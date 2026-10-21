"""Lazy Hugging Face Transformers backend with optional CPU/CUDA optimizations."""

from __future__ import annotations

import gc
import importlib.util
import math
import time
from contextlib import nullcontext
from typing import Any

from budgetroute.config import BackendConfig
from budgetroute.exceptions import BackendError
from budgetroute.schemas import (
    BackendGeneration,
    BackendName,
    ConfidenceSignals,
    GenerationRequest,
)


class TransformersBackend:
    def __init__(self, name: BackendName, config: BackendConfig) -> None:
        self.name = name
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._device = config.device
        self._optimization: dict[str, Any] = {
            "compile_requested": config.compile,
            "compile_enabled": False,
            "compile_first_execution_ms": None,
            "quantization_requested": config.quantization,
            "quantization_enabled": False,
        }

    def _require_dependencies(self) -> None:
        missing = [
            name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise BackendError(
                "Transformers backend requires optional dependencies: "
                + ", ".join(missing)
                + ". Install with `pip install -e .[transformers]`."
            )

    def initialize(self) -> None:
        if self._model is not None:
            return
        self._require_dependencies()
        import torch
        import transformers
        from transformers import AutoTokenizer

        self._torch = torch
        if self.config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.config.device == "cuda" and not torch.cuda.is_available():
            raise BackendError(
                "CUDA was explicitly requested but torch.cuda.is_available() is false. "
                "Install a CUDA-enabled PyTorch build or select device=cpu."
            )
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.config.precision]
        if self._device == "cuda" and self.config.precision == "bf16":
            supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            if not supported:
                raise BackendError(
                    "BF16 was requested but the selected CUDA device does not support it"
                )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            ("dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"): dtype,
        }
        if self.config.quantization != "none":
            if importlib.util.find_spec("bitsandbytes") is None:
                raise BackendError(
                    "quantization requires bitsandbytes; install the gpu extra on a supported platform"
                )
            from transformers import BitsAndBytesConfig

            bitsandbytes_config: Any = BitsAndBytesConfig
            model_kwargs["quantization_config"] = bitsandbytes_config(
                load_in_8bit=self.config.quantization == "int8",
                load_in_4bit=self.config.quantization == "int4",
            )
            self._optimization["quantization_enabled"] = True
        model_id = self.config.model_id
        if model_id is None:
            raise BackendError("model_id is required for a Transformers backend")
        tokenizer_id = self.config.tokenizer_id or model_id
        loader_name = (
            "AutoModelForImageTextToText"
            if self.config.model_loader == "image_text_to_text"
            else "AutoModelForCausalLM"
        )
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            raise BackendError(
                f"requested loader {loader_name} has no support in this Transformers version; upgrade the reviewed runtime or use the fallback profile"
            )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                revision=self.config.tokenizer_revision or self.config.revision,
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=self.config.local_files_only,
            )
            self._model = loader.from_pretrained(
                model_id,
                revision=self.config.revision,
                local_files_only=self.config.local_files_only,
                **model_kwargs,
            )
        except Exception as exc:
            self.cleanup()
            raise BackendError(
                f"could not load Transformers model {model_id!r}: {type(exc).__name__}: {exc}"
            ) from exc
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        if self.config.quantization == "none":
            self._model.to(self._device)
        self._model.eval()
        if self.config.compile:
            if not hasattr(torch, "compile"):
                raise BackendError(
                    "torch.compile was requested but is unavailable in this PyTorch build"
                )
            self._model = torch.compile(self._model)
            self._optimization["compile_enabled"] = True

    def health(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "type": "transformers",
            "initialized": self._model is not None,
            "ready": self._model is not None,
            "device": self._device,
        }

    def token_count(self, text: str) -> int:
        if self._tokenizer is None:
            raise BackendError("backend must be initialized before token counting")
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    def _render_prompt(self, prompt: str) -> str:
        if self._tokenizer is None:
            raise BackendError("backend is not initialized")
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            kwargs = (
                {"enable_thinking": self.config.generation.enable_thinking}
                if self.config.generation.enable_thinking is not None
                else {}
            )
            return str(
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            )
        return prompt

    def _check_input_limit(self, lengths: list[int]) -> None:
        limit = self.config.max_input_tokens
        if limit is not None and any(length > limit for length in lengths):
            raise BackendError(
                f"rendered input exceeds configured input token limit {limit}; shorten the prompt or start a separately identified configuration"
            )

    def _output_boundary(self, ids: list[int], budget: int) -> tuple[int, str]:
        eos = getattr(getattr(self._model, "generation_config", None), "eos_token_id", None)
        if eos is None:
            eos = getattr(self._tokenizer, "eos_token_id", None)
        stops = set(eos if isinstance(eos, list) else [eos])
        for index, token in enumerate(ids):
            if token in stops:
                return index + 1, "stop"
        return len(ids), "length" if len(ids) >= budget else "stop"

    def generate(self, request: GenerationRequest) -> BackendGeneration:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise BackendError("Transformers backend is not initialized")
        torch = self._torch
        prompt = self._render_prompt(request.prompt)
        encoded = self._tokenizer(prompt, return_tensors="pt", padding=True)
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[-1])
        self._check_input_limit([input_length])
        generation = self.config.generation
        max_new_tokens = request.max_new_tokens or generation.max_new_tokens
        synchronize = self._device == "cuda"
        torch.manual_seed(generation.seed)
        if synchronize:
            torch.cuda.manual_seed_all(generation.seed)
        if synchronize:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter_ns()
        autocast = (
            torch.autocast(
                device_type="cuda",
                dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[self.config.precision],
            )
            if self._device == "cuda" and self.config.precision in {"fp16", "bf16"}
            else nullcontext()
        )
        try:
            with torch.inference_mode(), autocast:
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": generation.temperature > 0,
                    "repetition_penalty": generation.repetition_penalty,
                    "pad_token_id": self._tokenizer.pad_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
                if generation.temperature > 0:
                    generation_kwargs.update(
                        {"temperature": generation.temperature, "top_p": generation.top_p}
                    )
                output = self._model.generate(
                    **encoded,
                    **generation_kwargs,
                )
        except Exception as exc:
            raise BackendError(
                f"Transformers generation failed: {type(exc).__name__}: {exc}"
            ) from exc
        if synchronize:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        if (
            self._optimization["compile_enabled"]
            and self._optimization["compile_first_execution_ms"] is None
        ):
            self._optimization["compile_first_execution_ms"] = elapsed_ms
        generated_ids = output.sequences[0, input_length:]
        output_tokens, finish_reason = self._output_boundary(generated_ids.tolist(), max_new_tokens)
        text = str(
            self._tokenizer.decode(generated_ids[:output_tokens], skip_special_tokens=True)
        ).strip()
        confidence: float | None = None
        signals: ConfidenceSignals | None = None
        if output.scores:
            transition_scores = self._model.compute_transition_scores(
                output.sequences,
                output.scores,
                getattr(output, "beam_indices", None),
                normalize_logits=True,
            )[0]
            selected = transition_scores[:output_tokens].detach().float().cpu()
            if selected.numel():
                sequence_log_probability = float(selected.sum().item())
                mean_log_probability = float(selected.mean().item())
                minimum_log_probability = float(selected.min().item())
                confidence = float(min(1.0, max(0.0, math.exp(mean_log_probability))))
                entropies: list[float] = []
                normalized_entropies: list[float] = []
                for score in output.scores[:output_tokens]:
                    logits = score[0].detach().float()
                    log_probabilities = torch.log_softmax(logits, dim=-1)
                    probabilities = torch.exp(log_probabilities)
                    entropy = float((-(probabilities * log_probabilities)).sum().item())
                    entropies.append(entropy)
                    normalized_entropies.append(entropy / max(math.log(logits.numel()), 1.0))
                signals = ConfidenceSignals(
                    method="length_normalized_token_likelihood",
                    token_count=int(selected.numel()),
                    sequence_log_probability=sequence_log_probability,
                    mean_token_log_probability=mean_log_probability,
                    minimum_token_log_probability=minimum_log_probability,
                    geometric_mean_token_probability=confidence,
                    mean_token_entropy=sum(entropies) / len(entropies),
                    normalized_mean_token_entropy=min(
                        1.0, max(0.0, sum(normalized_entropies) / len(normalized_entropies))
                    ),
                )
        return BackendGeneration(
            text=text,
            backend=self.name,
            input_tokens=input_length,
            output_tokens=output_tokens,
            usage_is_measured=True,
            confidence=confidence,
            confidence_signals=signals,
            generation_ms=elapsed_ms,
            metadata={
                "fake": False,
                "finish_reason": finish_reason,
                "model_loader": self.config.model_loader,
                "enable_thinking": self.config.generation.enable_thinking,
                "device": self._device,
                "tokens_per_second": output_tokens / (elapsed_ms / 1000) if elapsed_ms else None,
                "peak_cuda_mb": (
                    torch.cuda.max_memory_allocated() / (1024 * 1024) if synchronize else None
                ),
                "confidence_is_calibrated": False,
            },
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[BackendGeneration]:
        if not requests:
            return []
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise BackendError("Transformers backend is not initialized")
        grouped: dict[int, list[tuple[int, GenerationRequest]]] = {}
        for index, request in enumerate(requests):
            limit = request.max_new_tokens or self.config.generation.max_new_tokens
            grouped.setdefault(limit, []).append((index, request))
        ordered: list[BackendGeneration | None] = [None] * len(requests)
        for max_new_tokens, indexed_requests in grouped.items():
            outputs = self._generate_padded_batch(
                [request for _, request in indexed_requests], max_new_tokens
            )
            for (index, _), output in zip(indexed_requests, outputs, strict=True):
                ordered[index] = output
        if any(output is None for output in ordered):
            raise BackendError("Transformers batch did not produce every requested output")
        return [output for output in ordered if output is not None]

    def _generate_padded_batch(
        self, requests: list[GenerationRequest], max_new_tokens: int
    ) -> list[BackendGeneration]:
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None
        torch = self._torch
        prompts = [self._render_prompt(request.prompt) for request in requests]
        encoded = self._tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        padded_input_length = int(encoded["input_ids"].shape[-1])
        input_lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
        self._check_input_limit(input_lengths)
        generation = self.config.generation
        synchronize = self._device == "cuda"
        torch.manual_seed(generation.seed)
        if synchronize:
            torch.cuda.manual_seed_all(generation.seed)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        autocast = (
            torch.autocast(
                device_type="cuda",
                dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[self.config.precision],
            )
            if synchronize and self.config.precision in {"fp16", "bf16"}
            else nullcontext()
        )
        started = time.perf_counter_ns()
        try:
            with torch.inference_mode(), autocast:
                kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": generation.temperature > 0,
                    "repetition_penalty": generation.repetition_penalty,
                    "pad_token_id": self._tokenizer.pad_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
                if generation.temperature > 0:
                    kwargs.update(
                        {"temperature": generation.temperature, "top_p": generation.top_p}
                    )
                generated = self._model.generate(**encoded, **kwargs)
        except Exception as exc:
            raise BackendError(
                f"Transformers batch generation failed: {type(exc).__name__}: {exc}"
            ) from exc
        if synchronize:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        if (
            self._optimization["compile_enabled"]
            and self._optimization["compile_first_execution_ms"] is None
        ):
            self._optimization["compile_first_execution_ms"] = elapsed_ms
        generated_ids = generated.sequences[:, padded_input_length:]
        transition_scores = (
            self._model.compute_transition_scores(
                generated.sequences,
                generated.scores,
                getattr(generated, "beam_indices", None),
                normalize_logits=True,
            )
            if generated.scores
            else None
        )
        results: list[BackendGeneration] = []
        total_output_tokens = 0
        for row, _request in enumerate(requests):
            row_ids = generated_ids[row]
            output_tokens, finish_reason = self._output_boundary(row_ids.tolist(), max_new_tokens)
            total_output_tokens += output_tokens
            text = str(
                self._tokenizer.decode(row_ids[:output_tokens], skip_special_tokens=True)
            ).strip()
            confidence: float | None = None
            signals: ConfidenceSignals | None = None
            if transition_scores is not None and output_tokens:
                selected = transition_scores[row, :output_tokens].detach().float().cpu()
                sequence_log_probability = float(selected.sum().item())
                mean_log_probability = float(selected.mean().item())
                confidence = float(min(1.0, max(0.0, math.exp(mean_log_probability))))
                entropies: list[float] = []
                normalized_entropies: list[float] = []
                for score in generated.scores[:output_tokens]:
                    logits = score[row].detach().float()
                    log_probabilities = torch.log_softmax(logits, dim=-1)
                    probabilities = torch.exp(log_probabilities)
                    entropy = float((-(probabilities * log_probabilities)).sum().item())
                    entropies.append(entropy)
                    normalized_entropies.append(entropy / max(math.log(logits.numel()), 1.0))
                signals = ConfidenceSignals(
                    method="length_normalized_token_likelihood",
                    token_count=output_tokens,
                    sequence_log_probability=sequence_log_probability,
                    mean_token_log_probability=mean_log_probability,
                    minimum_token_log_probability=float(selected.min().item()),
                    geometric_mean_token_probability=confidence,
                    mean_token_entropy=sum(entropies) / len(entropies),
                    normalized_mean_token_entropy=min(
                        1.0, max(0.0, sum(normalized_entropies) / len(normalized_entropies))
                    ),
                )
            results.append(
                BackendGeneration(
                    text=text,
                    backend=self.name,
                    input_tokens=input_lengths[row],
                    output_tokens=output_tokens,
                    usage_is_measured=True,
                    confidence=confidence,
                    confidence_signals=signals,
                    generation_ms=elapsed_ms,
                    metadata={
                        "fake": False,
                        "finish_reason": finish_reason,
                        "model_loader": self.config.model_loader,
                        "enable_thinking": self.config.generation.enable_thinking,
                        "device": self._device,
                        "batch_size": len(requests),
                        "padded_input_tokens": padded_input_length,
                        "confidence_is_calibrated": False,
                        "peak_cuda_mb": (
                            torch.cuda.max_memory_allocated() / (1024 * 1024)
                            if synchronize
                            else None
                        ),
                    },
                )
            )
        aggregate_rate = total_output_tokens / (elapsed_ms / 1000) if elapsed_ms else None
        for result in results:
            result.metadata["batch_generated_tokens_per_second"] = aggregate_rate
        return results

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "type": "transformers",
            "model_id": self.config.model_id,
            "model_loader": self.config.model_loader,
            "max_input_tokens": self.config.max_input_tokens,
            "revision_requested": self.config.revision,
            "revision_resolved": getattr(
                getattr(self._model, "config", None), "_commit_hash", self.config.revision
            ),
            "tokenizer_id": self.config.tokenizer_id or self.config.model_id,
            "tokenizer_revision": self.config.tokenizer_revision or self.config.revision,
            "model_license": self.config.model_license,
            "model_card": self.config.model_card,
            "generation": self.config.generation.model_dump(mode="json"),
            "device_requested": self.config.device,
            "device_resolved": self._device,
            "precision": self.config.precision,
            "max_concurrency": self.config.max_concurrency,
            "input_cost_units_per_1k_tokens": self.config.input_cost_units_per_1k_tokens,
            "output_cost_units_per_1k_tokens": self.config.output_cost_units_per_1k_tokens,
            **self._optimization,
        }

    def cleanup(self) -> None:
        torch = self._torch
        self._model = None
        self._tokenizer = None
        self._torch = None
        # Repeated benchmark policies construct fresh services. Force cyclic
        # model/module references to release before loading the next checkpoint;
        # generational GC can otherwise retain several CPU state dicts on Windows.
        gc.collect()
        if torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
            torch.cuda.empty_cache()
