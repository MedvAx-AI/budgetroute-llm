"""OpenAI-compatible chat-completions backend for explicitly configured local runtimes."""

from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from budgetroute.config import BackendConfig
from budgetroute.exceptions import BackendError
from budgetroute.schemas import (
    BackendGeneration,
    BackendName,
    ConfidenceSignals,
    GenerationRequest,
)


class OpenAICompatibleBackend:
    """Call the portable `/v1/chat/completions` subset without vendor SDK coupling."""

    def __init__(self, name: BackendName, config: BackendConfig) -> None:
        self.name = name
        self.config = config
        self._initialized = False

    @property
    def endpoint(self) -> str:
        assert self.config.base_url is not None
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def initialize(self) -> None:
        if self.config.api_key_env and not os.environ.get(self.config.api_key_env):
            raise BackendError(
                f"OpenAI-compatible credential environment variable is unset: "
                f"{self.config.api_key_env}"
            )
        self._initialized = True

    def health(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "type": "openai_compatible",
            "initialized": self._initialized,
            "ready": self._initialized,
            "endpoint": self.config.base_url,
        }

    def token_count(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key_env:
            token = os.environ.get(self.config.api_key_env)
            if not token:
                raise BackendError(
                    f"OpenAI-compatible credential environment variable is unset: "
                    f"{self.config.api_key_env}"
                )
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _confidence(choice: dict[str, Any]) -> tuple[float | None, ConfidenceSignals | None]:
        content = choice.get("logprobs", {}).get("content") or []
        values = [float(item["logprob"]) for item in content if item.get("logprob") is not None]
        if not values:
            return None, None
        mean_log_probability = sum(values) / len(values)
        confidence = min(1.0, max(0.0, math.exp(mean_log_probability)))
        return confidence, ConfidenceSignals(
            method="openai_compatible_token_likelihood",
            token_count=len(values),
            sequence_log_probability=sum(values),
            mean_token_log_probability=mean_log_probability,
            minimum_token_log_probability=min(values),
            geometric_mean_token_probability=confidence,
        )

    def generate(self, request: GenerationRequest) -> BackendGeneration:
        if not self._initialized:
            raise BackendError(f"OpenAI-compatible {self.name.value} backend is not initialized")
        generation = self.config.generation
        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_new_tokens or generation.max_new_tokens,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "seed": generation.seed,
        }
        if self.config.request_logprobs:
            payload["logprobs"] = True
        if generation.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": generation.enable_thinking}
        http_request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        started = time.perf_counter_ns()
        try:
            with urlopen(http_request, timeout=self.config.request_timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise BackendError(
                f"OpenAI-compatible server returned HTTP {exc.code} at the configured endpoint"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise BackendError(
                f"OpenAI-compatible request failed: {type(exc).__name__}: {exc}"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError("OpenAI-compatible server returned invalid JSON") from exc
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        if not isinstance(response_data, dict):
            raise BackendError("OpenAI-compatible response must be a JSON object")
        returned_model = response_data.get("model")
        if returned_model is not None and returned_model != self.config.model_id:
            raise BackendError("OpenAI-compatible response model does not match configured model")
        try:
            choice = response_data["choices"][0]
            text = str(choice["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(
                "OpenAI-compatible response is missing choices[0].message.content"
            ) from exc
        usage = response_data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        reported_input = usage.get("prompt_tokens")
        reported_output = usage.get("completion_tokens")
        reported_input = (
            reported_input if type(reported_input) is int and reported_input >= 0 else None
        )
        reported_output = (
            reported_output if type(reported_output) is int and reported_output >= 0 else None
        )
        input_tokens = (
            reported_input if reported_input is not None else self.token_count(request.prompt)
        )
        output_tokens = reported_output if reported_output is not None else self.token_count(text)
        confidence, signals = self._confidence(choice)
        return BackendGeneration(
            text=text,
            backend=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_is_measured=reported_input is not None and reported_output is not None,
            confidence=confidence,
            confidence_signals=signals,
            generation_ms=elapsed_ms,
            metadata={
                "fake": False,
                "runtime": "openai_compatible",
                "runtime_identity": self.config.runtime_identity,
                "returned_model": returned_model,
                "system_fingerprint": response_data.get("system_fingerprint"),
                "finish_reason": choice.get("finish_reason"),
                "response_id": response_data.get("id"),
                "reported_input_tokens": reported_input,
                "reported_output_tokens": reported_output,
                "confidence_is_calibrated": False,
            },
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[BackendGeneration]:
        if not requests:
            return []
        workers = min(len(requests), self.config.max_concurrency)
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="budgetroute-openai"
        ) as pool:
            return list(pool.map(self.generate, requests))

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "type": "openai_compatible",
            "model_id": self.config.model_id,
            "base_url": self.config.base_url,
            "runtime_identity": self.config.runtime_identity,
            "remote_endpoint_allowed": self.config.allow_remote_endpoint,
            "credential_source": self.config.api_key_env,
            "request_timeout_seconds": self.config.request_timeout_seconds,
            "request_logprobs": self.config.request_logprobs,
            "max_concurrency": self.config.max_concurrency,
            "input_cost_units_per_1k_tokens": self.config.input_cost_units_per_1k_tokens,
            "output_cost_units_per_1k_tokens": self.config.output_cost_units_per_1k_tokens,
            "token_count_method": "usage_or_regex_approximation",
            "generation": self.config.generation.model_dump(mode="json"),
        }

    def cleanup(self) -> None:
        self._initialized = False
