"""Offline adapter mocks: these do not demonstrate CUDA/model feasibility."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from budgetroute.backends.transformers import TransformersBackend
from budgetroute.config import BackendConfig, GenerationConfig
from budgetroute.exceptions import BackendError
from budgetroute.schemas import BackendName, GenerationRequest


class Tensor:
    def __init__(self, values: Any) -> None:
        self.values = np.array(values)

    @property
    def shape(self) -> Any:
        return self.values.shape

    def to(self, device: str) -> Tensor:
        return self

    def __getitem__(self, key: Any) -> Tensor:
        return Tensor(self.values[key])

    def numel(self) -> int:
        return int(self.values.size)

    def tolist(self) -> Any:
        return self.values.tolist()

    def sum(self, dim: int) -> Tensor:
        return Tensor(self.values.sum(axis=dim))


@pytest.fixture
def mock_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    observed: dict[str, Any] = {"rendered": [], "templates": [], "generate": []}

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0
        chat_template = "mock-template"
        padding_side = "left"

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            observed["templates"].append(kwargs)
            return str(messages[0]["content"])

        def __call__(self, prompts: Any, **kwargs: Any) -> dict[str, Tensor]:
            texts = prompts if isinstance(prompts, list) else [prompts]
            observed["rendered"].extend(texts)
            n = 6 if any("long" in text for text in texts) else 2
            return {
                "input_ids": Tensor([[1] * n for _ in texts]),
                "attention_mask": Tensor([[1] * n for _ in texts]),
            }

        def decode(self, ids: Tensor, **kwargs: Any) -> str:
            return "A"

    class Model:
        def to(self, device: str) -> Model:
            observed.setdefault("model_moves", []).append(device)
            return self

        def eval(self) -> Model:
            return self

        def generate(self, **kwargs: Any) -> Any:
            observed["generate"].append(kwargs)
            n, width = kwargs["input_ids"].shape
            tails = [[7, 0, 0] if i == 0 else [8, 9, 0] for i in range(n)]
            return SimpleNamespace(
                sequences=Tensor([[1] * width + tail for tail in tails]), scores=()
            )

    def loader(kind: str) -> Any:
        def load(*args: Any, **kwargs: Any) -> Model:
            observed["loader"] = kind
            observed["load_kwargs"] = kwargs
            return Model()

        return SimpleNamespace(from_pretrained=load)

    transformers = SimpleNamespace(
        __version__="5.12.0",
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        AutoModelForCausalLM=loader("causal_lm"),
        AutoModelForImageTextToText=loader("image_text_to_text"),
    )
    torch = SimpleNamespace(
        float32="fp32",
        float16="fp16",
        bfloat16="bf16",
        cuda=SimpleNamespace(is_available=lambda: False),
        manual_seed=lambda seed: None,
        inference_mode=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(TransformersBackend, "_require_dependencies", lambda self: None)
    observed["transformers"] = transformers
    return observed


@pytest.mark.parametrize("loader", ["causal_lm", "image_text_to_text"])
def test_explicit_mock_loader_render_and_batch_boundaries(
    mock_runtime: dict[str, Any], loader: str
) -> None:
    config = BackendConfig(
        type="transformers",
        model_id="mock",
        model_loader=loader,
        precision="fp16",
        device="auto",
        max_input_tokens=4,
        generation=GenerationConfig(max_new_tokens=64, enable_thinking=False),
    )
    backend = TransformersBackend(BackendName.SMALL, config)
    backend.initialize()
    single = backend.generate(GenerationRequest(prompt="short"))
    batch = backend.generate_batch(
        [GenerationRequest(prompt="short"), GenerationRequest(prompt="short")]
    )
    assert mock_runtime["loader"] == loader
    assert mock_runtime["load_kwargs"]["dtype"] == "fp16"
    assert mock_runtime["load_kwargs"]["trust_remote_code"] is False
    assert all(t["enable_thinking"] is False for t in mock_runtime["templates"])
    assert all(k["max_new_tokens"] == 64 for k in mock_runtime["generate"])
    assert mock_runtime["rendered"][0] == mock_runtime["rendered"][1]
    assert single.output_tokens == batch[0].output_tokens == 2
    assert batch[1].output_tokens == 3
    assert single.metadata["finish_reason"] == "stop"
    assert backend.metadata()["model_loader"] == loader
    before = len(mock_runtime["generate"])
    with pytest.raises(BackendError, match=r"input.*limit"):
        backend.generate(GenerationRequest(prompt="long"))
    with pytest.raises(BackendError, match=r"input.*limit"):
        backend.generate_batch([GenerationRequest(prompt="long")])
    assert len(mock_runtime["generate"]) == before


def test_missing_loader_is_explicit_feasibility_failure(mock_runtime: dict[str, Any]) -> None:
    del mock_runtime["transformers"].AutoModelForImageTextToText
    backend = TransformersBackend(
        BackendName.SMALL,
        BackendConfig(type="transformers", model_id="mock", model_loader="image_text_to_text"),
    )
    with pytest.raises(BackendError, match=r"loader.*support"):
        backend.initialize()


def test_eos_at_budget_is_not_truncation(mock_runtime: dict[str, Any]) -> None:
    backend = TransformersBackend(
        BackendName.SMALL, BackendConfig(type="transformers", model_id="mock")
    )
    backend.initialize()
    assert backend._output_boundary([7, 8, 0], 3) == (3, "stop")
    assert backend._output_boundary([7, 8, 9], 3) == (3, "length")


def test_direct_device_loading_avoids_cpu_staging(mock_runtime: dict[str, Any]) -> None:
    config = BackendConfig(
        type="transformers", model_id="mock", device="cpu", load_direct_to_device=True
    )
    backend = TransformersBackend(BackendName.SMALL, config)
    backend.initialize()
    assert mock_runtime["load_kwargs"]["device_map"] == "cpu"
    assert mock_runtime.get("model_moves", []) == []
    assert backend.metadata()["load_direct_to_device"] is True


def test_direct_device_loading_has_separate_cache_identity() -> None:
    from budgetroute.experiments.cache import backend_fingerprint

    staged = BackendConfig(type="transformers", model_id="mock")
    direct = staged.model_copy(update={"load_direct_to_device": True})
    assert backend_fingerprint(BackendName.SMALL, staged) != backend_fingerprint(
        BackendName.SMALL, direct
    )
