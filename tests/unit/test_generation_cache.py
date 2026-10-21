from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from budgetroute.backends.cached import CachedGenerationBackend
from budgetroute.backends.fake import FakeBackend
from budgetroute.config import AppConfig, BackendConfig, BenchmarkConfig, GenerationCacheConfig
from budgetroute.exceptions import ArtifactError
from budgetroute.experiments.cache import GenerationCache, backend_fingerprint
from budgetroute.schemas import BackendGeneration, BackendName, GenerationRequest


def test_cache_key_ignores_request_identity_and_replays_without_delegate_init(
    tmp_path: Path,
) -> None:
    config = BackendConfig(type="fake", quality=1.0)
    cache = GenerationCache(tmp_path / "cache")
    live_delegate = FakeBackend(BackendName.SMALL, config)
    live = CachedGenerationBackend(BackendName.SMALL, config, live_delegate, cache, "read_write")
    live.initialize()
    first = live.generate(
        GenerationRequest(request_id="live", prompt="What is the capital of France?")
    )
    assert first.text == "Paris"
    assert first.metadata["cache_hit"] is False

    untouched_delegate = FakeBackend(BackendName.SMALL, config)
    replay = CachedGenerationBackend(
        BackendName.SMALL,
        config,
        untouched_delegate,
        GenerationCache(tmp_path / "cache"),
        "read_only",
    )
    replay.initialize()
    second = replay.generate(
        GenerationRequest(request_id="replay", prompt="What is the capital of France?")
    )
    assert second.text == first.text
    assert second.metadata["cache_hit"] is True
    assert untouched_delegate.health()["initialized"] is False


def test_read_only_cache_miss_is_actionable(tmp_path: Path) -> None:
    config = BackendConfig(type="fake")
    backend = CachedGenerationBackend(
        BackendName.LARGE,
        config,
        FakeBackend(BackendName.LARGE, config),
        GenerationCache(tmp_path / "empty"),
        "read_only",
    )
    backend.initialize()
    with pytest.raises(ArtifactError, match="collect baselines before replay"):
        backend.generate(GenerationRequest(prompt="uncached"))


def test_cache_batches_only_misses_and_preserves_order(tmp_path: Path) -> None:
    config = BackendConfig(type="fake", quality=1.0)
    cache_path = tmp_path / "batch-cache"
    live = CachedGenerationBackend(
        BackendName.SMALL,
        config,
        FakeBackend(BackendName.SMALL, config),
        GenerationCache(cache_path),
        "read_write",
    )
    live.initialize()
    requests = [
        GenerationRequest(prompt="What is the capital of France?"),
        GenerationRequest(prompt="What is 2 + 2?"),
    ]
    generated = live.generate_batch(requests)
    assert [item.text for item in generated] == ["Paris", "4"]
    assert all(item.metadata["cache_hit"] is False for item in generated)

    replay = CachedGenerationBackend(
        BackendName.SMALL,
        config,
        FakeBackend(BackendName.SMALL, config),
        GenerationCache(cache_path),
        "read_only",
    )
    replay.initialize()
    cached = replay.generate_batch(requests)
    assert [item.text for item in cached] == ["Paris", "4"]
    assert all(item.metadata["cache_hit"] is True for item in cached)


@pytest.mark.parametrize(
    "changes",
    [
        {"base_url": "http://localhost:9001/v1"},
        {"request_logprobs": True},
        {"runtime_identity": "runtime-b"},
        {"model_loader": "image_text_to_text"},
        {"max_input_tokens": 2048},
        {"generation": {"enable_thinking": False}},
    ],
)
def test_cache_identity_changes_for_endpoint_logprobs_runtime_template(
    tmp_path: Path,
    changes: dict[str, Any],
) -> None:
    fields = {
        "type": "openai_compatible",
        "model_id": "model",
        "base_url": "http://localhost:8001/v1",
        "runtime_identity": "runtime-a",
    }
    first = backend_fingerprint(BackendName.SMALL, BackendConfig.model_validate(fields))
    second = backend_fingerprint(
        BackendName.SMALL, BackendConfig.model_validate({**fields, **changes})
    )
    cache = GenerationCache(tmp_path)
    request = GenerationRequest(prompt="Choose A")
    assert cache.key_for(first, request)[0] != cache.key_for(second, request)[0]


def test_cache_ignores_credentials_and_request_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = BackendConfig(
        type="openai_compatible",
        model_id="model",
        base_url="http://localhost:8001/v1",
        api_key_env="TEST_MODEL_KEY",
    )
    monkeypatch.setenv("TEST_MODEL_KEY", "secret-one")
    first = backend_fingerprint(BackendName.SMALL, config)
    monkeypatch.setenv("TEST_MODEL_KEY", "secret-two")
    second = backend_fingerprint(BackendName.SMALL, config)
    cache = GenerationCache(tmp_path)
    assert (
        cache.key_for(first, GenerationRequest(prompt="Q", request_id="1"))[0]
        == cache.key_for(second, GenerationRequest(prompt="Q", request_id="2"))[0]
    )
    assert "secret-one" not in json.dumps(first)
    assert "secret-two" not in json.dumps(second)


def test_old_cache_not_silently_reused(tmp_path: Path) -> None:
    cache = GenerationCache(tmp_path)
    fingerprint = backend_fingerprint(BackendName.SMALL, BackendConfig())
    request = GenerationRequest(prompt="Q")
    key = cache.put(
        fingerprint,
        request,
        BackendGeneration(
            text="A", backend=BackendName.SMALL, input_tokens=1, output_tokens=1, generation_ms=1
        ),
    )
    path = cache.path_for(key)
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["schema_version"] = 1
    path.write_text(json.dumps(entry), encoding="utf-8")
    with pytest.raises(ArtifactError, match="cache"):
        cache.get(fingerprint, request)


def test_cached_http_requires_runtime_identity() -> None:
    backend = BackendConfig(
        type="openai_compatible", model_id="model", base_url="http://localhost:8001/v1"
    )
    with pytest.raises(ValidationError, match="runtime_identity"):
        AppConfig(
            mode="cpu",
            small_backend=backend,
            large_backend=backend,
            benchmark=BenchmarkConfig(cache=GenerationCacheConfig(mode="read_write")),
        )
    AppConfig(mode="cpu", small_backend=backend, large_backend=backend)


def test_cache_wrapper_refuses_unidentified_http_runtime(tmp_path: Path) -> None:
    config = BackendConfig(
        type="openai_compatible", model_id="model", base_url="http://localhost:8001/v1"
    )
    with pytest.raises(ArtifactError, match="runtime_identity"):
        CachedGenerationBackend(
            BackendName.SMALL,
            config,
            FakeBackend(BackendName.SMALL, BackendConfig()),
            GenerationCache(tmp_path),
            "read_only",
        )
