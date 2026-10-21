from __future__ import annotations

import json
from typing import Any
from urllib.request import Request

import pytest
from pydantic import ValidationError

from budgetroute.backends.openai_compatible import OpenAICompatibleBackend
from budgetroute.config import AppConfig, BackendConfig, validate_runtime_config
from budgetroute.exceptions import BackendError, ConfigurationError
from budgetroute.schemas import BackendName, GenerationRequest


class _Response:
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "id": "local-response",
                "choices": [
                    {
                        "message": {"content": "Paris"},
                        "finish_reason": "stop",
                        "logprobs": {"content": [{"token": "Paris", "logprob": -0.1}]},
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            }
        ).encode()


def test_openai_compatible_backend_parses_portable_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: float) -> _Response:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(bytes(request.data or b"{}").decode())
        return _Response()

    monkeypatch.setattr("budgetroute.backends.openai_compatible.urlopen", fake_urlopen)
    config = BackendConfig(
        type="openai_compatible",
        model_id="local-model",
        base_url="http://127.0.0.1:8001/v1",
    )
    backend = OpenAICompatibleBackend(BackendName.SMALL, config)
    backend.initialize()
    output = backend.generate(GenerationRequest(prompt="What is the capital of France?"))
    assert output.text == "Paris"
    assert output.confidence is not None
    assert output.input_tokens == 7
    assert captured["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert captured["payload"]["model"] == "local-model"


def test_openai_compatible_configuration_rejects_embedded_credentials() -> None:
    with pytest.raises(ValidationError, match="must not contain credentials"):
        BackendConfig(
            type="openai_compatible",
            model_id="local",
            base_url="https://user:secret@example.com/v1",
        )


def test_remote_openai_compatible_endpoint_requires_explicit_opt_in() -> None:
    backend = BackendConfig(
        type="openai_compatible", model_id="remote", base_url="https://example.com/v1"
    )
    config = AppConfig(mode="cpu", small_backend=backend, large_backend=backend)
    with pytest.raises(ConfigurationError, match="remote OpenAI-compatible endpoint is disabled"):
        validate_runtime_config(config)


def test_openai_compatible_batch_and_credential_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RUNTIME_KEY", "test-secret")
    monkeypatch.setattr(
        "budgetroute.backends.openai_compatible.urlopen",
        lambda request, timeout: _Response(),
    )
    config = BackendConfig(
        type="openai_compatible",
        model_id="local-model",
        base_url="http://localhost:8001/v1",
        api_key_env="LOCAL_RUNTIME_KEY",
        max_concurrency=2,
    )
    backend = OpenAICompatibleBackend(BackendName.SMALL, config)
    backend.initialize()
    outputs = backend.generate_batch(
        [GenerationRequest(prompt="one"), GenerationRequest(prompt="two")]
    )
    assert [output.text for output in outputs] == ["Paris", "Paris"]
    assert backend.metadata()["credential_source"] == "LOCAL_RUNTIME_KEY"
    backend.cleanup()
    assert backend.health()["ready"] is False
    with pytest.raises(BackendError, match="not initialized"):
        backend.generate(GenerationRequest(prompt="three"))


def test_openai_compatible_rejects_missing_credential_and_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = BackendConfig(
        type="openai_compatible",
        model_id="local-model",
        base_url="http://127.0.0.1:8001/v1",
        api_key_env="LOCAL_RUNTIME_KEY",
    )
    monkeypatch.delenv("LOCAL_RUNTIME_KEY", raising=False)
    backend = OpenAICompatibleBackend(BackendName.SMALL, config)
    with pytest.raises(BackendError, match="environment variable is unset"):
        backend.initialize()

    monkeypatch.setenv("LOCAL_RUNTIME_KEY", "test-secret")
    backend.initialize()

    class MalformedResponse(_Response):
        def read(self) -> bytes:
            return b'{"choices": []}'

    monkeypatch.setattr(
        "budgetroute.backends.openai_compatible.urlopen",
        lambda request, timeout: MalformedResponse(),
    )
    with pytest.raises(BackendError, match="missing choices"):
        backend.generate(GenerationRequest(prompt="test"))


def test_openai_compatible_empty_batch_is_stable() -> None:
    config = BackendConfig(
        type="openai_compatible",
        model_id="local-model",
        base_url="http://127.0.0.1:8001/v1",
    )
    backend = OpenAICompatibleBackend(BackendName.SMALL, config)
    backend.initialize()
    assert backend.generate_batch([]) == []


def test_returned_remote_model_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class WrongModel(_Response):
        def read(self) -> bytes:
            data = json.loads(super().read())
            data["model"] = "different-model"
            return json.dumps(data).encode()

    monkeypatch.setattr(
        "budgetroute.backends.openai_compatible.urlopen", lambda request, timeout: WrongModel()
    )
    backend = OpenAICompatibleBackend(
        BackendName.SMALL,
        BackendConfig(
            type="openai_compatible",
            model_id="local-model",
            base_url="http://localhost:8001/v1",
        ),
    )
    backend.initialize()
    with pytest.raises(BackendError, match="model"):
        backend.generate(GenerationRequest(prompt="Q"))


def test_unavailable_returned_identity_stays_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "budgetroute.backends.openai_compatible.urlopen", lambda request, timeout: _Response()
    )
    backend = OpenAICompatibleBackend(
        BackendName.SMALL,
        BackendConfig(
            type="openai_compatible",
            model_id="local-model",
            base_url="http://localhost:8001/v1",
            runtime_identity="pinned-server-a",
        ),
    )
    backend.initialize()
    output = backend.generate(GenerationRequest(prompt="Q"))
    assert output.metadata["returned_model"] is None
    assert output.metadata["runtime_identity"] == "pinned-server-a"


@pytest.mark.parametrize("body", [b"[]", b"null"])
def test_remote_identity_check_rejects_nonobject_response(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
) -> None:
    class NonObject(_Response):
        def read(self) -> bytes:
            return body

    monkeypatch.setattr(
        "budgetroute.backends.openai_compatible.urlopen", lambda request, timeout: NonObject()
    )
    backend = OpenAICompatibleBackend(
        BackendName.SMALL,
        BackendConfig(
            type="openai_compatible",
            model_id="model",
            base_url="http://localhost:8001/v1",
        ),
    )
    backend.initialize()
    with pytest.raises(BackendError, match="object"):
        backend.generate(GenerationRequest(prompt="Q"))


@pytest.mark.parametrize(
    "usage, measured, reported_input",
    [
        (None, False, None),
        ({"prompt_tokens": 7}, False, 7),
        ({"prompt_tokens": 7, "completion_tokens": 1}, True, 7),
    ],
)
def test_remote_usage_distinguishes_missing_counts_from_estimates(
    monkeypatch: pytest.MonkeyPatch,
    usage: dict[str, int] | None,
    measured: bool,
    reported_input: int | None,
) -> None:
    class UsageResponse(_Response):
        def read(self) -> bytes:
            data = json.loads(super().read())
            data["usage"] = usage
            return json.dumps(data).encode()

    monkeypatch.setattr(
        "budgetroute.backends.openai_compatible.urlopen", lambda request, timeout: UsageResponse()
    )
    backend = OpenAICompatibleBackend(
        BackendName.SMALL,
        BackendConfig(
            type="openai_compatible",
            model_id="model",
            base_url="http://localhost:8001/v1",
        ),
    )
    backend.initialize()
    output = backend.generate(GenerationRequest(prompt="Q"))
    assert output.usage_is_measured is measured
    assert output.metadata["reported_input_tokens"] == reported_input
