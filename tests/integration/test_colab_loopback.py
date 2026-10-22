"""Transport regressions with fake inference; never remote/GPU evidence."""

from __future__ import annotations

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from budgetroute.config import AppConfig
from budgetroute.inference.service import build_service
from budgetroute.validation.colab import _loopback_api


def test_loopback_probe_applies_and_restores_its_service_configuration(
    fake_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = build_service(fake_config)
    apps = []

    class InProcessServer:
        started = True
        should_exit = False

        def __init__(self, config):
            apps.append(config.app)

        def run(self, **kwargs):
            pass

    # The Windows execution environment blocks loopback HTTP. Exercise the real
    # ASGI endpoints in process; actual IPv4 transport remains a Colab gate.
    monkeypatch.setattr(uvicorn, "Server", InProcessServer)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: TestClient(apps[-1]))
    generate = service.generate
    # Only exercise the real HTTP/configuration boundary, not model feasibility.
    monkeypatch.setattr(
        service, "generate", lambda request: generate(request).model_copy(update={"fake": False})
    )
    assert service.config.api.feedback_enabled is False
    report = _loopback_api(fake_config, service)
    assert report["feedback_status"] == 202
    assert report["metrics_status"] == 200
    assert service.config is fake_config
