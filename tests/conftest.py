from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from budgetroute.config import AppConfig, load_config
from budgetroute.schemas import BackendName, PredictionRecord, RouteName


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_config(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.chdir(project_root)
    return load_config(project_root / "configs/serving/fake.yaml", apply_environment=False)


@pytest.fixture
def prediction_factory() -> Callable[..., PredictionRecord]:
    def make(**changes: Any) -> PredictionRecord:
        fields: dict[str, Any] = {
            "example_id": "q1",
            "group_id": "q1",
            "category": "test",
            "policy": "learned",
            "route": RouteName.LARGE,
            "backend": BackendName.LARGE,
            "raw_answer": "B",
            "normalized_answer": "b",
            "reference": "A",
            "quality_score": 0.0,
            "metric_name": "classification_accuracy",
            "latency_ms": 100.0,
            "input_tokens": 10,
            "output_tokens": 1,
            "retrieval_used": False,
            "escalated": False,
            "abstained": False,
            "confidence": 0.99,
        }
        return PredictionRecord.model_validate({**fields, **changes})

    return make
