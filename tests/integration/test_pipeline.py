from __future__ import annotations

import pytest

from budgetroute.backends.fake import FakeBackend
from budgetroute.config import AppConfig, BackendConfig, RoutingConfig
from budgetroute.inference.engine import InferenceEngine
from budgetroute.inference.service import build_service
from budgetroute.routing.registry import build_policy
from budgetroute.schemas import BackendName, GenerationRequest, RouteName


def test_fake_request_through_pipeline(fake_config: AppConfig) -> None:
    service = build_service(fake_config)
    try:
        response = service.generate(GenerationRequest(prompt="What is the capital of France?"))
        assert response.answer == "Paris"
        assert response.route == RouteName.SMALL
        assert response.fake
        assert response.timing.total_ms > 0
        abstained = service.generate(
            GenerationRequest(prompt="This is ambiguous without any context.")
        )
        assert abstained.route == RouteName.ABSTAIN
        assert abstained.backend_calls == []
    finally:
        service.close()


def test_retrieval_aware_route(fake_config: AppConfig) -> None:
    service = build_service(fake_config)
    try:
        response = service.generate(
            GenerationRequest(
                prompt="According to the project corpus, what is the internal project codename?",
                requires_retrieval=True,
                metadata={"fake_reference_answer": "Lantern", "fake_small_success": False},
            )
        )
        assert response.route == RouteName.SMALL_WITH_RETRIEVAL
        assert response.answer == "Lantern"
        assert response.retrieval
        assert response.timing.retrieval_ms > 0
    finally:
        service.close()


def test_cascade_escalation(fake_config: AppConfig) -> None:
    cascade = fake_config.model_copy(
        update={"routing": fake_config.routing.model_copy(update={"policy": "cascade"})}
    )
    service = build_service(cascade)
    try:
        response = service.generate(
            GenerationRequest(
                prompt="A difficult fake prompt",
                metadata={"fake_reference_answer": "large answer", "fake_small_success": False},
            )
        )
        assert response.execution.escalated
        assert response.execution.final_backend == "large"
        assert response.answer == "large answer"
        assert response.timing.escalation_ms > 0
    finally:
        service.close()


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    "policy, threshold, count, cost",
    [
        ("always_small", 1.0, 1, 0.12),
        ("always_large", 1.0, 1, 0.68),
        ("cascade", 1.0, 2, 0.80),
        ("cascade", 0.0, 1, 0.12),
    ],
)
def test_cascade_accounts_for_both_backend_calls(
    batch: bool,
    policy: str,
    threshold: float,
    count: int,
    cost: float,
) -> None:
    from budgetroute.evaluation.metrics import summarize_backend_calls

    class DoubleTokenFake(FakeBackend):
        def token_count(self, text: str) -> int:
            return 2 * super().token_count(text)

    small_config = BackendConfig(quality=1.0, measured_input_rate=1.0, measured_output_rate=2.0)
    large_config = BackendConfig(quality=1.0, measured_input_rate=3.0, measured_output_rate=4.0)
    small = FakeBackend(BackendName.SMALL, small_config)
    large = DoubleTokenFake(BackendName.LARGE, large_config)
    small.initialize()
    large.initialize()
    routing = RoutingConfig.model_validate(
        {"policy": policy, "cascade_confidence_threshold": threshold}
    )
    engine = InferenceEngine(
        small,
        large,
        build_policy(routing),
        routing,
        backend_configs={
            BackendName.SMALL: small_config,
            BackendName.LARGE: large_config,
        },
    )
    requests = [
        GenerationRequest(
            request_id=str(i),
            prompt="word " * 100,
            metadata={
                "fake_reference_answer": "answer " * 10,
                "fake_small_success": True,
            },
        )
        for i in range(2)
    ]
    try:
        responses = (
            engine.generate_batch(requests) if batch else [engine.generate(r) for r in requests]
        )
        for response in responses:
            assert response.backend_calls is not None
            assert len(response.backend_calls) == count
            assert summarize_backend_calls(response.backend_calls)[
                "estimated_cost_units"
            ] == pytest.approx(cost)
    finally:
        small.cleanup()
        large.cleanup()


def test_missing_usage_or_rate_keeps_total_unknown() -> None:
    from budgetroute.evaluation.metrics import summarize_backend_calls
    from budgetroute.schemas import BackendCallUsage

    known = BackendCallUsage(
        backend=BackendName.SMALL,
        input_tokens=100,
        output_tokens=10,
        input_rate=1.0,
        output_rate=2.0,
        estimated_cost_units=0.12,
    )
    unknown = BackendCallUsage(
        backend=BackendName.LARGE,
        input_tokens=200,
        output_tokens=None,
        input_rate=3.0,
        output_rate=4.0,
        estimated_cost_units=None,
    )
    summary = summarize_backend_calls([known, unknown])
    assert summary["estimated_cost_units"] is None
    assert summary["known_cost_units"] == pytest.approx(0.12)
    assert summary["unknown_call_count"] == 1
    assert summary["complete"] is False
    assert summarize_backend_calls(None)["estimated_cost_units"] is None
    assert summarize_backend_calls([])["estimated_cost_units"] == 0.0
    declared_zero = known.model_copy(update={"input_rate": 0.0, "output_rate": 0.0})
    declared_zero = BackendCallUsage.model_validate(declared_zero.model_dump())
    assert summarize_backend_calls([declared_zero])["estimated_cost_units"] == 0.0
    assert summarize_backend_calls([declared_zero])["complete"] is True
