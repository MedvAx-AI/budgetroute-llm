from __future__ import annotations

from collections.abc import Callable

import pytest

from budgetroute.evaluation.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
)
from budgetroute.evaluation.evaluators import exact_match, keyword, numeric, token_f1
from budgetroute.evaluation.routing_metrics import routing_quality_metrics, selective_metrics
from budgetroute.evaluation.uncertainty import grouped_bootstrap_mean
from budgetroute.experiments.runner import _warm_up
from budgetroute.schemas import BenchmarkRecord, EvaluationType, GenerationRequest, PredictionRecord


def test_deterministic_evaluators() -> None:
    assert exact_match("The Paris!", "the paris").score == 1
    assert token_f1("alpha beta", "alpha gamma").score == pytest.approx(0.5)
    assert numeric("The answer is 1,024.", "1024").score == 1
    assert numeric("First 2, then 3. Final answer: 3/4", "0.75").score == 1
    assert numeric("Reasoning 2 + 3 = 5, therefore 5", "5").score == 1
    assert keyword("A partial match", ["art"]).score == 0


def test_grouped_bootstrap_is_deterministic_and_group_aware() -> None:
    first = grouped_bootstrap_mean([0.0, 1.0, 0.5, 0.5], ["a", "a", "b", "b"], samples=100, seed=7)
    second = grouped_bootstrap_mean([0.0, 1.0, 0.5, 0.5], ["a", "a", "b", "b"], samples=100, seed=7)
    assert first == second
    assert first is not None
    assert first["group_count"] == 2


def test_calibration_handles_empty_and_constant_inputs() -> None:
    assert brier_score([], []) is None
    assert expected_calibration_error([0.5, 0.5], [0, 1]) == 0
    diagram = reliability_diagram_data([1.0], [1], bins=2)
    assert sum(item["count"] for item in diagram) == 1


def test_routing_metrics_handle_empty_classes() -> None:
    metrics = routing_quality_metrics([1, 1], [1, 1], [0.8, 0.9])
    assert metrics["routing_accuracy"] == 1
    assert metrics["escalation_precision"] is None
    assert selective_metrics([1.0, 0.0], [False, True]) == {
        "coverage": 0.5,
        "selective_accuracy": 1.0,
    }


def test_warm_up_uses_a_cache_isolation_marker() -> None:
    requests: list[GenerationRequest] = []

    class Service:
        def generate(self, request: GenerationRequest) -> None:
            requests.append(request)

    record = BenchmarkRecord(
        id="example",
        category="test",
        prompt="Question?",
        reference_answer="A",
        evaluation_type=EvaluationType.CLASSIFICATION,
    )

    _warm_up(Service(), [record], 1, fake=False)

    assert requests[0].metadata["benchmark_warmup"] is True


def test_missing_answer_confidence_is_unavailable(
    prediction_factory: Callable[..., PredictionRecord],
) -> None:
    from budgetroute.evaluation.metrics import aggregate_metrics

    metrics = aggregate_metrics([prediction_factory(confidence=0.99, backend_confidence=None)])
    assert metrics["brier_score"] is None
    assert metrics["expected_calibration_error"] is None
    assert metrics["probability_count"] == 0


@pytest.mark.parametrize("target, expected", [("small_success", 0.04), ("paired_quality", 0.64)])
def test_router_probability_is_not_large_answer_complement(
    prediction_factory: Callable[..., PredictionRecord],
    target: str,
    expected: float,
) -> None:
    from budgetroute.evaluation.routing_metrics import router_probability_metrics

    candidate = prediction_factory(
        backend_confidence=0.99,
        router_probability={
            "raw_probability": 0.3,
            "probability": 0.2,
            "target": target,
            "calibrated": True,
        },
    )
    baselines = [
        prediction_factory(policy=policy, quality_score=0.0)
        for policy in ["always_small", "always_large"]
    ]
    metrics = router_probability_metrics([candidate], baselines, 1.0)
    assert metrics["probability_count"] == 1
    assert metrics["target"] == target
    assert metrics["brier_score"] == pytest.approx(expected)
    expected_raw = 0.09 if target == "small_success" else 0.49
    assert metrics["raw_brier_score"] == pytest.approx(expected_raw)
    restored = PredictionRecord.model_validate_json(candidate.model_dump_json())
    assert restored.router_probability == candidate.router_probability


def test_missing_paired_baseline_does_not_fabricate_label(
    prediction_factory: Callable[..., PredictionRecord],
) -> None:
    from budgetroute.evaluation.routing_metrics import router_probability_metrics

    candidate = prediction_factory(
        router_probability={
            "raw_probability": 0.2,
            "probability": 0.2,
            "target": "paired_quality",
            "calibrated": False,
        }
    )
    metrics = router_probability_metrics(
        [candidate], [prediction_factory(policy="always_small")], 1.0
    )
    assert metrics["probability_count"] == 0
    assert metrics["missing_baseline_count"] == 1
    assert metrics["brier_score"] is None


def test_route_decisions_without_probabilities_have_no_calibration() -> None:
    metrics = routing_quality_metrics([1], [0])
    assert metrics["routing_accuracy"] == 0.0
    assert metrics["brier_score"] is None
    assert metrics["reliability"] == []


def test_legacy_missing_cost_is_not_zero(
    prediction_factory: Callable[..., PredictionRecord],
) -> None:
    from budgetroute.evaluation.metrics import aggregate_metrics

    assert aggregate_metrics([prediction_factory()])["estimated_cost_units"] is None


def test_legacy_unknown_record_preserves_other_known_cost(
    prediction_factory: Callable[..., PredictionRecord],
) -> None:
    from budgetroute.evaluation.metrics import aggregate_metrics

    known = prediction_factory(
        backend_calls=[
            {
                "backend": "small",
                "input_tokens": 100,
                "output_tokens": 10,
                "input_rate": 1,
                "output_rate": 2,
            }
        ]
    )
    result = aggregate_metrics([known, prediction_factory(example_id="legacy")])
    assert result["estimated_cost_units"] is None
    assert result["backend_cost_accounting"]["known_cost_units"] == pytest.approx(0.12)
    assert result["backend_cost_accounting"]["complete"] is False


def test_cost_derivation_rejects_unrepresentable_total() -> None:
    from pydantic import ValidationError

    from budgetroute.schemas import BackendCallUsage

    with pytest.raises(ValidationError, match="finite"):
        BackendCallUsage(
            backend="small", input_tokens=3000, output_tokens=1, input_rate=1e308, output_rate=0
        )


def test_unequal_groups_keep_observation_weighted_estimand() -> None:
    interval = grouped_bootstrap_mean([0.0, 1.0, 1.0, 1.0], ["a", "b", "b", "b"], samples=1, seed=7)
    assert interval is not None
    assert interval["estimate"] == 0.75
    assert interval["lower"] == 0.75
    assert interval["upper"] == 0.75
    assert interval["estimand"] == "observation_weighted"


def test_repeated_records_keep_original_independent_groups() -> None:
    from budgetroute.experiments.runner import _measured_records

    original = [
        BenchmarkRecord(
            id=f"q{i}",
            prompt="Question",
            category="test",
            reference_answer="A",
            evaluation_type=EvaluationType.CLASSIFICATION,
        )
        for i in range(2)
    ]
    repeated = _measured_records(original, 3)
    assert len(repeated) == 6
    assert {row.group_id for row in repeated} == {"q0", "q1"}
