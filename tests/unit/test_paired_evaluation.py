from __future__ import annotations

from collections.abc import Callable

import pytest

from budgetroute.schemas import PredictionRecord


def test_failed_answer_remains_in_paired_denominator(
    prediction_factory: Callable[..., PredictionRecord],
) -> None:
    from budgetroute.evaluation.metrics import aggregate_metrics
    from budgetroute.evaluation.paired import paired_policy_summary

    baseline = [
        prediction_factory(
            example_id=str(i), group_id=str(i), policy="always_large", quality_score=1.0
        )
        for i in range(4)
    ]
    candidate = [row.model_copy(update={"policy": "learned"}) for row in baseline]
    candidate[-1] = candidate[-1].model_copy(
        update={"error": "generation failed", "latency_ms": 0.0}
    )
    summary = paired_policy_summary(candidate, baseline)
    assert summary["paired_count"] == 4
    assert summary["accuracy_delta"] == -0.25
    assert summary["losses"] == 1
    assert summary["candidate_failures"] == 1
    assert summary["independent_group_count"] == 4
    assert summary["latency_ratio"] is None
    assert aggregate_metrics(candidate)["accuracy"] == 0.75
    assert aggregate_metrics(candidate)["successful_accuracy"] == 1.0


def test_incomplete_pairs_are_not_silently_dropped(
    prediction_factory: Callable[..., PredictionRecord],
) -> None:
    from budgetroute.evaluation.paired import paired_policy_summary

    result = paired_policy_summary(
        [prediction_factory(example_id="a")], [prediction_factory(example_id="b")]
    )
    assert result["complete"] is False
    assert result["accuracy_delta"] is None
    assert result["missing_candidate_ids"] == ["b"]
    assert result["missing_baseline_ids"] == ["a"]


def test_duplicate_pair_ids_are_rejected(
    prediction_factory: Callable[..., PredictionRecord],
) -> None:
    from budgetroute.evaluation.paired import paired_policy_summary

    row = prediction_factory()
    with pytest.raises(ValueError, match="duplicate"):
        paired_policy_summary([row, row], [row])


@pytest.mark.parametrize("changed", [{"reference": "C"}, {"group_id": "different"}])
def test_changed_reference_or_group_is_not_a_pair(
    prediction_factory: Callable[..., PredictionRecord], changed: dict[str, str]
) -> None:
    from budgetroute.evaluation.paired import paired_policy_summary

    row = prediction_factory()
    with pytest.raises(ValueError, match="paired"):
        paired_policy_summary([row], [row.model_copy(update=changed)])


@pytest.mark.parametrize("identity", [{"fake": True}, {"fake": None}, {"replayed": True}])
def test_non_live_identity_cannot_yield_a_latency_ratio(
    prediction_factory: Callable[..., PredictionRecord], identity: dict[str, object]
) -> None:
    from budgetroute.evaluation.paired import paired_policy_summary

    context = {
        "hardware_fingerprint": "gpu-a",
        "generation_fingerprint": "weights-a",
        "dataset_hash": "data-a",
        "trial_block": "block-1",
        "status": "completed",
        "fake": False,
        "replay": False,
    }
    rows = [prediction_factory(example_id=str(i), group_id=str(i), fake=False) for i in range(2)]
    result = paired_policy_summary(
        [row.model_copy(update=identity) for row in rows],
        rows,
        candidate_run=context,
        baseline_run=context,
    )
    assert result["latency_ratio"] is None


def test_live_ratio_requires_compatible_context_and_uses_paired_clusters(
    prediction_factory: Callable[..., PredictionRecord],
) -> None:
    from budgetroute.evaluation.paired import paired_policy_summary

    context = {
        "hardware_fingerprint": "gpu-a",
        "generation_fingerprint": "weights-a",
        "dataset_hash": "data-a",
        "trial_block": "block-1",
        "status": "completed",
        "fake": False,
        "replay": False,
    }
    baseline = [
        prediction_factory(
            example_id=str(i), group_id=str(i), policy="always_large", fake=False, latency_ms=100
        )
        for i in range(2)
    ]
    candidate = [row.model_copy(update={"policy": "learned", "latency_ms": 50}) for row in baseline]
    assert paired_policy_summary(candidate, baseline)["latency_ratio"] is None
    summary = paired_policy_summary(
        candidate, baseline, candidate_run=context, baseline_run=context
    )
    assert summary["latency_ratio"] == 0.5
    assert summary["latency_ratio_confidence_interval"]["estimate"] == 0.5
    different_gpu = {**context, "hardware_fingerprint": "gpu-b"}
    assert (
        paired_policy_summary(
            candidate, baseline, candidate_run=different_gpu, baseline_run=context
        )["latency_ratio"]
        is None
    )
    assert (
        paired_policy_summary(
            candidate,
            baseline,
            candidate_run={**context, "trial_block": "block-2"},
            baseline_run=context,
        )["latency_ratio"]
        is None
    )


def test_mixed_policy_population_is_rejected(
    prediction_factory: Callable[..., PredictionRecord],
) -> None:
    from budgetroute.evaluation.paired import paired_policy_summary

    rows = [prediction_factory(example_id="a"), prediction_factory(example_id="b", policy="other")]
    with pytest.raises(ValueError, match="one policy"):
        paired_policy_summary(rows, rows)


def test_empty_and_one_group_are_not_stable_evidence(
    prediction_factory: Callable[..., PredictionRecord],
) -> None:
    from budgetroute.evaluation.paired import paired_policy_summary

    assert paired_policy_summary([], [])["accuracy_delta"] is None
    row = prediction_factory()
    assert paired_policy_summary([row], [row])["accuracy_delta_confidence_interval"] is None
