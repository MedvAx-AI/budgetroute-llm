"""Routing, escalation, coverage, and selective-quality metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from budgetroute.evaluation.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
)
from budgetroute.schemas import PredictionRecord, RouteName


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def routing_quality_metrics(
    predicted_small_success: list[int],
    actual_small_success: list[int],
    probabilities: list[float] | None = None,
) -> dict[str, Any]:
    if len(predicted_small_success) != len(actual_small_success):
        raise ValueError("predicted and actual routing labels must have equal length")
    if not actual_small_success:
        return {
            "routing_accuracy": None,
            "escalation_precision": None,
            "escalation_recall": None,
            "false_escalation_rate": None,
            "false_non_escalation_rate": None,
            "brier_score": None,
            "expected_calibration_error": None,
            "reliability": [],
        }
    predicted = np.asarray(predicted_small_success, dtype=int)
    actual = np.asarray(actual_small_success, dtype=int)
    predicted_escalate = predicted == 0
    actual_escalate = actual == 0
    true_escalation = int((predicted_escalate & actual_escalate).sum())
    false_escalation = int((predicted_escalate & ~actual_escalate).sum())
    missed_escalation = int((~predicted_escalate & actual_escalate).sum())
    probabilities = probabilities if probabilities is not None else []
    true_positive = int(((predicted == 1) & (actual == 1)).sum())
    true_negative = int(((predicted == 0) & (actual == 0)).sum())
    false_positive = int(((predicted == 1) & (actual == 0)).sum())
    false_negative = int(((predicted == 0) & (actual == 1)).sum())
    return {
        "routing_accuracy": float((predicted == actual).mean()),
        "small_model_success_prediction_accuracy": float((predicted == actual).mean()),
        "escalation_precision": _safe_ratio(true_escalation, true_escalation + false_escalation),
        "escalation_recall": _safe_ratio(true_escalation, true_escalation + missed_escalation),
        "false_escalation_rate": _safe_ratio(false_escalation, int((actual == 1).sum())),
        "false_non_escalation_rate": _safe_ratio(missed_escalation, int((actual == 0).sum())),
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
        "brier_score": brier_score(probabilities, actual_small_success),
        "expected_calibration_error": expected_calibration_error(
            probabilities, actual_small_success
        ),
        "reliability": reliability_diagram_data(probabilities, actual_small_success),
    }


def router_probability_metrics(
    predictions: list[PredictionRecord],
    baselines: list[PredictionRecord],
    quality_threshold: float,
) -> dict[str, Any]:
    """Score only explicitly recorded gate probabilities against their own target."""
    lookup = {
        (row.example_id, row.policy): row.quality_score
        for row in baselines
        if row.policy in {"always_small", "always_large"} and row.error is None
    }
    targets = {
        row.router_probability.target for row in predictions if row.router_probability is not None
    }
    if len(targets) > 1:
        raise ValueError("cannot pool probabilities for different router targets")
    probabilities: list[float] = []
    raw: list[float] = []
    labels: list[int] = []
    decisions: list[int] = []
    missing = 0
    for row in predictions:
        probability = row.router_probability
        if probability is None:
            continue
        small = lookup.get((row.example_id, "always_small"))
        large = lookup.get((row.example_id, "always_large"))
        if small is None or (probability.target == "paired_quality" and large is None):
            missing += 1
            continue
        if probability.target == "small_success":
            label = int(small >= quality_threshold)
        else:
            assert large is not None
            label = int(small >= large)
        probabilities.append(probability.probability)
        raw.append(probability.raw_probability)
        labels.append(label)
        decisions.append(
            int(
                not row.escalated
                and row.route
                in {RouteName.SMALL, RouteName.SMALL_WITH_RETRIEVAL, RouteName.CASCADE}
            )
        )
    result = routing_quality_metrics(decisions, labels, probabilities)
    result.update(
        {
            "target": next(iter(targets), None),
            "probability_count": len(probabilities),
            "missing_probability_count": sum(row.router_probability is None for row in predictions),
            "missing_baseline_count": missing,
            "raw_brier_score": brier_score(raw, labels),
        }
    )
    if result["target"] == "paired_quality":
        result.pop("small_model_success_prediction_accuracy", None)
    return result


def selective_metrics(quality: list[float], abstained: list[bool]) -> dict[str, float | None]:
    if len(quality) != len(abstained):
        raise ValueError("quality and abstention vectors must have equal length")
    if not quality:
        return {"coverage": None, "selective_accuracy": None}
    covered = [
        score for score, is_abstained in zip(quality, abstained, strict=True) if not is_abstained
    ]
    return {
        "coverage": len(covered) / len(quality),
        "selective_accuracy": sum(covered) / len(covered) if covered else None,
    }
