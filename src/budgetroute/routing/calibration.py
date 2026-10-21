"""Threshold selection for quality-aware routing policies."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

import numpy as np

from budgetroute.exceptions import RouterTrainingError


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    coverage: float
    selective_accuracy: float


@dataclass(frozen=True)
class ThresholdSelection:
    status: Literal["feasible", "infeasible", "legacy"]
    threshold: float | None
    reason: str | None = None

    def accepts(self, probability: float | None) -> bool:
        return bool(
            self.status != "infeasible"
            and self.threshold is not None
            and probability is not None
            and isfinite(probability)
            and 0 <= probability <= 1
            and probability >= self.threshold
        )


def resolve_threshold_selection(
    artifact: dict[str, Any], override: float | None = None
) -> ThresholdSelection:
    """Numerical overrides cannot reopen explicitly disabled routing."""
    modern = int(artifact.get("schema_version", 1)) >= 4
    selection = artifact.get("threshold_selection")
    if modern:
        certificate = artifact.get("certification_status", "not_requested")
        if certificate not in {"not_requested", "passed"}:
            return ThresholdSelection("infeasible", None, f"Certification {certificate}")
        if not isinstance(selection, dict) or selection.get("status") != "feasible":
            reason = selection.get("reason") if isinstance(selection, dict) else None
            return ThresholdSelection("infeasible", None, str(reason or "No feasible threshold"))
        value = selection.get("threshold")
    else:
        value = artifact.get("serving_threshold", 0.5)
    try:
        selected_value = override if override is not None else value
        if selected_value is None:
            raise TypeError("missing numeric threshold")
        threshold = float(selected_value)
    except (TypeError, ValueError) as exc:
        raise RouterTrainingError("invalid routing threshold") from exc
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise RouterTrainingError("routing threshold must be finite and between zero and one")
    return ThresholdSelection(
        "feasible" if modern else "legacy",
        threshold,
        None if modern else "Legacy empirical threshold; not certified",
    )


@dataclass(frozen=True)
class ProbabilityCalibrator:
    """Portable scalar temperature calibration for binary probabilities."""

    method: Literal["identity", "temperature"] = "identity"
    temperature: float = 1.0
    fitted_samples: int = 0
    warning: str | None = None

    def transform_one(self, probability: float) -> float:
        clipped = min(1.0 - 1e-7, max(1e-7, float(probability)))
        if self.method == "identity":
            return clipped
        logit = np.log(clipped / (1.0 - clipped)) / self.temperature
        return float(1.0 / (1.0 + np.exp(-logit)))

    def transform(self, probabilities: list[float]) -> list[float]:
        return [self.transform_one(value) for value in probabilities]


def _binary_log_loss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return float(-(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped)).mean())


def fit_probability_calibrator(
    probabilities: list[float], outcomes: list[int]
) -> ProbabilityCalibrator:
    if not probabilities or len(probabilities) != len(outcomes):
        return ProbabilityCalibrator(warning="calibration data was empty or misaligned")
    if len(set(outcomes)) < 2:
        return ProbabilityCalibrator(
            fitted_samples=len(outcomes),
            warning="calibration split contained one outcome class; identity calibration used",
        )
    raw = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    target = np.asarray(outcomes, dtype=float)
    logits = np.log(raw / (1.0 - raw))
    candidates = np.geomspace(0.05, 20.0, 500)
    losses = []
    for temperature in candidates:
        calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
        losses.append(_binary_log_loss(calibrated, target))
    best = float(candidates[int(np.argmin(losses))])
    return ProbabilityCalibrator(
        method="temperature", temperature=best, fitted_samples=len(outcomes)
    )


def choose_threshold(
    probabilities: list[float], outcomes: list[int], target_accuracy: float
) -> ThresholdResult | None:
    if not probabilities or len(probabilities) != len(outcomes):
        return None
    best: ThresholdResult | None = None
    for threshold in sorted(set([0.0, *probabilities, 1.0])):
        mask = np.asarray(probabilities) >= threshold
        if not mask.any():
            continue
        accuracy = float(np.asarray(outcomes, dtype=float)[mask].mean())
        coverage = float(mask.mean())
        candidate = ThresholdResult(threshold, coverage, accuracy)
        if accuracy >= target_accuracy and (best is None or coverage > best.coverage):
            best = candidate
    return best
