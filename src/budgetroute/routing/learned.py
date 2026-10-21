"""Persisted scikit-learn small-model-success routing policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from budgetroute.exceptions import RouterTrainingError
from budgetroute.routing.base import feature_snapshot
from budgetroute.routing.calibration import ProbabilityCalibrator, resolve_threshold_selection
from budgetroute.schemas import (
    GenerationRequest,
    RequestFeatures,
    RouteDecision,
    RouteName,
    RouterProbability,
)


class LearnedPolicy:
    name = "learned"

    def __init__(self, artifact_path: Path, success_threshold: float | None = None) -> None:
        self.artifact_path = artifact_path
        self.success_threshold = success_threshold
        self._artifact: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._artifact is not None:
            return self._artifact
        if importlib.util.find_spec("joblib") is None:
            raise RouterTrainingError("learned routing requires the learned optional dependency")
        if not self.artifact_path.is_file():
            raise RouterTrainingError(
                f"learned router artifact does not exist: {self.artifact_path}"
            )
        import joblib

        artifact = joblib.load(self.artifact_path)
        if not isinstance(artifact, dict) or not {"model", "feature_order"} <= artifact.keys():
            raise RouterTrainingError("learned router artifact has an invalid schema")
        self._artifact = artifact
        return artifact

    def decide(self, request: GenerationRequest, features: RequestFeatures) -> RouteDecision:
        artifact = self._load()
        numeric = features.numeric_snapshot()
        feature_order = [str(item) for item in artifact["feature_order"]]
        vector = np.asarray([[numeric.get(name, 0.0) for name in feature_order]], dtype=float)
        model = artifact["model"]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vector)[0]
            classes = list(model.classes_)
            raw_success_probability = (
                float(probabilities[classes.index(1)]) if 1 in classes else float(classes[0] == 1)
            )
        else:
            raw_success_probability = float(model.predict(vector)[0])
        calibrator = artifact.get("calibrator", ProbabilityCalibrator())
        success_probability = calibrator.transform_one(raw_success_probability)
        selection = resolve_threshold_selection(artifact, self.success_threshold)
        route = RouteName.SMALL if selection.accepts(success_probability) else RouteName.LARGE
        return RouteDecision(
            route=route,
            confidence=max(success_probability, 1.0 - success_probability),
            difficulty_score=1.0 - success_probability,
            reason=(
                f"Routing infeasible: {selection.reason}"
                if selection.status == "infeasible"
                else f"Predicted small-model success {success_probability:.3f} is "
                f"{'above' if route == RouteName.SMALL else 'below'} the serving threshold"
            ),
            policy=self.name,
            features=feature_snapshot(features),
            thresholds={"small_success": selection.threshold}
            if selection.threshold is not None
            else {},
            router_probability=RouterProbability(
                raw_probability=raw_success_probability,
                probability=success_probability,
                target=artifact.get(
                    "label_strategy",
                    artifact.get("training_metadata", {}).get("label_strategy", "small_success"),
                ),
                calibrated=calibrator.method != "identity",
            ),
        )
