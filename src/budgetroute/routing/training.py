"""Leakage-safe learned-router training, calibration, persistence, and evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from budgetroute.evaluation.calibration_metrics import (
    brier_score,
    expected_calibration_error,
)
from budgetroute.evaluation.dataset import dataset_hash, load_dataset
from budgetroute.exceptions import RouterTrainingError
from budgetroute.experiments.pilot_data import PilotSplits
from budgetroute.features.semantic import (
    PREPROCESSING_IDENTITY,
    SemanticFeatures,
    SemanticFeatureSpec,
)
from budgetroute.routing.calibration import (
    ProbabilityCalibrator,
    ThresholdSelection,
    choose_threshold,
    fit_probability_calibrator,
    resolve_threshold_selection,
)


@dataclass(frozen=True)
class TrainingRows:
    features: list[dict[str, float]]
    labels: list[int]
    ids: list[str]
    groups: list[str]
    small_quality: list[float]
    large_quality: list[float | None]


LabelStrategy = Literal["small_success", "paired_quality"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RouterTrainingError(f"required training artifact does not exist: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_training_rows(
    artifact_dir: Path,
    quality_threshold: float,
    label_strategy: LabelStrategy = "small_success",
) -> TrainingRows:
    if label_strategy not in ("small_success", "paired_quality"):
        raise RouterTrainingError(f"unsupported router label strategy: {label_strategy}")
    predictions = _read_jsonl(artifact_dir / "predictions.jsonl")
    routes = _read_jsonl(artifact_dir / "routes.jsonl")
    route_lookup = {(str(item["example_id"]), str(item.get("policy", ""))): item for item in routes}
    preferred = [item for item in predictions if item.get("policy") == "always_small"]
    if label_strategy == "paired_quality" and not preferred:
        raise RouterTrainingError("paired-quality training requires always_small predictions")
    selected = preferred or predictions
    large_lookup = {
        str(item["example_id"]): item
        for item in predictions
        if item.get("policy") == "always_large" and item.get("error") is None
    }
    features: list[dict[str, float]] = []
    labels: list[int] = []
    ids: list[str] = []
    groups: list[str] = []
    small_quality: list[float] = []
    large_quality: list[float | None] = []
    missing_pairs: list[str] = []
    for item in selected:
        if item.get("error"):
            continue
        route = route_lookup.get((str(item["example_id"]), str(item.get("policy", ""))))
        if not route or not isinstance(route.get("features"), dict):
            continue
        numeric = {
            key: float(value)
            for key, value in route["features"].items()
            if isinstance(value, (bool, int, float))
        }
        example_id = str(item["example_id"])
        small_score = float(item["quality_score"])
        large_item = large_lookup.get(example_id)
        large_score = float(large_item["quality_score"]) if large_item is not None else None
        if label_strategy == "paired_quality" and large_score is None:
            missing_pairs.append(example_id)
            continue
        features.append(numeric)
        if label_strategy == "small_success":
            label = int(small_score >= quality_threshold)
        else:
            assert large_score is not None
            label = int(small_score >= large_score)
        labels.append(label)
        ids.append(example_id)
        groups.append(str(item.get("group_id") or route.get("group_id") or item["example_id"]))
        small_quality.append(small_score)
        large_quality.append(large_score)
    if missing_pairs:
        raise RouterTrainingError(
            "paired-quality training requires successful always_large predictions for every "
            f"always_small row; missing {len(missing_pairs)} pairs"
        )
    if not features:
        raise RouterTrainingError("no joined prediction/feature rows were available for training")
    return TrainingRows(features, labels, ids, groups, small_quality, large_quality)


def _label_definition(label_strategy: LabelStrategy, quality_threshold: float) -> str:
    if label_strategy == "paired_quality":
        return "small_model_quality >= large_model_quality on paired outcomes"
    return f"small_model_quality >= {quality_threshold}"


def _routing_outcome_metrics(
    rows: TrainingRows,
    indices: np.ndarray,
    route_small: list[int],
    quality_threshold: float,
) -> dict[str, Any] | None:
    if len(indices) != len(route_small):
        raise RouterTrainingError("route decisions do not match the selected outcome rows")
    large_scores = [rows.large_quality[int(index)] for index in indices]
    if any(value is None for value in large_scores):
        return None
    small_scores = [rows.small_quality[int(index)] for index in indices]
    paired_large = [float(value) for value in large_scores if value is not None]
    routed_scores = [
        small if use_small else large
        for small, large, use_small in zip(small_scores, paired_large, route_small, strict=True)
    ]
    oracle_scores = [
        max(small, large) for small, large in zip(small_scores, paired_large, strict=True)
    ]

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values))

    return {
        "sample_count": len(routed_scores),
        "small_route_count": sum(route_small),
        "large_route_count": len(route_small) - sum(route_small),
        "small_route_share": sum(route_small) / len(route_small),
        "routed_mean_quality": mean(routed_scores),
        "routed_success_rate": mean([float(value >= quality_threshold) for value in routed_scores]),
        "always_small_mean_quality": mean(small_scores),
        "always_large_mean_quality": mean(paired_large),
        "oracle_mean_quality": mean(oracle_scores),
        "quality_delta_vs_always_large": mean(routed_scores) - mean(paired_large),
        "quality_regret_vs_oracle": mean(oracle_scores) - mean(routed_scores),
    }


def _classification_metrics(
    labels: list[int], predictions: list[int], probabilities: list[float]
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if not labels:
        raise RouterTrainingError("cannot calculate router metrics for an empty split")
    unique = set(labels)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(unique) == 2 else None,
        "brier_score": brier_score(probabilities, labels),
        "expected_calibration_error": expected_calibration_error(probabilities, labels),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "route_distribution": {
            "small": sum(predictions) / len(predictions),
            "large": 1 - sum(predictions) / len(predictions),
        },
    }


def _probabilities(model: Any, matrix: np.ndarray) -> list[float]:
    raw = model.predict_proba(matrix)
    classes = list(model.classes_)
    return (
        raw[:, classes.index(1)].astype(float).tolist()
        if 1 in classes
        else [float(classes[0] == 1)] * len(matrix)
    )


def _group_partitions(
    groups: list[str], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    from sklearn.model_selection import GroupShuffleSplit

    indices = np.arange(len(groups))
    unique_groups = set(groups)
    if len(indices) < 9 or len(unique_groups) < 6:
        return (
            indices,
            indices,
            indices,
            "too few samples or groups for disjoint partitions; all metrics are in-sample",
        )
    group_values = np.asarray(groups)
    first = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=seed)
    train_indices, held_out_indices = next(first.split(indices, groups=group_values))
    held_out_groups = group_values[held_out_indices]
    if len(set(held_out_groups)) < 2:
        return (
            indices,
            indices,
            indices,
            "group split could not create calibration and test sets; all metrics are in-sample",
        )
    second = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    calibration_relative, test_relative = next(
        second.split(held_out_indices, groups=held_out_groups)
    )
    return (
        train_indices,
        held_out_indices[calibration_relative],
        held_out_indices[test_relative],
        None,
    )


def _split_summary(
    indices: np.ndarray, ids: list[str], groups: list[str], labels: list[int]
) -> dict[str, Any]:
    selected_labels = [labels[int(index)] for index in indices]
    return {
        "sample_count": len(indices),
        "group_count": len({groups[int(index)] for index in indices}),
        "class_counts": {
            "failure": selected_labels.count(0),
            "success": selected_labels.count(1),
        },
        "ids": [ids[int(index)] for index in indices],
        "groups": sorted({groups[int(index)] for index in indices}),
    }


def train_router(
    artifact_dir: Path,
    output_path: Path,
    quality_threshold: float = 0.8,
    seed: int = 42,
    target_selective_accuracy: float = 0.8,
    label_strategy: LabelStrategy = "small_success",
    *,
    split_manifest: Path | None = None,
    feature_mode: Literal["numeric", "semantic"] = "numeric",
    dataset_path: Path | None = None,
    semantic_spec: SemanticFeatureSpec | None = None,
) -> dict[str, Any]:
    try:
        import joblib
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RouterTrainingError(
            "router training requires scikit-learn and joblib; install the learned extra"
        ) from exc

    rows = build_training_rows(artifact_dir, quality_threshold, label_strategy)
    feature_order = sorted({name for item in rows.features for name in item})
    matrix = np.asarray(
        [[item.get(name, 0.0) for name in feature_order] for item in rows.features], dtype=float
    )
    target = np.asarray(rows.labels, dtype=int)
    truncation_count = 0
    if feature_mode not in {"numeric", "semantic"}:
        raise RouterTrainingError("unsupported feature mode")
    if feature_mode == "semantic":
        if dataset_path is None:
            raise RouterTrainingError("semantic features require a prompt dataset")
        prompts = {record.id: record.prompt for record in load_dataset(dataset_path)}
        if not set(rows.ids) <= prompts.keys():
            raise RouterTrainingError("missing prompt joins")
        encoder = SemanticFeatures(semantic_spec or SemanticFeatureSpec())
        try:
            matrix = encoder.transform([prompts[key] for key in rows.ids])
            truncation_count = encoder.truncation_count
        finally:
            encoder.close()
        feature_order = [f"embedding_{i}" for i in range(matrix.shape[1])]
    if split_manifest is None:
        train_indices, calibration_indices, test_indices, warning = _group_partitions(
            rows.groups, seed
        )
    else:
        manifest = PilotSplits.model_validate_json(split_manifest.read_text(encoding="utf-8"))
        if dataset_path is None or dataset_hash(dataset_path) != manifest.dataset_sha256:
            raise RouterTrainingError("pilot prompt dataset hash mismatch")
        partitions = [manifest.train_ids, manifest.calibration_ids, manifest.test_ids]
        if len(set(rows.ids)) != len(rows.ids) or set(rows.ids) != {
            key for part in partitions for key in part
        }:
            raise RouterTrainingError("pilot IDs missing, unknown or duplicated")
        lookup = {key: i for i, key in enumerate(rows.ids)}
        train_indices, calibration_indices, test_indices = [
            np.asarray([lookup[key] for key in part]) for part in partitions
        ]
        groups = [
            {rows.groups[int(i)] for i in indices}
            for indices in (train_indices, calibration_indices, test_indices)
        ]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise RouterTrainingError("pilot group leakage")
        warning = None
    train_labels = target[train_indices]
    if len(set(train_labels.tolist())) == 1:
        model: Any = DummyClassifier(strategy="constant", constant=int(train_labels[0]))
        warning = warning or "training split contained one class; constant classifier used"
    else:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
    model.fit(matrix[train_indices], train_labels)

    raw_calibration_probabilities = _probabilities(model, matrix[calibration_indices])
    calibration_labels = target[calibration_indices].astype(int).tolist()
    calibrator = fit_probability_calibrator(raw_calibration_probabilities, calibration_labels)
    calibrated_probabilities = calibrator.transform(raw_calibration_probabilities)
    threshold_result = choose_threshold(
        calibrated_probabilities, calibration_labels, target_selective_accuracy
    )
    selection = (
        ThresholdSelection("feasible", threshold_result.threshold)
        if threshold_result
        else ThresholdSelection(
            "infeasible", None, "No calibration subset met the requested accuracy"
        )
    )
    serving_threshold = selection.threshold

    raw_test_probabilities = _probabilities(model, matrix[test_indices])
    test_probabilities = calibrator.transform(raw_test_probabilities)
    test_labels = target[test_indices].astype(int).tolist()
    test_predictions = [int(selection.accepts(value)) for value in test_probabilities]
    metrics = _classification_metrics(test_labels, test_predictions, test_probabilities)
    metrics["raw_brier_score"] = brier_score(raw_test_probabilities, test_labels)
    metrics["raw_expected_calibration_error"] = expected_calibration_error(
        raw_test_probabilities, test_labels
    )
    outcome_metrics = _routing_outcome_metrics(
        rows, test_indices, test_predictions, quality_threshold
    )

    source_hash = hashlib.sha256(
        (artifact_dir / "predictions.jsonl").read_bytes()
        + (artifact_dir / "routes.jsonl").read_bytes()
    ).hexdigest()
    splits = {
        "train": _split_summary(train_indices, rows.ids, rows.groups, rows.labels),
        "calibration": _split_summary(calibration_indices, rows.ids, rows.groups, rows.labels),
        "test": _split_summary(test_indices, rows.ids, rows.groups, rows.labels),
    }
    train_groups = set(splits["train"]["groups"])
    calibration_groups = set(splits["calibration"]["groups"])
    test_groups = set(splits["test"]["groups"])
    disjoint = not (
        train_groups & calibration_groups
        or train_groups & test_groups
        or calibration_groups & test_groups
    )
    if warning is None and not disjoint:
        raise RouterTrainingError("group leakage was detected across router partitions")
    metadata = {
        "schema_version": 4,
        "feature_mode": feature_mode,
        "feature_dimension": int(matrix.shape[1]),
        "semantic_spec": (semantic_spec or SemanticFeatureSpec()).model_dump()
        if feature_mode == "semantic"
        else None,
        "preprocessing_identity": PREPROCESSING_IDENTITY
        if feature_mode == "semantic"
        else "numeric_snapshot_v1",
        "embedding_truncation_count": truncation_count,
        "split_role": "pilot" if split_manifest else "development",
        "label_strategy": label_strategy,
        "label_definition": _label_definition(label_strategy, quality_threshold),
        "quality_threshold": quality_threshold,
        "target_selective_accuracy": target_selective_accuracy,
        "serving_threshold": serving_threshold,
        "threshold_selection": asdict(selection),
        "empirical_threshold_metrics": asdict(threshold_result) if threshold_result else None,
        "certification_status": "not_requested",
        "calibrator": asdict(calibrator),
        "seed": seed,
        "dataset_hash": source_hash,
        "sample_count": len(rows.labels),
        "splits": splits,
        "group_disjoint": disjoint,
        "train_ids": splits["train"]["ids"],
        "calibration_ids": splits["calibration"]["ids"],
        "test_ids": splits["test"]["ids"],
        "metrics": metrics,
        "outcome_metrics": outcome_metrics,
        "warning": warning or calibrator.warning,
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "joblib")
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    joblib.dump(
        {
            "schema_version": 4,
            "model": model,
            "calibrator": calibrator,
            "serving_threshold": serving_threshold,
            "threshold_selection": asdict(selection),
            "certification_status": "not_requested",
            "feature_order": feature_order,
            **{
                key: metadata[key]
                for key in (
                    "feature_mode",
                    "feature_dimension",
                    "semantic_spec",
                    "preprocessing_identity",
                )
            },
            "label_definition": metadata["label_definition"],
            "label_strategy": label_strategy,
            "training_metadata": metadata,
        },
        temporary,
    )
    temporary.replace(output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def evaluate_router(
    artifact_path: Path,
    artifact_dir: Path,
    threshold: float = 0.8,
    *,
    test_only: bool = True,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    try:
        import joblib
    except ImportError as exc:
        raise RouterTrainingError("router evaluation requires joblib") from exc
    artifact = joblib.load(artifact_path)
    training_metadata = artifact.get("training_metadata", {})
    label_strategy: LabelStrategy = training_metadata.get("label_strategy", "small_success")
    quality_threshold = float(training_metadata.get("quality_threshold", threshold))
    rows = build_training_rows(artifact_dir, quality_threshold, label_strategy)
    order = [str(item) for item in artifact["feature_order"]]
    matrix = np.asarray([[item.get(name, 0.0) for name in order] for item in rows.features])
    if artifact.get("feature_mode") == "semantic":
        if dataset_path is None or artifact.get("preprocessing_identity") != PREPROCESSING_IDENTITY:
            raise RouterTrainingError("semantic evaluation requires matching prompt preprocessing")
        prompts = {row.id: row.prompt for row in load_dataset(dataset_path)}
        if not set(rows.ids) <= prompts.keys():
            raise RouterTrainingError("missing semantic evaluation prompt joins")
        encoder = SemanticFeatures(SemanticFeatureSpec.model_validate(artifact["semantic_spec"]))
        try:
            matrix = encoder.transform([prompts[key] for key in rows.ids])
        finally:
            encoder.close()
        if matrix.shape[1] != artifact["feature_dimension"]:
            raise RouterTrainingError("semantic evaluation dimension mismatch")
    selected = np.arange(len(rows.ids))
    if test_only and training_metadata.get("test_ids"):
        test_ids = set(str(item) for item in training_metadata["test_ids"])
        selected = np.asarray([index for index, item in enumerate(rows.ids) if item in test_ids])
        if not len(selected):
            raise RouterTrainingError("none of the persisted test IDs exist in these artifacts")
    raw_probabilities = _probabilities(artifact["model"], matrix[selected])
    calibrator = artifact.get("calibrator", ProbabilityCalibrator())
    probabilities = calibrator.transform(raw_probabilities)
    selection = resolve_threshold_selection(artifact)
    serving_threshold = selection.threshold
    labels = [rows.labels[int(index)] for index in selected]
    predictions = [int(selection.accepts(value)) for value in probabilities]
    result = _classification_metrics(labels, predictions, probabilities)
    result.update(
        {
            "evaluation_scope": "test" if test_only else "all",
            "evaluation_ids": [rows.ids[int(index)] for index in selected],
            "serving_threshold": serving_threshold,
            "raw_brier_score": brier_score(raw_probabilities, labels),
            "raw_expected_calibration_error": expected_calibration_error(raw_probabilities, labels),
            "label_strategy": label_strategy,
            "outcome_metrics": _routing_outcome_metrics(
                rows, selected, predictions, quality_threshold
            ),
        }
    )
    return result


def train_backend_confidence_calibrator(
    artifact_dir: Path,
    output_path: Path,
    *,
    policy: str = "always_small",
    quality_threshold: float = 0.8,
    target_selective_accuracy: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    predictions = [
        item
        for item in _read_jsonl(artifact_dir / "predictions.jsonl")
        if item.get("policy") == policy
        and item.get("error") is None
        and item.get("backend_confidence_raw") is not None
    ]
    if not predictions:
        raise RouterTrainingError(
            f"no raw backend-confidence rows were available for policy {policy!r}"
        )
    groups = [str(item.get("group_id") or item["example_id"]) for item in predictions]
    _, calibration_indices, test_indices, warning = _group_partitions(groups, seed)
    raw = [float(item["backend_confidence_raw"]) for item in predictions]
    labels = [int(float(item["quality_score"]) >= quality_threshold) for item in predictions]
    calibration_raw = [raw[int(index)] for index in calibration_indices]
    calibration_labels = [labels[int(index)] for index in calibration_indices]
    calibrator = fit_probability_calibrator(calibration_raw, calibration_labels)
    calibration_probabilities = calibrator.transform(calibration_raw)
    threshold_result = choose_threshold(
        calibration_probabilities, calibration_labels, target_selective_accuracy
    )
    selection = (
        ThresholdSelection("feasible", threshold_result.threshold)
        if threshold_result
        else ThresholdSelection(
            "infeasible", None, "No calibration subset met the requested accuracy"
        )
    )
    serving_threshold = selection.threshold
    test_raw = [raw[int(index)] for index in test_indices]
    test_labels = [labels[int(index)] for index in test_indices]
    test_probabilities = calibrator.transform(test_raw)
    test_predictions = [int(selection.accepts(value)) for value in test_probabilities]
    metadata = {
        "schema_version": 4,
        "kind": "backend_confidence_calibrator",
        "policy": policy,
        "quality_threshold": quality_threshold,
        "target_selective_accuracy": target_selective_accuracy,
        "serving_threshold": serving_threshold,
        "threshold_selection": asdict(selection),
        "empirical_threshold_metrics": asdict(threshold_result) if threshold_result else None,
        "certification_status": "not_requested",
        "calibrator": asdict(calibrator),
        "calibration_ids": [predictions[int(index)]["example_id"] for index in calibration_indices],
        "test_ids": [predictions[int(index)]["example_id"] for index in test_indices],
        "metrics": _classification_metrics(test_labels, test_predictions, test_probabilities),
        "warning": warning or calibrator.warning,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_backend_confidence_calibrator(
    path: Path,
) -> tuple[ProbabilityCalibrator, ThresholdSelection, dict[str, Any]]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        calibrator = ProbabilityCalibrator(**metadata["calibrator"])
        selection = resolve_threshold_selection(metadata)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RouterTrainingError(f"invalid backend confidence calibrator {path}: {exc}") from exc
    return calibrator, selection, metadata
