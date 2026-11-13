from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest

from budgetroute.features.request_features import RequestFeatureExtractor
from budgetroute.schemas import GenerationRequest


def write_training(tmp_path: Path) -> tuple[Path, Path]:
    from budgetroute.evaluation.dataset import dataset_hash
    from budgetroute.experiments.pilot_data import PilotSplits

    rows, predictions, routes = [], [], []
    for i in range(200):
        key = str(i)
        rows.append(
            {
                "id": key,
                "group_id": key,
                "category": "test",
                "prompt": f"Prompt {i}",
                "reference_answer": "SECRET_REFERENCE",
                "evaluation_type": "classification",
            }
        )
        for policy in ("always_small", "always_large"):
            predictions.append(
                {
                    "example_id": key,
                    "group_id": key,
                    "policy": policy,
                    "quality_score": i % 2,
                    "raw_answer": "SECRET_OUTPUT",
                }
            )
            routes.append({"example_id": key, "policy": policy, "features": {"token_count": i}})
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
    (tmp_path / "routes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in routes))
    splits = PilotSplits(
        seed=20261021,
        dataset_sha256=dataset_hash(data),
        exclusion_sha256="a" * 64,
        train_ids=[str(i) for i in range(120)],
        calibration_ids=[str(i) for i in range(120, 160)],
        test_ids=[str(i) for i in range(160, 200)],
    )
    split_file = tmp_path / "splits.json"
    split_file.write_text(splits.model_dump_json())
    return data, split_file


def vectors(texts: list[str]) -> np.ndarray:
    return np.array([[float(t.split()[-1]) % 2, 1.0] for t in texts], dtype=np.float32)


def test_semantic_inputs_exclude_labels_and_large_outputs(tmp_path: Path) -> None:
    from budgetroute.features.semantic import SemanticFeatures, SemanticFeatureSpec
    from budgetroute.routing.learned import LearnedPolicy
    from budgetroute.routing.training import train_router

    data, splits = write_training(tmp_path)
    observed: list[str] = []

    def embed(texts: list[str]) -> np.ndarray:
        observed.extend(texts)
        return vectors(texts)

    spec = SemanticFeatureSpec()
    with (
        patch("budgetroute.retrieval.embeddings.TransformersEmbedder.embed", side_effect=embed),
        patch(
            "budgetroute.routing.training._group_partitions",
            side_effect=AssertionError("must not resplit"),
        ),
    ):
        metadata = train_router(
            tmp_path,
            tmp_path / "router.joblib",
            split_manifest=splits,
            feature_mode="semantic",
            dataset_path=data,
            semantic_spec=spec,
            label_strategy="paired_quality",
        )
        request = GenerationRequest(prompt="Prompt 7")
        features = RequestFeatureExtractor().extract(request)
        policy = LearnedPolicy(tmp_path / "router.joblib")
        before = policy.decide(request, features)
        after = LearnedPolicy(tmp_path / "router.joblib").decide(request, features)
        assert before.model_dump() == after.model_dump()
        assert metadata["feature_dimension"] == 2
        assert metadata["split_role"] == "pilot"
        from budgetroute.routing.training import evaluate_router

        result = evaluate_router(tmp_path / "router.joblib", tmp_path, dataset_path=data)
        assert result["raw_brier_score"] == metadata["metrics"]["raw_brier_score"]
        assert all(t.startswith("Prompt ") for t in observed)
        encoder = SemanticFeatures(spec)
        assert np.array_equal(encoder.transform(["Prompt 7"]), vectors(["Prompt 7"]))
        encoder.close()
        policy.close()


@pytest.mark.parametrize("change", ["overlap", "unknown", "hash", "missing_prompt"])
def test_pilot_splits_fail_closed(tmp_path: Path, change: str) -> None:
    from budgetroute.routing.training import train_router

    data, splits = write_training(tmp_path)
    raw = json.loads(splits.read_text())
    if change == "overlap":
        raw["test_ids"][0] = raw["train_ids"][0]
    if change == "unknown":
        raw["test_ids"][0] = "unknown"
    if change == "hash":
        raw["dataset_sha256"] = "b" * 64
    if change == "missing_prompt":
        data.write_text("\n".join(data.read_text().splitlines()[:-1]) + "\n")
    splits.write_text(json.dumps(raw))
    with pytest.raises(
        (
            ValueError,
            __import__(
                "budgetroute.exceptions", fromlist=["RouterTrainingError"]
            ).RouterTrainingError,
        )
    ):
        train_router(tmp_path, tmp_path / "router.joblib", split_manifest=splits, dataset_path=data)


def test_semantic_revision_dimension_and_preprocessing_are_checked(tmp_path: Path) -> None:
    from budgetroute.exceptions import RouterTrainingError
    from budgetroute.features.semantic import SemanticFeatureSpec
    from budgetroute.routing.learned import LearnedPolicy

    with pytest.raises(ValueError):
        SemanticFeatureSpec(revision="main")
    data, splits = write_training(tmp_path)
    from budgetroute.routing.training import train_router

    with patch("budgetroute.retrieval.embeddings.TransformersEmbedder.embed", side_effect=vectors):
        train_router(
            tmp_path,
            tmp_path / "router.joblib",
            split_manifest=splits,
            dataset_path=data,
            feature_mode="semantic",
        )
        artifact = joblib.load(tmp_path / "router.joblib")
        artifact["feature_dimension"] = 9
        joblib.dump(artifact, tmp_path / "bad.joblib")
        with pytest.raises(RouterTrainingError, match="dimension"):
            LearnedPolicy(tmp_path / "bad.joblib").decide(
                GenerationRequest(prompt="Prompt 1"),
                RequestFeatureExtractor().extract(GenerationRequest(prompt="Prompt 1")),
            )
        artifact["preprocessing_identity"] = "unknown"
        joblib.dump(artifact, tmp_path / "bad.joblib")
        with pytest.raises(RouterTrainingError, match="preprocessing"):
            LearnedPolicy(tmp_path / "bad.joblib").decide(
                GenerationRequest(prompt="Prompt 1"),
                RequestFeatureExtractor().extract(GenerationRequest(prompt="Prompt 1")),
            )


def test_cleanup_does_not_initialize_encoder() -> None:
    from budgetroute.features.semantic import SemanticFeatures, SemanticFeatureSpec

    encoder = SemanticFeatures(SemanticFeatureSpec())
    with patch.object(
        encoder.embedder, "_initialize", side_effect=AssertionError("unexpected load")
    ):
        encoder.embedder._model = object()
        encoder.embedder._tokenizer = object()
        encoder.close()
    assert encoder.embedder._model is encoder.embedder._tokenizer is None


def test_numeric_v3_never_constructs_semantic_encoder(tmp_path: Path) -> None:
    from budgetroute.routing.learned import LearnedPolicy
    from budgetroute.routing.training import train_router

    write_training(tmp_path)
    train_router(tmp_path, tmp_path / "router.joblib")
    artifact = joblib.load(tmp_path / "router.joblib")
    artifact.pop("feature_mode")
    artifact["schema_version"] = 3
    joblib.dump(artifact, tmp_path / "legacy.joblib")
    with patch(
        "budgetroute.routing.learned.SemanticFeatures",
        side_effect=AssertionError("legacy numeric loaded encoder"),
    ):
        request = GenerationRequest(prompt="Prompt 1")
        policy = LearnedPolicy(tmp_path / "legacy.joblib")
        policy.decide(request, RequestFeatureExtractor().extract(request))
        policy.close()
