from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from budgetroute.config import AppConfig
from budgetroute.features.request_features import RequestFeatureExtractor
from budgetroute.schemas import BenchmarkRecord, GenerationRequest, RouteName


def questions(count: int = 200) -> list[BenchmarkRecord]:
    return [
        BenchmarkRecord(
            id=str(i),
            group_id=str(i),
            category="test",
            prompt=f"Question {i}?",
            reference_answer="A",
            evaluation_type="classification",
        )
        for i in range(count)
    ]


def test_collection_resumes_only_missing_cached_generations(
    fake_config: AppConfig, tmp_path: Path
) -> None:
    from budgetroute.backends.fake import FakeBackend
    from budgetroute.experiments.pilot import _collect

    original = FakeBackend.generate
    calls = 0

    def interrupted(self: FakeBackend, request: GenerationRequest):
        nonlocal calls
        calls += 1
        if calls == 121:
            raise KeyboardInterrupt("runtime disconnected")
        return original(self, request)

    with patch.object(FakeBackend, "generate", interrupted), pytest.raises(KeyboardInterrupt):
        _collect(fake_config, tmp_path, questions())
    calls = 0

    def resumed(self: FakeBackend, request: GenerationRequest):
        nonlocal calls
        calls += 1
        return original(self, request)

    with patch.object(FakeBackend, "generate", resumed):
        _collect(fake_config, tmp_path, questions())
    assert calls == 280
    rows = [
        json.loads(line)
        for line in (tmp_path / "collection/predictions.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len({(r["example_id"], r["policy"]) for r in rows}) == 400
    calls = 0
    with patch.object(FakeBackend, "generate", resumed):
        _collect(fake_config, tmp_path, questions())
    assert calls == 0


def test_matched_random_is_exact_count_and_persisted(tmp_path: Path) -> None:
    from budgetroute.experiments.pilot import _MatchedRandomPolicy

    first = _MatchedRandomPolicy.freeze(
        [str(i) for i in range(40)], 7, 20261021, tmp_path / "assignments.json"
    )
    again = _MatchedRandomPolicy.freeze(
        [str(i) for i in range(40)], 7, 20261021, tmp_path / "assignments.json"
    )
    extractor = RequestFeatureExtractor()
    routes = [
        first.decide(
            GenerationRequest(request_id=str(i), prompt="q"),
            extractor.extract(GenerationRequest(prompt="q")),
        ).route
        for i in range(40)
    ]
    assert routes.count(RouteName.SMALL) == 7
    assert first.small_ids == again.small_ids
    with pytest.raises(ValueError, match="assignment"):
        _MatchedRandomPolicy.freeze(
            [str(i) for i in range(40)], 8, 20261021, tmp_path / "assignments.json"
        )


@pytest.mark.parametrize(
    "changed",
    ["source", "configuration_hash", "dataset_hash", "generation_fingerprint", "environment_hash"],
)
def test_resume_rejects_identity_changes(tmp_path: Path, changed: str) -> None:
    from budgetroute.exceptions import ArtifactError
    from budgetroute.experiments.pilot import _freeze_identity

    identity = {
        key: "original"
        for key in (
            "source",
            "configuration_hash",
            "dataset_hash",
            "generation_fingerprint",
            "environment_hash",
        )
    }
    _freeze_identity(tmp_path, identity)
    with pytest.raises(ArtifactError, match="identity"):
        _freeze_identity(tmp_path, {**identity, changed: "changed"})


def test_timing_initializes_one_service_per_block(fake_config: AppConfig, tmp_path: Path) -> None:
    from budgetroute.experiments.pilot import _timing
    from budgetroute.inference.service import build_service
    from budgetroute.routing.registry import build_policy

    services = []

    def factory(config: AppConfig, **kwargs):
        service = build_service(config, **kwargs)
        services.append(service)
        return service

    def policies(
        config: AppConfig, workspace: Path, records: list[BenchmarkRecord], *, timing: bool
    ):
        return {
            name: build_policy(config.routing.model_copy(update={"policy": name}))
            for name in ("always_small", "always_large")
        }

    with (
        patch("budgetroute.experiments.pilot.build_service", side_effect=factory),
        patch("budgetroute.experiments.pilot._policies", side_effect=policies),
    ):
        _timing(fake_config, tmp_path, questions(40))
    assert len(services) == 3
    for block in range(3):
        result = json.loads((tmp_path / f"timing/block-{block}.json").read_text())
        assert result["status"] == "completed"
        assert len(result["predictions"]) == 80
        assert result["initialization_count"] == 1
        assert result["started_utc"].endswith("+00:00")


def test_mocked_pilot_pipeline_keeps_diagnostic_roles_and_safe_export(
    fake_config: AppConfig, tmp_path: Path
) -> None:
    """Simulation only: exercises orchestration, never produces accepted real evidence."""
    from budgetroute.evaluation.dataset import dataset_hash
    from budgetroute.exceptions import ArtifactError
    from budgetroute.experiments.artifacts import ArtifactWriter
    from budgetroute.experiments.pilot import (
        _collect,
        _diagnostic,
        _export,
        _fit,
        _summary,
        _timing,
        evidence_paths,
    )
    from budgetroute.experiments.pilot_data import PilotSplits
    from budgetroute.retrieval.embeddings import FakeEmbedder
    from budgetroute.validation.colab import validate_run_artifacts

    writer = ArtifactWriter(tmp_path)
    data = writer.write_jsonl(
        "data/pilot.jsonl", [row.model_dump(mode="json") for row in questions()]
    )
    splits = PilotSplits(
        seed=20261021,
        dataset_sha256=dataset_hash(data),
        exclusion_sha256="b" * 64,
        train_ids=[str(i) for i in range(120)],
        calibration_ids=[str(i) for i in range(120, 160)],
        test_ids=[str(i) for i in range(160, 200)],
    )
    writer.write_json("data/splits.json", splits.model_dump())
    writer.write_json(
        "identity.json",
        {
            "source": {"commit": "a" * 40, "dirty": False},
            "hardware_fingerprint": "fixture-hardware",
            "generation_fingerprint": "fixture-weights",
            "dataset_hash": splits.dataset_sha256,
        },
    )
    writer.write_json(
        "environment.json", {"gpu_available": True, "packages": {"fixture": "not-real"}}
    )
    writer.write_yaml("resolved-config.yaml", fake_config.model_dump(mode="json"))
    writer.write_json("data/pilot.manifest.json", {"role": "pilot"})
    writer.write_json(
        "components.json",
        {
            "status": "passed",
            "fake": False,
            "checks": {
                name: {"status": "passed", "fake": False}
                for name in ("generation", "semantic", "retrieval", "scheduling", "api", "cleanup")
            },
        },
    )
    _collect(fake_config, tmp_path, questions())
    with patch(
        "budgetroute.retrieval.embeddings.TransformersEmbedder.embed",
        side_effect=FakeEmbedder(384).embed,
    ):
        _fit(fake_config, tmp_path, questions())
        _diagnostic(fake_config, tmp_path, questions()[160:])
        _timing(fake_config, tmp_path, questions()[160:])
    summary_path = _summary(tmp_path)
    summary = json.loads(summary_path.read_text())
    assert summary["diagnostic_questions"] == 40
    assert summary["router_probability_evaluation"]["semantic_gate"]["probability_count"] == 40
    assert all(v["request_count"] == 40 for v in summary["diagnostic"].values())
    writer.write_json(
        "evidence-manifest.json",
        {p.relative_to(tmp_path).as_posix(): dataset_hash(p) for p in evidence_paths(tmp_path)},
    )
    assert validate_run_artifacts(tmp_path, require_real=False)["status"] == "passed"
    with pytest.raises(ArtifactError, match="fake"):
        validate_run_artifacts(tmp_path)
    import zipfile

    writer._atomic_text(".env", "DO-NOT-EXPORT")
    writer._atomic_text("weights.safetensors", "DO-NOT-EXPORT")
    with patch(
        "budgetroute.validation.colab.validate_run_artifacts",
        side_effect=lambda path: validate_run_artifacts(path, require_real=False),
    ):
        bundle = _export(tmp_path)
    with zipfile.ZipFile(bundle) as archive:
        assert ".env" not in archive.namelist()
        assert "weights.safetensors" not in archive.namelist()
        assert len(archive.namelist()) == len(evidence_paths(tmp_path)) + 1
    broken = json.loads((tmp_path / "timing/block-1.json").read_text())
    broken["status"] = "incomplete"
    broken["predictions"] = broken["predictions"][:1]
    writer.write_json("timing/block-1.json", broken)
    with patch(
        "budgetroute.retrieval.embeddings.TransformersEmbedder.embed",
        side_effect=FakeEmbedder(384).embed,
    ):
        _timing(fake_config, tmp_path, questions()[160:])
    assert len(json.loads((tmp_path / "timing/block-1.json").read_text())["predictions"]) == 240
