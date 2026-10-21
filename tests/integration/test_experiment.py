from __future__ import annotations

import json
from pathlib import Path

from budgetroute.config import AppConfig
from budgetroute.experiments.replay import verify_replay
from budgetroute.experiments.runner import run_benchmark
from budgetroute.inference.service import build_service
from budgetroute.routing.training import (
    evaluate_router,
    train_backend_confidence_calibrator,
    train_router,
)
from budgetroute.schemas import GenerationRequest, RouteName


def test_benchmark_report_and_fake_router(
    fake_config: AppConfig, tmp_path: Path, project_root: Path
) -> None:
    benchmark_config = fake_config.model_copy(
        update={
            "output_dir": tmp_path / "outputs",
            "benchmark": fake_config.benchmark.model_copy(
                update={
                    "policies": ["always_small", "always_large", "heuristic", "cascade"],
                    "fake": True,
                    "cache": fake_config.benchmark.cache.model_copy(
                        update={"directory": tmp_path / "cache"}
                    ),
                }
            ),
        }
    )
    run_dir = run_benchmark(benchmark_config, project_root)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["fake"] is True
    assert metrics["failed_requests"] == 0
    assert (run_dir / "report/figures/quality-latency.png").is_file()
    assert "NOT REAL MODEL PERFORMANCE" in (run_dir / "report/report.md").read_text(
        encoding="utf-8"
    )

    router_path = tmp_path / "router.joblib"
    metadata = train_router(run_dir, router_path, quality_threshold=0.8)
    assert router_path.is_file()
    assert metadata["label_definition"] == "small_model_quality >= 0.8"
    assert metadata["group_disjoint"] is True
    assert set(metadata["splits"]) == {"train", "calibration", "test"}
    evaluation = evaluate_router(router_path, run_dir, threshold=0.8)
    assert 0 <= evaluation["accuracy"] <= 1
    assert evaluation["evaluation_scope"] == "test"

    paired_router_path = tmp_path / "paired-router.joblib"
    paired_metadata = train_router(
        run_dir,
        paired_router_path,
        quality_threshold=0.8,
        label_strategy="paired_quality",
    )
    assert paired_metadata["label_strategy"] == "paired_quality"
    assert paired_metadata["label_definition"] == (
        "small_model_quality >= large_model_quality on paired outcomes"
    )
    assert paired_metadata["outcome_metrics"]["sample_count"] > 0
    paired_evaluation = evaluate_router(paired_router_path, run_dir, threshold=0.8)
    assert paired_evaluation["label_strategy"] == "paired_quality"
    assert -1 <= paired_evaluation["outcome_metrics"]["quality_delta_vs_always_large"] <= 1

    confidence_path = tmp_path / "small-confidence.json"
    confidence = train_backend_confidence_calibrator(run_dir, confidence_path)
    assert confidence_path.is_file()
    assert confidence["kind"] == "backend_confidence_calibrator"

    agreement = verify_replay(benchmark_config, sample_size=2)
    assert agreement["agreement"] == 1.0
    assert agreement["mismatch_count"] == 0

    learned_config = fake_config.model_copy(
        update={
            "routing": fake_config.routing.model_copy(
                update={"policy": "learned", "learned_model_path": router_path}
            )
        }
    )
    service = build_service(learned_config)
    try:
        response = service.generate(GenerationRequest(prompt="What is the capital of France?"))
        assert response.router.policy == "learned"
        assert response.route in {RouteName.SMALL, RouteName.LARGE}
        assert response.router.router_probability is not None
        assert response.router.router_probability.target == "small_success"
        from budgetroute.evaluation.dataset import load_dataset
        from budgetroute.evaluation.harness import evaluate_records

        records = load_dataset(learned_config.benchmark.dataset_path)[:2]
        predictions, _, _ = evaluate_records(service, records, fake=True)
        assert all(row.router_probability is not None for row in predictions)
        assert all(row.fake is True for row in predictions)
    finally:
        service.close()


def test_impossible_calibration_routes_all_large(tmp_path: Path) -> None:
    rows = [
        {
            "example_id": f"bad-{index}",
            "group_id": f"group-{index}",
            "policy": "always_small",
            "error": None,
            "backend_confidence_raw": 0.9,
            "quality_score": 0.0,
        }
        for index in range(12)
    ]
    (tmp_path / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "confidence.json"
    metadata = train_backend_confidence_calibrator(tmp_path, output, target_selective_accuracy=0.8)
    assert metadata["serving_threshold"] is None
    assert metadata["threshold_selection"]["status"] == "infeasible"
    assert metadata["metrics"]["confusion_matrix"] == [[len(metadata["test_ids"]), 0], [0, 0]]
    json.dumps(metadata, allow_nan=False)


def test_legacy_calibrator_remains_readable_and_uncertified(tmp_path: Path) -> None:
    from budgetroute.routing.training import load_backend_confidence_calibrator

    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({"schema_version": 1, "serving_threshold": 0.7, "calibrator": {}}),
        encoding="utf-8",
    )
    _, selection, _ = load_backend_confidence_calibrator(path)
    assert selection.status == "legacy"
    assert selection.threshold == 0.7
