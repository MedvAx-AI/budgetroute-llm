"""Benchmark runner that creates artifacts from actual executions only."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from budgetroute.config import AppConfig
from budgetroute.evaluation.dataset import (
    dataset_hash,
    load_dataset,
    validate_dataset_manifest,
)
from budgetroute.evaluation.harness import evaluate_records
from budgetroute.evaluation.metrics import aggregate_metrics
from budgetroute.evaluation.routing_metrics import (
    router_probability_metrics,
    routing_quality_metrics,
)
from budgetroute.experiments.artifacts import ArtifactWriter, create_run_directory
from budgetroute.experiments.metadata import environment_metadata, run_metadata
from budgetroute.inference.service import build_service
from budgetroute.reporting.report import generate_report
from budgetroute.schemas import BenchmarkRecord, GenerationRequest, PredictionRecord


def _measured_records(records: list[BenchmarkRecord], measured_runs: int) -> list[BenchmarkRecord]:
    if measured_runs == 1:
        return records
    result: list[BenchmarkRecord] = []
    for run_number in range(measured_runs):
        for record in records:
            result.append(record.model_copy(update={"id": f"{record.id}__run-{run_number:03d}"}))
    return result


def _warm_up(service: Any, records: list[BenchmarkRecord], runs: int, fake: bool) -> float:
    if not records:
        return 0.0
    start = time.perf_counter_ns()
    for index in range(runs):
        record = records[index % len(records)]
        metadata = dict(record.metadata)
        metadata["must_abstain"] = record.must_abstain
        # Keep warm-up generations out of measured cache entries. Request IDs are
        # intentionally excluded from cache keys, so an explicit marker is needed.
        metadata["benchmark_warmup"] = True
        if fake:
            metadata["fake_reference_answer"] = record.reference_answer
            metadata["fake_small_success"] = True
        service.generate(
            GenerationRequest(
                request_id=f"warmup-{index}",
                prompt=record.prompt,
                requires_retrieval=record.requires_retrieval,
                metadata=metadata,
            )
        )
    return (time.perf_counter_ns() - start) / 1_000_000


def run_benchmark(config: AppConfig, repository_root: Path | None = None) -> Path:
    root = (repository_root or Path.cwd()).resolve()
    records = load_dataset(config.benchmark.dataset_path)
    dataset_manifest: dict[str, Any] | None = None
    if config.benchmark.dataset_manifest_path is not None:
        dataset_manifest = validate_dataset_manifest(
            config.benchmark.dataset_path, config.benchmark.dataset_manifest_path
        ).model_dump(mode="json")
    if config.benchmark.fake and (
        config.small_backend.type != "fake" or config.large_backend.type != "fake"
    ):
        raise ValueError("a fake benchmark must use fake backends")
    slug = "fake-smoke" if config.benchmark.fake else "benchmark"
    run_id, run_dir = create_run_directory(config.output_dir, slug)
    writer = ArtifactWriter(run_dir)
    writer.write_yaml("config.resolved.yaml", config.model_dump(mode="json"))
    writer.write_json("environment.json", environment_metadata())

    all_predictions: list[PredictionRecord] = []
    all_routes: list[dict[str, Any]] = []
    all_timings: list[dict[str, Any]] = []
    policy_metrics: dict[str, Any] = {}
    backends: dict[str, Any] = {}
    total_measurement_wall_ms = 0.0
    measured = _measured_records(records, config.benchmark.measured_runs)
    for policy_name in config.benchmark.policies:
        policy_config = config.routing.model_copy(update={"policy": policy_name})
        policy_run_config = config.model_copy(update={"routing": policy_config})
        service = build_service(policy_run_config)
        try:
            service.initialize()
            warmup_runs = (
                0 if config.benchmark.cache.mode == "read_only" else config.benchmark.warmup_runs
            )
            warmup_ms = _warm_up(service, records, warmup_runs, config.benchmark.fake)
            measurement_started = time.perf_counter_ns()
            predictions, routes, timings = evaluate_records(
                service,
                measured,
                fake=config.benchmark.fake,
                batch_size=config.benchmark.batch_size,
                concurrency=config.benchmark.concurrency,
            )
            measurement_wall_ms = (time.perf_counter_ns() - measurement_started) / 1_000_000
            total_measurement_wall_ms += measurement_wall_ms
            all_predictions.extend(predictions)
            all_routes.extend(routes)
            all_timings.extend({"policy": policy_name, **item} for item in timings)
            policy_metrics[policy_name] = aggregate_metrics(
                predictions,
                wall_time_ms=measurement_wall_ms,
                bootstrap_samples=config.benchmark.bootstrap_samples,
                minimum_samples_for_claims=config.benchmark.minimum_samples_for_claims,
                quality_threshold=config.benchmark.quality_threshold,
                seed=config.seed,
            )
            backends[policy_name] = service.metadata()
            backends[policy_name]["warmup_ms"] = warmup_ms
        finally:
            service.close()

    small_success = {
        item.example_id: int(item.quality_score >= config.benchmark.quality_threshold)
        for item in all_predictions
        if item.policy == "always_small" and item.error is None
    }
    for policy_name in config.benchmark.policies:
        policy_predictions = [
            item
            for item in all_predictions
            if item.policy == policy_name
            and item.error is None
            and not item.abstained
            and item.example_id in small_success
        ]
        predicted_labels = [
            int(
                not item.escalated
                and item.route.value in {"small", "small_with_retrieval", "cascade"}
            )
            for item in policy_predictions
        ]
        actual_labels = [small_success[item.example_id] for item in policy_predictions]
        policy_metrics[policy_name]["route_decision_diagnostics"] = routing_quality_metrics(
            predicted_labels, actual_labels
        )
        policy_metrics[policy_name]["routing"] = router_probability_metrics(
            [row for row in all_predictions if row.policy == policy_name],
            all_predictions,
            config.benchmark.quality_threshold,
        )

    metrics = aggregate_metrics(
        all_predictions,
        wall_time_ms=total_measurement_wall_ms,
        bootstrap_samples=config.benchmark.bootstrap_samples,
        minimum_samples_for_claims=config.benchmark.minimum_samples_for_claims,
        quality_threshold=config.benchmark.quality_threshold,
        seed=config.seed,
    )
    metrics["policies"] = policy_metrics
    metrics["fake"] = config.benchmark.fake
    errors = [item.model_dump(mode="json") for item in all_predictions if item.error]
    writer.write_jsonl(
        "predictions.jsonl", [item.model_dump(mode="json") for item in all_predictions]
    )
    writer.write_jsonl("routes.jsonl", all_routes)
    writer.write_jsonl("timings.jsonl", all_timings)
    writer.write_jsonl("errors.jsonl", errors)
    writer.write_json("metrics.json", metrics)
    metadata = run_metadata(
        run_id,
        config,
        dataset_hash(config.benchmark.dataset_path),
        root,
        backends,
    )
    metadata["status"] = "completed_with_errors" if errors else "completed"
    metadata["error_count"] = len(errors)
    metadata["replay"] = config.benchmark.cache.mode == "read_only"
    metadata["generation_cache"] = config.benchmark.cache.model_dump(mode="json")
    metadata["dataset_manifest"] = dataset_manifest
    writer.write_json("run.json", metadata)
    generate_report(run_dir)
    return run_dir


def collect_baselines(config: AppConfig, repository_root: Path | None = None) -> Path:
    """Generate always-small and always-large outcomes and populate the shared cache."""

    policies = ["always_small", "always_large"]
    if config.retrieval.enabled:
        policies.extend(["retrieval_first", "cascade"])
    benchmark = config.benchmark.model_copy(
        update={
            "policies": policies,
            "cache": config.benchmark.cache.model_copy(update={"mode": "read_write"}),
        }
    )
    return run_benchmark(config.model_copy(update={"benchmark": benchmark}), repository_root)


def replay_benchmark(config: AppConfig, repository_root: Path | None = None) -> Path:
    """Evaluate configured policies without initializing model weights or generating tokens."""

    benchmark = config.benchmark.model_copy(
        update={
            "warmup_runs": 0,
            "cache": config.benchmark.cache.model_copy(update={"mode": "read_only"}),
        }
    )
    return run_benchmark(config.model_copy(update={"benchmark": benchmark}), repository_root)
