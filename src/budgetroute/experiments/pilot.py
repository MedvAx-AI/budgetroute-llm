"""Versioned, resumable development pilot. No confirmatory stage is exposed."""

from __future__ import annotations

import hashlib
import json
import random
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from budgetroute.config import AppConfig, load_config
from budgetroute.evaluation.dataset import dataset_hash, load_dataset
from budgetroute.evaluation.harness import evaluate_records, request_for_record
from budgetroute.evaluation.metrics import aggregate_metrics
from budgetroute.evaluation.paired import paired_policy_summary
from budgetroute.evaluation.routing_metrics import router_probability_metrics
from budgetroute.exceptions import ArtifactError
from budgetroute.experiments.artifacts import ArtifactWriter
from budgetroute.experiments.cache import GenerationCache, backend_fingerprint
from budgetroute.experiments.metadata import configuration_hash
from budgetroute.experiments.pilot_data import PilotSplits
from budgetroute.features.request_features import RequestFeatureExtractor
from budgetroute.inference.service import build_service
from budgetroute.routing.base import RoutingPolicy
from budgetroute.routing.calibration import (
    ThresholdSelection,
    choose_threshold,
    fit_probability_calibrator,
)
from budgetroute.routing.learned import LearnedPolicy
from budgetroute.routing.registry import build_policy
from budgetroute.routing.training import train_router
from budgetroute.schemas import (
    BackendName,
    BenchmarkRecord,
    GenerationRequest,
    PredictionRecord,
    RequestFeatures,
    RouteDecision,
    RouteName,
)

Stage = Literal["collect", "fit", "diagnostic", "timing", "summarize", "export"]


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"missing or invalid pilot artifact {path}: {exc}") from exc


def _rows(path: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if path.exists()
        else []
    )


def _freeze_identity(workspace: Path, identity: dict[str, Any]) -> None:
    path = workspace / "identity.json"
    if path.exists() and _json(path) != identity:
        raise ArtifactError(
            "pilot resume identity changed; start a separately identified workspace"
        )
    ArtifactWriter(workspace).write_json(path.name, identity)


class _MatchedRandomPolicy:
    name = "matched_random"

    def __init__(self, ids: list[str], small_ids: list[str]) -> None:
        self.ids = set(ids)
        self.small_ids = set(small_ids)

    @classmethod
    def freeze(
        cls, ids: list[str], cheap_count: int, seed: int, destination: Path
    ) -> _MatchedRandomPolicy:
        if len(set(ids)) != len(ids) or not 0 <= cheap_count <= len(ids):
            raise ValueError("invalid matched random assignment population/count")
        population = sorted(ids)
        small_ids = sorted(random.Random(seed).sample(population, cheap_count))
        material = {
            "ids": population,
            "small_ids": small_ids,
            "seed": seed,
            "cheap_count": cheap_count,
        }
        if destination.exists() and _json(destination) != material:
            raise ValueError("frozen matched random assignment changed")
        ArtifactWriter(destination.parent).write_json(destination.name, material)
        return cls(population, small_ids)

    def decide(self, request: GenerationRequest, features: RequestFeatures) -> RouteDecision:
        if request.request_id not in self.ids:
            raise ValueError("question missing from frozen random assignments")
        return RouteDecision(
            route=RouteName.SMALL if request.request_id in self.small_ids else RouteName.LARGE,
            confidence=0.5,
            difficulty_score=0.5,
            reason="Frozen exact-count random comparator",
            policy=self.name,
            features=features.model_dump(mode="json"),
        )


def _cache_config(config: AppConfig, workspace: Path, mode: str) -> AppConfig:
    return config.model_copy(
        update={
            "benchmark": config.benchmark.model_copy(
                update={
                    "cache": config.benchmark.cache.model_copy(
                        update={"mode": mode, "directory": workspace / "cache"}
                    )
                }
            )
        }
    )


def _collect(config: AppConfig, workspace: Path, records: list[BenchmarkRecord]) -> Path:
    directory = workspace / "collection"
    writer = ArtifactWriter(directory)
    config = _cache_config(config, workspace, "read_write")
    predictions = {
        (row["example_id"], row["policy"]): row for row in _rows(directory / "predictions.jsonl")
    }
    routes = {(row["example_id"], row["policy"]): row for row in _rows(directory / "routes.jsonl")}
    cache = GenerationCache(workspace / "cache")
    writer.write_json(
        "run.json", {"status": "incomplete", "role": "pilot", "fake": config.benchmark.fake}
    )
    service = build_service(config)
    try:
        service.initialize()
        for policy_name, backend, backend_config in (
            ("always_small", BackendName.SMALL, config.small_backend),
            ("always_large", BackendName.LARGE, config.large_backend),
        ):
            service.engine.policy = build_policy(
                config.routing.model_copy(update={"policy": policy_name})
            )
            for record in records:
                key = (record.id, policy_name)
                cached = cache.get(
                    backend_fingerprint(backend, backend_config),
                    request_for_record(record, fake=config.benchmark.fake),
                )
                if key in predictions and not predictions[key].get("error") and cached is not None:
                    continue
                measured, decisions, _ = evaluate_records(
                    service, [record], fake=config.benchmark.fake
                )
                predictions[key] = measured[0].model_dump(mode="json")
                if decisions:
                    routes[key] = decisions[0]
                writer.write_jsonl("predictions.jsonl", list(predictions.values()))
                writer.write_jsonl("routes.jsonl", list(routes.values()))
        failures = sum(row.get("error") is not None for row in predictions.values())
        writer.write_json(
            "run.json",
            {
                "status": "completed_with_errors" if failures else "completed",
                "role": "pilot",
                "fake": config.benchmark.fake,
                "record_count": len(predictions),
                "failure_count": failures,
                "latency_population": "collection_cache_not_live_timing",
            },
        )
    finally:
        service.close()
    return directory


def _fit(config: AppConfig, workspace: Path, records: list[BenchmarkRecord]) -> Path:
    directory = workspace / "fit"
    if _json(workspace / "collection/run.json").get("status") != "completed":
        raise ArtifactError("fit requires complete error-free paired collection")
    for mode in ("numeric", "semantic"):
        train_router(
            workspace / "collection",
            directory / f"{mode}.joblib",
            quality_threshold=1.0,
            seed=20261021,
            target_selective_accuracy=0.98,
            label_strategy="paired_quality",
            split_manifest=workspace / "data/splits.json",
            dataset_path=workspace / "data/pilot.jsonl",
            feature_mode=mode,
        )
    # Secondary confidence cascade: fit correctness calibration on fitting IDs only,
    # select a threshold on selection IDs only; never choose on diagnostic outcomes.
    splits = PilotSplits.model_validate(_json(workspace / "data/splits.json"))
    small = {
        row["example_id"]: row
        for row in _rows(workspace / "collection/predictions.jsonl")
        if row["policy"] == "always_small"
    }
    training = [small[key] for key in splits.train_ids]
    selection = [small[key] for key in splits.calibration_ids]
    available = all(row.get("backend_confidence_raw") is not None for row in training + selection)
    if available:
        calibrator = fit_probability_calibrator(
            [float(r["backend_confidence_raw"]) for r in training],
            [int(r["quality_score"] == 1) for r in training],
        )
        threshold = choose_threshold(
            calibrator.transform([float(r["backend_confidence_raw"]) for r in selection]),
            [int(r["quality_score"] == 1) for r in selection],
            0.98,
        )
    else:
        from budgetroute.routing.calibration import ProbabilityCalibrator

        calibrator, threshold = ProbabilityCalibrator(), None
    selected = (
        ThresholdSelection("feasible", threshold.threshold)
        if threshold
        else ThresholdSelection(
            "infeasible", None, "No feasible selection threshold or missing confidence"
        )
    )
    ArtifactWriter(directory).write_json(
        "cascade.json",
        {
            "schema_version": 4,
            "kind": "backend_confidence_calibrator",
            "calibrator": asdict(calibrator),
            "serving_threshold": selected.threshold,
            "threshold_selection": asdict(selected),
            "certification_status": "not_requested",
            "role": "pilot",
            "train_ids": splits.train_ids,
            "calibration_ids": splits.calibration_ids,
        },
    )
    return directory


def _policies(
    config: AppConfig, workspace: Path, records: list[BenchmarkRecord], *, timing: bool
) -> dict[str, RoutingPolicy]:
    policies: dict[str, RoutingPolicy] = {
        name: build_policy(config.routing.model_copy(update={"policy": name}))
        for name in ("always_small", "always_large", "cascade")
    }
    for mode in ("numeric", "semantic"):
        gate = LearnedPolicy(workspace / f"fit/{mode}.joblib")
        gate.name = f"{mode}_gate"
        policies[gate.name] = gate
    extractor = RequestFeatureExtractor()
    primary = policies["semantic_gate"]
    cheap = sum(
        primary.decide(
            request_for_record(record, fake=False),
            extractor.extract(request_for_record(record, fake=False)),
        ).route
        == RouteName.SMALL
        for record in records
    )
    policies["matched_random"] = _MatchedRandomPolicy.freeze(
        [row.id for row in records],
        cheap,
        20261022 if timing else 20261021,
        workspace / ("timing" if timing else "diagnostic") / "assignments.json",
    )
    # Do not retain primed per-question features; only the warm encoder weights remain.
    return policies


def _close_policies(policies: dict[str, RoutingPolicy]) -> None:
    for policy in policies.values():
        close = getattr(policy, "close", None)
        if close is not None:
            close()


def _with_cascade(config: AppConfig, workspace: Path) -> AppConfig:
    return config.model_copy(
        update={
            "routing": config.routing.model_copy(
                update={"cascade_calibrator_path": workspace / "fit/cascade.json"}
            )
        }
    )


def _diagnostic(config: AppConfig, workspace: Path, records: list[BenchmarkRecord]) -> Path:
    directory = workspace / "diagnostic"
    writer = ArtifactWriter(directory)
    policies = _policies(config, workspace, records, timing=False)
    service = build_service(_cache_config(_with_cascade(config, workspace), workspace, "read_only"))
    predictions: list[dict[str, Any]] = []
    routes: list[dict[str, object]] = []
    try:
        service.initialize()
        for policy in policies.values():
            service.engine.policy = policy
            measured, decisions, _ = evaluate_records(service, records, fake=config.benchmark.fake)
            predictions.extend(row.model_dump(mode="json") for row in measured)
            routes.extend(decisions)
        writer.write_jsonl("predictions.jsonl", predictions)
        writer.write_jsonl("routes.jsonl", routes)
        baselines = {
            (r["example_id"], r["policy"]): r
            for r in _rows(workspace / "collection/predictions.jsonl")
        }
        comparisons = [
            row for row in predictions if row["policy"] in {"always_small", "always_large"}
        ]
        fields = (
            "raw_answer",
            "quality_score",
            "answer_truncated",
            "input_tokens",
            "output_tokens",
            "backend_confidence_raw",
        )
        mismatch = [
            row["example_id"]
            for row in comparisons
            if any(
                row.get(key) != baselines[(row["example_id"], row["policy"])].get(key)
                for key in fields
            )
        ]
        writer.write_json(
            "replay.json",
            {
                "status": "passed" if not mismatch else "failed",
                "comparison_count": len(comparisons),
                "mismatch_count": len(mismatch),
                "fake": False,
                "scope": "diagnostic baseline signatures; no live latency claim",
            },
        )
    finally:
        service.close()
        _close_policies(policies)
    return directory


def _timing(config: AppConfig, workspace: Path, records: list[BenchmarkRecord]) -> Path:
    directory = workspace / "timing"
    writer = ArtifactWriter(directory)
    for block in range(3):
        path = directory / f"block-{block}.json"
        if path.exists() and _json(path).get("status") == "completed":
            continue
        policies = _policies(config, workspace, records, timing=True)
        order = list(policies)
        random.Random(20261021 + block).shuffle(order)
        material: dict[str, Any] = {
            "started_utc": datetime.now(UTC).isoformat(),
            "status": "incomplete",
            "block": block,
            "seed": 20261021 + block,
            "order": order,
            "cache_mode": "off",
            "initialization_count": 0,
            "predictions": [],
            "routes": [],
        }
        writer.write_json(path.name, material)
        effective = _cache_config(config, workspace, "off")
        if (workspace / "fit/cascade.json").exists():
            effective = _with_cascade(effective, workspace)
        service = build_service(effective)
        try:
            service.initialize()
            material["initialization_count"] = 1
            material["initialization_ms"] = service.initialization_ms
            for name in order:
                service.engine.policy = policies[name]
                service.generate(request_for_record(records[0], fake=config.benchmark.fake))
                measured, decisions, _ = evaluate_records(
                    service, records, fake=config.benchmark.fake, batch_size=1, concurrency=1
                )
                material["predictions"].extend(row.model_dump(mode="json") for row in measured)
                material["routes"].extend(decisions)
                writer.write_json(path.name, material)
            failures = any(row.get("error") for row in material["predictions"])
            material["status"] = "completed_with_errors" if failures else "completed"
            material["resident_backends"] = service.metadata()
            writer.write_json(path.name, material)
        finally:
            service.close()
            _close_policies(policies)
    return directory


def _summary(workspace: Path) -> Path:
    identity = _json(workspace / "identity.json")
    baseline = [
        PredictionRecord.model_validate(row)
        for row in _rows(workspace / "collection/predictions.jsonl")
    ]
    diagnostic = [
        PredictionRecord.model_validate(row)
        for row in _rows(workspace / "diagnostic/predictions.jsonl")
    ]
    live: list[PredictionRecord] = []
    for block in range(3):
        material = _json(workspace / f"timing/block-{block}.json")
        if material["status"] != "completed":
            raise ArtifactError("incomplete live timing block")
        for raw in material["predictions"]:
            row = PredictionRecord.model_validate(raw)
            live.append(row.model_copy(update={"example_id": f"{row.example_id}:block{block}"}))
    context = {
        **identity,
        "trial_block": "three_seeded_blocks",
        "status": "completed",
        "fake": False,
        "replay": False,
    }
    names = sorted({row.policy for row in diagnostic})
    large = [row for row in diagnostic if row.policy == "always_large"]
    live_large = [row for row in live if row.policy == "always_large"]
    pairs = {
        name: paired_policy_summary([row for row in diagnostic if row.policy == name], large)
        for name in names
        if name != "always_large"
    }
    timing = {
        name: paired_policy_summary(
            [row for row in live if row.policy == name],
            live_large,
            candidate_run=context,
            baseline_run=context,
        )
        for name in names
        if name != "always_large"
    }
    primary = pairs["semantic_gate"]
    discordance = (primary["wins"] + primary["losses"]) / 40
    planning_n = max(200, int(np.ceil((1.96 + 0.8416) ** 2 * max(discordance, 1 / 40) / (0.02**2))))
    elapsed = sum(row.latency_ms for row in live) / 1000
    summary = {
        "fake": any(row.fake is not False for row in baseline + diagnostic + live),
        "role": "development_pilot_not_confirmation",
        "summarized_utc": datetime.now(UTC).isoformat(),
        "baseline_questions": 200,
        "diagnostic_questions": 40,
        "independent_timed_questions": 40,
        "timing_blocks": 3,
        "comparison_population": "both generators resident; idle small remains loaded for always-large",
        "baseline": {
            name: aggregate_metrics([row for row in baseline if row.policy == name])
            for name in ("always_small", "always_large")
        },
        "diagnostic": {
            name: aggregate_metrics([row for row in diagnostic if row.policy == name])
            for name in names
        },
        "live": {
            name: aggregate_metrics([row for row in live if row.policy == name]) for name in names
        },
        "paired": pairs,
        "live_paired": timing,
        "router_probability_evaluation": {
            name: router_probability_metrics(
                [row for row in diagnostic if row.policy == name], baseline, 1.0
            )
            for name in names
        },
        "gates": {
            mode: {
                key: _json(workspace / f"fit/{mode}.joblib.metadata.json")[key]
                for key in (
                    "threshold_selection",
                    "certification_status",
                    "feature_mode",
                    "semantic_spec",
                    "embedding_truncation_count",
                    "label_strategy",
                )
            }
            for mode in ("numeric", "semantic")
        },
        "replay": _json(workspace / "diagnostic/replay.json"),
        "parsing_or_truncation_count": sum(
            row.answer_truncated or not row.normalized_answer for row in diagnostic
        ),
        "rough_confirmation_planning": {
            "method": "normal approximation; not a power guarantee or authorization",
            "paired_discordance": discordance,
            "independent_quality_questions_estimate": planning_n,
            "optimistic_runtime_hours_same_policy_mix": elapsed
            / len(live)
            * planning_n
            * len(names)
            * 3
            / 3600,
        },
        "recommendation": "stop_or_inconclusive_until_separately_approved_powered_protocol",
        "limitations": [
            "40 diagnostic questions cannot certify a two-point margin",
            "No confirmatory evidence",
            "Latency conditional on observed free-Colab hardware, co-residency and non-thinking greedy generation",
            "Configured costs are estimates, not bills",
        ],
    }
    writer = ArtifactWriter(workspace)
    path = writer.write_json("summary.json", summary)
    writer.write_json(
        "run.json",
        {
            "status": "completed",
            "fake": summary["fake"],
            "replay": False,
            "role": "pilot",
            "source": identity["source"],
        },
    )
    return path


def evidence_paths(workspace: Path) -> list[Path]:
    fixed = [
        "identity.json",
        "environment.json",
        "components.json",
        "run.json",
        "summary.json",
        "resolved-config.yaml",
        "data/pilot.jsonl",
        "data/splits.json",
        "data/pilot.manifest.json",
        "collection/predictions.jsonl",
        "collection/routes.jsonl",
        "collection/run.json",
        "diagnostic/predictions.jsonl",
        "diagnostic/routes.jsonl",
        "diagnostic/assignments.json",
        "diagnostic/replay.json",
        "timing/assignments.json",
        *[f"timing/block-{i}.json" for i in range(3)],
        "fit/numeric.joblib",
        "fit/semantic.joblib",
        "fit/numeric.joblib.metadata.json",
        "fit/semantic.joblib.metadata.json",
        "fit/cascade.json",
    ]
    return [workspace / relative for relative in fixed]


def _export(workspace: Path) -> Path:
    from budgetroute.validation.colab import validate_run_artifacts

    paths = evidence_paths(workspace)
    if any(
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(workspace.resolve())
        for path in paths
    ):
        raise ArtifactError("required evidence missing or outside workspace")
    ArtifactWriter(workspace).write_json(
        "evidence-manifest.json",
        {p.relative_to(workspace).as_posix(): dataset_hash(p) for p in paths},
    )
    validate_run_artifacts(workspace)
    destination = workspace / "routing-pilot-evidence.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*paths, workspace / "evidence-manifest.json"]:
            archive.write(path, path.relative_to(workspace).as_posix())
    return destination


def run_pilot_stage(config_path: Path, workspace: Path, *, stage: Stage) -> Path:
    from budgetroute.validation.colab import (
        source_identity,
        validate_colab_environment,
        validate_component_report,
    )

    if stage not in {"collect", "fit", "diagnostic", "timing", "summarize", "export"}:
        raise ArtifactError("only development pilot stages are supported")
    config = load_config(config_path, apply_environment=False)
    if (
        config.benchmark.fake
        or config.mode != "gpu"
        or any(b.type != "transformers" for b in (config.small_backend, config.large_backend))
    ):
        raise ArtifactError("Colab pilot requires two real Transformers GPU backends")
    environment = validate_colab_environment()
    splits = PilotSplits.model_validate(_json(workspace / "data/splits.json"))
    if dataset_hash(workspace / "data/pilot.jsonl") != splits.dataset_sha256:
        raise ArtifactError("pilot data hash mismatch")
    identity = {
        "source": source_identity(),
        "configuration_hash": configuration_hash(config),
        "dataset_hash": splits.dataset_sha256,
        "split_hash": dataset_hash(workspace / "data/splits.json"),
        "exclusion_hash": splits.exclusion_sha256,
        "generation_fingerprint": _hash(
            [
                backend_fingerprint(name, backend)
                for name, backend in (
                    (BackendName.SMALL, config.small_backend),
                    (BackendName.LARGE, config.large_backend),
                )
            ]
        ),
        "environment_hash": _hash(environment),
        "hardware_fingerprint": _hash(environment["hardware"]),
    }
    _freeze_identity(workspace, identity)
    report = _json(workspace / "components.json")
    validate_component_report(report)
    if (
        report.get("configuration_hash") != identity["configuration_hash"]
        or report.get("source") != identity["source"]
        or report.get("environment_hash") != identity["environment_hash"]
    ):
        raise ArtifactError("component checks do not match pilot identity")
    writer = ArtifactWriter(workspace)
    writer.write_json("environment.json", environment)
    writer.write_yaml("resolved-config.yaml", config.model_dump(mode="json"))
    config = config.model_copy(
        update={
            "benchmark": config.benchmark.model_copy(
                update={"dataset_path": workspace / "data/pilot.jsonl"}
            )
        }
    )
    records = load_dataset(workspace / "data/pilot.jsonl")
    if len(records) != 200 or set(row.id for row in records) != set(
        splits.train_ids + splits.calibration_ids + splits.test_ids
    ):
        raise ArtifactError("pilot population mismatch")
    diagnostic = [row for row in records if row.id in set(splits.test_ids)]
    if stage == "collect":
        return _collect(config, workspace, records)
    if stage == "fit":
        return _fit(config, workspace, records)
    if stage == "diagnostic":
        return _diagnostic(config, workspace, diagnostic)
    if stage == "timing":
        return _timing(config, workspace, diagnostic)
    if stage == "summarize":
        return _summary(workspace)
    return _export(workspace)
