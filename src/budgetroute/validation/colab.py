"""Real free-Colab component gates and inspectable artifact acceptance."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from budgetroute.config import AppConfig, RetrievalConfig, load_config
from budgetroute.evaluation.dataset import dataset_hash
from budgetroute.evaluation.harness import evaluate_records
from budgetroute.exceptions import ArtifactError
from budgetroute.experiments.artifacts import ArtifactWriter
from budgetroute.experiments.metadata import configuration_hash, git_metadata
from budgetroute.features.semantic import SemanticFeatures, SemanticFeatureSpec
from budgetroute.inference.service import build_service
from budgetroute.retrieval.embeddings import TransformersEmbedder
from budgetroute.retrieval.index import FaissHNSWIndex
from budgetroute.retrieval.service import RetrievalService
from budgetroute.routing.registry import build_policy
from budgetroute.schemas import BenchmarkRecord, EvaluationType, GenerationRequest, PredictionRecord

REQUIRED_CHECKS = {"generation", "semantic", "retrieval", "scheduling", "api", "cleanup"}


def source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    source = git_metadata(root)
    expected = os.environ.get("BUDGETROUTE_REVIEWED_SHA", "")
    if (
        not re.fullmatch(r"[a-f0-9]{40}", expected)
        or source["commit"] != expected
        or source["dirty"] is not False
    ):
        raise ArtifactError("source must be clean and match explicit BUDGETROUTE_REVIEWED_SHA")
    return source


def validate_colab_environment() -> dict[str, Any]:
    try:
        is_colab = (
            Path("/content").is_dir() and importlib.util.find_spec("google.colab") is not None
        )
    except ModuleNotFoundError:
        is_colab = False
    if not is_colab:
        raise ArtifactError(
            "important model checks require a Google Colab runtime, not local fallback"
        )
    import torch

    if not torch.cuda.is_available():
        raise ArtifactError("free Colab GPU unavailable; save checkpoints and stop, do not upgrade")
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ArtifactError("GPU hardware identity unavailable") from exc
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return {
        "status": "passed",
        "fake": False,
        "gpu_available": True,
        "hardware": {
            "gpu": gpu,
            "cuda": torch.version.cuda,
            "torch": str(torch.__version__),
            "python": sys.version,
        },
        "packages": dict(sorted(packages.items())),
    }


def validate_component_report(report: dict[str, Any]) -> None:
    if (
        report.get("status") != "passed"
        or report.get("fake") is not False
        or set(report.get("checks", {})) != REQUIRED_CHECKS
    ):
        raise ArtifactError("component report incomplete, failed or fake")

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            if (
                value.get("fake") is True
                or value.get("type") == "fake"
                or value.get("embedder") == "fake"
                or value.get("status")
                in {
                    "failed",
                    "completed_with_errors",
                    "incomplete",
                }
            ):
                raise ArtifactError("fake or failed nested component evidence")
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(report)
    for check in report["checks"].values():
        if check.get("status") != "passed" or check.get("fake") is not False:
            raise ArtifactError("each component must explicitly pass as real")
    try:
        checks = report["checks"]
        generation = checks["generation"]
        for name in ("small_backend", "large_backend"):
            backend = generation["backends"][name]
            if (
                backend["type"] != "transformers"
                or backend["device_resolved"] != "cuda"
                or backend["revision_requested"] != backend["revision_resolved"]
            ):
                raise ValueError("generator runtime identity mismatch")
        for name in ("always_small", "always_large"):
            evidence = generation["evidence"][name]
            if len(evidence["single"]) != 2 or len(evidence["batch"]) != 2:
                raise ValueError("missing single/batch generation evidence")
            for row in evidence["single"] + evidence["batch"]:
                if row["fake"] is not False or row["replayed"] or row["confidence_signals"] is None:
                    raise ValueError("invalid generation evidence")
        semantic = checks["semantic"]
        if semantic["spec"] != SemanticFeatureSpec().model_dump() or semantic["shape"] != [2, 384]:
            raise ValueError("MiniLM identity/shape mismatch")
        retrieval = checks["retrieval"]
        if (
            retrieval["metadata"]["index_type"] != "faiss_hnsw"
            or retrieval["metadata"]["embedding_revision"] != SemanticFeatureSpec().revision
            or not retrieval["hits"]
        ):
            raise ValueError("real pinned FAISS retrieval missing")
        grid = checks["scheduling"]["grid"]
        if {(r["concurrency"], r["batch_size"]) for r in grid} != {(1, 1), (2, 1), (1, 2)}:
            raise ValueError("scheduling grid incomplete")
        for condition in grid:
            if len(condition["predictions"]) != 2 or any(
                row["fake"] is not False or row["error"] for row in condition["predictions"]
            ):
                raise ValueError("scheduling prediction evidence invalid")
        api = checks["api"]
        if (
            api["transport"] != "authenticated_ipv4_loopback"
            or api["public_listener"] is not False
            or (
                api["generation_status"],
                api["unauthorized_status"],
                api["feedback_status"],
                api["metrics_status"],
            )
            != (200, 401, 202, 200)
        ):
            raise ValueError("authenticated loopback API evidence invalid")
        cleanup = checks["cleanup"]
        if (
            cleanup["encoder_released"] is not True
            or cleanup["generator_health"]["ready"] is not False
        ):
            raise ValueError("cleanup evidence invalid")
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"actual component evidence missing or incompatible: {exc}") from exc


def _assert_real_config(config: AppConfig) -> None:
    if config.mode != "gpu" or config.benchmark.fake:
        raise ArtifactError("component checks require real GPU configuration")
    for backend in (config.small_backend, config.large_backend):
        if (
            backend.type != "transformers"
            or backend.device != "cuda"
            or backend.trust_remote_code
            or not re.fullmatch(r"[a-f0-9]{40}", backend.revision or "")
        ):
            raise ArtifactError(
                "component checks require pinned built-in Transformers CUDA loaders"
            )


def _loopback_api(config: AppConfig, service: Any) -> dict[str, Any]:
    import httpx
    import uvicorn

    from budgetroute.api.app import create_app

    key_name = "BUDGETROUTE_COLAB_EPHEMERAL_KEY"
    previous = os.environ.get(key_name)
    key = secrets.token_urlsafe(32)
    os.environ[key_name] = key
    effective = config.model_copy(
        update={
            "api": config.api.model_copy(
                update={
                    "require_api_key": True,
                    "api_key_env": key_name,
                    "tenant_keys_env": None,
                    "feedback_enabled": True,
                    "rate_limit_requests": 100,
                }
            ),
            "batching": config.batching.model_copy(update={"enabled": False}),
        }
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(effective, service=service), log_level="error", lifespan="on")
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    try:
        thread.start()
        deadline = time.monotonic() + 120
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not server.started:
            raise ArtifactError("loopback API failed to start")
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180) as client:
            headers = {"Authorization": f"Bearer {key}"}
            if client.get("/v1/config").status_code != 401:
                raise ArtifactError("API authentication not enforced")
            generated = client.post(
                "/v1/generate",
                headers=headers,
                json={"prompt": "Reply with only A.", "request_id": "colab-api-check"},
            )
            if generated.status_code != 200 or generated.json().get("fake") is not False:
                raise ArtifactError("API generation failed real gate")
            feedback = client.post(
                "/v1/feedback",
                headers=headers,
                json={"request_id": "colab-api-check", "correct": True},
            )
            metrics = client.get("/metrics", headers=headers)
            if (
                feedback.status_code != 202
                or metrics.status_code != 200
                or "budgetroute_requests_total" not in metrics.text
            ):
                raise ArtifactError("feedback/metrics contract failed")
        return {
            "status": "passed",
            "fake": False,
            "transport": "authenticated_ipv4_loopback",
            "public_listener": False,
            "generation_status": 200,
            "unauthorized_status": 401,
            "feedback_status": 202,
            "metrics_status": 200,
        }
    finally:
        server.should_exit = True
        thread.join(timeout=180)
        listener.close()
        if previous is None:
            os.environ.pop(key_name, None)
        else:
            os.environ[key_name] = previous
        if thread.is_alive():
            raise ArtifactError("loopback API did not shut down cleanly")


def run_component_checks(config_path: Path, output_dir: Path) -> Path:
    from budgetroute.experiments.pilot import _hash

    environment = validate_colab_environment()
    source = source_identity()
    config = load_config(config_path, apply_environment=False)
    _assert_real_config(config)
    resolved_config = config.model_dump(mode="json")
    config = config.model_copy(
        update={
            "benchmark": config.benchmark.model_copy(
                update={"cache": config.benchmark.cache.model_copy(update={"mode": "off"})}
            ),
            "retrieval": config.retrieval.model_copy(update={"enabled": False}),
        }
    )
    report: dict[str, Any] = {
        "status": "incomplete",
        "fake": False,
        "source": source,
        "environment_hash": _hash(environment),
        "configuration_hash": configuration_hash(load_config(config_path, apply_environment=False)),
        "checks": {},
    }
    writer = ArtifactWriter(output_dir)
    writer.write_json("environment.json", environment)
    writer.write_yaml("resolved-config.yaml", resolved_config)
    writer.write_json("components.json", report)
    service = build_service(config)
    encoder = SemanticFeatures(SemanticFeatureSpec())
    retrieval_encoder = TransformersEmbedder(
        SemanticFeatureSpec().model_id, "cpu", SemanticFeatureSpec().revision, max_input_tokens=256
    )
    try:
        service.initialize()
        generation_evidence = {}
        for name, backend_config in (
            ("always_small", config.small_backend),
            ("always_large", config.large_backend),
        ):
            service.engine.policy = build_policy(config.routing.model_copy(update={"policy": name}))
            requests = [
                GenerationRequest(request_id=f"component-{i}", prompt=text)
                for i, text in enumerate(("Reply with only A.", "Reply with only B."))
            ]
            single = [service.generate(request) for request in requests]
            batch = service.generate_batch(requests)
            if any(
                row.fake
                or row.replayed
                or row.execution.answer_truncated
                or not row.answer
                or row.confidence_signals is None
                for row in single + batch
            ):
                raise ArtifactError("generation/confidence/batch real component gate failed")
            if [r.request_id for r in batch] != [r.request_id for r in requests] or [
                r.answer for r in single
            ] != [r.answer for r in batch]:
                raise ArtifactError("single/batch order or greedy answer agreement failed")
            generation_evidence[name] = {
                "single": [r.model_dump(mode="json") for r in single],
                "batch": [r.model_dump(mode="json") for r in batch],
                "requested_revision": backend_config.revision,
            }
        report["checks"]["generation"] = {
            "status": "passed",
            "fake": False,
            "backends": service.metadata(),
            "evidence": generation_evidence,
        }
        vectors = encoder.transform(["Prompt-only semantic component check", "A different prompt"])
        if vectors.shape != (2, 384) or not np_is_finite(vectors):
            raise ArtifactError("MiniLM real embedding shape/values invalid")
        report["checks"]["semantic"] = {
            "status": "passed",
            "fake": False,
            "spec": encoder.spec.model_dump(),
            "shape": list(vectors.shape),
            "truncation_count": encoder.truncation_count,
        }
        corpus = output_dir / "component-corpus"
        ArtifactWriter(corpus)._atomic_text("a.txt", "Paris is the capital of France.\n")
        ArtifactWriter(corpus)._atomic_text("b.txt", "Bananas are a yellow fruit.\n")
        retrieval_config = RetrievalConfig(
            enabled=True,
            corpus_path=corpus,
            index_path=output_dir / "component-index.npz",
            embedder="transformers",
            index_type="faiss_hnsw",
            embedding_model_id=encoder.spec.model_id,
            embedding_revision=encoder.spec.revision,
            min_similarity=-1.0,
        )
        retrieval = RetrievalService(retrieval_config, FaissHNSWIndex(retrieval_encoder))
        retrieval.initialize(rebuild=True)
        hits = retrieval.search("capital of France", top_k=1)
        if not hits or "Paris" not in hits[0].text:
            raise ArtifactError("real FAISS HNSW retrieval failed")
        report["checks"]["retrieval"] = {
            "status": "passed",
            "fake": False,
            "metadata": retrieval.metadata(),
            "hits": [h.model_dump(mode="json") for h in hits],
        }
        scheduling = []
        records = [
            BenchmarkRecord(
                id=f"schedule-{i}",
                category="component",
                prompt=prompt,
                reference_answer="A",
                evaluation_type=EvaluationType.CLASSIFICATION,
            )
            for i, prompt in enumerate(("Reply with only A.", "Reply with only A."))
        ]
        for concurrency, batch_size in ((1, 1), (2, 1), (1, 2)):
            measured, _, _ = evaluate_records(
                service, records, fake=False, concurrency=concurrency, batch_size=batch_size
            )
            if len(measured) != 2 or any(
                r.error or r.fake is not False or r.replayed for r in measured
            ):
                raise ArtifactError("real scheduling component failed")
            scheduling.append(
                {
                    "concurrency": concurrency,
                    "batch_size": batch_size,
                    "predictions": [r.model_dump(mode="json") for r in measured],
                }
            )
        report["checks"]["scheduling"] = {"status": "passed", "fake": False, "grid": scheduling}
        report["checks"]["api"] = _loopback_api(config, service)
        encoder.close()
        retrieval_encoder.cleanup()
        service.close()
        if encoder.embedder._model is not None or retrieval_encoder._model is not None:
            raise ArtifactError("encoder cleanup failed")
        report["checks"]["cleanup"] = {
            "status": "passed",
            "fake": False,
            "generator_health": service.health(),
            "encoder_released": True,
        }
        report["status"] = "passed"
        validate_component_report(report)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        encoder.close()
        retrieval_encoder.cleanup()
        service.close()
        writer.write_json("components.json", report)
    return output_dir / "components.json"


def np_is_finite(values: Any) -> bool:
    import numpy as np

    return bool(np.isfinite(values).all())


def validate_run_artifacts(run_dir: Path, *, require_real: bool = True) -> dict[str, Any]:
    from budgetroute.experiments.pilot import _json, _rows, evidence_paths

    files = evidence_paths(run_dir)
    manifest = _json(run_dir / "evidence-manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != {
        path.relative_to(run_dir).as_posix() for path in files
    }:
        raise ArtifactError("evidence manifest does not match required curated artifacts")
    for path in files:
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(run_dir.resolve())
            or manifest[path.relative_to(run_dir).as_posix()] != dataset_hash(path)
        ):
            raise ArtifactError(f"missing or hash-mismatched evidence: {path.name}")
    run = _json(run_dir / "run.json")
    if run.get("status") != "completed" or run.get("role") != "pilot":
        raise ArtifactError("run is not a complete development pilot")
    if require_real:
        if run.get("fake") is not False or run.get("replay") is not False:
            raise ArtifactError("fake/replay run cannot pass real gate")
        validate_component_report(_json(run_dir / "components.json"))
        identity = _json(run_dir / "identity.json")
        source = identity.get("source", {})
        if source.get("dirty") is not False or not re.fullmatch(
            r"[a-f0-9]{40}", source.get("commit") or ""
        ):
            raise ArtifactError("dirty/missing source identity")
        environment = _json(run_dir / "environment.json")
        if environment.get("gpu_available") is not True or not environment.get("packages"):
            raise ArtifactError("GPU or resolved environment evidence missing")
    collection = [
        PredictionRecord.model_validate(row)
        for row in _rows(run_dir / "collection/predictions.jsonl")
    ]
    diagnostic = [
        PredictionRecord.model_validate(row)
        for row in _rows(run_dir / "diagnostic/predictions.jsonl")
    ]
    if (
        len(collection) != 400
        or len({(r.example_id, r.policy) for r in collection}) != 400
        or any(r.error for r in collection)
    ):
        raise ArtifactError("baseline collection is incomplete or duplicated")
    from budgetroute.experiments.pilot_data import PilotSplits

    splits = PilotSplits.model_validate(_json(run_dir / "data/splits.json"))
    if dataset_hash(run_dir / "data/pilot.jsonl") != splits.dataset_sha256:
        raise ArtifactError("data does not match split manifest")
    names = {
        "always_small",
        "always_large",
        "numeric_gate",
        "semantic_gate",
        "matched_random",
        "cascade",
    }
    if (
        {r.policy for r in diagnostic} != names
        or len(diagnostic) != 240
        or len({(r.example_id, r.policy) for r in diagnostic}) != 240
        or {r.example_id for r in diagnostic} != set(splits.test_ids)
    ):
        raise ArtifactError(
            "diagnostic must cover only all 40 frozen diagnostic IDs for six policies"
        )
    replay = _json(run_dir / "diagnostic/replay.json")
    if (
        replay.get("status") != "passed"
        or replay.get("comparison_count") != 80
        or replay.get("mismatch_count") != 0
    ):
        raise ArtifactError("live/recorded baseline replay agreement failed")
    all_rows = collection + diagnostic
    for block in range(3):
        material = _json(run_dir / f"timing/block-{block}.json")
        timed = [PredictionRecord.model_validate(row) for row in material.get("predictions", [])]
        if (
            material.get("status") != "completed"
            or material.get("cache_mode") != "off"
            or material.get("initialization_count") != 1
            or len(timed) != 240
            or len({(r.example_id, r.policy) for r in timed}) != 240
            or {r.policy for r in timed} != names
            or {r.example_id for r in timed} != set(splits.test_ids)
        ):
            raise ArtifactError("timing block incomplete or wrong population")
        if any(r.error or r.replayed or r.latency_ms <= 0 for r in timed):
            raise ArtifactError("failed or replayed timing cannot pass live gate")
        all_rows.extend(timed)
    if require_real and any(row.fake is not False or row.error for row in all_rows):
        raise ArtifactError("fake, unknown or failed prediction in real evidence")
    return {
        "status": "passed",
        "role": "pilot",
        "fake": not require_real,
        "baseline_records": 400,
        "diagnostic_records": 240,
        "timed_records": 720,
        "source": run.get("source"),
    }
