"""Typer command-line interface for local development and experiments."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
from pathlib import Path
from typing import Any, Literal

import typer

from budgetroute import __version__
from budgetroute.adaptation.online import CalibrationRegistry
from budgetroute.config import AppConfig, load_config, validate_runtime_config
from budgetroute.environment import collect_environment, writable_directory
from budgetroute.evaluation.dataset import (
    dataset_summary,
    load_dataset,
    materialize_dataset,
    materialize_router_test_split,
    validate_dataset_manifest,
)
from budgetroute.experiments.replay import verify_replay
from budgetroute.experiments.runner import collect_baselines as collect_baseline_run
from budgetroute.experiments.runner import replay_benchmark as run_replay_benchmark
from budgetroute.experiments.runner import run_benchmark
from budgetroute.inference.service import build_service
from budgetroute.operations.auth import TenantAuthenticator
from budgetroute.operations.store import build_store
from budgetroute.reporting.report import generate_report, latest_run_directory
from budgetroute.retrieval.base import EmbeddingProvider
from budgetroute.retrieval.embeddings import FakeEmbedder, TransformersEmbedder
from budgetroute.retrieval.index import ExactCosineIndex, FaissCosineIndex, FaissHNSWIndex
from budgetroute.retrieval.service import RetrievalService
from budgetroute.routing.retrieval_training import train_retrieval_benefit_router
from budgetroute.routing.training import evaluate_router as evaluate_router_artifact
from budgetroute.routing.training import (
    train_backend_confidence_calibrator as train_confidence_artifact,
)
from budgetroute.routing.training import train_router as train_router_artifact
from budgetroute.schemas import GenerationRequest
from budgetroute.validation.compatibility import run_compatibility_matrix
from budgetroute.validation.load import run_load_test

app = typer.Typer(
    name="budgetroute",
    help="Quality-aware language-model routing, evaluation, and reporting.",
    no_args_is_help=True,
)


def _print_json(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False, default=str))


@app.command()
def doctor(
    config: Path = typer.Option(Path("configs/serving/fake.yaml"), exists=False),
) -> None:
    """Inspect Python, optional dependencies, CUDA, configuration, and outputs."""
    environment = collect_environment()
    configuration: dict[str, Any]
    try:
        loaded = load_config(config)
        try:
            validate_runtime_config(loaded)
            runtime: dict[str, Any] = {"valid": True}
        except Exception as exc:
            runtime = {"valid": False, "error": str(exc)}
        configuration = {
            "valid": True,
            "path": str(config),
            "mode": loaded.mode,
            "output_writable": writable_directory(loaded.output_dir),
            "runtime": runtime,
        }
    except Exception as exc:
        configuration = {"valid": False, "path": str(config), "error": str(exc)}
    _print_json({"budgetroute_version": __version__, **environment, "configuration": configuration})


@app.command("validate-config")
def validate_config_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Resolve and validate a YAML configuration."""
    loaded = load_config(config)
    validate_runtime_config(loaded)
    _print_json({"valid": True, "path": str(config), "summary": loaded.sanitized_summary()})


@app.command("security-check")
def security_check(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Audit deployment-sensitive settings without printing secret values."""
    loaded = load_config(config)
    validate_runtime_config(loaded)
    authenticator = TenantAuthenticator(loaded.api)
    legacy_secret = (
        os.environ.get(loaded.api.api_key_env) if loaded.api.tenant_keys_env is None else None
    )
    try:
        host_is_loopback = (
            loaded.api.host.lower() == "localhost"
            or ipaddress.ip_address(loaded.api.host).is_loopback
        )
    except ValueError:
        host_is_loopback = False
    checks: list[dict[str, object]] = [
        {
            "name": "non_loopback_authentication",
            "passed": host_is_loopback or loaded.api.require_api_key,
            "detail": "non-loopback bindings must require an API key",
        },
        {
            "name": "api_key_available",
            "passed": not loaded.api.require_api_key
            or bool(
                os.environ.get(
                    loaded.api.tenant_keys_env
                    if loaded.api.tenant_keys_env is not None
                    else loaded.api.api_key_env
                )
            ),
            "detail": (
                f"active credential source: {loaded.api.tenant_keys_env or loaded.api.api_key_env}"
            ),
        },
        {
            "name": "credential_validation",
            "passed": not loaded.api.require_api_key
            or (legacy_secret is None or len(legacy_secret) >= 16),
            "detail": {
                **authenticator.sanitized_summary(),
                "minimum_legacy_key_characters": 16,
            },
        },
        {
            "name": "bounded_admission",
            "passed": loaded.batching.max_queue_size > 0 and loaded.batching.request_timeout_ms > 0,
            "detail": "queue capacity and request deadline are configured",
        },
        {
            "name": "trusted_hosts",
            "passed": bool(loaded.api.trusted_hosts) and "*" not in loaded.api.trusted_hosts,
            "detail": "wildcard Host headers are not allowed",
        },
        {
            "name": "tls_boundary",
            "passed": host_is_loopback
            or loaded.api.tls_certfile is not None
            or loaded.api.external_tls_termination,
            "detail": "non-loopback traffic must use built-in or explicitly external TLS",
        },
        {
            "name": "shared_coordination",
            "passed": host_is_loopback or loaded.operations.backend in {"sqlite", "postgres"},
            "detail": "non-loopback profiles use transactional shared state",
        },
        {
            "name": "durable_review_feedback",
            "passed": not (loaded.api.review_enabled or loaded.api.feedback_enabled)
            or loaded.operations.backend in {"sqlite", "postgres"},
            "detail": "enabled review/feedback workflows require durable state",
        },
        {
            "name": "database_transport_security",
            "passed": loaded.operations.backend != "postgres"
            or loaded.operations.postgres_require_tls,
            "detail": "PostgreSQL profiles require an encrypted database connection",
        },
        {
            "name": "bounded_content_policy",
            "passed": host_is_loopback or loaded.content_policy.enabled,
            "detail": (
                "non-loopback profiles enable metadata bounds and deployment-specific "
                "literal input/output rules; these do not prove semantic safety"
            ),
        },
        {
            "name": "repository_security_policy",
            "passed": Path("SECURITY.md").is_file()
            and Path(".github/workflows/security.yml").is_file()
            and Path(".github/dependabot.yml").is_file(),
            "detail": "security policy, scanning workflow, and dependency updates exist",
        },
    ]
    passed = all(bool(check["passed"]) for check in checks)
    _print_json({"passed": passed, "config": str(config), "checks": checks})
    if not passed:
        raise typer.Exit(code=1)


@app.command("inspect-data")
def inspect_data(
    path: Path = typer.Option(..., "--path", exists=True, dir_okay=False),
    manifest: Path | None = typer.Option(None, "--manifest", exists=True, dir_okay=False),
) -> None:
    """Validate a benchmark JSONL file and show category statistics."""
    records = load_dataset(path)
    manifest_value = (
        validate_dataset_manifest(path, manifest).model_dump(mode="json")
        if manifest is not None
        else None
    )
    _print_json(
        {
            "valid": True,
            "path": str(path),
            "manifest": manifest_value,
            **dataset_summary(records),
        }
    )


@app.command("prepare-routing-pilot")
def prepare_routing_pilot_command(
    spec: Path = typer.Option(..., "--spec", exists=True, dir_okay=False),
    exclusions: Path = typer.Option(..., "--exclusions", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output"),
    seed: int = typer.Option(20261021, "--seed"),
) -> None:
    """Freeze 200 independent development questions and 120/40/40 pilot roles."""
    from budgetroute.experiments.pilot_data import prepare_pilot_data

    typer.echo(prepare_pilot_data(spec, exclusions, output, seed=seed).model_dump_json(indent=2))


@app.command("materialize-dataset")
def materialize_dataset_command(
    spec: Path = typer.Option(..., "--spec", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    limit: int | None = typer.Option(None, min=1),
    offset: int = typer.Option(0, min=0),
    sampling: Literal["head", "stratified"] = typer.Option("head"),
    seed: int = typer.Option(42),
    corpus_dir: Path | None = typer.Option(None, "--corpus-dir", file_okay=False),
) -> None:
    """Materialize a pinned public dataset and write its integrity manifest."""
    manifest = materialize_dataset(
        spec,
        output,
        limit=limit,
        offset=offset,
        sampling=sampling,
        seed=seed,
        corpus_dir=corpus_dir,
    )
    _print_json(
        {
            "dataset": str(output),
            "manifest": str(output.with_suffix(".manifest.json")),
            "details": manifest.model_dump(mode="json"),
        }
    )


@app.command("materialize-router-test-split")
def materialize_router_test_split_command(
    router_metadata: Path = typer.Option(..., "--router-metadata", exists=True, dir_okay=False),
    dataset: Path = typer.Option(..., "--dataset", exists=True, dir_okay=False),
    manifest: Path = typer.Option(..., "--manifest", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
) -> None:
    """Materialize the persisted untouched router test split with a derived manifest."""
    details = materialize_router_test_split(router_metadata, dataset, manifest, output)
    _print_json(
        {
            "dataset": str(output),
            "manifest": str(output.with_suffix(".manifest.json")),
            "details": details.model_dump(mode="json"),
        }
    )


@app.command("build-index")
def build_index(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Build and persist the configured exact cosine retrieval index."""
    loaded = load_config(config)
    if not loaded.retrieval.enabled:
        raise typer.BadParameter("retrieval.enabled must be true")
    if loaded.retrieval.embedder == "fake":
        embedder: EmbeddingProvider = FakeEmbedder(loaded.retrieval.embedding_dimension)
    else:
        assert loaded.retrieval.embedding_model_id is not None
        embedder = TransformersEmbedder(
            loaded.retrieval.embedding_model_id,
            "cuda" if loaded.mode == "gpu" else "cpu",
            loaded.retrieval.embedding_revision,
        )
    index: ExactCosineIndex
    if loaded.retrieval.index_type == "faiss_hnsw":
        index = FaissHNSWIndex(
            embedder,
            neighbors=loaded.retrieval.hnsw_neighbors,
            ef_construction=loaded.retrieval.hnsw_ef_construction,
            ef_search=loaded.retrieval.hnsw_ef_search,
        )
    elif loaded.retrieval.index_type == "faiss":
        index = FaissCosineIndex(embedder)
    else:
        index = ExactCosineIndex(embedder)
    service = RetrievalService(loaded.retrieval, index)
    service.initialize(rebuild=True, persist=True)
    _print_json(service.metadata())


@app.command()
def benchmark(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Run configured policies and save predictions, metrics, and a report."""
    run_directory = run_benchmark(load_config(config), Path.cwd())
    _print_json({"run_directory": str(run_directory), "report": str(run_directory / "report")})


@app.command("collect-baselines")
def collect_baselines_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Generate small/large baselines once and populate the content-addressed cache."""
    loaded = load_config(config)
    validate_runtime_config(loaded)
    run_directory = collect_baseline_run(loaded, Path.cwd())
    _print_json(
        {"run_directory": str(run_directory), "cache": str(loaded.benchmark.cache.directory)}
    )


@app.command("replay-benchmark")
def replay_benchmark_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Evaluate policies from cached generations without loading model weights."""
    run_directory = run_replay_benchmark(load_config(config), Path.cwd())
    _print_json({"run_directory": str(run_directory), "replay": True})


@app.command("verify-replay")
def verify_replay_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    sample_size: int = typer.Option(3, min=1, max=1000),
) -> None:
    """Regenerate a sample and verify exact semantic agreement with cache replay."""
    loaded = load_config(config)
    validate_runtime_config(loaded)
    result = verify_replay(loaded, sample_size)
    _print_json(result)
    if result["mismatch_count"]:
        raise typer.Exit(code=1)


@app.command("train-router")
def train_router(
    artifacts: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("outputs/router/router.joblib")),
    quality_threshold: float = typer.Option(0.8, min=0.0, max=1.0),
    seed: int = typer.Option(42),
    target_selective_accuracy: float = typer.Option(0.8, min=0.0, max=1.0),
    label_strategy: Literal["small_success", "paired_quality"] = typer.Option(
        "small_success", "--label-strategy"
    ),
    split_manifest: Path | None = typer.Option(
        None, "--split-manifest", exists=True, dir_okay=False
    ),
    feature_mode: Literal["numeric", "semantic"] = typer.Option("numeric", "--feature-mode"),
    dataset: Path | None = typer.Option(None, "--dataset", exists=True, dir_okay=False),
    embedding_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
    embedding_revision: str = typer.Option("1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    embedding_device: Literal["cpu", "cuda"] = typer.Option("cpu"),
    embedding_max_input_tokens: int = typer.Option(256, min=1, max=4096),
) -> None:
    """Train and persist a learned router from benchmark-derived labels."""
    from budgetroute.features.semantic import SemanticFeatureSpec

    result = train_router_artifact(
        artifacts,
        output,
        quality_threshold,
        seed,
        target_selective_accuracy,
        label_strategy,
        split_manifest=split_manifest,
        feature_mode=feature_mode,
        dataset_path=dataset,
        semantic_spec=SemanticFeatureSpec(
            model_id=embedding_model,
            revision=embedding_revision,
            device=embedding_device,
            max_input_tokens=embedding_max_input_tokens,
        ),
    )
    _print_json({"artifact": str(output), "training": result})


@app.command("evaluate-router")
def evaluate_router(
    router: Path = typer.Option(..., exists=True, dir_okay=False),
    artifacts: Path = typer.Option(..., exists=True, file_okay=False),
    quality_threshold: float = typer.Option(0.8, min=0.0, max=1.0),
    dataset: Path | None = typer.Option(None, "--dataset", exists=True, dir_okay=False),
) -> None:
    """Evaluate a learned router's labels, calibration, and route distribution."""
    _print_json(
        evaluate_router_artifact(router, artifacts, quality_threshold, dataset_path=dataset)
    )


@app.command("calibrate-confidence")
def calibrate_confidence(
    artifacts: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("outputs/router/small-confidence.json")),
    policy: str = typer.Option("always_small"),
    quality_threshold: float = typer.Option(0.8, min=0.0, max=1.0),
    target_selective_accuracy: float = typer.Option(0.8, min=0.0, max=1.0),
    seed: int = typer.Option(42),
) -> None:
    """Fit a portable correctness calibrator for small-backend confidence."""
    result = train_confidence_artifact(
        artifacts,
        output,
        policy=policy,
        quality_threshold=quality_threshold,
        target_selective_accuracy=target_selective_accuracy,
        seed=seed,
    )
    _print_json({"artifact": str(output), "calibration": result})


@app.command("train-retrieval-router")
def train_retrieval_router(
    artifacts: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("outputs/router/retrieval-benefit.joblib")),
    minimum_quality_improvement: float = typer.Option(0.01, min=0.0, max=1.0),
    seed: int = typer.Option(42),
    target_precision: float = typer.Option(0.8, min=0.0, max=1.0),
) -> None:
    """Train paired evidence for deciding whether retrieval improves small-model quality."""
    result = train_retrieval_benefit_router(
        artifacts,
        output,
        minimum_quality_improvement=minimum_quality_improvement,
        seed=seed,
        target_precision=target_precision,
    )
    _print_json({"artifact": str(output), "training": result})


@app.command("adapt-confidence")
def adapt_confidence(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    limit: int | None = typer.Option(None, min=20),
) -> None:
    """Build and gate an online calibration candidate from durable delayed labels."""
    loaded = load_config(config)
    if not loaded.adaptation.enabled:
        raise typer.BadParameter("adaptation.enabled must be true")
    if loaded.operations.backend not in {"sqlite", "postgres"}:
        raise typer.BadParameter("online adaptation requires a durable operations backend")
    store = build_store(loaded.operations)
    store.initialize()
    try:
        observations = store.labeled_observations(limit)
    finally:
        store.close()
    result = CalibrationRegistry(loaded.adaptation).adapt(observations)
    _print_json(result)
    if not result["promoted"]:
        raise typer.Exit(code=2)


@app.command("rollback-calibration")
def rollback_calibration(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    version: str | None = typer.Option(None),
) -> None:
    """Atomically restore a previously promoted online calibration version."""
    loaded = load_config(config)
    _print_json(CalibrationRegistry(loaded.adaptation).rollback(version))


@app.command("compatibility-matrix")
def compatibility_matrix(
    configs: list[Path] = typer.Option(..., "--config", exists=True, dir_okay=False),
    output_root: Path = typer.Option(Path("outputs/compatibility"), "--output-root"),
) -> None:
    """Probe exact runtime configurations and save non-comparative compatibility evidence."""
    run_directory = run_compatibility_matrix(configs, output_root)
    _print_json({"run_directory": str(run_directory)})


@app.command("load-test")
def load_test(
    targets: list[str] = typer.Option(..., "--target"),
    requests: int = typer.Option(100, min=1, max=1_000_000),
    concurrency: int = typer.Option(10, min=1, max=100_000),
    output_root: Path = typer.Option(Path("outputs/load"), "--output-root"),
    api_key_env: str | None = typer.Option(None),
    target_latency_ms: float = typer.Option(1000.0, min=0.001),
    request_timeout_seconds: float = typer.Option(120.0, min=0.001, max=3600.0),
    warmup_requests: int = typer.Option(0, min=0, max=100_000),
    allow_insecure_http: bool = typer.Option(False),
) -> None:
    """Send real traffic to one or more service endpoints and save load evidence."""
    if concurrency > requests:
        raise typer.BadParameter("concurrency cannot exceed requests")
    run_directory = asyncio.run(
        run_load_test(
            targets,
            request_count=requests,
            concurrency=concurrency,
            output_root=output_root,
            api_key_env=api_key_env,
            target_latency_ms=target_latency_ms,
            request_timeout_seconds=request_timeout_seconds,
            warmup_requests=warmup_requests,
            allow_insecure_http=allow_insecure_http,
        )
    )
    _print_json({"run_directory": str(run_directory)})


@app.command("audit-check")
def audit_check(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Verify the retained operational audit hash chain."""
    loaded = load_config(config)
    store = build_store(loaded.operations)
    store.initialize()
    try:
        result = store.verify_audit_chain()
    finally:
        store.close()
    _print_json(result)
    if not result["valid"]:
        raise typer.Exit(code=1)


@app.command("migrate-store")
def migrate_store(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Apply and checksum-verify PostgreSQL operational-store migrations."""
    loaded = load_config(config)
    if loaded.operations.backend != "postgres":
        raise typer.BadParameter("migrate-store requires operations.backend=postgres")
    migration_config = loaded.operations.model_copy(update={"postgres_auto_migrate": True})
    store = build_store(migration_config)
    store.initialize()
    try:
        result = store.health()
    finally:
        store.close()
    _print_json({"migrated": True, "operations": result})


@app.command("generate-report")
def generate_report_command(
    run_directory: Path | None = typer.Option(None, "--run-directory", file_okay=False),
    latest: bool = typer.Option(False, "--latest"),
    output_root: Path = typer.Option(Path("outputs"), "--output-root"),
) -> None:
    """Generate Markdown, CSV, and PNG files from existing experiment artifacts."""
    selected = latest_run_directory(output_root) if latest else run_directory
    if selected is None:
        raise typer.BadParameter("no run directory was supplied or found")
    report = generate_report(selected)
    _print_json({"report_directory": str(report)})


@app.command()
def serve(
    config: Path = typer.Option(Path("configs/serving/fake.yaml"), exists=True),
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None, min=1, max=65535),
) -> None:
    """Start the FastAPI service for the selected fake, CPU, or GPU configuration."""
    try:
        import uvicorn

        from budgetroute.api.app import create_app
    except ImportError as exc:
        raise typer.BadParameter("serve requires `pip install -e .[api]`") from exc
    loaded = load_config(config)
    if host is not None or port is not None:
        data = loaded.model_dump(mode="python")
        data["api"] = {
            **loaded.api.model_dump(mode="python"),
            "host": host or loaded.api.host,
            "port": port or loaded.api.port,
        }
        loaded = AppConfig.model_validate(data)
    application = create_app(loaded)
    uvicorn.run(
        application,
        host=loaded.api.host,
        port=loaded.api.port,
        log_level=loaded.api.log_level,
        ssl_certfile=(str(loaded.api.tls_certfile) if loaded.api.tls_certfile else None),
        ssl_keyfile=(str(loaded.api.tls_keyfile) if loaded.api.tls_keyfile else None),
    )


def _demo_case(label: str, config: AppConfig, request: GenerationRequest) -> None:
    service = build_service(config)
    try:
        service.initialize()
        response = service.generate(request)
        typer.echo(f"\n--- {label} ---")
        _print_json(response.model_dump(mode="json", exclude_none=True))
    finally:
        service.close()


@app.command()
def demo(
    config: Path = typer.Option(Path("configs/serving/fake.yaml"), exists=True),
) -> None:
    """Run five deterministic fake scenarios; output is not a real benchmark."""
    loaded = load_config(config)
    if loaded.mode != "fake":
        raise typer.BadParameter("demo requires a fake-mode configuration")
    typer.echo("FAKE MODE DEMO — outputs and timing are simulated, not model benchmark results.")
    heuristic = loaded.model_copy(
        update={"routing": loaded.routing.model_copy(update={"policy": "heuristic"})}
    )
    _demo_case(
        "easy -> small", heuristic, GenerationRequest(prompt="What is the capital of France?")
    )
    _demo_case(
        "difficult -> large",
        heuristic,
        GenerationRequest(
            prompt="Calculate the equation 17 * 6 + 5 step by step.",
            metadata={"fake_reference_answer": "107", "fake_small_success": False},
        ),
    )
    _demo_case(
        "retrieval -> small with context",
        heuristic,
        GenerationRequest(
            prompt="According to the project corpus, what is the internal project codename?",
            requires_retrieval=True,
            metadata={"fake_reference_answer": "Lantern", "fake_small_success": False},
        ),
    )
    cascade = loaded.model_copy(
        update={"routing": loaded.routing.model_copy(update={"policy": "cascade"})}
    )
    _demo_case(
        "cascade -> escalate",
        cascade,
        GenerationRequest(
            prompt="A difficult fake request that should escalate.",
            metadata={"fake_reference_answer": "Escalated answer", "fake_small_success": False},
        ),
    )
    _demo_case(
        "insufficient context -> abstain",
        heuristic,
        GenerationRequest(
            prompt="This is ambiguous and unanswerable without missing context.",
            metadata={"must_abstain": True},
        ),
    )
