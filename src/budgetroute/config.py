"""Typed YAML configuration loading, composition, and operational overrides."""

from __future__ import annotations

import importlib.util
import ipaddress
import os
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from budgetroute.exceptions import ConfigurationError


class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=10.0)
    seed: int = 42
    enable_thinking: bool | None = None


class BackendConfig(BaseModel):
    type: Literal["fake", "transformers", "openai_compatible"] = "fake"
    model_id: str | None = None
    tokenizer_id: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    model_license: str | None = None
    model_card: str | None = None
    local_files_only: bool = False
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    quantization: Literal["none", "int8", "int4"] = "none"
    compile: bool = False
    trust_remote_code: bool = False
    base_url: str | None = None
    runtime_identity: str | None = Field(default=None, min_length=1, max_length=512)
    model_loader: Literal["causal_lm", "image_text_to_text"] = "causal_lm"
    load_direct_to_device: bool = False
    max_input_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    api_key_env: str | None = None
    request_timeout_seconds: float = Field(default=120.0, gt=0.0, le=3600.0)
    request_logprobs: bool = False
    allow_remote_endpoint: bool = False
    max_concurrency: int = Field(default=1, ge=1, le=1024)
    input_cost_units_per_1k_tokens: float = Field(default=0.0, ge=0.0)
    output_cost_units_per_1k_tokens: float = Field(default=0.0, ge=0.0)
    measured_input_rate: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    measured_output_rate: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    artificial_latency_ms: float = Field(default=0.0, ge=0.0)
    fail_on_substrings: list[str] = Field(default_factory=list)
    quality: float = Field(default=0.8, ge=0.0, le=1.0)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @model_validator(mode="after")
    def validate_backend(self) -> BackendConfig:
        if self.runtime_identity is not None and not self.runtime_identity.strip():
            raise ValueError("runtime_identity must contain non-whitespace text")
        if self.type in {"transformers", "openai_compatible"} and not self.model_id:
            raise ValueError(f"{self.type} backend requires model_id")
        if self.type == "openai_compatible":
            if not self.base_url:
                raise ValueError("openai_compatible backend requires base_url")
            parsed = urlparse(self.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("openai_compatible base_url must be an HTTP(S) URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("base_url must not contain credentials, a query, or a fragment")
            if self.api_key_env is not None and not re.fullmatch(
                r"[A-Z][A-Z0-9_]*", self.api_key_env
            ):
                raise ValueError("api_key_env must be an uppercase environment variable name")
        if self.type != "transformers" and self.quantization != "none":
            raise ValueError("quantization is only valid for transformers backends")
        if self.type == "transformers" and self.device == "cpu" and self.precision == "fp16":
            raise ValueError("fp16 on CPU is unsupported; use fp32 or a supported bf16 setup")
        if self.type == "transformers" and self.device == "cpu" and self.quantization != "none":
            raise ValueError("quantization configuration currently requires a CUDA device")
        return self


class RetrievalConfig(BaseModel):
    enabled: bool = False
    corpus_path: Path = Path("data/sample_corpus")
    index_path: Path = Path("data/indexes/sample-index.npz")
    embedder: Literal["fake", "transformers"] = "fake"
    index_type: Literal["exact", "faiss", "faiss_hnsw"] = "exact"
    embedding_model_id: str | None = None
    embedding_revision: str | None = None
    embedding_dimension: int = Field(default=64, ge=8, le=4096)
    chunk_size: int = Field(default=500, ge=32)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=3, ge=1, le=50)
    min_similarity: float = Field(default=0.12, ge=-1.0, le=1.0)
    hnsw_neighbors: int = Field(default=32, ge=4, le=256)
    hnsw_ef_construction: int = Field(default=80, ge=8, le=2000)
    hnsw_ef_search: int = Field(default=64, ge=1, le=2000)

    @model_validator(mode="after")
    def validate_retrieval(self) -> RetrievalConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("retrieval chunk_overlap must be smaller than chunk_size")
        if self.embedder == "transformers" and not self.embedding_model_id:
            raise ValueError("transformers embedder requires embedding_model_id")
        return self


class RoutingConfig(BaseModel):
    policy: Literal[
        "always_small",
        "always_large",
        "random",
        "heuristic",
        "learned",
        "retrieval_first",
        "cascade",
        "load_aware",
        "budget_aware",
        "learned_retrieval",
    ] = "heuristic"
    seed: int = 42
    small_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    large_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    difficulty_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    abstain_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    cascade_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    retrieval_similarity_threshold: float = Field(default=0.12, ge=-1.0, le=1.0)
    learned_model_path: Path | None = None
    learned_success_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_benefit_model_path: Path | None = None
    retrieval_benefit_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    cascade_calibrator_path: Path | None = None
    retain_initial_answer: bool = True
    target_latency_ms: float = Field(default=1000.0, gt=0.0)
    max_estimated_cost_units: float | None = Field(default=None, ge=0.0)
    quality_weight: float = Field(default=0.6, ge=0.0)
    latency_weight: float = Field(default=0.25, ge=0.0)
    cost_weight: float = Field(default=0.15, ge=0.0)
    overload_threshold: float = Field(default=1.0, gt=0.0)
    human_review_enabled: bool = False
    human_review_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_probabilities(self) -> RoutingConfig:
        if self.policy == "random" and self.small_probability + self.large_probability <= 0:
            raise ValueError("random routing probabilities must sum to more than zero")
        if self.policy == "learned" and self.learned_model_path is None:
            raise ValueError("learned routing requires learned_model_path")
        if self.policy == "learned_retrieval" and self.retrieval_benefit_model_path is None:
            raise ValueError("learned_retrieval routing requires retrieval_benefit_model_path")
        if self.policy == "budget_aware" and (
            self.quality_weight + self.latency_weight + self.cost_weight <= 0
        ):
            raise ValueError("budget-aware routing weights must sum to more than zero")
        return self


class BatchingConfig(BaseModel):
    enabled: bool = False
    max_batch_size: int = Field(default=8, ge=1, le=1024)
    max_wait_ms: float = Field(default=10.0, ge=0.0, le=10_000.0)
    max_queue_size: int = Field(default=256, ge=1, le=100_000)
    submit_timeout_ms: float = Field(default=100.0, gt=0.0, le=60_000.0)
    request_timeout_ms: float = Field(default=120_000.0, gt=0.0, le=3_600_000.0)


class GenerationCacheConfig(BaseModel):
    mode: Literal["off", "read_write", "read_only", "refresh"] = "off"
    directory: Path = Path("outputs/cache/generations")


class BenchmarkConfig(BaseModel):
    dataset_path: Path = Path("data/sample_benchmark.jsonl")
    dataset_manifest_path: Path | None = None
    policies: list[str] = Field(default_factory=lambda: ["always_small", "heuristic", "cascade"])
    warmup_runs: int = Field(default=1, ge=0, le=100)
    measured_runs: int = Field(default=1, ge=1, le=10_000)
    concurrency: int = Field(default=1, ge=1, le=1024)
    batch_size: int = Field(default=1, ge=1, le=1024)
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    cache: GenerationCacheConfig = Field(default_factory=GenerationCacheConfig)
    replay_verify_samples: int = Field(default=0, ge=0, le=1000)
    require_pinned_revisions: bool = False
    fake: bool = False
    bootstrap_samples: int = Field(default=1000, ge=0, le=100_000)
    minimum_samples_for_claims: int = Field(default=100, ge=1, le=1_000_000)


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"
    max_prompt_chars: int = Field(default=50_000, ge=1)
    max_request_bytes: int = Field(default=1_048_576, ge=1024, le=100_000_000)
    require_api_key: bool = False
    api_key_env: str = "BUDGETROUTE_API_KEY"
    tenant_keys_env: str | None = None
    rate_limit_requests: int = Field(default=60, ge=1, le=1_000_000)
    rate_limit_window_seconds: float = Field(default=60.0, gt=0.0, le=86_400.0)
    trusted_hosts: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "testserver"]
    )
    metrics_enabled: bool = True
    feedback_enabled: bool = False
    review_enabled: bool = False
    tls_certfile: Path | None = None
    tls_keyfile: Path | None = None
    external_tls_termination: bool = False

    @model_validator(mode="after")
    def validate_api(self) -> ApiConfig:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.api_key_env):
            raise ValueError("api.api_key_env must be an uppercase environment variable name")
        if self.tenant_keys_env is not None and not re.fullmatch(
            r"[A-Z][A-Z0-9_]*", self.tenant_keys_env
        ):
            raise ValueError("api.tenant_keys_env must be an uppercase environment variable name")
        if not self.trusted_hosts:
            raise ValueError("api.trusted_hosts cannot be empty")
        if (self.tls_certfile is None) != (self.tls_keyfile is None):
            raise ValueError("api.tls_certfile and api.tls_keyfile must be configured together")
        return self


class OperationsConfig(BaseModel):
    backend: Literal["memory", "sqlite", "postgres"] = "memory"
    database_path: Path = Path("data/state/budgetroute.db")
    postgres_dsn_env: str = "BUDGETROUTE_POSTGRES_DSN"
    postgres_pool_min_size: int = Field(default=1, ge=1, le=1000)
    postgres_pool_max_size: int = Field(default=16, ge=1, le=1000)
    postgres_connect_timeout_seconds: float = Field(default=10.0, gt=0.0, le=300.0)
    postgres_require_tls: bool = True
    postgres_auto_migrate: bool = True
    replica_id_env: str = "BUDGETROUTE_REPLICA_ID"
    max_global_inflight: int = Field(default=64, ge=1, le=1_000_000)
    lease_ttl_seconds: float = Field(default=180.0, gt=1.0, le=86_400.0)
    quota_requests: int = Field(default=1000, ge=1, le=10_000_000)
    quota_window_seconds: float = Field(default=60.0, gt=0.0, le=86_400.0)
    audit_retention_events: int = Field(default=100_000, ge=100, le=100_000_000)

    @model_validator(mode="after")
    def validate_operations(self) -> OperationsConfig:
        for field_name, value in (
            ("replica_id_env", self.replica_id_env),
            ("postgres_dsn_env", self.postgres_dsn_env),
        ):
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", value):
                raise ValueError(
                    f"operations.{field_name} must be an uppercase environment variable name"
                )
        if self.postgres_pool_min_size > self.postgres_pool_max_size:
            raise ValueError(
                "operations.postgres_pool_min_size cannot exceed postgres_pool_max_size"
            )
        return self


class AdaptationConfig(BaseModel):
    enabled: bool = False
    registry_path: Path = Path("data/state/calibration-registry")
    minimum_labeled_samples: int = Field(default=100, ge=20, le=1_000_000)
    holdout_fraction: float = Field(default=0.25, gt=0.0, lt=0.5)
    minimum_brier_improvement: float = Field(default=0.002, ge=0.0, le=1.0)
    maximum_ece_regression: float = Field(default=0.01, ge=0.0, le=1.0)
    target_selective_accuracy: float = Field(default=0.8, ge=0.0, le=1.0)
    change_point_threshold: float = Field(default=5.0, gt=0.0)
    change_point_delta: float = Field(default=0.005, ge=0.0)


class ContentPolicyConfig(BaseModel):
    enabled: bool = False
    prompt_blocklist: list[str] = Field(default_factory=list)
    output_blocklist: list[str] = Field(default_factory=list)
    max_metadata_json_chars: int = Field(default=20_000, ge=100, le=1_000_000)
    max_metadata_depth: int = Field(default=6, ge=1, le=32)
    max_metadata_keys: int = Field(default=100, ge=1, le=100_000)

    @model_validator(mode="after")
    def validate_content_policy(self) -> ContentPolicyConfig:
        patterns = [*self.prompt_blocklist, *self.output_blocklist]
        if any(not item.strip() or len(item) > 500 for item in patterns):
            raise ValueError("content-policy blocklist items must contain 1-500 characters")
        return self


class MonitoringConfig(BaseModel):
    enabled: bool = True
    window_size: int = Field(default=500, ge=20, le=1_000_000)
    minimum_samples: int = Field(default=50, ge=10)
    drift_threshold: float = Field(default=0.2, gt=0.0)
    baseline_path: Path | None = None

    @model_validator(mode="after")
    def validate_monitoring(self) -> MonitoringConfig:
        if self.minimum_samples > self.window_size:
            raise ValueError("monitoring.minimum_samples cannot exceed window_size")
        return self


class AppConfig(BaseModel):
    name: str = "budgetroute"
    mode: Literal["fake", "cpu", "gpu"] = "fake"
    seed: int = 42
    output_dir: Path = Path("outputs")
    small_backend: BackendConfig = Field(default_factory=BackendConfig)
    large_backend: BackendConfig = Field(default_factory=lambda: BackendConfig(quality=0.95))
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    operations: OperationsConfig = Field(default_factory=OperationsConfig)
    adaptation: AdaptationConfig = Field(default_factory=AdaptationConfig)
    content_policy: ContentPolicyConfig = Field(default_factory=ContentPolicyConfig)

    @model_validator(mode="after")
    def validate_composition(self) -> AppConfig:
        if self.benchmark.cache.mode != "off":
            for backend in (self.small_backend, self.large_backend):
                if backend.type == "openai_compatible" and not backend.runtime_identity:
                    raise ValueError("cache-enabled HTTP backend requires runtime_identity")
        retrieval_policies = {"retrieval_first", "learned_retrieval"}
        if self.routing.policy in retrieval_policies and not self.retrieval.enabled:
            raise ValueError(f"{self.routing.policy} routing requires retrieval.enabled=true")
        if self.mode == "fake" and (
            self.small_backend.type != "fake" or self.large_backend.type != "fake"
        ):
            raise ValueError("fake mode requires both backends to use type=fake")
        if self.benchmark.require_pinned_revisions:
            unpinned = [
                name
                for name, backend in (
                    ("small_backend", self.small_backend),
                    ("large_backend", self.large_backend),
                )
                if backend.type == "transformers" and backend.revision is None
            ]
            if unpinned:
                raise ValueError(
                    "real benchmark requires pinned model revisions for: " + ", ".join(unpinned)
                )
            if (
                self.retrieval.enabled
                and self.retrieval.embedder == "transformers"
                and self.retrieval.embedding_revision is None
            ):
                raise ValueError("real benchmark requires a pinned retrieval embedding revision")
        if not _is_loopback_host(self.api.host) and not self.api.require_api_key:
            raise ValueError("non-loopback API binding requires api.require_api_key=true")
        if (
            not _is_loopback_host(self.api.host)
            and self.api.tls_certfile is None
            and not self.api.external_tls_termination
        ):
            raise ValueError(
                "non-loopback API binding requires built-in TLS or "
                "api.external_tls_termination=true"
            )
        return self

    def sanitized_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "small_backend": {
                "type": self.small_backend.type,
                "model_id": self.small_backend.model_id,
                "revision": self.small_backend.revision,
                "device": self.small_backend.device,
                "precision": self.small_backend.precision,
            },
            "large_backend": {
                "type": self.large_backend.type,
                "model_id": self.large_backend.model_id,
                "revision": self.large_backend.revision,
                "device": self.large_backend.device,
                "precision": self.large_backend.precision,
            },
            "retrieval": {
                "enabled": self.retrieval.enabled,
                "embedder": self.retrieval.embedder,
                "index_type": self.retrieval.index_type,
                "embedding_model_id": self.retrieval.embedding_model_id,
                "embedding_revision": self.retrieval.embedding_revision,
                "top_k": self.retrieval.top_k,
                "min_similarity": self.retrieval.min_similarity,
                "approximate": self.retrieval.index_type == "faiss_hnsw",
            },
            "routing": {
                "policy": self.routing.policy,
                "difficulty_threshold": self.routing.difficulty_threshold,
                "abstain_threshold": self.routing.abstain_threshold,
                "cascade_confidence_threshold": self.routing.cascade_confidence_threshold,
                "retrieval_similarity_threshold": self.routing.retrieval_similarity_threshold,
                "learned_model_configured": self.routing.learned_model_path is not None,
                "retrieval_benefit_model_configured": (
                    self.routing.retrieval_benefit_model_path is not None
                ),
                "cascade_calibrator_configured": (self.routing.cascade_calibrator_path is not None),
                "human_review_enabled": self.routing.human_review_enabled,
            },
            "batching": self.batching.model_dump(mode="json"),
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "max_prompt_chars": self.api.max_prompt_chars,
                "max_request_bytes": self.api.max_request_bytes,
                "authentication_required": self.api.require_api_key,
                "tenant_credentials_configured": self.api.tenant_keys_env is not None,
                "rate_limit_requests": self.api.rate_limit_requests,
                "rate_limit_window_seconds": self.api.rate_limit_window_seconds,
                "metrics_enabled": self.api.metrics_enabled,
                "feedback_enabled": self.api.feedback_enabled,
                "review_enabled": self.api.review_enabled,
                "tls_configured": self.api.tls_certfile is not None,
                "external_tls_termination": self.api.external_tls_termination,
            },
            "monitoring": {
                "enabled": self.monitoring.enabled,
                "window_size": self.monitoring.window_size,
                "minimum_samples": self.monitoring.minimum_samples,
                "drift_threshold": self.monitoring.drift_threshold,
                "external_baseline_configured": self.monitoring.baseline_path is not None,
            },
            "operations": {
                "backend": self.operations.backend,
                "durable": self.operations.backend in {"sqlite", "postgres"},
                "multi_host": self.operations.backend == "postgres",
                "postgres_tls_required": (
                    self.operations.postgres_require_tls
                    if self.operations.backend == "postgres"
                    else None
                ),
                "postgres_auto_migrate": (
                    self.operations.postgres_auto_migrate
                    if self.operations.backend == "postgres"
                    else None
                ),
                "max_global_inflight": self.operations.max_global_inflight,
                "lease_ttl_seconds": self.operations.lease_ttl_seconds,
                "quota_requests": self.operations.quota_requests,
                "quota_window_seconds": self.operations.quota_window_seconds,
                "audit_retention_events": self.operations.audit_retention_events,
            },
            "adaptation": {
                "enabled": self.adaptation.enabled,
                "minimum_labeled_samples": self.adaptation.minimum_labeled_samples,
                "holdout_fraction": self.adaptation.holdout_fraction,
                "minimum_brier_improvement": self.adaptation.minimum_brier_improvement,
                "maximum_ece_regression": self.adaptation.maximum_ece_regression,
            },
            "content_policy": {
                "enabled": self.content_policy.enabled,
                "prompt_rule_count": len(self.content_policy.prompt_blocklist),
                "output_rule_count": len(self.content_policy.output_blocklist),
                "max_metadata_json_chars": self.content_policy.max_metadata_json_chars,
                "max_metadata_depth": self.content_policy.max_metadata_depth,
                "max_metadata_keys": self.content_policy.max_metadata_keys,
            },
        }


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_include(include: str, current_file: Path) -> Path:
    candidate = Path(include)
    if candidate.is_absolute():
        return candidate
    relative = (current_file.parent / candidate).resolve()
    if relative.exists():
        return relative
    return (Path.cwd() / candidate).resolve()


def _load_yaml_tree(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    seen = set() if seen is None else seen
    if resolved in seen:
        raise ConfigurationError(f"cyclic configuration include detected at {resolved}")
    if not resolved.is_file():
        raise ConfigurationError(f"configuration file does not exist: {resolved}")
    seen.add(resolved)
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read YAML configuration {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {resolved}")
    includes = raw.pop("extends", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
        raise ConfigurationError("extends must be a string or list of YAML paths")
    merged: dict[str, Any] = {}
    for include in includes:
        merged = _deep_merge(merged, _load_yaml_tree(_resolve_include(include, resolved), seen))
    seen.remove(resolved)
    return _deep_merge(merged, raw)


def _apply_environment(data: dict[str, Any]) -> dict[str, Any]:
    result = dict(data)
    mappings: dict[str, tuple[str, ...]] = {
        "BUDGETROUTE_OUTPUT_DIR": ("output_dir",),
        "BUDGETROUTE_HOST": ("api", "host"),
        "BUDGETROUTE_PORT": ("api", "port"),
        "BUDGETROUTE_LOG_LEVEL": ("api", "log_level"),
    }
    for variable, path in mappings.items():
        value = os.environ.get(variable)
        if value is None:
            continue
        cursor = result
        for part in path[:-1]:
            nested = cursor.setdefault(part, {})
            if not isinstance(nested, dict):
                raise ConfigurationError(f"cannot apply {variable}: {part} is not a mapping")
            cursor = nested
        cursor[path[-1]] = int(value) if variable == "BUDGETROUTE_PORT" else value
    return result


def load_config(path: str | Path, *, apply_environment: bool = True) -> AppConfig:
    data = _load_yaml_tree(Path(path))
    if apply_environment:
        data = _apply_environment(data)
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid configuration {path}:\n{exc}") from exc


def dump_resolved_config(config: AppConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def validate_runtime_config(config: AppConfig) -> None:
    """Reject explicit optional-runtime requests that this environment cannot honor."""
    transformer_backends = [
        backend
        for backend in (config.small_backend, config.large_backend)
        if backend.type == "transformers"
    ]
    if transformer_backends and importlib.util.find_spec("torch") is None:
        raise ConfigurationError(
            "Transformers execution is configured but PyTorch is not installed; "
            "install the transformers extra"
        )
    if (
        config.retrieval.enabled
        and config.retrieval.embedder == "transformers"
        and any(importlib.util.find_spec(name) is None for name in ("torch", "transformers"))
    ):
        raise ConfigurationError(
            "Transformers retrieval is configured but torch/transformers are not installed"
        )
    if transformer_backends:
        import torch

        if (
            any(backend.device == "cuda" for backend in transformer_backends)
            and not torch.cuda.is_available()
        ):
            raise ConfigurationError(
                "CUDA is explicitly configured but torch.cuda.is_available() is false"
            )
        if any(
            backend.device == "cuda" and backend.precision == "bf16"
            for backend in transformer_backends
        ) and not bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
            raise ConfigurationError("BF16 is configured but the CUDA device does not support it")
        if (
            any(backend.quantization != "none" for backend in transformer_backends)
            and importlib.util.find_spec("bitsandbytes") is None
        ):
            raise ConfigurationError("quantization is configured but bitsandbytes is not installed")
    for backend in (config.small_backend, config.large_backend):
        if backend.type != "openai_compatible":
            continue
        assert backend.base_url is not None
        parsed = urlparse(backend.base_url)
        assert parsed.hostname is not None
        if not _is_loopback_host(parsed.hostname) and not backend.allow_remote_endpoint:
            raise ConfigurationError(
                "remote OpenAI-compatible endpoint is disabled; set "
                "allow_remote_endpoint=true only for an explicitly trusted server"
            )
        if not _is_loopback_host(parsed.hostname) and parsed.scheme != "https":
            raise ConfigurationError("remote OpenAI-compatible endpoints must use HTTPS")
        if backend.api_key_env and not os.environ.get(backend.api_key_env):
            raise ConfigurationError(
                f"OpenAI-compatible credential environment variable is unset: {backend.api_key_env}"
            )
    if config.operations.backend == "postgres":
        if any(importlib.util.find_spec(name) is None for name in ("psycopg", "psycopg_pool")):
            raise ConfigurationError(
                "PostgreSQL operations are configured but the postgres extra is not installed"
            )
        if not os.environ.get(config.operations.postgres_dsn_env):
            raise ConfigurationError(
                "PostgreSQL operations are configured but the named DSN environment variable "
                "is unset"
            )
    if config.api.require_api_key:
        if config.api.tenant_keys_env is not None:
            if not os.environ.get(config.api.tenant_keys_env):
                raise ConfigurationError(
                    "tenant authentication is enabled but its credential environment "
                    "variable is unset"
                )
        elif not os.environ.get(config.api.api_key_env):
            raise ConfigurationError(
                "API authentication is enabled but its legacy API key environment variable is unset"
            )
    if config.api.tls_certfile is not None:
        assert config.api.tls_keyfile is not None
        if not config.api.tls_certfile.is_file() or not config.api.tls_keyfile.is_file():
            raise ConfigurationError("configured TLS certificate or private-key file is missing")
    if config.routing.policy == "learned":
        assert config.routing.learned_model_path is not None
        if not config.routing.learned_model_path.is_file():
            raise ConfigurationError(
                f"learned router artifact does not exist: {config.routing.learned_model_path}"
            )
    if config.routing.policy == "learned_retrieval":
        assert config.routing.retrieval_benefit_model_path is not None
        if not config.routing.retrieval_benefit_model_path.is_file():
            raise ConfigurationError(
                "retrieval-benefit artifact does not exist: "
                f"{config.routing.retrieval_benefit_model_path}"
            )
    if (
        config.routing.cascade_calibrator_path is not None
        and not config.routing.cascade_calibrator_path.is_file()
    ):
        raise ConfigurationError(
            "cascade confidence calibrator does not exist: "
            f"{config.routing.cascade_calibrator_path}"
        )
    if config.benchmark.cache.mode == "read_only" and not config.benchmark.cache.directory.is_dir():
        raise ConfigurationError(
            f"read-only generation cache does not exist: {config.benchmark.cache.directory}"
        )
    if config.retrieval.enabled:
        if (
            config.retrieval.index_type.startswith("faiss")
            and importlib.util.find_spec("faiss") is None
        ):
            raise ConfigurationError("FAISS index requested but faiss is not installed")
        if not config.retrieval.index_path.is_file() and not config.retrieval.corpus_path.exists():
            raise ConfigurationError(
                "retrieval is enabled but neither a saved index nor corpus path is available"
            )
