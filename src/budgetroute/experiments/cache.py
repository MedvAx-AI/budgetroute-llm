"""Content-addressed, backend-neutral generation cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from budgetroute.config import BackendConfig
from budgetroute.exceptions import ArtifactError
from budgetroute.experiments.artifacts import ArtifactWriter
from budgetroute.schemas import BackendGeneration, BackendName, GenerationRequest

CACHE_SCHEMA_VERSION = 2


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def backend_fingerprint(name: BackendName, config: BackendConfig) -> dict[str, Any]:
    """Return only inputs that can change generated content or measured model cost."""

    return {
        "backend": name.value,
        "type": config.type,
        "model_id": config.model_id,
        "tokenizer_id": config.tokenizer_id or config.model_id,
        "revision": config.revision,
        "tokenizer_revision": config.tokenizer_revision or config.revision,
        "device": config.device,
        "precision": config.precision,
        "quantization": config.quantization,
        "compile": config.compile,
        "endpoint_namespace": _canonical_hash(config.base_url.rstrip("/"))
        if config.base_url
        else None,
        "runtime_identity": config.runtime_identity,
        "request_logprobs": config.request_logprobs,
        "model_loader": config.model_loader,
        "load_direct_to_device": config.load_direct_to_device,
        "max_input_tokens": config.max_input_tokens,
        "trust_remote_code": config.trust_remote_code,
        "quality": config.quality if config.type == "fake" else None,
        "artificial_latency_ms": config.artificial_latency_ms if config.type == "fake" else None,
        "generation": config.generation.model_dump(mode="json"),
    }


class GenerationCacheEntry(BaseModel):
    schema_version: Literal[2] = 2
    key: str = Field(min_length=64, max_length=64)
    backend_fingerprint: dict[str, Any]
    request_fingerprint: dict[str, Any]
    generation: BackendGeneration


@dataclass
class GenerationCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}


class GenerationCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.stats = GenerationCacheStats()

    def key_for(
        self,
        backend: dict[str, Any],
        request: GenerationRequest,
    ) -> tuple[str, dict[str, Any]]:
        request_fingerprint = request.model_dump(
            mode="json", exclude={"request_id"}, exclude_none=True
        )
        key = _canonical_hash(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "backend": backend,
                "request": request_fingerprint,
            }
        )
        return key, request_fingerprint

    def path_for(self, key: str) -> Path:
        return self.directory / f"v{CACHE_SCHEMA_VERSION}" / key[:2] / f"{key}.json"

    def get(self, backend: dict[str, Any], request: GenerationRequest) -> BackendGeneration | None:
        key, _ = self.key_for(backend, request)
        path = self.path_for(key)
        if not path.is_file():
            self.stats.misses += 1
            return None
        try:
            entry = GenerationCacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            raise ArtifactError(f"invalid generation cache entry {path}: {exc}") from exc
        if entry.key != key or entry.backend_fingerprint != backend:
            raise ArtifactError(f"generation cache integrity check failed: {path}")
        self.stats.hits += 1
        return entry.generation.model_copy(
            update={"metadata": {**entry.generation.metadata, "cache_hit": True, "cache_key": key}}
        )

    def put(
        self,
        backend: dict[str, Any],
        request: GenerationRequest,
        generation: BackendGeneration,
    ) -> str:
        key, request_fingerprint = self.key_for(backend, request)
        path = self.path_for(key)
        entry = GenerationCacheEntry(
            key=key,
            backend_fingerprint=backend,
            request_fingerprint=request_fingerprint,
            generation=generation.model_copy(
                update={
                    "metadata": {
                        key: value
                        for key, value in generation.metadata.items()
                        if key not in {"cache_hit", "cache_key"}
                    }
                }
            ),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        ArtifactWriter(path.parent).write_json(path.name, entry.model_dump(mode="json"))
        self.stats.writes += 1
        return key
