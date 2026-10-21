"""Generation backend wrapper supporting live collection and model-free replay."""

from __future__ import annotations

from typing import Any, Literal

from budgetroute.backends.base import GenerationBackend
from budgetroute.config import BackendConfig
from budgetroute.exceptions import ArtifactError
from budgetroute.experiments.cache import GenerationCache, backend_fingerprint
from budgetroute.schemas import BackendGeneration, BackendName, GenerationRequest

CacheMode = Literal["read_write", "read_only", "refresh"]


class CachedGenerationBackend:
    def __init__(
        self,
        name: BackendName,
        config: BackendConfig,
        delegate: GenerationBackend,
        cache: GenerationCache,
        mode: CacheMode,
    ) -> None:
        if config.type == "openai_compatible" and not config.runtime_identity:
            raise ArtifactError("cached HTTP backend requires an immutable runtime_identity")
        self.name = name
        self.config = config
        self.delegate = delegate
        self.cache = cache
        self.mode = mode
        self._fingerprint = backend_fingerprint(name, config)
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        if self.mode != "read_only":
            self.delegate.initialize()
        self._initialized = True

    def health(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "type": "generation_cache",
            "ready": self._initialized,
            "mode": self.mode,
            "delegate": self.delegate.health() if self.mode != "read_only" else None,
        }

    def token_count(self, text: str) -> int:
        if self.mode == "read_only":
            raise ArtifactError("token counting is unavailable during model-free replay")
        return self.delegate.token_count(text)

    def generate(self, request: GenerationRequest) -> BackendGeneration:
        if not self._initialized:
            raise ArtifactError("cached generation backend is not initialized")
        if self.mode != "refresh":
            cached = self.cache.get(self._fingerprint, request)
            if cached is not None:
                return cached
        if self.mode == "read_only":
            key, _ = self.cache.key_for(self._fingerprint, request)
            raise ArtifactError(
                f"generation cache miss for {self.name.value} backend key {key}; "
                "collect baselines before replay"
            )
        generation = self.delegate.generate(request)
        key = self.cache.put(self._fingerprint, request, generation)
        return generation.model_copy(
            update={"metadata": {**generation.metadata, "cache_hit": False, "cache_key": key}}
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[BackendGeneration]:
        if not self._initialized:
            raise ArtifactError("cached generation backend is not initialized")
        outputs: list[BackendGeneration | None] = [None] * len(requests)
        misses: list[tuple[int, GenerationRequest]] = []
        for index, request in enumerate(requests):
            cached = self.cache.get(self._fingerprint, request) if self.mode != "refresh" else None
            if cached is None:
                misses.append((index, request))
            else:
                outputs[index] = cached
        if misses and self.mode == "read_only":
            key, _ = self.cache.key_for(self._fingerprint, misses[0][1])
            raise ArtifactError(
                f"generation cache miss for {self.name.value} backend key {key}; "
                "collect baselines before replay"
            )
        if misses:
            generated = self.delegate.generate_batch([request for _, request in misses])
            if len(generated) != len(misses):
                raise ArtifactError("delegate returned a different number of batch generations")
            for (index, request), generation in zip(misses, generated, strict=True):
                key = self.cache.put(self._fingerprint, request, generation)
                outputs[index] = generation.model_copy(
                    update={
                        "metadata": {
                            **generation.metadata,
                            "cache_hit": False,
                            "cache_key": key,
                        }
                    }
                )
        if any(output is None for output in outputs):
            raise ArtifactError("generation cache batch did not produce every output")
        return [output for output in outputs if output is not None]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "type": "generation_cache",
            "mode": self.mode,
            "fingerprint": self._fingerprint,
            "cache": self.cache.stats.as_dict(),
            "delegate": self.delegate.metadata() if self.mode != "read_only" else None,
        }

    def cleanup(self) -> None:
        if self.mode != "read_only":
            self.delegate.cleanup()
        self._initialized = False
