"""Dependency composition and lifecycle management for inference services."""

from __future__ import annotations

import time
from typing import Any

from budgetroute.backends.base import GenerationBackend
from budgetroute.backends.cached import CachedGenerationBackend
from budgetroute.backends.fake import FakeBackend
from budgetroute.backends.openai_compatible import OpenAICompatibleBackend
from budgetroute.backends.transformers import TransformersBackend
from budgetroute.config import AppConfig, BackendConfig
from budgetroute.experiments.cache import GenerationCache
from budgetroute.inference.content_policy import ContentPolicy
from budgetroute.inference.engine import InferenceEngine, RouteResult
from budgetroute.inference.load import BackendLoadSnapshot, BackendLoadTracker, TrackedBackend
from budgetroute.retrieval.base import EmbeddingProvider
from budgetroute.retrieval.embeddings import FakeEmbedder, TransformersEmbedder
from budgetroute.retrieval.index import ExactCosineIndex, FaissCosineIndex, FaissHNSWIndex
from budgetroute.retrieval.service import RetrievalService
from budgetroute.routing.registry import build_policy
from budgetroute.routing.training import load_backend_confidence_calibrator
from budgetroute.schemas import BackendName, GenerationRequest, GenerationResponse


def _build_backend(name: BackendName, config: BackendConfig) -> GenerationBackend:
    if config.type == "fake":
        return FakeBackend(name, config)
    if config.type == "transformers":
        return TransformersBackend(name, config)
    return OpenAICompatibleBackend(name, config)


def _build_retriever(config: AppConfig) -> RetrievalService | None:
    if not config.retrieval.enabled:
        return None
    if config.retrieval.embedder == "fake":
        embedder: EmbeddingProvider = FakeEmbedder(config.retrieval.embedding_dimension)
    else:
        assert config.retrieval.embedding_model_id is not None
        device = "cuda" if config.mode == "gpu" else "cpu"
        embedder = TransformersEmbedder(
            config.retrieval.embedding_model_id,
            device,
            config.retrieval.embedding_revision,
        )
    index: ExactCosineIndex
    if config.retrieval.index_type == "faiss_hnsw":
        index = FaissHNSWIndex(
            embedder,
            neighbors=config.retrieval.hnsw_neighbors,
            ef_construction=config.retrieval.hnsw_ef_construction,
            ef_search=config.retrieval.hnsw_ef_search,
        )
    elif config.retrieval.index_type == "faiss":
        index = FaissCosineIndex(embedder)
    else:
        index = ExactCosineIndex(embedder)
    return RetrievalService(config.retrieval, index)


class InferenceService:
    def __init__(
        self,
        config: AppConfig,
        engine: InferenceEngine,
        retriever: RetrievalService | None,
        load_trackers: dict[BackendName, BackendLoadTracker],
        content_policy: ContentPolicy,
    ) -> None:
        self.config = config
        self.engine = engine
        self.retriever = retriever
        self.load_trackers = load_trackers
        self.content_policy = content_policy
        self._initialized = False
        self.initialization_ms = 0.0

    def initialize(self) -> None:
        if self._initialized:
            return
        start = time.perf_counter_ns()
        self.engine.small_backend.initialize()
        self.engine.large_backend.initialize()
        if self.retriever is not None:
            self.retriever.initialize()
        self.initialization_ms = (time.perf_counter_ns() - start) / 1_000_000
        self._initialized = True

    def route(self, request: GenerationRequest) -> RouteResult:
        if not self._initialized:
            self.initialize()
        self.content_policy.validate_request(request)
        return self.engine.route(request)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self._initialized:
            self.initialize()
        self.content_policy.validate_request(request)
        response = self.engine.generate(request)
        self.content_policy.validate_response(response)
        return response

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResponse]:
        if not self._initialized:
            self.initialize()
        for request in requests:
            self.content_policy.validate_request(request)
        responses = self.engine.generate_batch(requests)
        for response in responses:
            self.content_policy.validate_response(response)
        return responses

    def load_snapshot(self) -> dict[BackendName, BackendLoadSnapshot]:
        return {name: tracker.snapshot() for name, tracker in self.load_trackers.items()}

    def health(self) -> dict[str, Any]:
        retriever_ready = self.retriever is None or self.retriever.ready
        small = self.engine.small_backend.health()
        large = self.engine.large_backend.health()
        return {
            "initialized": self._initialized,
            "ready": bool(small["ready"] and large["ready"] and retriever_ready),
            "small_backend": small,
            "large_backend": large,
            "retrieval_ready": retriever_ready,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "initialization_ms": self.initialization_ms,
            "small_backend": self.engine.small_backend.metadata(),
            "large_backend": self.engine.large_backend.metadata(),
            "retrieval": self.retriever.metadata() if self.retriever else None,
        }

    def close(self) -> None:
        self.engine.small_backend.cleanup()
        self.engine.large_backend.cleanup()
        self._initialized = False


def build_service(config: AppConfig) -> InferenceService:
    small = _build_backend(BackendName.SMALL, config.small_backend)
    large = _build_backend(BackendName.LARGE, config.large_backend)
    cache_mode = config.benchmark.cache.mode
    if cache_mode != "off":
        cache = GenerationCache(config.benchmark.cache.directory)
        small = CachedGenerationBackend(
            BackendName.SMALL, config.small_backend, small, cache, cache_mode
        )
        large = CachedGenerationBackend(
            BackendName.LARGE, config.large_backend, large, cache, cache_mode
        )
    load_trackers = {
        BackendName.SMALL: BackendLoadTracker(
            BackendName.SMALL, config.small_backend.max_concurrency
        ),
        BackendName.LARGE: BackendLoadTracker(
            BackendName.LARGE, config.large_backend.max_concurrency
        ),
    }
    small = TrackedBackend(small, load_trackers[BackendName.SMALL])
    large = TrackedBackend(large, load_trackers[BackendName.LARGE])
    retriever = _build_retriever(config)

    def load_provider() -> dict[BackendName, BackendLoadSnapshot]:
        return {name: tracker.snapshot() for name, tracker in load_trackers.items()}

    policy = build_policy(
        config.routing,
        load_provider=load_provider,
        small_backend=config.small_backend,
        large_backend=config.large_backend,
    )
    confidence_calibrator = None
    cascade_threshold = None
    if config.routing.cascade_calibrator_path is not None:
        confidence_calibrator, cascade_threshold, _ = load_backend_confidence_calibrator(
            config.routing.cascade_calibrator_path
        )
    engine = InferenceEngine(
        small,
        large,
        policy,
        config.routing,
        retriever,
        confidence_calibrator=confidence_calibrator,
        cascade_confidence_threshold=cascade_threshold.threshold if cascade_threshold else None,
        cascade_threshold_selection=cascade_threshold,
        backend_configs={
            BackendName.SMALL: config.small_backend,
            BackendName.LARGE: config.large_backend,
        },
    )
    return InferenceService(
        config,
        engine,
        retriever,
        load_trackers,
        ContentPolicy(config.content_policy),
    )
