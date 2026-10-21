"""End-to-end routing, retrieval, generation, escalation, and trace orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass

from budgetroute.backends.base import GenerationBackend
from budgetroute.config import RoutingConfig
from budgetroute.exceptions import RetrievalError
from budgetroute.features.request_features import RequestFeatureExtractor
from budgetroute.profiling.memory import current_memory
from budgetroute.profiling.system import Timer
from budgetroute.retrieval.service import RetrievalService
from budgetroute.routing.base import RoutingPolicy
from budgetroute.routing.calibration import ProbabilityCalibrator, ThresholdSelection
from budgetroute.routing.cascade import escalation_reason
from budgetroute.schemas import (
    BackendGeneration,
    BackendName,
    ExecutionTrace,
    GenerationRequest,
    GenerationResponse,
    RetrievalHit,
    RouteDecision,
    RouteName,
    RouterTrace,
    TimingStats,
    UsageStats,
)


@dataclass(frozen=True)
class RouteResult:
    decision: RouteDecision
    retrieval_ms: float
    hits: list[RetrievalHit]


class InferenceEngine:
    def __init__(
        self,
        small_backend: GenerationBackend,
        large_backend: GenerationBackend,
        policy: RoutingPolicy,
        routing_config: RoutingConfig,
        retriever: RetrievalService | None = None,
        feature_extractor: RequestFeatureExtractor | None = None,
        confidence_calibrator: ProbabilityCalibrator | None = None,
        cascade_confidence_threshold: float | None = None,
        cascade_threshold_selection: ThresholdSelection | None = None,
    ) -> None:
        self.small_backend = small_backend
        self.large_backend = large_backend
        self.policy = policy
        self.routing_config = routing_config
        self.retriever = retriever
        self.feature_extractor = feature_extractor or RequestFeatureExtractor()
        self.confidence_calibrator = confidence_calibrator
        self.cascade_threshold_selection = cascade_threshold_selection
        self.cascade_confidence_threshold = (
            cascade_confidence_threshold
            if cascade_confidence_threshold is not None
            else routing_config.cascade_confidence_threshold
        )

    def _escalation_reason(self, confidence: float | None) -> str | None:
        if (
            self.cascade_threshold_selection is not None
            and self.cascade_threshold_selection.status == "infeasible"
        ):
            return f"Routing infeasible: {self.cascade_threshold_selection.reason}"
        return escalation_reason(confidence, self.cascade_confidence_threshold)

    def _calibrate_small_confidence(self, confidence: float | None) -> float | None:
        if confidence is None or self.confidence_calibrator is None:
            return confidence
        return self.confidence_calibrator.transform_one(confidence)

    def route(self, request: GenerationRequest) -> RouteResult:
        hits: list[RetrievalHit] = []
        retrieval_ms = 0.0
        should_retrieve = request.requires_retrieval or self.policy.name in {
            "retrieval_first",
            "learned_retrieval",
        }
        if should_retrieve:
            if self.retriever is None or not self.retriever.ready:
                if not request.allow_abstention:
                    raise RetrievalError(
                        "request requires retrieval but no ready retriever is configured"
                    )
                features = self.feature_extractor.extract(request)
                decision = RouteDecision(
                    route=RouteName.ABSTAIN,
                    confidence=1.0,
                    difficulty_score=None,
                    reason="Retrieval is required but no ready retrieval service is configured",
                    policy=self.policy.name,
                    features=features.model_dump(mode="json"),
                )
                return RouteResult(decision, retrieval_ms, hits)
            hits = self.retriever.search(request.prompt)
            retrieval_ms = self.retriever.last_retrieval_ms
        features = self.feature_extractor.extract(request, hits)
        with Timer() as timer:
            decision = self.policy.decide(request, features)
        decision.thresholds["routing_ms"] = timer.elapsed_ms
        return RouteResult(decision, retrieval_ms, hits)

    def _request_with_context(
        self, request: GenerationRequest, hits: list[RetrievalHit]
    ) -> GenerationRequest:
        context = "\n\n".join(f"[{hit.chunk_id}; score={hit.score:.3f}] {hit.text}" for hit in hits)
        return request.model_copy(
            update={"prompt": f"{request.prompt}\n\nReference context:\n{context}"}
        )

    def _generate(
        self,
        backend: GenerationBackend,
        request: GenerationRequest,
        hits: list[RetrievalHit],
        use_retrieval: bool,
    ) -> BackendGeneration:
        effective_request = self._request_with_context(request, hits) if use_retrieval else request
        return backend.generate(effective_request)

    def _response(
        self,
        request: GenerationRequest,
        route_result: RouteResult,
        *,
        routing_ms: float,
        execution: ExecutionTrace,
        answer: str,
        usage: UsageStats,
        final: BackendGeneration | None,
        initial: BackendGeneration | None = None,
        raw_confidence: float | None = None,
        confidence: float | None = None,
        generation_ms: float = 0.0,
        escalation_ms: float = 0.0,
        wall_total_ms: float,
        batch_size: int = 1,
    ) -> GenerationResponse:
        replayed = bool(
            (initial is not None and initial.metadata.get("cache_hit"))
            or (final is not None and final.metadata.get("cache_hit"))
        )
        total_ms = (
            routing_ms + route_result.retrieval_ms + generation_ms + escalation_ms
            if replayed
            else wall_total_ms
        )
        fake = bool(
            final.metadata.get("fake")
            if final is not None
            else self.small_backend.metadata().get("fake")
            and self.large_backend.metadata().get("fake")
        )
        decision = route_result.decision
        return GenerationResponse(
            request_id=request.request_id,
            answer=answer,
            route=decision.route,
            router=RouterTrace(
                policy=decision.policy,
                difficulty_score=decision.difficulty_score,
                confidence=decision.confidence,
                reason=decision.reason,
                features=decision.features,
                thresholds=decision.thresholds,
                router_probability=decision.router_probability,
            ),
            execution=execution,
            usage=usage,
            timing=TimingStats(
                total_ms=total_ms,
                routing_ms=routing_ms,
                retrieval_ms=route_result.retrieval_ms,
                generation_ms=generation_ms,
                escalation_ms=escalation_ms,
                time_to_first_token_ms=(
                    final.time_to_first_token_ms if final is not None else None
                ),
                replay_overhead_ms=wall_total_ms if replayed else None,
                batch_size=batch_size,
            ),
            memory=current_memory(),
            retrieval=route_result.hits if execution.retrieval_used else [],
            backend_confidence=confidence,
            backend_confidence_raw=raw_confidence,
            backend_confidence_calibrated=bool(
                self.confidence_calibrator is not None
                and final is not None
                and final.backend == BackendName.SMALL
            ),
            confidence_signals=final.confidence_signals if final is not None else None,
            replayed=replayed,
            fake=fake,
            estimated_cost_units=decision.estimated_cost_units,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        total_start = time.perf_counter_ns()
        route_result = self.route(request)
        decision = route_result.decision
        hits = route_result.hits
        routing_ms = float(decision.thresholds.pop("routing_ms", 0.0))
        execution = ExecutionTrace()
        generation_ms = 0.0
        escalation_ms = 0.0
        initial: BackendGeneration | None = None
        final: BackendGeneration | None = None
        raw_confidence: float | None = None

        if decision.route == RouteName.ABSTAIN:
            execution.abstained = True
            answer = "ABSTAIN: insufficient confidence or context."
            usage = UsageStats(input_tokens=0, output_tokens=0)
            confidence: float | None = None
        elif decision.route == RouteName.HUMAN_REVIEW:
            execution.abstained = True
            execution.human_review_required = True
            execution.human_review_reason = decision.reason
            answer = "HUMAN_REVIEW_REQUIRED: connect an external review workflow."
            usage = UsageStats(input_tokens=0, output_tokens=0)
            confidence = None
        elif decision.route in {RouteName.SMALL, RouteName.SMALL_WITH_RETRIEVAL}:
            execution.initial_backend = BackendName.SMALL
            execution.final_backend = BackendName.SMALL
            execution.retrieval_used = decision.route == RouteName.SMALL_WITH_RETRIEVAL
            if execution.retrieval_used and not hits:
                raise RetrievalError(
                    "routing selected retrieval but no context passed the threshold"
                )
            final = self._generate(self.small_backend, request, hits, execution.retrieval_used)
            generation_ms = final.generation_ms
            answer = final.text
            raw_confidence = final.confidence
            confidence = self._calibrate_small_confidence(raw_confidence)
            usage = UsageStats(input_tokens=final.input_tokens, output_tokens=final.output_tokens)
        elif decision.route == RouteName.LARGE:
            execution.initial_backend = BackendName.LARGE
            execution.final_backend = BackendName.LARGE
            final = self._generate(self.large_backend, request, hits, False)
            generation_ms = final.generation_ms
            answer = final.text
            raw_confidence = final.confidence
            confidence = raw_confidence
            usage = UsageStats(input_tokens=final.input_tokens, output_tokens=final.output_tokens)
        elif decision.route == RouteName.CASCADE:
            execution.initial_backend = BackendName.SMALL
            initial = self._generate(
                self.small_backend, request, hits, request.requires_retrieval and bool(hits)
            )
            generation_ms = initial.generation_ms
            calibrated_initial_confidence = self._calibrate_small_confidence(initial.confidence)
            execution.initial_confidence_raw = initial.confidence
            execution.initial_confidence_calibrated = calibrated_initial_confidence
            execution.cascade_threshold = self.cascade_confidence_threshold
            reason = self._escalation_reason(calibrated_initial_confidence)
            if reason is None:
                final = initial
                answer = initial.text
                raw_confidence = initial.confidence
                confidence = calibrated_initial_confidence
                execution.final_backend = BackendName.SMALL
                usage = UsageStats(
                    input_tokens=initial.input_tokens, output_tokens=initial.output_tokens
                )
            else:
                execution.escalated = True
                execution.escalation_reason = reason
                if self.routing_config.retain_initial_answer:
                    execution.initial_answer = initial.text
                final = self._generate(self.large_backend, request, hits, False)
                escalation_ms = final.generation_ms
                answer = final.text
                raw_confidence = final.confidence
                confidence = raw_confidence
                execution.final_backend = BackendName.LARGE
                usage = UsageStats(
                    input_tokens=initial.input_tokens,
                    output_tokens=initial.output_tokens,
                    escalation_input_tokens=final.input_tokens,
                    escalation_output_tokens=final.output_tokens,
                )
        else:  # pragma: no cover - enums and policy validation make this unreachable
            raise ValueError(f"unsupported route: {decision.route}")

        return self._response(
            request,
            route_result,
            routing_ms=routing_ms,
            execution=execution,
            answer=answer,
            usage=usage,
            final=final,
            initial=initial,
            raw_confidence=raw_confidence,
            confidence=confidence,
            generation_ms=generation_ms,
            escalation_ms=escalation_ms,
            wall_total_ms=(time.perf_counter_ns() - total_start) / 1_000_000,
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResponse]:
        """Execute an ordered request batch in small and large backend waves."""

        if not requests:
            return []
        batch_started = time.perf_counter_ns()
        route_results = [self.route(request) for request in requests]
        routing_times = [
            float(result.decision.thresholds.pop("routing_ms", 0.0)) for result in route_results
        ]
        responses: list[GenerationResponse | None] = [None] * len(requests)
        small_jobs: list[tuple[int, bool, bool]] = []
        large_jobs: list[tuple[int, BackendGeneration | None]] = []
        initial_generations: dict[int, BackendGeneration] = {}

        for index, (request, route_result) in enumerate(zip(requests, route_results, strict=True)):
            route = route_result.decision.route
            if route in {RouteName.ABSTAIN, RouteName.HUMAN_REVIEW}:
                human_review = route == RouteName.HUMAN_REVIEW
                execution = ExecutionTrace(
                    abstained=True,
                    human_review_required=human_review,
                    human_review_reason=route_result.decision.reason if human_review else None,
                )
                responses[index] = self._response(
                    request,
                    route_result,
                    routing_ms=routing_times[index],
                    execution=execution,
                    answer=(
                        "HUMAN_REVIEW_REQUIRED: connect an external review workflow."
                        if human_review
                        else "ABSTAIN: insufficient confidence or context."
                    ),
                    usage=UsageStats(input_tokens=0, output_tokens=0),
                    final=None,
                    wall_total_ms=(time.perf_counter_ns() - batch_started) / 1_000_000,
                    batch_size=len(requests),
                )
            elif route in {RouteName.SMALL, RouteName.SMALL_WITH_RETRIEVAL, RouteName.CASCADE}:
                use_retrieval = route == RouteName.SMALL_WITH_RETRIEVAL or (
                    route == RouteName.CASCADE
                    and request.requires_retrieval
                    and bool(route_result.hits)
                )
                if route == RouteName.SMALL_WITH_RETRIEVAL and not route_result.hits:
                    raise RetrievalError(
                        "routing selected retrieval but no context passed the threshold"
                    )
                small_jobs.append((index, use_retrieval, route == RouteName.CASCADE))
            elif route == RouteName.LARGE:
                large_jobs.append((index, None))

        if small_jobs:
            small_requests = [
                self._request_with_context(requests[index], route_results[index].hits)
                if use_retrieval
                else requests[index]
                for index, use_retrieval, _ in small_jobs
            ]
            small_outputs = self.small_backend.generate_batch(small_requests)
            if len(small_outputs) != len(small_jobs):
                raise RuntimeError("small backend returned a different number of batch outputs")
            for (index, use_retrieval, cascade), output in zip(
                small_jobs, small_outputs, strict=True
            ):
                if cascade:
                    initial_generations[index] = output
                    calibrated = self._calibrate_small_confidence(output.confidence)
                    if self._escalation_reason(calibrated) is not None:
                        large_jobs.append((index, output))
                        continue
                execution = ExecutionTrace(
                    initial_backend=BackendName.SMALL,
                    final_backend=BackendName.SMALL,
                    retrieval_used=use_retrieval,
                    initial_confidence_raw=output.confidence if cascade else None,
                    initial_confidence_calibrated=(
                        self._calibrate_small_confidence(output.confidence) if cascade else None
                    ),
                    cascade_threshold=self.cascade_confidence_threshold if cascade else None,
                )
                responses[index] = self._response(
                    requests[index],
                    route_results[index],
                    routing_ms=routing_times[index],
                    execution=execution,
                    answer=output.text,
                    usage=UsageStats(
                        input_tokens=output.input_tokens, output_tokens=output.output_tokens
                    ),
                    final=output,
                    initial=output if cascade else None,
                    raw_confidence=output.confidence,
                    confidence=self._calibrate_small_confidence(output.confidence),
                    generation_ms=output.generation_ms,
                    wall_total_ms=(time.perf_counter_ns() - batch_started) / 1_000_000,
                    batch_size=len(small_jobs),
                )

        if large_jobs:
            large_outputs = self.large_backend.generate_batch(
                [requests[index] for index, _ in large_jobs]
            )
            if len(large_outputs) != len(large_jobs):
                raise RuntimeError("large backend returned a different number of batch outputs")
            for (index, initial), output in zip(large_jobs, large_outputs, strict=True):
                escalated = initial is not None
                reason = (
                    self._escalation_reason(self._calibrate_small_confidence(initial.confidence))
                    if initial is not None
                    else None
                )
                execution = ExecutionTrace(
                    initial_backend=BackendName.SMALL if escalated else BackendName.LARGE,
                    final_backend=BackendName.LARGE,
                    escalated=escalated,
                    escalation_reason=reason,
                    initial_answer=(
                        initial.text
                        if initial is not None and self.routing_config.retain_initial_answer
                        else None
                    ),
                    initial_confidence_raw=initial.confidence if initial is not None else None,
                    initial_confidence_calibrated=(
                        self._calibrate_small_confidence(initial.confidence)
                        if initial is not None
                        else None
                    ),
                    cascade_threshold=self.cascade_confidence_threshold if escalated else None,
                )
                usage = UsageStats(
                    input_tokens=initial.input_tokens
                    if initial is not None
                    else output.input_tokens,
                    output_tokens=(
                        initial.output_tokens if initial is not None else output.output_tokens
                    ),
                    escalation_input_tokens=output.input_tokens if initial is not None else 0,
                    escalation_output_tokens=output.output_tokens if initial is not None else 0,
                )
                responses[index] = self._response(
                    requests[index],
                    route_results[index],
                    routing_ms=routing_times[index],
                    execution=execution,
                    answer=output.text,
                    usage=usage,
                    final=output,
                    initial=initial_generations.get(index),
                    raw_confidence=output.confidence,
                    confidence=output.confidence,
                    generation_ms=initial.generation_ms
                    if initial is not None
                    else output.generation_ms,
                    escalation_ms=output.generation_ms if initial is not None else 0.0,
                    wall_total_ms=(time.perf_counter_ns() - batch_started) / 1_000_000,
                    batch_size=len(large_jobs),
                )

        if any(response is None for response in responses):
            raise RuntimeError("batch execution did not produce every response")
        return [response for response in responses if response is not None]
