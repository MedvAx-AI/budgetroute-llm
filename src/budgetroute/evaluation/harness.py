"""Evaluate a configured inference service on validated benchmark records."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from budgetroute.evaluation.evaluators import evaluate_answer
from budgetroute.inference.service import InferenceService
from budgetroute.schemas import (
    BenchmarkRecord,
    GenerationRequest,
    GenerationResponse,
    PredictionRecord,
    RouteName,
)


def request_for_record(record: BenchmarkRecord, *, fake: bool) -> GenerationRequest:
    metadata = dict(record.metadata)
    metadata["must_abstain"] = record.must_abstain
    if fake:
        metadata["fake_reference_answer"] = record.reference_answer
        metadata.setdefault(
            "fake_small_success",
            record.category in {"factual", "classification", "simple_reasoning"}
            and not record.requires_retrieval,
        )
    return GenerationRequest(
        request_id=record.id,
        prompt=record.prompt,
        requires_retrieval=record.requires_retrieval,
        metadata=metadata,
    )


def evaluate_records(
    service: InferenceService,
    records: list[BenchmarkRecord],
    *,
    fake: bool,
    batch_size: int = 1,
    concurrency: int = 1,
) -> tuple[list[PredictionRecord], list[dict[str, object]], list[dict[str, object]]]:
    predictions: list[PredictionRecord] = []
    routes: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    batches = [
        records[offset : offset + batch_size] for offset in range(0, len(records), batch_size)
    ]

    def execute_batch(
        batch: list[BenchmarkRecord],
    ) -> tuple[list[BenchmarkRecord], list[GenerationResponse] | None, Exception | None]:
        requests = [request_for_record(record, fake=fake) for record in batch]
        try:
            responses = (
                service.generate_batch(requests)
                if len(requests) > 1 or batch_size > 1
                else [service.generate(requests[0])]
            )
        except Exception as exc:
            return batch, None, exc
        return batch, responses, None

    if concurrency == 1:
        results = [execute_batch(batch) for batch in batches]
    else:
        with ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="budgetroute-eval"
        ) as pool:
            results = list(pool.map(execute_batch, batches))

    for batch, responses, error in results:
        if error is not None:
            for record in batch:
                predictions.append(
                    PredictionRecord(
                        example_id=record.id,
                        group_id=record.group_id or record.id,
                        source=record.source,
                        source_split=record.source_split,
                        category=record.category,
                        policy=service.engine.policy.name,
                        route=RouteName.ABSTAIN,
                        backend=None,
                        raw_answer="",
                        normalized_answer="",
                        reference=record.reference_answer,
                        quality_score=0.0,
                        metric_name=record.evaluation_type.value,
                        latency_ms=0.0,
                        input_tokens=0,
                        output_tokens=0,
                        retrieval_used=False,
                        escalated=False,
                        abstained=True,
                        confidence=0.0,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
            continue
        assert responses is not None
        for record, response in zip(batch, responses, strict=True):
            evaluation = evaluate_answer(response.answer, record)
            effective_confidence = (
                response.backend_confidence
                if response.backend_confidence is not None
                else response.router.confidence
            )
            predictions.append(
                PredictionRecord(
                    example_id=record.id,
                    group_id=record.group_id or record.id,
                    source=record.source,
                    source_split=record.source_split,
                    category=record.category,
                    policy=response.router.policy,
                    route=response.route,
                    backend=response.execution.final_backend,
                    raw_answer=response.answer,
                    normalized_answer=evaluation.normalized_answer,
                    reference=record.reference_answer,
                    quality_score=0.0 if response.execution.answer_truncated else evaluation.score,
                    answer_truncated=response.execution.answer_truncated,
                    metric_name=evaluation.metric_name,
                    latency_ms=response.timing.total_ms,
                    generation_ms=response.timing.generation_ms,
                    retrieval_ms=response.timing.retrieval_ms,
                    queue_ms=response.timing.queue_ms,
                    batch_size=response.timing.batch_size,
                    time_to_first_token_ms=response.timing.time_to_first_token_ms,
                    input_tokens=response.usage.input_tokens
                    + response.usage.escalation_input_tokens,
                    output_tokens=response.usage.output_tokens
                    + response.usage.escalation_output_tokens,
                    process_rss_mb=response.memory.process_rss_mb,
                    peak_cuda_mb=response.memory.peak_cuda_mb,
                    retrieval_used=response.execution.retrieval_used,
                    escalated=response.execution.escalated,
                    abstained=response.execution.abstained,
                    human_review_required=response.execution.human_review_required,
                    estimated_cost_units=response.estimated_cost_units,
                    confidence=effective_confidence,
                    backend_confidence=response.backend_confidence,
                    backend_confidence_raw=response.backend_confidence_raw,
                    backend_confidence_calibrated=response.backend_confidence_calibrated,
                    confidence_method=(
                        response.confidence_signals.method
                        if response.confidence_signals is not None
                        else None
                    ),
                    replayed=response.replayed,
                    router_probability=response.router.router_probability,
                    fake=response.fake,
                    backend_calls=response.backend_calls,
                )
            )
            routes.append(
                {
                    "example_id": record.id,
                    "group_id": record.group_id or record.id,
                    "policy": response.router.policy,
                    "route": response.route.value,
                    "confidence": response.router.confidence,
                    "router_probability": response.router.router_probability.model_dump(mode="json")
                    if response.router.router_probability
                    else None,
                    "features": response.router.features,
                    "estimated_cost_units": response.estimated_cost_units,
                    "escalated": response.execution.escalated,
                    "human_review_required": response.execution.human_review_required,
                }
            )
            timings.append({"example_id": record.id, **response.timing.model_dump(mode="json")})
    return predictions, routes, timings
