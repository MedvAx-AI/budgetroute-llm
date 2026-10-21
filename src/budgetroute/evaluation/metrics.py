"""Aggregate quality, system, route, and calibration metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from budgetroute.evaluation.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
)
from budgetroute.evaluation.routing_metrics import selective_metrics
from budgetroute.evaluation.uncertainty import grouped_bootstrap_mean
from budgetroute.profiling.latency import summarize_latencies
from budgetroute.schemas import PredictionRecord


def aggregate_metrics(
    predictions: list[PredictionRecord],
    *,
    wall_time_ms: float | None = None,
    bootstrap_samples: int = 1000,
    minimum_samples_for_claims: int = 100,
    quality_threshold: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    successful = [item for item in predictions if item.error is None]
    qualities = [item.quality_score for item in successful]
    latencies = [item.latency_ms for item in successful]
    latency = summarize_latencies(latencies)
    generation_latency = summarize_latencies(
        [item.generation_ms for item in successful if item.generation_ms > 0]
    )
    retrieval_latency = summarize_latencies(
        [item.retrieval_ms for item in successful if item.retrieval_ms > 0]
    )
    ttft = summarize_latencies(
        [
            item.time_to_first_token_ms
            for item in successful
            if item.time_to_first_token_ms is not None
        ]
    )
    queue_latency = summarize_latencies([item.queue_ms for item in successful if item.queue_ms > 0])
    by_category: dict[str, list[float]] = defaultdict(list)
    for item in successful:
        by_category[item.category].append(item.quality_score)
    route_counts = Counter(item.route.value for item in successful)
    measurement_seconds = wall_time_ms / 1000.0 if wall_time_ms is not None else None
    answer_probabilities = [item for item in successful if item.backend_confidence is not None]
    outcomes = [int(item.quality_score >= quality_threshold) for item in answer_probabilities]
    confidences = [
        float(item.backend_confidence)
        for item in answer_probabilities
        if item.backend_confidence is not None
    ]
    selective = selective_metrics(qualities, [item.abstained for item in successful])
    return {
        "request_count": len(predictions),
        "successful_requests": len(successful),
        "failed_requests": len(predictions) - len(successful),
        "mean_quality": sum(qualities) / len(qualities) if qualities else None,
        "mean_quality_confidence_interval": grouped_bootstrap_mean(
            qualities,
            [item.group_id or item.example_id for item in successful],
            samples=bootstrap_samples,
            seed=seed,
        ),
        "scientific_claim_warning": (
            None
            if len({item.group_id or item.example_id for item in successful})
            >= minimum_samples_for_claims
            else (
                "too few independent groups for stable comparative claims: "
                f"{len({item.group_id or item.example_id for item in successful})} < "
                f"{minimum_samples_for_claims}"
            )
        ),
        "accuracy": sum(score == 1.0 for score in qualities) / len(qualities)
        if qualities
        else None,
        "quality_by_category": {
            category: sum(scores) / len(scores) for category, scores in sorted(by_category.items())
        },
        "route_distribution": {
            route: count / len(successful) for route, count in sorted(route_counts.items())
        }
        if successful
        else {},
        "route_counts": dict(sorted(route_counts.items())),
        "latency": {
            "count": latency.count,
            "mean_ms": latency.mean_ms,
            "p50_ms": latency.p50_ms,
            "p95_ms": latency.p95_ms,
            "p99_ms": latency.p99_ms,
            "warning": latency.warning,
        },
        "generation_latency": {
            "count": generation_latency.count,
            "mean_ms": generation_latency.mean_ms,
            "p50_ms": generation_latency.p50_ms,
            "p95_ms": generation_latency.p95_ms,
            "warning": generation_latency.warning,
        },
        "retrieval_latency": {
            "count": retrieval_latency.count,
            "mean_ms": retrieval_latency.mean_ms,
            "p50_ms": retrieval_latency.p50_ms,
            "p95_ms": retrieval_latency.p95_ms,
            "warning": retrieval_latency.warning,
        },
        "time_to_first_token": {
            "count": ttft.count,
            "mean_ms": ttft.mean_ms,
            "p50_ms": ttft.p50_ms,
            "p95_ms": ttft.p95_ms,
            "warning": ttft.warning,
        },
        "queue_latency": {
            "count": queue_latency.count,
            "mean_ms": queue_latency.mean_ms,
            "p50_ms": queue_latency.p50_ms,
            "p95_ms": queue_latency.p95_ms,
            "warning": queue_latency.warning,
        },
        "measurement_wall_ms": wall_time_ms,
        "throughput_requests_per_second": (
            len(successful) / measurement_seconds if measurement_seconds else None
        ),
        "input_tokens": sum(item.input_tokens for item in successful),
        "output_tokens": sum(item.output_tokens for item in successful),
        "generated_tokens_per_second": (
            sum(item.output_tokens for item in successful) / measurement_seconds
            if measurement_seconds
            else None
        ),
        "batch_sizes": dict(sorted(Counter(item.batch_size for item in successful).items())),
        "process_rss_mb_peak": max(
            (item.process_rss_mb for item in successful if item.process_rss_mb is not None),
            default=None,
        ),
        "peak_cuda_mb": max(
            (item.peak_cuda_mb for item in successful if item.peak_cuda_mb is not None),
            default=None,
        ),
        "escalations": sum(item.escalated for item in successful),
        "abstentions": sum(item.abstained for item in successful),
        "human_review_requests": sum(item.human_review_required for item in successful),
        "estimated_cost_units": sum(item.estimated_cost_units or 0.0 for item in successful),
        "replayed_requests": sum(item.replayed for item in successful),
        "confidence_methods": dict(
            sorted(
                Counter(
                    item.confidence_method or "router_or_unknown" for item in successful
                ).items()
            )
        ),
        **selective,
        "brier_score": brier_score(confidences, outcomes),
        "probability_count": len(confidences),
        "calibration_target": "final_answer_correctness",
        "expected_calibration_error": expected_calibration_error(confidences, outcomes),
        "reliability": reliability_diagram_data(confidences, outcomes),
        "confidence_correctness_threshold": quality_threshold,
    }
