"""Descriptive paired comparisons; not a confirmatory non-inferiority test."""

from __future__ import annotations

import random
from collections import defaultdict
from math import isfinite
from typing import Any

from budgetroute.evaluation.uncertainty import _quantile, grouped_bootstrap_mean
from budgetroute.schemas import PredictionRecord


def _indexed(rows: list[PredictionRecord]) -> dict[str, PredictionRecord]:
    if len({row.policy for row in rows}) > 1:
        raise ValueError("paired comparison requires one policy per population")
    result: dict[str, PredictionRecord] = {}
    for row in rows:
        if row.example_id in result:
            raise ValueError(f"duplicate paired example ID: {row.example_id}")
        result[row.example_id] = row
    return result


def _ratio_interval(
    pairs: list[tuple[PredictionRecord, PredictionRecord]],
    samples: int,
    seed: int,
) -> dict[str, Any] | None:
    clusters: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for left, right in pairs:
        clusters[left.group_id or left.example_id].append((left.latency_ms, right.latency_ms))
    if len(clusters) < 2 or samples <= 0:
        return None
    grouped = [rows for _, rows in sorted(clusters.items())]
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = [pair for _ in grouped for pair in generator.choice(grouped)]
        estimates.append(sum(pair[0] for pair in selected) / sum(pair[1] for pair in selected))
    estimates.sort()
    return {
        "method": "paired_cluster_percentile_bootstrap",
        "confidence_level": 0.95,
        "samples": samples,
        "group_count": len(grouped),
        "estimand": "ratio_of_mean_live_latencies",
        "estimate": sum(a.latency_ms for a, _ in pairs) / sum(b.latency_ms for _, b in pairs),
        "lower": _quantile(estimates, 0.025),
        "upper": _quantile(estimates, 0.975),
    }


def paired_policy_summary(
    candidate: list[PredictionRecord],
    baseline: list[PredictionRecord],
    *,
    samples: int = 2000,
    seed: int = 42,
    candidate_run: dict[str, Any] | None = None,
    baseline_run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    left, right = _indexed(candidate), _indexed(baseline)
    ids = sorted(left.keys() & right.keys())
    result: dict[str, Any] = {
        "claim_type": "descriptive_paired_comparison_not_noninferiority_certification",
        "complete": bool(ids) and left.keys() == right.keys(),
        "paired_count": len(ids),
        "missing_candidate_ids": sorted(right.keys() - left.keys()),
        "missing_baseline_ids": sorted(left.keys() - right.keys()),
        "candidate_failures": sum(row.error is not None for row in candidate),
        "baseline_failures": sum(row.error is not None for row in baseline),
        "accuracy_delta": None,
        "accuracy_delta_confidence_interval": None,
        "wins": None,
        "losses": None,
        "ties": None,
        "independent_group_count": 0,
        "latency_ratio": None,
        "latency_ratio_confidence_interval": None,
        "latency_unavailable_reason": "Incomplete pairs or unverified compatible live evidence",
    }
    if not result["complete"]:
        return result
    pairs = [(left[key], right[key]) for key in ids]
    for a, b in pairs:
        if (
            a.reference != b.reference
            or (a.group_id or a.example_id) != (b.group_id or b.example_id)
            or a.metric_name != b.metric_name
        ):
            raise ValueError("paired records must agree on reference, group and evaluator")
    differences = [
        float(a.error is None and a.quality_score == 1.0)
        - float(b.error is None and b.quality_score == 1.0)
        for a, b in pairs
    ]
    groups = [a.group_id or a.example_id for a, _ in pairs]
    result.update(
        accuracy_delta=sum(differences) / len(differences),
        accuracy_delta_confidence_interval=grouped_bootstrap_mean(
            differences, groups, samples=samples, seed=seed
        ),
        wins=sum(value > 0 for value in differences),
        losses=sum(value < 0 for value in differences),
        ties=sum(value == 0 for value in differences),
        independent_group_count=len(set(groups)),
    )
    contexts = (candidate_run, baseline_run)
    identities = ("hardware_fingerprint", "generation_fingerprint", "dataset_hash", "trial_block")
    compatible = bool(
        candidate_run
        and baseline_run
        and all(
            candidate_run.get(key) and candidate_run.get(key) == baseline_run.get(key)
            for key in identities
        )
        and all(
            context
            and context.get("status") == "completed"
            and context.get("fake") is False
            and context.get("replay") is False
            for context in contexts
        )
    )
    live = all(
        row.fake is False
        and not row.replayed
        and row.error is None
        and isfinite(row.latency_ms)
        and row.latency_ms > 0
        for row in [*candidate, *baseline]
    )
    if compatible and live:
        result.update(
            latency_ratio=sum(a.latency_ms for a, _ in pairs) / sum(b.latency_ms for _, b in pairs),
            latency_ratio_confidence_interval=_ratio_interval(pairs, samples, seed),
            latency_unavailable_reason=None,
        )
    return result
