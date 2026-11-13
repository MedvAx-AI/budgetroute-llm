"""Deterministic group-aware bootstrap uncertainty for benchmark summaries."""

from __future__ import annotations

import random
from collections import defaultdict


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a quantile from no values")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def grouped_bootstrap_mean(
    values: list[float],
    groups: list[str],
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, float | int | str] | None:
    if not values or len(values) != len(groups) or samples <= 0:
        return None
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped[group].append(value)
    clusters = [items for _, items in sorted(grouped.items())]
    if len(clusters) < 2:
        return None
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        drawn = [generator.choice(clusters) for _ in clusters]
        estimates.append(sum(sum(items) for items in drawn) / sum(len(items) for items in drawn))
    estimates.sort()
    return {
        "method": "grouped_percentile_bootstrap",
        "confidence_level": 0.95,
        "samples": samples,
        "group_count": len(clusters),
        "estimate": sum(values) / len(values),
        "estimand": "observation_weighted",
        "lower": _quantile(estimates, 0.025),
        "upper": _quantile(estimates, 0.975),
    }
