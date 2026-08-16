"""Cluster-aware confirmatory statistics for recommendation evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def clustered_bootstrap_mean(
    values: Iterable[float],
    clusters: Iterable[str],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float, np.ndarray]:
    """Return a mean and percentile interval by resampling complete clusters."""
    value_array = np.asarray(list(values), dtype=np.float64)
    cluster_array = np.asarray(list(clusters), dtype=str)
    if len(value_array) == 0 or len(value_array) != len(cluster_array):
        raise ValueError("Values and clusters must be non-empty and have equal length.")
    if not np.isfinite(value_array).all():
        raise ValueError("Bootstrap values must be finite.")
    unique, inverse = np.unique(cluster_array, return_inverse=True)
    sums = np.bincount(inverse, weights=value_array, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(replicates, len(unique)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(value_array.mean()), float(lower), float(upper), estimates


def two_sided_bootstrap_pvalue(estimates: Iterable[float]) -> float:
    """Calculate a conservative two-sided sign probability from bootstrap contrasts."""
    array = np.asarray(list(estimates), dtype=np.float64)
    if len(array) == 0:
        raise ValueError("Bootstrap estimates cannot be empty.")
    lower = (np.count_nonzero(array <= 0) + 1) / (len(array) + 1)
    upper = (np.count_nonzero(array >= 0) + 1) / (len(array) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Return Holm family-wise-error adjusted p-values in original order."""
    values = np.asarray(list(p_values), dtype=np.float64)
    if not len(values) or np.any((values < 0) | (values > 1)):
        raise ValueError("P-values must be a non-empty sequence in [0, 1].")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (len(values) - np.arange(len(values))) * values[order])
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted
