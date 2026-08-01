"""Paired bootstrap tests and multiple-comparison corrections."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def paired_bootstrap(
    first: np.ndarray,
    second: np.ndarray,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    if first.shape != second.shape:
        raise ValueError("Paired samples must have equal shape.")
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2])
    p_value = 2 * min(
        float((bootstrap_means <= 0).mean()),
        float((bootstrap_means >= 0).mean()),
    )
    return {
        "mean_difference": float(differences.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": min(p_value, 1.0),
    }


def outfit_clustered_paired_bootstrap(
    frame: pd.DataFrame,
    *,
    outfit_column: str,
    first_column: str,
    second_column: str,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int]:
    """Paired bootstrap that resamples outfits and retains every case in each outfit."""

    paired = frame[[outfit_column, first_column, second_column]].dropna().copy()
    paired["difference"] = paired[first_column].astype(float) - paired[second_column].astype(float)
    clusters = [group["difference"].to_numpy() for _, group in paired.groupby(outfit_column)]
    if not clusters:
        raise ValueError("No complete paired outfit clusters are available.")
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        bootstrap_means[index] = np.concatenate([clusters[value] for value in selected]).mean()
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_difference": float(paired["difference"].mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "cases": len(paired),
        "unique_outfits": len(clusters),
    }


def compare_variants_clustered(
    frame: pd.DataFrame,
    *,
    id_column: str,
    outfit_column: str,
    variant_column: str,
    metrics: list[str],
    comparisons: list[tuple[str, str]],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Evaluate only a predefined paired comparison family and correct within that family."""

    rows = []
    for metric in metrics:
        pivot = frame.pivot_table(
            index=[id_column, outfit_column],
            columns=variant_column,
            values=metric,
            aggfunc="mean",
        ).reset_index()
        for first, second in comparisons:
            if first not in pivot or second not in pivot:
                raise ValueError(f"Missing variant for primary comparison: {first} vs {second}")
            result = outfit_clustered_paired_bootstrap(
                pivot,
                outfit_column=outfit_column,
                first_column=first,
                second_column=second,
                samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=seed,
            )
            rows.append(
                {
                    "metric": metric,
                    "variant_a": first,
                    "variant_b": second,
                    **result,
                }
            )
    return pd.DataFrame(rows)


def holm_correction(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranks = np.arange(1, len(values) + 1)
    raw = values[order] * len(values) / ranks
    adjusted_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def compare_variants(
    frame: pd.DataFrame,
    id_column: str,
    variant_column: str,
    metrics: list[str],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    variants = sorted(frame[variant_column].unique())
    rows = []
    for metric in metrics:
        pivot = frame.pivot(index=id_column, columns=variant_column, values=metric)
        for first, second in combinations(variants, 2):
            paired = pivot[[first, second]].dropna()
            result = paired_bootstrap(
                paired[first].to_numpy(),
                paired[second].to_numpy(),
                bootstrap_samples,
                confidence_level,
                seed,
            )
            rows.append(
                {
                    "metric": metric,
                    "variant_a": first,
                    "variant_b": second,
                    "n": len(paired),
                    **result,
                }
            )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["p_value_holm"] = holm_correction(output["p_value"].to_numpy())
        output["p_value_fdr_bh"] = benjamini_hochberg(output["p_value"].to_numpy())
    return output
