"""Aggregation helpers for confirmatory recommendation evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from evidence_fashion.evaluation.statistics import clustered_bootstrap_mean


def aggregate_recommendation_metrics(
    case_metrics: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Create micro, category, and category-macro result rows with clustered CIs."""
    rows: list[dict[str, object]] = []
    for method_index, (method, method_rows) in enumerate(case_metrics.groupby("method", sort=True)):
        groups = [("micro", "all", method_rows)]
        groups.extend(
            ("category", category, group)
            for category, group in method_rows.groupby("target_category", sort=True)
        )
        for level, category, group in groups:
            for metric_index, metric in enumerate(metric_columns):
                estimate, lower, upper, _ = clustered_bootstrap_mean(
                    group[metric],
                    group["query_outfit_id"],
                    replicates=replicates,
                    confidence_level=confidence_level,
                    seed=seed + 1000 * method_index + 10 * metric_index,
                )
                rows.append(
                    {
                        "aggregation": level,
                        "category": category,
                        "method": method,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "cases": len(group),
                        "query_outfits": group["query_outfit_id"].nunique(),
                    }
                )
        category_means = method_rows.groupby("target_category")[list(metric_columns)].mean()
        unique_clusters, cluster_codes = np.unique(
            method_rows["query_outfit_id"].astype(str), return_inverse=True
        )
        unique_categories, category_codes = np.unique(
            method_rows["target_category"].astype(str), return_inverse=True
        )
        rng = np.random.default_rng(seed + 100_000 + method_index)
        draws = rng.integers(
            0,
            len(unique_clusters),
            size=(replicates, len(unique_clusters)),
        )
        alpha = (1.0 - confidence_level) / 2.0
        for metric in metric_columns:
            sums = np.zeros((len(unique_clusters), len(unique_categories)), dtype=float)
            counts = np.zeros_like(sums)
            np.add.at(
                sums,
                (cluster_codes, category_codes),
                method_rows[metric].to_numpy(float),
            )
            np.add.at(counts, (cluster_codes, category_codes), 1.0)
            sampled_sums = sums[draws].sum(axis=1)
            sampled_counts = counts[draws].sum(axis=1)
            bootstrap_macro = np.nanmean(
                np.divide(
                    sampled_sums,
                    sampled_counts,
                    out=np.full_like(sampled_sums, np.nan),
                    where=sampled_counts > 0,
                ),
                axis=1,
            )
            lower, upper = np.quantile(bootstrap_macro, [alpha, 1.0 - alpha])
            rows.append(
                {
                    "aggregation": "category_macro",
                    "category": "all",
                    "method": method,
                    "metric": metric,
                    "estimate": float(category_means[metric].mean()),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "cases": len(method_rows),
                    "query_outfits": method_rows["query_outfit_id"].nunique(),
                }
            )
    return pd.DataFrame(rows)
