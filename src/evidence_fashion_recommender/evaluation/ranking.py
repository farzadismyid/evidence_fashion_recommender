"""Ranking metrics and controlled candidate-set construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ranking_metrics(relevance: list[int] | np.ndarray, cutoffs: list[int]) -> dict[str, float]:
    labels = np.asarray(relevance, dtype=np.int8)
    positives = max(int(labels.sum()), 1)
    metrics: dict[str, float] = {}
    for k in cutoffs:
        top = labels[:k]
        hits = int(top.sum())
        discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
        dcg = float((top * discounts).sum())
        ideal_count = min(positives, k)
        idcg = float(discounts[:ideal_count].sum()) if ideal_count else 0.0
        metrics[f"precision_at_{k}"] = hits / k
        metrics[f"recall_at_{k}"] = hits / positives
        metrics[f"hit_rate_at_{k}"] = float(hits > 0)
        metrics[f"ndcg_at_{k}"] = dcg / idcg if idcg else 0.0
    positive_ranks = np.flatnonzero(labels) + 1
    metrics["reciprocal_rank"] = 1.0 / positive_ranks[0] if len(positive_ranks) else 0.0
    return metrics


def build_controlled_candidate_set(
    items: pd.DataFrame,
    query_outfit_id: str,
    target_category: str,
    negatives: int,
    rng: np.random.Generator,
    query_item_id: str | None = None,
) -> pd.DataFrame:
    category_items = items[items["broad_category"] == target_category]
    positives = category_items[category_items["outfit_ID"].astype(str) == str(query_outfit_id)]
    if query_item_id is not None:
        positives = positives[positives["item_ID"].astype(str) != str(query_item_id)]
    negative_pool = category_items[category_items["outfit_ID"].astype(str) != str(query_outfit_id)]
    if query_item_id is not None:
        negative_pool = negative_pool[negative_pool["item_ID"].astype(str) != str(query_item_id)]
    if positives.empty:
        raise ValueError("No same-outfit positive exists for the requested target category.")
    sample_size = min(negatives, len(negative_pool))
    sampled_negatives = negative_pool.sample(
        n=sample_size,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    result = pd.concat([positives.assign(is_positive=1), sampled_negatives.assign(is_positive=0)])
    return result.sample(
        frac=1.0,
        random_state=int(rng.integers(0, 1_000_000)),
    ).reset_index(drop=True)


def aggregate_ranking_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in results.columns
        if column.startswith(("precision_", "recall_", "hit_rate_", "ndcg_"))
        or column == "reciprocal_rank"
    ]
    return results.groupby("model_name")[metric_columns].mean().reset_index()
