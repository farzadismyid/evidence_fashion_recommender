"""Deterministic evidence-aware reranking and validation-only Pareto selection."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def minmax_normalize(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Scores must be non-empty and finite.")
    lower, upper = float(array.min()), float(array.max())
    return np.zeros_like(array) if upper == lower else (array - lower) / (upper - lower)


def rerank_candidates(
    candidates: pd.DataFrame,
    *,
    clip_weight: float,
    evidence_weight: float,
) -> pd.DataFrame:
    required = {"item_id", "clip_score", "evidence_score"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidates are missing score fields: {sorted(missing)}")
    if not np.isclose(clip_weight + evidence_weight, 1.0):
        raise ValueError("CLIP and evidence weights must sum to one.")
    result = candidates.copy()
    result["normalized_clip_score"] = minmax_normalize(result["clip_score"])
    result["normalized_evidence_score"] = minmax_normalize(result["evidence_score"])
    pre = result.sort_values(["clip_score", "item_id"], ascending=[False, True], kind="stable")
    result["pre_rerank_rank"] = result["item_id"].map(
        {item_id: rank for rank, item_id in enumerate(pre["item_id"], start=1)}
    )
    result["final_score"] = (
        clip_weight * result["normalized_clip_score"]
        + evidence_weight * result["normalized_evidence_score"]
    )
    result = result.sort_values(
        ["final_score", "clip_score", "item_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    result["post_rerank_rank"] = np.arange(1, len(result) + 1)
    return result


def ranking_metrics(relevance: Iterable[bool]) -> dict[str, float]:
    labels = np.asarray(list(relevance), dtype=np.int8)
    metrics = {}
    for cutoff in (1, 5, 10):
        top = labels[:cutoff]
        discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
        ideal_count = min(int(labels.sum()), cutoff)
        ideal = float(discounts[:ideal_count].sum()) if ideal_count else 0.0
        metrics[f"hr_at_{cutoff}"] = float(top.any())
        metrics[f"ndcg_at_{cutoff}"] = float((top * discounts).sum() / ideal) if ideal else 0.0
    positive_ranks = np.flatnonzero(labels) + 1
    metrics["mrr"] = float(1.0 / positive_ranks[0]) if len(positive_ranks) else 0.0
    return metrics


def pareto_frontier(points: pd.DataFrame, objective_columns: list[str]) -> pd.DataFrame:
    values = points[objective_columns].to_numpy(dtype=np.float64)
    dominated = np.zeros(len(points), dtype=bool)
    for index, value in enumerate(values):
        for other_index, other in enumerate(values):
            if index == other_index:
                continue
            if np.all(other >= value) and np.any(other > value):
                dominated[index] = True
                break
    result = points.copy()
    result["pareto_status"] = np.where(dominated, "dominated", "frontier")
    return result


def select_pareto_knee(
    points: pd.DataFrame,
    objective_columns: list[str],
    *,
    tie_columns: list[str],
) -> pd.Series:
    frontier = points[points["pareto_status"] == "frontier"].copy()
    normalized = []
    for column in objective_columns:
        normalized.append(minmax_normalize(frontier[column]))
    matrix = np.column_stack(normalized)
    frontier["knee_distance_to_ideal"] = np.linalg.norm(1.0 - matrix, axis=1)
    return frontier.sort_values(
        ["knee_distance_to_ideal", *tie_columns], kind="stable"
    ).iloc[0]
