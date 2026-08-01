"""Matched-candidate modality and CLIP fusion-weight evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.multimodal import fuse_embeddings
from .ranking import ranking_metrics


def score_modality_candidates(
    candidate_minilm: np.ndarray,
    candidate_clip_image: np.ndarray,
    candidate_clip_text: np.ndarray,
    query_minilm: np.ndarray,
    query_clip_image: np.ndarray,
    query_clip_text: np.ndarray,
    fusion_image_weights: list[float],
) -> dict[str, np.ndarray]:
    """Return all modality scores using one fixed candidate set."""

    scores = {
        "minilm_text": np.asarray(candidate_minilm) @ np.asarray(query_minilm),
        "clip_image": np.asarray(candidate_clip_image) @ np.asarray(query_clip_image),
        "clip_text": np.asarray(candidate_clip_text) @ np.asarray(query_clip_text),
    }
    for image_weight in fusion_image_weights:
        candidate_fused = fuse_embeddings(
            np.asarray(candidate_clip_image),
            np.asarray(candidate_clip_text),
            image_weight,
        )
        query_fused = fuse_embeddings(
            np.asarray(query_clip_image)[None, :],
            np.asarray(query_clip_text)[None, :],
            image_weight,
        )[0]
        scores[f"clip_fused_i{image_weight:.2f}"] = candidate_fused @ query_fused
    return scores


def evaluate_modality_case(
    *,
    case_id: str,
    outfit_id: str,
    target_category: str,
    relevance: np.ndarray,
    scores: dict[str, np.ndarray],
    cutoffs: list[int],
) -> pd.DataFrame:
    """Evaluate precomputed method scores while preserving paired case identity."""

    labels = np.asarray(relevance, dtype=np.int8)
    rows = []
    for method, values in scores.items():
        order = np.argsort(-np.asarray(values), kind="stable")
        rows.append(
            {
                "paper_case_id": case_id,
                "query_outfit_id": outfit_id,
                "target_category": target_category,
                "method": method,
                "num_candidates": len(labels),
                "num_positives": int(labels.sum()),
                **ranking_metrics(labels[order], cutoffs),
            }
        )
    return pd.DataFrame(rows)


def summarize_modality_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        column
        for column in results.columns
        if column.startswith(("precision_", "recall_", "hit_rate_", "ndcg_"))
        or column == "reciprocal_rank"
    ]
    return results.groupby("method", as_index=False)[metrics].mean()


def select_fusion_weight(summary: pd.DataFrame) -> pd.Series:
    """Apply the frozen validation hierarchy, ending with balanced-weight preference."""

    fused = summary[summary["method"].str.startswith("clip_fused_i")].copy()
    if fused.empty:
        raise ValueError("No fused CLIP rows are available for selection.")
    fused["image_weight"] = fused["method"].str.rsplit("i", n=1).str[-1].astype(float)
    fused["balance_distance"] = (fused["image_weight"] - 0.5).abs()
    ranked = fused.sort_values(
        ["ndcg_at_10", "hit_rate_at_10", "reciprocal_rank", "balance_distance"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    selected = ranked.iloc[0].copy()
    selected["text_weight"] = 1.0 - float(selected["image_weight"])
    return selected
