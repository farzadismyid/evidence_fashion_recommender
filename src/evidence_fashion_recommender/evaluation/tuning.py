"""Validation-only evidence-reranking hyperparameter selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..reranking import weighted_rerank
from .controlled import QueryEmbeddings
from .evidence_ranking import CandidateEvidenceScorer
from .ranking import build_controlled_candidate_set, ranking_metrics


def evaluate_reranking_grid(
    config: AppConfig,
    cases: pd.DataFrame,
    targets: pd.DataFrame,
    target_clip_embeddings: np.ndarray,
    query_embeddings: QueryEmbeddings,
    evidence_scorer: CandidateEvidenceScorer,
    clip_weights: list[float],
) -> pd.DataFrame:
    id_to_row = {item_id: index for index, item_id in enumerate(targets["item_ID"].astype(str))}
    rows = []
    cases = cases.reset_index(drop=True)
    for case_index, case in cases.iterrows():
        candidates = build_controlled_candidate_set(
            targets,
            str(case["query_outfit_id"]),
            str(case["target_category"]),
            config.evaluation.negatives_per_case,
            np.random.default_rng(config.project.seed + case_index),
            query_item_id=str(case["query_item_id"]),
        )
        embedding_rows = [id_to_row[item_id] for item_id in candidates["item_ID"].astype(str)]
        candidates = candidates.copy()
        candidates["clip_score"] = (
            np.asarray(target_clip_embeddings[embedding_rows])
            @ query_embeddings.clip_fused[case_index]
        )
        candidates["evidence_score"] = evidence_scorer.score(case, candidates)
        for clip_weight in clip_weights:
            ranked = weighted_rerank(
                candidates,
                clip_weight,
                1.0 - clip_weight,
                config.reranking.normalize_scores,
            )
            relevance = ranked["is_positive"].astype(int).tolist()
            rows.append(
                {
                    "paper_case_id": (f"VAL_{case_index:04d}_{case['target_category']}"),
                    "query_outfit_id": case["query_outfit_id"],
                    "target_category": case["target_category"],
                    "clip_weight": clip_weight,
                    "evidence_weight": 1.0 - clip_weight,
                    **ranking_metrics(relevance, config.evaluation.cutoffs),
                }
            )
    return pd.DataFrame(rows)


def select_reranking_weight(results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in results.columns
        if column.startswith(("precision_", "recall_", "hit_rate_", "ndcg_"))
        or column in {"reciprocal_rank", "positive_rank"}
    ]
    summary = (
        results.groupby(["clip_weight", "evidence_weight"])[metric_columns].mean().reset_index()
    )
    return summary.sort_values(
        ["ndcg_at_10", "hit_rate_at_10", "reciprocal_rank"],
        ascending=False,
    ).reset_index(drop=True)
