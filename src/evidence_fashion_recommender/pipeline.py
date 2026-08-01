"""End-to-end recommendation pipeline with inspectable intermediate results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AppConfig
from .evaluation.controlled import build_clip_query_text
from .evaluation.evidence_ranking import CandidateEvidenceScorer
from .indexes import PersistentCategoryIndexes
from .models.multimodal import CLIPEmbedder
from .reranking import weighted_rerank


@dataclass
class RecommendationResult:
    query: pd.Series
    candidates: pd.DataFrame
    recommendations: pd.DataFrame
    evidence: dict[str, pd.DataFrame]


def recommend(
    config: AppConfig,
    all_items: pd.DataFrame,
    target_items: pd.DataFrame,
    dataset_split,
    target_clip_embeddings: np.ndarray,
    clip_model: CLIPEmbedder,
    evidence_scorer: CandidateEvidenceScorer | None,
    query_item_id: str,
    target_category: str,
    user_request: str,
    category_indexes: PersistentCategoryIndexes | None = None,
) -> RecommendationResult:
    matches = all_items[all_items["item_ID"].astype(str) == str(query_item_id)]
    if matches.empty:
        raise ValueError(f"Unknown query item ID: {query_item_id}")
    query = matches.iloc[0]
    image = dataset_split[int(query["original_dataset_index"])]["image"].convert("RGB")
    return recommend_from_query(
        config,
        query,
        image,
        target_items,
        target_clip_embeddings,
        clip_model,
        evidence_scorer,
        target_category,
        user_request,
        category_indexes,
    )


def recommend_from_query(
    config: AppConfig,
    query: pd.Series,
    query_image,
    target_items: pd.DataFrame,
    target_clip_embeddings: np.ndarray,
    clip_model: CLIPEmbedder,
    evidence_scorer: CandidateEvidenceScorer | None,
    target_category: str,
    user_request: str,
    category_indexes: PersistentCategoryIndexes | None = None,
) -> RecommendationResult:
    if target_category not in config.dataset.target_categories:
        raise ValueError(f"Unsupported target category: {target_category}")
    case = pd.Series(
        {
            "query_category": query["category"],
            "query_group": query["query_category"],
            "query_text": query["text"],
            "user_request": user_request,
            "target_category": target_category,
        }
    )
    clip_text = clip_model.encode_text([build_clip_query_text(case)])
    clip_image = clip_model.encode_images([query_image])
    query_embedding = clip_model.fuse(clip_image, clip_text)[0]

    mask = target_items["broad_category"] == target_category
    if config.retrieval.exclude_query_outfit:
        mask &= target_items["outfit_ID"].astype(str) != str(query["outfit_ID"])
    if query.get("item_ID"):
        mask &= target_items["item_ID"].astype(str) != str(query["item_ID"])
    eligible = target_items[mask]
    if category_indexes is not None:
        category_size = len(category_indexes.rows[target_category])
        rows, index_scores = category_indexes.search(
            target_category, query_embedding, category_size
        )
        keep = np.isin(rows, eligible.index.to_numpy())
        rows, index_scores = rows[keep], index_scores[keep]
        pool_size = min(config.retrieval.candidate_pool_size, len(rows))
        candidates = target_items.loc[rows[:pool_size]].copy().reset_index(drop=True)
        candidates["clip_score"] = index_scores[:pool_size]
    else:
        eligible_rows = eligible.index.to_numpy()
        scores = np.asarray(target_clip_embeddings[eligible_rows]) @ query_embedding
        pool_size = min(config.retrieval.candidate_pool_size, len(scores))
        selected = np.argpartition(-scores, pool_size - 1)[:pool_size]
        selected = selected[np.argsort(-scores[selected])]
        candidates = eligible.iloc[selected].copy().reset_index(drop=True)
        candidates["clip_score"] = scores[selected]
    candidates["clip_rank"] = np.arange(1, len(candidates) + 1)

    evidence_tables: dict[str, pd.DataFrame] = {}
    if evidence_scorer is not None:
        candidates["evidence_score"] = evidence_scorer.score(case, candidates)
        ranked = weighted_rerank(
            candidates,
            config.reranking.clip_weight,
            config.reranking.evidence_weight,
            config.reranking.normalize_scores,
        )
    else:
        candidates["evidence_score"] = 0.0
        ranked = candidates.assign(final_score=candidates["clip_score"]).sort_values(
            "final_score", ascending=False
        )
    recommendations = ranked.head(config.retrieval.final_top_k).reset_index(drop=True)
    recommendations["rank"] = np.arange(1, len(recommendations) + 1)
    if evidence_scorer is not None:
        for _, candidate in recommendations.iterrows():
            evidence_tables[str(candidate["item_ID"])] = evidence_scorer.retrieve(case, candidate)
    return RecommendationResult(query, candidates, recommendations, evidence_tables)
