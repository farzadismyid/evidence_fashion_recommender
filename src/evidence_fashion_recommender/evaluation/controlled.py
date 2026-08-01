"""Controlled same-outfit ranking evaluation from the notebook methodology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..models.multimodal import CLIPEmbedder
from ..models.text import SentenceTransformerEmbedder
from ..reranking import weighted_rerank
from .evidence_ranking import CandidateEvidenceScorer
from .ranking import build_controlled_candidate_set, ranking_metrics

REQUEST_TEMPLATES = {
    "shoes": "recommend shoes that complete this outfit",
    "accessories": "recommend accessories that complete this outfit",
    "tops": "recommend a top that works with this outfit",
    "bottoms": "recommend bottoms that work with this outfit",
    "outerwear": "recommend outerwear that works with this outfit",
}


def build_evaluation_cases(
    all_items: pd.DataFrame,
    target_items: pd.DataFrame,
    target_categories: list[str],
    max_cases_per_target: int,
    seed: int,
) -> pd.DataFrame:
    query_items = all_items[
        all_items["query_category"].isin([*target_categories, "dresses"])
    ].copy()
    target_counts = target_items.groupby("outfit_ID").size()
    query_counts = query_items.groupby("outfit_ID").size()
    usable_outfits = set(
        target_counts[(target_counts >= 2) & (target_counts.index.isin(query_counts.index))].index
    )
    query_items = (
        query_items[query_items["outfit_ID"].isin(usable_outfits)]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    positive_lookup = {
        (str(outfit), category): group["item_ID"].astype(str).tolist()
        for (outfit, category), group in target_items.groupby(["outfit_ID", "broad_category"])
    }
    buckets: dict[str, list[dict[str, object]]] = {category: [] for category in target_categories}
    for row in query_items.to_dict("records"):
        for target_category in target_categories:
            if len(buckets[target_category]) >= max_cases_per_target:
                continue
            positives = [
                item_id
                for item_id in positive_lookup.get((str(row["outfit_ID"]), target_category), [])
                if item_id != str(row["item_ID"])
            ]
            if positives:
                buckets[target_category].append(
                    {
                        "query_item_id": str(row["item_ID"]),
                        "query_category": row["category"],
                        "query_group": row["query_category"],
                        "query_text": row["text"],
                        "query_outfit_id": str(row["outfit_ID"]),
                        "target_category": target_category,
                        "user_request": REQUEST_TEMPLATES[target_category],
                        "positive_item_ids": positives,
                        "num_positives": len(positives),
                    }
                )
        if all(len(values) >= max_cases_per_target for values in buckets.values()):
            break
    return pd.DataFrame([row for category in target_categories for row in buckets[category]])


def build_minilm_query_text(row: pd.Series) -> str:
    return " | ".join(
        [
            f"Query item category: {row['query_category']}",
            f"Query item type: {row['query_group']}",
            f"Query item description: {row['query_text']}",
            f"User request: {row['user_request']}",
            f"Recommended category: {row['target_category']}",
        ]
    )


def build_clip_query_text(row: pd.Series) -> str:
    return (
        f"Fashion item: {row['query_category']} | "
        f"Description: {row['query_text']} | "
        f"Input group: {row['query_group']} | "
        f"Request: {row['user_request']} | "
        f"Recommend: {row['target_category']}"
    )


@dataclass
class QueryEmbeddings:
    minilm: np.ndarray
    clip_fused: np.ndarray


def encode_evaluation_queries(
    cases: pd.DataFrame,
    all_items: pd.DataFrame,
    dataset_split,
    text_model: SentenceTransformerEmbedder,
    clip_model: CLIPEmbedder,
) -> QueryEmbeddings:
    minilm = text_model.encode([build_minilm_query_text(row) for _, row in cases.iterrows()])
    clip_text = clip_model.encode_text([build_clip_query_text(row) for _, row in cases.iterrows()])
    images = [
        dataset_split[
            int(all_items.loc[all_items["item_ID"] == item_id, "original_dataset_index"].iloc[0])
        ]["image"].convert("RGB")
        for item_id in cases["query_item_id"]
    ]
    clip_image = clip_model.encode_images(images)
    return QueryEmbeddings(minilm=minilm, clip_fused=clip_model.fuse(clip_image, clip_text))


def evaluate_controlled(
    config: AppConfig,
    cases: pd.DataFrame,
    target_items: pd.DataFrame,
    target_text_embeddings: np.ndarray,
    target_clip_embeddings: np.ndarray,
    query_embeddings: QueryEmbeddings,
    evidence_scorer: CandidateEvidenceScorer | None = None,
) -> pd.DataFrame:
    id_to_row = {
        item_id: index for index, item_id in enumerate(target_items["item_ID"].astype(str).tolist())
    }
    rows = []
    for case_index, case in cases.head(config.evaluation.controlled_cases).iterrows():
        candidates = build_controlled_candidate_set(
            target_items,
            str(case["query_outfit_id"]),
            str(case["target_category"]),
            config.evaluation.negatives_per_case,
            np.random.default_rng(config.project.seed + int(case_index)),
            query_item_id=str(case["query_item_id"]),
        )
        candidate_rows = [id_to_row[item_id] for item_id in candidates["item_ID"].astype(str)]
        model_scores = {
            "text_baseline": np.asarray(target_text_embeddings[candidate_rows])
            @ query_embeddings.minilm[case_index],
            "clip_multimodal": np.asarray(target_clip_embeddings[candidate_rows])
            @ query_embeddings.clip_fused[case_index],
        }
        if evidence_scorer is not None:
            evidence_scores = evidence_scorer.score(case, candidates)
            rerank_input = candidates.copy()
            rerank_input["clip_score"] = model_scores["clip_multimodal"]
            rerank_input["evidence_score"] = evidence_scores
            reranked = weighted_rerank(
                rerank_input,
                config.reranking.clip_weight,
                config.reranking.evidence_weight,
                config.reranking.normalize_scores,
            )
            item_to_final_score = dict(
                zip(reranked["item_ID"].astype(str), reranked["final_score"], strict=True)
            )
            model_scores["evidence_reranked"] = (
                candidates["item_ID"].astype(str).map(item_to_final_score).to_numpy()
            )
        for model_name, scores in model_scores.items():
            order = np.argsort(-scores)
            ranked = candidates.iloc[order].reset_index(drop=True)
            relevance = ranked["is_positive"].astype(int).tolist()
            positive_ranks = np.flatnonzero(np.asarray(relevance)) + 1
            rows.append(
                {
                    "case_index": int(case_index),
                    "query_item_id": case["query_item_id"],
                    "target_category": case["target_category"],
                    "num_candidates": len(ranked),
                    "num_positives": int(ranked["is_positive"].sum()),
                    "model_name": model_name,
                    "top_positive_rank": (int(positive_ranks[0]) if len(positive_ranks) else None),
                    **ranking_metrics(relevance, config.evaluation.cutoffs),
                }
            )
    return pd.DataFrame(rows)
