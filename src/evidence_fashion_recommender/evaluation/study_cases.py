"""Build the fixed explanation study through the modular recommendation pipeline."""

from __future__ import annotations

import pandas as pd

from ..config import AppConfig
from ..evaluation.evidence_ranking import CandidateEvidenceScorer
from ..models.multimodal import CLIPEmbedder
from ..pipeline import recommend


def build_modular_study_cases(
    config: AppConfig,
    schedule: pd.DataFrame,
    all_items: pd.DataFrame,
    targets: pd.DataFrame,
    dataset_split,
    target_clip_embeddings,
    clip_model: CLIPEmbedder,
    evidence_scorer: CandidateEvidenceScorer,
) -> pd.DataFrame:
    schedule_columns = [
        "case_index",
        "query_item_id",
        "query_category",
        "query_group",
        "query_text",
        "user_request",
        "target_category",
    ]
    schedule_columns.extend(
        column for column in ("query_outfit_id", "research_split") if column in schedule.columns
    )
    unique_schedule = schedule[schedule_columns].drop_duplicates().reset_index(drop=True)
    rows = []
    for case_number, case in unique_schedule.iterrows():
        result = recommend(
            config,
            all_items,
            targets,
            dataset_split,
            target_clip_embeddings,
            clip_model,
            evidence_scorer,
            str(case["query_item_id"]),
            str(case["target_category"]),
            str(case["user_request"]),
        )
        for recommendation_index, candidate in result.recommendations.head(
            config.evaluation.recommendations_per_case
        ).iterrows():
            item_id = str(candidate["item_ID"])
            evidence = result.evidence[item_id]
            rule_ids = ", ".join(evidence["rule_id"].astype(str))
            rule_text = "\n".join(
                f"{row['rule_id']}: {row['rule_text']}" for _, row in evidence.iterrows()
            )
            similar_items = result.candidates[
                result.candidates["item_ID"].astype(str) != item_id
            ].head(config.evidence.candidate_top_k)
            item_text = "\n".join(
                f"ITEM-{index + 1}: {row['category']} - {row['text']}"
                for index, (_, row) in enumerate(similar_items.iterrows())
            )
            rows.append(
                {
                    "paper_case_id": f"MOD_C{case_number:03d}_R{recommendation_index + 1}",
                    "kb_version": config.evidence.kb_version,
                    "evidence_filtering": (
                        "candidate_type_filtered"
                        if config.evidence.candidate_type_filtering
                        else "unfiltered"
                    ),
                    **case.to_dict(),
                    "recommendation_rank": recommendation_index + 1,
                    "recommended_item_id": item_id,
                    "recommended_category": candidate["category"],
                    "recommended_group": candidate["broad_category"],
                    "recommended_text": candidate["text"],
                    "clip_multimodal_score": candidate["clip_score"],
                    "evidence_score": candidate["evidence_score"],
                    "final_score": candidate["final_score"],
                    "rule_evidence_ids": rule_ids,
                    "rule_evidence_text": rule_text,
                    "item_evidence_text": item_text,
                    "hybrid_evidence_text": (
                        f"Expert styling rules:\n{rule_text}\n\n"
                        f"Similar item descriptions:\n{item_text}"
                    ),
                }
            )
    return pd.DataFrame(rows)
