"""Vectorized candidate-specific evidence scoring for controlled evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evidence import detect_item_types
from ..models.base import TextEmbedder

SOURCE_RELIABILITY_WEIGHTS = {"high": 1.0, "medium": 0.9, "low": 0.75}


def build_candidate_evidence_text(
    case: pd.Series,
    candidate: pd.Series,
) -> str:
    return " | ".join(
        [
            f"Query item category: {case['query_category']}",
            f"Query item group: {case['query_group']}",
            f"Query item description: {case['query_text']}",
            f"Candidate category: {candidate['category']}",
            f"Candidate group: {candidate['broad_category']}",
            f"Candidate description: {candidate['text']}",
            f"User request: {case['user_request']}",
            f"Need fashion evidence for recommending: {case['target_category']}",
        ]
    )


class CandidateEvidenceScorer:
    def __init__(
        self,
        knowledge_base: pd.DataFrame,
        kb_embeddings: np.ndarray,
        embedder: TextEmbedder,
        top_k: int,
        candidate_type_filtering: bool = False,
    ) -> None:
        self.kb = knowledge_base.reset_index(drop=True)
        self.embeddings = np.asarray(kb_embeddings)
        self.embedder = embedder
        self.top_k = top_k
        self.candidate_type_filtering = candidate_type_filtering
        self.category_rows = {
            str(category): group.index.to_numpy()
            for category, group in self.kb.groupby("recommended_category")
        }

    def score(self, case: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
        query_embeddings = self.embedder.encode(
            [
                build_candidate_evidence_text(case, candidate)
                for _, candidate in candidates.iterrows()
            ]
        )
        kb_rows = self.category_rows[str(case["target_category"])]
        similarities = query_embeddings @ self.embeddings[kb_rows].T
        top_count = min(self.top_k, len(kb_rows))
        top_local = np.argpartition(-similarities, top_count - 1, axis=1)[:, :top_count]
        scores = np.empty(len(candidates), dtype=np.float32)
        query_group = str(case["query_group"]).strip().lower()
        for index in range(len(candidates)):
            selected_rows = kb_rows[top_local[index]]
            selected_scores = similarities[index, top_local[index]]
            rules = self.kb.iloc[selected_rows]
            source_weights = (
                rules["source_reliability"].map(SOURCE_RELIABILITY_WEIGHTS).fillna(0.8).to_numpy()
            )
            input_bonuses = (
                rules["input_category"]
                .astype(str)
                .apply(
                    lambda value: (
                        0.05
                        if query_group in [part.strip().lower() for part in value.split(",")]
                        else 0.0
                    )
                )
            )
            weighted = selected_scores * source_weights + input_bonuses.to_numpy()
            scores[index] = 0.7 * weighted.max() + 0.3 * weighted.mean()
        return scores

    def retrieve(self, case: pd.Series, candidate: pd.Series) -> pd.DataFrame:
        query = self.embedder.encode([build_candidate_evidence_text(case, candidate)])
        kb_rows = self.category_rows[str(case["target_category"])]
        similarities = query[0] @ self.embeddings[kb_rows].T
        order = np.argsort(-similarities)
        if self.candidate_type_filtering and str(case["target_category"]) == "accessories":
            candidate_types = detect_item_types(
                f"{candidate.get('category', '')} {candidate.get('text', '')}"
            )
            compatible = []
            for local_index in order:
                rule = self.kb.iloc[kb_rows[local_index]]
                rule_types = detect_item_types(
                    f"{rule.get('rule_text', '')} {rule.get('evidence_keywords', '')}"
                )
                if not rule_types or not candidate_types or rule_types & candidate_types:
                    compatible.append(local_index)
                if len(compatible) >= self.top_k:
                    break
            if compatible:
                order = np.asarray(compatible)
        order = order[: self.top_k]
        result = self.kb.iloc[kb_rows[order]].copy()
        result["evidence_score"] = similarities[order]
        return result.reset_index(drop=True)
