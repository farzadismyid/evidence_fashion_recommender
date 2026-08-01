"""Knowledge-base loading, evidence text construction, and retrieval."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .retrieval import CategoryRetriever


def load_knowledge_base(path: Path) -> pd.DataFrame:
    kb = pd.read_csv(path)
    required = {"rule_id", "rule_text"}
    missing = required - set(kb.columns)
    if missing:
        raise ValueError(f"Knowledge base is missing required columns: {sorted(missing)}")
    return kb


def build_evidence_text(kb: pd.DataFrame) -> list[str]:
    preferred = [
        "rule_text",
        "input_category",
        "recommended_category",
        "occasion",
        "style",
        "source_title",
    ]
    columns = [column for column in preferred if column in kb.columns]
    return (
        kb[columns]
        .fillna("")
        .astype(str)
        .apply(lambda row: " | ".join(value for value in row if value), axis=1)
        .tolist()
    )


def detect_item_types(text: str) -> set[str]:
    vocabulary = {
        "boots",
        "shoes",
        "sandals",
        "sneakers",
        "heels",
        "flats",
        "bag",
        "handbag",
        "clutch",
        "belt",
        "scarf",
        "sunglasses",
        "earrings",
        "necklace",
        "bracelet",
        "top",
        "shirt",
        "blouse",
        "sweater",
        "jeans",
        "trousers",
        "pants",
        "skirt",
        "coat",
        "jacket",
        "blazer",
    }
    normalized = re.sub(r"[^a-z]+", " ", str(text).lower())
    return {term for term in vocabulary if re.search(rf"\b{re.escape(term)}\b", normalized)}


class EvidenceRetriever:
    def __init__(self, kb: pd.DataFrame, embeddings: np.ndarray, metric: str = "cosine") -> None:
        self.kb = kb.reset_index(drop=True)
        working = self.kb.copy()
        if "recommended_category" not in working:
            working["recommended_category"] = "all"
        working["broad_category"] = working["recommended_category"].fillna("all").astype(str)
        self.retriever = CategoryRetriever(metric).fit(working, embeddings)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        target_category: str,
        top_k: int,
        candidate_text: str | None = None,
        type_filtering: bool = False,
    ) -> pd.DataFrame:
        category = target_category if target_category in self.retriever.indexes else "all"
        rows, scores = self.retriever.search(query_embedding, category, max(top_k * 3, top_k))
        result = self.kb.iloc[rows].copy()
        result["evidence_score"] = scores
        if type_filtering and candidate_text:
            types = detect_item_types(candidate_text)
            if types:
                mask = (
                    result["rule_text"]
                    .astype(str)
                    .str.lower()
                    .apply(
                        lambda value: (
                            not detect_item_types(value) or bool(detect_item_types(value) & types)
                        )
                    )
                )
                filtered = result[mask]
                if not filtered.empty:
                    result = filtered
        return result.head(top_k).reset_index(drop=True)
