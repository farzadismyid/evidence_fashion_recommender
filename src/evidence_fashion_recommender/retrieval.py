"""Category-aware FAISS retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CategoryIndex:
    index: object
    row_indices: np.ndarray


class CategoryRetriever:
    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric
        self.indexes: dict[str, CategoryIndex] = {}

    def fit(
        self,
        items: pd.DataFrame,
        embeddings: np.ndarray,
        category_column: str = "broad_category",
    ) -> CategoryRetriever:
        import faiss

        if len(items) != len(embeddings):
            raise ValueError("Items and embeddings must be row-aligned.")
        for category, group in items.groupby(category_column):
            rows = group.index.to_numpy()
            vectors = np.ascontiguousarray(embeddings[rows].astype(np.float32))
            if self.metric == "cosine":
                faiss.normalize_L2(vectors)
                index = faiss.IndexFlatIP(vectors.shape[1])
            elif self.metric == "inner_product":
                index = faiss.IndexFlatIP(vectors.shape[1])
            else:
                index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(vectors)
            self.indexes[str(category)] = CategoryIndex(index, rows)
        return self

    def search(
        self,
        query_embedding: np.ndarray,
        target_category: str,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if target_category not in self.indexes:
            raise KeyError(f"No index for target category {target_category!r}")
        category_index = self.indexes[target_category]
        vector = np.ascontiguousarray(query_embedding.reshape(1, -1).astype(np.float32))
        if self.metric == "cosine":
            import faiss

            faiss.normalize_L2(vector)
        scores, local_rows = category_index.index.search(vector, top_k)
        valid = local_rows[0] >= 0
        rows = category_index.row_indices[local_rows[0][valid]]
        return rows, scores[0][valid]
