"""Persistent category-aware FAISS indexes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import ArtifactCache


@dataclass
class PersistentCategoryIndexes:
    indexes: dict[str, object]
    rows: dict[str, np.ndarray]

    def search(self, category: str, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        import faiss

        vector = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))
        faiss.normalize_L2(vector)
        scores, local = self.indexes[category].search(vector, top_k)
        valid = local[0] >= 0
        return self.rows[category][local[0][valid]], scores[0][valid]


def build_or_load_category_indexes(
    cache: ArtifactCache,
    items: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_fingerprint: str,
) -> tuple[PersistentCategoryIndexes, Path, bool]:
    import faiss

    inputs = {
        "embedding_fingerprint": embedding_fingerprint,
        "categories": items["broad_category"].astype(str).tolist(),
        "rows": len(items),
        "metric": "cosine",
        "schema_version": 1,
    }
    record = cache.location("faiss_indexes", inputs, "")
    manifest_path = record.path / "manifest.json"
    if record.hit and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        indexes, rows = {}, {}
        for category in manifest["categories"]:
            indexes[category] = faiss.read_index(str(record.path / f"{category}.faiss"))
            rows[category] = np.load(record.path / f"{category}.rows.npy")
        return PersistentCategoryIndexes(indexes, rows), record.path, True

    record.path.mkdir(parents=True, exist_ok=True)
    indexes, rows = {}, {}
    for category, group in items.groupby("broad_category"):
        category = str(category)
        row_indices = group.index.to_numpy(dtype=np.int64)
        vectors = np.ascontiguousarray(np.asarray(embeddings[row_indices]).astype(np.float32))
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(record.path / f"{category}.faiss"))
        np.save(record.path / f"{category}.rows.npy", row_indices)
        indexes[category], rows[category] = index, row_indices
    manifest_path.write_text(
        json.dumps({"categories": sorted(indexes)}, indent=2), encoding="utf-8"
    )
    return PersistentCategoryIndexes(indexes, rows), record.path, False
