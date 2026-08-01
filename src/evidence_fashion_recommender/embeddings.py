"""Cache-aware embedding computation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .cache import ArtifactCache
from .models.base import TextEmbedder

LOGGER = logging.getLogger(__name__)


def cached_text_embeddings(
    texts: list[str],
    embedder: TextEmbedder,
    cache: ArtifactCache,
    namespace: str,
    source_fingerprint: str,
) -> tuple[np.ndarray, Path, bool]:
    inputs = {
        "model": embedder.model_id,
        "source": source_fingerprint,
        "rows": len(texts),
        "schema_version": 1,
    }
    record = cache.location(namespace, inputs, ".npy")
    if record.hit:
        LOGGER.info("Reusing embeddings %s", record.fingerprint[:12])
        return np.load(record.path), record.path, True
    embeddings = embedder.encode(texts)
    if cache.policy != "disabled":
        record.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(record.path, embeddings)
    return embeddings, record.path, False
