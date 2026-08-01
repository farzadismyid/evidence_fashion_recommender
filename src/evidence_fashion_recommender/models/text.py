"""Sentence-transformers text embedding adapter."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import EmbeddingModelConfig


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingModelConfig, device: str = "auto") -> None:
        from sentence_transformers import SentenceTransformer

        resolved_device = None if device == "auto" else device
        self.config = config
        self.model_id = f"{config.name}@{config.revision or 'default'}"
        self.model = SentenceTransformer(
            config.name,
            revision=config.revision,
            device=resolved_device,
            local_files_only=config.local_files_only,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
