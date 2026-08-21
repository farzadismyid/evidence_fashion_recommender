"""Frozen Ollama/CLIP encoders and deterministic controlled-pool retrieval interfaces."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim not in {1, 2}:
        raise ValueError("Embeddings must be a vector or matrix.")
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("Embeddings must have finite, non-zero norms.")
    return array / norms


def fuse_clip_embeddings(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    *,
    image_weight: float,
    text_weight: float,
) -> np.ndarray:
    image = l2_normalize(image_embeddings)
    text = l2_normalize(text_embeddings)
    if image.shape != text.shape:
        raise ValueError("CLIP image and text embeddings must have identical shapes.")
    if image_weight < 0 or text_weight < 0 or image_weight + text_weight <= 0:
        raise ValueError("Fusion weights must be non-negative and have a positive sum.")
    return l2_normalize(image_weight * image + text_weight * text)


def cosine_scores(query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    query = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
    candidates = l2_normalize(candidate_embeddings)
    if candidates.shape[1] != query.shape[0]:
        raise ValueError("Query and candidate embedding dimensions differ.")
    return candidates @ query


def rank_candidates(
    query_embedding: np.ndarray,
    items: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    target_category: str,
    top_k: int,
    category_column: str = "broad_category",
    item_id_column: str = "item_id",
) -> pd.DataFrame:
    if len(items) != len(embeddings):
        raise ValueError("Items and embeddings must be row-aligned.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    mask = items[category_column].astype(str).eq(target_category).to_numpy()
    if not mask.any():
        raise KeyError(f"No candidates exist for target category {target_category!r}.")
    selected = items.loc[mask].copy()
    selected["compatibility_score"] = cosine_scores(query_embedding, embeddings[mask])
    return (
        selected.sort_values(
            ["compatibility_score", item_id_column],
            ascending=[False, True],
            kind="stable",
        )
        .head(top_k)
        .reset_index(drop=True)
    )


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class MiniLMEmbedder:
    settings: dict[str, Any]

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = _device(self.settings["device"])
        self.model = SentenceTransformer(
            self.settings["model_id"],
            revision=self.settings["revision"],
            device=self.device,
            local_files_only=True,
        )
        self.model.max_seq_length = self.settings["max_sequence_length"]

    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size or self.settings["batch_size"],
            convert_to_numpy=True,
            normalize_embeddings=self.settings["normalize"],
            show_progress_bar=False,
        )
        return l2_normalize(vectors) if self.settings["normalize"] else vectors.astype(np.float32)


@dataclass
class OllamaEmbedder:
    """Deterministic batched text embeddings from a locally pinned Ollama model."""

    settings: dict[str, Any]
    endpoint: str = "http://127.0.0.1:11434"

    def __post_init__(self) -> None:
        if self.settings.get("provider") != "ollama":
            raise ValueError("OllamaEmbedder requires an Ollama embedding-model configuration.")
        self.device = str(self.settings.get("device", "local_ollama"))

    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        size = batch_size or int(self.settings["batch_size"])
        batches = []
        for start in range(0, len(texts), size):
            chunk = list(texts[start : start + size])
            payload = json.dumps(
                {
                    "model": self.settings["model_id"],
                    "input": chunk,
                    "truncate": True,
                    "keep_alive": "10m",
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{self.endpoint.rstrip('/')}/api/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            vectors = np.asarray(result.get("embeddings", []), dtype=np.float32)
            if vectors.shape != (len(chunk), int(self.settings["dimension"])):
                raise ValueError(
                    "Ollama embedding response has an unexpected batch size or dimension."
                )
            batches.append(vectors)
        values = np.concatenate(batches).astype(np.float32)
        return l2_normalize(values) if self.settings["normalize"] else values


@dataclass
class CLIPEmbedder:
    settings: dict[str, Any]

    def __post_init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = _device(self.settings["device"])
        local_files_only = self.settings.get("local_files_only", True)
        self.processor = CLIPProcessor.from_pretrained(
            self.settings["model_id"],
            revision=self.settings["revision"],
            local_files_only=local_files_only,
        )
        self.model = CLIPModel.from_pretrained(
            self.settings["model_id"],
            revision=self.settings["revision"],
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()
        self.torch = torch

    @staticmethod
    def _feature_tensor(output):
        """Accept tensor-returning Transformers 4.x and structured Transformers 5.x APIs."""

        return output.pooler_output if hasattr(output, "pooler_output") else output

    def encode_text(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        size = batch_size or self.settings["batch_size"]
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), size):
                inputs = self.processor(
                    text=list(texts[start : start + size]),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.settings["max_sequence_length"],
                ).to(self.device)
                output = self._feature_tensor(self.model.get_text_features(**inputs))
                batches.append(output.cpu().numpy())
        vectors = np.concatenate(batches).astype(np.float32)
        return l2_normalize(vectors) if self.settings["normalize"] else vectors

    def encode_images(
        self, images: Sequence[Image.Image], batch_size: int | None = None
    ) -> np.ndarray:
        size = batch_size or self.settings["batch_size"]
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(images), size):
                inputs = self.processor(
                    images=[image.convert("RGB") for image in images[start : start + size]],
                    return_tensors="pt",
                ).to(self.device)
                output = self._feature_tensor(self.model.get_image_features(**inputs))
                batches.append(output.cpu().numpy())
        vectors = np.concatenate(batches).astype(np.float32)
        return l2_normalize(vectors) if self.settings["normalize"] else vectors
