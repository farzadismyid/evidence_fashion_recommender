"""Hugging Face CLIP adapter for image, text, and fused embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image

from ..config import MultimodalModelConfig


def fuse_embeddings(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    image_weight: float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Fuse aligned CLIP embeddings at an explicit validation-selected weight."""

    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("CLIP image and text embeddings must have the same shape.")
    if image_weight < 0 or image_weight > 1:
        raise ValueError("image_weight must be between 0 and 1.")
    fused = image_weight * image_embeddings + (1.0 - image_weight) * text_embeddings
    if not normalize:
        return fused.astype(np.float32)
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    return (fused / np.maximum(norms, 1e-12)).astype(np.float32)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class CLIPEmbedder:
    def __init__(self, config: MultimodalModelConfig, device: str = "auto") -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.config = config
        self.device = resolve_device(device)
        self.model_id = f"{config.name}@{config.revision or 'default'}"
        self.processor = CLIPProcessor.from_pretrained(
            config.name,
            revision=config.revision,
            local_files_only=config.local_files_only,
        )
        self.model = CLIPModel.from_pretrained(
            config.name,
            revision=config.revision,
            local_files_only=config.local_files_only,
        )
        self.model.to(self.device).eval()

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-12)

    @staticmethod
    def _feature_tensor(output):
        # transformers 4 returned a Tensor from get_*_features; transformers 5 returns
        # BaseModelOutputWithPooling. Supporting both keeps archived and fresh runs aligned.
        return output.pooler_output if hasattr(output, "pooler_output") else output

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        batches = []
        for start in range(0, len(texts), self.config.batch_size):
            inputs = self.processor(
                text=list(texts[start : start + self.config.batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.inference_mode():
                features = self.model.get_text_features(**inputs)
            batches.append(self._feature_tensor(features).float().cpu().numpy())
        result = np.concatenate(batches).astype(np.float32)
        return self._normalize(result) if self.config.normalize else result

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        batches = []
        for start in range(0, len(images), self.config.batch_size):
            inputs = self.processor(
                images=list(images[start : start + self.config.batch_size]),
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                features = self.model.get_image_features(**inputs)
            batches.append(self._feature_tensor(features).float().cpu().numpy())
        result = np.concatenate(batches).astype(np.float32)
        return self._normalize(result) if self.config.normalize else result

    def fuse(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        return fuse_embeddings(
            image_embeddings,
            text_embeddings,
            self.config.image_weight,
            normalize=self.config.normalize,
        )
