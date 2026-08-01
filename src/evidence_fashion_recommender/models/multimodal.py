"""Hugging Face CLIP adapter for image, text, and fused embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image

from ..config import MultimodalModelConfig


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
        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError("CLIP image and text embeddings must have the same shape.")
        fused = (
            self.config.image_weight * image_embeddings + self.config.text_weight * text_embeddings
        )
        return self._normalize(fused) if self.config.normalize else fused.astype(np.float32)
