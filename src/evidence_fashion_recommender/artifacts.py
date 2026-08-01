"""Named, fingerprinted artifacts used across experiment stages."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import ArtifactCache, stable_fingerprint
from .config import AppConfig


def item_table_fingerprint(items: pd.DataFrame) -> str:
    columns = ["item_ID", "outfit_ID", "broad_category", "item_text"]
    payload = items[columns].fillna("").astype(str).to_dict("records")
    return stable_fingerprint(payload)


def embedding_inputs(config: AppConfig, items: pd.DataFrame, modality: str) -> dict[str, object]:
    model = (
        config.models.text_embedding
        if modality == "minilm_text"
        else config.models.multimodal_embedding
    )
    return {
        "artifact": "target_embeddings",
        "modality": modality,
        "model": model.model_dump(mode="json"),
        "items": item_table_fingerprint(items),
        "rows": len(items),
        "schema_version": 1,
    }


def embedding_record(
    cache: ArtifactCache,
    config: AppConfig,
    items: pd.DataFrame,
    modality: str,
):
    return cache.location("embeddings", embedding_inputs(config, items, modality), ".npy")


@dataclass(frozen=True)
class LegacyImport:
    modality: str
    source: Path
    destination: Path
    shape: tuple[int, ...]


def import_legacy_embeddings(
    config: AppConfig,
    cache: ArtifactCache,
    items: pd.DataFrame,
    legacy_items_path: Path,
    legacy_embedding_dir: Path,
) -> list[LegacyImport]:
    legacy_items = pd.read_parquet(legacy_items_path, columns=["item_ID"])
    expected_ids = items["item_ID"].astype(str).reset_index(drop=True)
    actual_ids = legacy_items["item_ID"].astype(str).reset_index(drop=True)
    if not expected_ids.equals(actual_ids):
        raise ValueError("Legacy embeddings are not aligned with the modular target item table.")

    filenames = {
        "minilm_text": "target_text_embeddings_all_minilm_l6_v2.npy",
        "clip_image": "target_clip_image_embeddings.npy",
        "clip_text": "target_clip_text_embeddings.npy",
        "clip_fused": "target_clip_fused_embeddings.npy",
    }
    imports = []
    for modality, filename in filenames.items():
        source = legacy_embedding_dir / filename
        array = np.load(source, mmap_mode="r")
        if len(array) != len(items):
            raise ValueError(f"{filename} has {len(array)} rows; expected {len(items)}.")
        record = embedding_record(cache, config, items, modality)
        record.path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, record.path)
        metadata = {
            "fingerprint": record.fingerprint,
            "inputs": embedding_inputs(config, items, modality),
            "migration_source": str(source),
            "verified_item_alignment": True,
            "shape": list(array.shape),
        }
        record.path.with_suffix(".npy.meta.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        imports.append(LegacyImport(modality, source, record.path, tuple(array.shape)))
    return imports


def load_embedding_set(
    config: AppConfig,
    cache: ArtifactCache,
    items: pd.DataFrame,
) -> dict[str, np.ndarray]:
    result = {}
    for modality in ["minilm_text", "clip_image", "clip_text", "clip_fused"]:
        record = embedding_record(cache, config, items, modality)
        if not record.path.exists():
            raise FileNotFoundError(
                f"Missing {modality} embeddings. Run build-embeddings or import-legacy first."
            )
        result[modality] = np.load(record.path, mmap_mode="r")
    return result


def build_target_embeddings(
    config: AppConfig,
    cache: ArtifactCache,
    items: pd.DataFrame,
    dataset_split,
) -> dict[str, Path]:
    from .models.multimodal import CLIPEmbedder
    from .models.text import SentenceTransformerEmbedder

    outputs: dict[str, Path] = {}
    text_record = embedding_record(cache, config, items, "minilm_text")
    clip_text_record = embedding_record(cache, config, items, "clip_text")
    clip_image_record = embedding_record(cache, config, items, "clip_image")
    clip_fused_record = embedding_record(cache, config, items, "clip_fused")

    text_model = None
    if cache.policy == "refresh" or not text_record.path.exists():
        text_model = SentenceTransformerEmbedder(
            config.models.text_embedding, config.project.device
        )
        text_record.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(text_record.path, text_model.encode(items["item_text"].astype(str).tolist()))
    outputs["minilm_text"] = text_record.path

    clip_model = None
    if (
        cache.policy == "refresh"
        or not clip_text_record.path.exists()
        or not clip_image_record.path.exists()
    ):
        clip_model = CLIPEmbedder(config.models.multimodal_embedding, config.project.device)

    if cache.policy == "refresh" or not clip_text_record.path.exists():
        clip_text_record.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            clip_text_record.path,
            clip_model.encode_text(items["item_text"].astype(str).tolist()),
        )
    outputs["clip_text"] = clip_text_record.path

    if cache.policy == "refresh" or not clip_image_record.path.exists():
        clip_image_record.path.parent.mkdir(parents=True, exist_ok=True)
        partial = clip_image_record.path.with_suffix(".npy.partial")
        batch_size = config.models.multimodal_embedding.batch_size
        matrix = None
        for start in range(0, len(items), batch_size):
            batch = items.iloc[start : start + batch_size]
            images = [
                dataset_split[int(index)]["image"].convert("RGB")
                for index in batch["original_dataset_index"]
            ]
            encoded = clip_model.encode_images(images)
            if matrix is None:
                matrix = np.lib.format.open_memmap(
                    partial,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(items), encoded.shape[1]),
                )
            matrix[start : start + len(encoded)] = encoded
        if matrix is not None:
            matrix.flush()
            del matrix
        partial.replace(clip_image_record.path)
    outputs["clip_image"] = clip_image_record.path

    if cache.policy == "refresh" or not clip_fused_record.path.exists():
        if clip_model is None:
            clip_model = CLIPEmbedder(config.models.multimodal_embedding, config.project.device)
        image_embeddings = np.load(clip_image_record.path, mmap_mode="r")
        text_embeddings = np.load(clip_text_record.path, mmap_mode="r")
        clip_fused_record.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            clip_fused_record.path,
            clip_model.fuse(
                np.asarray(image_embeddings),
                np.asarray(text_embeddings),
            ),
        )
    outputs["clip_fused"] = clip_fused_record.path
    return outputs
