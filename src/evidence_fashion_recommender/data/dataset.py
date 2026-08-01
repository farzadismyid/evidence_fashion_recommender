"""Polyvore-style dataset adapter and cacheable metadata preparation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..cache import ArtifactCache
from ..config import AppConfig
from .categories import map_broad_category, map_query_category

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedDataset:
    items: pd.DataFrame
    metadata_cache_path: Path
    cache_hit: bool


def _find_cached_arrow_shards(config: AppConfig) -> list[Path]:
    from datasets.config import HF_DATASETS_CACHE

    cache_root = Path(os.getenv("HF_DATASETS_CACHE", HF_DATASETS_CACHE))
    dataset_cache_name = config.dataset.name.replace("/", "___")
    dataset_root = cache_root / dataset_cache_name
    if not dataset_root.exists():
        return []
    candidates = sorted(dataset_root.glob("**/*.arrow"))
    if config.dataset.revision:
        revision_matches = [path for path in candidates if config.dataset.revision in path.parts]
        if revision_matches:
            candidates = revision_matches
    split_marker = f"-{config.dataset.split}-"
    split_matches = [path for path in candidates if split_marker in path.name]
    return split_matches or candidates


def load_huggingface_split(config: AppConfig):
    from datasets import Dataset, concatenate_datasets, load_dataset

    if config.dataset.prefer_local_cache:
        shards = _find_cached_arrow_shards(config)
        if shards:
            LOGGER.info("Loading %d memory-mapped Arrow shards from the local cache", len(shards))
            parts = [Dataset.from_file(str(path)) for path in shards]
            return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    dataset = load_dataset(
        config.dataset.name,
        revision=config.dataset.revision,
    )
    return dataset[config.dataset.split]


def _prepare_frame(config: AppConfig) -> pd.DataFrame:
    split = load_huggingface_split(config)
    metadata_split = split.remove_columns(config.dataset.image_column)
    frame = metadata_split.to_pandas()
    item_id = config.dataset.item_id_column
    category = config.dataset.category_column
    text = config.dataset.text_column
    separator = config.dataset.outfit_id_separator

    frame[item_id] = frame[item_id].astype(str)
    parts = frame[item_id].str.rsplit(separator, n=1, expand=True)
    frame["outfit_ID"] = parts[0]
    frame["item_position"] = pd.to_numeric(parts[1], errors="coerce")
    frame["item_text"] = frame[category].fillna("").astype(str) + " | " + frame[text].fillna("")
    frame["broad_category"] = frame[category].map(map_broad_category)
    frame["query_category"] = frame[category].map(map_query_category)
    frame["original_dataset_index"] = frame.index
    # Images remain in the Hugging Face dataset cache. Duplicating 94k encoded images in
    # the project metadata cache would be wasteful and makes the table less portable.
    return frame


def load_prepared_dataset(config: AppConfig, cache: ArtifactCache) -> PreparedDataset:
    inputs = {
        "dataset": config.dataset.model_dump(mode="json"),
        "mapping": config.preprocessing.broad_category_mapping,
        "schema_version": 2,
    }
    record = cache.location("datasets", inputs, ".parquet")
    if record.hit:
        LOGGER.info("Reusing prepared dataset %s", record.fingerprint[:12])
        return PreparedDataset(pd.read_parquet(record.path), record.path, True)

    frame = _prepare_frame(config)
    if cache.policy != "disabled":
        record.path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(record.path, index=False)
    return PreparedDataset(frame, record.path, False)


def target_items(frame: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    allowed = set(config.dataset.target_categories)
    return frame[frame["broad_category"].isin(allowed)].reset_index(drop=True)
