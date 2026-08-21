"""Build hash-bound embedding caches; Stage 3 runs only the validation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from evidence_fashion.data import load_pinned_split, write_jsonl
from evidence_fashion.manifest import (
    configuration_hash,
    environment_summary,
    git_commit,
    load_resolved_configuration,
    sha256_file,
    utc_timestamp,
    write_json,
    write_new_json,
)
from evidence_fashion.retrieval import (
    CLIPEmbedder,
    OllamaEmbedder,
    fuse_clip_embeddings,
    rank_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _hash_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_validation_rows(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    validation = config["embedding_validation"]
    categories = config["preprocessing"]["target_categories"]
    eligible = frame[
        (frame["research_split"] == validation["split"]) & frame["broad_category"].isin(categories)
    ].copy()
    eligible["_order"] = eligible["item_id"].map(
        lambda value: _hash_order(str(value), validation["seed"])
    )
    eligible = eligible.sort_values(["_order", "item_id"], kind="stable")
    selected = []
    if validation["require_all_target_categories"]:
        for category in categories:
            selected.append(eligible[eligible["broad_category"] == category].iloc[[0]])
    initial = pd.concat(selected) if selected else eligible.iloc[0:0]
    remaining = eligible[~eligible["item_id"].isin(initial["item_id"])]
    result = pd.concat([initial, remaining.head(validation["sample_size"] - len(initial))])
    if len(result) != validation["sample_size"]:
        raise ValueError("Insufficient validation rows for the configured embedding sample.")
    return result.drop(columns="_order").reset_index(drop=True)


def validate_embeddings(
    arrays: dict[str, np.ndarray], sample: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    tolerance = config["embedding_validation"]["norm_absolute_tolerance"]
    minimum_distance = config["embedding_validation"]["minimum_pairwise_distance"]
    dimensions = {}
    maximum_norm_errors = {}
    minimum_distances = {}
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name} embeddings contain non-finite values.")
        dimensions[name] = int(array.shape[1])
        maximum_norm_errors[name] = float(np.max(np.abs(np.linalg.norm(array, axis=1) - 1)))
        if maximum_norm_errors[name] > tolerance:
            raise ValueError(f"{name} embeddings are not normalized within tolerance.")
        distances = np.linalg.norm(array[:, None, :] - array[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        minimum_distances[name] = float(np.min(distances))
        if minimum_distances[name] <= minimum_distance:
            raise ValueError(f"{name} contains duplicate or collapsed validation embeddings.")
    first = rank_candidates(
        arrays["clip_fused"][0],
        sample,
        arrays["clip_fused"],
        target_category=str(sample.iloc[0]["broad_category"]),
        top_k=len(sample),
    )
    second = rank_candidates(
        arrays["clip_fused"][0],
        sample,
        arrays["clip_fused"],
        target_category=str(sample.iloc[0]["broad_category"]),
        top_k=len(sample),
    )
    if first["item_id"].tolist() != second["item_id"].tolist():
        raise ValueError("Deterministic ranking validation failed.")
    return {
        "dimensions": dimensions,
        "maximum_norm_errors": maximum_norm_errors,
        "minimum_pairwise_distances": minimum_distances,
        "deterministic_ranking": True,
        "category_filtering": bool(
            first["broad_category"].eq(sample.iloc[0]["broad_category"]).all()
        ),
    }


def encode_split_images(
    clip: CLIPEmbedder,
    raw_split: Any,
    original_indices: list[int],
    image_column: str,
    *,
    batch_size: int,
) -> np.ndarray:
    """Embed images in bounded batches without materialising the full image corpus."""
    batches = []
    for start in range(0, len(original_indices), batch_size):
        indices = original_indices[start : start + batch_size]
        images = [raw_split[index][image_column] for index in indices]
        batches.append(clip.encode_images(images, batch_size=batch_size))
    return np.concatenate(batches).astype(np.float32)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    config_digest = configuration_hash(resolved)
    data_manifest_path = Path(config["paths"]["active_data_manifest"])
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    items_path = Path(
        next(
            path
            for path in data_manifest["output_artifact_hashes"]
            if path.endswith("prepared_items.parquet")
        )
    )
    input_hash = sha256_file(items_path)
    if input_hash != data_manifest["output_artifact_hashes"][str(items_path)]:
        raise ValueError("Prepared metadata hash does not match the Stage 2 manifest.")
    runtime_root = args.runtime_root or Path(config["paths"]["runtime_root"])
    scope = "validation" if args.validate_only else "full"
    run_id = f"embedding-{scope}-{config_digest[:12]}"
    run_dir = runtime_root / "embeddings" / run_id
    runtime_manifest_path = run_dir / "manifest.json"
    tracked_manifest_path = Path(config["paths"]["active_embedding_manifest"])
    if args.dry_run:
        print(json.dumps({"run_id": run_id, "would_call_models": True, "scope": scope}, indent=2))
        return
    if args.resume and runtime_manifest_path.exists():
        print(runtime_manifest_path.read_text(encoding="utf-8"))
        return
    if run_dir.exists():
        raise FileExistsError(f"Immutable run directory already exists: {run_dir}")

    frame = pd.read_parquet(items_path)
    validation_sample = select_validation_rows(frame, config)
    sample = validation_sample if args.validate_only else frame
    raw_split, dataset_fingerprint = load_pinned_split(config)
    image_column = config["dataset"]["columns"]["image"]
    texts = (sample["category"].astype(str) + " | " + sample["text"].astype(str)).tolist()
    text_embedder = OllamaEmbedder(models["embedders"]["qwen3_embedding"])
    clip = CLIPEmbedder(models["embedders"]["clip"])
    batch_size = config["embedding_validation"]["batch_size"] if args.validate_only else None
    qwen3_embedding_text = text_embedder.encode(texts, batch_size=batch_size)
    clip_text = clip.encode_text(texts, batch_size=batch_size)
    image_batch_size = batch_size or int(models["embedders"]["clip"]["batch_size"])
    clip_image = encode_split_images(
        clip,
        raw_split,
        [int(index) for index in sample["original_dataset_index"]],
        image_column,
        batch_size=image_batch_size,
    )
    fusion = config["retrieval"]["fusion"]
    clip_fused = fuse_clip_embeddings(
        clip_image,
        clip_text,
        image_weight=fusion["image_weight"],
        text_weight=fusion["text_weight"],
    )
    arrays = {
        "qwen3_embedding_text": qwen3_embedding_text,
        "clip_text": clip_text,
        "clip_image": clip_image,
        "clip_fused": clip_fused,
    }
    validation_positions = {
        str(item_id): index for index, item_id in enumerate(sample["item_id"].astype(str))
    }
    validation_arrays = {
        name: values[
            [validation_positions[str(item_id)] for item_id in validation_sample["item_id"]]
        ]
        for name, values in arrays.items()
    }
    validation = validate_embeddings(validation_arrays, validation_sample, config)
    run_dir.mkdir(parents=True)
    output_hashes = {}
    for name, array in arrays.items():
        path = run_dir / f"{name}.npy"
        with path.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
        output_hashes[str(path)] = sha256_file(path)
    metadata_path = run_dir / "sample_metadata.jsonl"
    write_jsonl(metadata_path, sample.to_dict("records"))
    output_hashes[str(metadata_path)] = sha256_file(metadata_path)
    manifest = {
        "schema_version": 1,
        "stage": 3,
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": config_digest,
        "resolved_configuration": resolved,
        "input_artifact_hashes": {
            str(items_path): input_hash,
            "dataset_fingerprint": dataset_fingerprint,
        },
        "output_artifact_hashes": output_hashes,
        "models": {
            name: {
                "model_id": settings["model_id"],
                "revision": settings["revision"],
                "immutable_digest": settings["immutable_digest"],
                "device": text_embedder.device if name == "qwen3_embedding" else clip.device,
                "precision": settings["precision"],
            }
            for name, settings in models["embedders"].items()
        },
        "row_counts": {"embedded_items": len(sample)},
        "failure_counts": {"embedding_failures": 0, "validation_failures": 0},
        "seed": config["embedding_validation"]["seed"],
        "environment": environment_summary(),
        "command": (
            "python scripts/build_embeddings.py --config configs/experiment.yaml --validate-only"
        )
        if args.validate_only
        else "python scripts/build_embeddings.py --config configs/experiment.yaml",
        "validation": validation,
        "scope": scope,
        "image_text_fusion": {
            "image_weight": fusion["image_weight"],
            "text_weight": fusion["text_weight"],
            "selection_status": "reference_pending_validation_search",
        },
        "explanation_evidence_boundary": "no_image_derived_text",
    }
    write_new_json(runtime_manifest_path, manifest)
    write_json(tracked_manifest_path, manifest)
    print(json.dumps({"run_id": run_id, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
