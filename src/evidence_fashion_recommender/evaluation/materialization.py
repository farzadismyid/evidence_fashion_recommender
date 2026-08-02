"""Fresh v2 materialization with safe reuse of fingerprint-compatible low-level caches."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import embedding_record
from ..cache import ArtifactCache, file_fingerprint, stable_fingerprint
from ..config import AppConfig

TARGET_NAMES = {
    "minilm_text": "target_minilm.npy",
    "clip_image": "target_clip_image.npy",
    "clip_text": "target_clip_text.npy",
}
QUERY_NAMES = ("query_minilm.npy", "query_clip_image.npy", "query_clip_text.npy")
CANDIDATE_COLUMNS = {
    "paper_case_id",
    "candidate_position",
    "target_row",
    "is_positive",
    "evidence_score",
}
SELECTED_CASE_COLUMNS = {
    "paper_case_id",
    "research_split",
    "recommended_item_id",
    "recommended_text",
    "item_evidence_text",
    "rule_evidence_ids",
    "rule_evidence_text",
    "packet_source_protocol",
}


def normalize_schedule(schedule: pd.DataFrame, split: str) -> pd.DataFrame:
    result = schedule.copy()
    if "research_split" not in result or set(result["research_split"].astype(str)) != {split}:
        raise ValueError(f"Schedule must contain only {split} rows.")
    if "paper_case_id" not in result:
        if "case_index" not in result:
            raise ValueError("Schedule needs paper_case_id or case_index.")
        result["paper_case_id"] = result.apply(
            lambda row: f"V2_{split.upper()}_{int(row['case_index']):04d}_{row['target_category']}",
            axis=1,
        )
    if result["paper_case_id"].duplicated().any():
        raise ValueError("Schedule paper_case_id values must be unique.")
    return result


def query_cache_fingerprint(config: AppConfig, schedule_path: Path, split: str) -> str:
    return stable_fingerprint(
        {
            "protocol": "final_eval_v2_query_embeddings",
            "schema_version": 2,
            "research_split": split,
            "schedule_hash": file_fingerprint(schedule_path),
            "text_model": config.models.text_embedding.model_dump(mode="json"),
            "clip_model": config.models.multimodal_embedding.model_dump(mode="json"),
        }
    )


def query_cache_directory(config: AppConfig, schedule_path: Path, split: str) -> Path:
    return (
        config.final_evaluation.output_root
        / "materialized"
        / "query_embeddings"
        / split
        / query_cache_fingerprint(config, schedule_path, split)
    )


def resolve_target_embeddings(
    config: AppConfig,
    target_items: pd.DataFrame,
    cache: ArtifactCache,
) -> tuple[dict[str, Path], dict[str, str]]:
    paths, fingerprints = {}, {}
    for modality, output_name in TARGET_NAMES.items():
        record = embedding_record(cache, config, target_items, modality)
        if not record.path.is_file():
            raise FileNotFoundError(
                f"Missing fingerprint-compatible cached {modality} target embeddings; "
                "target rebuilding is not allowed by materialization."
            )
        paths[output_name] = record.path
        fingerprints[modality] = record.fingerprint
    return paths, fingerprints


def _validate_source_manifest(
    source_path: Path,
    *,
    required_protocol: str,
    schedule_hash: str,
) -> dict[str, Any]:
    manifest_path = source_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != required_protocol:
        raise ValueError(
            f"Source protocol must be {required_protocol}; legacy v1 outputs are ineligible."
        )
    if manifest.get("schedule_hash") != schedule_hash:
        raise ValueError("Source manifest does not match the frozen schedule hash.")
    if manifest.get("output_hash") != file_fingerprint(source_path):
        raise ValueError("Source file hash does not match its manifest.")
    return manifest


def _resume_manifest(output_dir: Path, fingerprint: str, required: list[str]) -> bool:
    if not output_dir.exists():
        return False
    manifest_path = output_dir / "materialization_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"Refusing to overwrite unmanifested output: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_fingerprint") != fingerprint:
        raise ValueError("Existing materialized inputs were created from different sources.")
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"Existing materialization is incomplete: {missing}")
    for name, expected_hash in manifest.get("output_hashes", {}).items():
        output = output_dir / name
        if not output.is_file() or file_fingerprint(output) != expected_hash:
            raise ValueError(f"Existing materialized output hash differs: {output}")
    return True


def _validate_or_write_target_manifest(
    target_dir: Path,
    target_paths: dict[str, Path],
    target_fingerprints: dict[str, str],
) -> None:
    manifest_path = target_dir / "target_embedding_manifest.json"
    expected_hashes = {name: file_fingerprint(path) for name, path in target_paths.items()}
    if target_dir.exists():
        if not manifest_path.is_file():
            raise FileExistsError("Refusing to overwrite unmanifested target embedding directory.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("embedding_fingerprints") != target_fingerprints:
            raise ValueError("Existing target embedding fingerprints differ.")
        if manifest.get("output_hashes") != expected_hashes:
            raise ValueError("Existing target embedding manifest hashes differ.")
        for name, expected_hash in expected_hashes.items():
            output = target_dir / name
            if not output.is_file() or file_fingerprint(output) != expected_hash:
                raise ValueError(f"Existing target embedding differs: {output}")
        return
    target_dir.mkdir(parents=True)
    for output_name, source in target_paths.items():
        shutil.copy2(source, target_dir / output_name)
    target_manifest = {
        "protocol": "final_eval_v2_cached_target_embeddings",
        "embedding_fingerprints": target_fingerprints,
        "source_paths": {name: str(path) for name, path in target_paths.items()},
        "output_hashes": expected_hashes,
        "embeddings_copied": True,
        "embeddings_computed": False,
    }
    manifest_path.write_text(json.dumps(target_manifest, indent=2), encoding="utf-8")


def materialize_retrieval_inputs(
    *,
    config: AppConfig,
    split: str,
    schedule_path: Path,
    target_items: pd.DataFrame,
    candidate_source: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Copy validated cached inputs; never compute target or query embeddings."""

    schedule = normalize_schedule(pd.read_csv(schedule_path), split)
    schedule_hash = file_fingerprint(schedule_path)
    cache = ArtifactCache(config.paths.cache_dir, config.cache.policy)
    target_paths, target_fingerprints = resolve_target_embeddings(config, target_items, cache)
    query_dir = query_cache_directory(config, schedule_path, split)
    missing_query = [name for name in QUERY_NAMES if not (query_dir / name).is_file()]
    if missing_query:
        raise FileNotFoundError(
            "Separate v2 query embeddings are missing. Run the explicitly approved "
            f"query-only materialization command. Missing: {missing_query}"
        )
    candidate_manifest = _validate_source_manifest(
        candidate_source,
        required_protocol="final_eval_v2_candidate_sets",
        schedule_hash=schedule_hash,
    )
    candidates = pd.read_csv(candidate_source)
    missing_columns = CANDIDATE_COLUMNS - set(candidates.columns)
    if missing_columns:
        raise ValueError(f"Candidate source is missing columns: {sorted(missing_columns)}")
    if set(candidates["paper_case_id"]) != set(schedule["paper_case_id"]):
        raise ValueError("Candidate source and schedule case IDs differ.")
    source_paths = [
        schedule_path,
        candidate_source,
        *target_paths.values(),
        *[query_dir / name for name in QUERY_NAMES],
    ]
    fingerprint = stable_fingerprint(
        {
            "sources": {str(path): file_fingerprint(path) for path in source_paths},
            "target_fingerprints": target_fingerprints,
            "candidate_manifest": candidate_manifest,
            "split": split,
        }
    )
    split_dir = output_root / split
    target_dir = output_root / "target_embeddings"
    required = ["schedule.csv", "candidate_sets.csv", *QUERY_NAMES]
    _validate_or_write_target_manifest(target_dir, target_paths, target_fingerprints)
    if _resume_manifest(split_dir, fingerprint, required):
        return json.loads((split_dir / "materialization_manifest.json").read_text())
    split_dir.mkdir(parents=True, exist_ok=False)
    schedule.to_csv(split_dir / "schedule.csv", index=False)
    shutil.copy2(candidate_source, split_dir / "candidate_sets.csv")
    query_output = split_dir / "query_embeddings"
    query_output.mkdir()
    for name in QUERY_NAMES:
        shutil.copy2(query_dir / name, query_output / name)
        shutil.copy2(query_dir / name, split_dir / name)
    manifest = {
        "protocol": "final_eval_v2_materialized_inputs",
        "research_split": split,
        "input_fingerprint": fingerprint,
        "source_paths": [str(path) for path in source_paths],
        "source_hashes": {str(path): file_fingerprint(path) for path in source_paths},
        "schedule_hash": schedule_hash,
        "case_hash": stable_fingerprint(schedule.to_dict("records")),
        "embedding_fingerprints": {
            "target": target_fingerprints,
            "query": query_cache_fingerprint(config, schedule_path, split),
        },
        "embedding_actions": {"target": "copied", "query": "copied", "computed": False},
        "output_hashes": {
            "schedule.csv": file_fingerprint(split_dir / "schedule.csv"),
            "candidate_sets.csv": file_fingerprint(split_dir / "candidate_sets.csv"),
            **{name: file_fingerprint(split_dir / name) for name in QUERY_NAMES},
        },
        "primary_output_protocol": "fresh_final_eval_v2",
        "v1_primary_outputs_used": False,
    }
    (split_dir / "materialization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def materialize_query_embeddings(
    *,
    config: AppConfig,
    split: str,
    schedule_path: Path,
    builder: Callable[[pd.DataFrame], dict[str, np.ndarray]],
    approved: bool,
) -> Path:
    """Explicit query-only computation hook; target embeddings are never accepted or written."""

    if not approved:
        raise PermissionError("Query embedding computation requires explicit approval.")
    schedule = normalize_schedule(pd.read_csv(schedule_path), split)
    cache_dir = query_cache_directory(config, schedule_path, split)
    manifest_path = cache_dir / "query_embedding_manifest.json"
    if cache_dir.exists():
        if not manifest_path.is_file():
            raise FileExistsError("Refusing to overwrite unmanifested query cache.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != query_cache_fingerprint(config, schedule_path, split):
            raise ValueError("Existing query cache fingerprint differs.")
        for name, expected_hash in manifest.get("output_hashes", {}).items():
            output = cache_dir / name
            if not output.is_file() or file_fingerprint(output) != expected_hash:
                raise ValueError(f"Existing query cache hash differs: {output}")
        if set(manifest.get("output_hashes", {})) != set(QUERY_NAMES):
            raise ValueError("Existing query cache manifest is incomplete.")
        return cache_dir
    arrays = builder(schedule)
    expected = {name.removesuffix(".npy") for name in QUERY_NAMES}
    if set(arrays) != expected:
        raise ValueError(f"Query builder must return exactly: {sorted(expected)}")
    if any(len(value) != len(schedule) for value in arrays.values()):
        raise ValueError("Query embedding arrays must align with the schedule.")
    cache_dir.mkdir(parents=True, exist_ok=False)
    output_hashes = {}
    for name, array in arrays.items():
        output = cache_dir / f"{name}.npy"
        partial = output.with_suffix(".npy.partial")
        with partial.open("wb") as stream:
            np.save(stream, np.asarray(array, dtype=np.float32))
        partial.replace(output)
        output_hashes[output.name] = file_fingerprint(output)
    manifest = {
        "protocol": "final_eval_v2_query_embeddings",
        "fingerprint": query_cache_fingerprint(config, schedule_path, split),
        "schedule_hash": file_fingerprint(schedule_path),
        "config_hash": stable_fingerprint(config.model_dump(mode="json")),
        "text_model": config.models.text_embedding.model_dump(mode="json"),
        "clip_model": config.models.multimodal_embedding.model_dump(mode="json"),
        "target_embeddings_rebuilt": False,
        "query_embeddings_computed": True,
        "output_hashes": output_hashes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cache_dir


def materialize_selected_cases(
    *,
    split: str,
    schedule_path: Path,
    source_cases: Path,
    fusion_selection: Path,
    reranking_selection: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Materialize fresh selected-v2 cases only after both validation selections exist."""

    schedule = normalize_schedule(pd.read_csv(schedule_path), split)
    schedule_hash = file_fingerprint(schedule_path)
    _validate_source_manifest(
        source_cases,
        required_protocol="final_eval_v2_selected_cases",
        schedule_hash=schedule_hash,
    )
    cases = pd.read_csv(source_cases)
    missing = SELECTED_CASE_COLUMNS - set(cases.columns)
    if missing:
        raise ValueError(f"Selected cases are missing columns: {sorted(missing)}")
    if set(cases["packet_source_protocol"].astype(str)) != {"final_eval_v2_selected"}:
        raise ValueError("Old v1 cases cannot be used as primary final_eval_v2 selected cases.")
    if set(cases["paper_case_id"]) != set(schedule["paper_case_id"]):
        raise ValueError("Selected cases and schedule case IDs differ.")
    selection_hashes = {}
    for name, path in (("fusion", fusion_selection), ("reranking", reranking_selection)):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("selected_on") != "validation":
            raise ValueError(f"{name} selection must be frozen on validation.")
        selection_hashes[name] = file_fingerprint(path)
    fingerprint = stable_fingerprint(
        {
            "source": file_fingerprint(source_cases),
            "schedule": schedule_hash,
            "selections": selection_hashes,
            "split": split,
        }
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists():
        if not manifest_path.is_file():
            raise FileExistsError("Refusing to overwrite unmanifested selected cases.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_fingerprint") != fingerprint:
            raise ValueError("Existing selected cases used different inputs.")
        if manifest.get("output_hash") != file_fingerprint(output_path):
            raise ValueError("Existing selected cases hash differs.")
        return manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output_path, index=False)
    manifest = {
        "protocol": "final_eval_v2_selected_cases",
        "research_split": split,
        "input_fingerprint": fingerprint,
        "source_path": str(source_cases),
        "source_hash": file_fingerprint(source_cases),
        "schedule_hash": schedule_hash,
        "selection_hashes": selection_hashes,
        "output_hash": file_fingerprint(output_path),
        "primary_output_protocol": "fresh_final_eval_v2",
        "v1_primary_outputs_used": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
