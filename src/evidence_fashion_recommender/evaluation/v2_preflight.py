"""Read-only readiness inspection for final_eval_v2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..cache import file_fingerprint
from ..config import AppConfig
from .materialization import QUERY_NAMES, query_cache_directory
from .stage1 import load_stage1_bundle


def _check_manifested(
    path: Path, protocol: str, root: Path, *, optional: bool = False
) -> tuple[bool, str]:
    if not path.is_relative_to(root):
        return False, f"outside v2 root: {path}"
    if not path.is_file():
        return False, "not yet expected" if optional else f"missing: {path}"
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return False, f"missing manifest: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != protocol:
        return False, f"protocol mismatch/legacy reference: {path}"
    if manifest.get("output_hash") != file_fingerprint(path):
        return False, f"hash mismatch: {path}"
    if manifest.get("v1_primary_outputs_used") is True:
        return False, f"v1 primary reference detected: {path}"
    return True, "valid"


def inspect_readiness(
    config: AppConfig,
    schedules: dict[str, Path],
) -> dict[str, Any]:
    root = config.final_evaluation.output_root
    sources = root / "sources"
    materialized = root / "materialized"
    prepared = root / "prepared"
    validation = root / "validation"
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, ready: bool, detail: str) -> None:
        checks[name] = {"status": "READY" if ready else "BLOCKED", "detail": detail}

    ready, detail = _check_manifested(
        sources / "target_items.parquet", "final_eval_v2_target_items", root
    )
    record("target_items", ready, detail)
    target_manifest = materialized / "target_embeddings" / "target_embedding_manifest.json"
    target_ready = False
    target_detail = "not materialized"
    if target_manifest.is_file():
        value = json.loads(target_manifest.read_text(encoding="utf-8"))
        target_ready = value.get("protocol") == "final_eval_v2_cached_target_embeddings"
        for name, expected_hash in value.get("output_hashes", {}).items():
            path = materialized / "target_embeddings" / name
            target_ready &= path.is_file() and file_fingerprint(path) == expected_hash
        target_detail = "valid" if target_ready else "manifest/hash mismatch"
    record("target_embeddings", target_ready, target_detail)

    query_missing = False
    for split, schedule in schedules.items():
        query_dir = query_cache_directory(config, schedule, split)
        query_manifest_path = query_dir / "query_embedding_manifest.json"
        query_ready = query_manifest_path.is_file()
        if query_ready:
            query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
            query_ready = query_manifest.get("protocol") == "final_eval_v2_query_embeddings"
            for name, expected_hash in query_manifest.get("output_hashes", {}).items():
                path = query_dir / name
                query_ready &= path.is_file() and file_fingerprint(path) == expected_hash
            query_ready &= set(query_manifest.get("output_hashes", {})) == set(QUERY_NAMES)
        query_missing |= not query_ready
        record(
            f"{split}_query_embeddings",
            query_ready,
            "valid" if query_ready else f"query-only computation required: {query_dir}",
        )
        ready, detail = _check_manifested(
            sources / split / "candidate_sets.csv", "final_eval_v2_candidate_sets", root
        )
        record(f"{split}_candidate_sets", ready, detail)
        bundle = prepared / split
        bundle_files = (
            "schedule.csv",
            "candidate_sets.csv",
            "target_minilm.npy",
            "target_clip_image.npy",
            "target_clip_text.npy",
            *QUERY_NAMES,
            "preparation_manifest.json",
        )
        bundle_ready = all((bundle / name).is_file() for name in bundle_files)
        bundle_detail = "missing"
        if bundle_ready:
            try:
                load_stage1_bundle(bundle, split)
                bundle_detail = "valid"
            except (ValueError, OSError) as error:
                bundle_ready = False
                bundle_detail = f"invalid bundle: {error}"
        record(f"{split}_prepared_bundle", bundle_ready, bundle_detail)

    selection_paths = {
        "selected_fusion": validation / "fusion_tuning" / "selected_fusion.json",
        "selected_weight": validation / "reranking_tuning" / "selected_weight.json",
    }
    for name, path in selection_paths.items():
        if not path.is_file():
            record(name, False, f"not yet selected: {path}")
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            record(name, value.get("selected_on") == "validation", "validation-frozen")

    for split in schedules:
        ready, detail = _check_manifested(
            sources / split / "selected_cases.csv",
            "final_eval_v2_selected_cases",
            root,
            optional=True,
        )
        record(f"{split}_selected_cases", ready, detail)
        locked_candidates = (
            prepared / split / "locked_packets.csv",
            materialized / split / "locked_packets.csv",
        )
        locked = next((path for path in locked_candidates if path.is_file()), locked_candidates[0])
        ready, detail = _check_manifested(
            locked, "final_eval_v2_locked_packets", root, optional=True
        )
        record(f"{split}_locked_packets", ready, detail)

    candidates_ready = all(
        checks[f"{split}_candidate_sets"]["status"] == "READY" for split in schedules
    )
    target_source_ready = checks["target_items"]["status"] == "READY"
    if not target_source_ready:
        next_command = (
            "uv run efr --config configs/final_eval_v2.yaml materialize-final-eval-v2-target-items"
        )
    elif query_missing:
        next_command = (
            "APPROVAL REQUIRED: materialize-final-retrieval-v2-query-embeddings "
            "--split validation --approve-compute-query-embeddings"
        )
    elif not candidates_ready:
        next_command = (
            "uv run efr --config configs/final_eval_v2.yaml produce-final-eval-v2-candidates "
            "--split validation"
        )
    else:
        next_command = (
            "uv run efr --config configs/final_eval_v2.yaml "
            "prepare-final-retrieval-v2-bundle --split validation"
        )
    return {
        "protocol": "final_eval_v2_readiness",
        "checks": checks,
        "query_embedding_computation_required": query_missing,
        "v1_primary_reference_detected": any("v1" in value["detail"] for value in checks.values()),
        "exact_next_safe_command": next_command,
    }
