"""Preparation and validation-only reranking selection for final_eval_v2 Stage 1."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..cache import file_fingerprint, stable_fingerprint
from ..models.multimodal import fuse_embeddings
from ..reranking import weighted_rerank
from .protocol_gate import generation_packet_hash
from .ranking import ranking_metrics
from .stage1 import BUNDLE_FILES, load_stage1_bundle
from .tuning import select_reranking_weight

TARGET_ARRAYS = ("target_minilm.npy", "target_clip_image.npy", "target_clip_text.npy")
QUERY_ARRAYS = ("query_minilm.npy", "query_clip_image.npy", "query_clip_text.npy")


def _source_fingerprint(paths: list[Path], settings: dict[str, Any]) -> str:
    return stable_fingerprint(
        {"files": {str(path): file_fingerprint(path) for path in paths}, "settings": settings}
    )


def _resume_or_fail(output_dir: Path, fingerprint: str) -> bool:
    if not output_dir.exists():
        return False
    manifest_path = output_dir / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"Refusing to overwrite unmanifested directory: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_fingerprint") != fingerprint:
        raise ValueError("Existing prepared output was created from different inputs.")
    missing = [name for name in BUNDLE_FILES if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"Prepared bundle is incomplete: {missing}")
    return True


def prepare_stage1_bundle(
    *,
    split: str,
    schedule_path: Path,
    candidate_sets_path: Path,
    target_embedding_dir: Path,
    query_embedding_dir: Path,
    output_dir: Path,
) -> str:
    """Materialize a bundle without encoding or rebuilding any embedding."""

    if split not in {"validation", "test"}:
        raise ValueError("Prepared Stage 1 split must be validation or test.")
    source_paths = [schedule_path, candidate_sets_path]
    source_paths.extend(target_embedding_dir / name for name in TARGET_ARRAYS)
    source_paths.extend(query_embedding_dir / name for name in QUERY_ARRAYS)
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Prepared bundle requires existing cached/materialized inputs; no embeddings will "
            f"be rebuilt. Missing: {missing}"
        )
    fingerprint = _source_fingerprint(source_paths, {"split": split, "schema_version": 2})
    if _resume_or_fail(output_dir, fingerprint):
        return fingerprint
    schedule = pd.read_csv(schedule_path)
    if "research_split" not in schedule or set(schedule["research_split"].astype(str)) != {split}:
        raise ValueError(f"Schedule must contain only {split} rows.")
    candidates = pd.read_csv(candidate_sets_path)
    if set(candidates["paper_case_id"]) != set(schedule["paper_case_id"]):
        raise ValueError("Candidate sets must match schedule case IDs exactly.")
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(schedule_path, output_dir / "schedule.csv")
    shutil.copy2(candidate_sets_path, output_dir / "candidate_sets.csv")
    for name in TARGET_ARRAYS:
        shutil.copy2(target_embedding_dir / name, output_dir / name)
    for name in QUERY_ARRAYS:
        shutil.copy2(query_embedding_dir / name, output_dir / name)
    # Load after copying to validate alignment before declaring the bundle complete.
    load_stage1_bundle(output_dir, split)
    (output_dir / "preparation_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "final_eval_v2",
                "research_split": split,
                "input_fingerprint": fingerprint,
                "source_hashes": {str(path): file_fingerprint(path) for path in source_paths},
                "embeddings_rebuilt": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return fingerprint


def tune_reranking_v2_artifacts(
    *,
    bundle_dir: Path,
    fusion_selection_path: Path,
    resolved_config_path: Path,
    output_dir: Path,
    clip_weights: list[float],
    cutoffs: list[int],
) -> dict[str, Any]:
    bundle = load_stage1_bundle(bundle_dir, "validation")
    fusion = json.loads(fusion_selection_path.read_text(encoding="utf-8"))
    if fusion.get("selected_on") != "validation":
        raise ValueError("Fusion selection must be frozen on validation.")
    if "evidence_score" not in bundle.candidates:
        raise ValueError("Validation candidate sets require cached evidence_score values.")
    bundle_files = [bundle_dir / name for name in BUNDLE_FILES]
    input_fingerprint = _source_fingerprint(
        [fusion_selection_path, resolved_config_path, *bundle_files],
        {"clip_weights": clip_weights, "cutoffs": cutoffs},
    )
    manifest_path = output_dir / "stage_manifest.json"
    if output_dir.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"Refusing to overwrite unmanifested directory: {output_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_fingerprint") != input_fingerprint:
            raise ValueError("Existing reranking selection used different inputs.")
        selected_path = output_dir / "selected_weight.json"
        if not selected_path.is_file():
            raise ValueError("Existing reranking output is incomplete.")
        return json.loads(selected_path.read_text(encoding="utf-8"))
    image_weight = float(fusion["image_weight"])
    rows = []
    for query_index, case in bundle.schedule.iterrows():
        candidates = bundle.candidates[
            bundle.candidates["paper_case_id"] == case["paper_case_id"]
        ].copy()
        target_rows = candidates["target_row"].astype(int).to_numpy()
        target_fused = fuse_embeddings(
            bundle.target_clip_image[target_rows],
            bundle.target_clip_text[target_rows],
            image_weight,
        )
        query_fused = fuse_embeddings(
            bundle.query_clip_image[query_index][None, :],
            bundle.query_clip_text[query_index][None, :],
            image_weight,
        )[0]
        candidates["clip_score"] = target_fused @ query_fused
        for clip_weight in clip_weights:
            ranked = weighted_rerank(
                candidates,
                clip_weight,
                1.0 - clip_weight,
                normalize_scores=True,
            )
            rows.append(
                {
                    "paper_case_id": case["paper_case_id"],
                    "query_outfit_id": case["query_outfit_id"],
                    "target_category": case["target_category"],
                    "clip_weight": clip_weight,
                    "evidence_weight": 1.0 - clip_weight,
                    **ranking_metrics(ranked["is_positive"].astype(int), cutoffs),
                }
            )
    results = pd.DataFrame(rows)
    summary = select_reranking_weight(results)
    selected = summary.iloc[0].to_dict()
    selected.update(
        {
            "selected_on": "validation",
            "config_hash": file_fingerprint(resolved_config_path),
            "schedule_hash": file_fingerprint(bundle_dir / "schedule.csv"),
            "bundle_fingerprint": bundle.fingerprint,
            "metric_hierarchy": ["ndcg_at_10", "hit_rate_at_10", "reciprocal_rank"],
            "validation_results_path": str(output_dir / "validation_results.csv"),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    results.to_csv(output_dir / "validation_results.csv", index=False)
    summary.to_csv(output_dir / "validation_summary.csv", index=False)
    (output_dir / "selected_weight.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    (output_dir / "stage_manifest.json").write_text(
        json.dumps({"input_fingerprint": input_fingerprint, "selected_on": "validation"}, indent=2),
        encoding="utf-8",
    )
    return selected


def create_locked_packets_v2(
    *,
    source_cases_path: Path,
    fusion_selection_path: Path,
    reranking_selection_path: Path,
    output_path: Path,
    expected_split: str,
) -> str:
    """Freeze selected-v2 recommendation/evidence packets; reject legacy sources."""

    source = pd.read_csv(source_cases_path)
    required = {
        "paper_case_id",
        "research_split",
        "packet_source_protocol",
        "recommended_item_id",
        "item_evidence_text",
        "rule_evidence_ids",
        "rule_evidence_text",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Locked packet source is missing columns: {sorted(missing)}")
    if set(source["research_split"].astype(str)) != {expected_split}:
        raise ValueError(f"Locked packet source must contain only {expected_split} rows.")
    if set(source["packet_source_protocol"].astype(str)) != {"final_eval_v2_selected"}:
        raise ValueError("Legacy packet sources cannot be relabelled as final_eval_v2 selected.")
    selections = []
    for name, path in (
        ("fusion", fusion_selection_path),
        ("reranking", reranking_selection_path),
    ):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("selected_on") != "validation":
            raise ValueError(f"{name} selection must be frozen on validation.")
        selections.append(file_fingerprint(path))
    input_fingerprint = _source_fingerprint(
        [source_cases_path, fusion_selection_path, reranking_selection_path],
        {"expected_split": expected_split},
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"Refusing to overwrite unmanifested packets: {output_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_fingerprint") != input_fingerprint:
            raise ValueError("Existing locked packets used different inputs.")
        if manifest.get("protocol") != "final_eval_v2_locked_packets":
            raise ValueError("Existing locked packets use an invalid or legacy protocol.")
        if manifest.get("output_hash") != file_fingerprint(output_path):
            raise ValueError("Existing locked packet hash differs from its manifest.")
        return str(manifest["stage1_packet_hash"])
    packets = source.copy()
    packets["stage1_packet_protocol"] = "final_eval_v2_selected"
    packets["generation_packet_hash"] = packets.apply(generation_packet_hash, axis=1)
    packet_set_hash = stable_fingerprint(
        {
            "packets": packets.sort_values("paper_case_id")[
                ["paper_case_id", "generation_packet_hash"]
            ].to_dict("records"),
            "selection_hashes": selections,
        }
    )
    packets["stage1_packet_hash"] = packet_set_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    packets.to_csv(output_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": "final_eval_v2_locked_packets",
                "input_fingerprint": input_fingerprint,
                "stage1_packet_hash": packet_set_hash,
                "research_split": expected_split,
                "stage1_packet_protocol": "final_eval_v2_selected",
                "output_hash": file_fingerprint(output_path),
                "v1_primary_outputs_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return packet_set_hash
