"""Fresh, provenance-bound source producers for final_eval_v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..artifacts import item_table_fingerprint
from ..cache import file_fingerprint, stable_fingerprint
from ..config import AppConfig
from ..models.multimodal import fuse_embeddings
from ..reranking import weighted_rerank
from .materialization import normalize_schedule, resolve_target_embeddings
from .ranking import build_controlled_candidate_set


class EvidenceScorer(Protocol):
    def score(self, case: pd.Series, candidates: pd.DataFrame) -> np.ndarray: ...

    def retrieve(self, case: pd.Series, candidate: pd.Series) -> pd.DataFrame: ...


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def _resume_file(path: Path, protocol: str, input_fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        if _manifest_path(path).exists():
            raise ValueError(f"Orphan manifest exists without output: {_manifest_path(path)}")
        return None
    manifest_path = _manifest_path(path)
    if not manifest_path.is_file():
        raise FileExistsError(f"Refusing to overwrite unmanifested output: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != protocol:
        raise ValueError(f"Existing output is not {protocol}; legacy sources are ineligible.")
    if manifest.get("input_fingerprint") != input_fingerprint:
        raise ValueError(f"Existing {protocol} output was produced from different inputs.")
    if manifest.get("output_hash") != file_fingerprint(path):
        raise ValueError(f"Existing {protocol} output hash differs from its manifest.")
    return manifest


def materialize_target_item_table(
    *,
    config: AppConfig,
    items: pd.DataFrame,
    source_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    from ..cache import ArtifactCache

    required = {
        "item_ID",
        "outfit_ID",
        "broad_category",
        "item_text",
        "category",
        "text",
        "original_dataset_index",
    }
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"Target items are missing metadata: {sorted(missing)}")
    target_paths, compatibility = resolve_target_embeddings(
        config, items, ArtifactCache(config.paths.cache_dir, config.cache.policy)
    )
    source_hashes = {str(path): file_fingerprint(path) for path in source_paths}
    embedding_hashes = {name: file_fingerprint(path) for name, path in target_paths.items()}
    row_order = item_table_fingerprint(items)
    fingerprint = stable_fingerprint(
        {
            "row_order": row_order,
            "source_hashes": source_hashes,
            "embedding_compatibility": compatibility,
            "embedding_hashes": embedding_hashes,
        }
    )
    resumed = _resume_file(output_path, "final_eval_v2_target_items", fingerprint)
    if resumed:
        return resumed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(output_path, index=False)
    manifest = {
        "protocol": "final_eval_v2_target_items",
        "input_fingerprint": fingerprint,
        "row_count": len(items),
        "row_order_fingerprint": row_order,
        "source_hashes": source_hashes,
        "embedding_compatibility_hashes": compatibility,
        "embedding_file_hashes": embedding_hashes,
        "output_hash": file_fingerprint(output_path),
        "primary_output_protocol": "fresh_final_eval_v2",
        "v1_primary_outputs_used": False,
    }
    _manifest_path(output_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def produce_candidate_sets(
    *,
    config: AppConfig,
    split: str,
    schedule_path: Path,
    target_items_path: Path,
    scorer: EvidenceScorer,
    evidence_hash: str,
    output_path: Path,
) -> dict[str, Any]:
    schedule = normalize_schedule(pd.read_csv(schedule_path), split)
    target_manifest = json.loads(_manifest_path(target_items_path).read_text(encoding="utf-8"))
    if target_manifest.get("protocol") != "final_eval_v2_target_items":
        raise ValueError("Candidate production requires a fresh v2 target-item table.")
    if target_manifest.get("output_hash") != file_fingerprint(target_items_path):
        raise ValueError("Target-item hash differs from its manifest.")
    targets = pd.read_parquet(target_items_path).reset_index(drop=True)
    config_hash = stable_fingerprint(config.model_dump(mode="json"))
    inputs = {
        "schedule_hash": file_fingerprint(schedule_path),
        "target_items_hash": file_fingerprint(target_items_path),
        "config_hash": config_hash,
        "evidence_kb_hash": evidence_hash,
        "split": split,
        "candidate_seed": config.project.seed,
        "negatives_per_case": config.evaluation.negatives_per_case,
    }
    fingerprint = stable_fingerprint(inputs)
    resumed = _resume_file(output_path, "final_eval_v2_candidate_sets", fingerprint)
    if resumed:
        return resumed
    id_to_row = {value: index for index, value in enumerate(targets["item_ID"].astype(str))}
    rows: list[pd.DataFrame] = []
    for case_index, case in schedule.reset_index(drop=True).iterrows():
        candidates = build_controlled_candidate_set(
            targets,
            str(case["query_outfit_id"]),
            str(case["target_category"]),
            config.evaluation.negatives_per_case,
            np.random.default_rng(config.project.seed + case_index),
            query_item_id=str(case["query_item_id"]),
        )
        candidates = candidates.copy()
        candidates["paper_case_id"] = case["paper_case_id"]
        candidates["query_outfit_id"] = case["query_outfit_id"]
        candidates["target_category"] = case["target_category"]
        candidates["candidate_position"] = np.arange(len(candidates))
        candidates["target_row"] = candidates["item_ID"].astype(str).map(id_to_row)
        candidates["evidence_score"] = scorer.score(case, candidates)
        rows.append(candidates)
    result = pd.concat(rows, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    manifest = {
        "protocol": "final_eval_v2_candidate_sets",
        "input_fingerprint": fingerprint,
        **inputs,
        "row_count": len(result),
        "output_hash": file_fingerprint(output_path),
        "fresh_evidence_scoring": True,
        "v1_primary_outputs_used": False,
    }
    _manifest_path(output_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _selected_value(path: Path, kind: str) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("selected_on") != "validation":
        raise ValueError(f"Selected {kind} artifact must be frozen on validation.")
    return value, file_fingerprint(path)


def produce_selected_cases(
    *,
    split: str,
    schedule_path: Path,
    target_items_path: Path,
    candidate_sets_path: Path,
    target_clip_image_path: Path,
    target_clip_text_path: Path,
    query_clip_image_path: Path,
    query_clip_text_path: Path,
    fusion_selection_path: Path,
    reranking_selection_path: Path,
    scorer: EvidenceScorer,
    output_path: Path,
) -> dict[str, Any]:
    schedule = normalize_schedule(pd.read_csv(schedule_path), split).reset_index(drop=True)
    candidates_manifest = json.loads(
        _manifest_path(candidate_sets_path).read_text(encoding="utf-8")
    )
    if candidates_manifest.get("protocol") != "final_eval_v2_candidate_sets":
        raise ValueError("Selected cases require fresh v2 candidate sets, not v1 inputs.")
    if candidates_manifest.get("output_hash") != file_fingerprint(candidate_sets_path):
        raise ValueError("Candidate-set output hash differs from its manifest.")
    fusion, fusion_hash = _selected_value(fusion_selection_path, "fusion")
    reranking, reranking_hash = _selected_value(reranking_selection_path, "reranking")
    sources = [
        schedule_path,
        target_items_path,
        candidate_sets_path,
        target_clip_image_path,
        target_clip_text_path,
        query_clip_image_path,
        query_clip_text_path,
    ]
    inputs = {
        "schedule_hash": file_fingerprint(schedule_path),
        "target_items_hash": file_fingerprint(target_items_path),
        "candidate_set_hash": file_fingerprint(candidate_sets_path),
        "selected_fusion_hash": fusion_hash,
        "selected_reranking_hash": reranking_hash,
        "embedding_hashes": {str(path): file_fingerprint(path) for path in sources[3:]},
        "split": split,
    }
    fingerprint = stable_fingerprint(inputs)
    resumed = _resume_file(output_path, "final_eval_v2_selected_cases", fingerprint)
    if resumed:
        return resumed
    targets = pd.read_parquet(target_items_path).reset_index(drop=True)
    candidates = pd.read_csv(candidate_sets_path)
    target_image = np.load(target_clip_image_path, mmap_mode="r")
    target_text = np.load(target_clip_text_path, mmap_mode="r")
    query_image = np.load(query_clip_image_path, mmap_mode="r")
    query_text = np.load(query_clip_text_path, mmap_mode="r")
    rows = []
    for index, case in schedule.iterrows():
        pool = candidates[candidates["paper_case_id"] == case["paper_case_id"]].copy()
        target_rows = pool["target_row"].astype(int).to_numpy()
        fused_targets = fuse_embeddings(
            target_image[target_rows], target_text[target_rows], float(fusion["image_weight"])
        )
        fused_query = fuse_embeddings(
            query_image[index][None, :], query_text[index][None, :], float(fusion["image_weight"])
        )[0]
        pool["clip_score"] = fused_targets @ fused_query
        ranked = weighted_rerank(
            pool,
            float(reranking["clip_weight"]),
            float(reranking.get("evidence_weight", 1.0 - float(reranking["clip_weight"]))),
            normalize_scores=True,
        )
        chosen = ranked.iloc[0]
        item = targets.iloc[int(chosen["target_row"])]
        rules = scorer.retrieve(case, item)
        rule_ids = rules.get("rule_id", pd.Series(dtype=str)).astype(str).tolist()
        rule_texts = rules.get("rule_text", pd.Series(dtype=str)).astype(str).tolist()
        rule_sources = rules.get("source_title", pd.Series(dtype=str)).astype(str).tolist()
        rows.append(
            {
                **case.to_dict(),
                "recommended_item_id": str(item["item_ID"]),
                "recommended_text": str(item["item_text"]),
                "recommended_target_row": int(chosen["target_row"]),
                "locked_clip_score": float(chosen["clip_score"]),
                "locked_evidence_score": float(chosen["evidence_score"]),
                "locked_final_score": float(chosen["final_score"]),
                "item_evidence_text": str(item["item_text"]),
                "item_evidence_packet": json.dumps(item.to_dict(), default=str),
                "rule_evidence_ids": json.dumps(rule_ids),
                "rule_evidence_text": "\n".join(rule_texts),
                "rule_evidence_sources": json.dumps(rule_sources),
                "rule_evidence_packet": rules.to_json(orient="records"),
                "packet_source_protocol": "final_eval_v2_selected",
            }
        )
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    manifest = {
        "protocol": "final_eval_v2_selected_cases",
        "input_fingerprint": fingerprint,
        **inputs,
        "output_hash": file_fingerprint(output_path),
        "row_count": len(result),
        "packet_source_protocol": "final_eval_v2_selected",
        "v1_primary_outputs_used": False,
    }
    _manifest_path(output_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
