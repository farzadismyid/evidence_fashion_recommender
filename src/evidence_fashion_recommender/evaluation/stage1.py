"""Artifact orchestration for final_eval_v2 Stage 1 retrieval evaluation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cache import file_fingerprint, stable_fingerprint
from ..models.multimodal import fuse_embeddings
from ..reranking import weighted_rerank
from .modality import (
    evaluate_modality_case,
    score_modality_candidates,
    select_fusion_weight,
    summarize_modality_results,
)
from .protocol_gate import MaterialChangePolicy, compare_locked_packets, generation_packet_hash
from .ranking import ranking_metrics


@dataclass(frozen=True)
class Stage1Bundle:
    root: Path
    schedule: pd.DataFrame
    candidates: pd.DataFrame
    target_minilm: np.ndarray
    target_clip_image: np.ndarray
    target_clip_text: np.ndarray
    query_minilm: np.ndarray
    query_clip_image: np.ndarray
    query_clip_text: np.ndarray
    fingerprint: str


BUNDLE_FILES = (
    "schedule.csv",
    "candidate_sets.csv",
    "target_minilm.npy",
    "target_clip_image.npy",
    "target_clip_text.npy",
    "query_minilm.npy",
    "query_clip_image.npy",
    "query_clip_text.npy",
)


def load_stage1_bundle(root: Path, expected_split: str) -> Stage1Bundle:
    missing = [name for name in BUNDLE_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Stage 1 bundle is missing files: {missing}")
    schedule = pd.read_csv(root / "schedule.csv")
    if "research_split" not in schedule or set(schedule["research_split"].astype(str)) != {
        expected_split
    }:
        raise ValueError(f"Stage 1 bundle must contain only {expected_split} schedule rows.")
    if schedule["paper_case_id"].duplicated().any():
        raise ValueError("Stage 1 schedule contains duplicate paper_case_id values.")
    candidates = pd.read_csv(root / "candidate_sets.csv")
    required_candidates = {"paper_case_id", "target_row", "is_positive", "candidate_position"}
    missing_columns = required_candidates - set(candidates.columns)
    if missing_columns:
        raise ValueError(f"Candidate sets are missing columns: {sorted(missing_columns)}")
    if set(candidates["paper_case_id"]) != set(schedule["paper_case_id"]):
        raise ValueError("Candidate sets and schedule contain different case IDs.")
    hashes = {name: file_fingerprint(root / name) for name in BUNDLE_FILES}
    arrays = {
        name: np.load(root / f"{name}.npy", mmap_mode="r")
        for name in (
            "target_minilm",
            "target_clip_image",
            "target_clip_text",
            "query_minilm",
            "query_clip_image",
            "query_clip_text",
        )
    }
    query_rows = len(schedule)
    if any(len(arrays[name]) != query_rows for name in arrays if name.startswith("query_")):
        raise ValueError("Query embedding rows must align with the schedule.")
    target_rows = len(arrays["target_minilm"])
    if any(
        len(arrays[name]) != target_rows for name in ("target_clip_image", "target_clip_text")
    ):
        raise ValueError("Target embedding arrays must have identical row counts.")
    if candidates["target_row"].min() < 0 or candidates["target_row"].max() >= target_rows:
        raise ValueError("Candidate target_row values fall outside target embeddings.")
    return Stage1Bundle(
        root=root,
        schedule=schedule.reset_index(drop=True),
        candidates=candidates.sort_values(
            ["paper_case_id", "candidate_position"], kind="stable"
        ).reset_index(drop=True),
        fingerprint=stable_fingerprint(hashes),
        **arrays,
    )


def evaluate_bundle_modalities(
    bundle: Stage1Bundle, image_weights: list[float], cutoffs: list[int]
) -> pd.DataFrame:
    rows = []
    for query_index, case in bundle.schedule.iterrows():
        candidates = bundle.candidates[
            bundle.candidates["paper_case_id"] == case["paper_case_id"]
        ]
        target_rows = candidates["target_row"].astype(int).to_numpy()
        scores = score_modality_candidates(
            bundle.target_minilm[target_rows],
            bundle.target_clip_image[target_rows],
            bundle.target_clip_text[target_rows],
            bundle.query_minilm[query_index],
            bundle.query_clip_image[query_index],
            bundle.query_clip_text[query_index],
            image_weights,
        )
        rows.append(
            evaluate_modality_case(
                case_id=str(case["paper_case_id"]),
                outfit_id=str(case["query_outfit_id"]),
                target_category=str(case["target_category"]),
                relevance=candidates["is_positive"].astype(int).to_numpy(),
                scores=scores,
                cutoffs=cutoffs,
            )
        )
    return pd.concat(rows, ignore_index=True)


def _guard_output(output_dir: Path, input_fingerprint: str, expected: list[str]) -> bool:
    manifest_path = output_dir / "stage_manifest.json"
    if not output_dir.exists():
        return False
    if not manifest_path.is_file():
        raise FileExistsError(f"Refusing to overwrite unmanifested output directory: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_fingerprint") != input_fingerprint:
        raise ValueError("Existing Stage 1 outputs were produced from different inputs.")
    missing = [name for name in expected if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"Existing Stage 1 output is incomplete: {missing}")
    return True


def tune_clip_fusion_artifacts(
    bundle: Stage1Bundle,
    *,
    output_dir: Path,
    image_weights: list[float],
    cutoffs: list[int],
) -> dict[str, Any]:
    expected = [
        "modality_results.csv",
        "fusion_weight_validation.csv",
        "selected_fusion.json",
    ]
    fingerprint = stable_fingerprint(
        {"bundle": bundle.fingerprint, "image_weights": image_weights, "cutoffs": cutoffs}
    )
    if _guard_output(output_dir, fingerprint, expected):
        return json.loads((output_dir / "selected_fusion.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=False)
    results = evaluate_bundle_modalities(bundle, image_weights, cutoffs)
    summary = summarize_modality_results(results)
    fused = summary[summary["method"].str.startswith("clip_fused_i")].copy()
    selected = select_fusion_weight(summary).to_dict()
    selected.update(
        {
            "selected_on": "validation",
            "bundle_fingerprint": bundle.fingerprint,
            "selection_rule": "ndcg_at_10, hit_rate_at_10, reciprocal_rank, balance",
        }
    )
    results.to_csv(output_dir / "modality_results.csv", index=False)
    fused.to_csv(output_dir / "fusion_weight_validation.csv", index=False)
    (output_dir / "selected_fusion.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    (output_dir / "stage_manifest.json").write_text(
        json.dumps({"input_fingerprint": fingerprint, "outputs": expected}, indent=2),
        encoding="utf-8",
    )
    return selected


def _load_selected(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("selected_on") != "validation":
        raise ValueError(f"{name} artifact must be selected on validation.")
    return value


def evaluate_final_retrieval_artifacts(
    bundle: Stage1Bundle,
    *,
    output_dir: Path,
    fusion_selection: Path,
    reranking_selection: Path,
    locked_packets: Path,
    cutoffs: list[int],
) -> None:
    fusion = _load_selected(fusion_selection, "fusion")
    reranking = _load_selected(reranking_selection, "reranking")
    inputs = {
        "bundle": bundle.fingerprint,
        "fusion": file_fingerprint(fusion_selection),
        "reranking": file_fingerprint(reranking_selection),
        "locked_packets": file_fingerprint(locked_packets),
        "cutoffs": cutoffs,
    }
    fingerprint = stable_fingerprint(inputs)
    expected = [
        "test_ranking_results.csv",
        "selected_reranking.json",
        "locked_recommendation_evidence_packets.csv",
    ]
    if _guard_output(output_dir, fingerprint, expected):
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    image_weight = float(fusion["image_weight"])
    modality = evaluate_bundle_modalities(bundle, [image_weight], cutoffs)
    rows = [modality]
    clip_weight = float(reranking["clip_weight"])
    for query_index, case in bundle.schedule.iterrows():
        candidates = bundle.candidates[
            bundle.candidates["paper_case_id"] == case["paper_case_id"]
        ].copy()
        if "evidence_score" not in candidates:
            raise ValueError("Test candidate sets require evidence_score for v2 reranking.")
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
        ranked = weighted_rerank(
            candidates,
            clip_weight,
            1.0 - clip_weight,
            normalize_scores=True,
        )
        metrics = ranking_metrics(ranked["is_positive"].astype(int).to_numpy(), cutoffs)
        rows.append(
            pd.DataFrame(
                [
                    {
                        "paper_case_id": case["paper_case_id"],
                        "query_outfit_id": case["query_outfit_id"],
                        "target_category": case["target_category"],
                        "method": "evidence_reranked",
                        "num_candidates": len(ranked),
                        "num_positives": int(ranked["is_positive"].sum()),
                        **metrics,
                    }
                ]
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(
        output_dir / "test_ranking_results.csv", index=False
    )
    shutil.copy2(reranking_selection, output_dir / "selected_reranking.json")
    packets = pd.read_csv(locked_packets)
    if set(packets["paper_case_id"]) != set(bundle.schedule["paper_case_id"]):
        raise ValueError("Locked packets do not match the test schedule case IDs.")
    packets["stage1_packet_protocol"] = "final_eval_v2_selected"
    packets["generation_packet_hash"] = packets.apply(generation_packet_hash, axis=1)
    packet_set_hash = stable_fingerprint(
        packets.sort_values("paper_case_id")[
            ["paper_case_id", "generation_packet_hash"]
        ].to_dict("records")
    )
    packets["stage1_packet_hash"] = packet_set_hash
    packets.to_csv(output_dir / "locked_recommendation_evidence_packets.csv", index=False)
    (output_dir / "stage_manifest.json").write_text(
        json.dumps(
            {
                "input_fingerprint": fingerprint,
                "outputs": expected,
                "stage1_packet_hash": packet_set_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def compare_locked_artifact_outputs(
    *,
    legacy_packets: Path,
    v2_packets: Path,
    output_dir: Path,
) -> dict[str, Any]:
    fingerprint = stable_fingerprint(
        {
            "legacy": file_fingerprint(legacy_packets),
            "v2": file_fingerprint(v2_packets),
            "comparison_schema_version": 2,
        }
    )
    expected = ["packet_comparison.csv", "decision.json"]
    if _guard_output(output_dir, fingerprint, expected):
        return json.loads((output_dir / "decision.json").read_text(encoding="utf-8"))
    comparison, decision = compare_locked_packets(
        pd.read_csv(legacy_packets),
        pd.read_csv(v2_packets),
        policy=MaterialChangePolicy(),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    comparison.to_csv(output_dir / "packet_comparison.csv", index=False)
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    (output_dir / "stage_manifest.json").write_text(
        json.dumps({"input_fingerprint": fingerprint, "outputs": expected}, indent=2),
        encoding="utf-8",
    )
    return decision
