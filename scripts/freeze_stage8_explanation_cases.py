"""Freeze the 500 locked, trace-valid explanation cases for Stage 8."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from evidence_fashion.explanation import common_context
from evidence_fashion.grounding_contracts import require_trace_applicability
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


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--runtime-root", type=Path, default=Path(".runtime/current/explanations"))
    return parser.parse_args()


def _bound_output(manifest: Mapping[str, Any], suffix: str) -> tuple[Path, str]:
    matches = [
        (Path(path), str(digest))
        for path, digest in manifest["output_artifact_hashes"].items()
        if path.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Stage 7 manifest must bind exactly one {suffix} output.")
    path, digest = matches[0]
    if not path.exists() or sha256_file(path) != digest:
        raise ValueError(f"Hash-mismatched Stage 7 source artifact: {path}")
    return path, digest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _trace_score(trace: Mapping[str, Any]) -> float:
    values = np.asarray([rule["weighted_contribution"] for rule in trace["rules"]], dtype=float)
    return float(0.7 * values.max() + 0.3 * values.mean())


def _valid_trace(case: Mapping[str, Any], rule_count: int) -> bool:
    trace = case.get("evidence_trace", {})
    try:
        require_trace_applicability(trace)
    except ValueError:
        return False
    return (
        trace.get("candidate_id") == case.get("locked_candidate_id")
        and len(trace["rules"]) == rule_count
        and math.isclose(_trace_score(trace), float(trace["evidence_score"]), abs_tol=1e-12)
    )


def _select(
    locked: Sequence[Mapping[str, Any]], config: Mapping[str, Any], categories: Mapping[str, str]
) -> list[dict[str, Any]]:
    settings = config["stage7"]
    seed, quota = int(settings["case_selection_seed"]), int(settings["cases_per_category"])
    selected: list[dict[str, Any]] = []
    for target in config["preprocessing"]["target_categories"]:
        eligible = [
            dict(case)
            for case in locked
            if case["target_category"] == target
            and _valid_trace(case, int(config["stage4_validation"]["selected_rule_top_k"]))
            and (
                target != "bags"
                or categories.get(str(case["locked_candidate_id"]))
                in config["preprocessing"]["category_taxonomy"]["bag_allowlist"]
            )
        ]
        eligible.sort(
            key=lambda case: (
                hashlib.sha256(f"{seed}:{case['case_id']}".encode()).hexdigest(),
                case["case_id"],
            )
        )
        if len(eligible) < quota:
            raise ValueError(
                f"Only {len(eligible)} trace-valid locked cases for {target}; need {quota}."
            )
        selected.extend(eligible[:quota])
    if len(selected) != int(config["explanations"]["case_count"]) or len(
        {case["case_id"] for case in selected}
    ) != len(selected):
        raise ValueError("Stage 8 selection is incomplete or contains duplicate case IDs.")
    return selected


def _condition_records(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in selected:
        context = common_context(case)
        trace = case["evidence_trace"]
        for condition in ("no_rag", "rule_rag"):
            records.append(
                {
                    "calibration_case_id": case["case_id"],
                    "condition": condition,
                    "target_category": case["target_category"],
                    "locked_candidate_id": case["locked_candidate_id"],
                    "A_common_context": context,
                    "A_sha256": canonical_hash(context),
                    "B_exact_stored_trace": trace,
                    "B_sha256": canonical_hash(trace),
                    "stage7_locked_record_sha256": canonical_hash(case),
                }
            )
    return records


def _write_jsonl_new(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    digest = configuration_hash(resolved)
    source_manifest_path = Path("artifacts/manifests/stage6_recommendation_manifest.json")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    locked_path, locked_hash = _bound_output(source_manifest, "locked_cases.jsonl")
    prepared_path, prepared_hash = _bound_output(
        json.loads(Path(config["paths"]["active_data_manifest"]).read_text(encoding="utf-8")),
        "prepared_items.parquet",
    )
    items = pd.read_parquet(prepared_path, columns=["item_id", "category"])
    categories = dict(zip(items["item_id"].astype(str), items["category"].astype(str), strict=True))
    selected = _select(_read_jsonl(locked_path), config, categories)
    records = _condition_records(selected)
    if len(records) != 2 * len(selected):
        raise ValueError("Condition matrix is incomplete.")
    for case_id in {row["calibration_case_id"] for row in records}:
        pair = [row for row in records if row["calibration_case_id"] == case_id]
        if (
            len(pair) != 2
            or pair[0]["A_sha256"] != pair[1]["A_sha256"]
            or pair[0]["B_sha256"] != pair[1]["B_sha256"]
        ):
            raise ValueError(f"Conditions do not share a locked context for {case_id}.")
    run_id = f"stage8-selection-{digest[:12]}"
    run_dir = args.runtime_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Immutable Stage 8 selection already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    selected_path, records_path = (
        run_dir / "selected_locked_cases.jsonl",
        run_dir / "condition_inputs.jsonl",
    )
    _write_jsonl_new(selected_path, selected)
    _write_jsonl_new(records_path, records)
    manifest = {
        "schema_version": 1,
        "stage": 8,
        "status": "frozen",
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": digest,
        "resolved_configuration": resolved,
        "input_artifact_hashes": {str(locked_path): locked_hash, str(prepared_path): prepared_hash},
        "output_artifact_hashes": {
            str(selected_path): sha256_file(selected_path),
            str(records_path): sha256_file(records_path),
        },
        "row_counts": {"selected_cases": len(selected), "condition_inputs": len(records)},
        "validation": {
            "unique_case_ids": True,
            "per_category": {
                c: sum(row["target_category"] == c for row in selected)
                for c in config["preprocessing"]["target_categories"]
            },
            "trace_valid": True,
            "condition_contexts_identical": True,
            "bag_allowlist": True,
        },
        "environment": environment_summary(),
        "command": (
            "python scripts/freeze_stage8_explanation_cases.py --config configs/experiment.yaml "
            "--models-config configs/models.yaml --runtime-root .runtime/current/explanations"
        ),
    }
    write_new_json(run_dir / "manifest.json", manifest)
    write_json(
        Path("artifacts/manifests/stage8_explanation_case_selection_manifest.json"), manifest
    )
    print(
        json.dumps(
            {"run_id": run_id, "selected_cases": len(selected), "condition_inputs": len(records)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
