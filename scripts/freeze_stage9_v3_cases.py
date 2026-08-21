"""Freeze the V3-KB Stage 9 case matrix from existing locked recommendations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from evidence_fashion.explanation import common_context
from evidence_fashion.grounding_contracts import require_trace_applicability
from evidence_fashion.kb_audit import load_audited_rules
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
from evidence_fashion.retrieval import OllamaEmbedder
from evidence_fashion.rule_retrieval import RuleRetriever, candidate_rule_representation

SELECTION_SEED = 42


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")


def _bound_output(manifest: Mapping[str, Any], suffix: str) -> tuple[Path, str]:
    matches = [
        (Path(path), str(digest))
        for path, digest in manifest["output_artifact_hashes"].items()
        if path.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Locked recommendation manifest must bind exactly one {suffix} file.")
    path, digest = matches[0]
    if not path.exists() or sha256_file(path) != digest:
        raise ValueError(f"Hash-mismatched locked recommendation input: {path}")
    return path, digest


def _case_and_candidate(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    query_group = str(row["evidence_trace"]["query_group"])
    case = {
        "query_category": query_group,
        "query_group": query_group,
        "query_text": str(row["query_item_minimal_name"]),
        "outfit_context_text": "",
        "user_request": str(row["request"]),
        "applicability_contexts": [],
        "target_category": str(row["target_category"]),
    }
    candidate = {
        "item_id": str(row["locked_candidate_id"]),
        "category": str(row["target_category"]),
        "text": str(row["locked_candidate_minimal_name"]),
    }
    return case, candidate


def _selection_key(case_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{case_id}".encode()).hexdigest()


def _select(
    records: Sequence[Mapping[str, Any]], categories: Sequence[str]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category in categories:
        eligible = [
            dict(row)
            for row in records
            if row["target_category"] == category and row["evidence_trace"]["rules"]
        ]
        # Prefer genuine multi-rule packets first, then use a seeded stable order.
        eligible.sort(
            key=lambda row: (
                -int(len(row["evidence_trace"]["rules"]) >= 2),
                _selection_key(row["case_id"]),
            )
        )
        if len(eligible) < 100:
            raise ValueError(
                f"V3 has only {len(eligible)} strict-trace cases for {category}; need 100."
            )
        selected.extend(eligible[:100])
    if len(selected) != 500 or len({row["case_id"] for row in selected}) != 500:
        raise ValueError("V3 Stage 9 selection is incomplete or duplicated.")
    for row in selected:
        require_trace_applicability(row["evidence_trace"])
    return selected


def _condition_inputs(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in selected:
        context = common_context(case)
        trace = case["evidence_trace"]
        for condition in ("no_rag", "rule_rag"):
            rows.append(
                {
                    "calibration_case_id": case["case_id"],
                    "condition": condition,
                    "target_category": case["target_category"],
                    "locked_candidate_id": case["locked_candidate_id"],
                    "A_common_context": context,
                    "B_exact_stored_trace": trace,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--runtime-root", type=Path, default=Path(".runtime/current/explanations"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    digest = configuration_hash(resolved)
    source_manifest = json.loads(
        Path("artifacts/manifests/stage6_recommendation_manifest.json").read_text(encoding="utf-8")
    )
    locked_path, locked_hash = _bound_output(source_manifest, "locked_cases.jsonl")
    locked = _read_jsonl(locked_path)
    if len(locked) != 1000 or len({row["case_id"] for row in locked}) != 1000:
        raise ValueError(
            "The locked recommendation source must contain exactly 1,000 unique cases."
        )
    rules = load_audited_rules(config)
    embedder = OllamaEmbedder(models["embedders"]["qwen3_embedding"])
    retriever = RuleRetriever(
        rules,
        embedder.encode(rules["rule_text"].astype(str).tolist()),
        config["rule_retrieval"],
    )
    pairs = [_case_and_candidate(row) for row in locked]
    vectors = embedder.encode([candidate_rule_representation(*pair) for pair in pairs])
    records: list[dict[str, Any]] = []
    for row, (case, candidate), vector in zip(locked, pairs, vectors, strict=True):
        trace = retriever.retrieve_and_score(
            case=case, candidate=candidate, representation_embedding=vector
        ).to_dict()
        records.append({**row, "evidence_trace": trace})
    selected = _select(records, config["preprocessing"]["target_categories"])
    inputs = _condition_inputs(selected)
    run_id = f"stage9-v3-selection-{digest[:12]}"
    run_dir = args.runtime_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Immutable V3 selection already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    selected_path = run_dir / "selected_locked_cases.jsonl"
    inputs_path = run_dir / "condition_inputs.jsonl"
    _write_jsonl_new(selected_path, selected)
    _write_jsonl_new(inputs_path, inputs)
    sizes = Counter(len(row["evidence_trace"]["rules"]) for row in selected)
    manifest = {
        "schema_version": 1,
        "stage": 9,
        "status": "frozen_selection",
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": digest,
        "selection_seed": SELECTION_SEED,
        "selection_policy": "prefer_trace_size_at_least_2_then_sha256_seeded_case_order",
        "resolved_configuration": resolved,
        "input_artifact_hashes": {
            str(locked_path): locked_hash,
            config["paths"]["knowledge_base"]: sha256_file(Path(config["paths"]["knowledge_base"])),
        },
        "output_artifact_hashes": {
            str(selected_path): sha256_file(selected_path),
            str(inputs_path): sha256_file(inputs_path),
        },
        "row_counts": {"selected_cases": 500, "condition_inputs": 1000},
        "trace_size_distribution": {str(size): sizes[size] for size in sorted(sizes)},
        "per_target_category": {
            category: sum(row["target_category"] == category for row in selected)
            for category in config["preprocessing"]["target_categories"]
        },
        "validation": {
            "all_traces_nonempty_and_antecedent_applicable": True,
            "conditions_share_locked_case_and_recommendation": True,
            "multi_rule_packets_prioritized": True,
        },
        "environment": environment_summary(),
    }
    write_new_json(run_dir / "manifest.json", manifest)
    write_json(Path("artifacts/manifests/stage9_v3_case_selection_manifest.json"), manifest)
    print(
        json.dumps(
            {"run_id": run_id, "trace_size_distribution": manifest["trace_size_distribution"]}
        )
    )


if __name__ == "__main__":
    main()
