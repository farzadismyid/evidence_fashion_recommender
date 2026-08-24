"""Assemble the immutable Stage-1 preflight manifest from completed checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evidence_fashion.data import build_candidate_pool
from evidence_fashion.grounding_contracts import rule_applicability_gate
from evidence_fashion.kb_audit import coverage_matrix, load_canonical_rules
from evidence_fashion.manifest import (
    environment_summary,
    git_commit,
    sha256_file,
    utc_timestamp,
    write_new_json,
)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _manifest_output(manifest: dict[str, Any], suffix: str) -> Path:
    return Path(next(path for path in manifest["output_artifact_hashes"] if path.endswith(suffix)))


def _kb_audit(rules: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    settings = config["kb_validation"]
    target_counts = rules["recommended_category"].value_counts().sort_index().to_dict()
    expected_targets = {
        category: int(settings["expected_rules_per_category"])
        for category in config["preprocessing"]["target_categories"]
    }
    pairs = []
    threshold = float(settings["near_duplicate_sequence_similarity_threshold"])
    texts = rules["rule_text"].astype(str).tolist()
    ids = rules["rule_id"].astype(str).tolist()
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            similarity = SequenceMatcher(None, texts[left].lower(), texts[right].lower()).ratio()
            if similarity >= threshold:
                pairs.append(
                    {
                        "left_rule_id": ids[left],
                        "right_rule_id": ids[right],
                        "similarity": similarity,
                    }
                )
    return {
        "rule_count": len(rules),
        "target_counts": target_counts,
        "unique_nonempty_rule_ids": bool(
            rules["rule_id"].astype(str).str.strip().ne("").all()
            and rules["rule_id"].is_unique
        ),
        "exact_duplicate_rules": int(rules["rule_text"].astype(str).duplicated().sum()),
        "near_duplicate_threshold": threshold,
        "near_duplicate_pairs": pairs,
        "coverage": coverage_matrix(rules).to_dict(),
        "source_type_distribution": rules["source_type"].value_counts().sort_index().to_dict(),
        "provenance_complete": bool(
            rules[
                [
                    "source_url_or_reference",
                    "source_title",
                    "source_author_or_org",
                    "source_year",
                    "source_access_date",
                    "source_validation_status",
                ]
            ]
            .astype(str)
            .apply(lambda column: column.str.strip().ne(""))
            .all()
            .all()
        ),
        "pass": (
            len(rules) == int(settings["expected_rule_count"])
            and target_counts == expected_targets
            and not pairs
        ),
    }


def _structural_feasibility(
    items: pd.DataFrame, cases: list[dict[str, Any]], rules: pd.DataFrame, config: dict[str, Any]
) -> dict[str, int]:
    counts = {category: 0 for category in config["preprocessing"]["target_categories"]}
    for case in cases:
        target_rules = rules[rules["recommended_category"].eq(case["target_category"])]
        pool = build_candidate_pool(items, case, config)
        eligible = False
        for candidate in pool.to_dict("records"):
            if any(
                rule_applicability_gate(rule, case=case, candidate=candidate).established
                for rule in target_rules.to_dict("records")
            ):
                eligible = True
                break
        if eligible:
            counts[str(case["target_category"])] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts-config", type=Path, default=Path("configs/prompts.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    prompts = yaml.safe_load(args.prompts_config.read_text(encoding="utf-8"))
    resolved = {"experiment": config, "models": models, "prompts": prompts}
    resolved_hash = _json_hash(resolved)
    kb_path = Path(config["paths"]["knowledge_base"])
    rules = load_canonical_rules(kb_path)
    kb_audit = _kb_audit(rules, config)
    data_manifest_path = Path(config["paths"]["active_data_manifest"])
    embedding_manifest_path = Path(config["paths"]["active_embedding_manifest"])
    data_manifest = json.loads(data_manifest_path.read_text("utf-8"))
    embedding_manifest = json.loads(embedding_manifest_path.read_text("utf-8"))
    items_path = _manifest_output(data_manifest, "prepared_items.parquet")
    cases_path = _manifest_output(data_manifest, "evaluation_cases.jsonl")
    items = pd.read_parquet(items_path)
    cases = [json.loads(line) for line in cases_path.read_text("utf-8").splitlines()]
    feasible = _structural_feasibility(items, cases, rules, config)
    analysis_path = Path(config["paths"]["final_analysis_runs"])
    sensitivity_manifest_path = analysis_path / "stage1_validation" / "manifest.json"
    integrity_manifest_path = analysis_path / "stage1_integrity" / "manifest.json"
    sensitivity = json.loads(sensitivity_manifest_path.read_text("utf-8"))
    integrity = json.loads(integrity_manifest_path.read_text("utf-8"))
    defaults = {
        "image_weight": config["retrieval"]["fusion"]["image_weight"],
        "text_weight": config["retrieval"]["fusion"]["text_weight"],
        "clip_weight": config["reranking"]["clip_weight"],
        "evidence_weight": config["reranking"]["evidence_weight"],
        "rule_top_k": config["rule_retrieval"]["rule_top_k"],
    }
    fixed_defaults_pass = defaults == {
        "image_weight": 0.4,
        "text_weight": 0.6,
        "clip_weight": 0.75,
        "evidence_weight": 0.25,
        "rule_top_k": 5,
    }
    feasibility_pass = all(
        value >= config["explanations"]["cases_per_category"] for value in feasible.values()
    )
    sensitivity_pass = (
        sensitivity["row_counts"]["fusion_configurations"]
        == len(config["retrieval"]["fusion"]["validation_grid_image_weights"])
        and sensitivity["row_counts"]["reranking_configurations"]
        == len(config["validation_sensitivity"]["evidence_weights"])
        * len(config["validation_sensitivity"]["rule_top_k_values"])
        and not any(sensitivity["failure_counts"].values())
    )
    integrity_pass = all(integrity["checks"].values()) and not any(
        integrity["failure_counts"].values()
    )
    preflight_pass = kb_audit["pass"] and fixed_defaults_pass and feasibility_pass
    preflight_pass = preflight_pass and sensitivity_pass and integrity_pass
    output = Path(config["paths"]["stage1_manifest"])
    manifest = {
        "schema_version": 2,
        "stage": 1,
        "stage_name": "preflight_clean_reset_and_final_freeze",
        "status": "passed" if preflight_pass else "failed",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "resolved_configuration_hash": resolved_hash,
        "configuration_file_hashes": {
            str(args.config): sha256_file(args.config),
            str(args.models_config): sha256_file(args.models_config),
            str(args.prompts_config): sha256_file(args.prompts_config),
        },
        "input_artifact_hashes": {
            str(kb_path): sha256_file(kb_path),
            str(data_manifest_path): sha256_file(data_manifest_path),
            str(embedding_manifest_path): sha256_file(embedding_manifest_path),
            str(sensitivity_manifest_path): sha256_file(sensitivity_manifest_path),
            str(integrity_manifest_path): sha256_file(integrity_manifest_path),
        },
        "output_artifact_hashes": {
            **sensitivity["output_artifact_hashes"],
            **integrity["output_artifact_hashes"],
        },
        "models": {
            "embedders": models["embedders"],
            "extractor": models["extractor"],
            "verifier": models["verifier"],
        },
        "row_counts": {
            "final_kb_rules": len(rules),
            "prepared_items": len(items),
            "test_cases": len(cases),
            "validation_cases": sensitivity["row_counts"]["validation_cases"],
            "fusion_grid_configurations": sensitivity["row_counts"]["fusion_configurations"],
            "reranking_grid_configurations": sensitivity["row_counts"]["reranking_configurations"],
            "synthetic_integrity_claims": integrity["row_counts"]["extracted_claims"],
        },
        "failure_counts": {
            "data": sum(data_manifest["failure_counts"].values()),
            "embeddings": sum(embedding_manifest["failure_counts"].values()),
            "sensitivity": sum(sensitivity["failure_counts"].values()),
            "synthetic_integrity": sum(integrity["failure_counts"].values()),
        },
        "seeds": {
            "project": config["project"]["random_seed"],
            "split": config["splits"]["seed"],
            "test_cases": config["recommendation_evaluation"]["case_seed"],
            "validation_cases": config["validation_sensitivity"]["case_seed"],
        },
        "frozen_defaults": defaults,
        "kb_audit": kb_audit,
        "structural_explanation_feasibility_by_category": feasible,
        "integrity_check": integrity["checks"],
        "exact_trace_invariant_tests": "tests/test_final_trace_invariants.py",
        "canonical_runtime_state": {
            "root": config["paths"]["runtime_root"],
            "stage_paths": [
                config["paths"][key]
                for key in (
                    "data_runs",
                    "embedding_runs",
                    "recommendation_runs",
                    "explanation_runs",
                    "extraction_runs",
                    "verification_runs",
                    "final_analysis_runs",
                )
            ],
        },
        "commands": [
            "python scripts/prepare_data.py",
            "python scripts/build_embeddings.py",
            "python scripts/build_final_kb_embeddings.py",
            "python scripts/run_final_validation_sensitivity.py",
            "python scripts/run_final_integrity_check.py",
            "python scripts/finalize_stage1_preflight.py",
            "ruff check .",
            "pytest -q",
        ],
        "environment": environment_summary(),
    }
    if not feasibility_pass:
        raise ValueError(f"Insufficient structurally evidence-eligible pools: {feasible}")
    if not preflight_pass:
        raise ValueError("One or more Stage-1 preflight gates failed.")
    write_new_json(output, manifest)
    print(json.dumps({"status": manifest["status"], **manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
