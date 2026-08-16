"""Prepare pinned Polyvore metadata and deterministic controlled evaluation cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from evidence_fashion.data import (
    attach_candidate_pools,
    build_category_audit,
    build_evaluation_cases,
    load_pinned_split,
    prepare_metadata,
    resolve_exact_image_duplicate_splits,
    validate_prepared_data,
    write_jsonl,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_rows(config: dict[str, Any]):
    dataset = config["dataset"]
    columns = dataset["columns"]
    metadata_columns = [columns["category"], columns["text"], columns["item_id"]]
    split, fingerprint = load_pinned_split(config)
    metadata = split.select_columns(metadata_columns)
    image_column = columns["image"]
    images = split.data.column(image_column).to_pylist()
    image_hashes = [hashlib.sha256(entry["bytes"]).hexdigest() for entry in images]
    return metadata, image_hashes, fingerprint


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    config_digest = configuration_hash(resolved)
    runtime_root = args.runtime_root or Path(config["paths"]["runtime_root"])
    run_id = f"data-{config_digest[:12]}"
    run_dir = runtime_root / "data" / run_id
    items_path = run_dir / "prepared_items.parquet"
    cases_path = run_dir / "evaluation_cases.jsonl"
    bag_audit_cases_path = run_dir / "bag_audit_cases.jsonl"
    runtime_manifest_path = run_dir / "manifest.json"
    tracked_manifest_path = Path(config["paths"]["active_data_manifest"])

    if args.dry_run:
        print(
            json.dumps(
                {
                    "configuration_hash": config_digest,
                    "dataset_revision": config["dataset"]["revision"],
                    "run_dir": str(run_dir),
                    "would_call_model": False,
                },
                indent=2,
            )
        )
        return
    if args.resume and runtime_manifest_path.exists():
        print(runtime_manifest_path.read_text(encoding="utf-8"))
        return
    if run_dir.exists():
        raise FileExistsError(f"Immutable run directory already exists: {run_dir}")

    rows, image_hashes, observed_fingerprint = load_rows(config)
    audit = build_category_audit(rows, config)
    audit_path = Path(config["paths"]["category_audit_table"])
    audit.to_csv(audit_path, index=False)
    frame = prepare_metadata(rows, config, exact_image_hashes=image_hashes)
    frame, duplicate_resolution = resolve_exact_image_duplicate_splits(frame, config)
    cases = attach_candidate_pools(frame, build_evaluation_cases(frame, config), config)
    validation = validate_prepared_data(frame, cases, config, duplicate_resolution)
    run_dir.mkdir(parents=True)
    frame.to_parquet(items_path, index=False)
    write_jsonl(cases_path, cases.to_dict("records"))
    bag_audit_cases = cases.loc[cases["target_category"].eq("bags")].copy()
    expected_bag_cases = config["recommendation_evaluation"]["cases_per_category"]
    if len(bag_audit_cases) != expected_bag_cases:
        raise ValueError(
            f"Stage 1 requires {expected_bag_cases} deterministic bag audit cases; "
            f"found {len(bag_audit_cases)}."
        )
    write_jsonl(bag_audit_cases_path, bag_audit_cases.to_dict("records"))
    manifest = {
        "schema_version": 1,
        "stage": 2,
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": config_digest,
        "resolved_configuration": resolved,
        "input_artifact_hashes": {
            "dataset_revision": config["dataset"]["revision"],
            "dataset_fingerprint": observed_fingerprint,
        },
        "output_artifact_hashes": {
            str(items_path): sha256_file(items_path),
            str(cases_path): sha256_file(cases_path),
            str(bag_audit_cases_path): sha256_file(bag_audit_cases_path),
            str(audit_path): sha256_file(audit_path),
        },
        "models": {},
        "row_counts": validation.counts
        | {
            "evaluation_cases": len(cases),
            "bag_audit_cases": len(bag_audit_cases),
            "candidate_rows": int(cases["candidate_pool_size"].sum()),
        },
        "failure_counts": {"invalid_item_ids": validation.invalid_item_ids, "invalid_pools": 0},
        "seed": config["project"]["random_seed"],
        "environment": environment_summary(),
        "command": "python scripts/prepare_data.py --config configs/experiment.yaml --validate-only"
        if args.validate_only
        else "python scripts/prepare_data.py --config configs/experiment.yaml",
        "validation": {
            "category_audit": {
                "raw_categories": len(audit),
                "kept_categories": int(audit["decision"].eq("keep").sum()),
                "excluded_categories": int(audit["decision"].eq("exclude").sum()),
                "review_categories": int(audit["decision"].eq("review").sum()),
                "raw_items": int(audit["item_count"].sum()),
                "kept_items": int(
                    audit.loc[audit["decision"].eq("keep"), "item_count"].sum()
                ),
            },
            "case_counts": validation.case_counts,
            "candidate_pool_min_size": validation.candidate_pool_min_size,
            "candidate_pool_max_size": validation.candidate_pool_max_size,
            "split_outfit_overlap": validation.split_outfit_overlap,
            "duplicate_item_ids": validation.duplicate_item_ids,
            "exact_image_duplicate_groups": validation.exact_image_duplicate_groups,
            "cross_split_exact_image_groups": validation.cross_split_exact_image_groups,
            "duplicate_resolution": duplicate_resolution.__dict__,
        },
    }
    write_new_json(runtime_manifest_path, manifest)
    write_json(tracked_manifest_path, manifest)
    print(json.dumps(manifest["validation"] | manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
