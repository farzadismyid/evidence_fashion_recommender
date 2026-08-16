"""Freeze Stage 1 only after its pinned-data and taxonomy gates pass."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from evidence_fashion.manifest import (
    configuration_hash,
    load_resolved_configuration,
    sha256_file,
    utc_timestamp,
    write_new_json,
)

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs/experiment.yaml"
MODELS_PATH = ROOT / "configs/models.yaml"
OUTPUT_PATH = ROOT / "artifacts/manifests/stage1_taxonomy_freeze_manifest.json"


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data_manifest_path = ROOT / config["paths"]["active_data_manifest"]
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    current_hash = configuration_hash(load_resolved_configuration(CONFIG_PATH, MODELS_PATH))
    if data_manifest["configuration_hash"] != current_hash:
        raise ValueError("Stage 1 data manifest does not match the current resolved configuration.")
    expected_cases = {category: 200 for category in config["preprocessing"]["target_categories"]}
    if data_manifest["validation"]["case_counts"] != expected_cases:
        raise ValueError("Stage 1 does not contain exactly 200 cases per target category.")
    if data_manifest["row_counts"].get("bag_audit_cases") != 200:
        raise ValueError("Stage 1 requires exactly 200 deterministic bag audit cases.")
    if data_manifest["validation"]["cross_split_exact_image_groups"] != 0:
        raise ValueError("Stage 1 has cross-split exact-image leakage.")
    bag_outputs = [
        path
        for path in data_manifest["output_artifact_hashes"]
        if path.endswith("bag_audit_cases.jsonl")
    ]
    if len(bag_outputs) != 1:
        raise ValueError("Stage 1 data manifest must bind one bag audit-case artifact.")

    taxonomy = config["preprocessing"]["category_taxonomy"]
    manifest = {
        "schema_version": 1,
        "stage": 1,
        "status": "frozen",
        "timestamp_utc": utc_timestamp(),
        "experimental_condition_results_inspected": False,
        "configuration_hash": current_hash,
        "dataset_revision": config["dataset"]["revision"],
        "dataset_fingerprint": config["dataset"]["fingerprint"],
        "taxonomy_version": config["preprocessing"]["category_mapping_version"],
        "target_categories": config["preprocessing"]["target_categories"],
        "query_categories": config["preprocessing"]["query_categories"],
        "bag_allowlist": taxonomy["bag_allowlist"],
        "bag_excluded_categories": taxonomy["bag_excluded_categories"],
        "exact_outfit_counts": config["splits"]["exact_outfit_counts"],
        "expected_counts": config["dataset"]["expected_counts"],
        "case_counts": data_manifest["validation"]["case_counts"],
        "bag_audit_case_count": data_manifest["row_counts"]["bag_audit_cases"],
        "validation": {
            "split_outfit_overlap": data_manifest["validation"]["split_outfit_overlap"],
            "duplicate_item_ids": data_manifest["validation"]["duplicate_item_ids"],
            "cross_split_exact_image_groups": data_manifest["validation"][
                "cross_split_exact_image_groups"
            ],
        },
        "bound_artifact_hashes": data_manifest["output_artifact_hashes"]
        | {str(data_manifest_path.relative_to(ROOT)): sha256_file(data_manifest_path)},
        "next_gate": "Stage 2 bag-case applicability must pass before Stage 2 can be frozen.",
    }
    write_new_json(OUTPUT_PATH, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
