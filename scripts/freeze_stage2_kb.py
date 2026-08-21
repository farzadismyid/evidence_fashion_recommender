"""Freeze Stage 2 after provenance, balance, applicability and diversity gates pass."""

from __future__ import annotations

import json
from pathlib import Path

from evidence_fashion.kb_audit import load_canonical_rules
from evidence_fashion.manifest import git_commit, sha256_file, utc_timestamp, write_json

ROOT = Path(__file__).parents[1]
KB_PATH = ROOT / "data/kb/fashion_rules.csv"
AUDIT_PATH = ROOT / "reports/stage2_bag_case_applicability_audit.json"
SOURCE_REGISTRY = ROOT / "data/kb/kb_source_registry.csv"
SIMILARITY_AUDIT = ROOT / "data/kb/kb_rule_similarity_audit.csv"
COVERAGE_MATRIX = ROOT / "data/kb/coverage_matrix.csv"
STAGE1_MANIFEST = ROOT / "artifacts/manifests/stage1_taxonomy_freeze_manifest.json"
OUTPUT_PATH = ROOT / "artifacts/manifests/stage2_kb_freeze_manifest.json"


def main() -> None:
    rules = load_canonical_rules(KB_PATH)
    counts = rules["recommended_category"].value_counts().to_dict()
    minimum = {category: 20 for category in ("tops", "bottoms", "shoes", "outerwear", "bags")}
    if any(int(counts.get(category, 0)) < required for category, required in minimum.items()):
        raise ValueError("Stage 2 requires at least 20 source-audited rules per target category.")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("experimental_condition_results_inspected") is not False:
        raise ValueError("Stage 2 audit must not inspect experimental condition results.")
    if audit.get("stage2_pass") is not True:
        raise ValueError("Stage 2 applicability and diversity gates have not all passed.")
    stage1 = json.loads(STAGE1_MANIFEST.read_text(encoding="utf-8"))
    if stage1.get("status") != "frozen":
        raise ValueError("Stage 1 must be frozen before Stage 2.")

    artifacts = [
        KB_PATH,
        AUDIT_PATH,
        SOURCE_REGISTRY,
        SIMILARITY_AUDIT,
        COVERAGE_MATRIX,
        STAGE1_MANIFEST,
    ]
    manifest = {
        "schema_version": 1,
        "stage": 2,
        "status": "frozen",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "experimental_condition_results_inspected": False,
        "rule_count": len(rules),
        "rule_counts_by_target": counts,
        "source_page_count": int(rules["source_url_or_reference"].nunique()),
        "source_validation_statuses": sorted(set(rules["source_validation_status"])),
        "audit_gates": {
            key: audit[key]
            for key in (
                "coverage_pass",
                "prevalence_pass",
                "duplicate_packet_pass",
                "packet_diversity_pass",
                "stage2_pass",
            )
        },
        "audit_metrics": {
            key: audit[key]
            for key in (
                "case_count",
                "supported_case_count",
                "unsupported_case_count",
                "maximum_rule_prevalence",
                "duplicate_nonempty_packet_cases",
                "duplicate_packet_case_fraction",
                "unique_nonempty_packets",
            )
        },
        "thresholds": audit["thresholds"],
        "bound_artifact_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in artifacts
        },
        "next_gate": "Proceed to Stage 3 without changing the frozen taxonomy or KB.",
    }
    # The canonical Stage 2 manifest is replaced when a newly audited KB supersedes it.
    write_json(OUTPUT_PATH, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
