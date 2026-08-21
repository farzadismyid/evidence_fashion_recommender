"""Rebuild reviewer-facing KB audit tables without reading experiment outputs."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import yaml

from evidence_fashion.kb_audit import coverage_matrix, load_canonical_rules, load_legacy_audit

ROOT = Path(__file__).parents[1]
LEGACY_AUDIT = ROOT / "data/kb/legacy_kb_audit.yaml"
EXPERIMENT_CONFIG = ROOT / "configs/experiment.yaml"
LEGACY_RULE_AUDIT = ROOT / "data/kb/legacy_rule_audit.csv"
COVERAGE_MATRIX = ROOT / "data/kb/coverage_matrix.csv"
SOURCE_REGISTRY = ROOT / "data/kb/kb_source_registry.csv"
SIMILARITY_AUDIT = ROOT / "data/kb/kb_rule_similarity_audit.csv"


def main() -> None:
    experiment = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    canonical_kb = ROOT / experiment["paths"]["knowledge_base"]
    audit = load_legacy_audit(LEGACY_AUDIT)
    legacy_path = ROOT / audit["audited_asset"]["path"]
    legacy = pd.read_csv(legacy_path, dtype=str, keep_default_na=False)
    canonical = load_canonical_rules(canonical_kb)

    decisions = {
        rule_id: decision
        for decision, rule_ids in audit["decisions"].items()
        for rule_id in rule_ids
    }
    successors: dict[str, list[str]] = {}
    for _, rule in canonical.iterrows():
        for old_id in str(rule["supersedes_rule_ids"]).split("|"):
            if old_id:
                successors.setdefault(old_id, []).append(str(rule["rule_id"]))
    notes = audit["decision_policy"]

    output = legacy.copy()
    output.insert(1, "audit_decision", output["rule_id"].map(decisions))
    output.insert(
        2,
        "replacement_rule_ids",
        output["rule_id"].map(lambda value: "|".join(successors.get(value, []))),
    )
    output.insert(3, "audit_note", output["audit_decision"].map(notes))
    output.insert(4, "experimental_results_inspected", "false")
    if output["audit_decision"].isna().any() or len(output) != 126:
        raise ValueError("The row-level artifact must account for exactly 126 legacy rules.")
    output.to_csv(LEGACY_RULE_AUDIT, index=False, lineterminator="\n")

    matrix = coverage_matrix(canonical)
    matrix.index.name = "query_category"
    matrix.to_csv(COVERAGE_MATRIX, lineterminator="\n")

    source_fields = [
        "source_url_or_reference",
        "source_title",
        "source_author_or_org",
        "source_year",
        "source_access_date",
        "source_validation_status",
    ]
    consistency = canonical.groupby("source_url_or_reference")[source_fields[1:]].nunique()
    if (consistency > 1).any().any():
        raise ValueError("A source URL has inconsistent provenance metadata across rules.")
    registry_rows = []
    for source_url, group in canonical.groupby("source_url_or_reference", sort=True):
        first = group.iloc[0]
        registry_rows.append(
            {
                "source_url_or_reference": source_url,
                "source_title": first["source_title"],
                "source_author_or_org": first["source_author_or_org"],
                "source_year": first["source_year"],
                "source_access_date": first["source_access_date"],
                "source_validation_status": first["source_validation_status"],
                "rule_count": len(group),
                "rule_ids": "|".join(sorted(group["rule_id"])),
                "distinct_locators": group["source_locator"].nunique(),
                "reliability_labels": "|".join(sorted(set(group["source_reliability"]))),
            }
        )
    pd.DataFrame(registry_rows).to_csv(SOURCE_REGISTRY, index=False, lineterminator="\n")

    similarity_rows = []
    for left_index in range(len(canonical)):
        left = canonical.iloc[left_index]
        for right_index in range(left_index + 1, len(canonical)):
            right = canonical.iloc[right_index]
            similarity = SequenceMatcher(
                None, str(left["rule_text"]).lower(), str(right["rule_text"]).lower()
            ).ratio()
            if similarity < 0.72:
                continue
            same_target = left["recommended_category"] == right["recommended_category"]
            same_query = left["applicable_query_categories"] == right[
                "applicable_query_categories"
            ]
            decision = (
                "retain_distinct_target"
                if not same_target
                else "retain_distinct_query_or_context"
            )
            similarity_rows.append(
                {
                    "left_rule_id": left["rule_id"],
                    "right_rule_id": right["rule_id"],
                    "sequence_similarity": round(similarity, 6),
                    "same_target": str(same_target).lower(),
                    "same_query_applicability": str(same_query).lower(),
                    "same_source_page": str(
                        left["source_url_or_reference"] == right["source_url_or_reference"]
                    ).lower(),
                    "audit_decision": decision,
                }
            )
    pd.DataFrame(similarity_rows).to_csv(SIMILARITY_AUDIT, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
