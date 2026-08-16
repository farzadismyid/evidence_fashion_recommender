"""Fail-closed validation and audit utilities for the five-category KB."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ALLOWED_CATEGORIES = frozenset({"tops", "bottoms", "shoes", "outerwear", "bags"})
ALLOWED_RELIABILITY = frozenset({"high", "medium"})
REQUIRED_KB_COLUMNS = frozenset(
    {
        "rule_id",
        "rule_version",
        "rule_text",
        "input_category",
        "recommended_category",
        "applicable_query_categories",
        "required_context",
        "query_terms",
        "candidate_terms",
        "scenario_label",
        "gender_context",
        "formality_level",
        "occasion_tags",
        "season_tags",
        "source_type",
        "source_title",
        "source_author_or_org",
        "source_year",
        "source_url_or_reference",
        "source_locator",
        "source_access_date",
        "source_validation_status",
        "source_reliability",
        "evidence_summary",
        "rule_scope",
        "rule_limitations",
        "audit_status",
        "supersedes_rule_ids",
    }
)
LEGACY_DECISIONS = frozenset(
    {"outside_taxonomy", "legacy_accessory_ontology", "citation_overreach", "retained_or_rewritten"}
)


def _pipe_values(value: object) -> set[str]:
    return {part.strip() for part in str(value).split("|") if part.strip()}


def declared_values(value: object) -> set[str]:
    return {part.strip().lower() for part in str(value).split("|") if part.strip()}


def matches_declared_terms(value: object, text: str) -> bool:
    """Apply the KB's AND-of-OR term declaration to permitted case text."""
    declaration = str(value).strip().lower()
    if declaration == "none":
        return True
    normalized = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "
    clauses = [clause.strip() for clause in declaration.split("&") if clause.strip()]
    return bool(clauses) and all(
        any(
            f" {re.sub(r'[^a-z0-9]+', ' ', term).strip()} " in normalized
            for term in clause.split("|")
            if term.strip()
        )
        for clause in clauses
    )


def validate_canonical_rules(rules: pd.DataFrame) -> None:
    """Validate the evidence, provenance and applicability contract in one place."""
    missing = REQUIRED_KB_COLUMNS.difference(rules.columns)
    if missing:
        raise ValueError(f"Knowledge base is missing fields: {sorted(missing)}")
    if rules.empty:
        raise ValueError("Knowledge base must contain at least one rule.")
    if rules["rule_id"].duplicated().any() or rules["rule_id"].eq("").any():
        raise ValueError("Knowledge-base rule IDs must be non-empty and unique.")
    normalized_text = rules["rule_text"].astype(str).map(
        lambda value: re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    )
    if normalized_text.duplicated().any():
        raise ValueError("Knowledge base contains duplicate normalized rule text.")

    required_nonempty = REQUIRED_KB_COLUMNS.difference({"supersedes_rule_ids"})
    blank = {
        column: rules.index[rules[column].astype(str).str.strip().eq("")].tolist()
        for column in required_nonempty
        if rules[column].astype(str).str.strip().eq("").any()
    }
    if blank:
        raise ValueError(f"Knowledge base contains blank required fields: {blank}")

    for field in ("input_category", "recommended_category"):
        invalid = sorted(set(rules[field].astype(str)) - ALLOWED_CATEGORIES)
        if invalid:
            raise ValueError(f"{field} contains categories outside the frozen taxonomy: {invalid}")
    if not rules["audit_status"].eq("retain").all():
        raise ValueError("The canonical KB may contain only citation-audited retained rules.")
    invalid_reliability = sorted(
        set(rules["source_reliability"].astype(str).str.lower()) - ALLOWED_RELIABILITY
    )
    if invalid_reliability:
        raise ValueError(f"Unsupported source reliability labels: {invalid_reliability}")
    if not rules["source_url_or_reference"].str.startswith("https://").all():
        raise ValueError("Every retained rule must use an HTTPS source reference.")
    if not rules["source_access_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("Every retained rule must use an ISO source access date.")
    if not rules["source_validation_status"].eq(
        "verified_reachable_and_direct_2026-08-16"
    ).all():
        raise ValueError(
            "Every retained rule must have a current reachable-and-direct source audit."
        )
    source_metadata = [
        "source_title",
        "source_author_or_org",
        "source_year",
        "source_access_date",
        "source_validation_status",
    ]
    if (
        rules.groupby("source_url_or_reference")[source_metadata]
        .nunique()
        .gt(1)
        .any()
        .any()
    ):
        raise ValueError("A source URL has inconsistent provenance metadata across rules.")

    for _index, rule in rules.iterrows():
        applicable = _pipe_values(rule["applicable_query_categories"])
        if not applicable or not applicable.issubset(ALLOWED_CATEGORIES):
            raise ValueError(
                f"Rule {rule['rule_id']} has invalid applicable_query_categories: "
                f"{sorted(applicable)}"
            )
        if str(rule["recommended_category"]) in applicable:
            raise ValueError(
                f"Rule {rule['rule_id']} recommends the same category as its query applicability."
            )
        for term_field in ("query_terms", "candidate_terms", "required_context"):
            values = _pipe_values(rule[term_field])
            if not values or ("none" in values and len(values) > 1):
                raise ValueError(f"Rule {rule['rule_id']} has an invalid {term_field} declaration.")

    covered_targets = set(rules["recommended_category"].astype(str))
    if covered_targets != ALLOWED_CATEGORIES:
        raise ValueError(
            "The canonical KB must cover all five recommendation targets; "
            f"missing={sorted(ALLOWED_CATEGORIES - covered_targets)}"
        )


def load_canonical_rules(path: str | Path) -> pd.DataFrame:
    rules = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_canonical_rules(rules)
    return rules


def load_audited_rules(config: Mapping[str, Any]) -> pd.DataFrame:
    return load_canonical_rules(config["paths"]["knowledge_base"])


def load_legacy_audit(path: str | Path) -> dict[str, Any]:
    audit_path = Path(path)
    audit = yaml.safe_load(audit_path.read_text(encoding="utf-8"))
    decisions = audit.get("decisions", {})
    if set(decisions) != LEGACY_DECISIONS:
        raise ValueError("Legacy audit must use the four declared decision classes.")
    ids = [rule_id for values in decisions.values() for rule_id in values]
    expected_count = int(audit["audited_asset"]["row_count"])
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise ValueError("Legacy audit does not uniquely account for every legacy rule.")
    if not audit.get("result", {}).get("all_legacy_rows_accounted_for"):
        raise ValueError("Legacy audit is not marked complete.")
    if audit.get("result", {}).get("experimental_results_inspected") is not False:
        raise ValueError("Legacy audit must record that experimental results were not inspected.")
    workspace_root = audit_path.resolve().parents[2]
    legacy_path = workspace_root / audit["audited_asset"]["path"]
    digest = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    if digest != audit["audited_asset"]["sha256"]:
        raise ValueError("Archived legacy KB does not match the audit's SHA-256 binding.")
    legacy_ids = set(
        pd.read_csv(legacy_path, dtype=str, keep_default_na=False)["rule_id"].astype(str)
    )
    if legacy_ids != set(ids):
        raise ValueError("Legacy decision IDs do not match the archived KB.")
    return audit


def coverage_matrix(rules: pd.DataFrame) -> pd.DataFrame:
    """Return static rule coverage by query category and recommendation target."""
    validate_canonical_rules(rules)
    records = [
        {"query_category": query, "recommended_category": row["recommended_category"]}
        for _, row in rules.iterrows()
        for query in _pipe_values(row["applicable_query_categories"])
    ]
    matrix = pd.crosstab(
        pd.DataFrame(records)["query_category"],
        pd.DataFrame(records)["recommended_category"],
    )
    return matrix.reindex(
        index=sorted(ALLOWED_CATEGORIES), columns=sorted(ALLOWED_CATEGORIES), fill_value=0
    )


def audit_static_case_applicability(
    cases: list[dict[str, Any]], rules: pd.DataFrame, target_category: str
) -> dict[str, Any]:
    """Audit pre-experiment case-to-rule applicability without rankings or model outputs."""
    validate_canonical_rules(rules)
    target_rules = rules[
        rules["recommended_category"].eq(target_category)
        & rules["audit_status"].eq("retain")
    ]
    target_cases = [case for case in cases if case.get("target_category") == target_category]
    unsupported = []
    query_counts: dict[str, dict[str, int]] = {}
    frequency: Counter[str] = Counter()
    packet_counts: Counter[tuple[str, ...]] = Counter()
    for case in target_cases:
        query_group = str(case.get("query_group", ""))
        permitted_text = " | ".join(
            str(case.get(field, ""))
            for field in ("query_category", "query_text", "outfit_context_text", "user_request")
        )
        eligible = []
        for _, rule in target_rules.iterrows():
            applicable = declared_values(rule["applicable_query_categories"])
            if query_group.lower() not in applicable and "all" not in applicable:
                continue
            if not matches_declared_terms(rule["query_terms"], permitted_text):
                continue
            eligible.append(str(rule["rule_id"]))
        packet = tuple(sorted(eligible))
        packet_counts[packet] += 1
        frequency.update(packet)
        counts = query_counts.setdefault(query_group, {"cases": 0, "supported": 0})
        counts["cases"] += 1
        if packet:
            counts["supported"] += 1
        else:
            unsupported.append(
                {
                    "case_id": str(case.get("case_id", "")),
                    "query_group": query_group,
                    "query_category": str(case.get("query_category", "")),
                    "query_text": str(case.get("query_text", "")),
                }
            )
    supported_count = len(target_cases) - len(unsupported)
    return {
        "target_category": target_category,
        "case_count": len(target_cases),
        "supported_case_count": supported_count,
        "unsupported_case_count": len(unsupported),
        "supported_fraction": supported_count / len(target_cases) if target_cases else 0.0,
        "unsupported_cases": unsupported,
        "query_category_coverage": query_counts,
        "rule_frequency": dict(sorted(frequency.items())),
        "maximum_rule_prevalence": (
            max(frequency.values()) / len(target_cases) if target_cases and frequency else 0.0
        ),
        "unique_nonempty_packets": len([packet for packet in packet_counts if packet]),
        "duplicate_nonempty_packet_cases": sum(
            count for packet, count in packet_counts.items() if packet and count > 1
        ),
        "coverage_pass": bool(target_cases) and not unsupported,
        "experimental_condition_results_inspected": False,
    }


def audit_bag_case_packets(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure empirical bag coverage, concentration, duplication and overlap."""
    bag_cases = [case for case in cases if case.get("target_category") == "bags"]
    packets: list[tuple[str, ...]] = []
    unsupported: list[str] = []
    query_counts: dict[str, dict[str, int]] = {}
    rule_frequency: Counter[str] = Counter()
    for case in bag_cases:
        rules = case.get("evidence_trace", {}).get("rules", [])
        packet = tuple(str(rule["rule_id"]) for rule in rules)
        packets.append(packet)
        rule_frequency.update(packet)
        query_group = str(case.get("query_group", "unknown"))
        counts = query_counts.setdefault(query_group, {"cases": 0, "supported": 0})
        counts["cases"] += 1
        if packet:
            counts["supported"] += 1
        else:
            unsupported.append(str(case.get("case_id", "")))

    similarities = []
    for left, right in combinations((set(packet) for packet in packets), 2):
        union = left | right
        similarities.append(len(left & right) / len(union) if union else 1.0)
    packet_frequency = Counter(packets)
    duplicate_cases = sum(
        count for packet, count in packet_frequency.items() if packet and count > 1
    )
    maximum_rule_prevalence = (
        max(rule_frequency.values()) / len(bag_cases) if bag_cases and rule_frequency else 0.0
    )
    return {
        "bag_case_count": len(bag_cases),
        "supported_case_count": len(bag_cases) - len(unsupported),
        "unsupported_case_count": len(unsupported),
        "unsupported_case_ids": unsupported,
        "query_category_coverage": query_counts,
        "rule_frequency": dict(sorted(rule_frequency.items())),
        "maximum_rule_prevalence": maximum_rule_prevalence,
        "unique_nonempty_packets": len({packet for packet in packets if packet}),
        "duplicate_packet_cases": duplicate_cases,
        "mean_pairwise_jaccard": sum(similarities) / len(similarities) if similarities else 0.0,
        "coverage_pass": bool(bag_cases) and not unsupported,
    }
