"""Stage 5 human-calibration validation and model-quality gate calculations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

ENTAILMENT_FIELDS = (
    "full_kb_entailment",
    "exact_trace_entailment",
    "common_reference_item_fact_support",
)


def _normalized_claim(text: str) -> str:
    return " ".join(text.lower().split())


def _require_mapping(record: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = record.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping.")
    return value


def validate_human_calibration(
    records: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]
) -> dict[str, Any]:
    """Refuse anything that cannot serve as disjoint human calibration evidence."""
    calibration = _require_mapping(settings, "calibration")
    if not records:
        raise ValueError("Stage 5 has no human calibration annotations.")
    required_fields = set(calibration["required_human_fields"])
    required_tags = set(calibration["required_coverage_tags"])
    categories = set(calibration["target_categories"])
    conditions = set(calibration["conditions"])
    pairs: dict[str, set[str]] = {}
    observed_tags: set[str] = set()
    counts = Counter()
    for record in records:
        missing = required_fields - set(record)
        if missing:
            raise ValueError(f"Human annotation is missing fields: {sorted(missing)}.")
        if record.get("source_split") != calibration["source_split"]:
            raise ValueError(
                "Calibration records must come only from the reserved validation split."
            )
        if record.get("target_category") not in categories:
            raise ValueError("Calibration record has an unsupported target category.")
        condition = record.get("condition")
        if condition not in conditions:
            raise ValueError("Calibration record has an unsupported condition.")
        case_id = str(record.get("calibration_case_id", "")).strip()
        if not case_id:
            raise ValueError("Calibration records need a calibration_case_id.")
        pairs.setdefault(case_id, set()).add(str(condition))
        counts[str(record["target_category"])] += 1
        observed_tags.update(record.get("coverage_tags", []))
        if not str(record["annotator_id"]).strip() or not str(record["completed_at_utc"]).strip():
            raise ValueError(
                "Calibration annotations require a human annotator ID and completion time."
            )
        claims = record["human_claims"]
        if not isinstance(claims, list) or not claims:
            raise ValueError("Human annotation needs at least one atomic claim.")
        claim_ids = [claim.get("claim_id") for claim in claims if isinstance(claim, Mapping)]
        expected_ids = [f"C{index}" for index in range(1, len(claims) + 1)]
        if claim_ids != expected_ids:
            raise ValueError("Human claims must use consecutive C IDs in textual order.")
        verification = record["human_verification"]
        citations = record["human_citation_validation"]
        if not isinstance(verification, list) or not isinstance(citations, list):
            raise ValueError("Human verification and citation validation must be arrays.")
        for rows, label in ((verification, "verification"), (citations, "citation validation")):
            if [row.get("claim_id") for row in rows] != claim_ids:
                raise ValueError(f"Human {label} must cover each human claim exactly once.")
        for row in verification:
            if any(field not in row for field in ENTAILMENT_FIELDS):
                raise ValueError(
                    "Human verification must contain all three independent dimensions."
                )
    incomplete_pairs = sorted(case_id for case_id, values in pairs.items() if values != conditions)
    if incomplete_pairs:
        raise ValueError(f"Calibration cases must include both conditions: {incomplete_pairs}.")
    if len(pairs) < int(calibration["minimum_paired_cases"]):
        raise ValueError("Stage 5 has fewer than the required paired calibration cases.")
    missing_tags = required_tags - observed_tags
    if missing_tags:
        raise ValueError(f"Calibration coverage is incomplete: {sorted(missing_tags)}.")
    if "bags" not in counts:
        raise ValueError("Stage 5 calibration must include a bag example.")
    return {
        "paired_case_count": len(pairs),
        "record_count": len(records),
        "counts_by_category": dict(sorted(counts.items())),
        "coverage_tags": sorted(observed_tags),
        "disjointness": {
            "calibration_source_split": calibration["source_split"],
            "final_explanation_split": calibration["final_explanation_split"],
            "disjoint": calibration["source_split"] != calibration["final_explanation_split"],
        },
    }


def calibration_metrics(
    human_records: Sequence[Mapping[str, Any]], model_records: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    """Calculate transparent extractor/verifier quality metrics against human annotations."""
    model_by_key = {
        (str(row["calibration_case_id"]), str(row["condition"])): row for row in model_records
    }
    human_claim_count = model_claim_count = matched_claim_count = duplicate_claim_count = 0
    claim_id_matches = verification_total = citation_total = 0
    verification_matches = Counter()
    citation_matches = 0
    structured_success = 0
    for human in human_records:
        key = (str(human["calibration_case_id"]), str(human["condition"]))
        model = model_by_key.get(key)
        if model is None:
            continue
        if model.get("status") == "complete":
            structured_success += 1
        human_claims = list(human["human_claims"])
        model_claims = list(model.get("claims", []))
        human_claim_count += len(human_claims)
        model_claim_count += len(model_claims)
        human_texts = {_normalized_claim(str(row["claim_text"])) for row in human_claims}
        model_texts = [_normalized_claim(str(row.get("claim_text", ""))) for row in model_claims]
        matched_claim_count += len(human_texts & set(model_texts))
        duplicate_claim_count += len(model_texts) - len(set(model_texts))
        expected_ids = [f"C{index}" for index in range(1, len(model_claims) + 1)]
        if [row.get("claim_id") for row in model_claims] == expected_ids:
            claim_id_matches += 1
        human_verification = {row["claim_id"]: row for row in human["human_verification"]}
        raw_entailment = model.get("verification", model.get("entailment", []))
        if isinstance(raw_entailment, Mapping):
            raw_entailment = raw_entailment.get("claims", [])
        model_verification = {row["claim_id"]: row for row in raw_entailment}
        for claim_id, expected in human_verification.items():
            observed = model_verification.get(claim_id, {})
            for field in ENTAILMENT_FIELDS:
                verification_total += 1
                verification_matches[field] += observed.get(field) == expected.get(field)
        human_citations = {row["claim_id"]: row for row in human["human_citation_validation"]}
        raw_citations = model.get("citation_validation", [])
        if isinstance(raw_citations, Mapping):
            raw_citations = raw_citations.get("claims", [])
        model_citations = {row["claim_id"]: row for row in raw_citations}
        for claim_id, expected in human_citations.items():
            citation_total += 1
            observed = model_citations.get(claim_id, {})
            citation_matches += (
                observed.get("citation_present") == expected.get("citation_present")
                and observed.get("canonical_citation_format")
                == expected.get("canonical_citation_format")
                and observed.get("citation_entails_claim") == expected.get("citation_entails_claim")
            )
    record_count = len(human_records)
    human_claim_denominator = human_claim_count or 1
    model_claim_denominator = model_claim_count or 1
    record_denominator = record_count or 1
    verifier_denominator = verification_total / len(ENTAILMENT_FIELDS) or 1
    citation_denominator = citation_total or 1
    return {
        "extractor_claim_recall": matched_claim_count / human_claim_denominator,
        "extractor_claim_precision": matched_claim_count / model_claim_denominator,
        "extractor_duplicate_rate": duplicate_claim_count / model_claim_denominator,
        "extractor_claim_id_preservation_rate": claim_id_matches / record_denominator,
        "verifier_full_kb_accuracy": (
            verification_matches["full_kb_entailment"] / verifier_denominator
        ),
        "verifier_exact_trace_accuracy": (
            verification_matches["exact_trace_entailment"] / verifier_denominator
        ),
        "verifier_common_reference_accuracy": (
            verification_matches["common_reference_item_fact_support"] / verifier_denominator
        ),
        "citation_validity_accuracy": citation_matches / citation_denominator,
        "structured_output_success_rate": structured_success / record_denominator,
    }


def calibration_gates(metrics: Mapping[str, float], settings: Mapping[str, Any]) -> dict[str, bool]:
    criteria = _require_mapping(_require_mapping(settings, "calibration"), "pass_criteria")
    outcomes: dict[str, bool] = {}
    for metric, threshold in criteria.items():
        metric_name = metric.removesuffix("_minimum").removesuffix("_maximum")
        value = metrics.get(metric_name)
        if value is None:
            outcomes[metric_name] = False
        elif metric.endswith("_minimum"):
            outcomes[metric_name] = value >= float(threshold)
        else:
            outcomes[metric_name] = value <= float(threshold)
    outcomes["stage5_pass"] = all(outcomes.values())
    return outcomes
