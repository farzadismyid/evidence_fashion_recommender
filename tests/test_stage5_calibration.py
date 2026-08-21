from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from evidence_fashion.calibration import (
    align_calibration_claims,
    calibration_gates,
    calibration_metrics,
    validate_human_calibration,
)


def _settings() -> dict:
    return yaml.safe_load("""
calibration:
  source_split: validation
  final_explanation_split: test
  minimum_paired_cases: 1
  target_categories: [tops, bags]
  conditions: [no_rag, rule_rag]
  required_coverage_tags: [bag_example, compound_claim]
  required_human_fields:
    [annotator_id, completed_at_utc, human_claims, human_verification,
     human_citation_validation]
  pass_criteria:
    extractor_claim_recall_minimum: 0.9
    extractor_claim_precision_minimum: 0.9
    extractor_duplicate_rate_maximum: 0.0
    extractor_claim_id_preservation_rate_minimum: 1.0
    verifier_full_kb_accuracy_minimum: 0.8
    verifier_exact_trace_accuracy_minimum: 0.8
    verifier_common_reference_accuracy_minimum: 0.8
    citation_validity_accuracy_minimum: 0.8
    structured_output_success_rate_minimum: 1.0
""")


def _record(condition: str) -> dict:
    return {
        "calibration_case_id": "validation-1",
        "source_split": "validation",
        "target_category": "bags",
        "condition": condition,
        "coverage_tags": ["bag_example", "compound_claim"],
        "annotator_id": "reviewer-1",
        "completed_at_utc": "2026-08-20T00:00:00Z",
        "human_claims": [{"claim_id": "C1", "claim_text": "A bag works.", "claim_type": "other"}],
        "human_verification": [
            {
                "claim_id": "C1",
                "full_kb_candidate_applicable_rule_ids": ["K001"],
                "full_kb_entailment": "supported",
                "full_kb_rule_ids": ["K001"],
                "full_kb_reason": "The candidate rule applies and supports the claim.",
                "exact_trace_entailment": "supported",
                "exact_trace_rule_ids": ["K001"],
                "exact_trace_reason": "The exact trace supports the claim.",
                "common_reference_item_fact_support": "not_verifiable",
                "common_reference_fields": [],
                "common_reference_reason": "No item fact settles the relation.",
            }
        ],
        "human_citation_validation": [
            {
                "claim_id": "C1",
                "citation_present": False,
                "canonical_citation_format": True,
                "cited_rule_ids": [],
                "invalid_rule_ids": [],
                "citation_entails_claim": None,
                "brief_reason": "No citation is present.",
            }
        ],
    }


def test_stage5_requires_human_paired_disjoint_and_covered_annotations() -> None:
    records = [_record("no_rag"), _record("rule_rag")]
    summary = validate_human_calibration(records, _settings())
    assert summary["disjointness"]["disjoint"] is True
    assert summary["paired_case_count"] == 1
    invalid = deepcopy(records)
    invalid[0]["source_split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        validate_human_calibration(invalid, _settings())


def test_stage5_metrics_keep_the_three_verification_dimensions_separate() -> None:
    human = [_record("no_rag"), _record("rule_rag")]
    model = [
        {
            "calibration_case_id": row["calibration_case_id"],
            "condition": row["condition"],
            "status": "complete",
            "claims": deepcopy(row["human_claims"]),
            "verification": deepcopy(row["human_verification"]),
            "citation_validation": deepcopy(row["human_citation_validation"]),
        }
        for row in human
    ]
    model[0]["verification"][0]["full_kb_entailment"] = "unsupported"
    metrics = calibration_metrics(human, model)
    assert metrics["verifier_full_kb_accuracy"] == 0.5
    assert metrics["verifier_exact_trace_accuracy"] == 1.0
    assert calibration_gates(metrics, _settings())["stage5_pass"] is False


def test_calibration_alignment_matches_paraphrases_but_rejects_polarity_changes() -> None:
    aligned = align_calibration_claims(
        [{"claim_id": "C1", "claim_text": "The cashmere top has sophisticated texture."}],
        [{"claim_id": "C1", "claim_text": "The cashmere top provides sophisticated texture."}],
    )
    assert aligned["pairs"][0]["method"] == "semantic_proposition"
    negated = align_calibration_claims(
        [{"claim_id": "C1", "claim_text": "The bag is not suitable."}],
        [{"claim_id": "C1", "claim_text": "The bag is suitable."}],
    )
    assert negated["pairs"] == []


def test_calibration_alignment_is_one_to_one_for_split_or_merged_claims() -> None:
    aligned = align_calibration_claims(
        [{"claim_id": "C1", "claim_text": "The bag is suitable and blue."}],
        [
            {"claim_id": "C1", "claim_text": "The bag is suitable."},
            {"claim_id": "C2", "claim_text": "The bag is blue."},
        ],
    )
    assert aligned["pairs"] == []
    assert {row["decision"] for row in aligned["candidate_scores"]} >= {
        "split_or_merge_atomization"
    }


def test_calibration_alignment_resolves_locked_item_coreference() -> None:
    aligned = align_calibration_claims(
        [{"claim_id": "C1", "claim_text": "The exact item is waterproof."}],
        [{"claim_id": "C1", "claim_text": "The sock booties are waterproof."}],
        common_reference_item_facts={"locked_item_minimal_name": "sock booties"},
    )
    assert aligned["pairs"][0]["method"] == "coreference_normalized_text"


def test_calibration_metrics_do_not_score_absent_citations() -> None:
    human = [_record("no_rag"), _record("rule_rag")]
    model = [
        {
            "calibration_case_id": row["calibration_case_id"],
            "condition": row["condition"],
            "status": "complete",
            "claims": deepcopy(row["human_claims"]),
            "verification": deepcopy(row["human_verification"]),
            "citation_validation": [
                {
                    **deepcopy(row["human_citation_validation"][0]),
                    "canonical_citation_format": None,
                }
            ],
        }
        for row in human
    ]
    metrics = calibration_metrics(human, model)
    assert metrics["citation_scored_claim_count"] == 0
    assert metrics["citation_validity_accuracy"] == 1.0
