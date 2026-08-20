from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from evidence_fashion.calibration import (
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
        "human_claims": [
            {"claim_id": "C1", "claim_text": "A bag works.", "claim_type": "other"}
        ],
        "human_verification": [
            {
                "claim_id": "C1",
                "full_kb_entailment": "supported",
                "exact_trace_entailment": "supported",
                "common_reference_item_fact_support": "not_verifiable",
            }
        ],
        "human_citation_validation": [
            {
                "claim_id": "C1",
                "citation_present": False,
                "canonical_citation_format": True,
                "citation_entails_claim": None,
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
