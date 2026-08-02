import pandas as pd
import pytest

from evidence_fashion_recommender.evaluation.hybrid_v2 import (
    balanced_screening_subset,
    summarize_hybrid_phase,
)
from evidence_fashion_recommender.evaluation.robustness import (
    full_hybrid_specs,
    select_hybrid_finalists,
    validate_stage1_validation_packets,
)
from evidence_fashion_recommender.evaluation.study import _rule_frame


def test_full_hybrid_grid_has_36_specs_and_labels_rule_only_candidates() -> None:
    specs = full_hybrid_specs([35, 55, 75], [3, 5], [0, 2, 5], ["rules_first", "item_first"])
    assert len(specs) == 36
    assert len({spec.name for spec in specs}) == 36
    rule_only = [spec for spec in specs if spec.item_limit == 0]
    assert len(rule_only) == 12
    assert all(not spec.final_hybrid_eligible for spec in rule_only)
    assert all(spec.name.startswith("rule_only_candidate_") for spec in rule_only)


def test_hybrid_packets_must_come_from_selected_stage1_validation() -> None:
    valid = pd.DataFrame(
        [
            {
                "research_split": "validation",
                "stage1_packet_hash": "abc",
                "stage1_packet_protocol": "final_eval_v2_selected",
            }
        ]
    )
    assert validate_stage1_validation_packets(valid) == "abc"
    legacy = valid.assign(stage1_packet_protocol="legacy_v1_packets_only")
    with pytest.raises(ValueError, match="ineligible"):
        validate_stage1_validation_packets(legacy)


def test_priority_selection_excludes_rule_only_candidates() -> None:
    base = {
        "hallucinated_claim_rate": 0.1,
        "rule_supported_claim_rate": 0.8,
        "evidence_misuse_rate": 0.0,
        "candidate_substitution_rate": 0.0,
        "rule_evidence_overlap": 0.4,
        "item_evidence_overlap": 0.2,
        "general_clarity": 4.0,
    }
    summary = pd.DataFrame(
        [
            {**base, "grounding_variant": "rule-only", "item_limit": 0, "max_words": 35},
            {**base, "grounding_variant": "hybrid-55", "item_limit": 2, "max_words": 55},
            {**base, "grounding_variant": "hybrid-35", "item_limit": 5, "max_words": 35},
        ]
    )
    selected = select_hybrid_finalists(summary, practical_tie=0.01, finalist_count=4)
    assert set(selected["grounding_variant"]) == {"hybrid-35", "hybrid-55"}
    assert selected.iloc[0]["grounding_variant"] == "hybrid-35"


def test_v2_rule_packet_preserves_rule_text() -> None:
    row = pd.Series(
        {
            "rule_evidence_packet": '[{"rule_id":"R001","rule_text":"Use tonal colour."}]',
            "rule_evidence_ids": '["R001"]',
            "rule_evidence_text": "Use tonal colour.",
        }
    )
    assert _rule_frame(row).to_dict("records") == [
        {"rule_id": "R001", "rule_text": "Use tonal colour."}
    ]


def test_balanced_screening_subset_is_category_balanced() -> None:
    cases = pd.DataFrame(
        [
            {"paper_case_id": f"{category}-{index}", "target_category": category}
            for category in ("a", "b")
            for index in range(5)
        ]
    )
    selected = balanced_screening_subset(cases, 3, 42)
    assert selected.groupby("target_category").size().to_dict() == {"a": 3, "b": 3}


def test_hybrid_summary_uses_separate_claim_denominators() -> None:
    explanation = pd.DataFrame(
        [
            {
                "paper_case_id": "v1",
                "grounding_variant": "hybrid_w35_r3_i2_item_first",
                "max_words": 35,
                "rule_limit": 3,
                "item_limit": 2,
                "prompt_order": "item_first",
                "candidate_type": "hybrid",
                "generated_explanation": "A scarf works [R001].",
                "rule_evidence_ids": '["R001"]',
                "rule_evidence_text": "A scarf works.",
                "item_evidence_text": "scarf",
                "recommended_text": "scarf",
                "recommended_category": "accessories",
                "query_text": "blazer",
                "query_category": "outerwear",
            }
        ]
    )
    judged = explanation[["paper_case_id", "grounding_variant"]].assign(
        fashion_claim_count=2,
        hallucinated_fashion_claim_count=1,
        styling_claim_count=4,
        rule_supported_styling_claim_count=3,
        evidence_misuse=False,
        candidate_substitution=False,
        general_clarity=4.0,
    )
    _, summary = summarize_hybrid_phase(explanation, judged)
    assert summary.iloc[0]["hallucinated_claim_rate"] == 0.5
    assert summary.iloc[0]["rule_supported_claim_rate"] == 0.75
