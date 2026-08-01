import pandas as pd
import pytest

from evidence_fashion_recommender.evaluation.robustness import (
    full_hybrid_specs,
    select_hybrid_finalists,
    validate_stage1_validation_packets,
)


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
