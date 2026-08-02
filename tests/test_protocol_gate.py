import pandas as pd

from evidence_fashion_recommender.evaluation.protocol_gate import compare_locked_packets


def _packets(item_id: str = "i1", rules: str = "R001") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_case_id": "c1",
                "recommended_item_id": item_id,
                "item_evidence_text": "ITEM-1: shoes",
                "rule_evidence_ids": rules,
                "rule_evidence_text": f"{rules}: rule",
            }
        ]
    )


def test_gate_allows_only_explicit_legacy_generation_label_when_unchanged() -> None:
    comparison, summary = compare_locked_packets(_packets(), _packets())
    assert not comparison.loc[0, "recommendation_changed"]
    assert not summary["material_change"]
    assert summary["decision"] == "legacy_generation_v2_judging"


def test_gate_requires_all_variant_regeneration_for_evidence_change() -> None:
    _, summary = compare_locked_packets(_packets(), _packets(rules="R002"))
    assert summary["material_change"]
    assert summary["decision"] == "regenerate_all_variants"


def test_gate_aligns_protocol_specific_case_ids_on_semantic_key() -> None:
    legacy = pd.concat([_packets("i1"), _packets("i2")], ignore_index=True).assign(
        paper_case_id=["legacy-1", "legacy-2"],
        query_item_id=["q1", "q2"],
        target_category=["shoes", "tops"],
    )
    v2 = legacy.assign(paper_case_id=["v2-1", "v2-2"])
    comparison, summary = compare_locked_packets(legacy, v2)
    assert len(comparison) == 2
    assert summary["alignment_key"] == ["query_item_id", "target_category"]
    assert summary["changed_recommendation_rate"] == 0.0
