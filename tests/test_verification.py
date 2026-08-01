import pandas as pd

from evidence_fashion_recommender.evaluation.verification import (
    counterfactual_category_test,
    rule_relevance_agreement,
)


def test_rule_relevance_agreement_is_perfect_for_identical_judges() -> None:
    rows = []
    for judge in ("a", "b"):
        rows.extend(
            [
                {
                    "paper_case_id": "c",
                    "rule_id": "R1",
                    "rank": 1,
                    "judge_model": judge,
                    "relevant": 1,
                },
                {
                    "paper_case_id": "c",
                    "rule_id": "R2",
                    "rank": 2,
                    "judge_model": judge,
                    "relevant": 0,
                },
            ]
        )
    result = rule_relevance_agreement(pd.DataFrame(rows)).iloc[0]
    assert result["percent_agreement"] == 1
    assert result["cohen_kappa"] == 1


def test_counterfactual_target_rotation_has_no_false_match() -> None:
    cases = pd.DataFrame(
        [
            {
                "paper_case_id": "c1",
                "target_category": "shoes",
                "rule_evidence_ids": "R1",
            },
            {
                "paper_case_id": "c2",
                "target_category": "tops",
                "rule_evidence_ids": "R2",
            },
        ]
    )
    kb = pd.DataFrame(
        [
            {"rule_id": "R1", "recommended_category": "shoes"},
            {"rule_id": "R2", "recommended_category": "tops"},
        ]
    )
    result = counterfactual_category_test(cases, kb)
    assert result["counterfactual_false_match_rate"].sum() == 0
