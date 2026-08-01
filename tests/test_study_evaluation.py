import pandas as pd

from evidence_fashion_recommender.evaluation.explanations import (
    candidate_substitution_types,
)
from evidence_fashion_recommender.evaluation.study import (
    _parse_judge,
    evaluate_explanations,
    evaluate_rag_retrieval,
)


def test_evidence_aware_explanation_metrics() -> None:
    frame = pd.DataFrame(
        [
            {
                "paper_case_id": "C1",
                "grounding_variant": "rule_rag",
                "generated_explanation": "The locked pumps maintain polish [R001].",
                "rule_evidence_ids": "R001",
                "rule_evidence_text": "R001: Pumps maintain a polished look.",
                "recommended_category": "Pumps",
                "recommended_text": "black pumps",
            }
        ]
    )
    evaluated, summary = evaluate_explanations(frame)
    assert evaluated.loc[0, "citation_correctness"] == 1
    assert evaluated.loc[0, "unsupported_claim_count_evidence_aware"] == 0
    assert summary.loc[0, "citation_presence_rate"] == 1


def test_rag_retrieval_metrics_use_full_applicable_denominator() -> None:
    cases = pd.DataFrame(
        [
            {
                "paper_case_id": "C1",
                "query_group": "dresses",
                "target_category": "shoes",
                "rule_evidence_ids": "R001",
            }
        ]
    )
    kb = pd.DataFrame(
        [
            {
                "rule_id": "R001",
                "input_category": "dresses",
                "recommended_category": "shoes",
                "source_reliability": "high",
            },
            {
                "rule_id": "R002",
                "input_category": "dresses",
                "recommended_category": "shoes",
                "source_reliability": "high",
            },
        ]
    )
    results, _ = evaluate_rag_retrieval(cases, kb, [1])
    assert results.loc[0, "precision_at_1"] == 1
    assert results.loc[0, "recall_at_1"] == 0.5


def test_substitution_detector_distinguishes_context_from_alternative() -> None:
    allowed = {"jacket"}
    assert not candidate_substitution_types(
        "This jacket balances the skirt and works with the query shoes.", allowed
    )
    assert candidate_substitution_types("Choose a blazer instead of this jacket.", allowed) == {
        "blazer"
    }


def test_judge_parser_accepts_nested_claim_payload() -> None:
    parsed = _parse_judge(
        '{"faithfulness_to_available_information":4,"usefulness_to_user":5,'
        '"specificity":4,"style_appropriateness":5,"grounding_safety":5,'
        '"claims":[{"claim":"x","support":"supported"}]}'
    )
    assert parsed["faithfulness_to_available_information"] == 4


def test_judge_parser_repairs_observed_local_json_defects() -> None:
    parsed = _parse_judge(
        '{\n"faithfulness_to_available_information":4,\n"usefulness_to_user":4,'
        '\n"specificity":4,\n"style_appropriateness":4,\n"grounding_safety":4,'
        '\n"claims":[{"claim":"x","support":"unsupported"\n'
        "// unsupported because the evidence is absent\n"
        '"brief_reason":"why",}],\n"brief_reason":"ok"\n}'
    )
    assert parsed["claims"][0]["support"] == "unsupported"
    assert parsed["claims"][0]["support_label_compliant"] is True


def test_judge_parser_normalizes_support_labels_conservatively() -> None:
    parsed = _parse_judge(
        '{"faithfulness_to_available_information":4,"usefulness_to_user":4,'
        '"specificity":4,"style_appropriateness":4,"grounding_safety":4,'
        '"claims":[{"claim":"a","support":"supported (R099)"},'
        '{"claim":"b","support":"not verifiable"},'
        '{"claim":"c","support":"the colors match"}]}'
    )
    assert [claim["support"] for claim in parsed["claims"]] == [
        "supported",
        "not_verifiable",
        "not_verifiable",
    ]
    assert [claim["support_label_compliant"] for claim in parsed["claims"]] == [
        False,
        True,
        False,
    ]


def test_judge_parser_repairs_missing_claim_array_closure() -> None:
    parsed = _parse_judge(
        '{"faithfulness_to_available_information":4,"usefulness_to_user":4,'
        '"specificity":4,"style_appropriateness":4,"grounding_safety":4,'
        '"claims":[{"claim":"a","support":"supported"}\n,'
        '"brief_reason":"complete response"}'
    )
    assert parsed["claims"][0]["support"] == "supported"
    assert parsed["brief_reason"] == "complete response"
