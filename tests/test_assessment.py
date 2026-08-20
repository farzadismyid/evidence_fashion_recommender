import pytest

from evidence_fashion.assessment import (
    build_citation_validation_prompt,
    build_separated_entailment_prompt,
    build_true_blind_judge_prompt,
    cited_rule_ids,
    normalize_verification_payload,
    prepare_true_blind_pair,
    strip_condition_revealing_phrases,
    validate_canonical_citation_format,
    validate_extraction,
    validate_judgment,
    validate_verification,
)


def test_extraction_requires_complete_sequential_unique_claims() -> None:
    types = ["colour", "styling_relation"]
    claims = validate_extraction(
        {
            "claims": [
                {"claim_id": "C1", "claim_text": "The bag is black.", "claim_type": "colour"},
                {
                    "claim_id": "C2",
                    "claim_text": "The bag balances the top.",
                    "claim_type": "styling_relation",
                },
            ]
        },
        types,
    )
    assert [row["claim_id"] for row in claims] == ["C1", "C2"]
    with pytest.raises(ValueError, match="sequential"):
        validate_extraction(
            {"claims": [{"claim_id": "C2", "claim_text": "x", "claim_type": "colour"}]},
            types,
        )
    repaired = validate_extraction(
        {
            "claims": [
                {"claim_id": "C1", "claim_text": "Same claim.", "claim_type": "colour"},
                {"claim_id": "C2", "claim_text": "Same claim.", "claim_type": "colour"},
                {"claim_id": "C3", "claim_text": "New claim.", "claim_type": "colour"},
            ]
        },
        types,
    )
    assert [row["claim_id"] for row in repaired] == ["C1", "C2"]


def test_verification_allows_multiple_sources_and_requires_exact_coverage() -> None:
    claims = [{"claim_id": "C1", "claim_text": "The bag balances the top."}]
    verified = validate_verification(
        {
            "claims": [
                {
                    "claim_id": "C1",
                    "support_status": "supported",
                    "support_sources": [
                        "query_or_locked_item",
                        "rule_evidence",
                        "rule_evidence",
                    ],
                    "supporting_rule_ids": ["K001"],
                    "citation_entails_claim": True,
                    "brief_reason": "Both sources entail the relation.",
                }
            ]
        },
        claims,
        {"K001"},
        ["K001"],
    )
    assert len(verified[0]["support_sources"]) == 2
    with pytest.raises(ValueError, match="cover every claim"):
        validate_verification({"claims": []}, claims, {"K001"}, ["K001"])


def test_invalid_rule_ids_and_uncited_boolean_are_rejected() -> None:
    claims = [{"claim_id": "C1", "claim_text": "Claim"}]
    payload = {
        "claims": [
            {
                "claim_id": "C1",
                "support_status": "supported",
                "support_sources": ["rule_evidence"],
                "supporting_rule_ids": ["K999"],
                "citation_entails_claim": None,
                "brief_reason": "Reason",
            }
        ]
    }
    with pytest.raises(ValueError, match="outside"):
        validate_verification(payload, claims, {"K001"}, [])
    payload["claims"][0]["supporting_rule_ids"] = ["K001"]
    payload["claims"][0]["citation_entails_claim"] = True
    with pytest.raises(ValueError, match="must be null"):
        validate_verification(payload, claims, {"K001"}, [])


def test_verifier_structural_normalization_is_conservative() -> None:
    payload, actions = normalize_verification_payload(
        {
            "claims": [
                {
                    "claim_id": "C1",
                    "support_status": "supported",
                    "support_sources": ["rule_evidence"],
                    "supporting_rule_ids": ["K999"],
                    "citation_entails_claim": True,
                    "brief_reason": "Model supplied an unavailable rule.",
                }
            ]
        },
        {"K001"},
        [],
    )
    row = payload["claims"][0]
    assert row["support_status"] == "not_verifiable"
    assert row["support_sources"] == []
    assert row["supporting_rule_ids"] == []
    assert row["citation_entails_claim"] is None
    assert "conservatively_relabelled_sourceless_support" in actions


def test_citations_and_paired_judge_scores_are_validated() -> None:
    assert cited_rule_ids("Works [K001], repeats [K001], and [K126].", r"\[(K[0-9]{3})\]") == [
        "K001",
        "K126",
    ]
    dimensions = ["clarity", "specificity"]
    result = validate_judgment(
        {
            "first": {"clarity": 4, "specificity": 3, "brief_reason": "Clear."},
            "second": {"clarity": 5, "specificity": 5, "brief_reason": "Excellent."},
        },
        dimensions,
        1,
        5,
    )
    assert result["second"]["clarity"] == 5


def test_separated_entailment_is_citation_blind_and_citation_format_is_separate() -> None:
    explanation = "This works well [K025] [K099]."
    prompt = build_separated_entailment_prompt(
        explanation=explanation,
        claims=[{"claim_id": "C1", "claim_text": "This works well."}],
        full_kb_rules=[{"rule_id": "K025", "rule_text": "Works well."}],
        exact_trace_rules=[{"rule_id": "K025", "rule_text": "Works well."}],
        common_reference_item_facts={"locked_item": "bag"},
    )
    assert "[K025]" not in prompt
    assert "[K099]" not in prompt
    citation_prompt = build_citation_validation_prompt(
        claims=[{"claim_id": "C1", "claim_text": "This works well."}],
        explanation=explanation,
        exact_trace_rules=[{"rule_id": "K025", "rule_text": "Works well."}],
    )
    assert "[K025]" in citation_prompt
    assert validate_canonical_citation_format(explanation) == ["[K025]", "[K099]"]
    with pytest.raises(ValueError, match="Grouped"):
        validate_canonical_citation_format("This works [K025, K099].")


def test_true_blind_judging_removes_citations_and_condition_language() -> None:
    pair = prepare_true_blind_pair(
        {
            "no_rag": "No-RAG option: a clear choice.",
            "rule_rag": "Based on provided rules, this works [K025].",
        },
        case_id="case-1",
        generator="gemma",
        seed=42,
    )
    displayed = f"{pair['first_explanation']} {pair['second_explanation']}".lower()
    assert "[k025]" not in displayed
    assert "no-rag" not in displayed
    assert "provided rules" not in displayed
    prompt = build_true_blind_judge_prompt(
        common_reference_context={"request": "recommend a bag"},
        first_explanation=pair["first_explanation"],
        second_explanation=pair["second_explanation"],
    )
    assert "hallucination" not in prompt.lower()
    assert "non-redundancy" in prompt.lower()
    assert strip_condition_revealing_phrases("Rule-RAG [K025]") == ""
