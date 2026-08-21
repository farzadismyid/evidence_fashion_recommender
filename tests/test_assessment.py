import pytest

from evidence_fashion.assessment import (
    build_citation_validation_prompt,
    build_separated_entailment_prompt,
    build_true_blind_judge_prompt,
    cited_rule_ids,
    common_reference_eligibility,
    normalize_verification_payload,
    prepare_true_blind_pair,
    strip_condition_revealing_phrases,
    validate_canonical_citation_format,
    validate_citation_validation,
    validate_extraction,
    validate_judgment,
    validate_separated_entailment,
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


def test_separated_entailment_rejects_empty_missing_and_duplicate_claim_rows() -> None:
    claims = [
        {"claim_id": "C1", "claim_text": "First claim."},
        {"claim_id": "C2", "claim_text": "Second claim."},
    ]
    row = {
        "claim_id": "C1",
        "full_kb_candidate_applicable_rule_ids": ["K001"],
        "full_kb_entailment": "supported",
        "full_kb_rule_ids": ["K001"],
        "full_kb_reason": "The candidate rule entails it.",
        "exact_trace_entailment": "supported",
        "exact_trace_rule_ids": ["K001"],
        "exact_trace_reason": "The exact rule entails it.",
        "common_reference_item_fact_support": "not_verifiable",
        "common_reference_eligible": False,
        "common_reference_fields": [],
        "common_reference_reason": "No item fact settles it.",
    }
    with pytest.raises(ValueError, match="non-empty"):
        validate_separated_entailment(
            {"claims": []},
            claims,
            full_kb_rule_ids={"K001"},
            exact_trace_rule_ids={"K001"},
            common_reference_item_facts={},
        )
    with pytest.raises(ValueError, match="exactly preserve"):
        validate_separated_entailment(
            {"claims": [row, {**row, "claim_id": "C1"}]},
            claims,
            full_kb_rule_ids={"K001"},
            exact_trace_rule_ids={"K001"},
            common_reference_item_facts={},
        )
    with pytest.raises(ValueError, match="missing a required"):
        validate_separated_entailment(
            {"claims": [{**row, "claim_id": "C1"}, {"claim_id": "C2"}]},
            claims,
            full_kb_rule_ids={"K001"},
            exact_trace_rule_ids={"K001"},
            common_reference_item_facts={},
        )


def test_citation_validation_distinguishes_absent_and_malformed() -> None:
    claims = [{"claim_id": "C1", "claim_text": "Claim."}]
    absent = {
        "claims": [
            {
                "claim_id": "C1",
                "citation_entails_claim": None,
                "brief_reason": "No citation is present.",
            }
        ]
    }
    assert validate_citation_validation(absent, claims, occurrence_diagnostics=[]) == {
        "claims": [
            {
                "claim_id": "C1",
                "citation_present": False,
                "canonical_citation_format": None,
                "cited_rule_ids": [],
                "invalid_rule_ids": [],
                "citation_entails_claim": None,
                "brief_reason": "No citation is present.",
            }
        ]
    }
    malformed = {
        "claims": [
            {
                "claim_id": "C1",
                "citation_entails_claim": None,
                "brief_reason": "The grouped citation is malformed.",
            }
        ]
    }
    assert (
        validate_citation_validation(
            malformed,
            claims,
            occurrence_diagnostics=[
                {
                    "raw": "[K001, K001]",
                    "rule_ids": ["K001", "K001"],
                    "valid_canonical_occurrence": False,
                    "unknown_rule_ids": [],
                    "out_of_trace_rule_ids": [],
                }
            ],
        )["claims"][0]["canonical_citation_format"]
        is False
    )


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


def test_shared_entailment_rubric_forbids_category_analogy_and_closed_world_exclusion() -> None:
    prompt = build_separated_entailment_prompt(
        explanation="The ankle booties are suitable. The blazer is unsuitable.",
        claims=[
            {"claim_id": "C1", "claim_text": "K016 supports ankle booties."},
            {"claim_id": "C2", "claim_text": "K026 entails a blazer is unsuitable."},
        ],
        full_kb_rules=[
            {
                "rule_id": "K016",
                "rule_text": "Pumps such as slingbacks or mules are a documented direction.",
            },
            {
                "rule_id": "K026",
                "rule_text": "A denim jacket or bomber is category-appropriate.",
            },
        ],
        exact_trace_rules=[],
        common_reference_item_facts={},
    )
    assert "Do not use category similarity" in prompt
    assert "examples as exhaustive lists" in prompt


def test_common_reference_eligibility_allows_only_literal_supplied_case_facts() -> None:
    facts = {
        "locked_item_minimal_name": "blue suede jacket",
        "locked_item_category": "Jackets",
        "query_item_minimal_name": "black trousers",
        "outfit_context_text": "Jackets: blue suede jacket | Bottoms: black trousers",
    }
    assert common_reference_eligibility({"claim_text": "The blue suede jacket is suede."}, facts)[
        "eligible"
    ]
    for text in (
        "The blue suede jacket is suitable for the black trousers.",
        "The blue suede jacket complements the black trousers.",
        "The blue suede jacket is sophisticated.",
        "A jacket is a documented styling direction when trousers are present.",
        "The blue suede jacket is waterproof.",
    ):
        assert not common_reference_eligibility({"claim_text": text}, facts)["eligible"]


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
