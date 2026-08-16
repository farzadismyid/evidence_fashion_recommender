import pytest

from evidence_fashion.assessment import (
    cited_rule_ids,
    normalize_verification_payload,
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
                    "supporting_rule_ids": ["R001"],
                    "citation_entails_claim": True,
                    "brief_reason": "Both sources entail the relation.",
                }
            ]
        },
        claims,
        {"R001"},
        ["R001"],
    )
    assert len(verified[0]["support_sources"]) == 2
    with pytest.raises(ValueError, match="cover every claim"):
        validate_verification({"claims": []}, claims, {"R001"}, ["R001"])


def test_invalid_rule_ids_and_uncited_boolean_are_rejected() -> None:
    claims = [{"claim_id": "C1", "claim_text": "Claim"}]
    payload = {
        "claims": [
            {
                "claim_id": "C1",
                "support_status": "supported",
                "support_sources": ["rule_evidence"],
                "supporting_rule_ids": ["R999"],
                "citation_entails_claim": None,
                "brief_reason": "Reason",
            }
        ]
    }
    with pytest.raises(ValueError, match="outside"):
        validate_verification(payload, claims, {"R001"}, [])
    payload["claims"][0]["supporting_rule_ids"] = ["R001"]
    payload["claims"][0]["citation_entails_claim"] = True
    with pytest.raises(ValueError, match="must be null"):
        validate_verification(payload, claims, {"R001"}, [])


def test_verifier_structural_normalization_is_conservative() -> None:
    payload, actions = normalize_verification_payload(
        {
            "claims": [
                {
                    "claim_id": "C1",
                    "support_status": "supported",
                    "support_sources": ["rule_evidence"],
                    "supporting_rule_ids": ["R999"],
                    "citation_entails_claim": True,
                    "brief_reason": "Model supplied an unavailable rule.",
                }
            ]
        },
        {"R001"},
        [],
    )
    row = payload["claims"][0]
    assert row["support_status"] == "not_verifiable"
    assert row["support_sources"] == []
    assert row["supporting_rule_ids"] == []
    assert row["citation_entails_claim"] is None
    assert "conservatively_relabelled_sourceless_support" in actions


def test_citations_and_paired_judge_scores_are_validated() -> None:
    assert cited_rule_ids("Works [R001], repeats [R001], and [R126].", r"\[(R[0-9]{3})\]") == [
        "R001",
        "R126",
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
