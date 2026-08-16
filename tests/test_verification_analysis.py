import json

from evidence_fashion.verification_analysis import (
    claim_role,
    classify_refusal,
    normalization_reason_codes,
    source_bucket,
    visible_status,
)


def test_claim_role_is_schema_only_and_conservative() -> None:
    assert claim_role("item_type") == "identity_context"
    assert claim_role("styling_relation") == "substantive"
    assert claim_role("colour") == "substantive"
    assert claim_role("other") == "substantive"


def test_visible_status_does_not_count_hidden_b_for_no_rag() -> None:
    b_only = {"support_status": "supported", "support_sources": ["rule_evidence"]}
    both = {
        "support_status": "supported",
        "support_sources": ["query_or_locked_item", "rule_evidence"],
    }
    assert source_bucket(b_only) == "b_only"
    assert visible_status("no_rag", b_only) == "unsupported"
    assert visible_status("rule_rag", b_only) == "supported"
    assert source_bucket(both) == "both_a_b"
    assert visible_status("no_rag", both) == "supported"


def test_non_supported_claims_are_in_neither_source_bucket() -> None:
    claim = {"support_status": "unsupported", "support_sources": ["rule_evidence"]}
    assert source_bucket(claim) == "neither"
    assert visible_status("no_rag", claim) == "unsupported"


def test_normalization_reasons_are_reconstructed_from_raw_fields() -> None:
    raw = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "C1",
                    "support_status": "supported",
                    "support_sources": ["rule_evidence", "rule_evidence"],
                    "supporting_rule_ids": ["R999"],
                    "citation_entails_claim": True,
                }
            ]
        }
    )
    final = [
        {
            "claim_id": "C1",
            "support_status": "not_verifiable",
            "support_sources": [],
            "supporting_rule_ids": [],
            "citation_entails_claim": None,
        }
    ]
    reasons = normalization_reason_codes(
        raw, final, allowed_rule_ids={"R001"}, citation_ids=[]
    )
    assert "duplicate_support_sources_removed" in reasons
    assert "invalid_rule_ids_removed" in reasons
    assert "sourceless_support_relabelled_not_verifiable" in reasons
    assert "uncited_entailment_set_null" in reasons


def test_refusal_audit_distinguishes_genuine_and_non_refusal_text() -> None:
    assert classify_refusal("I can't provide guidance on choosing eyewear.") == (
        "genuine_refusal",
        "misguided_eyewear_safety_refusal",
    )
    assert (
        classify_refusal("There is not enough evidence to assess the colour.")[0]
        == "non_refusal"
    )
