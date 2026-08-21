import pytest

from evidence_fashion.grounding_contracts import (
    canonical_citation_ids,
    citation_occurrences,
    require_trace_applicability,
    rule_applicability_gate,
    validate_generated_explanation,
)


def _rule() -> dict:
    return {
        "rule_id": "K001",
        "recommended_category": "tops",
        "applicable_query_categories": "bottoms",
        "required_context": "none",
        "query_terms": "wide leg",
        "candidate_terms": "cardigan",
    }


def _case() -> dict:
    return {
        "target_category": "tops",
        "query_group": "bottoms",
        "query_category": "bottoms",
        "query_text": "wide leg trousers",
        "outfit_context_text": "Bottoms: wide leg trousers",
        "user_request": "Recommend a top.",
    }


def _valid_explanation(citation: str = "[K001]") -> str:
    return (
        "The rib cardigan is the recommended item because it gives the outfit a balanced upper "
        "layer while keeping proportions clear with wide-leg trousers. Its simple cardigan role "
        "supports the requested top category without assuming colour, fabric, occasion, or other "
        "unprovided details. This explanation stays focused on the locked recommendation and the "
        f"supplied rule evidence. {citation}"
    )


def test_shared_gate_fails_closed_and_exact_trace_requires_the_passed_decision() -> None:
    decision = rule_applicability_gate(
        _rule(), case=_case(), candidate={"category": "Tops", "text": "rib cardigan"}
    )
    assert decision.established is True
    require_trace_applicability({"rules": [{"rule_id": "K001", **decision.trace_metadata()}]})
    failed = rule_applicability_gate(
        _rule(), case=_case(), candidate={"category": "Tops", "text": "silk shirt"}
    )
    assert failed.established is False
    with pytest.raises(ValueError, match="without established antecedents"):
        require_trace_applicability({"rules": [{"rule_id": "K001", **failed.trace_metadata()}]})


def test_locked_item_and_citation_contracts_reject_drift_grouping_and_out_of_trace_ids() -> None:
    valid = _valid_explanation()
    assert validate_generated_explanation(
        valid,
        locked_item_name="rib cardigan",
        target_category="tops",
        trace_rule_ids=["K001"],
        citations_required=True,
    ) == ["K001"]
    with pytest.raises(ValueError, match="locked recommendation"):
        validate_generated_explanation(
            "A silk shirt is recommended. [K001]",
            locked_item_name="rib cardigan",
            target_category="tops",
            trace_rule_ids=["K001"],
            citations_required=True,
        )


def test_citation_occurrences_preserve_grouped_duplicate_and_unknown_diagnostics() -> None:
    diagnostics = citation_occurrences(
        "Evidence [K001, K001] [K999] [bad].", known_rule_ids=["K001"], trace_rule_ids=["K001"]
    )
    assert [row["raw"] for row in diagnostics] == ["[K001, K001]", "[K999]", "[bad]"]
    assert diagnostics[0]["duplicate_rule_ids"] == ["K001"]
    assert diagnostics[1]["unknown_rule_ids"] == ["K999"]
    assert all(not row["valid_canonical_occurrence"] for row in diagnostics)
    with pytest.raises(ValueError, match="Grouped"):
        canonical_citation_ids("The rib cardigan works. [K001, K002]")
    with pytest.raises(ValueError, match="outside"):
        validate_generated_explanation(
            _valid_explanation("[K002]"),
            locked_item_name="rib cardigan",
            target_category="tops",
            trace_rule_ids=["K001"],
            citations_required=True,
        )
