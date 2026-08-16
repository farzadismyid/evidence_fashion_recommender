import csv
import json
from pathlib import Path

import pytest

from evidence_fashion.study_metrics import (
    classify_item_attribute,
    dta_entailed,
    exact_b_entails_attribute,
    explicit_a_entails_attribute,
    strip_condition_revealing_citations,
    valid_rule_citation,
)

ROOT = Path(__file__).parents[1]


def _context() -> dict[str, str]:
    return {
        "user_request": "Recommend bottoms that work with this outfit.",
        "query_item_id": "q1",
        "query_item_category": "shoes",
        "query_item_text": "Christian Louboutin satin pumps black",
        "locked_candidate_id": "c1",
        "locked_item_category": "bottoms",
        "locked_item_text": "Theory silk straight-leg pants charcoal",
    }


def test_uiar_uses_all_explicit_A_text_without_brand_inference() -> None:
    supported = {
        "claim_type": "material",
        "claim_text": "The recommended Theory pants are made from silk.",
    }
    unsupported = {
        "claim_type": "material",
        "claim_text": "The recommended Theory pants use premium cashmere.",
    }
    assert classify_item_attribute(supported, _context()).bucket == "item_attribute"
    assert explicit_a_entails_attribute(supported["claim_text"], _context())
    assert not explicit_a_entails_attribute(unsupported["claim_text"], _context())


def test_A_attribute_support_cannot_leak_between_query_and_recommended_items() -> None:
    assert not explicit_a_entails_attribute("The recommended Theory pants are black.", _context())
    assert explicit_a_entails_attribute("The Louboutin pumps are black.", _context())


def test_generic_rule_does_not_establish_instance_level_attribute() -> None:
    rules = [
        {
            "rule_id": "R101",
            "rule_text": "Recommend bottoms whose colour is compatible with the existing outfit.",
        }
    ]
    claim = "The recommended Theory pants are charcoal and match the pumps."
    assert not exact_b_entails_attribute(claim, _context(), rules)


def test_relational_styling_judgment_is_not_forced_into_uiar() -> None:
    claim = {
        "claim_type": "visual_match",
        "claim_text": "The charcoal pants balance the black pumps.",
    }
    assert classify_item_attribute(claim, _context()).bucket == "ambiguous"


def test_dta_and_citation_validity_require_exact_B_attribution_and_observed_id() -> None:
    claim = {
        "support_status": "supported",
        "support_sources": ["rule_evidence"],
        "supporting_rule_ids": ["R101"],
        "citation_entails_claim": True,
    }
    assert dta_entailed(claim)
    assert valid_rule_citation(claim, ["R101"])
    assert not valid_rule_citation(claim, ["R099"])


def test_blinded_rendering_removes_rule_ids_and_citation_markers() -> None:
    rendered = strip_condition_revealing_citations(
        "Based on Rule R101, these work [R101, R099]. Rule R099 also applies."
    )
    assert "R101" not in rendered
    assert "R099" not in rendered
    assert "[" not in rendered
    assert "rule" not in rendered.casefold()
    assert "evidence" not in rendered.casefold()


@pytest.mark.skip(reason="Stage 8 outputs belong to the archived experiment")
def test_canonical_study_metrics_preserve_baselines_and_no_rag_citations_are_na() -> None:
    manifest = json.loads(
        (ROOT / "artifacts/manifests/stage8_study_metrics_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["model_calls"] == 0
    assert manifest["preserved_baselines"]["canonical_baselines_modified"] is False
    assert (
        manifest["length_matched_sensitivity"]["role"]
        == "sensitivity_only_not_primary_confirmatory_estimate"
    )
    with (ROOT / "artifacts/tables/table_stage8_study_specific_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    full_micro = {
        row["condition"]: row
        for row in rows
        if row["analysis_subset"] == "full_corpus"
        and row["scope"] == "overall"
        and row["aggregation"] == "micro_claim"
    }
    assert full_micro["no_rag"]["citation_precision"] == ""
    assert full_micro["no_rag"]["citation_coverage"] == ""
    # The cross-model verifier found higher post-hoc B alignment for No-RAG.
    # Preserve this negative result instead of encoding the original hypothesis.
    assert float(full_micro["rule_rag"]["dta_rate"]) < float(full_micro["no_rag"]["dta_rate"])
