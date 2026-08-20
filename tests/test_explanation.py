import pytest

from evidence_fashion.explanation import (
    ASSESSMENT_SCHEMA,
    build_no_rag_prompt,
    build_rule_rag_prompt,
    common_context,
)


def _case() -> dict:
    return {
        "request": "Recommend a bag.",
        "query_item_minimal_name": "blue blouse",
        "locked_candidate_minimal_name": "black shoulder bag",
        "evidence_trace": {
            "candidate_item_id": "candidate-1",
            "representation_hash": "hash",
            "target_category": "accessories",
            "evidence_score": 0.8,
            "filtering": {"category_filter": "accessories"},
            "rules": [
                {
                    "rule_id": "R2",
                    "rule_text": "A structured bag can balance a soft top.",
                    "semantic_similarity": 0.8,
                    "reliability_label": "high",
                    "reliability_weight": 1.0,
                    "query_group_bonus": 0.0,
                    "weighted_contribution": 0.8,
                    "rank": 1,
                    "filtering_decision": "retained_after_category_filter_and_top_k",
                    "antecedent_established": True,
                    "antecedent_checks": {"query_terms": True, "candidate_terms": True},
                }
            ],
        },
    }


def _settings() -> dict:
    return {
        "id": "test",
        "word_limit": 60,
        "rule_count": 1,
        "evidence_format": "compact_json",
        "evidence_ordering": "retrieval_rank",
        "citation_mode": "required",
        "include_rule_scores": True,
        "include_reliability_labels": True,
        "grounding_prompt_variant": "concise",
    }


def test_common_context_contains_only_approved_A_fields() -> None:
    assert set(common_context(_case())) == {
        "user_request",
        "query_item_minimal_name",
        "locked_item_minimal_name",
    }


def test_no_rag_prompt_has_no_rule_citation_or_grounding_instruction() -> None:
    prompt = build_no_rag_prompt(_case())
    assert "R2" not in prompt
    assert "citation" not in prompt.lower()
    assert "words" not in prompt.lower()
    assert "only" not in prompt.lower()
    assert "do not invent" not in prompt.lower()
    assert "evidence" not in prompt.lower()


def test_no_rag_prompt_can_share_rule_rag_word_limit_without_evidence() -> None:
    prompt = build_no_rag_prompt(_case(), 75)
    assert "at most 75 words" in prompt
    assert "citation" not in prompt.lower()
    assert "evidence" not in prompt.lower()


def test_rule_prompt_is_derived_from_stored_trace_and_records_ids() -> None:
    prompt, rule_ids = build_rule_rag_prompt(_case(), _settings())
    assert rule_ids == ["R2"]
    assert "R2" in prompt
    assert "60 words" in prompt
    assert "A structured bag can balance a soft top." in prompt


def test_rule_prompt_rejects_a_rule_count_that_differs_from_complete_trace() -> None:
    settings = {**_settings(), "rule_count": 3}
    with pytest.raises(ValueError, match="complete stored reranking trace"):
        build_rule_rag_prompt(_case(), settings)


def test_assessment_schema_allows_multiple_support_sources_without_claim_cap() -> None:
    claim = ASSESSMENT_SCHEMA["$defs"]["claim"]["properties"]
    assert claim["support_source"]["type"] == "array"
    condition = ASSESSMENT_SCHEMA["$defs"]["condition"]["properties"]
    assert "maxItems" not in condition["claims"]
