import numpy as np
import pandas as pd
import pytest

from evidence_fashion.rule_retrieval import (
    RuleRetriever,
    candidate_rule_representation,
    truncate_trace,
)


def _settings() -> dict:
    return {
        "candidate_top_k": 2,
        "category_filter_field": "recommended_category",
        "applicability_filter_field": "applicable_query_categories",
        "required_context_field": "required_context",
        "query_terms_field": "query_terms",
        "candidate_terms_field": "candidate_terms",
        "audit_status_field": "audit_status",
        "approved_audit_status": "retain",
        "reliability_weights": {"high": 1.0, "medium": 0.85, "low": 0.65},
        "query_group_bonus": 0.1,
        "score_max_weight": 0.7,
        "score_mean_weight": 0.3,
    }


def _rules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_id": "R2",
                "rule_text": "medium shoe rule",
                "input_category": "dresses",
                "recommended_category": "shoes",
                "source_reliability": "medium",
                "audit_status": "retain",
                "applicable_query_categories": "dresses",
                "required_context": "none",
                "query_terms": "none",
                "candidate_terms": "none",
            },
            {
                "rule_id": "R1",
                "rule_text": "high shoe rule",
                "input_category": "dresses",
                "recommended_category": "shoes",
                "source_reliability": "high",
                "audit_status": "retain",
                "applicable_query_categories": "dresses",
                "required_context": "none",
                "query_terms": "none",
                "candidate_terms": "none",
            },
            {
                "rule_id": "R3",
                "rule_text": "top rule",
                "input_category": "dresses",
                "recommended_category": "tops",
                "source_reliability": "high",
                "audit_status": "retain",
                "applicable_query_categories": "dresses",
                "required_context": "none",
                "query_terms": "none",
                "candidate_terms": "none",
            },
        ]
    )


def _case() -> dict:
    return {
        "query_category": "Day Dresses",
        "query_group": "dresses",
        "query_text": "day dress",
        "user_request": "Recommend shoes.",
        "target_category": "shoes",
    }


def _candidate() -> dict:
    return {"item_id": "c1", "category": "Pumps", "text": "black pumps"}


def test_filter_happens_before_top_k_and_trace_reproduces_score() -> None:
    embeddings = np.array([[0.8, 0.6], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    retriever = RuleRetriever(_rules(), embeddings, _settings())
    trace = retriever.retrieve_and_score(
        case=_case(), candidate=_candidate(), representation_embedding=np.array([1.0, 0.0])
    )
    assert [rule.rule_id for rule in trace.rules] == ["R1", "R2"]
    assert trace.filtering["rules_after_category_filter"] == 2
    scores = [rule.weighted_contribution for rule in trace.rules]
    assert trace.evidence_score == pytest.approx(0.7 * max(scores) + 0.3 * np.mean(scores))
    assert all(rule.filtering_decision.startswith("retained") for rule in trace.rules)


def test_reliability_weight_and_query_bonus_are_recorded() -> None:
    embeddings = np.array([[0.8, 0.6], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    retriever = RuleRetriever(_rules(), embeddings, _settings())
    trace = retriever.retrieve_and_score(
        case=_case(), candidate=_candidate(), representation_embedding=np.array([1.0, 0.0])
    )
    medium = next(rule for rule in trace.rules if rule.rule_id == "R2")
    assert medium.reliability_weight == 0.85
    assert medium.query_group_bonus == 0.1


def test_truncated_trace_recomputes_exact_score_and_filtering() -> None:
    embeddings = np.array([[0.8, 0.6], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    retriever = RuleRetriever(_rules(), embeddings, _settings())
    full = retriever.retrieve_and_score(
        case=_case(), candidate=_candidate(), representation_embedding=np.array([1.0, 0.0])
    )
    exact = truncate_trace(full, 1, _settings())
    assert [rule.rule_id for rule in exact.rules] == ["R1"]
    assert exact.evidence_score == pytest.approx(exact.rules[0].weighted_contribution)
    assert exact.filtering["top_k_requested"] == 1
    assert exact.filtering["rules_retained"] == 1


def test_candidate_representation_contains_only_approved_dataset_and_request_fields() -> None:
    text = candidate_rule_representation(_case(), _candidate())
    assert "day dress" in text
    assert "black pumps" in text
    assert "image" not in text.lower()


def test_query_term_conjunction_yields_empty_trace_until_every_clause_is_present() -> None:
    rules = _rules().iloc[[0]].copy()
    rules.loc[:, "query_terms"] = "dress & smart casual|smart-casual"
    retriever = RuleRetriever(rules, np.array([[1.0, 0.0]]), _settings())
    empty = retriever.retrieve_and_score(
        case=_case(),
        candidate=_candidate(),
        representation_embedding=np.array([1.0, 0.0]),
    )
    assert empty.rules == ()
    assert empty.evidence_score == 0.0
    assert empty.filtering["empty_trace_reason"] == "no_rule_with_established_antecedent"
    case = _case()
    case["user_request"] = "Recommend shoes for a smart-casual look."
    trace = retriever.retrieve_and_score(
        case=case,
        candidate=_candidate(),
        representation_embedding=np.array([1.0, 0.0]),
    )
    assert [rule.rule_id for rule in trace.rules] == ["R2"]
