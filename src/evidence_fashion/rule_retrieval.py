"""Unified candidate-specific expert-rule retrieval, scoring, and exact evidence traces."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .grounding_contracts import rule_applicability_gate
from .kb_audit import declared_values
from .retrieval import l2_normalize


def full_kb_candidate_retrieval(
    rules: pd.DataFrame, *, target_category: str, query_group: str
) -> list[dict[str, Any]]:
    """Return verification candidates only; this deliberately does not assess antecedents."""
    candidates = rules[
        rules["recommended_category"].astype(str).eq(target_category)
        & rules["audit_status"].astype(str).eq("retain")
        & rules["applicable_query_categories"]
        .astype(str)
        .map(
            lambda value: (
                query_group.lower() in declared_values(value) or "all" in declared_values(value)
            )
        )
    ]
    return candidates.sort_values("rule_id").to_dict(orient="records")


@dataclass(frozen=True)
class RuleContribution:
    rule_id: str
    rule_text: str
    semantic_similarity: float
    reliability_label: str
    reliability_weight: float
    query_group_bonus: float
    weighted_contribution: float
    retrieval_rank: int
    filtering_decision: str
    antecedent_established: bool
    antecedent_checks: dict[str, bool]


@dataclass(frozen=True)
class CandidateEvidenceTrace:
    candidate_id: str
    evidence_score: float
    query_group: str
    target_category: str
    representation_sha256: str
    filtering: dict[str, int | str]
    rules: tuple[RuleContribution, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_score": self.evidence_score,
            "query_group": self.query_group,
            "target_category": self.target_category,
            "representation_sha256": self.representation_sha256,
            "filtering": self.filtering,
            "rules": [asdict(rule) for rule in self.rules],
        }


def candidate_rule_representation(case: dict[str, Any], candidate: dict[str, Any]) -> str:
    return " | ".join(
        [
            f"Query category: {case['query_category']}",
            f"Query text: {case['query_text']}",
            f"User request: {case['user_request']}",
            f"Candidate category: {candidate['category']}",
            f"Candidate text: {candidate['text']}",
            f"Target category: {case['target_category']}",
        ]
    )


class RuleRetriever:
    """One shared function for filtering, top-k, scoring, storage, and later B evidence."""

    def __init__(
        self,
        rules: pd.DataFrame,
        rule_embeddings: np.ndarray,
        settings: dict[str, Any],
    ) -> None:
        required = {
            "rule_id",
            "rule_text",
            "input_category",
            "recommended_category",
            "source_reliability",
            "audit_status",
            "applicable_query_categories",
            "required_context",
            "query_terms",
            "candidate_terms",
        }
        missing = required.difference(rules.columns)
        if missing:
            raise ValueError(f"Knowledge base is missing fields: {sorted(missing)}")
        if len(rules) != len(rule_embeddings):
            raise ValueError("Rules and rule embeddings must be row-aligned.")
        self.rules = rules.reset_index(drop=True).copy()
        self.rule_embeddings = l2_normalize(rule_embeddings)
        self.settings = settings

    def retrieve_and_score(
        self,
        *,
        case: dict[str, Any],
        candidate: dict[str, Any],
        representation_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> CandidateEvidenceTrace:
        selected_top_k = top_k or self.settings["candidate_top_k"]
        target = str(case["target_category"])
        eligible_mask = self.rules[self.settings["category_filter_field"]].astype(str).eq(target)
        category_eligible_count = int(eligible_mask.sum())
        audit_excluded = 0
        applicability_excluded = 0
        context_excluded = 0
        query_terms_excluded = 0
        candidate_terms_excluded = 0
        audit_field = self.settings["audit_status_field"]
        approved = self.rules[audit_field].astype(str).eq(self.settings["approved_audit_status"])
        audit_excluded = category_eligible_count - int((eligible_mask & approved).sum())
        decisions = [
            rule_applicability_gate(rule, case=case, candidate=candidate)
            for rule in self.rules.to_dict(orient="records")
        ]
        decision_by_index = dict(enumerate(decisions))
        query_group = str(case["query_group"])
        applicable = pd.Series(
            [decision.checks["query_group"] for decision in decisions], index=self.rules.index
        )
        applicability_excluded = int((eligible_mask & approved & ~applicable).sum())
        context_applicable = pd.Series(
            [decision.checks["required_context"] for decision in decisions],
            index=self.rules.index,
        )
        context_excluded = int((eligible_mask & approved & applicable & ~context_applicable).sum())
        query_matches = pd.Series(
            [decision.checks["query_terms"] for decision in decisions], index=self.rules.index
        )
        query_terms_excluded = int(
            (eligible_mask & approved & applicable & context_applicable & ~query_matches).sum()
        )
        candidate_matches = pd.Series(
            [decision.checks["candidate_terms"] for decision in decisions], index=self.rules.index
        )
        candidate_terms_excluded = int(
            (
                eligible_mask
                & approved
                & applicable
                & context_applicable
                & query_matches
                & ~candidate_matches
            ).sum()
        )
        eligible_mask &= (
            approved & applicable & context_applicable & query_matches & candidate_matches
        )
        eligible_rows = np.flatnonzero(eligible_mask.to_numpy())
        representation = candidate_rule_representation(case, candidate)
        if not len(eligible_rows):
            # A sparse KB must not cause an otherwise valid recommendation case to
            # disappear.  An empty trace is explicit evidence that no rule passed
            # the antecedent gate; it is never backfilled with a merely similar
            # rule.  Downstream Rule-RAG selection excludes empty traces.
            return CandidateEvidenceTrace(
                candidate_id=str(candidate["item_id"]),
                evidence_score=0.0,
                query_group=query_group,
                target_category=target,
                representation_sha256=hashlib.sha256(representation.encode()).hexdigest(),
                filtering={
                    "rules_before_filter": len(self.rules),
                    "category_filter": target,
                    "rules_after_category_filter": 0,
                    "rules_excluded_by_category": len(self.rules) - category_eligible_count,
                    "rules_excluded_by_audit": audit_excluded,
                    "rules_excluded_by_applicability": applicability_excluded,
                    "rules_excluded_by_context": context_excluded,
                    "rules_excluded_by_query_terms": query_terms_excluded,
                    "rules_excluded_by_candidate_terms": candidate_terms_excluded,
                    "top_k_requested": selected_top_k,
                    "rules_retained": 0,
                    "rules_not_selected_after_scoring": 0,
                    "empty_trace_reason": "no_rule_with_established_antecedent",
                },
                rules=(),
            )
        vector = l2_normalize(np.asarray(representation_embedding).reshape(1, -1))[0]
        similarities = self.rule_embeddings[eligible_rows] @ vector
        scored = []
        query_group = str(case["query_group"])
        for local_index, rule_row in enumerate(eligible_rows):
            rule = self.rules.iloc[rule_row]
            reliability_label = str(rule["source_reliability"]).lower()
            reliability_weight = float(self.settings["reliability_weights"][reliability_label])
            bonus = (
                float(self.settings["query_group_bonus"])
                if str(rule["input_category"]) == query_group
                else 0.0
            )
            contribution = float(similarities[local_index]) * reliability_weight + bonus
            scored.append(
                (
                    -contribution,
                    str(rule["rule_id"]),
                    int(rule_row),
                    float(similarities[local_index]),
                    reliability_label,
                    reliability_weight,
                    bonus,
                    contribution,
                )
            )
        scored.sort(key=lambda value: (value[0], value[1]))
        retained = scored[:selected_top_k]
        contributions = tuple(
            RuleContribution(
                rule_id=rule_id,
                rule_text=str(self.rules.iloc[row_index]["rule_text"]),
                semantic_similarity=similarity,
                reliability_label=reliability_label,
                reliability_weight=reliability_weight,
                query_group_bonus=bonus,
                weighted_contribution=weighted,
                retrieval_rank=rank,
                filtering_decision="retained_after_category_filter_and_top_k",
                antecedent_established=decision_by_index[row_index].established,
                antecedent_checks=decision_by_index[row_index].checks,
            )
            for rank, (
                _,
                rule_id,
                row_index,
                similarity,
                reliability_label,
                reliability_weight,
                bonus,
                weighted,
            ) in enumerate(retained, start=1)
        )
        weighted_scores = np.asarray(
            [rule.weighted_contribution for rule in contributions], dtype=np.float64
        )
        evidence_score = float(
            self.settings["score_max_weight"] * weighted_scores.max()
            + self.settings["score_mean_weight"] * weighted_scores.mean()
        )
        return CandidateEvidenceTrace(
            candidate_id=str(candidate["item_id"]),
            evidence_score=evidence_score,
            query_group=query_group,
            target_category=target,
            representation_sha256=hashlib.sha256(representation.encode()).hexdigest(),
            filtering={
                "rules_before_filter": len(self.rules),
                "category_filter": target,
                "rules_after_category_filter": len(eligible_rows),
                "rules_excluded_by_category": len(self.rules) - category_eligible_count,
                "rules_excluded_by_audit": audit_excluded,
                "rules_excluded_by_applicability": applicability_excluded,
                "rules_excluded_by_context": context_excluded,
                "rules_excluded_by_query_terms": query_terms_excluded,
                "rules_excluded_by_candidate_terms": candidate_terms_excluded,
                "top_k_requested": selected_top_k,
                "rules_retained": len(contributions),
                "rules_not_selected_after_scoring": len(eligible_rows) - len(contributions),
            },
            rules=contributions,
        )


def truncate_trace(
    trace: CandidateEvidenceTrace,
    top_k: int,
    settings: dict[str, Any],
) -> CandidateEvidenceTrace:
    """Return the exact top-k scoring trace and recompute its evidence score."""
    rules = trace.rules[:top_k]
    if not rules:
        raise ValueError("An evidence trace must retain at least one rule.")
    scores = np.asarray([rule.weighted_contribution for rule in rules], dtype=np.float64)
    evidence_score = float(
        settings["score_max_weight"] * scores.max() + settings["score_mean_weight"] * scores.mean()
    )
    filtering = dict(trace.filtering)
    filtering["top_k_requested"] = top_k
    filtering["rules_retained"] = len(rules)
    filtering["rules_not_selected_after_scoring"] = int(
        filtering["rules_after_category_filter"]
    ) - len(rules)
    return CandidateEvidenceTrace(
        candidate_id=trace.candidate_id,
        evidence_score=evidence_score,
        query_group=trace.query_group,
        target_category=trace.target_category,
        representation_sha256=trace.representation_sha256,
        filtering=filtering,
        rules=rules,
    )
