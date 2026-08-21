"""Shared grounding invariants for rules, citations, and locked recommendations."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .kb_audit import declared_values, matches_declared_terms

CANONICAL_RULE_ID_RE = re.compile(r"K\d{3}\Z")
CANONICAL_CITATION_RE = re.compile(r"\[K\d{3}\]")
BRACKETED_RE = re.compile(r"\[[^\]]*\]")
RULE_LIKE_ID_RE = re.compile(r"\b[A-Za-z]\d{3}\b")


@dataclass(frozen=True)
class RuleApplicabilityDecision:
    """Auditable antecedent decision made before a rule enters an exact trace."""

    rule_id: str
    established: bool
    checks: dict[str, bool]

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "antecedent_established": self.established,
            "antecedent_checks": self.checks,
        }


def rule_applicability_gate(
    rule: Mapping[str, Any], *, case: Mapping[str, Any], candidate: Mapping[str, Any]
) -> RuleApplicabilityDecision:
    """Fail closed unless every declared rule antecedent is established by case data."""
    query_group = str(case.get("query_group", case.get("query_category", ""))).lower()
    observed_context = case.get("applicability_contexts", [])
    if isinstance(observed_context, str):
        observed_context = observed_context.split("|")
    query_text = " | ".join(
        str(case.get(field, ""))
        for field in (
            "query_category",
            "query_group",
            "query_text",
            "outfit_context_text",
            "user_request",
        )
    )
    candidate_text = " | ".join(str(candidate.get(field, "")) for field in ("category", "text"))
    declared_query_groups = declared_values(rule["applicable_query_categories"])
    required_context = declared_values(rule["required_context"])
    checks = {
        "target_category": str(rule["recommended_category"]) == str(case["target_category"]),
        "query_group": query_group in declared_query_groups or "all" in declared_query_groups,
        "required_context": "none" in required_context
        or required_context.issubset({str(value).lower() for value in observed_context}),
        "query_terms": matches_declared_terms(rule["query_terms"], query_text),
        "candidate_terms": matches_declared_terms(rule["candidate_terms"], candidate_text),
    }
    return RuleApplicabilityDecision(
        rule_id=str(rule["rule_id"]), established=all(checks.values()), checks=checks
    )


def require_trace_applicability(trace: Mapping[str, Any]) -> None:
    """Reject a supplied exact trace unless every retained rule carries a passed gate."""
    rules = trace.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("An exact rule trace must contain at least one rule.")
    invalid = [
        str(rule.get("rule_id", ""))
        for rule in rules
        if not isinstance(rule, Mapping)
        or rule.get("antecedent_established") is not True
        or not isinstance(rule.get("antecedent_checks"), Mapping)
        or not all(bool(value) for value in rule["antecedent_checks"].values())
    ]
    if invalid:
        raise ValueError(f"Exact trace contains rule(s) without established antecedents: {invalid}")


def citation_occurrences(
    text: str,
    *,
    known_rule_ids: Sequence[str] = (),
    trace_rule_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Return occurrence-level K-citation diagnostics without dropping malformed brackets.

    Syntax is entirely deterministic.  A canonical citation is exactly one ``[K###]``
    bracket; an unknown or out-of-trace ID is not a valid occurrence even if its
    bracket syntax is canonical.  The optional sets intentionally let production
    generation, Stage 5, and final evaluation share the same parser.
    """
    known = set(known_rule_ids)
    trace = set(trace_rule_ids)
    occurrences = []
    for index, raw in enumerate(BRACKETED_RE.findall(text), start=1):
        ids = RULE_LIKE_ID_RE.findall(raw)
        canonical_syntax = bool(CANONICAL_CITATION_RE.fullmatch(raw))
        duplicate_ids = sorted({rule_id for rule_id in ids if ids.count(rule_id) > 1})
        unknown_ids = sorted(set(ids).difference(known)) if known else []
        out_of_trace_ids = sorted(set(ids).difference(trace)) if trace else []
        valid = (
            canonical_syntax
            and len(ids) == 1
            and not duplicate_ids
            and not unknown_ids
            and not out_of_trace_ids
        )
        occurrences.append(
            {
                "occurrence_index": index,
                "raw": raw,
                "rule_ids": ids,
                "canonical_syntax": canonical_syntax,
                "duplicate_rule_ids": duplicate_ids,
                "unknown_rule_ids": unknown_ids,
                "out_of_trace_rule_ids": out_of_trace_ids,
                "valid_canonical_occurrence": valid,
            }
        )
    return occurrences


def canonical_citation_ids(text: str) -> list[str]:
    """Return unique K IDs, rejecting grouped, foreign, and malformed rule citations."""
    ids: list[str] = []
    for occurrence in citation_occurrences(text):
        if not occurrence["canonical_syntax"]:
            raise ValueError(
                "Grouped or malformed rule citations must use separate canonical [K###] brackets; "
                f"found {occurrence['raw']!r}."
            )
        ids.extend(occurrence["rule_ids"])
    return list(dict.fromkeys(ids))


def require_citations_in_trace(
    text: str, trace_rule_ids: Sequence[str], *, required: bool
) -> list[str]:
    citation_ids = canonical_citation_ids(text)
    if required and not citation_ids:
        raise ValueError("A rule-grounded explanation requires at least one canonical citation.")
    invalid = set(citation_ids).difference(trace_rule_ids)
    if invalid:
        raise ValueError(f"Explanation cites rule(s) outside its exact trace: {sorted(invalid)}")
    return citation_ids


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def require_locked_recommendation(
    explanation: str, *, locked_item_name: str, target_category: str
) -> None:
    """Require the response to name the exact locked item and retain its target category."""
    locked = _normalized(locked_item_name)
    rendered = _normalized(explanation)
    if not locked or locked not in rendered:
        raise ValueError("Explanation does not preserve the exact locked recommendation.")
    if not _normalized(target_category):
        raise ValueError("Locked recommendation has no target category.")


def validate_generated_explanation(
    explanation: str,
    *,
    locked_item_name: str,
    target_category: str,
    trace_rule_ids: Sequence[str] = (),
    citations_required: bool = False,
) -> list[str]:
    """Apply the shared post-generation contract before a response is persisted."""
    require_locked_recommendation(
        explanation, locked_item_name=locked_item_name, target_category=target_category
    )
    return require_citations_in_trace(explanation, trace_rule_ids, required=citations_required)
