"""Deterministic Stage 8 verification re-aggregation helpers."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

IDENTITY_CONTEXT_TYPES = frozenset({"item_type"})
SUPPORT_A = "query_or_locked_item"
SUPPORT_B = "rule_evidence"
ALLOWED_SOURCES = frozenset({SUPPORT_A, SUPPORT_B})


def claim_role(claim_type: str) -> str:
    """Apply the conservative schema-only role split used in the revision.

    ``item_type`` is the only schema label dedicated to identity/category claims. Other
    labels are treated as substantive because colour, material, and ``other`` mix factual
    and explanatory uses that cannot be split reproducibly from the saved schema alone.
    """
    return "identity_context" if claim_type in IDENTITY_CONTEXT_TYPES else "substantive"


def source_bucket(claim: Mapping[str, Any]) -> str:
    """Return mutually exclusive support provenance for an A+B verification decision."""
    if claim["support_status"] != "supported":
        return "neither"
    sources = set(claim.get("support_sources", []))
    has_a = SUPPORT_A in sources
    has_b = SUPPORT_B in sources
    if has_a and has_b:
        return "both_a_b"
    if has_a:
        return "a_only"
    if has_b:
        return "b_only"
    return "neither"


def visible_status(condition: str, claim: Mapping[str, Any]) -> str:
    """Project the preserved A+B decision onto evidence visible during generation."""
    status = str(claim["support_status"])
    if condition == "rule_rag":
        return status
    if condition != "no_rag":
        raise ValueError(f"Unknown condition: {condition}")
    if status == "supported" and SUPPORT_A in claim.get("support_sources", []):
        return "supported"
    if status == "contradicted":
        return "contradicted"
    if status == "not_verifiable":
        return "not_verifiable"
    # This includes B-only A+B support: it is post-hoc alignment, not visible grounding.
    return "unsupported"


def normalization_reason_codes(
    raw_text: str,
    final_claims: Sequence[Mapping[str, Any]],
    *,
    allowed_rule_ids: set[str],
    citation_ids: Sequence[str],
) -> set[str]:
    """Reconstruct deterministic normalization reasons from preserved raw/final fields."""
    raw = json.loads(raw_text)
    raw_claims = raw.get("claims", [])
    reasons: set[str] = set()
    for before, after in zip(raw_claims, final_claims, strict=False):
        raw_sources = before.get("support_sources")
        raw_rules = before.get("supporting_rule_ids")
        if isinstance(raw_sources, list):
            if len(raw_sources) != len(dict.fromkeys(raw_sources)):
                reasons.add("duplicate_support_sources_removed")
            if any(source not in ALLOWED_SOURCES for source in raw_sources):
                reasons.add("invalid_support_sources_removed")
        if isinstance(raw_rules, list):
            if len(raw_rules) != len(dict.fromkeys(raw_rules)):
                reasons.add("duplicate_rule_ids_removed")
            if any(rule_id not in allowed_rule_ids for rule_id in raw_rules):
                reasons.add("invalid_rule_ids_removed")
        before_sources = set(raw_sources) if isinstance(raw_sources, list) else set()
        after_sources = set(after.get("support_sources", []))
        before_rules = set(raw_rules) if isinstance(raw_rules, list) else set()
        valid_before_rules = before_rules & allowed_rule_ids
        if valid_before_rules and SUPPORT_B not in before_sources and SUPPORT_B in after_sources:
            reasons.add("rule_source_added_for_valid_rule_ids")
        if (
            not valid_before_rules
            and SUPPORT_B in before_sources
            and SUPPORT_B not in after_sources
        ):
            reasons.add("rule_source_removed_without_valid_rule_ids")
        if before.get("support_status") == "supported" and after.get(
            "support_status"
        ) == "not_verifiable":
            reasons.add("sourceless_support_relabelled_not_verifiable")
        if not citation_ids and before.get("citation_entails_claim") is not None and after.get(
            "citation_entails_claim"
        ) is None:
            reasons.add("uncited_entailment_set_null")
    return reasons


def classify_refusal(text: str) -> tuple[str, str]:
    """Classify a Stage 8 refusal deterministically without altering its N/A status."""
    lowered = text.casefold()
    refusal_phrases = (
        "i can't provide",
        "i cannot provide",
        "i can't fulfill",
        "i cannot fulfill",
        "i can't help",
        "i cannot help",
    )
    if not any(phrase in lowered for phrase in refusal_phrases):
        return "non_refusal", "no_first_person_refusal_phrase"
    if "choosing eyewear" in lowered:
        return "genuine_refusal", "misguided_eyewear_safety_refusal"
    if "cheap women's outerwear" in lowered or "cheap womens outerwear" in lowered:
        return "genuine_refusal", "misguided_product_safety_refusal"
    if "explicit content" in lowered or "illegal or harmful" in lowered:
        return "genuine_refusal", "content_safety_refusal"
    return "genuine_refusal", "generic_refusal"


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def aggregate_claims(
    explanations: Sequence[Mapping[str, Any]], aggregation: str
) -> dict[str, Any]:
    """Aggregate saved claim rows at claim-micro or explanation-macro level."""
    substantive = [
        claim
        for explanation in explanations
        for claim in explanation["claims"]
        if claim["claim_role"] == "substantive"
    ]
    counts = Counter(
        claim["claim_role"]
        for explanation in explanations
        for claim in explanation["claims"]
    )
    status_names = ("supported", "unsupported", "contradicted", "not_verifiable")
    source_names = ("a_only", "b_only", "both_a_b", "neither")
    eligible = [
        explanation
        for explanation in explanations
        if any(claim["claim_role"] == "substantive" for claim in explanation["claims"])
    ]

    def micro_rate(key: str, value: str) -> float | None:
        if not substantive:
            return None
        return sum(claim[key] == value for claim in substantive) / len(substantive)

    def macro_rate(key: str, value: str) -> float | None:
        per_explanation = []
        for explanation in eligible:
            claims = [
                claim
                for claim in explanation["claims"]
                if claim["claim_role"] == "substantive"
            ]
            per_explanation.append(sum(claim[key] == value for claim in claims) / len(claims))
        return mean(per_explanation)

    rate = micro_rate if aggregation == "micro_claim" else macro_rate
    result: dict[str, Any] = {
        "explanations": len(explanations),
        "eligible_explanations": len(eligible),
        "total_claims": sum(len(explanation["claims"]) for explanation in explanations),
        "identity_context_claims": counts["identity_context"],
        "substantive_explanatory_claims": counts["substantive"],
    }
    for status in status_names:
        result[f"visible_{status}_rate"] = rate("visible_status", status)
        result[f"shared_ab_{status}_rate"] = rate("shared_ab_status", status)
    for source in source_names:
        result[f"source_{source}_claims"] = sum(
            claim["source_bucket"] == source for claim in substantive
        )
        result[f"source_{source}_rate"] = rate("source_bucket", source)
    result["posthoc_b_aligned_claims"] = sum(
        claim["source_bucket"] in {"b_only", "both_a_b"} for claim in substantive
    )
    result["posthoc_b_aligned_rate"] = rate_in_claims_or_explanations(
        eligible if aggregation == "macro_explanation" else substantive,
        aggregation,
        lambda claim: claim["source_bucket"] in {"b_only", "both_a_b"},
    )
    return result


def rate_in_claims_or_explanations(
    values: Sequence[Mapping[str, Any]], aggregation: str, predicate: Any
) -> float | None:
    if aggregation == "micro_claim":
        return sum(predicate(claim) for claim in values) / len(values) if values else None
    rates = []
    for explanation in values:
        claims = [
            claim
            for claim in explanation["claims"]
            if claim["claim_role"] == "substantive"
        ]
        rates.append(sum(predicate(claim) for claim in claims) / len(claims))
    return mean(rates)
