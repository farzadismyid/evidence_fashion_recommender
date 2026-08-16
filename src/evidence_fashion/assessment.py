"""Stage 8 atomic extraction, multi-source verification, and general judging contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

CLAIM_STATUSES = {"supported", "unsupported", "contradicted", "not_verifiable"}
SUPPORT_SOURCES = {"query_or_locked_item", "rule_evidence"}


def extraction_schema(claim_types: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "claim_text": {"type": "string"},
                        "claim_type": {"type": "string", "enum": list(claim_types)},
                    },
                    "required": ["claim_id", "claim_text", "claim_type"],
                },
            }
        },
        "required": ["claims"],
    }


def verification_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "support_status": {
                            "type": "string",
                            "enum": sorted(CLAIM_STATUSES),
                        },
                        "support_sources": {
                            "type": "array",
                            "items": {"type": "string", "enum": sorted(SUPPORT_SOURCES)},
                            "uniqueItems": True,
                        },
                        "supporting_rule_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "uniqueItems": True,
                        },
                        "citation_entails_claim": {
                            "type": ["boolean", "null"],
                        },
                        "brief_reason": {"type": "string"},
                    },
                    "required": [
                        "claim_id",
                        "support_status",
                        "support_sources",
                        "supporting_rule_ids",
                        "citation_entails_claim",
                        "brief_reason",
                    ],
                },
            }
        },
        "required": ["claims"],
    }


def judge_schema(dimensions: Sequence[str], minimum: int, maximum: int) -> dict[str, Any]:
    score_properties = {
        dimension: {"type": "integer", "minimum": minimum, "maximum": maximum}
        for dimension in dimensions
    }
    item = {
        "type": "object",
        "properties": {**score_properties, "brief_reason": {"type": "string"}},
        "required": [*dimensions, "brief_reason"],
    }
    return {
        "type": "object",
        "properties": {"first": item, "second": item},
        "required": ["first", "second"],
    }


def build_extraction_prompt(explanation: str, claim_types: Sequence[str]) -> str:
    return (
        "Extract every independent atomic fashion or styling proposition from the complete "
        "explanation. Split conjunctions when they make independently checkable claims. Do not "
        "use named-entity recognition as a substitute for proposition extraction. Do not assess "
        "truth or support. Do not omit claims, repeat claims, or impose a claim cap. Assign IDs "
        "C1, C2, and so on in textual order. Use only these claim types: "
        f"{', '.join(claim_types)}.\n\nComplete explanation:\n{explanation}"
    )


def validate_extraction(
    payload: Mapping[str, Any], claim_types: Sequence[str], *, require_claims: bool = True
) -> list[dict[str, str]]:
    claims = payload.get("claims")
    if not isinstance(claims, list) or (require_claims and not claims):
        raise ValueError("Extraction must contain a non-empty claims array.")
    expected_ids = [f"C{index}" for index in range(1, len(claims) + 1)]
    observed_ids = [row.get("claim_id") for row in claims if isinstance(row, Mapping)]
    if observed_ids != expected_ids or len(observed_ids) != len(claims):
        raise ValueError("Claim IDs must be complete, unique, and sequential.")
    normalized_texts: set[str] = set()
    validated = []
    for row in claims:
        text = str(row.get("claim_text", "")).strip()
        claim_type = str(row.get("claim_type", ""))
        if not text or claim_type not in claim_types:
            raise ValueError("Every claim needs non-empty text and an allowed type.")
        normalized = " ".join(text.lower().split())
        if normalized in normalized_texts:
            continue
        normalized_texts.add(normalized)
        validated.append(
            {"claim_id": str(row["claim_id"]), "claim_text": text, "claim_type": claim_type}
        )
    for index, row in enumerate(validated, start=1):
        row["claim_id"] = f"C{index}"
    return validated


def cited_rule_ids(explanation: str, pattern: str) -> list[str]:
    return list(dict.fromkeys(re.findall(pattern, explanation)))


def build_verification_prompt(
    *,
    explanation: str,
    claims: Sequence[Mapping[str, Any]],
    packet_a: Mapping[str, Any],
    packet_b: Mapping[str, Any],
    evidence_shown: str,
    citation_ids: Sequence[str],
) -> str:
    evidence = {
        "common_context_A": packet_a,
        "exact_rule_trace_B": packet_b,
        "evidence_shown_during_generation": evidence_shown,
        "citations_observed_in_explanation": list(citation_ids),
    }
    return (
        "Act as a strict entailment classifier. Verify every supplied atomic claim against the "
        "same common union evidence packet A+B, regardless of what was visible during generation. "
        "Keep support status separate from support sources. A claim may have both allowed support "
        "sources, but each source label may appear at most once. Supported means semantic "
        "entailment; unsupported means relevant evidence exists "
        "but does not entail it; contradicted requires affirmative conflict; not_verifiable means "
        "the packet cannot settle it. Absence of support is not contradiction. Rule support "
        "requires an exact supplied rule ID whose text entails the claim. For an uncited claim, "
        "citation_entails_claim must be null; otherwise it is true only if an observed valid "
        "citation entails that claim. Return each input claim ID exactly once and no others.\n\n"
        f"Evidence packet: {json.dumps(evidence, sort_keys=True, ensure_ascii=False)}\n\n"
        f"Complete explanation: {explanation}\n\n"
        f"Atomic claims: {json.dumps(list(claims), sort_keys=True, ensure_ascii=False)}"
    )


def validate_verification(
    payload: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    allowed_rule_ids: set[str],
    citation_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows = payload.get("claims")
    if not isinstance(rows, list) or len(rows) != len(claims):
        raise ValueError("Verifier output must cover every claim exactly once.")
    expected_ids = [str(row["claim_id"]) for row in claims]
    if [row.get("claim_id") for row in rows if isinstance(row, Mapping)] != expected_ids:
        raise ValueError("Verifier claim IDs are missing, duplicated, unknown, or out of order.")
    has_citations = bool(citation_ids)
    validated = []
    for row in rows:
        status = str(row.get("support_status", ""))
        sources = row.get("support_sources")
        rule_ids = row.get("supporting_rule_ids")
        citation_entails = row.get("citation_entails_claim")
        reason = str(row.get("brief_reason", "")).strip()
        if status not in CLAIM_STATUSES or not isinstance(sources, list):
            raise ValueError("Invalid support status or source array.")
        sources = list(dict.fromkeys(sources))
        if not set(sources).issubset(SUPPORT_SOURCES):
            raise ValueError("Support sources must contain only allowed labels.")
        if not isinstance(rule_ids, list):
            raise ValueError("Supporting rule IDs must be an array.")
        rule_ids = list(dict.fromkeys(rule_ids))
        if not set(rule_ids).issubset(allowed_rule_ids):
            raise ValueError("Verifier cited a rule outside the supplied B packet.")
        if status == "supported" and not sources:
            raise ValueError("A supported claim must identify at least one support source.")
        if "rule_evidence" in sources and not rule_ids:
            raise ValueError("Rule support requires at least one exact supporting rule ID.")
        if rule_ids and "rule_evidence" not in sources:
            raise ValueError("Supporting rule IDs require the rule_evidence source label.")
        if not has_citations and citation_entails is not None:
            raise ValueError("Citation entailment must be null when the explanation is uncited.")
        if has_citations and citation_entails is not None and not isinstance(
            citation_entails, bool
        ):
            raise ValueError("Citation entailment must be boolean or null for a cited explanation.")
        if not reason:
            raise ValueError("Every verifier decision requires a brief reason.")
        validated.append(
            {
                "claim_id": str(row["claim_id"]),
                "support_status": status,
                "support_sources": list(sources),
                "supporting_rule_ids": list(rule_ids),
                "citation_entails_claim": citation_entails,
                "brief_reason": reason,
            }
        )
    return validated


def normalize_verification_payload(
    payload: Mapping[str, Any],
    allowed_rule_ids: set[str],
    citation_ids: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    """Mechanically reconcile verifier fields without inventing evidence.

    The raw response remains the authoritative model output.  This normalization only
    removes impossible source identifiers, reconciles the source/rule parallel fields,
    and applies the predeclared N/A citation convention.  If those repairs leave a
    nominally supported claim with no valid source, the conservative status is
    ``not_verifiable``.
    """
    rows = payload.get("claims")
    if not isinstance(rows, list):
        return dict(payload), []
    actions: list[str] = []
    normalized_rows: list[Any] = []
    for item in rows:
        if not isinstance(item, Mapping):
            normalized_rows.append(item)
            continue
        row = dict(item)
        sources_value = row.get("support_sources")
        rules_value = row.get("supporting_rule_ids")
        if isinstance(sources_value, list):
            sources = list(
                dict.fromkeys(source for source in sources_value if source in SUPPORT_SOURCES)
            )
        else:
            sources = sources_value
        if isinstance(rules_value, list):
            rule_ids = list(
                dict.fromkeys(rule for rule in rules_value if rule in allowed_rule_ids)
            )
        else:
            rule_ids = rules_value
        if isinstance(sources, list) and isinstance(rule_ids, list):
            if rule_ids and "rule_evidence" not in sources:
                sources.append("rule_evidence")
                actions.append("added_rule_source_for_valid_rule_ids")
            if not rule_ids and "rule_evidence" in sources:
                sources = [source for source in sources if source != "rule_evidence"]
                actions.append("removed_rule_source_without_valid_rule_ids")
            if row.get("support_status") == "supported" and not sources:
                row["support_status"] = "not_verifiable"
                actions.append("conservatively_relabelled_sourceless_support")
            row["support_sources"] = sources
            row["supporting_rule_ids"] = rule_ids
        if not citation_ids and row.get("citation_entails_claim") is not None:
            row["citation_entails_claim"] = None
            actions.append("set_uncited_entailment_to_null")
        normalized_rows.append(row)
    return {**dict(payload), "claims": normalized_rows}, list(dict.fromkeys(actions))


def build_judge_prompt(
    *,
    packet_a: Mapping[str, Any],
    packet_b: Mapping[str, Any],
    first_text: str,
    second_text: str,
    first_evidence_shown: str,
    second_evidence_shown: str,
    dimensions: Sequence[str],
) -> str:
    evidence = {
        "common_context_A": packet_a,
        "exact_rule_trace_B": packet_b,
        "first_displayed_sources": first_evidence_shown,
        "second_displayed_sources": second_evidence_shown,
    }
    return (
        "Judge each complete explanation separately from atomic-claim verification. Assign "
        "anchored integer scores from 1 (very poor) to 5 (excellent), with higher always better, "
        f"for: {', '.join(dimensions)}. Input consistency measures consistency with A; evidence "
        "use correctness measures whether claims use only the sources displayed for that text; "
        "hallucination control penalizes unsupported details. Do not score word-count compliance. "
        "Do not infer or name experimental conditions. Give a concise reason for each text.\n\n"
        f"Evaluation context: {json.dumps(evidence, sort_keys=True, ensure_ascii=False)}\n\n"
        f"First explanation:\n{first_text}\n\nSecond explanation:\n{second_text}"
    )


def validate_judgment(
    payload: Mapping[str, Any], dimensions: Sequence[str], minimum: int, maximum: int
) -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    for position in ("first", "second"):
        row = payload.get(position)
        if not isinstance(row, Mapping):
            raise ValueError("Paired judgment must contain first and second objects.")
        result: dict[str, Any] = {}
        for dimension in dimensions:
            value = row.get(dimension)
            if not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError("Judge scores must be anchored integers in range.")
            result[dimension] = value
        reason = str(row.get("brief_reason", "")).strip()
        if not reason:
            raise ValueError("Each judgment requires a brief reason.")
        result["brief_reason"] = reason
        validated[position] = result
    return validated
