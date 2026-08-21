"""Stage 8 atomic extraction, multi-source verification, and general judging contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

from .grounding_contracts import (
    canonical_citation_ids,
)
from .grounding_contracts import (
    citation_occurrences as parse_citation_occurrences,
)

CLAIM_STATUSES = {"supported", "unsupported", "contradicted", "not_verifiable"}
SUPPORT_SOURCES = {"query_or_locked_item", "rule_evidence"}
ENTAILMENT_FIELDS = (
    "full_kb_entailment",
    "exact_trace_entailment",
    "common_reference_item_fact_support",
)
ENTAILMENT_REQUIRED_FIELDS = (
    "claim_id",
    "full_kb_candidate_applicable_rule_ids",
    "full_kb_entailment",
    "full_kb_rule_ids",
    "full_kb_reason",
    "exact_trace_entailment",
    "exact_trace_rule_ids",
    "exact_trace_reason",
    "common_reference_item_fact_support",
    "common_reference_fields",
    "common_reference_reason",
)
CITATION_REQUIRED_FIELDS = (
    "claim_id",
    "citation_present",
    "canonical_citation_format",
    "cited_rule_ids",
    "invalid_rule_ids",
    "citation_entails_claim",
    "brief_reason",
)

VERDICT_DEFINITIONS = """SUPPORTED: the supplied source directly semantically entails the
complete claim.
CONTRADICTED: the supplied source directly entails an incompatible or negated proposition.
UNSUPPORTED: the claim is within the source's evaluable scope and relevant source material exists,
but that material does not entail the claim.
NOT_VERIFIABLE: the claim falls outside what that source can establish from the supplied packet.
Absence of evidence alone never implies contradiction."""

_SUBJECTIVE_OR_RULE_TERMS = frozenset(
    {
        "appropriate",
        "balance",
        "balanced",
        "contrast",
        "comfort",
        "comfortable",
        "complement",
        "complements",
        "coordinate",
        "coordinates",
        "elegant",
        "formal",
        "formality",
        "good",
        "harmony",
        "harmonious",
        "pairing",
        "proportion",
        "sophisticated",
        "style",
        "styling",
        "suitable",
        "suitability",
        "works",
    }
)

_FACT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "with",
    }
)


def _fact_texts(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [text for nested in value.values() for text in _fact_texts(nested)]
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [text for nested in value for text in _fact_texts(nested)]
    return [str(value).lower()]


def common_reference_eligibility(
    claim: Mapping[str, Any], common_reference_item_facts: Mapping[str, Any]
) -> dict[str, Any]:
    """Decide whether common item/context facts can assess this factual proposition.

    This is deliberately an eligibility decision, not a support verdict.  Styling
    rationales and subjective suitability statements remain outside the factual metric.
    """
    text = str(claim.get("claim_text", "")).lower()
    tokens = {token for token in re.findall(r"[a-z0-9]+", text) if token not in _FACT_STOPWORDS}
    if tokens & _SUBJECTIVE_OR_RULE_TERMS:
        return {"eligible": False, "reason": "subjective_or_general_styling_claim"}
    fact_text = " ".join(_fact_texts(common_reference_item_facts))
    fact_tokens = set(re.findall(r"[a-z0-9]+", fact_text))
    item_aliases = [
        str(common_reference_item_facts.get(field, "")).lower()
        for field in (
            "locked_item_minimal_name",
            "query_item_minimal_name",
            "locked_item_category",
            "query_item_category",
            "outfit_context_text",
        )
    ]
    subject_tokens = set()
    for alias in item_aliases:
        alias_tokens = set(re.findall(r"[a-z0-9]+", alias)) - _FACT_STOPWORDS
        if alias and (alias in text or alias_tokens & tokens):
            subject_tokens.update(alias_tokens)
    concrete_subject = bool(subject_tokens) or bool(
        re.search(r"\b(exact|locked|query) item\b", text)
    )
    if not concrete_subject:
        return {"eligible": False, "reason": "no_concrete_case_entity"}
    predicate_match = re.search(r"\b(?:is|are|has|have|contains)\b\s+(.+)", text)
    predicate_tokens = (
        {
            token
            for token in re.findall(r"[a-z0-9]+", predicate_match.group(1))
            if token not in _FACT_STOPWORDS
        }
        if predicate_match
        else tokens - subject_tokens - {"exact", "locked", "query", "item"}
    )
    if not predicate_tokens or not predicate_tokens.issubset(fact_tokens):
        return {"eligible": False, "reason": "not_a_literal_supplied_case_fact"}
    return {
        "eligible": True,
        "reason": "literal_supplied_case_fact",
    }


CONDITION_REVEALING_PHRASES = (
    r"\bno[- ]?rag\b",
    r"\brule[- ]?rag\b",
    r"\bretrieval[- ]augmented\b",
    r"\bprovided rules?\b",
    r"\brule evidence\b",
    r"\bwithout rules?\b",
)


def separated_entailment_schema() -> dict[str, Any]:
    """Schema for independent KB, trace, and common-reference assessments."""
    status = {"type": "string", "enum": sorted(CLAIM_STATUSES)}
    item = {
        "type": "object",
        "properties": {
            "claim_id": {"type": "string"},
            "full_kb_candidate_applicable_rule_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "full_kb_entailment": status,
            "full_kb_rule_ids": {"type": "array", "items": {"type": "string"}},
            "full_kb_reason": {"type": "string"},
            "exact_trace_entailment": status,
            "exact_trace_rule_ids": {"type": "array", "items": {"type": "string"}},
            "exact_trace_reason": {"type": "string"},
            "common_reference_item_fact_support": status,
            "common_reference_eligible": {"type": "boolean"},
            "common_reference_fields": {"type": "array", "items": {"type": "string"}},
            "common_reference_reason": {"type": "string"},
        },
        "required": [
            "claim_id",
            "full_kb_candidate_applicable_rule_ids",
            "full_kb_entailment",
            "full_kb_rule_ids",
            "full_kb_reason",
            "exact_trace_entailment",
            "exact_trace_rule_ids",
            "exact_trace_reason",
            "common_reference_item_fact_support",
            "common_reference_eligible",
            "common_reference_fields",
            "common_reference_reason",
        ],
    }
    return {
        "type": "object",
        "properties": {"claims": {"type": "array", "minItems": 1, "items": item}},
        "required": ["claims"],
    }


def citation_validation_schema() -> dict[str, Any]:
    """Schema for Phi's semantic pass; syntax and ID validity are deterministic."""
    item = {
        "type": "object",
        "properties": {
            "claim_id": {"type": "string"},
            "citation_entails_claim": {"type": ["boolean", "null"]},
            "brief_reason": {"type": "string"},
        },
        "required": [
            "claim_id",
            "citation_entails_claim",
            "brief_reason",
        ],
    }
    return {
        "type": "object",
        "properties": {"claims": {"type": "array", "minItems": 1, "items": item}},
        "required": ["claims"],
    }


def validate_separated_entailment(
    payload: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    *,
    full_kb_rule_ids: set[str],
    exact_trace_rule_ids: set[str],
    common_reference_item_facts: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless Phi returns one complete judgment for every supplied claim."""
    rows = payload.get("claims")
    if not isinstance(rows, list) or len(rows) != len(claims) or not rows:
        raise ValueError(
            "Entailment output must contain exactly one non-empty row per input claim."
        )
    expected_ids = [str(claim["claim_id"]) for claim in claims]
    observed_ids = [row.get("claim_id") for row in rows if isinstance(row, Mapping)]
    if observed_ids != expected_ids or len(observed_ids) != len(rows):
        raise ValueError("Entailment claim IDs must exactly preserve the supplied order and IDs.")
    validated = []
    for row in rows:
        if any(field not in row for field in ENTAILMENT_REQUIRED_FIELDS):
            raise ValueError(
                "Entailment output is missing a required verdict, evidence, or reason field."
            )
        candidate_ids = row["full_kb_candidate_applicable_rule_ids"]
        kb_ids = row["full_kb_rule_ids"]
        trace_ids = row["exact_trace_rule_ids"]
        if not all(isinstance(value, list) for value in (candidate_ids, kb_ids, trace_ids)):
            raise ValueError("Entailment rule evidence fields must be arrays.")
        if not set(candidate_ids).issubset(full_kb_rule_ids) or not set(kb_ids).issubset(
            full_kb_rule_ids
        ):
            raise ValueError("Entailment cited a rule outside the supplied full-KB candidates.")
        if not set(trace_ids).issubset(exact_trace_rule_ids):
            raise ValueError("Entailment cited a rule outside the supplied exact trace.")
        if any(str(row[field]) not in CLAIM_STATUSES for field in ENTAILMENT_FIELDS):
            raise ValueError("Entailment contains an invalid verdict.")
        if not all(
            str(row[field]).strip()
            for field in ("full_kb_reason", "exact_trace_reason", "common_reference_reason")
        ):
            raise ValueError("Entailment requires a non-empty reason for every evidence dimension.")
        if not isinstance(row["common_reference_fields"], list):
            raise ValueError("Entailment common-reference fields must be an array.")
        eligibility = common_reference_eligibility(
            claims[len(validated)], common_reference_item_facts
        )
        normalized = dict(row)
        normalized["common_reference_eligible"] = eligibility["eligible"]
        if not eligibility["eligible"]:
            normalized["common_reference_item_fact_support"] = "not_verifiable"
            normalized["common_reference_fields"] = []
            normalized["common_reference_reason"] = (
                "N/A: common-reference facts cannot establish styling or subjective rationale."
            )
        validated.append(normalized)
    return {"claims": validated}


def validate_citation_validation(
    payload: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    *,
    occurrence_diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine Phi semantic judgments with deterministic citation diagnostics."""
    rows = payload.get("claims")
    if not isinstance(rows, list) or len(rows) != len(claims) or not rows:
        raise ValueError("Citation output must contain exactly one non-empty row per input claim.")
    expected_ids = [str(claim["claim_id"]) for claim in claims]
    observed_ids = [row.get("claim_id") for row in rows if isinstance(row, Mapping)]
    if observed_ids != expected_ids or len(observed_ids) != len(rows):
        raise ValueError("Citation claim IDs must exactly preserve the supplied order and IDs.")
    citation_present = bool(occurrence_diagnostics)
    valid_occurrences = [
        occurrence
        for occurrence in occurrence_diagnostics
        if occurrence.get("valid_canonical_occurrence") is True
    ]
    canonical = (
        None
        if not citation_present
        else bool(valid_occurrences) and len(valid_occurrences) == len(occurrence_diagnostics)
    )
    cited_rule_ids = list(
        dict.fromkeys(
            rule_id
            for occurrence in occurrence_diagnostics
            for rule_id in occurrence.get("rule_ids", [])
        )
    )
    invalid_rule_ids = sorted(
        {
            rule_id
            for occurrence in occurrence_diagnostics
            for field in ("unknown_rule_ids", "out_of_trace_rule_ids")
            for rule_id in occurrence.get(field, [])
        }
    )
    validated = []
    for row in rows:
        if any(
            field not in row for field in ("claim_id", "citation_entails_claim", "brief_reason")
        ):
            raise ValueError("Citation output is missing a required field.")
        entails = row["citation_entails_claim"]
        if not str(row["brief_reason"]).strip():
            raise ValueError("Citation output requires a brief reason.")
        if not citation_present or canonical is False:
            entails = None
        elif not isinstance(entails, bool):
            raise ValueError("Valid citations require a boolean citation-entailment judgment.")
        validated.append(
            {
                "claim_id": str(row["claim_id"]),
                "citation_present": citation_present,
                "canonical_citation_format": canonical,
                "cited_rule_ids": cited_rule_ids,
                "invalid_rule_ids": invalid_rule_ids,
                "citation_entails_claim": entails,
                "brief_reason": str(row["brief_reason"]),
            }
        )
    return {"claims": validated}


def citation_occurrences(
    explanation: str,
    *,
    known_rule_ids: Sequence[str] = (),
    trace_rule_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Preserve citations for the dedicated post-entailment citation assessment only."""
    return parse_citation_occurrences(
        explanation, known_rule_ids=known_rule_ids, trace_rule_ids=trace_rule_ids
    )


def strip_rule_citations(text: str) -> str:
    """Remove all bracketed rule citations before the citation-blind entailment pass."""
    return re.sub(r"\[[^\]]*K\d{3}[^\]]*\]", "", text).replace("  ", " ").strip()


def validate_canonical_citation_format(explanation: str) -> list[str]:
    """Accept only individually bracketed IDs, e.g. ``[K025] [K099]``."""
    return [f"[{rule_id}]" for rule_id in canonical_citation_ids(explanation)]


def build_separated_entailment_prompt(
    *,
    explanation: str,
    claims: Sequence[Mapping[str, Any]],
    full_kb_rules: Sequence[Mapping[str, Any]],
    exact_trace_rules: Sequence[Mapping[str, Any]],
    common_reference_item_facts: Mapping[str, Any],
) -> str:
    """Build a citation-blind prompt with three non-derived verification dimensions."""
    evidence = {
        "frozen_full_knowledge_base": list(full_kb_rules),
        "exact_stored_rule_trace": list(exact_trace_rules),
        "common_reference_item_facts": dict(common_reference_item_facts),
    }
    return (
        "Independently assess full-KB entailment, exact-trace entailment, and common-reference "
        "item-fact support for every claim. Do not derive any field from another.\n"
        f"{VERDICT_DEFINITIONS}\n"
        "For KB and trace support, a rule counts only if its antecedent is established and its "
        "stated consequent directly entails the complete claim. Do not use category similarity, "
        "examples as exhaustive lists, or absence from a list as negative evidence. Common "
        "reference support is eligible only for concrete supplied item/query/context facts; "
        "styling principles, suitability, and subjective rationale are not verifiable there. "
        "No citation information is available in this pass.\n\n"
        f"Evidence: {json.dumps(evidence, sort_keys=True, ensure_ascii=False)}\n\n"
        f"Explanation with citations removed: {strip_rule_citations(explanation)}\n\n"
        f"Atomic claims: {json.dumps(list(claims), sort_keys=True, ensure_ascii=False)}"
    )


def build_citation_validation_prompt(
    *,
    claims: Sequence[Mapping[str, Any]],
    explanation: str,
    exact_trace_rules: Sequence[Mapping[str, Any]],
) -> str:
    """Build the separate citation-validity pass after blind entailment is complete."""
    serialized_trace = json.dumps(list(exact_trace_rules), sort_keys=True, ensure_ascii=False)
    serialized_occurrences = json.dumps(citation_occurrences(explanation), sort_keys=True)
    serialized_claims = json.dumps(list(claims), sort_keys=True, ensure_ascii=False)
    return (
        "The citation-occurrence diagnostics are deterministic and authoritative. Assess only "
        "whether valid canonical cited rules semantically entail each claim. Do not produce or "
        "modify general support verdicts. For absent or malformed citations, return null "
        "citation entailment.\n\n"
        f"Exact stored rule trace: {serialized_trace}\n\n"
        f"Citation occurrences: {serialized_occurrences}\n\n"
        f"Atomic claims: {serialized_claims}"
    )


def strip_condition_revealing_phrases(text: str) -> str:
    sanitized = strip_rule_citations(text)
    for phrase in CONDITION_REVEALING_PHRASES:
        sanitized = re.sub(phrase, "", sanitized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", sanitized).strip()


def prepare_true_blind_pair(
    explanations: Mapping[str, str], *, case_id: str, generator: str, seed: int
) -> dict[str, Any]:
    """Sanitize and deterministically randomize positions without leaking the assignment."""
    if set(explanations) != {"no_rag", "rule_rag"}:
        raise ValueError("A blind pair requires exactly no_rag and rule_rag explanations.")
    ordering = ["no_rag", "rule_rag"]
    digest = sha256(f"{seed}:{case_id}:{generator}".encode()).hexdigest()
    if int(digest, 16) % 2:
        ordering.reverse()
    return {
        "first_explanation": strip_condition_revealing_phrases(explanations[ordering[0]]),
        "second_explanation": strip_condition_revealing_phrases(explanations[ordering[1]]),
        "position_assignment": {"first": ordering[0], "second": ordering[1]},
    }


def build_true_blind_judge_prompt(
    *,
    common_reference_context: Mapping[str, Any],
    first_explanation: str,
    second_explanation: str,
) -> str:
    dimensions = (
        "relevance, clarity, usefulness, coherence, appropriate specificity, and non-redundancy"
    )
    serialized_context = json.dumps(dict(common_reference_context), sort_keys=True)
    return (
        "Score each anonymized explanation independently from 1 to 5 only for "
        f"{dimensions}. Do not infer evidence access or experimental conditions.\n\n"
        f"Common reference context: {serialized_context}\n\n"
        f"First explanation:\n{first_explanation}\n\nSecond explanation:\n{second_explanation}"
    )


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


def cited_rule_ids(explanation: str, pattern: str | None = None) -> list[str]:
    """Return only syntactically canonical IDs; retain malformed occurrences separately."""
    del pattern
    return list(
        dict.fromkeys(
            rule_id
            for occurrence in citation_occurrences(explanation)
            if occurrence["canonical_syntax"]
            for rule_id in occurrence["rule_ids"]
        )
    )


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
        f"entailment. {VERDICT_DEFINITIONS} Rule support requires an exact supplied rule ID, an "
        "established antecedent, and a consequent that directly entails the complete claim. Do not "
        "broaden item categories by analogy and do not infer unsuitability from omission in a "
        "non-exhaustive example list. Common packet facts support only concrete supplied case "
        "facts, "
        "not styling principles, suitability, or subjective rationale. For an uncited claim, "
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
    common_reference_eligibility_by_claim: Mapping[str, Mapping[str, Any]] | None = None,
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
        if (
            has_citations
            and citation_entails is not None
            and not isinstance(citation_entails, bool)
        ):
            raise ValueError("Citation entailment must be boolean or null for a cited explanation.")
        if not reason:
            raise ValueError("Every verifier decision requires a brief reason.")
        eligibility = (common_reference_eligibility_by_claim or {}).get(
            str(row["claim_id"]), {"eligible": False, "reason": "not_assessed"}
        )
        if (
            common_reference_eligibility_by_claim is not None
            and "query_or_locked_item" in sources
            and not eligibility["eligible"]
        ):
            raise ValueError(
                "Common-reference source cannot support a styling or subjective claim."
            )
        validated.append(
            {
                "claim_id": str(row["claim_id"]),
                "support_status": status,
                "support_sources": list(sources),
                "supporting_rule_ids": list(rule_ids),
                "citation_entails_claim": citation_entails,
                "brief_reason": reason,
                "common_reference_eligible": bool(eligibility["eligible"]),
                "common_reference_eligibility_reason": str(eligibility["reason"]),
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
            rule_ids = list(dict.fromkeys(rule for rule in rules_value if rule in allowed_rule_ids))
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
