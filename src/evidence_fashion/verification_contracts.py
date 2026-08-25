"""Strict final-run contract for the four independent Stage-4 verdicts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

TRACE_AND_KB = frozenset({"supported", "not_supported"})
COMMON = frozenset({"supported", "not_supported", "N/A"})
CITATION = frozenset({"entails", "does_not_entail", "N/A"})


def verification_schema() -> dict[str, Any]:
    """Return the frozen structured-output schema for a claim record."""
    item = {
        "type": "object",
        "properties": {
            "claim_id": {"type": "string"},
            "trace_support": {"type": "string", "enum": sorted(TRACE_AND_KB)},
            "full_kb_support": {"type": "string", "enum": sorted(TRACE_AND_KB)},
            "common_reference_support": {"type": "string", "enum": sorted(COMMON)},
            "citation_entailment": {"type": "string", "enum": sorted(CITATION)},
        },
        "required": [
            "claim_id",
            "trace_support",
            "full_kb_support",
            "common_reference_support",
            "citation_entailment",
        ],
    }
    return {
        "type": "object",
        "properties": {"claims": {"type": "array", "minItems": 1, "items": item}},
        "required": ["claims"],
    }


def validate_verdicts(
    payload: Mapping[str, Any],
    *,
    claims: Sequence[Mapping[str, Any]],
    common_reference_eligible: Mapping[str, bool],
    valid_citations_present: bool,
) -> list[dict[str, str]]:
    """Reject altered IDs, extra fields, and invalid N/A semantics."""
    rows = payload.get("claims")
    expected_ids = [str(claim["claim_id"]) for claim in claims]
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError("Verifier output must contain exactly one row for each input claim.")
    validated: list[dict[str, str]] = []
    for expected_id, row in zip(expected_ids, rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != {
            "claim_id",
            "trace_support",
            "full_kb_support",
            "common_reference_support",
            "citation_entailment",
        }:
            raise ValueError("Each verdict must contain exactly the five frozen fields.")
        if str(row["claim_id"]) != expected_id:
            raise ValueError("Verifier must preserve claim IDs exactly and in order.")
        trace = str(row["trace_support"])
        full_kb = str(row["full_kb_support"])
        common = str(row["common_reference_support"])
        citation = str(row["citation_entailment"])
        if trace not in TRACE_AND_KB or full_kb not in TRACE_AND_KB:
            raise ValueError("Trace and full-KB verdicts must be binary.")
        if common not in COMMON or citation not in CITATION:
            raise ValueError("Verifier returned an invalid common-reference or citation verdict.")
        eligible = bool(common_reference_eligible.get(expected_id, False))
        if not eligible and common != "N/A":
            raise ValueError("Ineligible common-reference claims must be N/A.")
        if eligible and common == "N/A":
            raise ValueError("Eligible common-reference claims require a binary verdict.")
        if not valid_citations_present and citation != "N/A":
            raise ValueError("Citation entailment must be N/A without valid citations.")
        validated.append(
            {
                "claim_id": expected_id,
                "trace_support": trace,
                "full_kb_support": full_kb,
                "common_reference_support": common,
                "citation_entailment": citation,
            }
        )
    return validated
