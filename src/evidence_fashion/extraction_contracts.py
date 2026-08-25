"""Strict final-run contract for evidence-independent atomic-claim extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def validate_atomic_claims(
    payload: Mapping[str, Any], *, claim_types: Sequence[str]
) -> list[dict[str, str]]:
    """Accept only ordered, unique, evidence-independent atomic claims."""
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("Extraction must contain a non-empty claims array.")
    allowed = set(claim_types)
    expected_ids = [f"C{index}" for index in range(1, len(claims) + 1)]
    validated: list[dict[str, str]] = []
    seen_text: set[str] = set()
    for expected_id, row in zip(expected_ids, claims, strict=True):
        if not isinstance(row, Mapping) or set(row) != {"claim_id", "claim_text", "claim_type"}:
            raise ValueError("Each extracted claim must contain only ID, text, and type.")
        claim_id = str(row["claim_id"])
        text = str(row["claim_text"]).strip()
        claim_type = str(row["claim_type"])
        if claim_id != expected_id:
            raise ValueError("Claim IDs must be consecutive C identifiers in textual order.")
        if not text or claim_type not in allowed:
            raise ValueError("Every claim needs non-empty text and an allowed type.")
        normalized = " ".join(text.casefold().split())
        if normalized in seen_text:
            raise ValueError("Duplicate atomic claims are not allowed.")
        seen_text.add(normalized)
        validated.append(
            {"claim_id": claim_id, "claim_text": text, "claim_type": claim_type}
        )
    return validated
