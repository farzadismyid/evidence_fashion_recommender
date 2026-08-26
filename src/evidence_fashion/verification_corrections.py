"""Deterministic logical-consistency corrections for frozen verification labels."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any


def enforce_trace_implies_full_kb(records: Sequence[MutableMapping[str, Any]]) -> int:
    """Apply the trace-subset invariant, failing closed on malformed evidence packets."""
    corrected = 0
    for record in records:
        trace_rule_ids = set(record["exact_trace_rule_ids"])
        full_kb_rule_ids = set(record["full_kb_candidate_rule_ids"])
        for claim in record["claims"]:
            if (
                claim["trace_support"] == "supported"
                and claim["full_kb_support"] == "not_supported"
            ):
                if not trace_rule_ids or not trace_rule_ids.issubset(full_kb_rule_ids):
                    raise ValueError(
                        "Cannot correct a claim unless every exact-trace rule is present in the "
                        "record's full-KB candidate packet."
                    )
                claim["full_kb_support"] = "supported"
                corrected += 1
    return corrected


def trace_support_implies_full_kb(record: Mapping[str, Any]) -> bool:
    """Return whether every stored claim obeys the logical trace-subset invariant."""
    return all(
        claim["trace_support"] != "supported" or claim["full_kb_support"] == "supported"
        for claim in record["claims"]
    )
