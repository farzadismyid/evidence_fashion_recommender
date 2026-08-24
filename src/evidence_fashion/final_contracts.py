"""Final-run trace and explanation-input invariants.

These narrow functions are deliberately the only bridge between reranking and Rule-RAG input
construction.  They accept a stored trace; they neither receive a retriever nor reconstruct one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash a packet with a stable, byte-identical JSON representation."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def reproduce_evidence_score(trace: Mapping[str, Any], settings: Mapping[str, Any]) -> float:
    """Reproduce a stored candidate evidence score from precisely its stored rules."""
    rules = trace.get("rules", [])
    if not rules:
        if float(trace["evidence_score"]) != 0.0:
            raise ValueError("An empty trace must have an evidence score of zero.")
        return 0.0
    values = [float(rule["weighted_contribution"]) for rule in rules]
    score = float(settings["score_max_weight"]) * max(values) + float(
        settings["score_mean_weight"]
    ) * sum(values) / len(values)
    if abs(score - float(trace["evidence_score"])) > 1e-12:
        raise ValueError("Stored trace does not reproduce its candidate evidence score.")
    return score


def build_rule_rag_evidence_packet(locked_trace: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Pass the locked stored trace through unchanged and return its immutable hash link."""
    packet = dict(locked_trace)
    packet["rules"] = [dict(rule) for rule in locked_trace.get("rules", [])]
    return packet, canonical_json_sha256(packet)
