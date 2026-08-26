"""Deterministic Stage-5 metrics for the frozen final verification schema."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

METRICS = (
    "trace_support_rate",
    "full_kb_support_rate",
    "unsupported_item_fact_rate",
    "trace_supported_claims_per_100_words",
)


def word_count(text: str) -> int:
    return len(text.split())


def record_metrics(
    record: Mapping[str, Any], explanation: Mapping[str, Any]
) -> dict[str, float | None]:
    """Calculate frozen claim metrics without obsolete support-source fields."""
    claims = list(record["claims"])
    if not claims:
        return {metric: None for metric in METRICS}
    eligible = [claim for claim in claims if claim["common_reference_support"] != "N/A"]
    words = word_count(str(explanation["explanation"]))
    trace_supported = sum(claim["trace_support"] == "supported" for claim in claims)
    return {
        "trace_support_rate": trace_supported / len(claims),
        "full_kb_support_rate": sum(claim["full_kb_support"] == "supported" for claim in claims)
        / len(claims),
        "unsupported_item_fact_rate": (
            sum(claim["common_reference_support"] == "not_supported" for claim in eligible)
            / len(eligible)
            if eligible
            else None
        ),
        "trace_supported_claims_per_100_words": trace_supported * 100 / words if words else None,
    }


def paired_complete_rows(
    rows: Sequence[Mapping[str, Any]], generator: str | None = None
) -> list[dict[str, Any]]:
    """Return only case-level No-RAG/Rule-RAG pairs, optionally for one generator."""
    selected = [row for row in rows if generator is None or row["generator_model_id"] == generator]
    by_key = {
        (row["case_id"], row["generator_model_id"], row["condition"]): row for row in selected
    }
    pairs = []
    for case_id, model, condition in by_key:
        if (
            condition == "no_rag"
            and (rule_rag := by_key.get((case_id, model, "rule_rag"))) is not None
        ):
            pairs.append(
                {
                    "case_id": case_id,
                    "generator_model_id": model,
                    "no_rag": by_key[(case_id, model, condition)],
                    "rule_rag": rule_rag,
                }
            )
    return sorted(pairs, key=lambda row: (row["case_id"], row["generator_model_id"]))


def bootstrap_paired_difference(
    pairs: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
    aggregate_generators_by_case: bool,
) -> dict[str, float | int | None]:
    """Case-clustered paired bootstrap; overall analysis averages generators within a case."""
    values: dict[str, list[float]] = {}
    for pair in pairs:
        before, after = pair["no_rag"][metric], pair["rule_rag"][metric]
        if before is not None and after is not None:
            cluster = str(pair["case_id"] if aggregate_generators_by_case else pair["case_id"])
            values.setdefault(cluster, []).append(float(after) - float(before))
    differences = np.asarray([np.mean(value) for value in values.values()], dtype=float)
    if not len(differences):
        return {"estimate": None, "ci_lower": None, "ci_upper": None, "p_value": None, "n": 0}
    rng = np.random.default_rng(seed)
    estimates = differences[
        rng.integers(0, len(differences), size=(replicates, len(differences)))
    ].mean(axis=1)
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(estimates, [alpha, 1 - alpha])
    p_value = min(
        1.0,
        2
        * min(
            (np.count_nonzero(estimates <= 0) + 1) / (replicates + 1),
            (np.count_nonzero(estimates >= 0) + 1) / (replicates + 1),
        ),
    )
    return {
        "estimate": float(differences.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(p_value),
        "n": len(differences),
    }


def holm_adjust(p_values: Iterable[float | None]) -> list[float | None]:
    values = list(p_values)
    ordered = sorted(
        ((index, value) for index, value in enumerate(values) if value is not None),
        key=lambda item: float(item[1]),
    )
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * float(value)))
        adjusted[index] = running
    return [adjusted.get(index) for index in range(len(values))]
