"""Construct-valid v2 result-table transformations."""

from __future__ import annotations

import pandas as pd


def external_grounding_table(claims: pd.DataFrame) -> pd.DataFrame:
    """Aggregate support sources without treating unavailable rule grounding as zero."""

    valid = claims[~claims["claim_extraction_failed"].astype(bool)].copy()
    valid["rule_supported"] = valid["support_label"] == "supported_by_rule_evidence"
    valid["item_supported"] = valid["support_label"] == "supported_by_item_evidence"
    valid["unsupported_fashion_claim"] = valid["support_label"].isin(
        {"unsupported", "contradicted"}
    )
    rows = []
    for variant, group in valid.groupby("grounding_variant"):
        has_rules = variant in {"rule_rag", "hybrid_rag"}
        rows.append(
            {
                "grounding_variant": variant,
                "claim_count": len(group),
                "rule_supported_claim_rate": (
                    float(group["rule_supported"].mean()) if has_rules else pd.NA
                ),
                "item_supported_claim_rate": (
                    float(group["item_supported"].mean())
                    if variant in {"item_rag", "hybrid_rag"}
                    else pd.NA
                ),
                "unsupported_fashion_claim_rate": float(
                    group["unsupported_fashion_claim"].mean()
                ),
                "external_rule_grounding_status": "measured" if has_rules else "N/A",
            }
        )
    return pd.DataFrame(rows)
