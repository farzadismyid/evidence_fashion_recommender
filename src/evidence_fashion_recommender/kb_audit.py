"""Knowledge-base quality and coverage audit."""

from __future__ import annotations

import pandas as pd


def audit_knowledge_base(kb: pd.DataFrame, target_categories: list[str]) -> dict[str, object]:
    required = [
        "rule_id",
        "rule_text",
        "input_category",
        "recommended_category",
        "source_type",
        "source_title",
        "source_url_or_reference",
        "source_reliability",
    ]
    missing_columns = sorted(set(required) - set(kb.columns))
    duplicate_ids = int(kb["rule_id"].duplicated().sum()) if "rule_id" in kb else None
    blank_rule_count = (
        int(kb["rule_text"].fillna("").str.strip().eq("").sum()) if "rule_text" in kb else None
    )
    represented = (
        set(kb["recommended_category"].dropna().astype(str))
        if "recommended_category" in kb
        else set()
    )
    category_counts = (
        kb["recommended_category"].value_counts(dropna=False).to_dict()
        if "recommended_category" in kb
        else {}
    )
    return {
        "rows": len(kb),
        "missing_required_columns": missing_columns,
        "duplicate_rule_ids": duplicate_ids,
        "blank_rule_count": blank_rule_count,
        "missing_target_categories": sorted(set(target_categories) - represented),
        "recommended_category_counts": category_counts,
        "source_type_counts": (
            kb["source_type"].value_counts(dropna=False).to_dict() if "source_type" in kb else {}
        ),
        "reliability_counts": (
            kb["source_reliability"].value_counts(dropna=False).to_dict()
            if "source_reliability" in kb
            else {}
        ),
    }
