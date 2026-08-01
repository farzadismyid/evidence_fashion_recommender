"""Blinded human-review pack creation and agreement metrics."""

from __future__ import annotations

import pandas as pd

REVIEW_COLUMNS = [
    "rater_id",
    "issue_label",
    "issue_notes",
    "acceptability",
    "faithfulness_score",
    "usefulness_score",
]


def create_blinded_review_pack(
    explanations: pd.DataFrame,
    id_column: str = "case_id",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        id_column,
        "grounding_variant",
        "query_text",
        "user_request",
        "recommended_text",
        "generated_explanation",
    }
    missing = required - set(explanations.columns)
    if missing:
        raise ValueError(f"Explanation table is missing columns: {sorted(missing)}")
    shuffled = explanations.sample(frac=1, random_state=seed).reset_index(drop=True).copy()
    shuffled["review_id"] = [f"H{index:05d}" for index in range(1, len(shuffled) + 1)]
    key = shuffled[["review_id", id_column, "grounding_variant"]].copy()
    visible = shuffled[
        [
            "review_id",
            "query_text",
            "user_request",
            "recommended_text",
            "generated_explanation",
        ]
    ].copy()
    for column in REVIEW_COLUMNS:
        visible[column] = ""
    return visible, key


def cohen_kappa(labels_a: pd.Series, labels_b: pd.Series) -> float:
    paired = pd.DataFrame({"a": labels_a, "b": labels_b}).dropna()
    paired = paired[(paired["a"] != "") & (paired["b"] != "")]
    if paired.empty:
        raise ValueError("No paired ratings are available.")
    observed = float((paired["a"] == paired["b"]).mean())
    categories = sorted(set(paired["a"]) | set(paired["b"]))
    expected = sum(
        float((paired["a"] == category).mean()) * float((paired["b"] == category).mean())
        for category in categories
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0
