"""Evidence-aware candidate reranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


def minmax(values: pd.Series) -> pd.Series:
    low, high = float(values.min()), float(values.max())
    if np.isclose(low, high):
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - low) / (high - low)


def weighted_rerank(
    candidates: pd.DataFrame,
    clip_weight: float,
    evidence_weight: float,
    normalize_scores: bool = True,
) -> pd.DataFrame:
    required = {"clip_score", "evidence_score"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidates are missing scores: {sorted(missing)}")
    result = candidates.copy()
    clip = minmax(result["clip_score"]) if normalize_scores else result["clip_score"]
    evidence = minmax(result["evidence_score"]) if normalize_scores else result["evidence_score"]
    result["final_score"] = clip_weight * clip + evidence_weight * evidence
    return result.sort_values("final_score", ascending=False).reset_index(drop=True)
