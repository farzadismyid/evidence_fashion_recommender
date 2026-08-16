import numpy as np
import pandas as pd

from evidence_fashion.reranking import (
    minmax_normalize,
    pareto_frontier,
    rerank_candidates,
    select_pareto_knee,
)


def test_reranking_stores_all_scores_ranks_and_deterministic_ties() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["b", "a", "c"],
            "clip_score": [0.9, 0.9, 0.1],
            "evidence_score": [0.1, 0.8, 0.9],
        }
    )
    ranked = rerank_candidates(frame, clip_weight=0.75, evidence_weight=0.25)
    assert set(
        [
            "normalized_clip_score",
            "normalized_evidence_score",
            "final_score",
            "pre_rerank_rank",
            "post_rerank_rank",
        ]
    ).issubset(ranked.columns)
    assert ranked.iloc[0]["item_id"] == "a"


def test_constant_minmax_scores_become_zero() -> None:
    np.testing.assert_array_equal(minmax_normalize([2.0, 2.0]), [0.0, 0.0])


def test_pareto_frontier_and_knee_are_deterministic() -> None:
    points = pd.DataFrame(
        {
            "evidence_weight": [0.0, 0.25, 0.5],
            "rule_top_k": [1, 3, 5],
            "quality": [1.0, 0.9, 0.7],
            "evidence": [0.1, 0.8, 0.7],
        }
    )
    marked = pareto_frontier(points, ["quality", "evidence"])
    assert marked["pareto_status"].tolist() == ["frontier", "frontier", "dominated"]
    knee = select_pareto_knee(
        marked,
        ["quality", "evidence"],
        tie_columns=["evidence_weight", "rule_top_k"],
    )
    assert knee["evidence_weight"] in {0.0, 0.25}
