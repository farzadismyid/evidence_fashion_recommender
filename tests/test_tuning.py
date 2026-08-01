import pandas as pd

from evidence_fashion_recommender.evaluation.tuning import select_reranking_weight


def test_reranking_selection_prioritizes_ndcg() -> None:
    frame = pd.DataFrame(
        [
            {
                "clip_weight": 0.8,
                "evidence_weight": 0.2,
                "ndcg_at_10": 0.2,
                "hit_rate_at_10": 0.3,
                "reciprocal_rank": 0.1,
            },
            {
                "clip_weight": 0.9,
                "evidence_weight": 0.1,
                "ndcg_at_10": 0.3,
                "hit_rate_at_10": 0.2,
                "reciprocal_rank": 0.1,
            },
        ]
    )
    assert select_reranking_weight(frame).iloc[0]["clip_weight"] == 0.9
