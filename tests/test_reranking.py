import pandas as pd

from evidence_fashion_recommender.reranking import weighted_rerank


def test_weighted_rerank_preserves_score_components() -> None:
    candidates = pd.DataFrame(
        {"item_ID": ["a", "b"], "clip_score": [0.9, 0.5], "evidence_score": [0.1, 1.0]}
    )
    result = weighted_rerank(candidates, clip_weight=0.9, evidence_weight=0.1)
    assert list(result["item_ID"]) == ["a", "b"]
    assert "final_score" in result
