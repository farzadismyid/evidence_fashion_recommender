import pytest

from evidence_fashion_recommender.evaluation.ranking import ranking_metrics


def test_ranking_metrics() -> None:
    result = ranking_metrics([0, 1, 0, 1], [1, 3])
    assert result["precision_at_1"] == 0
    assert result["hit_rate_at_3"] == 1
    assert result["recall_at_3"] == pytest.approx(0.5)
    assert result["reciprocal_rank"] == pytest.approx(0.5)
