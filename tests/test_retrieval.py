import numpy as np
import pandas as pd
import pytest

from evidence_fashion.retrieval import (
    cosine_scores,
    fuse_clip_embeddings,
    l2_normalize,
    rank_candidates,
)


def test_l2_normalization() -> None:
    vectors = l2_normalize(np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32))
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_zero_embedding_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        l2_normalize(np.zeros((1, 2), dtype=np.float32))


def test_fusion_uses_configurable_normalized_weighted_sum() -> None:
    image = np.array([[1.0, 0.0]], dtype=np.float32)
    text = np.array([[0.0, 1.0]], dtype=np.float32)
    fused = fuse_clip_embeddings(image, text, image_weight=0.4, text_weight=0.6)
    expected = np.array([[0.4, 0.6]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(fused, expected, atol=1e-6)


def test_cosine_scores_validate_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions differ"):
        cosine_scores(np.ones(2), np.ones((3, 4)))


def test_ranking_filters_category_and_breaks_ties_by_item_id() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["b", "a", "c"],
            "broad_category": ["shoes", "shoes", "tops"],
        }
    )
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    ranked = rank_candidates(
        np.array([1.0, 0.0]),
        items,
        embeddings,
        target_category="shoes",
        top_k=5,
    )
    assert ranked["item_id"].tolist() == ["a", "b"]
    assert ranked["broad_category"].tolist() == ["shoes", "shoes"]

