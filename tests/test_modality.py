import numpy as np
import pandas as pd

from evidence_fashion_recommender.evaluation.modality import (
    evaluate_modality_case,
    score_modality_candidates,
    select_fusion_weight,
)
from evidence_fashion_recommender.models.multimodal import fuse_embeddings


def test_fuse_embeddings_validates_and_normalizes() -> None:
    image = np.array([[1.0, 0.0]], dtype=np.float32)
    text = np.array([[0.0, 1.0]], dtype=np.float32)
    fused = fuse_embeddings(image, text, 0.5)
    assert np.isclose(np.linalg.norm(fused[0]), 1.0)
    assert np.allclose(fused[0], [2**-0.5, 2**-0.5])


def test_modality_methods_share_one_relevance_vector() -> None:
    candidates = np.eye(2, dtype=np.float32)
    scores = score_modality_candidates(
        candidates,
        candidates,
        candidates,
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        [1.0, 0.5, 0.0],
    )
    result = evaluate_modality_case(
        case_id="c1",
        outfit_id="o1",
        target_category="shoes",
        relevance=np.array([1, 0]),
        scores=scores,
        cutoffs=[1, 5, 10],
    )
    assert set(result["method"]) == {
        "minilm_text",
        "clip_image",
        "clip_text",
        "clip_fused_i1.00",
        "clip_fused_i0.50",
        "clip_fused_i0.00",
    }
    assert result["paper_case_id"].nunique() == 1


def test_fusion_selection_prefers_balanced_setting_after_exact_ties() -> None:
    summary = pd.DataFrame(
        [
            {
                "method": "clip_fused_i0.80",
                "ndcg_at_10": 0.2,
                "hit_rate_at_10": 0.3,
                "reciprocal_rank": 0.1,
            },
            {
                "method": "clip_fused_i0.60",
                "ndcg_at_10": 0.2,
                "hit_rate_at_10": 0.3,
                "reciprocal_rank": 0.1,
            },
        ]
    )
    selected = select_fusion_weight(summary)
    assert selected["image_weight"] == 0.6
    assert np.isclose(selected["text_weight"], 0.4)
