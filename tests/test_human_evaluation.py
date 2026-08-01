import pandas as pd
import pytest

from evidence_fashion_recommender.evaluation.human import (
    cohen_kappa,
    create_blinded_review_pack,
)


def test_review_pack_hides_variant() -> None:
    explanations = pd.DataFrame(
        [
            {
                "case_id": "C1",
                "grounding_variant": "rule_rag",
                "query_text": "dress",
                "user_request": "formal",
                "recommended_text": "black pumps",
                "generated_explanation": "These work well.",
            }
        ]
    )
    review, key = create_blinded_review_pack(explanations)
    assert "grounding_variant" not in review
    assert key.loc[0, "grounding_variant"] == "rule_rag"


def test_cohen_kappa_perfect_agreement() -> None:
    assert cohen_kappa(pd.Series(["a", "b"]), pd.Series(["a", "b"])) == pytest.approx(1.0)
