import pandas as pd

from evidence_fashion_recommender.cache import ArtifactCache
from evidence_fashion_recommender.evaluation.robustness import (
    judge_robustness_study,
    one_factor_hybrid_specs,
)


def test_one_factor_grid_avoids_full_factorial() -> None:
    specs = one_factor_hybrid_specs([55, 75, 100], [1, 2, 3, 5], ["candidate_first", "rules_first"])
    assert len(specs) == 8
    assert len({spec.name for spec in specs}) == 8


class _FakeJudge:
    model_id = "fake@1"

    def generate(self, prompt: str) -> str:
        return (
            '{"faithfulness_to_available_information":4,"usefulness_to_user":5,'
            '"specificity":4,"style_appropriateness":5,"grounding_safety":5,'
            '"claims":[{"claim":"compatible","support":"supported"}],'
            '"brief_reason":"grounded"}'
        )


def test_robustness_judge_combines_scores_and_claims(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "paper_case_id": "c1",
                "grounding_variant": "item_rag",
                "generation_model": "generator@1",
                "query_text": "black dress",
                "user_request": "recommend shoes",
                "recommended_text": "black pumps",
                "item_evidence_text": "black pumps",
                "generated_explanation": "The pumps are compatible.",
            }
        ]
    )
    judged, errors = judge_robustness_study(frame, [_FakeJudge()], ArtifactCache(tmp_path))
    assert errors.empty
    assert list(errors.columns) == [
        "paper_case_id",
        "grounding_variant",
        "generation_model",
        "judge_model",
        "error",
    ]
    assert judged.loc[0, "claim_support_rate"] == 1
    assert judged.loc[0, "overall_judge_score"] == 4.6
    assert not judged.loc[0, "self_judge"]
