from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_fashion_recommender.config import FinalEvaluationConfig, load_config


def test_final_eval_v2_config_uses_versioned_roots_and_full_grid() -> None:
    config = load_config("configs/final_eval_v2.yaml")
    final = config.final_evaluation
    assert final.output_root == Path("outputs/final_eval_v2")
    assert final.report_root == Path("reports/final_eval_v2")
    assert len(final.fusion_image_weights) == 11
    assert (
        len(final.hybrid_word_budgets)
        * len(final.hybrid_rule_counts)
        * len(final.hybrid_item_counts)
        * len(final.hybrid_evidence_orders)
        == 36
    )


def test_final_eval_rejects_non_versioned_output_root() -> None:
    with pytest.raises(ValidationError, match="outputs/final_eval_v2"):
        FinalEvaluationConfig(output_root=Path("outputs/final_evaluation"))
