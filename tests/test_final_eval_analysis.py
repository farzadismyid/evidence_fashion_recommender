import numpy as np
import pandas as pd

from evidence_fashion_recommender.evaluation.final_judging import (
    is_cross_model_judgment,
    model_family,
    primary_and_sensitivity_summaries,
)
from evidence_fashion_recommender.evaluation.final_reporting import external_grounding_table
from evidence_fashion_recommender.evaluation.statistics import (
    outfit_clustered_paired_bootstrap,
)


def test_cross_model_primary_excludes_same_base_family() -> None:
    assert model_family("mistral:latest@abc:temperature=0") == "mistral"
    assert not is_cross_model_judgment("mistral@abc", "mistral:latest@def")
    assert is_cross_model_judgment("mistral@abc", "qwen3:8b@def")


def test_primary_summary_excludes_self_family_rows() -> None:
    frame = pd.DataFrame(
        [
            {"grounding_variant": "no_rag", "cross_model_primary_eligible": False},
            {"grounding_variant": "no_rag", "cross_model_primary_eligible": True},
        ]
    )
    for dimension in (
        "input_consistency",
        "general_quality",
        "clarity",
        "specificity",
        "hallucination_risk",
        "evidence_misuse",
    ):
        frame[dimension] = [1, 5]
    primary, sensitivity = primary_and_sensitivity_summaries(frame)
    assert primary.loc[0, "general_quality"] == 5
    assert sensitivity.loc[0, "general_quality"] == 3


def test_clustered_bootstrap_reports_case_and_outfit_counts() -> None:
    frame = pd.DataFrame(
        {
            "outfit": ["o1", "o1", "o2"],
            "a": [1.0, 1.0, 0.0],
            "b": [0.0, 0.0, 0.0],
        }
    )
    result = outfit_clustered_paired_bootstrap(
        frame,
        outfit_column="outfit",
        first_column="a",
        second_column="b",
        samples=100,
        confidence_level=0.95,
        seed=42,
    )
    assert result["cases"] == 3
    assert result["unique_outfits"] == 2
    assert np.isclose(result["mean_difference"], 2 / 3)


def test_external_grounding_uses_na_not_zero_for_no_rag() -> None:
    claims = pd.DataFrame(
        [
            {
                "grounding_variant": "no_rag",
                "claim_extraction_failed": False,
                "support_label": "supported_by_query_or_locked_item",
            },
            {
                "grounding_variant": "rule_rag",
                "claim_extraction_failed": False,
                "support_label": "supported_by_rule_evidence",
            },
        ]
    )
    table = external_grounding_table(claims).set_index("grounding_variant")
    assert pd.isna(table.loc["no_rag", "rule_supported_claim_rate"])
    assert table.loc["no_rag", "external_rule_grounding_status"] == "N/A"
    assert table.loc["rule_rag", "rule_supported_claim_rate"] == 1
