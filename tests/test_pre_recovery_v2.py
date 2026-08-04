import numpy as np
import pandas as pd

from evidence_fashion_recommender.cli import build_parser
from evidence_fashion_recommender.evaluation.pre_recovery_v2 import (
    _claim_summary,
    _effect_size,
)


def test_pre_recovery_command_is_exposed() -> None:
    args = build_parser().parse_args(["analyze-final-eval-v2"])
    assert args.command == "analyze-final-eval-v2"
    assert args.output_dir == "reports/final_eval_v2/pre_recovery"


def test_claim_summary_excludes_na_from_rate_denominator() -> None:
    verified = pd.DataFrame(
        [
            {
                "grounding_variant": "no_rag",
                "verification_status": "complete",
                "support_label": "unsupported",
            },
            {
                "grounding_variant": "no_rag",
                "verification_status": "N/A",
                "support_label": pd.NA,
            },
        ]
    )
    checkpoint = [
        {
            "grounding_variant": "no_rag",
            "verification_status": "N/A",
            "claim_extraction_failed": True,
            "claim_verification_failed": False,
        }
    ]
    summary = _claim_summary(verified, checkpoint).set_index("grounding_variant")
    assert summary.loc["no_rag", "verified_claims"] == 1
    assert summary.loc["no_rag", "unsupported_rate"] == 1.0
    assert summary.loc["no_rag", "na_explanations"] == 1


def test_paired_effect_size_handles_constant_difference() -> None:
    assert _effect_size(np.array([1.0, 1.0, 1.0])) == 0.0
