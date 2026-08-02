import json
from pathlib import Path

import pandas as pd

from evidence_fashion_recommender.evaluation.stage1_preparation import (
    freeze_evidence_in_loop_reranking_selection,
)


def test_freezes_pareto_selection_and_preserves_accuracy_baseline(tmp_path: Path) -> None:
    summary = tmp_path / "validation_summary.csv"
    pd.DataFrame(
        [
            {
                "clip_weight": 1.0,
                "evidence_weight": 0.0,
                "hit_rate_at_10": 0.27,
                "ndcg_at_10": 0.145,
                "reciprocal_rank": 0.148,
            },
            {
                "clip_weight": 0.75,
                "evidence_weight": 0.25,
                "hit_rate_at_10": 0.25,
                "ndcg_at_10": 0.136,
                "reciprocal_rank": 0.140,
            },
        ]
    ).to_csv(summary, index=False)
    selected_path = tmp_path / "selected_weight.json"
    selected_path.write_text(
        json.dumps(
            {"clip_weight": 1.0, "evidence_weight": 0.0, "selected_on": "validation"}
        ),
        encoding="utf-8",
    )
    (tmp_path / "stage_manifest.json").write_text("{}", encoding="utf-8")

    selected = freeze_evidence_in_loop_reranking_selection(
        summary_path=summary,
        selected_path=selected_path,
        clip_weight=0.75,
        selection_policy="evidence_in_loop_pareto_v2",
    )

    assert selected["clip_weight"] == 0.75
    assert selected["evidence_weight"] == 0.25
    assert selected["selection_policy"] == "evidence_in_loop_pareto_v2"
    baseline = json.loads(
        (tmp_path / "selected_weight_accuracy_optimal.json").read_text(encoding="utf-8")
    )
    assert baseline["method_role"] == "accuracy_optimal_baseline"
    assert baseline["superseded_as_proposed_method"] is True
