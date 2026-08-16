import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evidence_fashion.evaluation.statistics import (
    clustered_bootstrap_mean,
    holm_adjust,
    two_sided_bootstrap_pvalue,
)

ROOT = Path(__file__).parents[1]


def test_clustered_bootstrap_is_deterministic_and_keeps_clustered_rows_together() -> None:
    values = [0.0, 1.0, 1.0, 1.0]
    clusters = ["a", "a", "b", "c"]
    first = clustered_bootstrap_mean(
        values, clusters, replicates=200, confidence_level=0.95, seed=42
    )
    second = clustered_bootstrap_mean(
        values, clusters, replicates=200, confidence_level=0.95, seed=42
    )
    assert first[0] == 0.75
    assert np.array_equal(first[3], second[3])
    assert first[1] <= first[0] <= first[2]


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
    assert two_sided_bootstrap_pvalue([1.0] * 99) == 0.02


def test_stage6_kb_summary_fields_exist() -> None:
    columns = pd.read_csv("data/kb/fashion_rules.csv", nrows=0).columns
    assert {"recommended_category", "source_reliability"}.issubset(columns)


def test_stage6_manifest_outputs_and_confirmatory_contract() -> None:
    manifest = json.loads(
        (ROOT / "artifacts/manifests/stage6_recommendation_manifest.json").read_text()
    )
    assert manifest["row_counts"]["confirmatory_cases"] == 1000
    assert manifest["row_counts"]["candidate_rows"] == 99238
    assert manifest["trace_validation"] == {
        "complete_five_rule_traces": True,
        "locked_cases_checked": 1000,
    }
    for raw_path, expected in manifest["output_artifact_hashes"].items():
        # The central registry is append-only across later stages; the latest stage manifest
        # binds its current hash while Stage 6 continues to bind all immutable Stage 6 outputs.
        if raw_path.endswith("figure_table_registry.csv"):
            continue
        path = ROOT / raw_path
        assert path.exists(), raw_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    results = pd.read_csv(ROOT / "artifacts/tables/table_02_recommendation_results.csv")
    assert len(results) == 245
    assert not results[["estimate", "ci_lower", "ci_upper"]].isna().any().any()
    assert set(results["aggregation"]) == {"micro", "category", "category_macro"}
