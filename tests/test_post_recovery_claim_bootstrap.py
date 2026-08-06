from __future__ import annotations

import pandas as pd

from evidence_fashion_recommender.evaluation.post_recovery_claim_bootstrap import (
    build_case_clusters,
    run_case_clustered_claim_bootstrap,
)

VARIANTS = ("no_rag", "item_rag", "rule_rag", "hybrid_rag")


def _fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_rows = []
    claim_rows = []
    for case in ("case_1", "case_2"):
        for variant in VARIANTS:
            status = "N/A" if case == "case_2" and variant == "hybrid_rag" else "complete"
            checkpoint_rows.append(
                {
                    "paper_case_id": case,
                    "grounding_variant": variant,
                    "generation_model": "generator",
                    "verification_status": status,
                }
            )
            if status == "complete":
                label = (
                    "supported_by_rule_evidence"
                    if variant in {"rule_rag", "hybrid_rag"}
                    else "unsupported"
                )
                claim_rows.append(
                    {
                        "paper_case_id": case,
                        "grounding_variant": variant,
                        "generation_model": "generator",
                        "verification_status": "complete",
                        "support_label": label,
                    }
                )
    return pd.DataFrame(claim_rows), pd.DataFrame(checkpoint_rows)


def test_clusters_retain_all_claims_within_each_case() -> None:
    claims, checkpoint = _fixture_frames()
    extra = pd.DataFrame(
        [
            {
                "paper_case_id": "case_1",
                "grounding_variant": "no_rag",
                "generation_model": "generator",
                "verification_status": "complete",
                "support_label": "unsupported",
            }
            for _ in range(10)
        ]
    )
    clusters = build_case_clusters(pd.concat([claims, extra], ignore_index=True), checkpoint)
    no_rag = clusters[clusters["grounding_variant"] == "no_rag"].set_index("paper_case_id")
    assert no_rag.loc["case_1", "claim_denominator"] == 11
    assert no_rag.loc["case_2", "claim_denominator"] == 1


def test_bootstrap_is_deterministic_and_resamples_cases_not_claims() -> None:
    claims, checkpoint = _fixture_frames()
    extra = pd.DataFrame(
        [
            {
                "paper_case_id": "case_1",
                "grounding_variant": "no_rag",
                "generation_model": "generator",
                "verification_status": "complete",
                "support_label": "supported_by_rule_evidence",
            }
            for _ in range(10)
        ]
    )
    first = run_case_clustered_claim_bootstrap(
        pd.concat([claims, extra], ignore_index=True), checkpoint, samples=200, seed=42
    )
    second = run_case_clustered_claim_bootstrap(
        pd.concat([claims, extra], ignore_index=True), checkpoint, samples=200, seed=42
    )
    pd.testing.assert_frame_equal(first[0], second[0])
    no_rag = (
        first[0]
        .query("metric == 'any_permitted_evidence_support' and grounding_variant == 'no_rag'")
        .iloc[0]
    )
    assert no_rag["ci_lower"] == 0
    assert no_rag["ci_upper"] == 10 / 11


def test_na_rows_are_excluded_from_claim_denominators_and_reported_separately() -> None:
    claims, checkpoint = _fixture_frames()
    estimates, _, _ = run_case_clustered_claim_bootstrap(claims, checkpoint, samples=20, seed=42)
    hybrid_support = estimates.query(
        "metric == 'any_permitted_evidence_support' and grounding_variant == 'hybrid_rag'"
    ).iloc[0]
    hybrid_na = estimates[
        (estimates["metric"] == "claim_evaluation_na_coverage")
        & (estimates["grounding_variant"] == "hybrid_rag")
    ].iloc[0]
    assert hybrid_support["denominator"] == 1
    assert hybrid_support["estimate"] == 1
    assert hybrid_na["numerator"] == 1
    assert hybrid_na["denominator"] == 2
    assert hybrid_na["estimate"] == 0.5


def test_paired_comparisons_align_same_case_clusters() -> None:
    claims, checkpoint = _fixture_frames()
    _, differences, _ = run_case_clustered_claim_bootstrap(claims, checkpoint, samples=100, seed=42)
    comparison = differences[
        (differences["metric"] == "any_permitted_evidence_support")
        & (differences["variant_a"] == "rule_rag")
        & (differences["variant_b"] == "no_rag")
    ].iloc[0]
    assert comparison["difference_percentage_points"] == 100
    assert comparison["cluster_count"] == 2


def test_expected_metric_variant_and_comparison_counts() -> None:
    claims, checkpoint = _fixture_frames()
    estimates, differences, metadata = run_case_clustered_claim_bootstrap(
        claims, checkpoint, samples=10
    )
    assert len(estimates) == 36
    assert len(differences) == 12
    assert metadata == {
        "test_cases": 2,
        "case_variant_clusters": 8,
        "bootstrap_samples": 10,
        "confidence_level": 0.95,
        "seed": 42,
    }
