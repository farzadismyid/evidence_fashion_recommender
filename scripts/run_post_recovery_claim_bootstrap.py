"""Reproduce post-recovery case-clustered claim bootstrap outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evidence_fashion_recommender.evaluation.post_recovery_claim_bootstrap import (
    run_case_clustered_claim_bootstrap,
)


def _rate(value: float) -> str:
    return f"{100 * value:.2f}%"


def _write_report(
    estimates: pd.DataFrame,
    differences: pd.DataFrame,
    metadata: dict[str, int | float],
    path: Path,
) -> None:
    lines = [
        "# Post-recovery claim-level clustered bootstrap",
        "",
        "Claim rates use only `verification_status == complete`; extraction and verification "
        "N/A rows "
        "are excluded from those denominators. N/A coverage is separately the fraction of "
        "explanation-level checkpoint records with `verification_status == N/A`.",
        "",
        f"Case-level clusters: {metadata['test_cases']}; replicates: "
        f"{metadata['bootstrap_samples']}; seed: {metadata['seed']}; intervals: 95% percentile.",
        "",
        "## Variant estimates",
        "",
        "| Metric | Variant | Estimate (95% CI) | Numerator / denominator |",
        "|---|---|---:|---:|",
    ]
    for row in estimates.itertuples(index=False):
        estimate = (
            "N/A (not applicable)"
            if not row.applicable
            else (f"{_rate(row.estimate)} ({_rate(row.ci_lower)}, {_rate(row.ci_upper)})")
        )
        lines.append(
            f"| {row.metric} | {row.grounding_variant} | {estimate} | "
            f"{row.numerator} / {row.denominator} |"
        )
    lines.extend(
        [
            "",
            "## Paired case-clustered differences",
            "",
            "| Metric | Contrast | Difference pp (95% CI) | Bootstrap p-value |",
            "|---|---|---:|---:|",
        ]
    )
    for row in differences.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.variant_a} minus {row.variant_b} | "
            f"{row.difference_percentage_points:.2f} ("
            f"{row.ci_lower_percentage_points:.2f}, {row.ci_upper_percentage_points:.2f}) | "
            f"{row.p_value_display} |"
        )
    lines.extend(
        [
            "",
            "Machine-readable CSV and JSON outputs are in "
            "`outputs/final_eval_v2/post_recovery/statistics/`. The script preserves complete "
            "case clusters, including all generators, variants, and valid claims for every "
            "sampled test case.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("outputs/final_eval_v2/post_recovery")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/final_eval_v2/post_recovery/POST_RECOVERY_CLAIM_BOOTSTRAP.md"),
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    verification_root = args.artifact_root / "claims/verification"
    verified = pd.read_csv(verification_root / "verified_claims.csv", keep_default_na=False)
    checkpoint = pd.read_json(verification_root / "verification_checkpoint.jsonl", lines=True)
    estimates, differences, metadata = run_case_clustered_claim_bootstrap(
        verified, checkpoint, samples=args.samples, confidence_level=0.95, seed=args.seed
    )
    if metadata["test_cases"] != 300 or metadata["case_variant_clusters"] != 1200:
        raise ValueError(f"Unexpected post-recovery cluster dimensions: {metadata}")
    output_root = args.artifact_root / "statistics"
    output_root.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(output_root / "claim_metric_estimates.csv", index=False)
    differences.to_csv(output_root / "claim_metric_paired_differences.csv", index=False)
    (output_root / "claim_bootstrap_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _write_report(estimates, differences, metadata, args.report)


if __name__ == "__main__":
    main()
