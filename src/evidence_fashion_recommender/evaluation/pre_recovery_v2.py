"""Build the paper-ready final_eval_v2 analysis without recovering N/A rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from ..cache import file_fingerprint
from .final_judging import GENERAL_DIMENSIONS
from .statistics import paired_bootstrap

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

RETRIEVAL_LABELS = {
    "minilm_text": "Text-only (MiniLM)",
    "clip_image": "CLIP image-only",
    "clip_text": "CLIP text-only",
    "clip_fused_i0.40": "Fused CLIP",
    "evidence_reranked": "Fused CLIP + evidence",
}
VARIANT_ORDER = ["no_rag", "item_rag", "rule_rag", "hybrid_rag"]
CLAIM_LABELS = [
    "supported_by_rule_evidence",
    "supported_by_item_evidence",
    "supported_by_query_or_locked_item",
    "unsupported",
    "contradicted",
    "not_verifiable",
]


def _effect_size(differences: np.ndarray) -> float:
    standard_deviation = differences.std(ddof=1)
    return float(differences.mean() / standard_deviation) if standard_deviation else 0.0


def _paired_comparisons(
    frame: pd.DataFrame,
    *,
    id_columns: list[str],
    variant_column: str,
    metrics: list[str],
    comparisons: list[tuple[str, str]],
    samples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        pivot = frame.pivot_table(
            index=id_columns, columns=variant_column, values=metric, aggfunc="mean"
        )
        for first, second in comparisons:
            paired = pivot[[first, second]].dropna()
            result = paired_bootstrap(
                paired[first].to_numpy(),
                paired[second].to_numpy(),
                samples,
                confidence_level,
                seed,
            )
            differences = paired[first].to_numpy() - paired[second].to_numpy()
            rows.append(
                {
                    "metric": metric,
                    "variant_a": first,
                    "variant_b": second,
                    "n": len(paired),
                    **result,
                    "paired_effect_size_dz": _effect_size(differences),
                }
            )
    return pd.DataFrame(rows)


def _claim_summary(verified: pd.DataFrame, checkpoint: list[dict[str, Any]]) -> pd.DataFrame:
    valid = verified[verified["verification_status"] == "complete"].copy()
    rows = []
    for variant in VARIANT_ORDER:
        group = valid[valid["grounding_variant"] == variant]
        failures = [
            row
            for row in checkpoint
            if row["grounding_variant"] == variant and row["verification_status"] == "N/A"
        ]
        counts = group["support_label"].value_counts()
        total = len(group)
        row: dict[str, Any] = {
            "grounding_variant": variant,
            "verified_claims": total,
            "na_explanations": len(failures),
            "extraction_failure_na": sum(
                bool(value["claim_extraction_failed"]) for value in failures
            ),
            "verification_failure_na": sum(
                bool(value.get("claim_verification_failed")) for value in failures
            ),
        }
        for label in CLAIM_LABELS:
            row[f"{label}_count"] = int(counts.get(label, 0))
            row[f"{label}_rate"] = float(counts.get(label, 0) / total) if total else np.nan
        row["any_supported_claim_rate"] = float(
            group["support_label"].str.startswith("supported_by_").mean()
        )
        row["generation_rule_supported_rate"] = (
            row["supported_by_rule_evidence_rate"]
            if variant in {"rule_rag", "hybrid_rag"}
            else pd.NA
        )
        row["generation_item_supported_rate"] = (
            row["supported_by_item_evidence_rate"]
            if variant in {"item_rag", "hybrid_rag"}
            else pd.NA
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _qualitative_examples(
    explanations: pd.DataFrame, judged: pd.DataFrame, verified: pd.DataFrame
) -> pd.DataFrame:
    primary = judged[
        (~judged["judging_failed"].astype(bool))
        & judged["cross_model_primary_eligible"].astype(bool)
    ]
    scores = (
        primary.groupby(["paper_case_id", "generation_model", "grounding_variant"], as_index=False)[
            "general_quality"
        ]
        .mean()
        .pivot_table(
            index=["paper_case_id", "generation_model"],
            columns="grounding_variant",
            values="general_quality",
        )
        .dropna()
    )
    scores["hybrid_minus_no_rag"] = scores["hybrid_rag"] - scores["no_rag"]
    selected = pd.concat(
        [scores.nlargest(3, "hybrid_minus_no_rag"), scores.nsmallest(3, "hybrid_minus_no_rag")]
    ).reset_index()
    text = explanations[
        ["paper_case_id", "generation_model", "grounding_variant", "generated_explanation"]
    ]
    output = selected.merge(text, on=["paper_case_id", "generation_model"], how="left")
    valid_claims = verified[verified["verification_status"] == "complete"].copy()
    valid_claims["supported"] = valid_claims["support_label"].str.startswith("supported_by_")
    support = valid_claims.groupby(
        ["paper_case_id", "generation_model", "grounding_variant"], as_index=False
    ).agg(claims=("claim_id", "size"), supported_claim_rate=("supported", "mean"))
    return output.merge(
        support, on=["paper_case_id", "generation_model", "grounding_variant"], how="left"
    ).sort_values(["hybrid_minus_no_rag", "paper_case_id", "grounding_variant"])


def _save_figures(
    output_dir: Path,
    retrieval: pd.DataFrame,
    general: pd.DataFrame,
    claims: pd.DataFrame,
    lengths: pd.DataFrame,
) -> None:
    plt.figure(figsize=(9, 4.8))
    positions = np.arange(len(retrieval))
    width = 0.36
    plt.bar(positions - width / 2, retrieval["hit_rate_at_10"], width, label="HR@10")
    plt.bar(positions + width / 2, retrieval["ndcg_at_10"], width, label="NDCG@10")
    plt.xticks(positions, retrieval["method_label"], rotation=22, ha="right")
    plt.ylabel("Mean test score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "figure_retrieval.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    ordered = general.set_index("grounding_variant").loc[VARIANT_ORDER].reset_index()
    plt.bar(ordered["grounding_variant"], ordered["general_quality"])
    plt.ylim(1, 5)
    plt.ylabel("Cross-model general quality")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_general_quality.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    ordered = claims.set_index("grounding_variant").loc[VARIANT_ORDER].reset_index()
    plt.bar(ordered["grounding_variant"], ordered["any_supported_claim_rate"], label="Supported")
    plt.bar(
        ordered["grounding_variant"],
        ordered["unsupported_rate"],
        bottom=ordered["any_supported_claim_rate"],
        label="Unsupported",
    )
    plt.ylabel("Claim rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "figure_claim_rates.png", dpi=180)
    plt.close()

    length_plot = lengths.copy()
    length_plot["generator"] = length_plot["generation_model"].str.split("@").str[0]
    pivot = length_plot.pivot(
        index="grounding_variant", columns="generator", values="length_compliance_rate"
    )
    pivot.loc[VARIANT_ORDER].plot(kind="bar", figsize=(8, 4.5))
    plt.ylabel("35-word compliance rate")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_length_compliance.png", dpi=180)
    plt.close()


def build_pre_recovery_analysis(
    *,
    artifact_root: Path,
    output_dir: Path,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    analysis_label: str = "PRE-RECOVERY FINAL ANALYSIS",
    stage4d_recovery_run: bool = False,
) -> dict[str, Any]:
    """Aggregate immutable Stage 1--4 artifacts into a labeled result bundle."""

    paths = {
        "retrieval": artifact_root / "retrieval/test/test_ranking_results.csv",
        "reranking_validation": artifact_root
        / "validation/reranking_tuning/validation_summary.csv",
        "reranking_selection": artifact_root / "validation/reranking_tuning/selected_weight.json",
        "explanations": artifact_root / "explanations/explanations.csv",
        "verified_claims": artifact_root / "claims/verification/verified_claims.csv",
        "verification_checkpoint": artifact_root
        / "claims/verification/verification_checkpoint.jsonl",
        "judgments": artifact_root / "judging/general/judge_results.csv",
        "judgment_checkpoint": artifact_root / "judging/general/general_judging_checkpoint.jsonl",
        "length_summary": artifact_root / "judging/general/length_compliance_summary.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing pre-recovery inputs: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking = pd.read_csv(paths["retrieval"])
    explanations = pd.read_csv(paths["explanations"])
    verified = pd.read_csv(paths["verified_claims"])
    judged = pd.read_csv(paths["judgments"])
    lengths = pd.read_csv(paths["length_summary"])
    verification_checkpoint = [
        json.loads(line)
        for line in paths["verification_checkpoint"].read_text(encoding="utf-8").splitlines()
        if line
    ]
    judgment_checkpoint = [
        json.loads(line)
        for line in paths["judgment_checkpoint"].read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(explanations) != 3600 or len(verification_checkpoint) != 3600:
        raise ValueError("Stage 4B coverage must be exactly 3,600 unique explanations.")
    if len(judgment_checkpoint) != 10800:
        raise ValueError("Stage 4C coverage must be exactly 10,800 judgments.")
    if len({row["explanation_key"] for row in verification_checkpoint}) != 3600:
        raise ValueError("Stage 4B checkpoint contains duplicate explanation keys.")
    if len({row["judgment_key"] for row in judgment_checkpoint}) != 10800:
        raise ValueError("Stage 4C checkpoint contains duplicate judgment keys.")

    retrieval_summary = (
        ranking.groupby("method", as_index=False)
        .mean(numeric_only=True)
        .loc[lambda frame: frame["method"].isin(RETRIEVAL_LABELS)]
    )
    retrieval_summary["method_label"] = retrieval_summary["method"].map(RETRIEVAL_LABELS)
    retrieval_summary["method_order"] = retrieval_summary["method"].map(
        {name: index for index, name in enumerate(RETRIEVAL_LABELS)}
    )
    retrieval_summary = retrieval_summary.sort_values("method_order")
    retrieval_summary = retrieval_summary[
        [
            "method",
            "method_label",
            *[f"hit_rate_at_{cutoff}" for cutoff in (1, 3, 5, 10)],
            *[f"ndcg_at_{cutoff}" for cutoff in (1, 3, 5, 10)],
            "reciprocal_rank",
        ]
    ]
    retrieval_summary.to_csv(output_dir / "recommendation_metrics.csv", index=False)

    retrieval_stats = _paired_comparisons(
        ranking,
        id_columns=["paper_case_id"],
        variant_column="method",
        metrics=["hit_rate_at_10", "ndcg_at_10", "reciprocal_rank"],
        comparisons=[
            ("clip_fused_i0.40", "minilm_text"),
            ("clip_fused_i0.40", "clip_image"),
            ("clip_fused_i0.40", "clip_text"),
            ("evidence_reranked", "clip_fused_i0.40"),
        ],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    retrieval_stats.to_csv(output_dir / "recommendation_statistical_comparisons.csv", index=False)

    claim_summary = _claim_summary(verified, verification_checkpoint)
    claim_summary.to_csv(output_dir / "claim_support_summary.csv", index=False)

    successful_judgments = judged[~judged["judging_failed"].astype(bool)].copy()
    primary = successful_judgments[
        successful_judgments["cross_model_primary_eligible"].astype(bool)
    ]
    primary_summary = primary.groupby("grounding_variant", as_index=False)[
        list(GENERAL_DIMENSIONS)
    ].mean()
    sensitivity_summary = successful_judgments.groupby("grounding_variant", as_index=False)[
        list(GENERAL_DIMENSIONS)
    ].mean()
    primary_summary.to_csv(output_dir / "general_quality_primary_cross_model.csv", index=False)
    sensitivity_summary.to_csv(
        output_dir / "general_quality_sensitivity_all_judges.csv", index=False
    )

    per_explanation = primary.groupby(
        ["paper_case_id", "generation_model", "grounding_variant"], as_index=False
    )[list(GENERAL_DIMENSIONS)].mean()
    explanation_stats = _paired_comparisons(
        per_explanation,
        id_columns=["paper_case_id", "generation_model"],
        variant_column="grounding_variant",
        metrics=list(GENERAL_DIMENSIONS),
        comparisons=[
            ("rule_rag", "no_rag"),
            ("hybrid_rag", "no_rag"),
            ("rule_rag", "item_rag"),
            ("hybrid_rag", "rule_rag"),
        ],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    explanation_stats.to_csv(output_dir / "explanation_statistical_comparisons.csv", index=False)
    lengths.to_csv(output_dir / "length_compliance_summary.csv", index=False)
    qualitative = _qualitative_examples(explanations, judged, verified)
    qualitative.to_csv(output_dir / "qualitative_examples.csv", index=False)

    reranking_validation = pd.read_csv(paths["reranking_validation"])
    reranking_validation.to_csv(output_dir / "reranking_validation_tradeoff.csv", index=False)
    selection = json.loads(paths["reranking_selection"].read_text(encoding="utf-8"))
    (output_dir / "reranking_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    _save_figures(output_dir, retrieval_summary, primary_summary, claim_summary, lengths)

    inventory = []
    for name, path in paths.items():
        if path.suffix == ".csv":
            row_count = len(pd.read_csv(path))
        elif path.suffix == ".jsonl":
            row_count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
        else:
            row_count = 1
        inventory.append(
            {
                "artifact": name,
                "path": str(path),
                "sha256": file_fingerprint(path),
                "rows": row_count,
            }
        )
    pd.DataFrame(inventory).to_csv(output_dir / "artifact_inventory.csv", index=False)

    extraction_na = sum(bool(row["claim_extraction_failed"]) for row in verification_checkpoint)
    verification_na = sum(
        bool(row.get("claim_verification_failed")) for row in verification_checkpoint
    )
    judgment_na = sum(bool(row["judging_failed"]) for row in judgment_checkpoint)
    manifest = {
        "analysis_label": analysis_label,
        "stage4d_recovery_run": stage4d_recovery_run,
        "explanations": 3600,
        "verification_rows": 3600,
        "judgment_rows": 10800,
        "extraction_na": extraction_na,
        "verification_na": verification_na,
        "judgment_na": judgment_na,
        "primary_judging": "cross_model_only",
        "sensitivity_judging": "all_judges_including_self",
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "seed": seed,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
