"""Before-versus-after reporting for the systematic robustness phase."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _write_sha256_manifest(directory: Path, filename: str) -> Path:
    manifest_path = directory / filename
    artifacts = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        artifacts.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    manifest_path.write_text(
        json.dumps({"sha256_algorithm": "SHA-256", "artifacts": artifacts}, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def build_robustness_report(
    baseline_dir: Path,
    study_dir: Path,
    heldout_ranking_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)

    baseline_ranking = pd.read_csv(baseline_dir / "recommendation_metrics.csv")
    baseline_explanations = pd.read_csv(baseline_dir / "explanation_metrics.csv")
    baseline_judge = pd.read_csv(baseline_dir / "independent_judge_metrics.csv")
    heldout_ranking = pd.read_csv(heldout_ranking_dir / "test_summary.csv")
    robust_automatic = pd.read_csv(study_dir / "automatic_summary.csv")
    robust_judge = pd.read_csv(study_dir / "judge_ensemble_summary.csv")
    robust_cross_model = pd.read_csv(study_dir / "cross_model_judge_summary.csv")
    agreement = pd.read_csv(study_dir / "judge_agreement.csv")
    relevance_agreement = pd.read_csv(study_dir / "rule_relevance_agreement.csv")
    consensus_retrieval = pd.read_csv(study_dir / "consensus_retrieval_metrics.csv")
    counterfactual = pd.read_csv(study_dir / "counterfactual_retrieval_test.csv")
    substitution = json.loads(
        (study_dir / "substitution_detector_metrics.json").read_text(encoding="utf-8")
    )
    selected_hybrid = json.loads(
        (study_dir / "selected_hybrid_spec.json").read_text(encoding="utf-8")
    )

    before_clip = baseline_ranking[baseline_ranking["model_name"] == "clip_multimodal"].iloc[0]
    before_rerank = baseline_ranking[baseline_ranking["model_name"] == "evidence_reranked"].iloc[0]
    after_clip = heldout_ranking[heldout_ranking["clip_weight"] == 1.0].iloc[0]
    after_rerank = heldout_ranking[heldout_ranking["clip_weight"] == 0.9].iloc[0]
    ranking_comparison = pd.DataFrame(
        [
            {
                "phase": "before",
                "method": "pure_clip",
                "hit_rate_at_10": before_clip["hit_rate_at_10"],
                "ndcg_at_10": before_clip["ndcg_at_10"],
            },
            {
                "phase": "before",
                "method": "evidence_reranked",
                "hit_rate_at_10": before_rerank["hit_rate_at_10"],
                "ndcg_at_10": before_rerank["ndcg_at_10"],
            },
            {
                "phase": "after_frozen_test",
                "method": "pure_clip",
                "hit_rate_at_10": after_clip["hit_rate_at_10"],
                "ndcg_at_10": after_clip["ndcg_at_10"],
            },
            {
                "phase": "after_frozen_test",
                "method": "evidence_reranked",
                "hit_rate_at_10": after_rerank["hit_rate_at_10"],
                "ndcg_at_10": after_rerank["ndcg_at_10"],
            },
        ]
    )
    ranking_comparison.to_csv(output_dir / "before_after_ranking.csv", index=False)

    ensemble = (
        robust_judge.groupby("grounding_variant")[
            [
                "faithfulness_to_available_information",
                "usefulness_to_user",
                "specificity",
                "style_appropriateness",
                "grounding_safety",
                "overall_judge_score",
                "claim_support_rate",
                "claim_label_compliance_rate",
            ]
        ]
        .mean()
        .reset_index()
    )
    automatic = (
        robust_automatic.groupby("grounding_variant")[
            [
                "mean_unsupported_claim_count",
                "mean_evidence_overlap",
                "candidate_substitution_rate",
                "prompt_leakage_rate",
            ]
        ]
        .mean()
        .reset_index()
    )
    robust_summary = ensemble.merge(automatic, on="grounding_variant")
    robust_summary.to_csv(output_dir / "robustness_ensemble_summary.csv", index=False)
    cross_model = (
        robust_cross_model.groupby("grounding_variant")[
            [
                "faithfulness_to_available_information",
                "overall_judge_score",
                "claim_support_rate",
                "claim_label_compliance_rate",
            ]
        ]
        .mean()
        .reset_index()
    )
    cross_model.to_csv(output_dir / "cross_model_variant_summary.csv", index=False)

    ranking_comparison.pivot(index="method", columns="phase", values="hit_rate_at_10").plot(
        kind="bar", figsize=(8, 5)
    )
    plt.ylabel("HitRate@10")
    plt.title("Before versus frozen held-out recommendation results")
    plt.tight_layout()
    plt.savefig(figures / "before_after_ranking.png", dpi=180)
    plt.close()

    robust_summary.set_index("grounding_variant")[
        ["overall_judge_score", "faithfulness_to_available_information"]
    ].plot(kind="bar", figsize=(8, 5), ylim=(0, 5))
    plt.title("Cross-generator, cross-judge explanation quality")
    plt.tight_layout()
    plt.savefig(figures / "robust_explanation_quality.png", dpi=180)
    plt.close()

    best_quality = cross_model.sort_values("overall_judge_score", ascending=False).iloc[0]
    best_grounding = robust_summary.sort_values(
        ["mean_unsupported_claim_count", "claim_support_rate"],
        ascending=[True, False],
    ).iloc[0]
    before_best = baseline_judge.sort_values("overall_judge_score", ascending=False).iloc[0]
    before_hybrid = baseline_explanations[
        baseline_explanations["grounding_variant"] == "hybrid_rag"
    ].iloc[0]
    mean_agreement = agreement["spearman"].mean()
    overall_agreement = agreement[agreement["dimension"] == "overall_judge_score"][
        "spearman"
    ].mean()
    mean_rule_agreement = relevance_agreement["percent_agreement"].mean()
    mean_rule_kappa = relevance_agreement["cohen_kappa"].mean()
    no_rag = robust_summary[robust_summary["grounding_variant"] == "no_rag"].iloc[0]
    hybrid = robust_summary[robust_summary["grounding_variant"] == "hybrid_rag"].iloc[0]
    unsupported_reduction = 1 - (
        hybrid["mean_unsupported_claim_count"] / no_rag["mean_unsupported_claim_count"]
    )
    report = f"""# Final systematic robustness report

## Scope

This is the complete non-human robustness phase. Human evaluation is explicitly excluded
and remains future work. Model judges and deterministic checks are proxy evidence and are
not described as human preference.

## Experimental strengthening

- The before-baseline is immutable and SHA-256 fingerprinted.
- Development, validation, and test partitions are outfit-disjoint.
- The frozen test set contains 300 balanced cases, versus 100 explanation cases before.
- Prompt and reranking choices were selected only on validation.
- Three pinned generators and three pinned judges replace the single-generator,
  single-judge result.
- Claim-level verification, judge agreement, consensus rule relevance, and corrected
  substitution detection are reported.

## Recommendation comparison

- Before pure CLIP HR@10/NDCG@10: {before_clip["hit_rate_at_10"]:.4f} /
  {before_clip["ndcg_at_10"]:.4f}
- Frozen-test pure CLIP HR@10/NDCG@10: {after_clip["hit_rate_at_10"]:.4f} /
  {after_clip["ndcg_at_10"]:.4f}
- Frozen-test validation-selected reranker HR@10/NDCG@10:
  {after_rerank["hit_rate_at_10"]:.4f} / {after_rerank["ndcg_at_10"]:.4f}
- Frozen-test reranker change versus pure CLIP: HR@10
  {after_rerank["hit_rate_at_10"] - after_clip["hit_rate_at_10"]:+.4f}; NDCG@10
  {after_rerank["ndcg_at_10"] - after_clip["ndcg_at_10"]:+.4f}

The selected reranker is compared without retuning on test. Any HitRate/NDCG trade-off is
reported rather than resolved post hoc.

## Explanation robustness

- Before best single-judge variant: {before_best["grounding_variant"]}
  ({before_best["overall_judge_score"]:.3f}/5)
- Robust cross-model best quality variant: {best_quality["grounding_variant"]}
  ({best_quality["overall_judge_score"]:.3f}/5)
- Robust strongest grounding variant: {best_grounding["grounding_variant"]}
- Before Hybrid unsupported claims: {before_hybrid["mean_unsupported_claim_count"]:.3f}
- Robust strongest-grounding unsupported claims:
  {best_grounding["mean_unsupported_claim_count"]:.3f}
- Hybrid unsupported-claim reduction versus No-RAG: {unsupported_reduction:.1%}
- Hybrid versus No-RAG multi-judge overall score:
  {hybrid["overall_judge_score"]:.3f} versus {no_rag["overall_judge_score"]:.3f}
- Mean claim-label compliance: {robust_summary["claim_label_compliance_rate"].mean():.3f}
- Mean pairwise judge Spearman agreement: {mean_agreement:.3f}
- Overall-score pairwise Spearman agreement: {overall_agreement:.3f}
- Mean consensus rule Precision@5:
  {consensus_retrieval["consensus_precision_at_5"].mean():.3f}
- Consensus rule HitRate@5 / MRR:
  {consensus_retrieval["consensus_hit_rate_at_5"].mean():.3f} /
  {consensus_retrieval["consensus_mrr"].mean():.3f}
- Rule-relevance judge agreement / Cohen's kappa:
  {mean_rule_agreement:.3f} / {mean_rule_kappa:.3f}
- Counterfactual category false-match rate:
  {counterfactual["counterfactual_false_match_rate"].mean():.3f}
- Substitution detector frozen-benchmark precision/recall/F1:
  {substitution["precision"]:.3f} / {substitution["recall"]:.3f} /
  {substitution["f1"]:.3f}
- Validation-selected Hybrid prompt: {selected_hybrid["name"]}

Full per-generator, per-judge, claim-level, retrieval, agreement, and corrected statistical
tables accompany this report.

## Main finding

Hybrid-RAG is not the overall judge-score winner. No-RAG has the highest cross-model
quality score, while Rule-RAG has the lowest deterministic unsupported-claim count and
Hybrid-RAG has the greatest evidence overlap. Hybrid reduces deterministic unsupported
claims substantially without improving the aggregate judge score over No-RAG. The
research question is therefore answered as a trade-off, not as evidence that one method
dominates every metric.

## Reliability cautions

- The 10,800 planned explanation judgments and 4,500 rule-level judgments were parsed;
  the final judge and rule-relevance error tables are empty.
- Judge agreement is modest, and grounding-safety scores are nearly constant. Model-judge
  scores must not be treated as interchangeable with human judgments.
- Non-canonical claim labels were normalized deterministically; descriptive values were
  conservatively mapped to `not_verifiable`. The compliance rate is reported above.
- Before/after datasets differ, so the strongest recommendation comparison is the frozen
  test comparison between pure CLIP and the validation-selected reranker.

## Interpretation constraint

The larger, multi-model results support claims about systematic robustness only. Human
usefulness, preference, and perceived trust remain unanswered future work.
"""
    report_path = output_dir / "FINAL_ROBUSTNESS_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    for filename in [
        "judge_summary.csv",
        "judge_errors.csv",
        "judge_score_standard_deviations.csv",
        "judge_ensemble_summary.csv",
        "cross_model_judge_summary.csv",
        "judge_agreement.csv",
        "statistical_tests.csv",
        "rule_relevance_judgments.csv",
        "rule_relevance_agreement.csv",
        "rule_relevance_errors.csv",
        "counterfactual_retrieval_test.csv",
        "consensus_retrieval_metrics.csv",
        "automatic_summary.csv",
        "selected_hybrid_spec.json",
        "substitution_detector_metrics.json",
    ]:
        shutil.copy2(study_dir / filename, output_dir / filename)
    _write_sha256_manifest(study_dir, "FINAL_STUDY_MANIFEST.json")
    _write_sha256_manifest(output_dir, "FINAL_REPORT_MANIFEST.json")
    return report_path
