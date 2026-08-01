"""Final reproducibility bundle and systematic-evaluation report."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_final_report(
    baseline_run: Path,
    improved_run: Path,
    study_run: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    baseline = pd.read_csv(baseline_run / "metrics" / "controlled_ranking_summary.csv")
    improved = pd.read_csv(improved_run / "metrics" / "controlled_ranking_summary.csv")
    study_dir = study_run / "metrics" / "explanation_study"
    automatic = pd.read_csv(study_dir / "automatic_summary.csv")
    judge = pd.read_csv(study_dir / "independent_judge_summary.csv")
    rag = pd.read_csv(study_dir / "rag_retrieval_summary.csv")
    statistics = pd.read_csv(study_dir / "statistical_tests.csv")

    baseline.assign(experiment="paper_baseline_v3").to_csv(
        output_dir / "recommendation_metrics.csv", index=False
    )
    improved.assign(experiment="paper_improved_light_rerank").to_csv(
        output_dir / "recommendation_metrics_improved.csv", index=False
    )
    for name, frame in [
        ("explanation_metrics", automatic),
        ("independent_judge_metrics", judge),
        ("rag_retrieval_metrics", rag),
        ("statistical_tests", statistics),
    ]:
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    ranking_plot = baseline.set_index("model_name")[
        ["hit_rate_at_1", "hit_rate_at_5", "hit_rate_at_10"]
    ]
    ranking_plot.plot(kind="bar", figsize=(9, 5))
    plt.ylabel("Hit rate")
    plt.title("Controlled recommendation accuracy")
    plt.tight_layout()
    plt.savefig(figures / "recommendation_hit_rates.png", dpi=180)
    plt.close()

    automatic.set_index("grounding_variant")[
        ["mean_unsupported_claim_count", "mean_evidence_overlap"]
    ].plot(kind="bar", figsize=(9, 5))
    plt.title("Automatic explanation faithfulness")
    plt.tight_layout()
    plt.savefig(figures / "explanation_faithfulness.png", dpi=180)
    plt.close()

    judge.set_index("grounding_variant")[
        [
            "faithfulness_to_available_information",
            "usefulness_to_user",
            "grounding_safety",
        ]
    ].plot(kind="bar", figsize=(9, 5), ylim=(0, 5))
    plt.title("Independent LLM judge")
    plt.tight_layout()
    plt.savefig(figures / "independent_judge.png", dpi=180)
    plt.close()

    for source in [baseline_run, improved_run, study_run]:
        destination = output_dir / "manifests" / source.name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ["config_resolved.yaml", "run_manifest.json"]:
            shutil.copy2(source / filename, destination / filename)

    best_clip = baseline.loc[baseline["model_name"] == "clip_multimodal"].iloc[0]
    best_evidence = improved.loc[improved["model_name"] == "evidence_reranked"].iloc[0]
    report_path = output_dir / "FINAL_SYSTEMATIC_REPORT.md"
    report_path.write_text(
        f"""# Final systematic evaluation report

## Scope

This report contains the complete non-human evaluation. Human evaluation is explicitly
excluded from the current work and recorded as future work.

## Recommendation evaluation

- Controlled cases: 300
- Same-category negatives per case: 99
- CLIP HitRate@10: {best_clip["hit_rate_at_10"]:.4f}
- CLIP NDCG@10: {best_clip["ndcg_at_10"]:.4f}
- Light evidence-reranked HitRate@10: {best_evidence["hit_rate_at_10"]:.4f}
- Light evidence-reranked NDCG@10: {best_evidence["ndcg_at_10"]:.4f}

The result is an accuracy-grounding trade-off; evidence reranking is not claimed to
outperform pure CLIP.

## Systematic explanation and faithfulness evaluation

The study contains 100 fixed recommendation cases and 400 newly generated explanations:
No-RAG, Item-RAG, Rule-RAG, and Hybrid-RAG. Metrics cover citation behaviour, unsupported
claims, occasion drift, evidence overlap, candidate substitution, prompt leakage,
explanation length, and an independent Qwen3 judge.

## RAG retrieval evaluation

Reported measures include evidence coverage, retrieved-rule count, source reliability,
category/input compatibility, Precision@K, Recall@K against applicable KB rules,
HitRate@K, NDCG@K, reciprocal rank, and unique-rule usage. These are rule-applicability
proxies; no human relevance judgments are claimed.

## Statistical evaluation

Paired bootstrap confidence intervals use the configured 5,000 resamples. All pairwise
variant tests include Holm and Benjamini-Hochberg corrected p-values.

## Important limitation

Human evaluation is future work. Independent LLM judging and deterministic automatic
checks are systematic supporting evidence, not a claim about human preference or
human-perceived explanation quality.
""",
        encoding="utf-8",
    )
    return report_path
