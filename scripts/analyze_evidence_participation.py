"""Analyse frozen expanded-ranking evidence participation without model calls.

This script deliberately consumes only cached embeddings and frozen candidate scores.  The
candidate scorer has no binary support/applicability decision: its five highest-scoring
category rules all enter the continuous score.  Consequently, score-threshold coverage is
reported only as a validation-derived sensitivity analysis, never as confirmed rule support.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evidence_fashion_recommender.models.multimodal import fuse_embeddings
from evidence_fashion_recommender.reranking import weighted_rerank

ROOT = Path("outputs/recommendation_eval_expanded")
OUT = ROOT / "evidence_participation"
REPORT = Path("reports/recommendation_eval_expanded/EVIDENCE_PARTICIPATION_ANALYSIS.md")
FINAL = Path("outputs/final_eval_v2")
CUTS = (1, 5, 10)
SEED = 42
REPLICATES = 5_000
SCORING_RULE_COUNT = 5  # frozen `evidence.candidate_top_k`, not a support count.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def minmax(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    return np.ones(len(values)) if np.isclose(low, high) else (values - low) / (high - low)


def cohort_name(historical: bool | None) -> str:
    if historical is None:
        return "expanded_3000"
    return "historical_300" if historical else "new_2700"


def ranks(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    # Explicit deterministic tie-breaker; normal data have no material ties.
    order = np.lexsort((ids.astype(str), -scores))
    result = np.empty(len(scores), dtype=int)
    result[order] = np.arange(1, len(scores) + 1)
    return result


def thresholds_from_validation() -> dict[str, float]:
    scores = pd.read_csv(
        FINAL / "sources/validation/candidate_sets.csv", usecols=["evidence_score"]
    )["evidence_score"].to_numpy()
    # Pre-declared score-distribution sensitivity points: no outcome is used.
    return {f"validation_q{q}": float(np.quantile(scores, q / 100)) for q in (25, 50, 75)}


def reconstruct_cases() -> pd.DataFrame:
    schedule = pd.read_csv(ROOT / "expanded_schedule.csv")
    candidates = pd.read_csv(ROOT / "candidate_sets.csv")
    image = np.load(ROOT / "query_embeddings/query_clip_image.npy", mmap_mode="r")
    text = np.load(ROOT / "query_embeddings/query_clip_text.npy", mmap_mode="r")
    target_image = np.load(
        FINAL / "materialized/target_embeddings/target_clip_image.npy", mmap_mode="r"
    )
    target_text = np.load(
        FINAL / "materialized/target_embeddings/target_clip_text.npy", mmap_mode="r"
    )
    fusion = json.loads((FINAL / "validation/fusion_tuning/selected_fusion.json").read_text())
    weight = float(fusion["image_weight"])
    rows: list[dict[str, object]] = []
    for i, case in schedule.iterrows():
        pool = candidates[candidates.paper_case_id == case.paper_case_id].copy()
        target_rows = pool.target_row.astype(int).to_numpy()
        clip = (
            fuse_embeddings(target_image[target_rows], target_text[target_rows], weight)
            @ fuse_embeddings(image[i][None, :], text[i][None, :], weight)[0]
        )
        pool["clip_score"] = clip
        reranked = weighted_rerank(pool, 0.75, 0.25, True)
        clip_rank = ranks(clip, pool.item_ID.to_numpy())
        rerank_rank = (
            pd.Series(np.arange(1, len(reranked) + 1), index=reranked.item_ID.astype(str))
            .reindex(pool.item_ID.astype(str))
            .to_numpy(dtype=int)
        )
        clip_order = np.argsort(clip)[::-1]
        rerank_order = (
            reranked.item_ID.astype(str)
            .map(pd.Series(np.arange(len(pool)), index=pool.item_ID.astype(str)))
            .to_numpy(dtype=int)
        )
        evidence = pool.evidence_score.to_numpy(float)
        positives = pool.is_positive.to_numpy(bool)
        top1_changed = pool.item_ID.iloc[clip_order[0]] != reranked.item_ID.iloc[0]
        clip_best_positive = clip_rank[positives].min()
        rerank_best_positive = rerank_rank[positives].min()
        base = {
            "paper_case_id": case.paper_case_id,
            "query_outfit_id": str(case.query_outfit_id),
            "target_category": case.target_category,
            "historical_subset": bool(case.historical_subset),
            "num_candidates": len(pool),
            "num_positives": int(positives.sum()),
            "top1_changed": bool(top1_changed),
            "top1_promoted_score_increased": bool(
                evidence[rerank_order[0]] > evidence[clip_order[0]]
            ),
            "top1_evidence_delta": float(evidence[rerank_order[0]] - evidence[clip_order[0]]),
            "relevant_item_moved_lower": bool(rerank_best_positive > clip_best_positive),
            "improved_evidence_and_relevant_lower": bool(
                top1_changed
                and evidence[rerank_order[0]] > evidence[clip_order[0]]
                and rerank_best_positive > clip_best_positive
            ),
            "mean_abs_rank_shift": float(np.abs(rerank_rank - clip_rank).mean()),
            "median_abs_rank_shift": float(np.median(np.abs(rerank_rank - clip_rank))),
        }
        for method, order in (("clip_fused", clip_order), ("evidence_reranked", rerank_order)):
            record = dict(base, method=method)
            for k in CUTS:
                top = order[:k]
                record[f"mean_evidence_score_at_{k}"] = float(evidence[top].mean())
                record[f"scoring_rule_count_at_{k}"] = float(SCORING_RULE_COUNT)
                record[f"top_{k}_overlap"] = float(
                    len(
                        set(pool.item_ID.iloc[clip_order[:k]])
                        & set(pool.item_ID.iloc[rerank_order[:k]])
                    )
                    / k
                )
            rows.append(record)
    return pd.DataFrame(rows)


def summarize(
    per_case: pd.DataFrame, thresholds: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output: list[dict[str, object]] = []
    categories: list[dict[str, object]] = []
    for historical in (True, False, None):
        subset = (
            per_case if historical is None else per_case[per_case.historical_subset == historical]
        )
        for method, group in subset.groupby("method"):
            base = {"cohort": cohort_name(historical), "method": method, "cases": len(group)}
            values = {
                **{
                    f"mean_evidence_score_at_{k}": group[f"mean_evidence_score_at_{k}"].mean()
                    for k in CUTS
                },
                **{
                    f"mean_scoring_rule_count_at_{k}": group[f"scoring_rule_count_at_{k}"].mean()
                    for k in CUTS
                },
                **{f"rule_backed_coverage_at_{k}": np.nan for k in CUTS},
            }
            if method == "evidence_reranked":
                values |= {
                    "top1_changed_count": int(group.top1_changed.sum()),
                    "top1_changed_rate": group.top1_changed.mean(),
                    "mean_top5_overlap": group.top_5_overlap.mean(),
                    "mean_top10_overlap": group.top_10_overlap.mean(),
                    "mean_abs_rank_shift": group.mean_abs_rank_shift.mean(),
                    "median_abs_rank_shift": group.median_abs_rank_shift.median(),
                    "promoted_higher_score_count": int(
                        (group.top1_changed & group.top1_promoted_score_increased).sum()
                    ),
                    "promoted_higher_score_rate_among_changed": group.loc[
                        group.top1_changed, "top1_promoted_score_increased"
                    ].mean(),
                    "mean_top1_evidence_delta_among_changed": group.loc[
                        group.top1_changed, "top1_evidence_delta"
                    ].mean(),
                    "promoted_with_candidate_applicable_rule_count": np.nan,
                    "promoted_with_candidate_applicable_rule_rate": np.nan,
                    "improved_evidence_relevant_lower_count": int(
                        group.improved_evidence_and_relevant_lower.sum()
                    ),
                    "improved_evidence_relevant_lower_rate": group.improved_evidence_and_relevant_lower.mean(),
                }
            for name, threshold in thresholds.items():
                for k in CUTS:
                    values[f"score_sensitivity_coverage_at_{k}_{name}"] = (
                        group[f"mean_evidence_score_at_{k}"] >= threshold
                    ).mean()
            output.append(base | values)
            for category, cat in group.groupby("target_category"):
                categories.append(
                    base | {"target_category": category, **values}
                    if False
                    else {
                        "cohort": cohort_name(historical),
                        "method": method,
                        "target_category": category,
                        "cases": len(cat),
                        **{
                            f"mean_evidence_score_at_{k}": cat[f"mean_evidence_score_at_{k}"].mean()
                            for k in CUTS
                        },
                        **{
                            f"mean_scoring_rule_count_at_{k}": cat[
                                f"scoring_rule_count_at_{k}"
                            ].mean()
                            for k in CUTS
                        },
                        **{f"rule_backed_coverage_at_{k}": np.nan for k in CUTS},
                        "top1_changed_rate": cat.top1_changed.mean()
                        if method == "evidence_reranked"
                        else np.nan,
                        "mean_top5_overlap": cat.top_5_overlap.mean()
                        if method == "evidence_reranked"
                        else np.nan,
                        "mean_top10_overlap": cat.top_10_overlap.mean()
                        if method == "evidence_reranked"
                        else np.nan,
                    }
                )
    return pd.DataFrame(output), pd.DataFrame(categories)


def bootstrap(per_case: pd.DataFrame) -> pd.DataFrame:
    # A cluster is a query outfit; both method rows and every case in the outfit remain together.
    fused = per_case[per_case.method == "clip_fused"].set_index("paper_case_id")
    rerank = per_case[per_case.method == "evidence_reranked"].set_index("paper_case_id")
    cases = fused.index.to_numpy()
    outfits = fused.query_outfit_id.to_numpy()
    unique, inverse = np.unique(outfits, return_inverse=True)
    metric_map = {
        "mean_evidence_score_at_1": rerank.mean_evidence_score_at_1
        - fused.mean_evidence_score_at_1,
        "mean_evidence_score_at_5": rerank.mean_evidence_score_at_5
        - fused.mean_evidence_score_at_5,
        "mean_evidence_score_at_10": rerank.mean_evidence_score_at_10
        - fused.mean_evidence_score_at_10,
        "scoring_rule_count_at_1": rerank.scoring_rule_count_at_1 - fused.scoring_rule_count_at_1,
        "scoring_rule_count_at_5": rerank.scoring_rule_count_at_5 - fused.scoring_rule_count_at_5,
        "scoring_rule_count_at_10": rerank.scoring_rule_count_at_10
        - fused.scoring_rule_count_at_10,
        "top5_overlap": rerank.top_5_overlap,
        "top10_overlap": rerank.top_10_overlap,
        "mean_abs_rank_shift": rerank.mean_abs_rank_shift,
        "evidence_gain_changed_top1": rerank.top1_evidence_delta.where(rerank.top1_changed),
    }
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(unique), size=(REPLICATES, len(unique)))
    results: list[dict[str, object]] = []
    for metric, series in metric_map.items():
        values = series.reindex(cases).to_numpy(float)
        # Per-outfit sum and denominator make case-weighted cluster bootstrap exact.
        sums = np.bincount(inverse, weights=np.nan_to_num(values), minlength=len(unique))
        counts = np.bincount(inverse, weights=np.isfinite(values), minlength=len(unique))
        denom = counts[draws].sum(axis=1)
        reps = sums[draws].sum(axis=1) / np.where(denom == 0, np.nan, denom)
        estimate = float(np.nansum(values) / np.isfinite(values).sum())
        p = min(1.0, 2 * min(float(np.nanmean(reps <= 0)), float(np.nanmean(reps >= 0))))
        results.append(
            {
                "metric": metric,
                "comparison": "evidence_reranked_minus_clip_fused"
                if "score" in metric or "rule" in metric
                else "paired_structure",
                "estimate": estimate,
                "ci_low": float(np.nanpercentile(reps, 2.5)),
                "ci_high": float(np.nanpercentile(reps, 97.5)),
                "bootstrap_p_value": p,
                "bootstrap_p_display": "p < 1/5000" if p == 0 else f"p = {p:.4f}",
                "clusters": len(unique),
                "cases": len(cases),
                "seed": SEED,
                "replicates": REPLICATES,
            }
        )
    return pd.DataFrame(results)


def validation_sweep() -> pd.DataFrame:
    # Validation result rows give frozen accuracy; candidate score participation is reconstructed
    # from the frozen validation candidate table and cached query/target embeddings.
    summary = pd.read_csv(FINAL / "validation/reranking_tuning/validation_summary.csv")
    candidates = pd.read_csv(FINAL / "sources/validation/candidate_sets.csv")
    schedule = pd.read_csv(FINAL / "materialized/validation/schedule.csv")
    qroot = next((FINAL / "materialized/query_embeddings/validation").iterdir())
    image, text = (
        np.load(qroot / "query_clip_image.npy", mmap_mode="r"),
        np.load(qroot / "query_clip_text.npy", mmap_mode="r"),
    )
    ti, tt = (
        np.load(FINAL / "materialized/target_embeddings/target_clip_image.npy", mmap_mode="r"),
        np.load(FINAL / "materialized/target_embeddings/target_clip_text.npy", mmap_mode="r"),
    )
    fusion = float(
        json.loads((FINAL / "validation/fusion_tuning/selected_fusion.json").read_text())[
            "image_weight"
        ]
    )
    rows = []
    for _, setting in summary.iterrows():
        evidence_means, changes = [], []
        for i, case in schedule.iterrows():
            pool = candidates[candidates.paper_case_id == case.paper_case_id].copy()
            idx = pool.target_row.astype(int).to_numpy()
            clip = (
                fuse_embeddings(ti[idx], tt[idx], fusion)
                @ fuse_embeddings(image[i][None, :], text[i][None, :], fusion)[0]
            )
            pool["clip_score"] = clip
            clip_top = pool.item_ID.iloc[np.argmax(clip)]
            ranked = weighted_rerank(
                pool, float(setting.clip_weight), float(setting.evidence_weight), True
            )
            evidence_means.append(float(ranked.evidence_score.head(10).mean()))
            changes.append(clip_top != ranked.item_ID.iloc[0])
        rows.append(
            dict(setting)
            | {
                "recommendation_change_rate": float(np.mean(changes)),
                "mean_candidate_evidence_score_at_10": float(np.mean(evidence_means)),
                "mean_scoring_rule_count_at_10": float(SCORING_RULE_COUNT),
                "rule_backed_coverage": np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result["is_fused_clip_baseline"] = result.evidence_weight.eq(0.0)
    result["is_selected_075_025"] = result.clip_weight.eq(0.75) & result.evidence_weight.eq(0.25)
    values = result[
        ["hit_rate_at_10", "ndcg_at_10", "reciprocal_rank", "mean_candidate_evidence_score_at_10"]
    ].to_numpy()
    result["pareto_efficient"] = [
        not any(np.all(other >= point) and np.any(other > point) for other in values)
        for point in values
    ]
    return result


def write_report(aggregate: pd.DataFrame, boot: pd.DataFrame, thresholds: dict[str, float]) -> None:
    expanded = aggregate[aggregate.cohort == "expanded_3000"].set_index("method")
    r, f = expanded.loc["evidence_reranked"], expanded.loc["clip_fused"]
    h = aggregate[aggregate.cohort == "historical_300"].set_index("method").loc["evidence_reranked"]
    n = aggregate[aggregate.cohort == "new_2700"].set_index("method").loc["evidence_reranked"]
    lines = [
        "# Evidence participation analysis",
        "",
        "This analysis uses frozen candidate evidence scores, cached embeddings, the frozen 3,000-case schedule, and the validation-selected 0.75 CLIP / 0.25 evidence reranker. It did not call an LLM, Ollama, judge, or external API, and did not modify frozen ranking outputs.",
        "",
        "## Definitions and threshold status",
        "",
        "No frozen binary applicability/support threshold exists. `CandidateEvidenceScorer.score` selects the top five rules within the target category and combines their continuous similarities; it does not determine whether a rule supports a candidate. Thus **rule-backed Coverage@K is not confirmed and is reported as N/A**, not as the presence of a retrieved rule. The scorer-selected rule count is a structural count of five score-contributing rules, not an entailment/applicability count.",
        "",
        "Validation-only, outcome-blind score sensitivity thresholds (not rule-backed coverage): "
        + ", ".join(f"{k}={v:.6f}" for k, v in thresholds.items())
        + ".",
        "",
        "A retrieved rule is merely returned by retrieval; a query-relevant rule concerns the query; a candidate-applicable rule would need a binary applicability rule (absent); a score-contributing rule is one of the frozen top-five terms in the continuous scorer. Higher evidence score indicates stronger support under that frozen system, not objective recommendation correctness.",
        "",
        "## Cohort results",
        "",
        f"Historical 300: top-1 changed in {int(h.top1_changed_count)}/300 ({h.top1_changed_rate:.1%}); mean evidence score@10 was {h.mean_evidence_score_at_10:.4f}. New 2,700: {int(n.top1_changed_count)}/2,700 ({n.top1_changed_rate:.1%}); score@10 {n.mean_evidence_score_at_10:.4f}. Expanded 3,000: {int(r.top1_changed_count):,}/3,000 ({r.top1_changed_rate:.1%}); score@10 {r.mean_evidence_score_at_10:.4f}.",
        "",
        f"For 3,000 cases, reranking increased mean evidence score from {f.mean_evidence_score_at_1:.4f} to {r.mean_evidence_score_at_1:.4f} at 1, {f.mean_evidence_score_at_5:.4f} to {r.mean_evidence_score_at_5:.4f} at 5, and {f.mean_evidence_score_at_10:.4f} to {r.mean_evidence_score_at_10:.4f} at 10. Among changed top-1 results, {r.promoted_higher_score_rate_among_changed:.1%} promoted a higher-score candidate (mean difference {r.mean_top1_evidence_delta_among_changed:.4f}); top-five/top-ten overlaps were {r.mean_top5_overlap:.1%}/{r.mean_top10_overlap:.1%}, with mean absolute rank shift {r.mean_abs_rank_shift:.2f}.",
        "",
        "## Accuracy–evidence interpretation",
        "",
        "The paired cluster bootstrap finds positive continuous evidence-score shifts (see CSV) while the frozen expanded recommendation report found HR@10 close but inconclusive and NDCG@10/MRR significantly lower for reranking. This supports an accuracy–evidence participation trade-off under the frozen scorer, not an accuracy improvement or rule-entailment claim.",
        "",
        "## Reproducibility and limitations",
        "",
        "`scripts/analyze_evidence_participation.py` reconstructs only deterministic ranks from cached embeddings and saved evidence scores. Candidate-level rule identities/applicability labels were not saved in the frozen expanded output, so no candidate-applicable supporting-rule count or binary coverage can be recovered without a new retrieval pass; such a pass was intentionally not performed. The requested expanded handoff file was absent at analysis start, so it was not altered.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [
        ROOT / "ranking_results.csv",
        ROOT / "candidate_sets.csv",
        ROOT / "expanded_schedule.csv",
    ]
    before = {str(path): sha256(path) for path in protected}
    thresholds = thresholds_from_validation()
    per_case = reconstruct_cases()
    aggregate, category = summarize(per_case, thresholds)
    boot = bootstrap(per_case)
    sweep = validation_sweep()
    per_case.to_csv(OUT / "per_case_evidence_participation.csv", index=False)
    aggregate.to_csv(OUT / "aggregate_evidence_participation.csv", index=False)
    category.to_csv(OUT / "category_evidence_participation.csv", index=False)
    boot.to_csv(OUT / "paired_outfit_cluster_bootstrap.csv", index=False)
    sweep.to_csv(OUT / "validation_weight_sweep_evidence_participation.csv", index=False)
    distribution = (
        per_case[per_case.method == "evidence_reranked"]
        .groupby("target_category")
        .size()
        .rename("recommended_items")
        .reset_index()
    )
    distribution["scoring_rule_count"] = SCORING_RULE_COUNT
    distribution.to_csv(OUT / "scoring_rule_count_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(
        sweep.ndcg_at_10,
        sweep.mean_candidate_evidence_score_at_10,
        color="#4c78a8",
        label="validation-tested",
    )
    for _, row in sweep.iterrows():
        label = f"{row.clip_weight:.2f}/{row.evidence_weight:.2f}"
        ax.annotate(
            label,
            (row.ndcg_at_10, row.mean_candidate_evidence_score_at_10),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    baseline = sweep[sweep.is_fused_clip_baseline]
    selected = sweep[sweep.is_selected_075_025]
    efficient = sweep[sweep.pareto_efficient]
    ax.scatter(
        baseline.ndcg_at_10,
        baseline.mean_candidate_evidence_score_at_10,
        marker="s",
        s=65,
        color="#59a14f",
        label="fused CLIP (0 evidence)",
    )
    ax.scatter(
        selected.ndcg_at_10,
        selected.mean_candidate_evidence_score_at_10,
        marker="*",
        s=130,
        color="#e15759",
        label="selected 0.75/0.25",
    )
    ax.scatter(
        efficient.ndcg_at_10,
        efficient.mean_candidate_evidence_score_at_10,
        facecolors="none",
        edgecolors="black",
        s=95,
        label="Pareto-efficient",
    )
    ax.set(
        xlabel="Validation NDCG@10",
        ylabel="Mean candidate evidence score@10",
        title="Validation accuracy–evidence participation trade-off (frozen sweep)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "validation_pareto_accuracy_evidence.png", dpi=300)
    plt.close(fig)
    write_report(aggregate, boot, thresholds)
    after = {str(path): sha256(path) for path in protected}
    if before != after:
        raise RuntimeError("Frozen expanded recommendation outputs changed.")
    manifest = {
        "seed": SEED,
        "replicates": REPLICATES,
        "threshold_status": "no frozen binary threshold; validation score quantile sensitivity only",
        "thresholds": thresholds,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "outputs": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()},
        "report_hash": sha256(REPORT),
        "llm_or_external_api_called": False,
    }
    (OUT / "completion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
