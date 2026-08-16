from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from derive_stage8_study_metrics import combine_records, locate_output, read_json, read_jsonl
from scipy import stats

from evidence_fashion.manifest import git_commit, sha256_file, utc_timestamp, write_json

ROOT = Path(__file__).parents[1]
STAGE7_MANIFEST = ROOT / "artifacts/manifests/stage7_explanation_generation_manifest.json"
STAGE8_MANIFEST = ROOT / "artifacts/manifests/stage8_assessment_manifest.json"
STUDY_TABLE = ROOT / "artifacts/tables/table_stage8_study_specific_metrics.csv"
TABLE_DIR = ROOT / "artifacts/tables"
FIGURE_DIR = ROOT / "artifacts/figures"
EXAMPLE_PATH = ROOT / "artifacts/examples/stage8_deterministic_examples.md"
MANIFEST_PATH = ROOT / "artifacts/manifests/stage8_publication_analysis_manifest.json"

JUDGE_DIMENSIONS = (
    "input_consistency",
    "general_quality",
    "clarity",
    "specificity",
    "hallucination_control",
    "evidence_use_correctness",
)
BOOTSTRAP_REPLICATES = 2_000
SEED = 42


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values))


def rate(values: Sequence[bool]) -> float | None:
    return mean(values) if values else None


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float, float, np.ndarray]:
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["case_id"])].append(row)
    cluster_ids = sorted(clusters)
    observed = statistic(rows)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sample = [row for cluster_id in sampled for row in clusters[str(cluster_id)]]
        draws[index] = statistic(sample)
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, float(low), float(high), draws


def bootstrap_pvalue(draws: np.ndarray) -> float:
    return float(2 * min((np.sum(draws <= 0) + 1) / (len(draws) + 1),
                         (np.sum(draws >= 0) + 1) / (len(draws) + 1)))


def explanation_metrics(explanations: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for explanation in explanations:
        substantive = [c for c in explanation["claims"] if c["claim_role"] == "substantive"]
        attributes = [c for c in explanation["claims"] if c["attribute_bucket"] == "item_attribute"]
        unsupported = sum(bool(c["uiar_unsupported"]) for c in attributes)
        output[(explanation["case_id"], explanation["generator"], explanation["condition"])] = {
            "case_id": explanation["case_id"],
            "generator": explanation["generator"],
            "category": explanation["category"],
            "condition": explanation["condition"],
            "dta": rate([bool(c["dta_entailed"]) for c in substantive]),
            "uiar": rate([bool(c["uiar_unsupported"]) for c in attributes]),
            "attribute_density": unsupported / explanation["word_count"] * 100,
            "visible_support": rate([c["visible_status"] == "supported" for c in substantive]),
            "word_count": explanation["word_count"],
            "substantive_claims": len(substantive),
            "attribute_claims": len(attributes),
            "output_text": explanation["output_text"],
            "context_a": explanation["context_a"],
            "trace_b": explanation["trace_b"],
            "claims": explanation["claims"],
            "status": explanation["status"],
        }
    return output


def merge_judgments(
    metrics: dict[tuple[str, str, str], dict[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for judgment in judgments:
        for condition in ("no_rag", "rule_rag"):
            key = (judgment["case_id"], judgment["generator"], condition)
            rows.append({**metrics[key], **judgment["judgments"][condition]})
    return rows


def paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[(row["case_id"], row["generator"])][row["condition"]] = row
    paired = []
    for (case_id, generator), conditions in sorted(by_pair.items()):
        if set(conditions) != {"no_rag", "rule_rag"}:
            continue
        paired.append({
            "case_id": case_id,
            "generator": generator,
            "category": conditions["no_rag"]["category"],
            "no_rag": conditions["no_rag"],
            "rule_rag": conditions["rule_rag"],
        })
    return paired


def paired_judge_statistics(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, list[Mapping[str, Any]]]] = [("overall", "all", list(pairs))]
    scopes += [("generator", value, [p for p in pairs if p["generator"] == value])
               for value in sorted({p["generator"] for p in pairs})]
    scopes += [("category", value, [p for p in pairs if p["category"] == value])
               for value in sorted({p["category"] for p in pairs})]
    for scope, value, subset in scopes:
        scope_rows = []
        raw_p = []
        for dimension in JUDGE_DIMENSIONS:
            stat = lambda sample, d=dimension: mean(
                [p["rule_rag"][d] - p["no_rag"][d] for p in sample]
            )
            offset = int(hashlib.sha256(f"judge|{scope}|{value}|{dimension}".encode()).hexdigest()[:8], 16)
            difference, low, high, draws = cluster_bootstrap(subset, stat, seed=SEED + offset)
            differences = np.array([p["rule_rag"][dimension] - p["no_rag"][dimension] for p in subset])
            dz = difference / float(np.std(differences, ddof=1)) if np.std(differences, ddof=1) else math.inf
            p_value = float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
            raw_p.append(p_value)
            scope_rows.append({
                "scope": scope,
                "scope_value": value,
                "dimension": dimension,
                "paired_explanations": len(subset),
                "case_clusters": len({p["case_id"] for p in subset}),
                "no_rag_mean": mean([p["no_rag"][dimension] for p in subset]),
                "rule_rag_mean": mean([p["rule_rag"][dimension] for p in subset]),
                "paired_difference": difference,
                "ci_lower": low,
                "ci_upper": high,
                "p_value_raw": p_value,
                "p_value_holm": "",
                "paired_effect_dz": dz,
            })
        for row, adjusted in zip(scope_rows, holm_adjust(raw_p), strict=True):
            row["p_value_holm"] = adjusted
        rows.extend(scope_rows)
    return rows


def spearman_rows(rows: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = ("dta", "uiar", "attribute_density", "visible_support", "word_count")
    for condition in ("no_rag", "rule_rag"):
        subset = [row for row in rows if row["condition"] == condition]
        for metric in metrics:
            eligible = [r for r in subset if r[metric] is not None]
            for dimension in JUDGE_DIMENSIONS:
                def corr(sample: Sequence[Mapping[str, Any]], m=metric, d=dimension) -> float:
                    return float(stats.spearmanr([r[m] for r in sample], [r[d] for r in sample]).statistic)
                offset = int(hashlib.sha256(f"association|{condition}|{metric}|{dimension}".encode()).hexdigest()[:8], 16)
                rho, low, high, draws = cluster_bootstrap(eligible, corr, seed=SEED + offset, replicates=1_000)
                asymptotic_p = float(stats.spearmanr([r[metric] for r in eligible], [r[dimension] for r in eligible]).pvalue)
                output.append({
                    "analysis": "within_condition",
                    "condition": condition,
                    "metric": metric,
                    "judge_dimension": dimension,
                    "explanations": len(eligible),
                    "case_clusters": len({r["case_id"] for r in eligible}),
                    "spearman_rho": rho,
                    "ci_lower": low,
                    "ci_upper": high,
                    "p_value_asymptotic": asymptotic_p,
                })
    for metric in ("dta", "uiar", "attribute_density", "word_count"):
        eligible = [p for p in pairs if p["no_rag"][metric] is not None and p["rule_rag"][metric] is not None]
        for dimension in JUDGE_DIMENSIONS:
            x = [p["rule_rag"][metric] - p["no_rag"][metric] for p in eligible]
            y = [p["rule_rag"][dimension] - p["no_rag"][dimension] for p in eligible]
            result = stats.spearmanr(x, y)
            output.append({
                "analysis": "paired_change",
                "condition": "rule_rag_minus_no_rag",
                "metric": metric,
                "judge_dimension": dimension,
                "explanations": len(eligible),
                "case_clusters": len({p["case_id"] for p in eligible}),
                "spearman_rho": float(result.statistic),
                "ci_lower": "",
                "ci_upper": "",
                "p_value_asymptotic": float(result.pvalue),
            })
    return output


def metric_difference(pair: Mapping[str, Any], metric: str) -> float | None:
    no_value = pair["no_rag"][metric]
    rule_value = pair["rule_rag"][metric]
    return None if no_value is None or rule_value is None else rule_value - no_value


def heterogeneity_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric in ("dta", "uiar"):
        eligible = [p for p in pairs if metric_difference(p, metric) is not None]
        for grouping in ("generator", "category"):
            levels = sorted({p[grouping] for p in eligible})
            level_rows = []
            for level in levels:
                subset = [p for p in eligible if p[grouping] == level]
                stat = lambda sample, m=metric: mean([metric_difference(p, m) for p in sample])
                offset = int(hashlib.sha256(f"heterogeneity|{metric}|{grouping}|{level}".encode()).hexdigest()[:8], 16)
                effect, low, high, draws = cluster_bootstrap(subset, stat, seed=SEED + offset)
                level_rows.append({
                    "metric": metric,
                    "grouping": grouping,
                    "level": level,
                    "paired_explanations": len(subset),
                    "case_clusters": len({p["case_id"] for p in subset}),
                    "paired_difference": effect,
                    "ci_lower": low,
                    "ci_upper": high,
                    "omnibus_wald_chi2": "",
                    "omnibus_df": "",
                    "omnibus_p_value": "",
                })
            # Estimate the joint effect covariance by resampling complete case clusters.
            clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for pair in eligible:
                clusters[pair["case_id"]].append(pair)
            cluster_ids = sorted(clusters)
            rng = np.random.default_rng(SEED + int(hashlib.sha256(
                f"heterogeneity-joint|{metric}|{grouping}".encode()
            ).hexdigest()[:8], 16))
            joint_draws = np.empty((BOOTSTRAP_REPLICATES, len(levels)), dtype=float)
            for draw_index in range(BOOTSTRAP_REPLICATES):
                sampled_ids = rng.choice(cluster_ids, len(cluster_ids), replace=True)
                sample = [p for cluster_id in sampled_ids for p in clusters[str(cluster_id)]]
                for level_index, level in enumerate(levels):
                    values = [metric_difference(p, metric) for p in sample if p[grouping] == level]
                    joint_draws[draw_index, level_index] = mean(values)
            effects = np.array([r["paired_difference"] for r in level_rows], dtype=float)
            covariance = np.cov(joint_draws, rowvar=False)
            contrast = np.zeros((len(levels) - 1, len(levels)))
            for index in range(len(levels) - 1):
                contrast[index, index] = 1
                contrast[index, -1] = -1
            contrast_effect = contrast @ effects
            contrast_covariance = contrast @ covariance @ contrast.T
            chi2 = float(contrast_effect.T @ np.linalg.pinv(contrast_covariance) @ contrast_effect)
            df = len(levels) - 1
            p_value = float(stats.chi2.sf(chi2, df))
            for row in level_rows:
                row.update({"omnibus_wald_chi2": chi2, "omnibus_df": df, "omnibus_p_value": p_value})
            output.extend(level_rows)
    return output


def leave_one_generator_out(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    generators = sorted({p["generator"] for p in pairs})
    for metric in ("dta", "uiar"):
        for omitted in generators:
            subset = [p for p in pairs if p["generator"] != omitted and metric_difference(p, metric) is not None]
            stat = lambda sample, m=metric: mean([metric_difference(p, m) for p in sample])
            offset = int(hashlib.sha256(f"loo|{metric}|{omitted}".encode()).hexdigest()[:8], 16)
            effect, low, high, draws = cluster_bootstrap(subset, stat, seed=SEED + offset)
            output.append({
                "metric": metric,
                "omitted_generator": omitted,
                "included_generators": "; ".join(g for g in generators if g != omitted),
                "paired_explanations": len(subset),
                "case_clusters": len({p["case_id"] for p in subset}),
                "paired_difference": effect,
                "ci_lower": low,
                "ci_upper": high,
                "p_value": bootstrap_pvalue(draws),
            })
    return output


def select_examples(pairs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    selected = []
    for category in sorted({p["category"] for p in pairs}):
        candidates = [p for p in pairs if p["category"] == category and p["no_rag"]["status"] != "not_applicable"]
        dta = np.array([metric_difference(p, "dta") for p in candidates], dtype=float)
        density = np.array([metric_difference(p, "attribute_density") for p in candidates], dtype=float)
        medians = (float(np.median(dta)), float(np.median(density)))
        scored = []
        for p, dta_value, density_value in zip(candidates, dta, density, strict=True):
            distance = abs(dta_value - medians[0]) + abs(density_value - medians[1])
            tie = hashlib.sha256(f"{category}|{p['case_id']}|{p['generator']}".encode()).hexdigest()
            scored.append((distance, tie, p))
        selected.append(min(scored, key=lambda item: (item[0], item[1]))[2])
    return selected


def write_examples(path: Path, examples: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Deterministically selected Stage 8 explanation pairs",
        "",
        "One pair per broad category is selected without manual outcome screening. Within each category, "
        "the selected pair is closest to the category medians of the paired DTA change and unsupported-attribute-density change; SHA-256 order breaks ties.",
        "",
    ]
    for index, pair in enumerate(examples, start=1):
        no = pair["no_rag"]
        rule = pair["rule_rag"]
        context = no["context_a"]
        lines.extend([
            f"## Example {index}: {pair['category']} — {pair['generator']}",
            "",
            f"Case ID: `{pair['case_id']}`",
            "",
            f"Common context A: request ‘{context['user_request']}’; query ‘{context['query_item_text']}’; locked recommendation ‘{context['locked_item_text']}’.",
            "",
            "Exact trace B: " + "; ".join(f"{r['rule_id']} — {r['rule_text']}" for r in rule["trace_b"]["rules"]),
            "",
            f"No-RAG metrics: DTA={no['dta']:.3f}; UIAR={no['uiar']:.3f}" if no["uiar"] is not None else f"No-RAG metrics: DTA={no['dta']:.3f}; UIAR=N/A",
            "",
            "### No-RAG explanation",
            "",
            no["output_text"],
            "",
            f"Rule-RAG metrics: DTA={rule['dta']:.3f}; UIAR={rule['uiar']:.3f}" if rule["uiar"] is not None else f"Rule-RAG metrics: DTA={rule['dta']:.3f}; UIAR=N/A",
            "",
            "### Rule-RAG explanation",
            "",
            rule["output_text"],
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [FIGURE_DIR / f"{stem}.png", FIGURE_DIR / f"{stem}.svg"]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def make_figures(judge_rows: Sequence[Mapping[str, Any]], association_rows: Sequence[Mapping[str, Any]], heterogeneity: Sequence[Mapping[str, Any]]) -> list[Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    outputs: list[Path] = []
    study = list(csv.DictReader(STUDY_TABLE.open(encoding="utf-8")))
    primary = [r for r in study if r["analysis_subset"] == "full_corpus" and r["scope"] == "overall" and r["aggregation"] in {"micro_claim", "macro_explanation"} and r["condition"] in {"no_rag", "rule_rag"}]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, metric, title in zip(axes, ("dta_rate", "uiar_rate"), ("Decision-trace alignment", "Unsupported item-attribute rate"), strict=True):
        x = np.arange(2); width = 0.34
        for offset, condition, color in [(-width/2, "no_rag", "#7A7A7A"), (width/2, "rule_rag", "#1F6E8C")]:
            vals = [float(next(r[metric] for r in primary if r["aggregation"] == agg and r["condition"] == condition)) for agg in ("micro_claim", "macro_explanation")]
            ax.bar(x + offset, vals, width, label=condition.replace("_", "-").title(), color=color)
            for xi, value in zip(x + offset, vals, strict=True):
                ax.text(xi, value + .018, f"{value:.1%}", ha="center", fontsize=8)
        ax.set_title(title); ax.set_xticks(x, ["Micro", "Macro"]); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Rate"); axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Stage 8 primary explanation metrics")
    outputs += save_figure(fig, "fig_10_stage8_primary_metrics")

    overall = [r for r in judge_rows if r["scope"] == "overall"]
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    y = np.arange(len(overall)); effects = np.array([r["paired_difference"] for r in overall])
    low = effects - np.array([r["ci_lower"] for r in overall]); high = np.array([r["ci_upper"] for r in overall]) - effects
    ax.errorbar(effects, y, xerr=[low, high], fmt="o", color="#1F6E8C", capsize=3)
    ax.axvline(0, color="black", lw=.8); ax.set_yticks(y, [r["dimension"].replace("_", " ").title() for r in overall]); ax.invert_yaxis()
    ax.set_xlabel("Paired score difference (Rule-RAG − No-RAG, 1–5 scale)"); ax.grid(axis="x", alpha=.2)
    outputs += save_figure(fig, "fig_11_stage8_judge_paired_effects")

    assoc = [r for r in association_rows if r["analysis"] == "within_condition" and r["metric"] in {"dta", "uiar"}]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.8), sharex=True)
    for ax, condition, metric in zip(axes.flat, ("no_rag", "no_rag", "rule_rag", "rule_rag"), ("dta", "uiar", "dta", "uiar"), strict=True):
        subset = [r for r in assoc if r["condition"] == condition and r["metric"] == metric]
        vals = [r["spearman_rho"] for r in subset]
        ax.bar(np.arange(len(vals)), vals, color="#1F6E8C" if condition == "rule_rag" else "#7A7A7A")
        ax.axhline(0, color="black", lw=.7); ax.set_ylim(-.5, .5); ax.set_title(f"{condition.replace('_','-').title()}: {metric.upper()}")
        ax.set_xticks(np.arange(len(vals)), [r["judge_dimension"].replace("_", " ") for r in subset], rotation=35, ha="right", fontsize=7)
    fig.suptitle("Within-condition association with automated quality judgments (Spearman ρ)")
    fig.tight_layout()
    outputs += save_figure(fig, "fig_12_stage8_metric_judge_associations")

    subset = [r for r in heterogeneity if r["grouping"] == "generator"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric in zip(axes, ("dta", "uiar"), strict=True):
        rows = [r for r in subset if r["metric"] == metric]
        y = np.arange(len(rows)); effect = np.array([r["paired_difference"] for r in rows])
        ax.errorbar(effect, y, xerr=[effect-np.array([r["ci_lower"] for r in rows]), np.array([r["ci_upper"] for r in rows])-effect], fmt="o", color="#7B2CBF", capsize=3)
        ax.axvline(0, color="black", lw=.7); ax.set_yticks(y, [r["level"].split(":")[0] for r in rows]); ax.set_title(metric.upper()); ax.set_xlabel("Paired difference (Rule-RAG − No-RAG)"); ax.grid(axis="x", alpha=.2)
    fig.suptitle("Generator heterogeneity in primary explanation metrics")
    outputs += save_figure(fig, "fig_13_stage8_generator_heterogeneity")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    labels = ["No-RAG words", "Rule-RAG words", "No-RAG UAD", "Rule-RAG UAD"]
    values = [197.77, 60.55, .691, .324]
    ax2 = ax.twinx(); ax.bar([0,1], values[:2], color=["#7A7A7A", "#1F6E8C"]); ax2.bar([3,4], values[2:], color=["#B0B0B0", "#55A6C1"])
    ax.set_xticks([0,1,3,4], labels, rotation=15); ax.set_ylabel("Mean explanation words"); ax2.set_ylabel("Unsupported attributes per 100 words"); ax.set_title("Length confound and verbosity-normalised unsupported-attribute density")
    outputs += save_figure(fig, "fig_14_stage8_length_and_density")
    return outputs


def main() -> None:
    stage7 = read_json(STAGE7_MANIFEST)
    stage8 = read_json(STAGE8_MANIFEST)
    generation_path = locate_output(stage7, "generations.jsonl")
    packet_path = locate_output(stage7, "case_evidence_packets.jsonl")
    extraction_path = locate_output(stage8, "extractions.jsonl")
    verification_path = locate_output(stage8, "verifications.jsonl")
    judgment_path = locate_output(stage8, "judgments.jsonl")
    inputs = [STAGE7_MANIFEST, STAGE8_MANIFEST, STUDY_TABLE, generation_path, packet_path,
              extraction_path, verification_path, judgment_path]
    explanations = combine_records(read_jsonl(generation_path), read_jsonl(packet_path),
                                   read_jsonl(extraction_path), read_jsonl(verification_path))
    metrics = explanation_metrics(explanations)
    merged = merge_judgments(metrics, read_jsonl(judgment_path))
    pairs = paired_rows(merged)
    if len(merged) != 3_000 or len(pairs) != 1_500:
        raise ValueError("Expected 3,000 explanations and 1,500 matched pairs.")

    judge_rows = paired_judge_statistics(pairs)
    association_rows = spearman_rows(merged, pairs)
    heterogeneity = heterogeneity_rows(pairs)
    loo = leave_one_generator_out(pairs)
    output_paths = [
        TABLE_DIR / "table_stage8_judge_paired_statistics.csv",
        TABLE_DIR / "table_stage8_metric_associations.csv",
        TABLE_DIR / "table_stage8_heterogeneity.csv",
        TABLE_DIR / "table_stage8_leave_one_generator_out.csv",
    ]
    write_csv(output_paths[0], judge_rows)
    write_csv(output_paths[1], association_rows)
    write_csv(output_paths[2], heterogeneity)
    write_csv(output_paths[3], loo)
    examples = select_examples(pairs)
    write_examples(EXAMPLE_PATH, examples)
    output_paths.append(EXAMPLE_PATH)
    output_paths += make_figures(judge_rows, association_rows, heterogeneity)

    manifest = {
        "stage": "stage8_publication_analysis",
        "status": "complete",
        "generated_at_utc": utc_timestamp(),
        "git_commit_at_generation": git_commit(),
        "method": {
            "model_calls": 0,
            "paired_unit": "case_id × generator",
            "bootstrap_cluster": "case_id",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": 0.95,
            "seed": SEED,
            "judge_multiplicity": "Holm adjustment across six dimensions within each scope",
            "association": "within-condition and paired-change Spearman rank correlation",
            "qualitative_selection": "category-median DTA/density distance with SHA-256 tie-breaking",
            "flops": "not estimated; no kernel-level operation counters were captured and no new inference was run",
        },
        "counts": {"explanations": len(merged), "matched_pairs": len(pairs), "examples": len(examples)},
        "input_artifact_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in inputs},
        "output_artifact_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in output_paths},
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps({"manifest": str(MANIFEST_PATH), "outputs": len(output_paths)}, indent=2))


if __name__ == "__main__":
    main()
