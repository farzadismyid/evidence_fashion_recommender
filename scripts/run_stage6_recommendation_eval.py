"""Run the frozen Stage 6 confirmatory recommendation evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from run_recommendation_eval import _active_items, _item_images

from evidence_fashion.data import (
    attach_candidate_pools,
    build_evaluation_cases,
    load_pinned_split,
    write_jsonl,
)
from evidence_fashion.evaluation.recommendation import aggregate_recommendation_metrics
from evidence_fashion.evaluation.statistics import (
    clustered_bootstrap_mean,
    holm_adjust,
    two_sided_bootstrap_pvalue,
)
from evidence_fashion.kb_audit import load_audited_rules
from evidence_fashion.manifest import (
    configuration_hash,
    environment_summary,
    git_commit,
    load_resolved_configuration,
    sha256_file,
    utc_timestamp,
    write_json,
    write_new_json,
)
from evidence_fashion.reranking import ranking_metrics, rerank_candidates
from evidence_fashion.retrieval import cosine_scores, fuse_clip_embeddings
from evidence_fashion.rule_retrieval import RuleRetriever, candidate_rule_representation

METRICS = [
    "hr_at_1",
    "hr_at_5",
    "hr_at_10",
    "ndcg_at_1",
    "ndcg_at_5",
    "ndcg_at_10",
    "mrr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _rank(item_ids: list[str], relevance: list[bool], scores: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"item_id": item_ids, "is_positive": relevance, "score": scores}
    ).sort_values(["score", "item_id"], ascending=[False, True], kind="stable")


def _contrasts(case_metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    settings = config["statistics"]
    stage6 = config["stage6"]
    reference = stage6["reference_method"]
    index = ["case_id", "query_outfit_id"]
    rows = []
    for comparison_index, comparison in enumerate(stage6["contrast_methods"]):
        left = case_metrics[case_metrics["method"].eq(reference)].set_index(index)
        right = case_metrics[case_metrics["method"].eq(comparison)].set_index(index)
        joined = left.join(right, lsuffix="_reference", rsuffix="_comparison", how="inner")
        for metric_index, metric in enumerate(METRICS):
            differences = joined[f"{metric}_reference"] - joined[f"{metric}_comparison"]
            estimate, lower, upper, estimates = clustered_bootstrap_mean(
                differences,
                joined.index.get_level_values("query_outfit_id"),
                replicates=int(settings["bootstrap_replicates"]),
                confidence_level=float(settings["confidence_level"]),
                seed=int(config["project"]["random_seed"]) + 100 * comparison_index + metric_index,
            )
            rows.append(
                {
                    "reference_method": reference,
                    "comparison_method": comparison,
                    "metric": metric,
                    "mean_paired_difference": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "p_value": two_sided_bootstrap_pvalue(estimates),
                    "cases": len(joined),
                    "bootstrap_unit": settings["bootstrap_unit"],
                }
            )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = holm_adjust(result["p_value"])
    result["reject_at_0_05"] = result["holm_adjusted_p_value"].lt(0.05)
    return result


def _svg_document(title: str, boxes: list[tuple[int, int, int, int, str]]) -> str:
    width, height = 1200, 680
    elements = []
    for x, y, w, h, label in boxes:
        safe = label.replace("&", "&amp;")
        elements.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
            'fill="#E8F1F8" stroke="#0072B2" stroke-width="2"/>'
            f'<text x="{x + w / 2}" y="{y + h / 2 + 5}" text-anchor="middle" '
            f'font-family="Arial" font-size="16">{safe}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        f'<text x="600" y="38" text-anchor="middle" font-family="Arial" font-size="24" '
        f'font-weight="bold">{title}</text>{"".join(elements)}</svg>\n'
    )


def _write_method_figures() -> list[Path]:
    figures = Path("artifacts/figures")
    specifications = {
        "fig_01_system_architecture.svg": (
            "Evidence-constrained multimodal recommendation pipeline",
            [
                (40, 90, 170, 65, "Image + request"),
                (260, 90, 180, 65, "CLIP encoders"),
                (490, 90, 180, 65, "0.40/0.60 fusion"),
                (720, 90, 190, 65, "Rule retrieval"),
                (960, 90, 190, 65, "0.75/0.25 rerank"),
                (490, 230, 180, 65, "Locked item + B"),
                (260, 370, 180, 65, "No-RAG: A"),
                (720, 370, 180, 65, "Rule-RAG: A + B"),
                (490, 510, 180, 65, "Claims + judging"),
            ],
        ),
        "fig_02_evidence_ablation.svg": (
            "Explanation evidence ablation",
            [
                (120, 110, 300, 90, "Common context A"),
                (120, 300, 300, 90, "No-RAG = A"),
                (780, 110, 300, 90, "Exact five-rule trace B"),
                (780, 300, 300, 90, "Rule-RAG = A + B"),
                (450, 500, 300, 90, "Paired evaluation"),
            ],
        ),
        "fig_03_evidence_trace.svg": (
            "Exact scoring-trace reuse",
            [
                (70, 100, 210, 75, "Five retrieved rules"),
                (350, 100, 210, 75, "Weighted scores"),
                (630, 100, 210, 75, "Evidence score"),
                (910, 100, 210, 75, "Reranked item"),
                (350, 360, 210, 75, "Stored trace B"),
                (630, 360, 210, 75, "Rule-RAG prompt"),
            ],
        ),
        "fig_04_dataset_pipeline.svg": (
            "Dataset processing and controlled evaluation",
            [
                (50, 100, 190, 75, "94,096 raw items"),
                (290, 100, 190, 75, "Exact taxonomy"),
                (530, 100, 190, 75, "69,725 kept items"),
                (770, 100, 190, 75, "Leakage repair"),
                (1010, 100, 140, 75, "Outfit splits"),
                (290, 350, 190, 75, "Test cases"),
                (530, 350, 190, 75, "99 negatives"),
                (770, 350, 190, 75, "Known positives"),
            ],
        ),
    }
    paths = []
    for name, (title, boxes) in specifications.items():
        path = figures / name
        path.write_text(_svg_document(title, boxes), encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def _write_metric_figure(results: pd.DataFrame, path: Path) -> None:
    micro = results[
        results["aggregation"].eq("micro")
        & results["metric"].isin(["hr_at_10", "ndcg_at_10", "mrr"])
    ]
    methods = sorted(micro["method"].unique())
    colours = {"hr_at_10": "#0072B2", "ndcg_at_10": "#D55E00", "mrr": "#009E73"}
    width, height = 1000, 620
    plot_height = 430
    maximum = max(float(micro["ci_upper"].max()), 0.01)
    elements = []
    group_width = 800 / len(methods)
    for method_index, method in enumerate(methods):
        subset = micro[micro["method"].eq(method)]
        for metric_index, metric in enumerate(("hr_at_10", "ndcg_at_10", "mrr")):
            row = subset[subset["metric"].eq(metric)].iloc[0]
            x = 110 + method_index * group_width + metric_index * 30
            y = 500 - float(row["estimate"]) / maximum * plot_height
            low = 500 - float(row["ci_lower"]) / maximum * plot_height
            high = 500 - float(row["ci_upper"]) / maximum * plot_height
            elements.append(
                f'<line x1="{x}" y1="{low}" x2="{x}" y2="{high}" stroke="black"/>'
                f'<circle cx="{x}" cy="{y}" r="6" fill="{colours[metric]}"/>'
            )
        label = method.replace("_", " ")
        elements.append(
            f'<text x="{135 + method_index * group_width}" y="540" text-anchor="middle" '
            f'font-family="Arial" font-size="11">{label}</text>'
        )
    legend = " ".join(
        f'<tspan fill="{colour}">● {metric.upper()}</tspan>' for metric, colour in colours.items()
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        '<text x="500" y="35" text-anchor="middle" font-family="Arial" font-size="22" '
        'font-weight="bold">Stage 6 recommendation performance (95% clustered CI)</text>'
        f'<text x="500" y="585" text-anchor="middle" font-family="Arial" '
        f'font-size="14">{legend}</text>'
        f"{''.join(elements)}</svg>\n"
    )
    path.write_text(svg, encoding="utf-8", newline="\n")


def _write_series_figure(
    frame: pd.DataFrame,
    *,
    x_column: str,
    series_column: str,
    value_column: str,
    title: str,
    y_label: str,
    path: Path,
) -> None:
    """Write an accessible vector line chart from a tidy source table."""
    width, height = 1000, 620
    left, right, top, bottom = 90, 950, 70, 520
    x_values = sorted(frame[x_column].unique())
    y_max = max(float(frame[value_column].max()) * 1.1, 0.01)
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    elements = [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = bottom - (bottom - top) * tick / 5
        elements.append(
            f'<text x="{left - 10}" y="{y + 5}" text-anchor="end" '
            f'font-family="Arial" font-size="12">{value:.3f}</text>'
        )
    for index, x_value in enumerate(x_values):
        x = left + (right - left) * index / max(len(x_values) - 1, 1)
        elements.append(
            f'<text x="{x}" y="{bottom + 25}" text-anchor="middle" '
            f'font-family="Arial" font-size="12">{x_value}</text>'
        )
    for series_index, (series, group) in enumerate(frame.groupby(series_column, sort=True)):
        colour = palette[series_index % len(palette)]
        lookup = dict(zip(group[x_column], group[value_column], strict=True))
        points = []
        for index, x_value in enumerate(x_values):
            if x_value not in lookup:
                continue
            x = left + (right - left) * index / max(len(x_values) - 1, 1)
            y = bottom - float(lookup[x_value]) / y_max * (bottom - top)
            points.append(f"{x:.2f},{y:.2f}")
            elements.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{colour}"/>')
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colour}" '
            'stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{left + 150 * series_index}" y="585" font-family="Arial" '
            f'font-size="12" fill="{colour}">● {series}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        f'<text x="500" y="35" text-anchor="middle" font-family="Arial" font-size="22" '
        f'font-weight="bold">{title}</text>'
        f'<text x="25" y="300" transform="rotate(-90 25 300)" text-anchor="middle" '
        f'font-family="Arial" font-size="14">{y_label}</text>'
        f'<text x="520" y="560" text-anchor="middle" font-family="Arial" '
        f'font-size="14">{x_column.replace("_", " ")}</text>{"".join(elements)}</svg>\n'
    )
    path.write_text(svg, encoding="utf-8", newline="\n")


def _write_additional_publication_figures() -> list[Path]:
    figures = Path("artifacts/figures")
    specifications = []
    fusion_path = Path("artifacts/tables/table_stage4_fusion_search.csv")
    if fusion_path.exists():
        fusion = pd.read_csv(fusion_path).melt(
            id_vars="image_weight",
            value_vars=["hr_at_10", "ndcg_at_10", "mrr"],
            var_name="metric",
            value_name="estimate",
        )
        specifications.append(
            (
                fusion,
                "image_weight",
                "metric",
                "estimate",
                "Validation fusion sensitivity",
                "ranking metric",
                figures / "fig_07_fusion_sensitivity.svg",
            )
        )
    pool_path = Path("artifacts/tables/table_stage4_pool_sensitivity.csv")
    if pool_path.exists():
        pool = pd.read_csv(pool_path)
        specifications.append(
            (
                pool,
                "pool_target",
                "method",
                "ndcg_at_10",
                "Candidate-pool-size sensitivity",
                "NDCG@10",
                figures / "fig_08_pool_sensitivity.svg",
            )
        )
    prompt_path = Path("artifacts/tables/table_stage5_optimization.csv")
    if prompt_path.exists():
        prompt = pd.read_csv(prompt_path)
        prompt_plot = prompt.assign(
            mean_quality=prompt[["general_quality", "clarity", "specificity"]].mean(axis=1) / 5
        ).melt(
            id_vars="configuration_id",
            value_vars=["support_rate", "mean_quality"],
            var_name="metric",
            value_name="estimate",
        )
        specifications.append(
            (
                prompt_plot,
                "configuration_id",
                "metric",
                "estimate",
                "Explanation prompt ablation",
                "rate or normalized score",
                figures / "fig_09_prompt_ablation.svg",
            )
        )
    for frame, x, series, value, title, y_label, path in specifications:
        _write_series_figure(
            frame,
            x_column=x,
            series_column=series,
            value_column=value,
            title=title,
            y_label=y_label,
            path=path,
        )
    return [row[-1] for row in specifications]


def _image_panel(
    selected: list[dict[str, Any]], item_lookup: pd.DataFrame, raw_split, path: Path
) -> None:
    thumb = 150
    header = 60
    row_height = 2 * thumb + 65
    canvas = Image.new("RGB", (11 * thumb, header + len(selected) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 10), "Query | top five before reranking | top five after reranking", fill="black"
    )
    for row_index, record in enumerate(selected):
        y = header + row_index * row_height
        draw.text((10, y), f"{record['target_category']} · {record['case_id']}", fill="black")
        ids = [record["query_item_id"], *record["pre_top5"], *record["post_top5"]]
        for column, item_id in enumerate(ids):
            source_index = int(item_lookup.loc[item_id, "original_dataset_index"])
            image = raw_split[source_index]["image"].convert("RGB")
            image.thumbnail((thumb - 8, thumb - 8))
            x = column * thumb + (thumb - image.width) // 2
            canvas.paste(image, (x, y + 25))
            draw.text((column * thumb + 4, y + thumb + 28), str(item_id)[:18], fill="black")
    canvas.save(path, format="PNG", optimize=True)


def _registry_rows(config_digest: str, artifacts: dict[str, Path]) -> list[dict[str, str]]:
    specs = {
        "table_01_dataset_statistics": ("table", "Dataset and split statistics", "Dataset"),
        "table_02_recommendation_results": (
            "table",
            "Confirmatory recommendation results",
            "Results",
        ),
        "table_03_recommendation_contrasts": (
            "table",
            "Paired recommendation contrasts",
            "Results",
        ),
        "table_04_evidence_participation": (
            "table",
            "Evidence participation diagnostics",
            "Results",
        ),
        "table_05_stage6_rule_frequency": ("table", "Confirmatory rule-frequency audit", "Results"),
        "table_06_kb_summary": ("table", "Knowledge-base composition", "Methods"),
        "table_07_publication_readiness": (
            "table",
            "Publication-readiness checklist",
            "Results",
        ),
        "fig_01_system_architecture": ("figure", "End-to-end system architecture", "Methods"),
        "fig_02_evidence_ablation": ("figure", "A/B evidence ablation", "Methods"),
        "fig_03_evidence_trace": ("figure", "Exact evidence-trace reuse", "Methods"),
        "fig_04_dataset_pipeline": ("figure", "Dataset processing and splits", "Methods"),
        "fig_05_recommendation_metrics": (
            "figure",
            "Recommendation metrics with uncertainty",
            "Results",
        ),
        "fig_06_recommendation_examples": (
            "example",
            "Deterministic recommendation examples",
            "Results",
        ),
        "fig_07_fusion_sensitivity": (
            "figure",
            "Validation fusion sensitivity",
            "Results",
        ),
        "fig_08_pool_sensitivity": (
            "figure",
            "Candidate-pool-size sensitivity",
            "Results",
        ),
        "fig_09_prompt_ablation": (
            "figure",
            "Explanation prompt ablation",
            "Results",
        ),
    }
    rows = []
    for artifact_id, (kind, title, chapter) in specs.items():
        path = artifacts[artifact_id]
        rows.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": kind,
                "title": title,
                "research_question": "Stage 6 recommendation quality, uncertainty, or provenance",
                "source_data": ".runtime/stage6",
                "generation_function_or_script": "scripts/run_stage6_recommendation_eval.py",
                "configuration_hash": config_digest,
                "output_path": str(path),
                "caption": f"{title}; frozen 1,000-case test evaluation.",
                "intended_thesis_chapter": chapter,
                "intended_paper_section": chapter,
                "status": "final",
                "notes": "Researcher-selected approximately 100-candidate main protocol.",
            }
        )
    return rows


def _update_registry(config_digest: str, artifacts: dict[str, Path]) -> None:
    path = Path("artifacts/manifests/figure_table_registry.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        existing = list(csv.DictReader(handle))
    additions = _registry_rows(config_digest, artifacts)
    ids = {row["artifact_id"] for row in additions}
    rows = [row for row in existing if row["artifact_id"] not in ids] + additions
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    config_digest = configuration_hash(resolved)
    run_id = f"stage6-confirmatory-{config_digest[:12]}"
    runtime_root = args.runtime_root or Path(config["paths"]["runtime_root"])
    run_dir = runtime_root / "stage6" / run_id
    if args.dry_run:
        print(
            json.dumps(
                {"run_id": run_id, "cases": config["recommendation_evaluation"]["case_count"]},
                indent=2,
            )
        )
        return
    if run_dir.exists():
        raise FileExistsError(f"Immutable Stage 6 run already exists: {run_dir}")

    from evidence_fashion.retrieval import CLIPEmbedder, OllamaEmbedder

    items, items_path, items_hash = _active_items(config)
    cases = attach_candidate_pools(items, build_evaluation_cases(items, config), config)
    expected = int(config["recommendation_evaluation"]["case_count"])
    if len(cases) != expected:
        raise ValueError(f"Expected {expected} confirmatory cases; found {len(cases)}.")
    item_lookup = items.set_index("item_id", drop=False)
    candidate_ids = sorted({item for values in cases["candidate_item_ids"] for item in values})
    candidate_rows = item_lookup.loc[candidate_ids].reset_index(drop=True)
    query_rows = item_lookup.loc[cases["query_item_id"].tolist()].reset_index(drop=True)
    raw_split, dataset_fingerprint = load_pinned_split(config)
    clip = CLIPEmbedder(models["embedders"]["clip"])
    text_embedder = OllamaEmbedder(models["embedders"]["qwen3_embedding"])
    batch_size = int(config["stage4_validation"]["embedding_batch_size"])

    candidate_texts = (
        candidate_rows["category"].astype(str) + " | " + candidate_rows["text"].astype(str)
    ).tolist()
    candidate_clip_text = clip.encode_text(candidate_texts, batch_size=batch_size)
    candidate_clip_image = _item_images(candidate_rows, raw_split, clip, batch_size)
    candidate_qwen3_embedding = text_embedder.encode(candidate_texts, batch_size=batch_size)
    fusion = config["retrieval"]["fusion"]
    candidate_fused = fuse_clip_embeddings(
        candidate_clip_image,
        candidate_clip_text,
        image_weight=float(fusion["image_weight"]),
        text_weight=float(fusion["text_weight"]),
    )
    query_texts = [
        " | ".join(
            [
                f"Query category: {case['query_category']}",
                f"Query text: {case['query_text']}",
                f"User request: {case['user_request']}",
                f"Target category: {case['target_category']}",
            ]
        )
        for case in cases.to_dict("records")
    ]
    query_clip_text = clip.encode_text(query_texts, batch_size=batch_size)
    query_clip_image = _item_images(query_rows, raw_split, clip, batch_size)
    query_qwen3_embedding = text_embedder.encode(query_texts, batch_size=batch_size)
    query_fused = fuse_clip_embeddings(
        query_clip_image,
        query_clip_text,
        image_weight=float(fusion["image_weight"]),
        text_weight=float(fusion["text_weight"]),
    )

    kb = load_audited_rules(config)
    rule_embeddings = text_embedder.encode(
        kb["rule_text"].astype(str).tolist(), batch_size=batch_size
    )
    retriever = RuleRetriever(kb, rule_embeddings, config["rule_retrieval"])
    case_candidate_pairs = []
    representations = []
    for case in cases.to_dict("records"):
        for item_id, relevant in zip(
            case["candidate_item_ids"], case["candidate_relevance"], strict=True
        ):
            candidate = item_lookup.loc[item_id].to_dict()
            case_candidate_pairs.append((case, candidate, bool(relevant)))
            representations.append(candidate_rule_representation(case, candidate))
    representation_embeddings = text_embedder.encode(representations, batch_size=batch_size)
    traces = [
        retriever.retrieve_and_score(
            case=case,
            candidate=candidate,
            representation_embedding=representation_embeddings[index],
            top_k=int(config["stage4_validation"]["selected_rule_top_k"]),
        )
        for index, (case, candidate, _) in enumerate(case_candidate_pairs)
    ]

    candidate_index = {item_id: index for index, item_id in enumerate(candidate_ids)}
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    locked_cases: list[dict[str, Any]] = []
    ranking_records: list[dict[str, Any]] = []
    rule_packet_counter: Counter[str] = Counter()
    rule_slot_counter: Counter[str] = Counter()
    rule_text = dict(zip(kb["rule_id"], kb["rule_text"], strict=True))
    offset = 0
    for case_index, case in enumerate(cases.to_dict("records")):
        count = len(case["candidate_item_ids"])
        case_traces = traces[offset : offset + count]
        offset += count
        indices = [candidate_index[item] for item in case["candidate_item_ids"]]
        score_sets = {
            "qwen3_embedding_text": cosine_scores(
                query_qwen3_embedding[case_index], candidate_qwen3_embedding[indices]
            ),
            "clip_image": cosine_scores(
                query_clip_image[case_index], candidate_clip_image[indices]
            ),
            "clip_text": cosine_scores(query_clip_text[case_index], candidate_clip_text[indices]),
            "fused_clip_main_040_060": cosine_scores(
                query_fused[case_index], candidate_fused[indices]
            ),
        }
        rankings = {
            method: _rank(case["candidate_item_ids"], case["candidate_relevance"], scores)
            for method, scores in score_sets.items()
        }
        rerank_input = pd.DataFrame(
            {
                "item_id": case["candidate_item_ids"],
                "is_positive": case["candidate_relevance"],
                "clip_score": score_sets["fused_clip_main_040_060"],
                "evidence_score": [trace.evidence_score for trace in case_traces],
                "trace": case_traces,
            }
        )
        reranked = rerank_candidates(
            rerank_input,
            clip_weight=float(config["reranking"]["reference_clip_weight"]),
            evidence_weight=float(config["reranking"]["reference_evidence_weight"]),
        )
        rankings["evidence_rerank_main_075_025"] = reranked
        for method, ranking in rankings.items():
            metric_rows.append(
                {
                    "case_id": case["case_id"],
                    "query_outfit_id": case["query_outfit_id"],
                    "target_category": case["target_category"],
                    "actual_pool_size": count,
                    "method": method,
                    **ranking_metrics(ranking["is_positive"]),
                }
            )
        pre = rankings["fused_clip_main_040_060"]
        pre_ids = pre["item_id"].tolist()
        post_ids = reranked["item_id"].tolist()
        evidence_by_item = rerank_input.set_index("item_id")["evidence_score"]
        diagnostic = {
            "case_id": case["case_id"],
            "query_outfit_id": case["query_outfit_id"],
            "target_category": case["target_category"],
            "top1_changed": float(pre_ids[0] != post_ids[0]),
            "top5_changed": float(pre_ids[:5] != post_ids[:5]),
            "top5_set_overlap": len(set(pre_ids[:5]) & set(post_ids[:5])) / 5,
            "top10_set_overlap": len(set(pre_ids[:10]) & set(post_ids[:10])) / 10,
        }
        for cutoff in (1, 5, 10):
            diagnostic[f"evidence_gain_at_{cutoff}"] = float(
                evidence_by_item.loc[post_ids[:cutoff]].mean()
                - evidence_by_item.loc[pre_ids[:cutoff]].mean()
            )
        diagnostic["mean_absolute_rank_shift_top10_union"] = float(
            np.mean(
                [
                    abs(pre_ids.index(item) - post_ids.index(item))
                    for item in set(pre_ids[:10]) | set(post_ids[:10])
                ]
            )
        )
        diagnostic_rows.append(diagnostic)
        top = reranked.iloc[0]
        trace = top["trace"]
        locked_cases.append(
            {
                "case_id": case["case_id"],
                "query_item_id": case["query_item_id"],
                "query_outfit_id": case["query_outfit_id"],
                "query_item_minimal_name": str(case["query_text"] or case["query_category"]),
                "request": case["user_request"],
                "target_category": case["target_category"],
                "locked_candidate_id": str(top["item_id"]),
                "locked_candidate_minimal_name": str(
                    item_lookup.loc[top["item_id"], "text"]
                    or item_lookup.loc[top["item_id"], "category"]
                ),
                "pre_rerank_rank": int(top["pre_rerank_rank"]),
                "post_rerank_rank": int(top["post_rerank_rank"]),
                "clip_score": float(top["clip_score"]),
                "evidence_score": float(top["evidence_score"]),
                "final_score": float(top["final_score"]),
                "evidence_trace": trace.to_dict(),
            }
        )
        ranking_records.append(
            {
                "case_id": case["case_id"],
                "target_category": case["target_category"],
                "query_item_id": case["query_item_id"],
                "pre_top5": pre_ids[:5],
                "post_top5": post_ids[:5],
            }
        )
        for candidate_trace in case_traces:
            ids = {rule.rule_id for rule in candidate_trace.rules}
            rule_packet_counter.update(ids)
            rule_slot_counter.update(rule.rule_id for rule in candidate_trace.rules)

    case_metrics = pd.DataFrame(metric_rows)
    evidence_diagnostics = pd.DataFrame(diagnostic_rows)
    stats = config["statistics"]
    recommendation_results = aggregate_recommendation_metrics(
        case_metrics,
        metric_columns=METRICS,
        replicates=int(stats["bootstrap_replicates"]),
        confidence_level=float(stats["confidence_level"]),
        seed=int(config["project"]["random_seed"]),
    )
    contrasts = _contrasts(case_metrics, config)
    evidence_summary = (
        evidence_diagnostics.drop(columns=["case_id", "query_outfit_id", "target_category"])
        .mean()
        .rename_axis("metric")
        .reset_index(name="estimate")
    )
    frequency = pd.DataFrame(
        [
            {
                "rule_id": rule_id,
                "rule_text": rule_text[rule_id],
                "packet_count": rule_packet_counter[rule_id],
                "packet_prevalence": rule_packet_counter[rule_id] / len(traces),
                "slot_count": rule_slot_counter[rule_id],
                "slot_share": rule_slot_counter[rule_id] / (len(traces) * 5),
            }
            for rule_id in sorted(rule_slot_counter)
        ]
    )
    dataset_statistics = pd.DataFrame(
        [
            ("prepared_items", len(items)),
            ("prepared_outfits", items["outfit_id"].nunique()),
            ("confirmatory_cases", len(cases)),
            ("candidate_rows", int(cases["candidate_item_ids"].map(len).sum())),
            ("candidate_pool_min", int(cases["candidate_item_ids"].map(len).min())),
            ("candidate_pool_mean", float(cases["candidate_item_ids"].map(len).mean())),
            ("candidate_pool_max", int(cases["candidate_item_ids"].map(len).max())),
        ],
        columns=["measure", "value"],
    )
    kb_summary = (
        kb.groupby(["recommended_category", "source_reliability"], as_index=False)
        .size()
        .rename(columns={"size": "rule_count"})
    )
    publication_readiness = pd.DataFrame(
        [
            ("Major quantitative results have source tables", "complete_stage6", "tables 01-06"),
            ("Validation trends and sensitivities are preserved", "complete", "stage 4/5 tables"),
            (
                "Important validation trends have vector visualisations",
                "complete_stage6",
                "figures 07-09",
            ),
            ("Methodology diagrams are vector graphics", "complete_stage6", "figures 01-04"),
            ("Recommendation uncertainty is visualised", "complete_stage6", "figure 05"),
            ("Recommendation examples use original images", "complete_stage6", "figure 06"),
            (
                "Examples follow a fixed selection rule",
                "complete_stage6",
                "lowest case ID/category",
            ),
            (
                "Explanation conditions shown side by side",
                "pilot_only",
                "full experiment is Stage 7",
            ),
            ("Worked claim and judge examples", "pilot_only", "final examples require Stage 8"),
            ("Automated explanation evaluation", "pending_stage8", "requires Stages 7-8"),
            (
                "Artifact captions and provenance registry",
                "complete_stage6",
                "figure/table registry",
            ),
            ("Full paper/thesis release review", "pending_stage10", "after Stages 7-8"),
        ],
        columns=["requirement", "status", "evidence_or_next_step"],
    )

    run_dir.mkdir(parents=True)
    runtime_outputs = {
        "case_metrics": run_dir / "case_metrics.csv",
        "evidence_diagnostics": run_dir / "evidence_diagnostics.csv",
        "locked_cases": run_dir / "locked_cases.jsonl",
        "rankings": run_dir / "candidate_rankings.jsonl",
    }
    case_metrics.to_csv(runtime_outputs["case_metrics"], index=False)
    evidence_diagnostics.to_csv(runtime_outputs["evidence_diagnostics"], index=False)
    write_jsonl(runtime_outputs["locked_cases"], locked_cases)
    write_jsonl(runtime_outputs["rankings"], ranking_records)
    tracked = {
        "table_01_dataset_statistics": Path("artifacts/tables/table_01_dataset_statistics.csv"),
        "table_02_recommendation_results": Path(
            "artifacts/tables/table_02_recommendation_results.csv"
        ),
        "table_03_recommendation_contrasts": Path(
            "artifacts/tables/table_03_recommendation_contrasts.csv"
        ),
        "table_04_evidence_participation": Path(
            "artifacts/tables/table_04_evidence_participation.csv"
        ),
        "table_05_stage6_rule_frequency": Path(
            "artifacts/tables/table_05_stage6_rule_frequency.csv"
        ),
        "table_06_kb_summary": Path("artifacts/tables/table_06_kb_summary.csv"),
        "table_07_publication_readiness": Path(
            "artifacts/tables/table_07_publication_readiness.csv"
        ),
    }
    for frame, path in zip(
        (
            dataset_statistics,
            recommendation_results,
            contrasts,
            evidence_summary,
            frequency,
            kb_summary,
            publication_readiness,
        ),
        tracked.values(),
        strict=True,
    ):
        frame.to_csv(path, index=False)
    figure_paths = _write_method_figures()
    sensitivity_figure_paths = _write_additional_publication_figures()
    metric_figure = Path("artifacts/figures/fig_05_recommendation_metrics.svg")
    _write_metric_figure(recommendation_results, metric_figure)
    selected_examples = []
    for category in config["preprocessing"]["target_categories"]:
        selected_examples.extend(
            sorted(
                [row for row in ranking_records if row["target_category"] == category],
                key=lambda row: row["case_id"],
            )[: int(config["stage6"]["publication_example_cases_per_category"])]
        )
    example_path = Path("artifacts/examples/fig_06_recommendation_examples.png")
    _image_panel(selected_examples, item_lookup, raw_split, example_path)
    artifacts = {
        **tracked,
        **{path.stem: path for path in figure_paths},
        "fig_05_recommendation_metrics": metric_figure,
        "fig_06_recommendation_examples": example_path,
        **{path.stem: path for path in sensitivity_figure_paths},
    }
    _update_registry(config_digest, artifacts)

    all_outputs = [
        *runtime_outputs.values(),
        *tracked.values(),
        *figure_paths,
        *sensitivity_figure_paths,
        metric_figure,
        example_path,
        Path("artifacts/manifests/figure_table_registry.csv"),
    ]
    manifest = {
        "schema_version": 1,
        "stage": 6,
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": config_digest,
        "resolved_configuration": resolved,
        "input_artifact_hashes": {
            str(items_path): items_hash,
            config["paths"]["knowledge_base"]: sha256_file(Path(config["paths"]["knowledge_base"])),
            config["paths"]["kb_expansion_source"]: sha256_file(
                Path(config["paths"]["kb_expansion_source"])
            ),
            config["paths"]["legacy_kb_audit"]: sha256_file(
                Path(config["paths"]["legacy_kb_audit"])
            ),
            config["paths"]["legacy_rule_audit"]: sha256_file(
                Path(config["paths"]["legacy_rule_audit"])
            ),
            config["paths"]["kb_coverage_matrix"]: sha256_file(
                Path(config["paths"]["kb_coverage_matrix"])
            ),
            config["paths"]["kb_source_registry"]: sha256_file(
                Path(config["paths"]["kb_source_registry"])
            ),
            config["paths"]["kb_rule_similarity_audit"]: sha256_file(
                Path(config["paths"]["kb_rule_similarity_audit"])
            ),
            "dataset_fingerprint": dataset_fingerprint,
        },
        "output_artifact_hashes": {str(path): sha256_file(path) for path in all_outputs},
        "row_counts": {
            "confirmatory_cases": len(cases),
            "candidate_rows": len(traces),
            "case_metric_rows": len(case_metrics),
            "locked_cases": len(locked_cases),
            "recommendation_result_rows": len(recommendation_results),
            "contrast_rows": len(contrasts),
        },
        "models": models["embedders"],
        "statistics": stats,
        "environment": environment_summary(),
        "failure_counts": {"ranking_failures": 0, "trace_failures": 0},
        "trace_validation": {
            "complete_five_rule_traces": all(
                len(row["evidence_trace"]["rules"]) == 5 for row in locked_cases
            ),
            "locked_cases_checked": len(locked_cases),
        },
        "command": (
            "python scripts/run_stage6_recommendation_eval.py --config configs/experiment.yaml"
        ),
    }
    runtime_manifest = run_dir / "manifest.json"
    write_new_json(runtime_manifest, manifest)
    write_json(Path("artifacts/manifests/stage6_recommendation_manifest.json"), manifest)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "cases": len(cases),
                "candidate_rows": len(traces),
                "micro_results": recommendation_results[
                    recommendation_results["aggregation"].eq("micro")
                ].to_dict("records"),
                "evidence_participation": evidence_summary.to_dict("records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
