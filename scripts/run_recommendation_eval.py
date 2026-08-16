"""Stage 4 validation-only evidence retrieval and reranking analysis."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from evidence_fashion.data import (
    attach_candidate_pools,
    build_evaluation_cases,
    load_pinned_split,
    write_jsonl,
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
from evidence_fashion.reranking import (
    pareto_frontier,
    ranking_metrics,
    rerank_candidates,
    select_pareto_knee,
)
from evidence_fashion.retrieval import (
    CLIPEmbedder,
    MiniLMEmbedder,
    cosine_scores,
    fuse_clip_embeddings,
)
from evidence_fashion.rule_retrieval import (
    CandidateEvidenceTrace,
    RuleRetriever,
    candidate_rule_representation,
    truncate_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fusion-search-only", action="store_true")
    return parser.parse_args()


def _series_record(row: pd.Series) -> dict[str, Any]:
    """Convert pandas scalar values to JSON-safe Python values."""
    return json.loads(row.to_json())


def _active_items(config: dict[str, Any]) -> tuple[pd.DataFrame, Path, str]:
    manifest = json.loads(Path(config["paths"]["active_data_manifest"]).read_text(encoding="utf-8"))
    path = Path(
        next(
            key
            for key in manifest["output_artifact_hashes"]
            if key.endswith("prepared_items.parquet")
        )
    )
    digest = sha256_file(path)
    if digest != manifest["output_artifact_hashes"][str(path)]:
        raise ValueError("Active prepared metadata does not match its manifest hash.")
    return pd.read_parquet(path), path, digest


def _stage4_cases(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    local = copy.deepcopy(config)
    settings = config["stage4_validation"]
    local["recommendation_evaluation"].update(
        {
            "case_count": settings["case_count"],
            "cases_per_category": settings["cases_per_category"],
            "case_split": settings["split"],
            "case_seed": settings["case_seed"],
        }
    )
    local["candidate_pool"]["max_negatives"] = settings["candidate_max_negatives"]
    return attach_candidate_pools(frame, build_evaluation_cases(frame, local), local)


def _item_images(rows: pd.DataFrame, raw_split, clip: CLIPEmbedder, batch_size: int) -> np.ndarray:
    batches = []
    image_column = "image"
    for start in range(0, len(rows), batch_size):
        indices = rows.iloc[start : start + batch_size]["original_dataset_index"]
        images = [raw_split[int(index)][image_column] for index in indices]
        batches.append(clip.encode_images(images, batch_size=batch_size))
    return np.concatenate(batches)


def _trace_score(trace: CandidateEvidenceTrace, top_k: int, settings: dict[str, Any]) -> float:
    scores = np.asarray(
        [rule.weighted_contribution for rule in trace.rules[:top_k]], dtype=np.float64
    )
    return float(
        settings["score_max_weight"] * scores.max()
        + settings["score_mean_weight"] * scores.mean()
    )


def _case_metrics(
    candidates: pd.DataFrame,
    clip_weight: float,
    evidence_weight: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    ranked = rerank_candidates(
        candidates,
        clip_weight=clip_weight,
        evidence_weight=evidence_weight,
    )
    metrics = ranking_metrics(ranked["is_positive"])
    pre = candidates.sort_values(["clip_score", "item_id"], ascending=[False, True])
    metrics.update(
        {
            "top1_evidence": float(ranked.iloc[0]["evidence_score"]),
            "evidence_gain": float(
                ranked.iloc[0]["evidence_score"] - pre.iloc[0]["evidence_score"]
            ),
            "changed_top1": float(ranked.iloc[0]["item_id"] != pre.iloc[0]["item_id"]),
            "trace_participation": float(len(ranked.iloc[0]["trace"].rules) > 0),
        }
    )
    return metrics, ranked


def _aggregate_search(rows: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["rule_top_k", "evidence_weight", "clip_weight"], as_index=False)
        .agg(
            hr_at_10=("hr_at_10", "mean"),
            ndcg_at_10=("ndcg_at_10", "mean"),
            mrr=("mrr", "mean"),
            mean_top1_evidence=("top1_evidence", "mean"),
            mean_evidence_gain=("evidence_gain", "mean"),
            changed_top1_rate=("changed_top1", "mean"),
            rule_trace_participation=("trace_participation", "mean"),
        )
        .sort_values(["rule_top_k", "evidence_weight"])
        .reset_index(drop=True)
    )
    objectives = config["reranking_search"]["objectives"]
    return pareto_frontier(grouped, objectives)


def _fusion_search(
    cases: pd.DataFrame,
    candidate_ids: list[str],
    candidate_image: np.ndarray,
    candidate_text: np.ndarray,
    query_image: np.ndarray,
    query_text: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    fusion = config["retrieval"]["fusion"]
    rows = []
    for image_weight in fusion["validation_grid_image_weights"]:
        text_weight = 1.0 - float(image_weight)
        candidate_fused = fuse_clip_embeddings(
            candidate_image,
            candidate_text,
            image_weight=float(image_weight),
            text_weight=text_weight,
        )
        query_fused = fuse_clip_embeddings(
            query_image,
            query_text,
            image_weight=float(image_weight),
            text_weight=text_weight,
        )
        candidate_vector = {
            item_id: candidate_fused[index] for index, item_id in enumerate(candidate_ids)
        }
        metrics = []
        for case_index, case in enumerate(cases.to_dict("records")):
            frame = pd.DataFrame(
                {
                    "item_id": case["candidate_item_ids"],
                    "is_positive": case["candidate_relevance"],
                    "score": cosine_scores(
                        query_fused[case_index],
                        np.stack(
                            [candidate_vector[item] for item in case["candidate_item_ids"]]
                        ),
                    ),
                }
            ).sort_values(["score", "item_id"], ascending=[False, True], kind="stable")
            metrics.append(ranking_metrics(frame["is_positive"]))
        rows.append(
            {
                "image_weight": float(image_weight),
                "text_weight": text_weight,
                "hr_at_10": float(np.mean([row["hr_at_10"] for row in metrics])),
                "ndcg_at_10": float(np.mean([row["ndcg_at_10"] for row in metrics])),
                "mrr": float(np.mean([row["mrr"] for row in metrics])),
                "distance_from_reference": abs(
                    float(image_weight) - float(fusion["image_weight"])
                ),
            }
        )
    table = pd.DataFrame(rows)
    selected = table.sort_values(
        [
            fusion["selection_metric"],
            "mrr",
            "hr_at_10",
            "distance_from_reference",
            "image_weight",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).iloc[0]
    table["selected"] = table["image_weight"].eq(selected["image_weight"])
    return table, selected


def _candidate_pool_sensitivity(
    cases: pd.DataFrame,
    candidate_ids: list[str],
    candidate_minilm: np.ndarray,
    candidate_clip_image: np.ndarray,
    candidate_clip_text: np.ndarray,
    candidate_clip_fused: np.ndarray,
    query_minilm: np.ndarray,
    query_clip_image: np.ndarray,
    query_clip_text: np.ndarray,
    query_clip_fused: np.ndarray,
    traces_by_case_item: dict[tuple[str, str], CandidateEvidenceTrace],
    config: dict[str, Any],
) -> pd.DataFrame:
    candidate_index = {item_id: index for index, item_id in enumerate(candidate_ids)}
    methods = (
        "minilm_text",
        "clip_image",
        "clip_text",
        "fused_clip",
        "evidence_rerank_fixed_075_025",
    )
    rows: list[dict[str, Any]] = []
    seed = int(config["project"]["random_seed"])
    selected_top_k = int(config["stage4_validation"]["selected_rule_top_k"])
    rule_settings = config["rule_retrieval"]
    for pool_target in config["candidate_pool"]["validation_sensitivity_pool_sizes"]:
        for case_index, case in enumerate(cases.to_dict("records")):
            positives = [
                item
                for item, relevant in zip(
                    case["candidate_item_ids"], case["candidate_relevance"], strict=True
                )
                if relevant
            ]
            negatives = [
                item
                for item, relevant in zip(
                    case["candidate_item_ids"], case["candidate_relevance"], strict=True
                )
                if not relevant
            ]
            negatives.sort(
                key=lambda item: (
                    hashlib.sha256(f"{seed}:{case['case_id']}:{item}".encode()).hexdigest(),
                    item,
                )
            )
            retained = positives + negatives[: max(int(pool_target) - len(positives), 0)]
            retained = sorted(set(retained))
            indices = [candidate_index[item] for item in retained]
            relevance = pd.Series([item in set(positives) for item in retained], index=retained)
            score_sets = {
                "minilm_text": cosine_scores(query_minilm[case_index], candidate_minilm[indices]),
                "clip_image": cosine_scores(
                    query_clip_image[case_index], candidate_clip_image[indices]
                ),
                "clip_text": cosine_scores(
                    query_clip_text[case_index], candidate_clip_text[indices]
                ),
                "fused_clip": cosine_scores(
                    query_clip_fused[case_index], candidate_clip_fused[indices]
                ),
            }
            for method in methods[:-1]:
                ranked = pd.DataFrame(
                    {
                        "item_id": retained,
                        "is_positive": relevance.loc[retained].to_numpy(bool),
                        "score": score_sets[method],
                    }
                ).sort_values(["score", "item_id"], ascending=[False, True], kind="stable")
                metrics = ranking_metrics(ranked["is_positive"])
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "target_category": case["target_category"],
                        "pool_target": int(pool_target),
                        "actual_pool_size": len(retained),
                        "method": method,
                        **metrics,
                    }
                )
            exact_traces = [
                truncate_trace(
                    traces_by_case_item[(case["case_id"], item)],
                    selected_top_k,
                    rule_settings,
                )
                for item in retained
            ]
            reranked = rerank_candidates(
                pd.DataFrame(
                    {
                        "item_id": retained,
                        "is_positive": relevance.loc[retained].to_numpy(bool),
                        "clip_score": score_sets["fused_clip"],
                        "evidence_score": [trace.evidence_score for trace in exact_traces],
                    }
                ),
                clip_weight=float(config["reranking"]["reference_clip_weight"]),
                evidence_weight=float(config["reranking"]["reference_evidence_weight"]),
            )
            metrics = ranking_metrics(reranked["is_positive"])
            rows.append(
                {
                    "case_id": case["case_id"],
                    "target_category": case["target_category"],
                    "pool_target": int(pool_target),
                    "actual_pool_size": len(retained),
                    "method": methods[-1],
                    **metrics,
                }
            )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["pool_target", "method"], as_index=False)
        .agg(
            cases=("case_id", "count"),
            actual_pool_min=("actual_pool_size", "min"),
            actual_pool_mean=("actual_pool_size", "mean"),
            actual_pool_max=("actual_pool_size", "max"),
            hr_at_10=("hr_at_10", "mean"),
            ndcg_at_10=("ndcg_at_10", "mean"),
            mrr=("mrr", "mean"),
        )
        .sort_values(["pool_target", "method"], kind="stable")
        .reset_index(drop=True)
    )


def _diversity(
    all_traces: list[dict[str, Any]],
    locked_cases: list[dict[str, Any]],
    config: dict[str, Any],
    kb_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    packet_count = len(all_traces)
    slot_counter: Counter[str] = Counter()
    packet_counter: Counter[str] = Counter()
    rule_text: dict[str, str] = {}
    packets_by_category: dict[str, list[set[str]]] = defaultdict(list)
    packets_by_case: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for record in all_traces:
        ids = tuple(rule["rule_id"] for rule in record["trace"]["rules"])
        packet_counter.update(set(ids))
        slot_counter.update(ids)
        packets_by_category[record["target_category"]].append(set(ids))
        packets_by_case[record["case_id"]].add(ids)
        for rule in record["trace"]["rules"]:
            rule_text[rule["rule_id"]] = rule["rule_text"]
    total_slots = sum(slot_counter.values())
    frequency = pd.DataFrame(
        [
            {
                "rule_id": rule_id,
                "rule_text": rule_text.get(rule_id, ""),
                "packet_count": packet_counter[rule_id],
                "packet_prevalence": packet_counter[rule_id] / packet_count,
                "slot_count": slot_counter[rule_id],
                "slot_share": slot_counter[rule_id] / total_slots,
                "generic_flag": packet_counter[rule_id] / packet_count
                >= config["stage4_validation"]["rule_generic_packet_prevalence_threshold"],
            }
            for rule_id in sorted(slot_counter)
        ]
    )
    probabilities = np.asarray(list(slot_counter.values()), dtype=float) / total_slots

    def mean_jaccard(groups: list[set[str]], limit: int = 2000) -> float:
        values = []
        for left_index, left in enumerate(groups):
            for right in groups[left_index + 1 :]:
                values.append(len(left & right) / len(left | right))
                if len(values) >= limit:
                    return float(np.mean(values))
        return float(np.mean(values)) if values else 0.0

    within = {
        category: mean_jaccard(groups) for category, groups in sorted(packets_by_category.items())
    }
    between_values = []
    categories = sorted(packets_by_category)
    for left_index, left_category in enumerate(categories):
        for right_category in categories[left_index + 1 :]:
            left = packets_by_category[left_category][:50]
            right = packets_by_category[right_category][:50]
            between_values.extend(len(a & b) / len(a | b) for a in left for b in right)
    locked_packets = [
        tuple(rule["rule_id"] for rule in row["evidence_trace"]["rules"])
        for row in locked_cases
    ]
    return frequency, {
        "candidate_packets": packet_count,
        "kb_rules_used": len(slot_counter),
        "kb_rule_coverage": len(slot_counter) / kb_size,
        "shannon_entropy": float(-(probabilities * np.log2(probabilities)).sum()),
        "within_category_mean_jaccard": within,
        "between_category_mean_jaccard": float(np.mean(between_values)),
        "distinct_candidate_packets": len({
            tuple(rule["rule_id"] for rule in record["trace"]["rules"])
            for record in all_traces
        }),
        "identical_locked_packet_rate": 1 - len(set(locked_packets)) / len(locked_packets),
        "mean_candidate_packet_variants_per_case": float(
            np.mean([len(value) for value in packets_by_case.values()])
        ),
        "generic_flagged_rules": frequency.loc[frequency["generic_flag"], "rule_id"].tolist(),
    }


def _write_pareto_svg(points: pd.DataFrame, selected: pd.Series, path: Path) -> None:
    width, height, margin = 720, 480, 60
    x = points["mean_top1_evidence"].to_numpy(float)
    y = points["ndcg_at_10"].to_numpy(float)
    x_span = max(float(x.max() - x.min()), 1e-12)
    y_span = max(float(y.max() - y.min()), 1e-12)
    circles = []
    for _, row in points.iterrows():
        cx = margin + (row["mean_top1_evidence"] - x.min()) / x_span * (width - 2 * margin)
        cy = height - margin - (row["ndcg_at_10"] - y.min()) / y_span * (height - 2 * margin)
        colour = "#0072B2" if row["pareto_status"] == "frontier" else "#999999"
        radius = 7 if (
            row["rule_top_k"] == selected["rule_top_k"]
            and row["evidence_weight"] == selected["evidence_weight"]
        ) else 4
        circles.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius}" fill="{colour}"/>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" '
        f'y2="{height-margin}" stroke="black"/>'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
        f'y2="{height-margin}" stroke="black"/>\n'
        f'{"".join(circles)}'
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle">'
        'Mean top-1 evidence score</text>'
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 '
        f'{height/2})" text-anchor="middle">Validation NDCG@10</text>'
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="16">'
        'Recommendation quality versus evidence participation</text></svg>'
    )
    path.write_text(svg, encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    if not args.validate_only and not args.dry_run and not args.resume:
        raise SystemExit("Stage 6 confirmatory evaluation is not authorized; use --validate-only.")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    config_digest = configuration_hash(resolved)
    runtime_root = args.runtime_root or Path(config["paths"]["runtime_root"])
    run_id = f"stage4-validation-{config_digest[:12]}"
    run_dir = runtime_root / "stage4" / run_id
    manifest_path = run_dir / "manifest.json"
    if args.dry_run:
        summary = {"run_id": run_id, "case_count": config["stage4_validation"]["case_count"]}
        print(json.dumps(summary, indent=2))
        return
    if args.resume and manifest_path.exists():
        print(manifest_path.read_text(encoding="utf-8"))
        return
    if run_dir.exists():
        raise FileExistsError(f"Immutable Stage 4 run already exists: {run_dir}")

    items, items_path, items_hash = _active_items(config)
    cases = _stage4_cases(items, config)
    item_lookup = items.set_index("item_id", drop=False)
    candidate_ids = sorted({item for values in cases["candidate_item_ids"] for item in values})
    candidate_rows = item_lookup.loc[candidate_ids].reset_index(drop=True)
    query_rows = item_lookup.loc[cases["query_item_id"].tolist()].reset_index(drop=True)
    raw_split, dataset_fingerprint = load_pinned_split(config)
    clip = CLIPEmbedder(models["embedders"]["clip"])
    mini = MiniLMEmbedder(models["embedders"]["minilm"])
    batch_size = config["stage4_validation"]["embedding_batch_size"]

    candidate_texts = (
        candidate_rows["category"].astype(str) + " | " + candidate_rows["text"].astype(str)
    ).tolist()
    candidate_clip_text = clip.encode_text(candidate_texts, batch_size=batch_size)
    candidate_clip_image = _item_images(candidate_rows, raw_split, clip, batch_size)
    candidate_minilm = mini.encode(candidate_texts, batch_size=batch_size)
    fusion = config["retrieval"]["fusion"]
    candidate_clip = fuse_clip_embeddings(
        candidate_clip_image,
        candidate_clip_text,
        image_weight=fusion["image_weight"],
        text_weight=fusion["text_weight"],
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
    query_minilm = mini.encode(query_texts, batch_size=batch_size)
    tracked_fusion = Path("artifacts/tables/table_stage4_fusion_search.csv")
    fusion_table = pd.read_csv(tracked_fusion)
    fusion_selected = fusion_table.sort_values(
        ["ndcg_at_10", "mrr", "hr_at_10"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0]
    if args.fusion_search_only:
        print(
            json.dumps(
                {
                    "validation_cases": len(cases),
                    "selected": _series_record(fusion_selected),
                },
                indent=2,
            )
        )
        return
    if not np.isclose(float(fusion_selected["image_weight"]), 0.20):
        raise ValueError("The preserved fusion diagnostic no longer records 0.20/0.80 as best.")
    query_clip = fuse_clip_embeddings(
        query_clip_image,
        query_clip_text,
        image_weight=fusion["image_weight"],
        text_weight=fusion["text_weight"],
    )
    candidate_vector = {
        item_id: candidate_clip[index] for index, item_id in enumerate(candidate_ids)
    }

    kb = load_audited_rules(config)
    rule_embeddings = mini.encode(kb["rule_text"].astype(str).tolist(), batch_size=batch_size)
    retriever = RuleRetriever(kb, rule_embeddings, config["rule_retrieval"])
    representation_records = []
    case_candidate_pairs = []
    for case in cases.to_dict("records"):
        for item_id, relevance in zip(
            case["candidate_item_ids"], case["candidate_relevance"], strict=True
        ):
            candidate = item_lookup.loc[item_id].to_dict()
            representation_records.append(candidate_rule_representation(case, candidate))
            case_candidate_pairs.append((case, candidate, bool(relevance)))
    representation_embeddings = mini.encode(representation_records, batch_size=batch_size)

    traces: list[CandidateEvidenceTrace] = []
    trace_records = []
    for index, (case, candidate, relevance) in enumerate(case_candidate_pairs):
        trace = retriever.retrieve_and_score(
            case=case,
            candidate=candidate,
            representation_embedding=representation_embeddings[index],
            top_k=config["stage4_validation"]["selected_rule_top_k"],
        )
        traces.append(trace)
        trace_records.append(
            {
                "case_id": case["case_id"],
                "target_category": case["target_category"],
                "is_positive": relevance,
                "trace": trace.to_dict(),
            }
        )

    tracked_sensitivity = Path("artifacts/tables/table_stage4_pool_sensitivity.csv")
    sensitivity = pd.read_csv(tracked_sensitivity)

    selected_rankings: dict[str, pd.DataFrame] = {}
    main_metric_rows: list[dict[str, Any]] = []
    candidate_index = {item_id: index for index, item_id in enumerate(candidate_ids)}
    offset = 0
    settings = config["rule_retrieval"]
    for case_index, case in enumerate(cases.to_dict("records")):
        count = len(case["candidate_item_ids"])
        case_traces = traces[offset : offset + count]
        offset += count
        base = pd.DataFrame(
            {
                "item_id": case["candidate_item_ids"],
                "is_positive": case["candidate_relevance"],
                "clip_score": [
                    float(score)
                    for score in cosine_scores(
                        query_clip[case_index],
                        np.stack([candidate_vector[item] for item in case["candidate_item_ids"]]),
                    )
                ],
                "trace": case_traces,
            }
        )
        indices = [candidate_index[item] for item in case["candidate_item_ids"]]
        baseline_scores = {
            "minilm_text": cosine_scores(query_minilm[case_index], candidate_minilm[indices]),
            "clip_image": cosine_scores(
                query_clip_image[case_index], candidate_clip_image[indices]
            ),
            "clip_text": cosine_scores(
                query_clip_text[case_index], candidate_clip_text[indices]
            ),
            "fused_clip_main_040_060": base["clip_score"].to_numpy(float),
        }
        for method, scores in baseline_scores.items():
            ranked = pd.DataFrame(
                {
                    "item_id": case["candidate_item_ids"],
                    "is_positive": case["candidate_relevance"],
                    "score": scores,
                }
            ).sort_values(["score", "item_id"], ascending=[False, True], kind="stable")
            main_metric_rows.append(
                {
                    "case_id": case["case_id"],
                    "method": method,
                    **ranking_metrics(ranked["is_positive"]),
                }
            )
        exact_traces = [
            truncate_trace(trace, config["stage4_validation"]["selected_rule_top_k"], settings)
            for trace in case_traces
        ]
        base["trace"] = exact_traces
        base["evidence_score"] = [trace.evidence_score for trace in exact_traces]
        metrics, ranking = _case_metrics(
            base,
            float(config["stage4_validation"]["selected_clip_weight"]),
            float(config["stage4_validation"]["selected_evidence_weight"]),
        )
        main_metric_rows.append(
            {"case_id": case["case_id"], "method": "evidence_rerank_main_075_025", **metrics}
        )
        selected_rankings[case["case_id"]] = ranking

    main_results = (
        pd.DataFrame(main_metric_rows)
        .groupby("method", as_index=False)
        .agg(
            cases=("case_id", "count"),
            hr_at_1=("hr_at_1", "mean"),
            hr_at_5=("hr_at_5", "mean"),
            hr_at_10=("hr_at_10", "mean"),
            ndcg_at_1=("ndcg_at_1", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
            ndcg_at_10=("ndcg_at_10", "mean"),
            mrr=("mrr", "mean"),
        )
        .sort_values("method", kind="stable")
    )
    main_results.insert(
        1,
        "candidate_max_negatives",
        config["stage4_validation"]["candidate_max_negatives"],
    )
    main_results.insert(2, "image_weight", config["retrieval"]["fusion"]["image_weight"])
    main_results.insert(3, "text_weight", config["retrieval"]["fusion"]["text_weight"])

    tracked_points = Path("artifacts/tables/table_stage4_pareto.csv")
    points = pd.read_csv(tracked_points)
    knee = select_pareto_knee(
        points,
        config["reranking_search"]["objectives"],
        tie_columns=["evidence_weight", "rule_top_k"],
    )
    locked_cases = []
    trace_reproduction_errors = []
    for case in cases.to_dict("records"):
        top = selected_rankings[case["case_id"]].iloc[0]
        candidate = item_lookup.loc[top["item_id"]]
        exact_trace = top["trace"]
        reproduced_score = _trace_score(
            exact_trace, len(exact_trace.rules), config["rule_retrieval"]
        )
        trace_reproduction_errors.append(abs(reproduced_score - float(top["evidence_score"])))
        if not np.isclose(reproduced_score, float(top["evidence_score"]), atol=1e-12):
            raise ValueError("Stored B trace does not reproduce the candidate evidence score.")
        locked_cases.append(
            {
                "case_id": case["case_id"],
                "query_item_id": case["query_item_id"],
                "query_outfit_id": case["query_outfit_id"],
                "query_item_minimal_name": str(case["query_text"] or case["query_category"]),
                "request": case["user_request"],
                "target_category": case["target_category"],
                "locked_candidate_id": str(top["item_id"]),
                "locked_candidate_minimal_name": str(candidate["text"] or candidate["category"]),
                "clip_score": float(top["clip_score"]),
                "normalized_clip_score": float(top["normalized_clip_score"]),
                "evidence_score": float(top["evidence_score"]),
                "normalized_evidence_score": float(top["normalized_evidence_score"]),
                "final_score": float(top["final_score"]),
                "pre_rerank_rank": int(top["pre_rerank_rank"]),
                "post_rerank_rank": int(top["post_rerank_rank"]),
                "evidence_trace": exact_trace.to_dict(),
            }
        )
    frequency, diversity = _diversity(trace_records, locked_cases, config, len(kb))

    run_dir.mkdir(parents=True)
    cases_path = run_dir / "validation_cases.jsonl"
    traces_path = run_dir / "candidate_evidence_traces.jsonl"
    locked_path = run_dir / "locked_cases.jsonl"
    points_path = run_dir / "pareto_points.csv"
    frequency_path = run_dir / "rule_frequency.csv"
    diversity_path = run_dir / "rule_diversity.json"
    main_results_path = run_dir / "main_results.csv"
    write_jsonl(cases_path, cases.to_dict("records"))
    write_jsonl(traces_path, trace_records)
    write_jsonl(locked_path, locked_cases)
    points.to_csv(points_path, index=False)
    frequency.to_csv(frequency_path, index=False)
    main_results.to_csv(main_results_path, index=False)
    diversity_path.write_text(
        json.dumps(diversity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    tracked_frequency = Path("artifacts/tables/table_stage4_rule_frequency.csv")
    tracked_figure = Path("artifacts/figures/fig_stage4_pareto_frontier.svg")
    tracked_main_results = Path("artifacts/tables/table_stage4_main_results.csv")
    main_results.to_csv(tracked_main_results, index=False)
    selected_point = points[
        (points["rule_top_k"] == config["stage4_validation"]["selected_rule_top_k"])
        & (points["evidence_weight"] == config["stage4_validation"]["selected_evidence_weight"])
    ].iloc[0]
    output_paths = [
        cases_path,
        traces_path,
        locked_path,
        points_path,
        frequency_path,
        diversity_path,
        main_results_path,
        tracked_main_results,
    ]
    output_hashes = {str(path): sha256_file(path) for path in output_paths}
    supporting_paths = [
        tracked_points,
        tracked_frequency,
        tracked_fusion,
        tracked_sensitivity,
        tracked_figure,
        Path(config["paths"]["category_audit_table"]),
    ]
    manifest = {
        "schema_version": 1,
        "stage": 4,
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
        "output_artifact_hashes": output_hashes,
        "supporting_diagnostic_artifact_hashes": {
            str(path): sha256_file(path) for path in supporting_paths
        },
        "models": {
            name: {
                "model_id": value["model_id"],
                "immutable_digest": value["immutable_digest"],
                "device": mini.device if name == "minilm" else clip.device,
            }
            for name, value in models["embedders"].items()
        },
        "row_counts": {
            "validation_cases": len(cases),
            "candidate_traces": len(trace_records),
            "pareto_configurations": len(points),
            "pareto_frontier_points": int(points["pareto_status"].eq("frontier").sum()),
            "locked_cases": len(locked_cases),
            "pool_sensitivity_rows": len(sensitivity),
            "main_result_rows": len(main_results),
        },
        "failure_counts": {"trace_failures": 0, "ranking_failures": 0},
        "trace_validation": {
            "locked_traces_checked": len(trace_reproduction_errors),
            "maximum_absolute_score_reproduction_error": max(trace_reproduction_errors),
            "complete_trace_matches_selected_rule_count": all(
                len(row["evidence_trace"]["rules"])
                == config["stage4_validation"]["selected_rule_top_k"]
                for row in locked_cases
            ),
        },
        "seed": config["stage4_validation"]["case_seed"],
        "environment": environment_summary(),
        "command": (
            "python scripts/run_recommendation_eval.py "
            "--config configs/experiment.yaml --validate-only"
        ),
        "selection": {
            "fusion_validation_selection": _series_record(fusion_selected),
            "main_fusion_operating_point": {
                "image_weight": fusion["image_weight"],
                "text_weight": fusion["text_weight"],
                "policy": fusion["operating_point_policy"],
                "validation_optimum": False,
            },
            "pareto_knee_recommendation": _series_record(knee),
            "frozen_stage5_operating_point": {
                "rule_top_k": config["stage4_validation"]["selected_rule_top_k"],
                "clip_weight": config["stage4_validation"]["selected_clip_weight"],
                "evidence_weight": config["stage4_validation"]["selected_evidence_weight"],
                "policy": config["reranking_search"]["stage5_operating_point_policy"],
            },
            "policy": config["reranking_search"]["stage5_operating_point_policy"],
            "researcher_approval_required_for_reference_change": False,
            "selection_method": config["reranking_search"]["selection_method"],
            "maximum_ndcg10_relative_loss": config["reranking_search"][
                "maximum_ndcg10_relative_loss"
            ],
        },
        "diversity": diversity,
        "exact_trace_reuse_required": True,
    }
    write_new_json(manifest_path, manifest)
    write_json(Path("artifacts/manifests/evidence_trace_manifest.json"), manifest)
    write_json(Path("artifacts/manifests/stage4_reranking_manifest.json"), manifest)

    registry_path = Path("artifacts/manifests/figure_table_registry.csv")
    existing_rows = []
    if registry_path.exists():
        with registry_path.open(encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.reader(handle))
    with registry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "artifact_id",
            "artifact_type",
            "title",
            "research_question",
            "source_data",
            "generation_function_or_script",
            "configuration_hash",
            "output_path",
            "caption",
            "intended_thesis_chapter",
            "intended_paper_section",
            "status",
            "notes",
        ]
        writer.writerow(header)
        writer.writerows(
            row for row in existing_rows[1:] if row[0] != "table_stage4_main_results"
        )
        writer.writerow(
            [
                "table_stage4_main_results",
                "table",
                "Researcher-selected main Stage 4 operating point",
                "How do the frozen recommendation methods perform in the main controlled pool?",
                str(main_results_path),
                "scripts/run_recommendation_eval.py:main",
                config_digest,
                str(tracked_main_results),
                "Main approximately 100-candidate validation run under 0.40/0.60 "
                "fusion and fixed 0.75/0.25 five-rule reranking.",
                "Methods and results",
                "Recommendation evaluation",
                "final",
                "Researcher-selected design defaults; completed larger-pool diagnostics "
                "are preserved separately.",
            ]
        )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "cases": len(cases),
                "candidate_traces": len(trace_records),
                "main_results": json.loads(main_results.to_json(orient="records")),
                "pareto_knee": _series_record(knee),
                "frozen_stage5": _series_record(selected_point),
                "diversity": diversity,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
