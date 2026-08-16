"""Run an additive FashionCLIP baseline on the frozen Stage 6 candidate pools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from run_recommendation_eval import _active_items, _item_images

from evidence_fashion.data import attach_candidate_pools, build_evaluation_cases, load_pinned_split
from evidence_fashion.evaluation.recommendation import aggregate_recommendation_metrics
from evidence_fashion.evaluation.statistics import (
    clustered_bootstrap_mean,
    holm_adjust,
    two_sided_bootstrap_pvalue,
)
from evidence_fashion.manifest import (
    environment_summary,
    git_commit,
    sha256_file,
    utc_timestamp,
    write_json,
    write_new_json,
)
from evidence_fashion.reranking import ranking_metrics
from evidence_fashion.retrieval import CLIPEmbedder, cosine_scores, fuse_clip_embeddings

METRICS = ["hr_at_1", "hr_at_5", "hr_at_10", "ndcg_at_1", "ndcg_at_5", "ndcg_at_10", "mrr"]
METHOD_COMPARISONS = {
    "fashionclip_image": "clip_image",
    "fashionclip_text": "clip_text",
    "fashionclip_fused_040_060": "fused_clip_main_040_060",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--baseline-config", type=Path, default=Path("configs/fashionclip_baseline.yaml")
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("artifacts/manifests/stage6_recommendation_manifest.json"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_output(manifest: dict[str, Any], suffix: str) -> tuple[Path, str]:
    matches = [
        (Path(raw), digest)
        for raw, digest in manifest["output_artifact_hashes"].items()
        if raw.replace("\\", "/").endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one Stage 6 output ending in {suffix!r}; found {len(matches)}")
    path, expected = matches[0]
    if not path.exists() or sha256_file(path) != expected:
        raise ValueError(f"Frozen Stage 6 input is missing or changed: {path}")
    return path, expected


def _rank(item_ids: list[str], relevance: list[bool], scores: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"item_id": item_ids, "is_positive": relevance, "score": scores}
    ).sort_values(["score", "item_id"], ascending=[False, True], kind="stable")


def _contrasts(
    fashion: pd.DataFrame, reference: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    index = ["case_id"]
    stats = config["statistics"]
    rows: list[dict[str, Any]] = []
    for comparison_index, (method, reference_method) in enumerate(
        METHOD_COMPARISONS.items()
    ):
        left = fashion[fashion["method"].eq(method)].set_index(index)
        right = reference[reference["method"].eq(reference_method)].set_index(index)
        joined = left.join(
            right, lsuffix="_fashionclip", rsuffix="_reference", how="inner"
        )
        if len(joined) != int(config["recommendation_evaluation"]["case_count"]):
            raise ValueError(f"Incomplete FashionCLIP/reference matrix for {method}.")
        for metric_index, metric in enumerate(METRICS):
            differences = joined[f"{metric}_fashionclip"] - joined[f"{metric}_reference"]
            estimate, lower, upper, estimates = clustered_bootstrap_mean(
                differences,
                joined["query_outfit_id_fashionclip"],
                replicates=int(stats["bootstrap_replicates"]),
                confidence_level=float(stats["confidence_level"]),
                seed=(
                    int(config["project"]["random_seed"])
                    + 900
                    + 100 * comparison_index
                    + metric_index
                ),
            )
            rows.append(
                {
                    "method": method,
                    "reference_method": reference_method,
                    "metric": metric,
                    "mean_paired_difference_fashionclip_minus_reference": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "p_value": two_sided_bootstrap_pvalue(estimates),
                    "cases": len(joined),
                    "bootstrap_unit": stats["bootstrap_unit"],
                }
            )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = holm_adjust(result["p_value"])
    result["reject_at_0_05"] = result["holm_adjusted_p_value"].lt(0.05)
    return result


def _update_registry(config_digest: str, outputs: dict[str, Path]) -> None:
    registry = Path("artifacts/manifests/figure_table_registry.csv")
    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = set(outputs)
    rows = [row for row in rows if row["artifact_id"] not in ids]
    titles = {
        "table_stage6b_fashionclip_results": "Additive FashionCLIP recommendation results",
        "table_stage6b_fashionclip_contrasts": "Paired FashionCLIP versus general CLIP contrasts",
    }
    for artifact_id, path in outputs.items():
        rows.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": "table",
                "title": titles[artifact_id],
                "research_question": (
                    "Does a fashion-domain CLIP baseline improve controlled-pool "
                    "recommendation?"
                ),
                "source_data": ".runtime/stage6b",
                "generation_function_or_script": "scripts/run_stage6b_fashionclip_baseline.py",
                "configuration_hash": config_digest,
                "output_path": str(path),
                "caption": titles[artifact_id] + "; additive frozen-pool analysis.",
                "intended_thesis_chapter": "Results",
                "intended_paper_section": "Recommendation evaluation",
                "status": "final_additive",
                "notes": (
                    "Same 1,000 cases, candidate pools, texts, images, fusion weights, "
                    "and metrics as Stage 6."
                ),
            }
        )
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    baseline = yaml.safe_load(args.baseline_config.read_text(encoding="utf-8"))
    model = dict(baseline["fashionclip"])
    model["local_files_only"] = not args.allow_download
    digest = _canonical_hash(
        {
            "stage6_frozen_sections": {
                key: config[key]
                for key in (
                    "dataset",
                    "preprocessing",
                    "splits",
                    "recommendation_evaluation",
                    "candidate_pool",
                    "retrieval",
                    "statistics",
                )
            },
            "fashionclip": baseline,
        }
    )
    run_id = f"stage6b-fashionclip-{digest[:12]}"
    runtime_root = args.runtime_root or Path(config["paths"]["runtime_root"])
    run_dir = runtime_root / "stage6b" / run_id
    if args.dry_run:
        print(json.dumps({"run_id": run_id, "model": model, "cases": 1000}, indent=2))
        return
    if run_dir.exists():
        raise FileExistsError(f"Immutable FashionCLIP run already exists: {run_dir}")

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    frozen_source = source_manifest["resolved_configuration"]["experiment"]
    frozen_keys = (
        "dataset",
        "preprocessing",
        "splits",
        "recommendation_evaluation",
        "candidate_pool",
        "retrieval",
        "statistics",
    )
    changed = [key for key in frozen_keys if config[key] != frozen_source[key]]
    if changed:
        raise ValueError(f"Current configuration changed frozen Stage 6 fields: {changed}")
    reference_path, reference_hash = _manifest_output(source_manifest, "case_metrics.csv")
    reference = pd.read_csv(reference_path)

    items, items_path, items_hash = _active_items(config)
    cases = attach_candidate_pools(items, build_evaluation_cases(items, config), config)
    expected = int(config["recommendation_evaluation"]["case_count"])
    reference_case_ids = set(
        reference[reference["method"].eq("fused_clip_main_040_060")]["case_id"]
    )
    if len(cases) != expected or set(cases["case_id"]) != reference_case_ids:
        raise ValueError("Rebuilt cases do not match the frozen Stage 6 reference cases.")
    item_lookup = items.set_index("item_id", drop=False)
    candidate_ids = sorted({item for values in cases["candidate_item_ids"] for item in values})
    candidate_rows = item_lookup.loc[candidate_ids].reset_index(drop=True)
    query_rows = item_lookup.loc[cases["query_item_id"].tolist()].reset_index(drop=True)
    raw_split, dataset_fingerprint = load_pinned_split(config)
    embedder = CLIPEmbedder(model)
    batch_size = int(model["batch_size"])

    candidate_texts = (
        candidate_rows["category"].astype(str) + " | " + candidate_rows["text"].astype(str)
    ).tolist()
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
    candidate_text = embedder.encode_text(candidate_texts, batch_size=batch_size)
    candidate_image = _item_images(candidate_rows, raw_split, embedder, batch_size)
    query_text = embedder.encode_text(query_texts, batch_size=batch_size)
    query_image = _item_images(query_rows, raw_split, embedder, batch_size)
    fusion = config["retrieval"]["fusion"]
    candidate_fused = fuse_clip_embeddings(
        candidate_image,
        candidate_text,
        image_weight=float(fusion["image_weight"]),
        text_weight=float(fusion["text_weight"]),
    )
    query_fused = fuse_clip_embeddings(
        query_image,
        query_text,
        image_weight=float(fusion["image_weight"]),
        text_weight=float(fusion["text_weight"]),
    )
    if not all(np.isfinite(array).all() for array in (candidate_fused, query_fused)):
        raise ValueError("FashionCLIP produced non-finite embeddings.")

    candidate_index = {item_id: index for index, item_id in enumerate(candidate_ids)}
    metric_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    score_standard_deviations: list[float] = []
    for case_index, case in enumerate(cases.to_dict("records")):
        indices = [candidate_index[item] for item in case["candidate_item_ids"]]
        score_sets = {
            "fashionclip_image": cosine_scores(
                query_image[case_index], candidate_image[indices]
            ),
            "fashionclip_text": cosine_scores(query_text[case_index], candidate_text[indices]),
            "fashionclip_fused_040_060": cosine_scores(
                query_fused[case_index], candidate_fused[indices]
            ),
        }
        for method, scores in score_sets.items():
            score_standard_deviations.append(float(np.std(scores)))
            ranking = _rank(case["candidate_item_ids"], case["candidate_relevance"], scores)
            metric_rows.append(
                {
                    "case_id": case["case_id"],
                    "query_outfit_id": case["query_outfit_id"],
                    "target_category": case["target_category"],
                    "target_accessory_subcategory": case.get(
                        "target_accessory_subcategory", ""
                    ),
                    "actual_pool_size": len(case["candidate_item_ids"]),
                    "method": method,
                    **ranking_metrics(ranking["is_positive"]),
                }
            )
            ranking_rows.append(
                {
                    "case_id": case["case_id"],
                    "method": method,
                    "top10": ranking["item_id"].head(10).tolist(),
                    "top10_scores": ranking["score"].head(10).astype(float).tolist(),
                }
            )
    if min(score_standard_deviations) <= 1e-8:
        raise ValueError("FashionCLIP produced a degenerate candidate score vector.")

    case_metrics = pd.DataFrame(metric_rows)
    stats = config["statistics"]
    results = aggregate_recommendation_metrics(
        case_metrics,
        metric_columns=METRICS,
        replicates=int(stats["bootstrap_replicates"]),
        confidence_level=float(stats["confidence_level"]),
        seed=int(config["project"]["random_seed"]) + 800,
    )
    contrasts = _contrasts(case_metrics, reference, config)

    run_dir.mkdir(parents=True)
    runtime_outputs = {
        "case_metrics": run_dir / "case_metrics.csv",
        "candidate_rankings": run_dir / "candidate_rankings.jsonl",
        "results": run_dir / "fashionclip_results.csv",
        "contrasts": run_dir / "fashionclip_contrasts.csv",
    }
    case_metrics.to_csv(runtime_outputs["case_metrics"], index=False)
    with runtime_outputs["candidate_rankings"].open("w", encoding="utf-8", newline="\n") as handle:
        for row in ranking_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    results.to_csv(runtime_outputs["results"], index=False)
    contrasts.to_csv(runtime_outputs["contrasts"], index=False)
    tracked = {
        "table_stage6b_fashionclip_results": Path(
            "artifacts/tables/table_stage6b_fashionclip_results.csv"
        ),
        "table_stage6b_fashionclip_contrasts": Path(
            "artifacts/tables/table_stage6b_fashionclip_contrasts.csv"
        ),
    }
    results.to_csv(tracked["table_stage6b_fashionclip_results"], index=False)
    contrasts.to_csv(tracked["table_stage6b_fashionclip_contrasts"], index=False)
    _update_registry(digest, tracked)
    registry = Path("artifacts/manifests/figure_table_registry.csv")
    outputs = [*runtime_outputs.values(), *tracked.values(), registry]
    manifest = {
        "schema_version": 1,
        "stage": "6b",
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": digest,
        "analysis_role": "additive_stronger_domain_baseline_not_used_for_selection",
        "model": baseline["fashionclip"],
        "input_artifact_hashes": {
            str(args.config): sha256_file(args.config),
            str(args.baseline_config): sha256_file(args.baseline_config),
            str(args.source_manifest): sha256_file(args.source_manifest),
            str(reference_path): reference_hash,
            str(items_path): items_hash,
        },
        "output_artifact_hashes": {str(path): sha256_file(path) for path in outputs},
        "row_counts": {
            "cases": expected,
            "case_method_rows": len(case_metrics),
            "rankings": len(ranking_rows),
        },
        "integrity_checks": {
            "same_frozen_case_ids": True,
            "same_candidate_protocol": True,
            "finite_embeddings": True,
            "nondegenerate_scores": True,
            "fashionclip_embedding_dimension": int(candidate_fused.shape[1]),
            "minimum_within_case_score_sd": min(score_standard_deviations),
        },
        "dataset_fingerprint": dataset_fingerprint,
        "environment": environment_summary(),
        "status": "complete_additive_baseline",
    }
    write_new_json(run_dir / "manifest.json", manifest)
    write_json(Path("artifacts/manifests/stage6b_fashionclip_manifest.json"), manifest)
    print(json.dumps({"run_id": run_id, "results": results.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
