"""Run deterministic Stage 5 using only the frozen final Stage 1–4 artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from evidence_fashion.final_analysis import (
    METRICS,
    bootstrap_paired_difference,
    holm_adjust,
    paired_complete_rows,
    record_metrics,
)
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def locate(manifest: dict[str, Any], suffix: str) -> Path:
    path = Path(next(name for name in manifest["output_artifact_hashes"] if name.endswith(suffix)))
    if sha256_file(path) != manifest["output_artifact_hashes"][str(path)]:
        raise ValueError(f"Frozen input hash mismatch: {path}")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot create empty required source table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rec_case_metrics(rankings: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    methods = ("minilm_text", "clip_image", "clip_text", "fused_clip", "evidence_rerank")
    for row in rankings:
        positives = {
            candidate_id
            for candidate_id, relevant in zip(
                row["case"]["candidate_item_ids"], row["case"]["candidate_relevance"], strict=True
            )
            if relevant
        }
        for method in methods:
            ranked = row["ranked_candidate_ids"][method]
            rank = next(
                (index + 1 for index, candidate in enumerate(ranked) if candidate in positives),
                None,
            )
            metric = {
                "hr_at_1": float(rank == 1),
                "hr_at_5": float(rank is not None and rank <= 5),
                "hr_at_10": float(rank is not None and rank <= 10),
                "ndcg_at_1": float(rank == 1),
                "ndcg_at_5": 1 / np.log2(rank + 1) if rank is not None and rank <= 5 else 0.0,
                "ndcg_at_10": 1 / np.log2(rank + 1) if rank is not None and rank <= 10 else 0.0,
                "mrr": 1 / rank if rank is not None else 0.0,
            }
            rows.append(
                {
                    "case_id": row["case"]["case_id"],
                    "query_outfit_id": row["case"]["query_outfit_id"],
                    "target_category": row["case"]["target_category"],
                    "method": method,
                    **metric,
                }
            )
    return pd.DataFrame(rows)


def save_figure(path: Path, contrasts: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in contrasts if row["scope"] == "overall"]
    labels = [row["metric"] for row in selected]
    values = [row["estimate"] * 100 for row in selected]
    errors = [
        [(row["estimate"] - row["ci_lower"]) * 100 for row in selected],
        [(row["ci_upper"] - row["estimate"]) * 100 for row in selected],
    ]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(labels, values, color="#0072B2", yerr=errors, capsize=4)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Rule-RAG − No-RAG (percentage points)")
    axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".svg"))
    figure.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(figure)


def main() -> None:
    experiment = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
    prompts = yaml.safe_load(Path("configs/prompts.yaml").read_text(encoding="utf-8"))
    stage1_path = Path("artifacts/manifests/final_stage1_preflight_manifest.json")
    stage2_path = Path("artifacts/manifests/final_stage2_manifest.json")
    stage3_path = Path("artifacts/manifests/final_stage3_manifest.json")
    stage4_path = Path("artifacts/manifests/final_stage4_manifest.json")
    stage1, stage2, stage3, stage4 = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (stage1_path, stage2_path, stage3_path, stage4_path)
    ]
    rec_manifest_path = Path(
        next(path for path in stage2["input_manifest_hashes"] if "recommendations" in path)
    )
    exp_manifest_path = Path(
        next(path for path in stage2["input_manifest_hashes"] if "explanations" in path)
    )
    rec_manifest = json.loads(rec_manifest_path.read_text(encoding="utf-8"))
    exp_manifest = json.loads(exp_manifest_path.read_text(encoding="utf-8"))
    ext_manifest = json.loads(Path(stage3["stage3_manifest"]["path"]).read_text(encoding="utf-8"))
    ver_manifest = json.loads(Path(stage4["stage4_manifest"]["path"]).read_text(encoding="utf-8"))
    paths = {
        "rankings": locate(rec_manifest, "candidate_rankings.jsonl"),
        "cases": locate(rec_manifest, "explanation_cases.jsonl"),
        "explanations": locate(exp_manifest, "explanations.jsonl"),
        "extractions": locate(ext_manifest, "extractions.jsonl"),
        "verifications": locate(ver_manifest, "verifications.jsonl"),
        "recommendation_metrics": locate(rec_manifest, "recommendation_metrics.csv"),
    }

    amendment = {
        "amended_at_utc": utc_timestamp(),
        "reason": "authorized_stage4_verifier_contract_correction",
        "original_stage1_prompts_config_sha256": stage1["configuration_file_hashes"][
            "configs\\prompts.yaml"
        ],
        "actual_final_prompts_config_sha256": sha256_file(Path("configs/prompts.yaml")),
        "stage4_prompt_hashes": ver_manifest["prompt_hashes"],
        "stage4_configuration_hash": ver_manifest["configuration_hash"],
        "stage4_manifest_sha256": sha256_file(Path(stage4["stage4_manifest"]["path"])),
    }
    stage1["post_stage1_provenance_amendment"] = amendment
    write_json(stage1_path, stage1)

    cases = {row["case_id"]: row for row in read_jsonl(paths["cases"])}
    explanations = {
        (row["case_id"], row["generator_model_id"], row["condition"]): row
        for row in read_jsonl(paths["explanations"])
        if row["status"] == "accepted"
    }
    verifications = [
        row for row in read_jsonl(paths["verifications"]) if row["status"] == "accepted"
    ]
    analysis_rows = []
    for verification in verifications:
        key = (
            verification["case_id"],
            verification["generator_model_id"],
            verification["condition"],
        )
        explanation = explanations[key]
        trace_size = len(cases[verification["case_id"]]["exact_stored_rule_trace_B"]["rules"])
        analysis_rows.append(
            {
                "case_id": verification["case_id"],
                "generator_model_id": verification["generator_model_id"],
                "condition": verification["condition"],
                "target_category": verification["target_category"],
                "trace_size": trace_size,
                "word_count": len(explanation["explanation"].split()),
                **record_metrics(verification, explanation),
            }
        )
    pairs = paired_complete_rows(analysis_rows)
    required_pairs = {
        "gemma4:12b": 474,
        "llama3.1:8b-instruct-q8_0": 438,
        "ministral-3:14b-instruct-2512-q4_K_M": 456,
    }
    observed_pairs = {
        model: len(paired_complete_rows(analysis_rows, model)) for model in required_pairs
    }
    if observed_pairs != required_pairs:
        raise ValueError(f"Generator-specific complete pairs changed: {observed_pairs}")
    settings = experiment["statistics"]
    contrasts = []
    scopes = [("overall", "all", pairs, True)] + [
        ("generator", model, paired_complete_rows(analysis_rows, model), False)
        for model in required_pairs
    ]
    for category in sorted({row["target_category"] for row in analysis_rows}):
        scopes.append(
            (
                "category",
                category,
                [pair for pair in pairs if pair["no_rag"]["target_category"] == category],
                True,
            )
        )
    for scope, value, scoped_pairs, aggregate_cases in scopes:
        for metric_index, metric in enumerate(METRICS):
            result = bootstrap_paired_difference(
                scoped_pairs,
                metric,
                replicates=int(settings["bootstrap_replicates"]),
                confidence_level=float(settings["confidence_level"]),
                seed=int(settings["bootstrap_seed"]) + metric_index,
                aggregate_generators_by_case=aggregate_cases,
            )
            contrasts.append(
                {
                    "scope": scope,
                    "scope_value": value,
                    "metric": metric,
                    "paired_cases": result["n"],
                    "estimate": result["estimate"],
                    "ci_lower": result["ci_lower"],
                    "ci_upper": result["ci_upper"],
                    "p_value": result["p_value"],
                    "holm_p_value": None,
                }
            )
    overall = [row for row in contrasts if row["scope"] == "overall"]
    for row, adjusted in zip(overall, holm_adjust(row["p_value"] for row in overall), strict=True):
        row["holm_p_value"] = adjusted

    failures = []
    for stage, path in (
        ("stage2", paths["explanations"]),
        ("stage3", paths["extractions"]),
        ("stage4", paths["verifications"]),
    ):
        for row in read_jsonl(path):
            if row["status"] != "accepted":
                failures.append(
                    {
                        "stage": stage,
                        "generator_model_id": row["generator_model_id"],
                        "condition": row["condition"],
                        "count": 1,
                    }
                )
    failure_rows = [
        {
            "stage": stage,
            "generator_model_id": model,
            "condition": condition,
            "terminal_failures": count,
        }
        for (stage, model, condition), count in sorted(
            Counter(
                (row["stage"], row["generator_model_id"], row["condition"]) for row in failures
            ).items()
        )
    ]
    output = Path(experiment["paths"]["final_analysis_runs"]) / "final-stage5"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Stage-5 output: {output}")
    output.mkdir(parents=True)
    tables = Path(experiment["paths"]["tables"])
    figures = Path(experiment["paths"]["figures"])
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metric_path = output / "explanation_record_metrics.csv"
    contrast_path = output / "explanation_paired_contrasts.csv"
    failure_path = output / "terminal_failures.csv"
    write_csv(metric_path, analysis_rows)
    write_csv(contrast_path, contrasts)
    write_csv(failure_path, failure_rows)
    rec_metrics = rec_case_metrics(read_jsonl(paths["rankings"]))
    from evidence_fashion.evaluation.recommendation import aggregate_recommendation_metrics

    rec_summary = aggregate_recommendation_metrics(
        rec_metrics,
        metric_columns=[
            "hr_at_1",
            "hr_at_5",
            "hr_at_10",
            "ndcg_at_1",
            "ndcg_at_5",
            "ndcg_at_10",
            "mrr",
        ],
        replicates=int(settings["bootstrap_replicates"]),
        confidence_level=float(settings["confidence_level"]),
        seed=int(settings["bootstrap_seed"]),
    )
    rec_path = output / "recommendation_metrics_with_ci.csv"
    rec_summary.to_csv(rec_path, index=False)
    save_figure(figures / "final_explanation_paired_contrasts", contrasts)
    qualitative = sorted(analysis_rows, key=lambda row: (row["trace_support_rate"], row["case_id"]))
    write_json(
        output / "qualitative_examples.json",
        {
            "lowest_trace_support": qualitative[:6],
            "highest_trace_support": qualitative[-6:],
            "selection": "deterministic trace-support ordering",
        },
    )
    release = Path(experiment["paths"]["release"])
    release.mkdir(parents=True, exist_ok=True)
    for name, source in {
        "candidate_rankings.jsonl": paths["rankings"],
        "explanation_cases.jsonl": paths["cases"],
        "explanations.jsonl": paths["explanations"],
        "extractions.jsonl": paths["extractions"],
        "verifications.jsonl": paths["verifications"],
        "explanation_record_metrics.csv": metric_path,
        "explanation_paired_contrasts.csv": contrast_path,
        "recommendation_metrics_with_ci.csv": rec_path,
        "terminal_failures.csv": failure_path,
    }.items():
        shutil.copy2(source, release / name)
    for source in (
        stage1_path,
        stage2_path,
        stage3_path,
        stage4_path,
        Path("configs/experiment.yaml"),
        Path("configs/models.yaml"),
        Path("configs/prompts.yaml"),
    ):
        shutil.copy2(source, release / source.name)
    source_hashes = {
        str(path): sha256_file(path)
        for path in (
            metric_path,
            contrast_path,
            failure_path,
            rec_path,
            output / "qualitative_examples.json",
        )
    }
    manifest = {
        "schema_version": 1,
        "stage": 5,
        "stage_name": "final_results_and_release",
        "status": "complete",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": configuration_hash(
            {
                **load_resolved_configuration(
                    Path("configs/experiment.yaml"), Path("configs/models.yaml")
                ),
                "prompts": prompts,
            }
        ),
        "input_manifests": {
            str(path): sha256_file(path)
            for path in (stage1_path, stage2_path, stage3_path, stage4_path)
        },
        "output_artifact_hashes": source_hashes,
        "pairing": {"policy": "generator_specific_complete_case_only", "counts": observed_pairs},
        "statistics": settings,
        "provenance_amendment": amendment,
        "environment": environment_summary(),
    }
    write_new_json(output / "manifest.json", manifest)
    write_json(
        Path("artifacts/manifests/final_stage5_manifest.json"),
        {
            **manifest,
            "stage5_manifest_path": str(output / "manifest.json"),
            "stage5_manifest_sha256": sha256_file(output / "manifest.json"),
        },
    )


if __name__ == "__main__":
    main()
