"""Run the frozen 1,000-case final recommendation evaluation and lock explanation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from evidence_fashion.final_contracts import (
    build_rule_rag_evidence_packet,
    canonical_json_sha256,
    reproduce_evidence_score,
)
from evidence_fashion.manifest import (
    configuration_hash,
    environment_summary,
    git_commit,
    load_resolved_configuration,
    sha256_file,
    utc_timestamp,
    write_new_json,
)
from evidence_fashion.reranking import ranking_metrics, rerank_candidates
from evidence_fashion.retrieval import OllamaEmbedder, cosine_scores, fuse_clip_embeddings
from evidence_fashion.rule_retrieval import RuleRetriever, candidate_rule_representation

BASELINES = ("minilm_text", "clip_image", "clip_text", "fused_clip", "evidence_rerank")


def _manifest_output(manifest: dict[str, Any], suffix: str) -> Path:
    return Path(next(path for path in manifest["output_artifact_hashes"] if path.endswith(suffix)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _hash_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _ranked_ids(ids: list[str], scores: np.ndarray) -> list[str]:
    return (
        pd.DataFrame({"item_id": ids, "score": scores})
        .sort_values(["score", "item_id"], ascending=[False, True], kind="stable")["item_id"]
        .astype(str)
        .tolist()
    )


def _summarise_metrics(records: list[dict[str, Any]]) -> pd.DataFrame:
    metric_names = (
        "hr_at_1",
        "hr_at_5",
        "hr_at_10",
        "ndcg_at_1",
        "ndcg_at_5",
        "ndcg_at_10",
        "mrr",
    )
    frame = pd.DataFrame(records)
    rows = []
    for method in BASELINES:
        subset = frame[frame["method"].eq(method)]
        groups = [("micro", "all", subset)]
        groups.extend(
            ("category", str(category), values)
            for category, values in subset.groupby("target_category")
        )
        for aggregation, label, values in groups:
            for metric in metric_names:
                rows.append(
                    {
                        "method": method,
                        "aggregation": aggregation,
                        "target_category": label,
                        "metric": metric,
                        "estimate": float(values[metric].mean()),
                        "cases": len(values),
                    }
                )
        category_means = subset.groupby("target_category")[list(metric_names)].mean()
        for metric in metric_names:
            rows.append(
                {
                    "method": method,
                    "aggregation": "macro",
                    "target_category": "all",
                    "metric": metric,
                    "estimate": float(category_means[metric].mean()),
                    "cases": len(category_means),
                }
            )
    return pd.DataFrame(rows)


def _minimal_item_identity(row: dict[str, Any]) -> str:
    return f"{row['category']} | {row['text']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts-config", type=Path, default=Path("configs/prompts.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    prompts = yaml.safe_load(args.prompts_config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    resolved["prompts"] = prompts
    config_hash = configuration_hash(resolved)
    data_manifest_path = Path(config["paths"]["active_data_manifest"])
    embedding_manifest_path = Path(config["paths"]["active_embedding_manifest"])
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    items = pd.read_parquet(_manifest_output(data_manifest, "prepared_items.parquet"))
    cases = _read_jsonl(_manifest_output(data_manifest, "evaluation_cases.jsonl"))
    if len(cases) != int(config["recommendation_evaluation"]["case_count"]):
        raise ValueError("Final recommendation cases do not match the frozen quota.")
    category_counts = Counter(str(case["target_category"]) for case in cases)
    expected_category_count = int(config["recommendation_evaluation"]["cases_per_category"])
    if any(count != expected_category_count for count in category_counts.values()):
        raise ValueError(f"Final recommendation category quotas are invalid: {category_counts}")
    metadata = _read_jsonl(_manifest_output(embedding_manifest, "sample_metadata.jsonl"))
    if items["item_id"].astype(str).tolist() != [str(row["item_id"]) for row in metadata]:
        raise ValueError("Embedding metadata is not aligned to the final prepared item table.")
    arrays = {
        name: np.load(_manifest_output(embedding_manifest, f"{name}.npy"), allow_pickle=False)
        for name in ("minilm_text", "clip_image", "clip_text")
    }
    kb_manifest_paths = sorted(
        Path(config["paths"]["embedding_runs"]).glob("final-kb-*/manifest.json")
    )
    if len(kb_manifest_paths) != 1:
        raise ValueError("Expected exactly one frozen final KB embedding run.")
    kb_manifest = json.loads(kb_manifest_paths[0].read_text(encoding="utf-8"))
    rules = pd.DataFrame(_read_jsonl(_manifest_output(kb_manifest, "rule_metadata.jsonl")))
    rule_vectors = np.load(_manifest_output(kb_manifest, "rule_embeddings.npy"), allow_pickle=False)
    run_dir = (
        Path(config["paths"]["recommendation_runs"]) / f"final-recommendations-{config_hash[:12]}"
    )
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite a frozen recommendation run: {run_dir}")
    run_dir.mkdir(parents=True)
    item_index = {str(item_id): index for index, item_id in enumerate(items["item_id"].astype(str))}
    item_lookup = items.set_index("item_id", drop=False)
    retriever = RuleRetriever(rules, rule_vectors, config["rule_retrieval"])
    embedder = OllamaEmbedder(models["embedders"]["qwen3_embedding"])
    metric_records: list[dict[str, Any]] = []
    ranking_records: list[dict[str, Any]] = []
    locked_records: list[dict[str, Any]] = []
    rule_usage = Counter()
    trace_sizes = []
    diagnostics = []
    for position, case in enumerate(cases, start=1):
        candidate_ids = [str(value) for value in case["candidate_item_ids"]]
        candidates = item_lookup.loc[candidate_ids].reset_index(drop=True)
        candidate_rows = candidates.to_dict("records")
        candidate_indices = [item_index[item_id] for item_id in candidate_ids]
        query_index = item_index[str(case["query_item_id"])]
        scores = {
            "minilm_text": cosine_scores(
                arrays["minilm_text"][query_index], arrays["minilm_text"][candidate_indices]
            ),
            "clip_image": cosine_scores(
                arrays["clip_image"][query_index], arrays["clip_image"][candidate_indices]
            ),
            "clip_text": cosine_scores(
                arrays["clip_text"][query_index], arrays["clip_text"][candidate_indices]
            ),
        }
        query_fused = fuse_clip_embeddings(
            arrays["clip_image"][[query_index]],
            arrays["clip_text"][[query_index]],
            image_weight=float(config["retrieval"]["fusion"]["image_weight"]),
            text_weight=float(config["retrieval"]["fusion"]["text_weight"]),
        )[0]
        candidate_fused = fuse_clip_embeddings(
            arrays["clip_image"][candidate_indices],
            arrays["clip_text"][candidate_indices],
            image_weight=float(config["retrieval"]["fusion"]["image_weight"]),
            text_weight=float(config["retrieval"]["fusion"]["text_weight"]),
        )
        scores["fused_clip"] = cosine_scores(query_fused, candidate_fused)
        representations = [candidate_rule_representation(case, row) for row in candidate_rows]
        representation_vectors = embedder.encode(
            representations, batch_size=int(config["embeddings"]["batch_size"])
        )
        traces = [
            retriever.retrieve_and_score(
                case=case,
                candidate=row,
                representation_embedding=vector,
                top_k=int(config["rule_retrieval"]["rule_top_k"]),
            )
            for row, vector in zip(candidate_rows, representation_vectors, strict=True)
        ]
        reranked = rerank_candidates(
            pd.DataFrame(
                {
                    "item_id": candidate_ids,
                    "clip_score": scores["fused_clip"],
                    "evidence_score": [trace.evidence_score for trace in traces],
                }
            ),
            clip_weight=float(config["reranking"]["clip_weight"]),
            evidence_weight=float(config["reranking"]["evidence_weight"]),
        )
        trace_by_id = {trace.candidate_id: trace.to_dict() for trace in traces}
        positive_ids = set(str(value) for value in case["positive_item_ids"])
        baseline_rankings = {
            method: _ranked_ids(candidate_ids, method_scores)
            for method, method_scores in scores.items()
        }
        baseline_rankings["evidence_rerank"] = reranked["item_id"].astype(str).tolist()
        for method, ranked_ids in baseline_rankings.items():
            metric_records.append(
                {
                    "case_id": case["case_id"],
                    "query_outfit_id": case["query_outfit_id"],
                    "target_category": case["target_category"],
                    "method": method,
                    **ranking_metrics([item_id in positive_ids for item_id in ranked_ids]),
                }
            )
        rerank_by_id = {str(row["item_id"]): row for row in reranked.to_dict("records")}
        candidate_details = []
        for candidate, candidate_id, index in zip(
            candidate_rows, candidate_ids, range(len(candidate_ids)), strict=True
        ):
            rerank = rerank_by_id[candidate_id]
            trace = trace_by_id[candidate_id]
            reproduce_evidence_score(trace, config["rule_retrieval"])
            candidate_details.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_category": candidate["category"],
                    "candidate_text": candidate["text"],
                    "is_positive": candidate_id in positive_ids,
                    "minilm_text_score": float(scores["minilm_text"][index]),
                    "clip_image_score": float(scores["clip_image"][index]),
                    "clip_text_score": float(scores["clip_text"][index]),
                    "original_clip_score": float(scores["fused_clip"][index]),
                    "normalized_clip_score": float(rerank["normalized_clip_score"]),
                    "raw_evidence_score": float(rerank["evidence_score"]),
                    "normalized_evidence_score": float(rerank["normalized_evidence_score"]),
                    "final_score": float(rerank["final_score"]),
                    "pre_rerank_rank": int(rerank["pre_rerank_rank"]),
                    "post_rerank_rank": int(rerank["post_rerank_rank"]),
                    "exact_stored_rule_trace": trace,
                    "trace_hash": canonical_json_sha256(trace),
                }
            )
        top = reranked.iloc[0]
        locked_id = str(top["item_id"])
        locked_candidate = next(row for row in candidate_rows if str(row["item_id"]) == locked_id)
        locked_trace = trace_by_id[locked_id]
        rule_packet, trace_hash = build_rule_rag_evidence_packet(locked_trace)
        if trace_hash != canonical_json_sha256(locked_trace):
            raise ValueError("Rule-RAG packet hash differs from the stored locked trace hash.")
        trace_rule_ids = [str(rule["rule_id"]) for rule in locked_trace["rules"]]
        trace_sizes.append(len(trace_rule_ids))
        rule_usage.update(trace_rule_ids)
        fused_top = baseline_rankings["fused_clip"][0]
        pre_evidence = trace_by_id[fused_top]["evidence_score"]
        diagnostics.append(
            {
                "case_id": case["case_id"],
                "target_category": case["target_category"],
                "top1_changed": locked_id != fused_top,
                "top5_overlap": len(
                    set(baseline_rankings["fused_clip"][:5])
                    & set(baseline_rankings["evidence_rerank"][:5])
                ),
                "top1_evidence_score_gain": float(locked_trace["evidence_score"])
                - float(pre_evidence),
                "pre_to_post_rank_shift": int(top["pre_rerank_rank"])
                - int(top["post_rerank_rank"]),
                "locked_trace_size": len(trace_rule_ids),
            }
        )
        ranking_records.append(
            {
                "case": case,
                "candidates": candidate_details,
                "ranked_candidate_ids": baseline_rankings,
            }
        )
        locked_records.append(
            {
                "case_id": case["case_id"],
                "target_category": case["target_category"],
                "query_item_id": case["query_item_id"],
                "locked_candidate_id": locked_id,
                "locked_candidate_minimal_name": _minimal_item_identity(locked_candidate),
                "top1_evidence_score": float(locked_trace["evidence_score"]),
                "exact_stored_rule_trace": locked_trace,
                "trace_hash": trace_hash,
            }
        )
        if position % 25 == 0:
            print(f"final recommendations: {position}/{len(cases)}", flush=True)
    metric_table = _summarise_metrics(metric_records)
    selected: list[dict[str, Any]] = []
    required = int(config["explanations"]["cases_per_category"])
    for category in config["recommendation_evaluation"]["target_category_order"]:
        eligible = [
            record
            for record in locked_records
            if record["target_category"] == category and record["exact_stored_rule_trace"]["rules"]
        ]
        eligible.sort(
            key=lambda record: (
                _hash_order(str(record["case_id"]), int(config["explanations"]["selection_seed"])),
                record["case_id"],
            )
        )
        if len(eligible) < required:
            raise ValueError(
                f"Only {len(eligible)} evidence-eligible {category} cases; {required} required."
            )
        selected.extend(eligible[:required])
    case_by_id = {str(case["case_id"]): case for case in cases}
    explanation_cases = []
    for locked in selected:
        case = case_by_id[str(locked["case_id"])]
        query = item_lookup.loc[str(case["query_item_id"])].to_dict()
        common_a = {
            "user_request": str(case["user_request"]),
            "query_item_minimal_name": _minimal_item_identity(query),
            "locked_item_minimal_name": str(locked["locked_candidate_minimal_name"]),
        }
        evidence_packet, trace_hash = build_rule_rag_evidence_packet(
            locked["exact_stored_rule_trace"]
        )
        explanation_cases.append(
            {
                "case_id": locked["case_id"],
                "target_category": locked["target_category"],
                "query_outfit_id": case["query_outfit_id"],
                "locked_candidate_id": locked["locked_candidate_id"],
                "common_context_A": common_a,
                "common_context_A_hash": canonical_json_sha256(common_a),
                "exact_stored_rule_trace_B": evidence_packet,
                "trace_hash": trace_hash,
                "no_rag_input": {"common_context_A": common_a},
                "rule_rag_input": {
                    "common_context_A": common_a,
                    "exact_stored_rule_trace_B": evidence_packet,
                },
            }
        )
    selection_counts = Counter(record["target_category"] for record in explanation_cases)
    if len(explanation_cases) != int(config["explanations"]["case_count"]) or any(
        selection_counts[category] != required
        for category in config["recommendation_evaluation"]["target_category_order"]
    ):
        raise ValueError("Final explanation selection does not match its frozen quota.")
    ranking_path = run_dir / "candidate_rankings.jsonl"
    locked_path = run_dir / "locked_recommendations.jsonl"
    explanation_path = run_dir / "explanation_cases.jsonl"
    metrics_path = run_dir / "recommendation_metrics.csv"
    diagnostics_path = run_dir / "evidence_participation_diagnostics.json"
    _write_jsonl(ranking_path, ranking_records)
    _write_jsonl(locked_path, locked_records)
    _write_jsonl(explanation_path, explanation_cases)
    metric_table.to_csv(metrics_path, index=False, lineterminator="\n")
    diagnostics_payload = {
        "top1_change_rate": float(pd.DataFrame(diagnostics)["top1_changed"].mean()),
        "mean_top5_overlap": float(pd.DataFrame(diagnostics)["top5_overlap"].mean()),
        "mean_top1_evidence_score_gain": float(
            pd.DataFrame(diagnostics)["top1_evidence_score_gain"].mean()
        ),
        "mean_pre_to_post_rank_shift": float(
            pd.DataFrame(diagnostics)["pre_to_post_rank_shift"].mean()
        ),
        "locked_trace_size_distribution": dict(sorted(Counter(trace_sizes).items())),
        "rules_used_at_least_once": len(rule_usage),
        "rule_usage": dict(sorted(rule_usage.items())),
        "diagnostic_cases": diagnostics,
    }
    diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": 2,
        "stage": 2,
        "stage_name": "final_recommendations_and_explanation_input_lock",
        "status": "frozen_recommendations_complete",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": config_hash,
        "input_artifact_hashes": {
            str(data_manifest_path): sha256_file(data_manifest_path),
            str(embedding_manifest_path): sha256_file(embedding_manifest_path),
            str(kb_manifest_paths[0]): sha256_file(kb_manifest_paths[0]),
        },
        "output_artifact_hashes": {
            str(ranking_path): sha256_file(ranking_path),
            str(locked_path): sha256_file(locked_path),
            str(explanation_path): sha256_file(explanation_path),
            str(metrics_path): sha256_file(metrics_path),
            str(diagnostics_path): sha256_file(diagnostics_path),
        },
        "models": {
            "minilm": models["embedders"]["minilm"],
            "clip": models["embedders"]["clip"],
            "qwen3_embedding": models["embedders"]["qwen3_embedding"],
        },
        "row_counts": {
            "recommendation_cases": len(cases),
            "candidate_rows": sum(len(record["candidates"]) for record in ranking_records),
            "locked_recommendations": len(locked_records),
            "explanation_cases": len(explanation_cases),
            "explanation_cases_by_category": dict(sorted(selection_counts.items())),
            "metric_rows": len(metric_table),
        },
        "failure_counts": {"recommendation_terminal_failures": 0},
        "seeds": {
            "case_selection": config["recommendation_evaluation"]["case_seed"],
            "explanation_selection": config["explanations"]["selection_seed"],
        },
        "fixed_defaults": {
            "image_weight": config["retrieval"]["fusion"]["image_weight"],
            "text_weight": config["retrieval"]["fusion"]["text_weight"],
            "clip_weight": config["reranking"]["clip_weight"],
            "evidence_weight": config["reranking"]["evidence_weight"],
            "rule_top_k": config["rule_retrieval"]["rule_top_k"],
        },
        "trace_invariant": {
            "stored_trace_reproduces_evidence_score": True,
            "rule_rag_B_hash_equals_locked_trace_hash": True,
            "second_retrieval_after_locking": False,
        },
        "environment": environment_summary(),
    }
    write_new_json(manifest_path, manifest)
    embedder = None
    print(json.dumps(manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
