"""Run only the pre-registered Stage-1 validation sensitivity analyses."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from evidence_fashion.data import attach_candidate_pools, build_evaluation_cases
from evidence_fashion.manifest import sha256_file, write_new_json
from evidence_fashion.reranking import pareto_frontier, ranking_metrics, rerank_candidates
from evidence_fashion.retrieval import OllamaEmbedder, cosine_scores, fuse_clip_embeddings
from evidence_fashion.rule_retrieval import (
    RuleRetriever,
    candidate_rule_representation,
    truncate_trace,
)


def _manifest_output(manifest: dict[str, Any], suffix: str) -> Path:
    return Path(next(path for path in manifest["output_artifact_hashes"] if path.endswith(suffix)))


def _validation_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    settings = result["validation_sensitivity"]
    result["recommendation_evaluation"].update(
        {
            "case_count": int(settings["case_count"]),
            "case_split": str(settings["split"]),
            "case_seed": int(settings["case_seed"]),
            "cases_per_category": int(settings["cases_per_category"]),
        }
    )
    return result


def _append_metrics(
    accumulator: dict[tuple[float, int] | float, list[dict[str, Any]]],
    key: tuple[float, int] | float,
    relevance: list[bool],
    *,
    category: str,
    diagnostics: dict[str, float] | None = None,
) -> None:
    accumulator[key].append(
        {"target_category": category, **ranking_metrics(relevance), **(diagnostics or {})}
    )


def _aggregate(records: list[dict[str, Any]], fields: list[str]) -> dict[str, float]:
    frame = pd.DataFrame(records)
    values = {field: float(frame[field].mean()) for field in fields}
    values["cases"] = int(len(frame))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    data_manifest = json.loads(Path(config["paths"]["active_data_manifest"]).read_text("utf-8"))
    embedding_manifest = json.loads(
        Path(config["paths"]["active_embedding_manifest"]).read_text("utf-8")
    )
    items = pd.read_parquet(_manifest_output(data_manifest, "prepared_items.parquet"))
    metadata = pd.DataFrame(
        json.loads(line)
        for line in _manifest_output(embedding_manifest, "sample_metadata.jsonl").read_text(
            "utf-8"
        ).splitlines()
    )
    if items["item_id"].astype(str).tolist() != metadata["item_id"].astype(str).tolist():
        raise ValueError("Full embedding metadata is not aligned to final prepared items.")
    arrays = {
        name: np.load(_manifest_output(embedding_manifest, f"{name}.npy"), allow_pickle=False)
        for name in ("clip_image", "clip_text")
    }
    kb_manifest_paths = sorted(
        Path(config["paths"]["embedding_runs"]).glob("final-kb-*/manifest.json")
    )
    if len(kb_manifest_paths) != 1:
        raise ValueError("Expected exactly one final KB embedding manifest.")
    kb_manifest = json.loads(kb_manifest_paths[0].read_text("utf-8"))
    rules = pd.DataFrame(
        json.loads(line)
        for line in _manifest_output(kb_manifest, "rule_metadata.jsonl")
        .read_text("utf-8")
        .splitlines()
    )
    rule_vectors = np.load(_manifest_output(kb_manifest, "rule_embeddings.npy"), allow_pickle=False)
    validation_config = _validation_config(config)
    cases = attach_candidate_pools(
        items, build_evaluation_cases(items, validation_config), validation_config
    )
    item_index = {str(item_id): index for index, item_id in enumerate(items["item_id"].astype(str))}
    output_dir = args.output_dir or (
        Path(config["paths"]["final_analysis_runs"]) / "stage1_validation"
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite validation sensitivity output: {output_dir}")
    output_dir.mkdir(parents=True)
    fusion_records: dict[float, list[dict[str, Any]]] = defaultdict(list)
    rerank_records: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    fusion = config["retrieval"]["fusion"]
    fixed_image_weight = float(fusion["image_weight"])
    fixed_text_weight = float(fusion["text_weight"])
    retriever = RuleRetriever(rules, rule_vectors, config["rule_retrieval"])
    embedder = OllamaEmbedder(
        models["embedders"]["qwen3_embedding"],
        endpoint=str(models["inference_defaults"]["endpoint"]),
    )
    metrics = ["hr_at_10", "ndcg_at_10", "mrr"]
    rerank_metric_fields = metrics + [
        "mean_top1_evidence_score",
        "mean_evidence_score_gain",
        "changed_top1_rate",
        "rule_trace_participation",
    ]
    for position, case in enumerate(cases.to_dict("records"), start=1):
        candidates = items.set_index("item_id").loc[case["candidate_item_ids"]].reset_index()
        candidate_indices = [item_index[str(item_id)] for item_id in candidates["item_id"]]
        query_index = item_index[str(case["query_item_id"])]
        for image_weight in fusion["validation_grid_image_weights"]:
            text_weight = 1.0 - float(image_weight)
            query = fuse_clip_embeddings(
                arrays["clip_image"][[query_index]],
                arrays["clip_text"][[query_index]],
                image_weight=float(image_weight),
                text_weight=text_weight,
            )[0]
            candidate_vectors = fuse_clip_embeddings(
                arrays["clip_image"][candidate_indices],
                arrays["clip_text"][candidate_indices],
                image_weight=float(image_weight),
                text_weight=text_weight,
            )
            scores = cosine_scores(query, candidate_vectors)
            ranked = pd.DataFrame(
                {"item_id": candidates["item_id"].astype(str), "score": scores}
            ).sort_values(["score", "item_id"], ascending=[False, True], kind="stable")
            positive_ids = set(case["positive_item_ids"])
            _append_metrics(
                fusion_records,
                float(image_weight),
                [item_id in positive_ids for item_id in ranked["item_id"]],
                category=str(case["target_category"]),
            )
        fixed_query = fuse_clip_embeddings(
            arrays["clip_image"][[query_index]],
            arrays["clip_text"][[query_index]],
            image_weight=fixed_image_weight,
            text_weight=fixed_text_weight,
        )[0]
        fixed_candidates = fuse_clip_embeddings(
            arrays["clip_image"][candidate_indices],
            arrays["clip_text"][candidate_indices],
            image_weight=fixed_image_weight,
            text_weight=fixed_text_weight,
        )
        clip_scores = cosine_scores(fixed_query, fixed_candidates)
        candidate_rows = candidates.to_dict("records")
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
        positive_ids = set(case["positive_item_ids"])
        clip_ranked = pd.DataFrame(
            {"item_id": candidates["item_id"].astype(str), "score": clip_scores}
        )
        clip_top_id = str(
            clip_ranked.sort_values(
                ["score", "item_id"], ascending=[False, True], kind="stable"
            ).iloc[0]["item_id"]
        )
        clip_evidence = {trace.candidate_id: trace.evidence_score for trace in traces}
        for top_k in config["validation_sensitivity"]["rule_top_k_values"]:
            evidence = []
            reduced_traces = []
            for trace in traces:
                reduced = (
                    truncate_trace(trace, int(top_k), config["rule_retrieval"])
                    if trace.rules
                    else trace
                )
                reduced_traces.append(reduced)
                evidence.append(reduced.evidence_score)
            for evidence_weight in config["validation_sensitivity"]["evidence_weights"]:
                clip_weight = 1.0 - float(evidence_weight)
                ranked = rerank_candidates(
                    pd.DataFrame(
                        {
                            "item_id": candidates["item_id"].astype(str),
                            "clip_score": clip_scores,
                            "evidence_score": evidence,
                        }
                    ),
                    clip_weight=clip_weight,
                    evidence_weight=float(evidence_weight),
                )
                top = ranked.iloc[0]
                top_trace = next(
                    trace for trace in reduced_traces if trace.candidate_id == str(top["item_id"])
                )
                _append_metrics(
                    rerank_records,
                    (float(evidence_weight), int(top_k)),
                    [item_id in positive_ids for item_id in ranked["item_id"]],
                    category=str(case["target_category"]),
                    diagnostics={
                        "mean_top1_evidence_score": float(top["evidence_score"]),
                        "mean_evidence_score_gain": float(top["evidence_score"])
                        - float(clip_evidence[clip_top_id]),
                        "changed_top1_rate": float(str(top["item_id"]) != clip_top_id),
                        "rule_trace_participation": float(bool(top_trace.rules)),
                    },
                )
        if position % 25 == 0:
            print(f"validation sensitivity: {position}/{len(cases)}", flush=True)
    fusion_table = pd.DataFrame(
        [
            {
                "image_weight": image_weight,
                "text_weight": 1.0 - image_weight,
                **_aggregate(records, metrics),
            }
            for image_weight, records in sorted(fusion_records.items())
        ]
    )
    rerank_table = pd.DataFrame(
        [
            {
                "evidence_weight": evidence_weight,
                "clip_weight": 1.0 - evidence_weight,
                "rule_top_k": top_k,
                **_aggregate(records, rerank_metric_fields),
            }
            for (evidence_weight, top_k), records in sorted(rerank_records.items())
        ]
    )
    pareto = pareto_frontier(
        rerank_table, list(config["validation_sensitivity"]["pareto_objectives"])
    )
    fusion_path = output_dir / "fusion_grid.csv"
    rerank_path = output_dir / "reranking_grid.csv"
    pareto_path = output_dir / "reranking_pareto_frontier.csv"
    fusion_table.to_csv(fusion_path, index=False, lineterminator="\n")
    rerank_table.to_csv(rerank_path, index=False, lineterminator="\n")
    pareto.to_csv(pareto_path, index=False, lineterminator="\n")
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 2,
        "stage": "final_stage1_validation_sensitivity",
        "input_artifact_hashes": {
            str(Path(config["paths"]["active_data_manifest"])): sha256_file(
                Path(config["paths"]["active_data_manifest"])
            ),
            str(Path(config["paths"]["active_embedding_manifest"])): sha256_file(
                Path(config["paths"]["active_embedding_manifest"])
            ),
            str(kb_manifest_paths[0]): sha256_file(kb_manifest_paths[0]),
        },
        "output_artifact_hashes": {
            str(fusion_path): sha256_file(fusion_path),
            str(rerank_path): sha256_file(rerank_path),
            str(pareto_path): sha256_file(pareto_path),
        },
        "row_counts": {
            "validation_cases": len(cases),
            "fusion_configurations": len(fusion_table),
            "reranking_configurations": len(rerank_table),
            "pareto_frontier_configurations": int(pareto["pareto_status"].eq("frontier").sum()),
        },
        "failure_counts": {"ranking_failures": 0, "trace_failures": 0},
        "fixed_confirmatory_defaults": {
            "image_weight": fixed_image_weight,
            "text_weight": fixed_text_weight,
            "clip_weight": float(config["reranking"]["clip_weight"]),
            "evidence_weight": float(config["reranking"]["evidence_weight"]),
            "rule_top_k": int(config["rule_retrieval"]["rule_top_k"]),
        },
        "policy": "validation_only_diagnostic_does_not_select_confirmatory_defaults",
    }
    write_new_json(manifest_path, manifest)
    print(json.dumps(manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
