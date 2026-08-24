"""Measure the impact of reranking every frozen Stage 6 pool with the V3 KB.

This is deliberately an isolated counterfactual.  It reads immutable Stage 6
inputs and the active V3 KB, calls only the configured embedding services, and
writes a new experiment directory.  It never invokes a generator, extractor,
or verifier and never replaces canonical recommendation or explanation files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from evidence_fashion.explanation import common_context
from evidence_fashion.grounding_contracts import require_trace_applicability
from evidence_fashion.kb_audit import load_audited_rules
from evidence_fashion.reranking import rerank_candidates
from evidence_fashion.retrieval import (
    CLIPEmbedder,
    OllamaEmbedder,
    cosine_scores,
    fuse_clip_embeddings,
)
from evidence_fashion.rule_retrieval import RuleRetriever, candidate_rule_representation

SELECTION_SEED = 42


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _selection_key(case_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{case_id}".encode()).hexdigest()


def _select(
    records: Sequence[Mapping[str, Any]], categories: Sequence[str]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category in categories:
        eligible = [
            dict(row)
            for row in records
            if row["target_category"] == category and row["evidence_trace"]["rules"]
        ]
        eligible.sort(
            key=lambda row: (
                -int(len(row["evidence_trace"]["rules"]) >= 2),
                _selection_key(str(row["case_id"])),
            )
        )
        if len(eligible) < 100:
            raise ValueError(
                f"V3 has only {len(eligible)} eligible cases for {category}; need 100."
            )
        selected.extend(eligible[:100])
    if len(selected) != 500 or len({row["case_id"] for row in selected}) != 500:
        raise ValueError("V3 selection is incomplete or duplicated.")
    for row in selected:
        require_trace_applicability(row["evidence_trace"])
    return selected


def _query_prompt(case: Mapping[str, Any]) -> str:
    return " | ".join(
        [
            f"Query category: {case['query_category']}",
            f"Query text: {case['query_text']}",
            f"User request: {case['user_request']}",
            f"Target category: {case['target_category']}",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".runtime/current/data/data-a65702f6537a"),
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path(".runtime/current/embeddings/embedding-full-414ac73b4696"),
    )
    parser.add_argument(
        "--old-locked-cases",
        type=Path,
        default=Path(
            ".runtime/current/recommendations/stage6/"
            "stage6-confirmatory-414ac73b4696/locked_cases.jsonl"
        ),
    )
    parser.add_argument(
        "--old-selection",
        type=Path,
        default=Path(
            ".runtime/current/explanations/"
            "stage9-v3-selection-8e0dedea27a1/selected_locked_cases.jsonl"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path(".runtime/current/experiments"))
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    items = pd.read_parquet(args.data_dir / "prepared_items.parquet")
    cases = _read_jsonl(args.data_dir / "evaluation_cases.jsonl")
    old_locked = _read_jsonl(args.old_locked_cases)
    old_selected = _read_jsonl(args.old_selection)
    if len(cases) != 1000 or len({str(row["case_id"]) for row in cases}) != 1000:
        raise ValueError("Expected exactly 1,000 frozen evaluation cases.")
    if len(old_locked) != 1000 or len({str(row["case_id"]) for row in old_locked}) != 1000:
        raise ValueError("Expected exactly 1,000 frozen old locked cases.")
    if len(old_selected) != 500 or len({str(row["case_id"]) for row in old_selected}) != 500:
        raise ValueError("Expected exactly 500 existing V3 explanation cases.")

    metadata = _read_jsonl(args.embeddings_dir / "sample_metadata.jsonl")
    embedding_index = {str(row["item_id"]): index for index, row in enumerate(metadata)}
    if len(embedding_index) != len(metadata):
        raise ValueError("Embedding metadata has duplicate item IDs.")
    item_lookup = items.set_index("item_id", drop=False)
    required_ids = {
        str(item_id)
        for case in cases
        for item_id in [case["query_item_id"], *case["candidate_item_ids"]]
    }
    missing = required_ids.difference(embedding_index).union(
        required_ids.difference(item_lookup.index)
    )
    if missing:
        raise ValueError(f"Frozen cases reference unavailable items: {sorted(missing)[:5]}")

    clip_fused = np.load(args.embeddings_dir / "clip_fused.npy", mmap_mode="r")
    clip_image = np.load(args.embeddings_dir / "clip_image.npy", mmap_mode="r")
    if len(clip_fused) != len(metadata) or len(clip_image) != len(metadata):
        raise ValueError("Cached CLIP embeddings are not aligned with sample metadata.")

    print("Encoding the 1,000 frozen query prompts with CLIP on the configured device.", flush=True)
    clip = CLIPEmbedder(models["embedders"]["clip"])
    query_text = clip.encode_text(
        [_query_prompt(case) for case in cases],
        batch_size=int(config["stage4_validation"]["embedding_batch_size"]),
    )
    fusion = config["retrieval"]["fusion"]
    query_image = np.asarray(
        [clip_image[embedding_index[str(case["query_item_id"])]] for case in cases],
        dtype=np.float32,
    )
    query_fused = fuse_clip_embeddings(
        query_image,
        query_text,
        image_weight=float(fusion["image_weight"]),
        text_weight=float(fusion["text_weight"]),
    )

    rules = load_audited_rules(config)
    print(f"Embedding {len(rules)} V3 rules with Ollama only.", flush=True)
    embedder = OllamaEmbedder(models["embedders"]["qwen3_embedding"], endpoint=args.ollama_endpoint)
    rule_embeddings = embedder.encode(
        rules["rule_text"].astype(str).tolist(),
        batch_size=int(config["stage4_validation"]["embedding_batch_size"]),
    )
    retriever = RuleRetriever(rules, rule_embeddings, config["rule_retrieval"])
    old_locked_by_case = {str(row["case_id"]): row for row in old_locked}

    records: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []
    top_k = int(config["stage4_validation"]["selected_rule_top_k"])
    batch_size = int(config["stage4_validation"]["embedding_batch_size"])
    for case_number, case in enumerate(cases, start=1):
        candidate_ids = [str(item_id) for item_id in case["candidate_item_ids"]]
        candidates = [item_lookup.loc[item_id].to_dict() for item_id in candidate_ids]
        representations = [
            candidate_rule_representation(case, candidate) for candidate in candidates
        ]
        vectors = embedder.encode(representations, batch_size=batch_size)
        traces = [
            retriever.retrieve_and_score(
                case=case,
                candidate=candidate,
                representation_embedding=vector,
                top_k=top_k,
            )
            for candidate, vector in zip(candidates, vectors, strict=True)
        ]
        indices = [embedding_index[item_id] for item_id in candidate_ids]
        reranked = rerank_candidates(
            pd.DataFrame(
                {
                    "item_id": candidate_ids,
                    "clip_score": cosine_scores(query_fused[case_number - 1], clip_fused[indices]),
                    "evidence_score": [trace.evidence_score for trace in traces],
                    "trace_index": np.arange(len(candidate_ids)),
                }
            ),
            clip_weight=float(config["reranking"]["reference_clip_weight"]),
            evidence_weight=float(config["reranking"]["reference_evidence_weight"]),
        )
        top = reranked.iloc[0]
        top_trace = traces[int(top["trace_index"])]
        candidate = item_lookup.loc[str(top["item_id"])].to_dict()
        old = old_locked_by_case[str(case["case_id"])]
        record = {
            "case_id": str(case["case_id"]),
            "query_item_id": str(case["query_item_id"]),
            "query_outfit_id": str(case["query_outfit_id"]),
            "query_item_minimal_name": str(case["query_text"] or case["query_category"]),
            "request": str(case["user_request"]),
            "target_category": str(case["target_category"]),
            "locked_candidate_id": str(top["item_id"]),
            "locked_candidate_minimal_name": str(candidate["text"] or candidate["category"]),
            "pre_rerank_rank": int(top["pre_rerank_rank"]),
            "post_rerank_rank": int(top["post_rerank_rank"]),
            "clip_score": float(top["clip_score"]),
            "evidence_score": float(top["evidence_score"]),
            "final_score": float(top["final_score"]),
            "evidence_trace": top_trace.to_dict(),
        }
        records.append(record)
        per_case.append(
            {
                "case_id": record["case_id"],
                "target_category": record["target_category"],
                "old_top1_id": str(old["locked_candidate_id"]),
                "new_top1_id": record["locked_candidate_id"],
                "top1_changed": bool(old["locked_candidate_id"] != record["locked_candidate_id"]),
                "old_top1_clip_score": float(old["clip_score"]),
                "new_top1_clip_score": record["clip_score"],
                "new_top1_evidence_score": record["evidence_score"],
                "new_top1_final_score": record["final_score"],
                "new_top1_trace_sha256": _sha256(record["evidence_trace"]),
                "new_top1_rule_ids": [
                    rule["rule_id"] for rule in record["evidence_trace"]["rules"]
                ],
            }
        )
        if case_number % 25 == 0 or case_number == len(cases):
            print(f"Completed {case_number}/{len(cases)} full candidate pools.", flush=True)

    selected = _select(records, config["preprocessing"]["target_categories"])
    old_selected_by_case = {str(row["case_id"]): row for row in old_selected}
    new_selected_by_case = {str(row["case_id"]): row for row in selected}
    old_ids = set(old_selected_by_case)
    new_ids = set(new_selected_by_case)
    unchanged_slots = 0
    needed_no_rag_inputs = 0
    needed_rule_rag_inputs = 0
    workload_rows = []
    for case_id, new_row in sorted(new_selected_by_case.items()):
        old_row = old_selected_by_case.get(case_id)
        no_rag_reusable = bool(
            old_row
            and old_row["locked_candidate_id"] == new_row["locked_candidate_id"]
            and common_context(old_row) == common_context(new_row)
        )
        rule_rag_reusable = bool(
            no_rag_reusable
            and _canonical_json(old_row["evidence_trace"])
            == _canonical_json(new_row["evidence_trace"])
        )
        needed_no_rag_inputs += int(not no_rag_reusable)
        needed_rule_rag_inputs += int(not rule_rag_reusable)
        unchanged_slots += int(rule_rag_reusable)
        workload_rows.append(
            {
                "case_id": case_id,
                "target_category": new_row["target_category"],
                "previously_selected": old_row is not None,
                "candidate_changed": old_row is None
                or old_row["locked_candidate_id"] != new_row["locked_candidate_id"],
                "no_rag_input_reusable": no_rag_reusable,
                "rule_rag_input_reusable": rule_rag_reusable,
                "old_candidate_id": old_row["locked_candidate_id"] if old_row else None,
                "new_candidate_id": new_row["locked_candidate_id"],
                "old_trace_sha256": _sha256(old_row["evidence_trace"]) if old_row else None,
                "new_trace_sha256": _sha256(new_row["evidence_trace"]),
            }
        )

    category_counts = Counter(row["target_category"] for row in per_case if row["top1_changed"])
    summary = {
        "schema_version": 1,
        "purpose": "isolated_full_pool_v3_kb_reranking_counterfactual",
        "model_calls": {
            "clip_encoder": models["embedders"]["clip"]["model_id"],
            "ollama_embedding_model": models["embedders"]["qwen3_embedding"]["model_id"],
            "generator_called": False,
            "extractor_called": False,
            "verifier_called": False,
        },
        "inputs": {
            "kb_path": str(Path(config["paths"]["knowledge_base"])),
            "kb_sha256": hashlib.sha256(
                Path(config["paths"]["knowledge_base"]).read_bytes()
            ).hexdigest(),
            "cases": len(cases),
            "old_locked_cases": str(args.old_locked_cases),
            "old_v3_selection": str(args.old_selection),
        },
        "top1_comparison": {
            "changed": sum(row["top1_changed"] for row in per_case),
            "unchanged": sum(not row["top1_changed"] for row in per_case),
            "changed_by_target_category": dict(sorted(category_counts.items())),
        },
        "selection_comparison": {
            "old_selected_cases": len(old_ids),
            "new_selected_cases": len(new_ids),
            "case_ids_retained": len(old_ids & new_ids),
            "case_ids_removed": len(old_ids - new_ids),
            "case_ids_added": len(new_ids - old_ids),
            "same_case_same_candidate": sum(
                old_selected_by_case[case_id]["locked_candidate_id"]
                == new_selected_by_case[case_id]["locked_candidate_id"]
                for case_id in old_ids & new_ids
            ),
        },
        "incremental_workload": {
            "new_or_changed_no_rag_condition_inputs": needed_no_rag_inputs,
            "new_or_changed_rule_rag_condition_inputs": needed_rule_rag_inputs,
            "reusable_full_condition_inputs": unchanged_slots * 2,
            "generator_models": len(models["generators"]["roster"]),
            "new_explanation_generations": (needed_no_rag_inputs + needed_rule_rag_inputs)
            * len(models["generators"]["roster"]),
            "new_extractions": (needed_no_rag_inputs + needed_rule_rag_inputs)
            * len(models["generators"]["roster"]),
            "new_verifications": (needed_no_rag_inputs + needed_rule_rag_inputs)
            * len(models["generators"]["roster"]),
        },
        "selection_policy": "prefer_trace_size_at_least_2_then_sha256_seeded_case_order",
    }
    run_id = f"v3-full-pool-counterfactual-{summary['inputs']['kb_sha256'][:12]}"
    run_dir = args.output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Experiment output already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "per_case_top1_comparison.jsonl", per_case)
    _write_jsonl(run_dir / "new_selected_locked_cases.jsonl", selected)
    _write_jsonl(run_dir / "explanation_reuse_audit.jsonl", workload_rows)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
