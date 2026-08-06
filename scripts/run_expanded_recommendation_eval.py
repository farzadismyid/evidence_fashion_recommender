"""Run the outcome-blind frozen 3,000-case recommendation evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evidence_fashion_recommender.cache import file_fingerprint
from evidence_fashion_recommender.cli import _build_v2_evidence_scorer
from evidence_fashion_recommender.config import load_config
from evidence_fashion_recommender.data.dataset import load_huggingface_split, load_prepared_dataset
from evidence_fashion_recommender.evaluation.controlled import (
    build_evaluation_cases,
    encode_evaluation_queries,
)
from evidence_fashion_recommender.evaluation.ranking import build_controlled_candidate_set, ranking_metrics
from evidence_fashion_recommender.evaluation.splits import assign_outfit_splits
from evidence_fashion_recommender.models.multimodal import CLIPEmbedder, fuse_embeddings
from evidence_fashion_recommender.models.text import SentenceTransformerEmbedder
from evidence_fashion_recommender.reranking import weighted_rerank
from evidence_fashion_recommender.run import start_run


ROOT = Path("outputs/recommendation_eval_expanded")
FINAL_ROOT = Path("outputs/final_eval_v2")
PROTOCOL = ROOT / "frozen_protocol.json"
CATEGORIES = ("accessories", "bottoms", "outerwear", "shoes", "tops")
METHODS = ("minilm_text", "clip_image", "clip_text", "clip_fused", "evidence_reranked")
CUTS = [1, 5, 10]


def _key_hash(frame: pd.DataFrame) -> str:
    values = frame[["query_item_id", "target_category"]].astype(str).sort_values(
        ["query_item_id", "target_category"]
    )
    payload = json.dumps(values.to_dict("records"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _build_schedule(protocol: dict[str, object]) -> pd.DataFrame:
    items = pd.read_parquet("data/processed/items_clean.parquet")
    targets = pd.read_parquet("data/processed/target_items_clean.parquet")
    cases = build_evaluation_cases(items, targets, list(CATEGORIES), len(items) * len(CATEGORIES), 42)
    cases = assign_outfit_splits(
        cases,
        outfit_column="query_outfit_id",
        seed=42,
        development_fraction=0.6,
        validation_fraction=0.2,
    )
    eligible = cases[cases["research_split"] == "test"].copy()
    historical = pd.read_csv("outputs/robustness/schedules/test_schedule.csv")
    historical["paper_case_id"] = historical.apply(
        lambda row: f"V2_TEST_{int(row['case_index']):04d}_{row['target_category']}", axis=1
    )
    historical["historical_subset"] = True
    base = set(zip(historical["query_item_id"].astype(str), historical["target_category"].astype(str)))
    selected = [historical]
    allocation = protocol["cohort_allocation"]
    for category in CATEGORIES:
        group = eligible[eligible["target_category"] == category].copy()
        fixed = group[
            [
                (str(query), str(target)) in base
                for query, target in zip(group["query_item_id"], group["target_category"])
            ]
        ]
        count = int(allocation[category])
        remaining = group.drop(fixed.index).copy()
        remaining["_rank"] = remaining.apply(
            lambda row: hashlib.sha256(
                f"42|{row['query_item_id']}|{row['target_category']}".encode()
            ).hexdigest(),
            axis=1,
        )
        selected.append(remaining.sort_values(["_rank", "query_item_id"]).head(count - len(fixed)))
    schedule = pd.concat(selected, ignore_index=True)
    historical_keys = set(zip(historical["query_item_id"].astype(str), historical["target_category"].astype(str)))
    schedule["historical_subset"] = [
        (str(query), str(target)) in historical_keys
        for query, target in zip(schedule["query_item_id"], schedule["target_category"])
    ]
    new_mask = ~schedule["historical_subset"]
    schedule.loc[new_mask, "paper_case_id"] = [
        f"V2_EXPANDED_{index:04d}_{category}"
        for index, category in enumerate(schedule.loc[new_mask, "target_category"], 300)
    ]
    schedule = schedule.sort_values(["historical_subset", "paper_case_id"], ascending=[False, True]).reset_index(drop=True)
    schedule["case_index"] = np.arange(len(schedule))
    if len(schedule) != 3000 or _key_hash(schedule) != protocol["selection_method"]["selected_identity_key_sha256"]:
        raise ValueError("Expanded schedule does not match frozen protocol.")
    return schedule


def _write_candidates(config, schedule: pd.DataFrame) -> pd.DataFrame:
    historical = pd.read_csv(FINAL_ROOT / "sources/test/candidate_sets.csv")
    new = schedule[~schedule["historical_subset"]]
    targets = pd.read_parquet(FINAL_ROOT / "sources/target_items.parquet").reset_index(drop=True)
    id_to_row = {item: index for index, item in enumerate(targets["item_ID"].astype(str))}
    scorer, _ = _build_v2_evidence_scorer(config)
    rows = [historical]
    for _, case in new.iterrows():
        pool = build_controlled_candidate_set(
            targets,
            str(case["query_outfit_id"]),
            str(case["target_category"]),
            config.evaluation.negatives_per_case,
            np.random.default_rng(config.project.seed + int(case["case_index"])),
            query_item_id=str(case["query_item_id"]),
        ).copy()
        pool["paper_case_id"] = case["paper_case_id"]
        pool["query_outfit_id"] = case["query_outfit_id"]
        pool["target_category"] = case["target_category"]
        pool["candidate_position"] = np.arange(len(pool))
        pool["target_row"] = pool["item_ID"].astype(str).map(id_to_row)
        pool["evidence_score"] = scorer.score(case, pool)
        rows.append(pool)
    return pd.concat(rows, ignore_index=True)


def _write_query_embeddings(config, schedule: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_root = FINAL_ROOT / "materialized/query_embeddings/test/e659d4e048de87eff77b2a4edca4823398cf2fd3815e409e24a204d9e495704d"
    old = {name: np.load(old_root / f"{name}.npy") for name in ("query_minilm", "query_clip_image", "query_clip_text")}
    new = schedule[~schedule["historical_subset"]]
    context = start_run(config)
    prepared = load_prepared_dataset(config, context.cache)
    split = load_huggingface_split(config)
    encoded = encode_evaluation_queries(
        new.reset_index(drop=True),
        prepared.items,
        split,
        SentenceTransformerEmbedder(config.models.text_embedding, config.project.device),
        CLIPEmbedder(config.models.multimodal_embedding, config.project.device),
    )
    return (
        np.concatenate([old["query_minilm"], encoded.minilm]),
        np.concatenate([old["query_clip_image"], encoded.clip_image]),
        np.concatenate([old["query_clip_text"], encoded.clip_text]),
    )


def _evaluate(schedule, candidates, query_minilm, query_image, query_text) -> pd.DataFrame:
    target_root = FINAL_ROOT / "materialized/target_embeddings"
    target_minilm = np.load(target_root / "target_minilm.npy", mmap_mode="r")
    target_image = np.load(target_root / "target_clip_image.npy", mmap_mode="r")
    target_text = np.load(target_root / "target_clip_text.npy", mmap_mode="r")
    fusion = json.loads((FINAL_ROOT / "validation/fusion_tuning/selected_fusion.json").read_text())
    image_weight = float(fusion["image_weight"])
    rows = []
    for index, case in schedule.iterrows():
        pool = candidates[candidates["paper_case_id"] == case["paper_case_id"]].copy()
        target_rows = pool["target_row"].astype(int).to_numpy()
        scores = {
            "minilm_text": target_minilm[target_rows] @ query_minilm[index],
            "clip_image": target_image[target_rows] @ query_image[index],
            "clip_text": target_text[target_rows] @ query_text[index],
        }
        fused_targets = fuse_embeddings(target_image[target_rows], target_text[target_rows], image_weight)
        fused_query = fuse_embeddings(query_image[index][None, :], query_text[index][None, :], image_weight)[0]
        scores["clip_fused"] = fused_targets @ fused_query
        pool["clip_score"] = scores["clip_fused"]
        reranked_evidence = weighted_rerank(pool, 0.75, 0.25, True)
        for method, values in scores.items():
            ranked = pool.iloc[np.argsort(-values)]
            rows.append(
                {
                    "paper_case_id": case["paper_case_id"],
                    "query_outfit_id": case["query_outfit_id"],
                    "target_category": case["target_category"],
                    "historical_subset": bool(case["historical_subset"]),
                    "method": method,
                    "num_candidates": len(pool),
                    "num_positives": int(pool["is_positive"].sum()),
                    **ranking_metrics(ranked["is_positive"].to_numpy(), CUTS),
                }
            )
        rows.append(
            {
                "paper_case_id": case["paper_case_id"],
                "query_outfit_id": case["query_outfit_id"],
                "target_category": case["target_category"],
                "historical_subset": bool(case["historical_subset"]),
                "method": "evidence_reranked",
                "num_candidates": len(pool),
                "num_positives": int(pool["is_positive"].sum()),
                **ranking_metrics(reranked_evidence["is_positive"].to_numpy(), CUTS),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    config = load_config("configs/final_eval_v2.yaml", [])
    schedule = _build_schedule(protocol)
    ROOT.mkdir(exist_ok=True)
    schedule.to_csv(ROOT / "expanded_schedule.csv", index=False)
    candidates = _write_candidates(config, schedule)
    candidates.to_csv(ROOT / "candidate_sets.csv", index=False)
    query_minilm, query_image, query_text = _write_query_embeddings(config, schedule)
    embedding_root = ROOT / "query_embeddings"
    embedding_root.mkdir(exist_ok=True)
    np.save(embedding_root / "query_minilm.npy", query_minilm)
    np.save(embedding_root / "query_clip_image.npy", query_image)
    np.save(embedding_root / "query_clip_text.npy", query_text)
    results = _evaluate(schedule, candidates, query_minilm, query_image, query_text)
    results.to_csv(ROOT / "ranking_results.csv", index=False)
    (ROOT / "run_manifest.json").write_text(json.dumps({
        "protocol_hash": file_fingerprint(PROTOCOL), "schedule_hash": file_fingerprint(ROOT / "expanded_schedule.csv"),
        "candidate_hash": file_fingerprint(ROOT / "candidate_sets.csv"), "results_hash": file_fingerprint(ROOT / "ranking_results.csv"),
        "methods": METHODS, "reranker_weights": {"clip": 0.75, "evidence": 0.25},
        "historical_cases_reused": 300, "new_cases_computed": 2700,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
