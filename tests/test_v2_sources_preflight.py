import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evidence_fashion_recommender.artifacts import embedding_record
from evidence_fashion_recommender.cache import ArtifactCache, file_fingerprint
from evidence_fashion_recommender.config import load_config
from evidence_fashion_recommender.evaluation.materialization import query_cache_directory
from evidence_fashion_recommender.evaluation.v2_preflight import inspect_readiness
from evidence_fashion_recommender.evaluation.v2_sources import (
    materialize_target_item_table,
    produce_candidate_sets,
    produce_selected_cases,
)


class TinyScorer:
    def score(self, case, candidates):
        return np.linspace(0.1, 0.9, len(candidates), dtype=np.float32)

    def retrieve(self, case, candidate):
        return pd.DataFrame(
            [
                {
                    "rule_id": "r1",
                    "rule_text": "Coordinate colour and formality.",
                    "source_title": "KB",
                }
            ]
        )


def _fixtures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/final_eval_v2.yaml")
    items = pd.DataFrame(
        [
            {
                "item_ID": "p1",
                "outfit_ID": "o1",
                "broad_category": "shoes",
                "item_text": "black shoe",
                "category": "shoe",
                "text": "black",
                "original_dataset_index": 0,
            },
            {
                "item_ID": "n1",
                "outfit_ID": "o2",
                "broad_category": "shoes",
                "item_text": "white shoe",
                "category": "shoe",
                "text": "white",
                "original_dataset_index": 1,
            },
            {
                "item_ID": "n2",
                "outfit_ID": "o3",
                "broad_category": "shoes",
                "item_text": "blue shoe",
                "category": "shoe",
                "text": "blue",
                "original_dataset_index": 2,
            },
        ]
    )
    source = Path("outputs/cache/datasets/source.parquet")
    source.parent.mkdir(parents=True)
    items.to_parquet(source, index=False)
    cache = ArtifactCache(config.paths.cache_dir)
    for modality in ("minilm_text", "clip_image", "clip_text"):
        record = embedding_record(cache, config, items, modality)
        record.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(record.path, np.eye(3, dtype=np.float32))
    schedules = {}
    for split in ("validation", "test"):
        path = Path(f"{split}.csv")
        pd.DataFrame(
            [
                {
                    "paper_case_id": f"{split}-1",
                    "case_index": 0,
                    "query_item_id": "q1",
                    "query_category": "dress",
                    "query_group": "dresses",
                    "query_text": "black dress",
                    "user_request": "recommend shoes",
                    "query_outfit_id": "o1",
                    "target_category": "shoes",
                    "research_split": split,
                }
            ]
        ).to_csv(path, index=False)
        schedules[split] = path
    return config, items, source, schedules


def test_fresh_target_and_candidate_sources_are_row_aligned_and_resumable(tmp_path, monkeypatch):
    config, items, source, schedules = _fixtures(tmp_path, monkeypatch)
    target_path = Path("outputs/final_eval_v2/sources/target_items.parquet")
    target_manifest = materialize_target_item_table(
        config=config, items=items, source_paths=[source], output_path=target_path
    )
    assert target_manifest["row_count"] == 3
    assert set(target_manifest["embedding_compatibility_hashes"]) == {
        "minilm_text",
        "clip_image",
        "clip_text",
    }
    assert pd.read_parquet(target_path)["item_ID"].tolist() == ["p1", "n1", "n2"]
    assert (
        materialize_target_item_table(
            config=config, items=items, source_paths=[source], output_path=target_path
        )
        == target_manifest
    )

    candidate_path = Path("outputs/final_eval_v2/sources/validation/candidate_sets.csv")
    manifest = produce_candidate_sets(
        config=config,
        split="validation",
        schedule_path=schedules["validation"],
        target_items_path=target_path,
        scorer=TinyScorer(),
        evidence_hash="fresh-kb-hash",
        output_path=candidate_path,
    )
    candidates = pd.read_csv(candidate_path)
    assert {
        "paper_case_id",
        "query_outfit_id",
        "target_category",
        "candidate_position",
        "target_row",
        "is_positive",
        "evidence_score",
    } <= set(candidates)
    assert candidates["evidence_score"].notna().all()
    assert manifest["protocol"] == "final_eval_v2_candidate_sets"
    assert manifest["output_hash"] == file_fingerprint(candidate_path)
    assert (
        produce_candidate_sets(
            config=config,
            split="validation",
            schedule_path=schedules["validation"],
            target_items_path=target_path,
            scorer=TinyScorer(),
            evidence_hash="fresh-kb-hash",
            output_path=candidate_path,
        )
        == manifest
    )
    candidates.loc[0, "evidence_score"] = 99
    candidates.to_csv(candidate_path, index=False)
    with pytest.raises(ValueError, match="hash"):
        produce_candidate_sets(
            config=config,
            split="validation",
            schedule_path=schedules["validation"],
            target_items_path=target_path,
            scorer=TinyScorer(),
            evidence_hash="fresh-kb-hash",
            output_path=candidate_path,
        )


def test_selected_cases_use_validation_artifacts_and_reject_v1(tmp_path, monkeypatch):
    config, items, source, schedules = _fixtures(tmp_path, monkeypatch)
    target_path = Path("outputs/final_eval_v2/sources/target_items.parquet")
    materialize_target_item_table(
        config=config, items=items, source_paths=[source], output_path=target_path
    )
    candidate_path = Path("outputs/final_eval_v2/sources/validation/candidate_sets.csv")
    produce_candidate_sets(
        config=config,
        split="validation",
        schedule_path=schedules["validation"],
        target_items_path=target_path,
        scorer=TinyScorer(),
        evidence_hash="kb",
        output_path=candidate_path,
    )
    bundle = Path("outputs/final_eval_v2/prepared/validation")
    bundle.mkdir(parents=True)
    target = np.eye(3, dtype=np.float32)
    query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    for name in ("target_clip_image.npy", "target_clip_text.npy"):
        np.save(bundle / name, target)
    for name in ("query_clip_image.npy", "query_clip_text.npy"):
        np.save(bundle / name, query)
    fusion = Path("outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json")
    rerank = Path("outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json")
    fusion.parent.mkdir(parents=True)
    rerank.parent.mkdir(parents=True)
    fusion.write_text(json.dumps({"selected_on": "validation", "image_weight": 0.5}))
    rerank.write_text(
        json.dumps({"selected_on": "validation", "clip_weight": 0.5, "evidence_weight": 0.5})
    )
    output = Path("outputs/final_eval_v2/sources/validation/selected_cases.csv")
    manifest = produce_selected_cases(
        split="validation",
        schedule_path=schedules["validation"],
        target_items_path=target_path,
        candidate_sets_path=candidate_path,
        target_clip_image_path=bundle / "target_clip_image.npy",
        target_clip_text_path=bundle / "target_clip_text.npy",
        query_clip_image_path=bundle / "query_clip_image.npy",
        query_clip_text_path=bundle / "query_clip_text.npy",
        fusion_selection_path=fusion,
        reranking_selection_path=rerank,
        scorer=TinyScorer(),
        output_path=output,
    )
    row = pd.read_csv(output).iloc[0]
    assert row["packet_source_protocol"] == "final_eval_v2_selected"
    assert row["rule_evidence_ids"] == '["r1"]'
    assert manifest["selected_fusion_hash"] == file_fingerprint(fusion)
    assert (
        produce_selected_cases(
            split="validation",
            schedule_path=schedules["validation"],
            target_items_path=target_path,
            candidate_sets_path=candidate_path,
            target_clip_image_path=bundle / "target_clip_image.npy",
            target_clip_text_path=bundle / "target_clip_text.npy",
            query_clip_image_path=bundle / "query_clip_image.npy",
            query_clip_text_path=bundle / "query_clip_text.npy",
            fusion_selection_path=fusion,
            reranking_selection_path=rerank,
            scorer=TinyScorer(),
            output_path=output,
        )
        == manifest
    )
    candidate_manifest_path = candidate_path.with_suffix(".manifest.json")
    candidate_manifest = json.loads(candidate_manifest_path.read_text())
    candidate_manifest["protocol"] = "legacy_v1_candidate_sets"
    candidate_manifest_path.write_text(json.dumps(candidate_manifest))
    output.unlink()
    output.with_suffix(".manifest.json").unlink()
    with pytest.raises(ValueError, match="fresh v2"):
        produce_selected_cases(
            split="validation",
            schedule_path=schedules["validation"],
            target_items_path=target_path,
            candidate_sets_path=candidate_path,
            target_clip_image_path=bundle / "target_clip_image.npy",
            target_clip_text_path=bundle / "target_clip_text.npy",
            query_clip_image_path=bundle / "query_clip_image.npy",
            query_clip_text_path=bundle / "query_clip_text.npy",
            fusion_selection_path=fusion,
            reranking_selection_path=rerank,
            scorer=TinyScorer(),
            output_path=output,
        )


def test_preflight_reports_blocked_then_ready_sources(tmp_path, monkeypatch):
    config, items, source, schedules = _fixtures(tmp_path, monkeypatch)
    blocked = inspect_readiness(config, schedules)
    assert blocked["checks"]["target_items"]["status"] == "BLOCKED"
    assert blocked["query_embedding_computation_required"] is True
    target_path = Path("outputs/final_eval_v2/sources/target_items.parquet")
    materialize_target_item_table(
        config=config, items=items, source_paths=[source], output_path=target_path
    )
    for split, schedule in schedules.items():
        candidate = Path(f"outputs/final_eval_v2/sources/{split}/candidate_sets.csv")
        produce_candidate_sets(
            config=config,
            split=split,
            schedule_path=schedule,
            target_items_path=target_path,
            scorer=TinyScorer(),
            evidence_hash="kb",
            output_path=candidate,
        )
        query_dir = query_cache_directory(config, schedule, split)
        query_dir.mkdir(parents=True)
        hashes = {}
        for name in ("query_minilm.npy", "query_clip_image.npy", "query_clip_text.npy"):
            np.save(query_dir / name, np.ones((1, 2), dtype=np.float32))
            hashes[name] = file_fingerprint(query_dir / name)
        (query_dir / "query_embedding_manifest.json").write_text(
            json.dumps(
                {
                    "protocol": "final_eval_v2_query_embeddings",
                    "output_hashes": hashes,
                }
            )
        )
    ready = inspect_readiness(config, schedules)
    assert ready["checks"]["target_items"]["status"] == "READY"
    assert ready["checks"]["validation_candidate_sets"]["status"] == "READY"
    assert ready["query_embedding_computation_required"] is False
    assert ready["v1_primary_reference_detected"] is False
