import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evidence_fashion_recommender.artifacts import embedding_record
from evidence_fashion_recommender.cache import ArtifactCache, file_fingerprint
from evidence_fashion_recommender.cli import main
from evidence_fashion_recommender.config import load_config
from evidence_fashion_recommender.evaluation.materialization import (
    materialize_query_embeddings,
    query_cache_directory,
)


def _schedule(path: Path, split: str = "validation") -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "paper_case_id": "c1",
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
    )
    frame.to_csv(path, index=False)
    return frame


def _target_items(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "item_ID": "i1",
                "outfit_ID": "o1",
                "broad_category": "shoes",
                "item_text": "black pumps",
            },
            {
                "item_ID": "i2",
                "outfit_ID": "o2",
                "broad_category": "shoes",
                "item_text": "white trainers",
            },
        ]
    )
    frame.to_csv(path, index=False)
    return frame


def _candidate_source(path: Path, schedule_path: Path, protocol: str) -> None:
    frame = pd.DataFrame(
        [
            {
                "paper_case_id": "c1",
                "candidate_position": position,
                "target_row": position,
                "is_positive": int(position == 0),
                "evidence_score": float(position == 0),
            }
            for position in (0, 1)
        ]
    )
    frame.to_csv(path, index=False)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "schedule_hash": file_fingerprint(schedule_path),
                "output_hash": file_fingerprint(path),
            }
        ),
        encoding="utf-8",
    )


def _cache_targets(config, targets: pd.DataFrame) -> None:
    cache = ArtifactCache(config.paths.cache_dir)
    array = np.eye(2, dtype=np.float32)
    for modality in ("minilm_text", "clip_image", "clip_text"):
        record = embedding_record(cache, config, targets, modality)
        record.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(record.path, array)


def test_materialization_resolves_cache_and_never_uses_v1_primary_outputs(
    tmp_path, monkeypatch
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = repository / "configs/final_eval_v2.yaml"
    monkeypatch.chdir(tmp_path)
    config = load_config(config_path)
    schedule_path = Path("validation.csv")
    _schedule(schedule_path)
    target_path = Path("targets.csv")
    targets = _target_items(target_path)
    _cache_targets(config, targets)

    with pytest.raises(PermissionError):
        materialize_query_embeddings(
            config=config,
            split="validation",
            schedule_path=schedule_path,
            builder=lambda _: {},
            approved=False,
        )

    def tiny_builder(schedule: pd.DataFrame) -> dict[str, np.ndarray]:
        value = np.ones((len(schedule), 2), dtype=np.float32)
        return {
            "query_minilm": value,
            "query_clip_image": value,
            "query_clip_text": value,
        }

    query_dir = materialize_query_embeddings(
        config=config,
        split="validation",
        schedule_path=schedule_path,
        builder=tiny_builder,
        approved=True,
    )
    assert query_dir.is_relative_to(Path("outputs/final_eval_v2"))
    assert query_dir == query_cache_directory(config, schedule_path, "validation")
    assert materialize_query_embeddings(
        config=config,
        split="validation",
        schedule_path=schedule_path,
        builder=lambda _: pytest.fail("resume must not recompute"),
        approved=True,
    ) == query_dir

    candidates = Path("candidates.csv")
    _candidate_source(candidates, schedule_path, "final_eval_v2_candidate_sets")
    args = [
        "--config",
        str(config_path),
        "materialize-final-retrieval-v2-inputs",
        "--split",
        "validation",
        "--schedule",
        str(schedule_path),
        "--target-items",
        str(target_path),
        "--candidate-source",
        str(candidates),
    ]
    assert main(args) == 0
    assert main(args) == 0
    root = Path("outputs/final_eval_v2/materialized")
    split_dir = root / "validation"
    manifest = json.loads((split_dir / "materialization_manifest.json").read_text())
    assert manifest["v1_primary_outputs_used"] is False
    assert manifest["embedding_actions"]["computed"] is False
    assert set(pd.read_csv(split_dir / "candidate_sets.csv").columns) >= {
        "paper_case_id",
        "candidate_position",
        "target_row",
        "is_positive",
        "evidence_score",
    }
    for name in ("target_minilm.npy", "target_clip_image.npy", "target_clip_text.npy"):
        assert (root / "target_embeddings" / name).is_file()
    assert manifest["output_hashes"]["candidate_sets.csv"] == file_fingerprint(
        split_dir / "candidate_sets.csv"
    )

    changed_candidates = pd.read_csv(candidates)
    changed_candidates.loc[0, "evidence_score"] = 0.5
    changed_candidates.to_csv(candidates, index=False)
    candidates.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "protocol": "final_eval_v2_candidate_sets",
                "schedule_hash": file_fingerprint(schedule_path),
                "output_hash": file_fingerprint(candidates),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different sources"):
        main(args)


def test_missing_query_cache_fails_and_v1_candidate_source_is_rejected(
    tmp_path, monkeypatch
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = repository / "configs/final_eval_v2.yaml"
    monkeypatch.chdir(tmp_path)
    config = load_config(config_path)
    schedule_path = Path("test.csv")
    _schedule(schedule_path, "test")
    target_path = Path("targets.csv")
    targets = _target_items(target_path)
    _cache_targets(config, targets)
    candidates = Path("candidates.csv")
    _candidate_source(candidates, schedule_path, "final_eval_v2_candidate_sets")
    args = [
        "--config",
        str(config_path),
        "materialize-final-retrieval-v2-inputs",
        "--split",
        "test",
        "--schedule",
        str(schedule_path),
        "--target-items",
        str(target_path),
        "--candidate-source",
        str(candidates),
    ]
    with pytest.raises(FileNotFoundError, match="query-only materialization"):
        main(args)

    value = np.ones((1, 2), dtype=np.float32)
    materialize_query_embeddings(
        config=config,
        split="test",
        schedule_path=schedule_path,
        builder=lambda _: {
            "query_minilm": value,
            "query_clip_image": value,
            "query_clip_text": value,
        },
        approved=True,
    )
    _candidate_source(candidates, schedule_path, "legacy_v1_candidate_sets")
    with pytest.raises(ValueError, match="legacy v1 outputs are ineligible"):
        main(args)


def test_selected_cases_schema_hashes_and_v1_rejection(tmp_path, monkeypatch) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = repository / "configs/final_eval_v2.yaml"
    monkeypatch.chdir(tmp_path)
    schedule_path = Path("validation.csv")
    _schedule(schedule_path)
    fusion = Path("fusion.json")
    reranking = Path("reranking.json")
    fusion.write_text(json.dumps({"selected_on": "validation"}), encoding="utf-8")
    reranking.write_text(json.dumps({"selected_on": "validation"}), encoding="utf-8")
    source = Path("selected_source.csv")
    frame = pd.DataFrame(
        [
            {
                "paper_case_id": "c1",
                "research_split": "validation",
                "recommended_item_id": "i1",
                "recommended_text": "black pumps",
                "item_evidence_text": "ITEM-1: pumps",
                "rule_evidence_ids": "R001",
                "rule_evidence_text": "R001: formality",
                "packet_source_protocol": "final_eval_v2_selected",
            }
        ]
    )
    frame.to_csv(source, index=False)
    _candidate_source(source, schedule_path, "final_eval_v2_selected_cases")
    # Restore the selected-case contents after the manifest helper's candidate fixture write.
    frame.to_csv(source, index=False)
    source.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "protocol": "final_eval_v2_selected_cases",
                "schedule_hash": file_fingerprint(schedule_path),
                "output_hash": file_fingerprint(source),
            }
        ),
        encoding="utf-8",
    )
    output = Path("outputs/final_eval_v2/materialized/validation/selected_cases.csv")
    args = [
        "--config",
        str(config_path),
        "materialize-final-retrieval-v2-selected-cases",
        "--split",
        "validation",
        "--schedule",
        str(schedule_path),
        "--source-cases",
        str(source),
        "--fusion-selection",
        str(fusion),
        "--reranking-selection",
        str(reranking),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    assert main(args) == 0
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["v1_primary_outputs_used"] is False
    assert manifest["output_hash"] == file_fingerprint(output)
    assert set(pd.read_csv(output).columns) >= {
        "recommended_item_id",
        "item_evidence_text",
        "rule_evidence_text",
        "packet_source_protocol",
    }

    legacy = frame.assign(packet_source_protocol="legacy_v1")
    legacy.to_csv(source, index=False)
    source.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "protocol": "final_eval_v2_selected_cases",
                "schedule_hash": file_fingerprint(schedule_path),
                "output_hash": file_fingerprint(source),
            }
        ),
        encoding="utf-8",
    )
    output2 = Path("outputs/final_eval_v2/materialized/validation/legacy.csv")
    args[-1] = str(output2)
    with pytest.raises(ValueError, match="Old v1 cases"):
        main(args)
