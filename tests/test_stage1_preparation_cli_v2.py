import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evidence_fashion_recommender.cli import main
from evidence_fashion_recommender.evaluation.stage1_preparation import prepare_stage1_bundle


def _sources(root: Path, split: str) -> dict[str, Path]:
    root.mkdir(parents=True)
    schedule = root / "schedule.csv"
    candidates = root / "candidate_sets.csv"
    pd.DataFrame(
        [
            {
                "paper_case_id": "c1",
                "query_outfit_id": "o1",
                "target_category": "shoes",
                "research_split": split,
            },
            {
                "paper_case_id": "c2",
                "query_outfit_id": "o2",
                "target_category": "shoes",
                "research_split": split,
            },
        ]
    ).to_csv(schedule, index=False)
    pd.DataFrame(
        [
            {
                "paper_case_id": case,
                "candidate_position": position,
                "target_row": row,
                "is_positive": int(position == 0),
                "evidence_score": float(position == 0),
            }
            for case in ("c1", "c2")
            for position, row in enumerate((0, 1))
        ]
    ).to_csv(candidates, index=False)
    target_dir = root / "target"
    query_dir = root / "query"
    target_dir.mkdir()
    query_dir.mkdir()
    target = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    for name in ("target_minilm", "target_clip_image", "target_clip_text"):
        np.save(target_dir / f"{name}.npy", target)
    for name in ("query_minilm", "query_clip_image", "query_clip_text"):
        np.save(query_dir / f"{name}.npy", query)
    return {
        "schedule": schedule,
        "candidates": candidates,
        "target_dir": target_dir,
        "query_dir": query_dir,
    }


def test_preparation_reranking_and_packet_cli_are_resumable(tmp_path, monkeypatch) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = repository / "configs/final_eval_v2.yaml"
    monkeypatch.chdir(tmp_path)
    source = _sources(Path("fixtures"), "validation")
    bundle = Path("outputs/final_eval_v2/prepared/validation")
    prepare_args = [
        "--config",
        str(config),
        "prepare-final-retrieval-v2-bundle",
        "--split",
        "validation",
        "--schedule",
        str(source["schedule"]),
        "--candidate-sets",
        str(source["candidates"]),
        "--target-embedding-dir",
        str(source["target_dir"]),
        "--query-embedding-dir",
        str(source["query_dir"]),
        "--output-dir",
        str(bundle),
    ]
    assert main(prepare_args) == 0
    assert main(prepare_args) == 0
    preparation = json.loads((bundle / "preparation_manifest.json").read_text())
    assert preparation["embeddings_rebuilt"] is False

    fusion_dir = Path("outputs/final_eval_v2/validation/fusion_tuning")
    fusion_dir.mkdir(parents=True)
    fusion = fusion_dir / "selected_fusion.json"
    fusion.write_text(
        json.dumps({"selected_on": "validation", "image_weight": 0.6}), encoding="utf-8"
    )
    rerank_args = [
        "--config",
        str(config),
        "tune-reranking-v2",
        "--bundle",
        str(bundle),
        "--fusion-selection",
        str(fusion),
    ]
    assert main(rerank_args) == 0
    assert main(rerank_args) == 0
    rerank_dir = Path("outputs/final_eval_v2/validation/reranking_tuning")
    selected = json.loads((rerank_dir / "selected_weight.json").read_text())
    assert selected["selected_on"] == "validation"
    assert selected["config_hash"]
    assert selected["schedule_hash"]
    assert selected["metric_hierarchy"] == [
        "ndcg_at_10",
        "hit_rate_at_10",
        "reciprocal_rank",
    ]
    assert selected["validation_results_path"].endswith("validation_results.csv")

    cases = Path("fixtures/selected_cases.csv")
    pd.DataFrame(
        [
            {
                "paper_case_id": case,
                "research_split": "validation",
                "packet_source_protocol": "final_eval_v2_selected",
                "recommended_item_id": "item-1",
                "item_evidence_text": "ITEM-1: pumps",
                "rule_evidence_ids": "R001",
                "rule_evidence_text": "R001: match formality",
            }
            for case in ("c1", "c2")
        ]
    ).to_csv(cases, index=False)
    packet_path = Path("outputs/final_eval_v2/prepared/validation/locked_packets.csv")
    packet_args = [
        "--config",
        str(config),
        "create-locked-packets-v2",
        "--split",
        "validation",
        "--source-cases",
        str(cases),
        "--fusion-selection",
        str(fusion),
        "--reranking-selection",
        str(rerank_dir / "selected_weight.json"),
        "--output",
        str(packet_path),
    ]
    assert main(packet_args) == 0
    assert main(packet_args) == 0
    packets = pd.read_csv(packet_path)
    assert set(packets["stage1_packet_protocol"]) == {"final_eval_v2_selected"}
    assert packets["generation_packet_hash"].str.len().eq(64).all()
    assert packets["stage1_packet_hash"].nunique() == 1


def test_bundle_preparation_never_rebuilds_missing_embeddings(tmp_path) -> None:
    source = _sources(tmp_path / "fixtures", "test")
    (source["query_dir"] / "query_clip_image.npy").unlink()
    with pytest.raises(FileNotFoundError, match="no embeddings will be rebuilt"):
        prepare_stage1_bundle(
            split="test",
            schedule_path=source["schedule"],
            candidate_sets_path=source["candidates"],
            target_embedding_dir=source["target_dir"],
            query_embedding_dir=source["query_dir"],
            output_dir=tmp_path / "bundle",
        )
