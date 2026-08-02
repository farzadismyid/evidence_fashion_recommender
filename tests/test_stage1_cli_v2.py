import json
from pathlib import Path

import numpy as np
import pandas as pd

from evidence_fashion_recommender.cli import main


def _write_bundle(root: Path, split: str) -> None:
    root.mkdir(parents=True)
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
    ).to_csv(root / "schedule.csv", index=False)
    pd.DataFrame(
        [
            {
                "paper_case_id": case,
                "candidate_position": position,
                "target_row": target,
                "is_positive": int(position == 0),
                "evidence_score": float(position == 0),
            }
            for case in ("c1", "c2")
            for position, target in enumerate((0, 1))
        ]
    ).to_csv(root / "candidate_sets.csv", index=False)
    target_minilm = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    target_image = target_minilm.copy()
    target_text = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
    query_minilm = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    query_image = query_minilm.copy()
    query_text = query_minilm.copy()
    for name, value in {
        "target_minilm": target_minilm,
        "target_clip_image": target_image,
        "target_clip_text": target_text,
        "query_minilm": query_minilm,
        "query_clip_image": query_image,
        "query_clip_text": query_text,
    }.items():
        np.save(root / f"{name}.npy", value)


def _write_packets(path: Path, recommended: str = "item-1") -> None:
    pd.DataFrame(
        [
            {
                "paper_case_id": case,
                "recommended_item_id": recommended,
                "item_evidence_text": "ITEM-1: pumps",
                "rule_evidence_ids": "R001",
                "rule_evidence_text": "R001: match formality",
            }
            for case in ("c1", "c2")
        ]
    ).to_csv(path, index=False)


def test_stage1_cli_synthetic_bundle_is_resumable(tmp_path, monkeypatch) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = repository / "configs/final_eval_v2.yaml"
    monkeypatch.chdir(tmp_path)
    validation = Path("fixtures/validation")
    test = Path("fixtures/test")
    _write_bundle(validation, "validation")
    _write_bundle(test, "test")
    _write_packets(test / "locked_packets.csv")

    fusion_args = [
        "--config",
        str(config),
        "tune-clip-fusion",
        "--bundle",
        str(validation),
    ]
    assert main(fusion_args) == 0
    assert main(fusion_args) == 0
    fusion_output = Path("outputs/final_eval_v2/validation/fusion_tuning")
    assert (fusion_output / "modality_results.csv").is_file()
    assert (fusion_output / "fusion_weight_validation.csv").is_file()
    selected = json.loads((fusion_output / "selected_fusion.json").read_text())
    assert selected["selected_on"] == "validation"

    reranking = Path("fixtures/selected_weight.json")
    reranking.write_text(
        json.dumps({"selected_on": "validation", "clip_weight": 0.9}), encoding="utf-8"
    )
    retrieval_args = [
        "--config",
        str(config),
        "evaluate-final-retrieval-v2",
        "--bundle",
        str(test),
        "--reranking-selection",
        str(reranking),
        "--locked-packets",
        str(test / "locked_packets.csv"),
    ]
    assert main(retrieval_args) == 0
    assert main(retrieval_args) == 0
    retrieval_output = Path("outputs/final_eval_v2/retrieval/test")
    assert (retrieval_output / "test_ranking_results.csv").is_file()
    packets = pd.read_csv(retrieval_output / "locked_recommendation_evidence_packets.csv")
    assert packets["stage1_packet_hash"].nunique() == 1

    legacy = Path("fixtures/legacy.csv")
    _write_packets(legacy)
    gate_args = [
        "--config",
        str(config),
        "compare-locked-artifacts-v2",
        "--legacy-packets",
        str(legacy),
    ]
    assert main(gate_args) == 0
    assert main(gate_args) == 0
    decision = json.loads(
        Path("outputs/final_eval_v2/decision_gate/decision.json").read_text()
    )
    assert decision["decision"] == "legacy_generation_v2_judging"
