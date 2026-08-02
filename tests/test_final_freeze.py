import json
from pathlib import Path

import pytest

from evidence_fashion_recommender.final_freeze import create_final_eval_v2_freeze


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else str(value), encoding="utf-8")
    return path


def test_freeze_requires_all_validation_selections_and_stage1_binding(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _write(Path("resolved.yaml"), "config")
    fusion = _write(Path("fusion.json"), {"selected_on": "validation", "image_weight": 0.6})
    reranking = _write(Path("reranking.json"), {"selected_on": "validation", "clip_weight": 0.9})
    hybrid = _write(
        Path("hybrid.json"),
        {
            "selected_on": "validation",
            "candidate_type": "hybrid",
            "item_count": 2,
            "stage1_packet_protocol": "final_eval_v2_selected",
            "stage1_packet_hash": "packet-hash",
        },
    )
    schedule = _write(Path("validation.csv"), "case")
    cases = _write(Path("cases.csv"), "case")
    kb = _write(Path("kb.csv"), "rule")
    lock = _write(Path("uv.lock"), "lock")
    prompt = _write(Path("prompt.txt"), "prompt")
    manifest = create_final_eval_v2_freeze(
        destination=Path("outputs/final_eval_v2/freeze"),
        resolved_config=config,
        fusion_selection=fusion,
        reranking_selection=reranking,
        hybrid_selection=hybrid,
        schedules=[schedule],
        cases=[cases],
        knowledge_base=kb,
        dependency_lock=lock,
        prompt_files=[prompt],
        command_list=["command"],
        expected_stage1_packet_hash="packet-hash",
        gate_definition={"max_changed_rate": 0.0},
        source_state={"commit": "abc", "dirty": False},
    )
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    assert frozen["source"] == {"commit": "abc", "dirty": False}
    assert frozen["selections"]["hybrid"]["value"]["item_count"] == 2


def test_freeze_rejects_legacy_hybrid_packets(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("resolved.yaml", "schedule.csv", "cases.csv", "kb.csv", "uv.lock", "prompt"):
        _write(Path(name), name)
    fusion = _write(Path("fusion.json"), {"selected_on": "validation"})
    reranking = _write(Path("reranking.json"), {"selected_on": "validation"})
    hybrid = _write(
        Path("hybrid.json"),
        {
            "selected_on": "validation",
            "candidate_type": "hybrid",
            "item_count": 5,
            "stage1_packet_protocol": "legacy_v1_packets_only",
            "stage1_packet_hash": "old",
        },
    )
    with pytest.raises(ValueError, match="legacy-only"):
        create_final_eval_v2_freeze(
            destination=Path("outputs/final_eval_v2/freeze"),
            resolved_config=Path("resolved.yaml"),
            fusion_selection=fusion,
            reranking_selection=reranking,
            hybrid_selection=hybrid,
            schedules=[Path("schedule.csv")],
            cases=[Path("cases.csv")],
            knowledge_base=Path("kb.csv"),
            dependency_lock=Path("uv.lock"),
            prompt_files=[Path("prompt")],
            command_list=[],
            expected_stage1_packet_hash="new",
            gate_definition={},
            source_state={"commit": "abc", "dirty": False},
        )


def test_freeze_accepts_stage2_packet_source_protocol(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("resolved.yaml", "schedule.csv", "cases.csv", "kb.csv", "uv.lock", "prompt"):
        _write(Path(name), name)
    fusion = _write(Path("fusion.json"), {"selected_on": "validation"})
    reranking = _write(Path("reranking.json"), {"selected_on": "validation"})
    hybrid = _write(
        Path("hybrid.json"),
        {
            "selected_on": "validation",
            "candidate_type": "hybrid",
            "item_count": 2,
            "packet_source_protocol": "final_eval_v2_selected",
            "stage1_packet_hash": "packet",
        },
    )
    manifest = create_final_eval_v2_freeze(
        destination=Path("outputs/final_eval_v2/freeze"),
        resolved_config=Path("resolved.yaml"),
        fusion_selection=fusion,
        reranking_selection=reranking,
        hybrid_selection=hybrid,
        schedules=[Path("schedule.csv")],
        cases=[Path("cases.csv")],
        knowledge_base=Path("kb.csv"),
        dependency_lock=Path("uv.lock"),
        prompt_files=[Path("prompt")],
        command_list=[],
        expected_stage1_packet_hash="packet",
        gate_definition={},
        source_state={"commit": "abc", "dirty": False},
    )
    assert manifest.is_file()
