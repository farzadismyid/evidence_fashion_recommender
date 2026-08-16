import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_stage7_explanation_generation",
    ROOT / "scripts/run_stage7_explanation_generation.py",
)
assert SPEC and SPEC.loader
STAGE7 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE7)
build_case_packets = STAGE7.build_case_packets
canonical_hash = STAGE7.canonical_hash
generation_summary = STAGE7.generation_summary
select_stage7_cases = STAGE7.select_stage7_cases
validate_frozen_stage2_to_6 = STAGE7.validate_frozen_stage2_to_6


def _stage6() -> tuple[dict, list[dict]]:
    manifest = json.loads(
        (ROOT / "artifacts/manifests/stage6_recommendation_manifest.json").read_text()
    )
    locked_path = ROOT / next(
        path for path in manifest["output_artifact_hashes"] if path.endswith("locked_cases.jsonl")
    )
    locked = [json.loads(line) for line in locked_path.read_text().splitlines()]
    return manifest, locked


@pytest.mark.skip(reason="Stage 6 inputs have not been regenerated")
def test_stage7_selection_is_balanced_deterministic_and_stage6_frozen() -> None:
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    manifest, locked = _stage6()
    validate_frozen_stage2_to_6(config, manifest)
    first = select_stage7_cases(locked, config)
    second = select_stage7_cases(locked, config)
    assert [row["case_id"] for row in first] == [row["case_id"] for row in second]
    assert len(first) == 500
    assert pd.Series(row["target_category"] for row in first).value_counts().to_dict() == {
        category: 100 for category in config["preprocessing"]["target_categories"]
    }


@pytest.mark.skip(reason="Stage 6 inputs have not been regenerated")
def test_stage7_packets_preserve_exact_five_rule_stage6_trace() -> None:
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    _, locked = _stage6()
    selected = select_stage7_cases(locked, config)
    frozen = json.loads(
        (ROOT / "artifacts/manifests/stage5_frozen_settings.json").read_text()
    )
    packets = build_case_packets(selected, frozen["settings"])
    original = {row["case_id"]: row for row in selected}
    assert len(packets) == 500
    for packet in packets:
        trace = packet["B_exact_stored_trace"]
        assert trace == original[packet["case_id"]]["evidence_trace"]
        assert len(trace["rules"]) == 5
        assert packet["B_sha256"] == canonical_hash(trace)


def test_generation_summary_keeps_conditions_and_generators_separate() -> None:
    records = [
        {
            "generator": generator,
            "condition": condition,
            "word_count": words,
            "latency_seconds": 1.0,
            "retry_count": 0,
            "refusal_detected": False,
            "malformed_or_empty": False,
            "requested_word_limit": 75,
        }
        for generator, words in (("g1", 20), ("g2", 80))
        for condition in ("no_rag", "rule_rag")
    ]
    summary = generation_summary(records)
    assert len(summary) == 6
    assert set(summary["condition"]) == {"no_rag", "rule_rag"}
    assert summary.loc[
        summary["generator"].eq("all_generators")
        & summary["condition"].eq("rule_rag"),
        "word_limit_violations",
    ].iloc[0] == 1
    assert summary.loc[
        summary["generator"].eq("all_generators")
        & summary["condition"].eq("no_rag"),
        "word_limit_violations",
    ].iloc[0] == 1


def test_stage7_manifest_contract_when_present() -> None:
    path = ROOT / "artifacts/manifests/stage7_explanation_generation_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    assert manifest["stage"] == 7
    assert manifest["row_counts"]["selected_cases"] == 500
    assert manifest["row_counts"]["generations"] == 3000
    assert manifest["integrity_checks"]["five_rule_packets"] == 500
    assert manifest["status"]["claim_verification"] == "not_started_stage8"
    for raw_path, expected in manifest["output_artifact_hashes"].items():
        # The registry is an intentionally append-only cross-stage index. Later stages add
        # rows, so the Stage 7 manifest can only bind its immutable Stage 7 artifacts.
        if Path(raw_path).name == "figure_table_registry.csv":
            continue
        artifact = ROOT / raw_path
        assert artifact.exists(), raw_path
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected
