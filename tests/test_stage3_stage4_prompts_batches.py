from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_fashion.model_batches import SequentialBatchExecutor, stage4_batches
from evidence_fashion.prompt_registry import (
    load_prompt_registry,
    prompt_manifest_fields,
    render_prompt,
)

ROOT = Path(__file__).parents[1]


def test_stage3_registry_has_all_roles_and_renders_exact_prompt_provenance() -> None:
    registry = load_prompt_registry(ROOT / "configs/prompts.yaml")
    rendered = render_prompt(
        registry,
        "no_rag_explanation",
        {
            "user_request": "Recommend a bag.",
            "query_item_minimal_name": "blue blouse",
            "locked_item_minimal_name": "black shoulder bag",
        },
    )
    assert "rule" not in rendered["user_prompt"].lower()
    assert len(rendered["rendered_prompt_sha256"]) == 64
    assert prompt_manifest_fields(rendered)["user_prompt"] == rendered["user_prompt"]
    with pytest.raises(ValueError, match="variables differ"):
        render_prompt(registry, "no_rag_explanation", {})


def test_stage3_freeze_binds_exact_contracts_and_stage4_freeze_is_ordered() -> None:
    stage3_path = ROOT / "artifacts/manifests/stage3_prompt_freeze_manifest.json"
    stage4_path = ROOT / "artifacts/manifests/stage4_sequential_batch_manifest.json"
    stage3 = json.loads(stage3_path.read_text())
    stage4 = json.loads(stage4_path.read_text())
    assert stage3["status"] == stage4["status"] == "frozen"
    assert set(stage3["role_contract_sha256"]) == {
        "no_rag_explanation", "rule_rag_explanation", "claim_extraction",
        "claim_verification", "blind_judge",
    }
    assert [row["model_id"] for row in stage4["batches"]] == [
        "gemma4:12b", "llama3.1:8b-instruct-q8_0",
        "ministral-3:14b-instruct-2512-q4_K_M", "qwen3.5:9b", "phi4:14b",
    ]
    assert stage4["optional_blind_judge"]["included_in_frozen_default_schedule"] is False


def test_stage4_executor_unloads_each_complete_batch_before_the_next_one() -> None:
    import yaml

    policy = yaml.safe_load((ROOT / "configs/model_batches.yaml").read_text())
    batches = stage4_batches(policy)
    seen: list[str] = []
    unloaded: list[str] = []
    events = SequentialBatchExecutor(batches).execute(
        lambda batch: (seen.append(batch.model_id) or {"records": 1}), unloaded.append
    )
    assert seen == unloaded == [batch.model_id for batch in batches]
    assert [event["event"] for event in events].count("model_unloaded") == len(batches)
