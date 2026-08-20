"""Freeze and validate the Stage 4 sequential model-batch policy."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from evidence_fashion.manifest import git_commit, sha256_file, utc_timestamp, write_new_json
from evidence_fashion.model_batches import OPTIONAL_JUDGE, stage4_batches

ROOT = Path(__file__).parents[1]
MODELS_PATH = ROOT / "configs/models.yaml"
PROMPTS_PATH = ROOT / "configs/prompts.yaml"
POLICY_PATH = ROOT / "configs/model_batches.yaml"
STAGE3_PATH = ROOT / "artifacts/manifests/stage3_prompt_freeze_manifest.json"
OUTPUT_PATH = ROOT / "artifacts/manifests/stage4_sequential_batch_manifest.json"
BATCH_IMPLEMENTATION = ROOT / "src/evidence_fashion/model_batches.py"


def main() -> None:
    stage3 = json.loads(STAGE3_PATH.read_text(encoding="utf-8"))
    if stage3.get("status") != "frozen":
        raise ValueError("Stage 3 prompts must be frozen before Stage 4.")
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    batches = stage4_batches(policy)
    payload = {
        "schema_version": 1,
        "stage": 4,
        "status": "frozen",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "execution_mode": "complete_batch_then_unload",
        "max_loaded_models": 1,
        "batches": [batch.__dict__ for batch in batches],
        "pipeline_stop": {"after_batch": batches[-1].batch_id, "reason": "inspection_required"},
        "optional_blind_judge": {
            "batch_id": OPTIONAL_JUDGE[0],
            "model_id": OPTIONAL_JUDGE[1],
            "role": OPTIONAL_JUDGE[2],
            "requires_separate_approval": True,
            "included_in_frozen_default_schedule": False,
        },
        "runtime_enforcement": "evidence_fashion.model_batches.SequentialBatchExecutor",
        "bound_artifact_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                MODELS_PATH,
                PROMPTS_PATH,
                POLICY_PATH,
                STAGE3_PATH,
                BATCH_IMPLEMENTATION,
                Path(__file__),
            )
        },
        "next_gate": (
            "Stage 5 calibration must approve extraction and verification before any full run."
        ),
    }
    write_new_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
