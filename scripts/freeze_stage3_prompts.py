"""Freeze the canonical Stage 3 prompt registry after the Stage 1-2 gates."""

from __future__ import annotations

import json
from pathlib import Path

from evidence_fashion.manifest import git_commit, sha256_file, utc_timestamp, write_new_json
from evidence_fashion.prompt_registry import canonical_sha256, load_prompt_registry, text_sha256

ROOT = Path(__file__).parents[1]
PROMPTS_PATH = ROOT / "configs/prompts.yaml"
MODELS_PATH = ROOT / "configs/models.yaml"
STAGE1_PATH = ROOT / "artifacts/manifests/stage1_taxonomy_freeze_manifest.json"
STAGE2_PATH = ROOT / "artifacts/manifests/stage2_kb_freeze_manifest.json"
OUTPUT_PATH = ROOT / "artifacts/manifests/stage3_prompt_freeze_manifest.json"
REGISTRY_IMPLEMENTATION = ROOT / "src/evidence_fashion/prompt_registry.py"


def main() -> None:
    stage1 = json.loads(STAGE1_PATH.read_text(encoding="utf-8"))
    stage2 = json.loads(STAGE2_PATH.read_text(encoding="utf-8"))
    if stage1.get("status") != "frozen" or stage2.get("status") != "frozen":
        raise ValueError("Stages 1 and 2 must be frozen before freezing prompts.")
    registry = load_prompt_registry(PROMPTS_PATH)
    roles = registry["roles"]
    payload = {
        "schema_version": 1,
        "stage": 3,
        "status": "frozen",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "prompt_registry_schema_version": registry["schema_version"],
        "prompt_registry_sha256": sha256_file(PROMPTS_PATH),
        "role_contract_sha256": {name: canonical_sha256(role) for name, role in roles.items()},
        "template_sha256": {
            name: {
                "system_prompt": text_sha256(role["system_prompt"]),
                "user_template": text_sha256(role["user_template"]),
            }
            for name, role in roles.items()
        },
        "rendered_prompt_manifest_contract": {
            "required_per_completed_model_call": [
                "role",
                "system_prompt",
                "user_prompt",
                "system_prompt_sha256",
                "user_prompt_sha256",
                "rendered_prompt_sha256",
                "role_contract_sha256",
            ],
            "implementation": "evidence_fashion.prompt_registry.prompt_manifest_fields",
        },
        "bound_artifact_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                PROMPTS_PATH,
                MODELS_PATH,
                STAGE1_PATH,
                STAGE2_PATH,
                REGISTRY_IMPLEMENTATION,
                Path(__file__),
            )
        },
        "blind_judge": {
            "enabled": roles["blind_judge"].get("enabled", False),
            "approval_required": True,
        },
        "next_gate": "Stage 4 must enforce complete sequential model batches before calibration.",
    }
    write_new_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
