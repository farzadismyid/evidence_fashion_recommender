"""Freeze the canonical runtime destination contract required by Stage 6."""

from __future__ import annotations

import json
from pathlib import Path

from evidence_fashion.manifest import sha256_file, utc_timestamp, write_new_json

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = Path(__file__).resolve()
STAGE5_MANIFEST = ROOT / "artifacts/manifests/stage5_calibration_manifest.json"
DESTINATIONS = (
    ".runtime/current/data",
    ".runtime/current/embeddings",
    ".runtime/current/recommendations",
    ".runtime/current/explanations",
    ".runtime/current/extraction",
    ".runtime/current/verification",
    ".runtime/current/judging",
)


def main() -> None:
    stage5 = json.loads(STAGE5_MANIFEST.read_text(encoding="utf-8"))
    if stage5.get("status") != "frozen":
        raise ValueError("Stage 5 must be frozen before Stage 6.")
    for relative in DESTINATIONS:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "stage": 6,
        "stage_name": "canonical_runtime_destinations",
        "status": "frozen",
        "timestamp_utc": utc_timestamp(),
        "canonical_destinations": list(DESTINATIONS),
        "lifecycle": {
            "working_output": "resumable temporary location",
            "completed_output": "atomically replace canonical destination",
            "failed_or_partial_output": "must not be represented as completed",
        },
        "bound_artifact_hashes": {
            str(STAGE5_MANIFEST.relative_to(ROOT)): sha256_file(STAGE5_MANIFEST),
            str(SCRIPT_PATH.relative_to(ROOT)): sha256_file(SCRIPT_PATH),
        },
        "next_stage": "Stage 7 regenerate data, embeddings, and recommendation results",
    }
    output = ROOT / "artifacts/manifests/stage6_runtime_destinations_manifest.json"
    write_new_json(output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
