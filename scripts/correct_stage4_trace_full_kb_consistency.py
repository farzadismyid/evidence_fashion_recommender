"""Apply the authorised Stage-4 trace-subset correction in place without model calls."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evidence_fashion.manifest import sha256_file, utc_timestamp, write_json
from evidence_fashion.verification_corrections import enforce_trace_implies_full_kb

RUN_DIR = Path(".runtime/current/verification/final-verification-955ab963af41")
VERIFICATIONS = RUN_DIR / "verifications.jsonl"
RUN_MANIFEST = RUN_DIR / "manifest.json"
STAGE_MANIFEST = Path("artifacts/manifests/final_stage4_manifest.json")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl_in_place(path: Path, records: list[dict[str, Any]]) -> None:
    """Replace the explicitly authorised canonical file; no derived dataset is created."""
    temporary = path.with_suffix(path.suffix + ".correction.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary correction file: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    records = _read_jsonl(VERIFICATIONS)
    original_hash = sha256_file(VERIFICATIONS)
    corrected = enforce_trace_implies_full_kb(records)
    _write_jsonl_in_place(VERIFICATIONS, records)
    corrected_hash = sha256_file(VERIFICATIONS)
    run_manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    run_manifest["output_artifact_hashes"][str(VERIFICATIONS)] = corrected_hash
    run_manifest["verdict_counts"] = dict(
        sorted(
            Counter(
                f"{field}:{claim[field]}"
                for record in records
                if record["status"] == "accepted"
                for claim in record["claims"]
                for field in (
                    "trace_support",
                    "full_kb_support",
                    "common_reference_support",
                    "citation_entailment",
                )
            ).items()
        )
    )
    run_manifest["deterministic_logical_consistency_correction"] = {
        "applied_at_utc": utc_timestamp(),
        "rule": "trace_support_supported_implies_full_kb_support_supported",
        "corrected_claims": corrected,
        "pre_correction_verifications_sha256": original_hash,
        "post_correction_verifications_sha256": corrected_hash,
        "trace_rule_packet_containment_required": True,
        "model_calls": 0,
    }
    write_json(RUN_MANIFEST, run_manifest)
    stage_manifest = json.loads(STAGE_MANIFEST.read_text(encoding="utf-8"))
    stage_manifest["stage4_manifest"]["sha256"] = sha256_file(RUN_MANIFEST)
    stage_manifest["frozen_outputs"]["verifications"]["sha256"] = corrected_hash
    stage_manifest["deterministic_logical_consistency_correction"] = {
        "rule": "trace_support_supported_implies_full_kb_support_supported",
        "corrected_claims": corrected,
        "canonical_file_updated_in_place": True,
        "model_calls": 0,
    }
    write_json(STAGE_MANIFEST, stage_manifest)
    print(json.dumps({"corrected_claims": corrected, "verification_sha256": corrected_hash}))


if __name__ == "__main__":
    main()
