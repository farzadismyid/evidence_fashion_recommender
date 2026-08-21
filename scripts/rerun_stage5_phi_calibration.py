"""Regenerate only sealed Stage 5 Phi outputs after a calibration reliability fix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from prepare_stage5_human_calibration import _run_phi_batch, _write_jsonl_new

from evidence_fashion.explanation import OllamaClient
from evidence_fashion.manifest import sha256_file, utc_timestamp, write_new_json
from evidence_fashion.prompt_registry import load_prompt_registry

ROOT = Path(__file__).parents[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet", type=Path, default=Path("data/calibration/stage5_annotations_done.jsonl")
    )
    parser.add_argument(
        "--qwen-outputs",
        type=Path,
        default=Path(
            ".runtime/current/calibration/stage5-aad7353c3f4b/qwen_claim_extraction_sealed.jsonl"
        ),
    )
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts-config", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    packet_path = ROOT / args.packet
    qwen_path = ROOT / args.qwen_outputs
    models_path = ROOT / args.models_config
    prompts_path = ROOT / args.prompts_config
    records = _read_jsonl(packet_path)
    qwen = _read_jsonl(qwen_path)
    qwen_by_key = {(row["calibration_case_id"], row["condition"]): row for row in qwen}
    if len(qwen_by_key) != len(records):
        raise ValueError("Sealed Qwen outputs must contain exactly one row per calibration record.")
    if any((row["calibration_case_id"], row["condition"]) not in qwen_by_key for row in records):
        raise ValueError("Sealed Qwen outputs do not cover the completed calibration packet.")

    run_identity = {
        "purpose": "phi_only_regeneration_after_completeness_fix",
        "packet_sha256": sha256_file(packet_path),
        "qwen_sha256": sha256_file(qwen_path),
        "models_sha256": sha256_file(models_path),
        "prompts_sha256": sha256_file(prompts_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "phi_implementation_sha256": sha256_file(
            ROOT / "scripts/prepare_stage5_human_calibration.py"
        ),
        "assessment_contract_sha256": sha256_file(ROOT / "src/evidence_fashion/assessment.py"),
    }
    run_id = hashlib.sha256(json.dumps(run_identity, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = ROOT / ".runtime/current/calibration" / f"stage5-phi-{run_id}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite sealed Phi rerun: {run_dir}")
    run_dir.mkdir(parents=True)

    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    registry = load_prompt_registry(prompts_path)
    client = OllamaClient(models["generation_defaults"], endpoint=args.ollama_endpoint)
    phi, phi_raw = _run_phi_batch(client, registry, models, records, qwen_by_key)
    phi_path = run_dir / "phi_verification_sealed.jsonl"
    raw_path = run_dir / "phi_raw_responses_sealed.jsonl"
    _write_jsonl_new(phi_path, phi)
    _write_jsonl_new(raw_path, phi_raw)
    provenance = {
        "schema_version": 1,
        "stage": 5,
        "status": "phi_regenerated_pending_calibration",
        "timestamp_utc": utc_timestamp(),
        "run_id": run_id,
        "purpose": run_identity["purpose"],
        "sealed_outputs": {
            "qwen_claim_extraction_reused": str(qwen_path.relative_to(ROOT)),
            "phi_verification": str(phi_path.relative_to(ROOT)),
            "phi_raw_responses": str(raw_path.relative_to(ROOT)),
        },
        "bound_artifact_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                packet_path,
                qwen_path,
                models_path,
                prompts_path,
                Path(__file__),
                phi_path,
                raw_path,
            )
        },
    }
    write_new_json(run_dir / "provenance.json", provenance)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
