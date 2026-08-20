"""Validate human calibration annotations and assess Qwen/Phi outputs against them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evidence_fashion.calibration import (
    calibration_gates,
    calibration_metrics,
    validate_human_calibration,
)
from evidence_fashion.manifest import sha256_file, utc_timestamp, write_new_json

ROOT = Path(__file__).parents[1]
STAGE3_MANIFEST = ROOT / "artifacts/manifests/stage3_prompt_freeze_manifest.json"
STAGE4_MANIFEST = ROOT / "artifacts/manifests/stage4_sequential_batch_manifest.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/calibration.yaml"))
    parser.add_argument("--model-outputs", type=Path)
    parser.add_argument("--qwen-outputs", type=Path)
    parser.add_argument("--phi-outputs", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen_stages = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (STAGE3_MANIFEST, STAGE4_MANIFEST)
    ]
    if any(manifest.get("status") != "frozen" for manifest in frozen_stages):
        raise ValueError("Stages 3 and 4 must be frozen before Stage 5 calibration.")
    annotation_path = ROOT / settings["calibration"]["annotation_path"]
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Human annotation file is required before Stage 5: {annotation_path.relative_to(ROOT)}"
        )
    human_records = _read_jsonl(annotation_path)
    validation = validate_human_calibration(human_records, settings)
    if args.validate_only:
        print(json.dumps({"status": "human_annotations_valid", **validation}, indent=2))
        return
    if args.qwen_outputs or args.phi_outputs:
        if not args.qwen_outputs or not args.phi_outputs:
            raise ValueError("Provide both sealed Qwen and Phi output paths together.")
        qwen_path = ROOT / args.qwen_outputs
        phi_path = ROOT / args.phi_outputs
        if not qwen_path.exists() or not phi_path.exists():
            raise FileNotFoundError("Sealed Qwen/Phi calibration outputs are required.")
        qwen_rows = {
            (row["calibration_case_id"], row["condition"]): row for row in _read_jsonl(qwen_path)
        }
        phi_rows = {
            (row["calibration_case_id"], row["condition"]): row for row in _read_jsonl(phi_path)
        }
        model_rows = [
            {
                "calibration_case_id": case_id,
                "condition": condition,
                "status": qwen_rows[(case_id, condition)]["status"],
                "claims": qwen_rows[(case_id, condition)]["claims"],
                "entailment": phi_rows[(case_id, condition)]["entailment"],
                "citation_validation": phi_rows[(case_id, condition)]["citation_validation"],
            }
            for case_id, condition in sorted(qwen_rows)
            if (case_id, condition) in phi_rows
        ]
        output_paths = (qwen_path, phi_path)
    else:
        output_path = ROOT / (args.model_outputs or settings["calibration"]["model_output_path"])
        if not output_path.exists():
            raise FileNotFoundError(
                "Qwen/Phi calibration outputs are required for the Stage 5 gate."
            )
        model_rows = _read_jsonl(output_path)
        output_paths = (output_path,)
    metrics = calibration_metrics(human_records, model_rows)
    gates = calibration_gates(metrics, settings)
    if not gates["stage5_pass"]:
        raise ValueError(f"Stage 5 calibration did not pass: {gates}")
    manifest = {
        "schema_version": 1,
        "stage": 5,
        "status": "frozen",
        "timestamp_utc": utc_timestamp(),
        "human_annotation_validation": validation,
        "metrics": metrics,
        "gates": gates,
        "bound_artifact_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                config_path,
                annotation_path,
                *output_paths,
                STAGE3_MANIFEST,
                STAGE4_MANIFEST,
            )
        },
        "next_gate": "Stages 6-8 may begin only with this approved calibration manifest.",
    }
    write_new_json(ROOT / "artifacts/manifests/stage5_calibration_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
