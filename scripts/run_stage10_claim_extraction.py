"""Run the Stage 10 Qwen atomic-claim extraction batch only.

This deliberately stops before verification.  It consumes the frozen Stage 9
matrix and retains every accepted extraction plus every rejected raw attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evidence_fashion.assessment import extraction_schema, validate_extraction
from evidence_fashion.explanation import OllamaClient
from evidence_fashion.manifest import (
    configuration_hash,
    environment_summary,
    git_commit,
    load_resolved_configuration,
    sha256_file,
    utc_timestamp,
    write_json,
    write_new_json,
)
from evidence_fashion.prompt_registry import (
    load_prompt_registry,
    prompt_manifest_fields,
    render_prompt,
    text_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument(
        "--stage9-manifest",
        type=Path,
        default=Path("artifacts/manifests/stage9_explanation_generation_manifest.json"),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("artifacts/manifests/stage9_v3_case_selection_manifest.json"),
    )
    parser.add_argument(
        "--runtime-root", type=Path, default=Path(".runtime/current/extraction")
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("artifacts/manifests/stage10_claim_extraction_manifest.json"),
    )
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def bound_output(manifest: Mapping[str, Any], suffix: str) -> tuple[Path, str]:
    matches = [
        (Path(raw_path), str(digest))
        for raw_path, digest in manifest["output_artifact_hashes"].items()
        if raw_path.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Manifest must bind exactly one {suffix} artifact.")
    path, expected_hash = matches[0]
    if not path.exists() or sha256_file(path) != expected_hash:
        raise ValueError(f"Hash-mismatched input artifact: {path}")
    return path, expected_hash


def installed_model_digest(endpoint: str, model_id: str) -> str | None:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/api/tags", timeout=10) as response:
        models = json.loads(response.read().decode("utf-8")).get("models", [])
    return next((str(row.get("digest")) for row in models if row.get("name") == model_id), None)


def validate_stage9_matrix(
    generations: Sequence[Mapping[str, Any]], selection_rows: Sequence[Mapping[str, Any]]
) -> None:
    expected = len(selection_rows) * 3
    keys = {(r["case_id"], r["generator"], r["condition"]) for r in generations}
    if len(selection_rows) != 1000 or len(generations) != expected or len(keys) != expected:
        raise ValueError("Stage 10 requires a complete unique 3,000-row Stage 9 matrix.")
    expected_pairs = {
        (r["calibration_case_id"], r["condition"]): r for r in selection_rows
    }
    if len(expected_pairs) != 1000:
        raise ValueError("The frozen selection condition matrix is incomplete or duplicate.")
    for row in generations:
        source = expected_pairs.get((row["case_id"], row["condition"]))
        if source is None:
            raise ValueError("A Stage 9 output is outside the frozen V3 selection.")
        if row["locked_candidate_id"] != source["locked_candidate_id"]:
            raise ValueError("Stage 9 output does not preserve the selected locked item.")
        if row["A_sha256"] != canonical_hash(source["A_common_context"]):
            raise ValueError("Stage 9 output common context does not match the frozen selection.")
        if row["B_sha256"] != canonical_hash(source["B_exact_stored_trace"]):
            raise ValueError("Stage 9 output trace does not match the frozen selection.")


def extraction_summary(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    records: list[dict[str, Any]] = []
    for generator, condition in sorted(
        {(str(r["generator"]), str(r["condition"])) for r in rows}
    ):
        group = frame[(frame["generator"] == generator) & (frame["condition"] == condition)]
        records.append(
            {
                "generator": generator,
                "condition": condition,
                "explanations": len(group),
                "complete_extractions": int(group["status"].eq("complete").sum()),
                "terminal_failures": int(group["status"].eq("terminal_failure").sum()),
                "generation_failures": int(
                    group["status"].eq("not_applicable_generation_failure").sum()
                ),
                "total_claims": int(group["claim_count"].sum()),
                "mean_claims": float(group["claim_count"].mean()),
                "median_claims": float(group["claim_count"].median()),
                "max_claims": int(group["claim_count"].max()),
                "retried_extractions": int((group["retry_count"] > 0).sum()),
                "normalized_duplicate_outputs": int(
                    group["structural_normalization_applied"].sum()
                ),
            }
        )
    return pd.DataFrame(records)


def review_sample(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One deterministic completed row per category/model/condition stratum (up to 30)."""
    strata: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] == "complete":
            strata.setdefault(
                (str(row["target_category"]), str(row["generator"]), str(row["condition"])), []
            ).append(row)
    selected = []
    for _key, candidates in sorted(strata.items()):
        selected.append(
            min(
                candidates,
                key=lambda row: hashlib.sha256(
                    f"42:{row['case_id']}:{row['generator']}:{row['condition']}".encode()
                ).hexdigest(),
            )
        )
    return selected


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    registry = load_prompt_registry(args.prompts)
    resolved = load_resolved_configuration(args.config, args.models_config)
    stage9_manifest = json.loads(args.stage9_manifest.read_text(encoding="utf-8"))
    selection_manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    generations_path, generations_hash = bound_output(stage9_manifest, "explanations.jsonl")
    selection_path, selection_hash = bound_output(selection_manifest, "condition_inputs.jsonl")
    all_stage9_rows = read_jsonl(generations_path)
    selection_rows = read_jsonl(selection_path)
    validate_stage9_matrix(all_stage9_rows, selection_rows)
    generations = [row for row in all_stage9_rows if row["status"] == "success"]
    frozen_accepted = int(
        stage9_manifest.get("stage9_freeze", {}).get("accepted_matrix_cells", len(generations))
    )
    if len(generations) != frozen_accepted:
        raise ValueError(
            "Accepted Stage 9 output count differs from the frozen Stage 9 completion manifest."
        )
    extractor = models["extractor"]
    observed_digest = installed_model_digest(args.ollama_endpoint, str(extractor["model_id"]))
    if observed_digest != str(extractor["immutable_digest"]):
        raise ValueError(
            f"Configured extractor digest mismatch: {observed_digest} != "
            f"{extractor['immutable_digest']}"
        )
    run_hash = configuration_hash(
        {
            "resolved": resolved,
            "prompts_sha256": sha256_file(args.prompts),
            "stage9_manifest_sha256": sha256_file(args.stage9_manifest),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
        }
    )
    run_id = f"stage10-claim-extraction-{run_hash[:12]}"
    run_dir = args.runtime_root / run_id
    progress_path = run_dir / "extraction_progress.jsonl"
    raw_attempts_path = run_dir / "raw_extraction_attempts.jsonl"
    extractions_path = run_dir / "extractions.jsonl"
    summary_path = run_dir / "claim_extraction_summary.csv"
    sample_path = run_dir / "stratified_manual_review_sample.jsonl"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Immutable Stage 10 output already exists: {manifest_path}")
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Partial Stage 10 output exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    history = read_jsonl(progress_path) if progress_path.exists() else []
    latest = {(r["case_id"], r["generator"], r["condition"]): r for r in history}
    expected_keys = {(r["case_id"], r["generator"], r["condition"]) for r in generations}
    if not set(latest).issubset(expected_keys):
        raise ValueError("Existing extraction progress has a key outside the Stage 9 matrix.")
    defaults = models["generation_defaults"]
    retries = registry["roles"]["claim_extraction"]["retry"]
    maximum_attempts = int(retries["max_attempts"]) + int(retries["repair_attempts"])
    claim_types = list(config["stage8"]["extraction_claim_types"])
    client = OllamaClient(defaults, endpoint=args.ollama_endpoint)
    try:
        for position, generation in enumerate(generations, start=1):
            key = (generation["case_id"], generation["generator"], generation["condition"])
            if key in latest:
                continue
            base = {
                "case_id": generation["case_id"],
                "generator": generation["generator"],
                "condition": generation["condition"],
                "target_category": generation["target_category"],
                "locked_candidate_id": generation["locked_candidate_id"],
                "explanation_sha256": (
                    text_sha256(str(generation["explanation"]))
                    if generation.get("explanation") is not None
                    else None
                ),
                "extractor_model_id": extractor["model_id"],
                "extractor_immutable_digest": extractor["immutable_digest"],
            }
            if generation["status"] != "success":
                record = {
                    **base,
                    "status": "not_applicable_generation_failure",
                    "claims": [],
                    "claim_count": 0,
                    "raw_response_text": None,
                    "raw_response_sha256": None,
                    "latency_seconds": 0.0,
                    "retry_count": 0,
                    "retry_errors": ["upstream_stage9_terminal_failure"],
                    "structural_normalization_applied": False,
                    "prompt_provenance": None,
                }
            else:
                rendered = render_prompt(
                    registry,
                    "claim_extraction",
                    {
                        "claim_types_json": json.dumps(claim_types, ensure_ascii=False),
                        "explanation": str(generation["explanation"]),
                    },
                )
                attempts: list[dict[str, Any]] = []
                validated: list[dict[str, str]] | None = None
                result = None
                raw_payload: Mapping[str, Any] | None = None
                for attempt in range(maximum_attempts + 1):
                    prompt = str(rendered["user_prompt"])
                    if attempt:
                        prompt += "\n\n" + str(retries["retry_instruction"])
                    try:
                        candidate = client.generate(
                            str(extractor["model_id"]),
                            prompt,
                            system_prompt=str(rendered["system_prompt"]),
                            json_format=extraction_schema(claim_types),
                            token_limit=int(defaults["structured_token_limit"]) * (2**attempt),
                            timeout_seconds=float(defaults["timeout_seconds"]) * (2**attempt),
                        )
                        payload = json.loads(candidate.text)
                        checked = validate_extraction(payload, claim_types)
                        attempts.append(
                            {
                                "attempt": attempt,
                                "raw_response_text": candidate.text,
                                "raw_response_sha256": text_sha256(candidate.text),
                                "validation_error": None,
                                "latency_seconds": candidate.latency_seconds,
                            }
                        )
                        validated, result, raw_payload = checked, candidate, payload
                        break
                    except Exception as error:  # retain failed responses/errors for auditability
                        attempts.append(
                            {
                                "attempt": attempt,
                                "raw_response_text": (
                                    candidate.text if "candidate" in locals() else None
                                ),
                                "raw_response_sha256": (
                                    text_sha256(candidate.text)
                                    if "candidate" in locals()
                                    else None
                                ),
                                "validation_error": f"{type(error).__name__}: {error}",
                                "latency_seconds": (
                                    candidate.latency_seconds if "candidate" in locals() else None
                                ),
                            }
                        )
                    finally:
                        if "candidate" in locals():
                            del candidate
                for attempt_row in attempts:
                    append_jsonl(raw_attempts_path, {**base, **attempt_row})
                record = {
                    **base,
                    "status": "complete" if validated is not None else "terminal_failure",
                    "claims": validated or [],
                    "claim_count": len(validated or []),
                    "raw_response_text": result.text if result is not None else None,
                    "raw_response_sha256": (
                        text_sha256(result.text) if result is not None else None
                    ),
                    "latency_seconds": result.latency_seconds if result is not None else 0.0,
                    "retry_count": len(attempts) - 1,
                    "retry_errors": [
                        str(row["validation_error"])
                        for row in attempts
                        if row["validation_error"] is not None
                    ],
                    "structural_normalization_applied": bool(
                        raw_payload is not None
                        and canonical_hash(raw_payload.get("claims", []))
                        != canonical_hash(validated)
                    ),
                    "prompt_provenance": prompt_manifest_fields(rendered),
                }
            append_jsonl(progress_path, record)
            latest[key] = record
            if position % 50 == 0:
                print(f"stage10 extraction {position}/{len(generations)}", flush=True)
    finally:
        client.unload(str(extractor["model_id"]))
    if set(latest) != expected_keys:
        raise RuntimeError("Stage 10 extraction matrix is incomplete.")
    extractions = [latest[(r["case_id"], r["generator"], r["condition"])] for r in generations]
    write_jsonl(extractions_path, extractions)
    table = extraction_summary(extractions)
    table.to_csv(summary_path, index=False)
    write_jsonl(sample_path, review_sample(extractions))
    retry_rows = read_jsonl(raw_attempts_path) if raw_attempts_path.exists() else []
    integrity = {
        "complete_unique_accepted_explanation_matrix": len(extractions) == len(generations)
        and len({(r["case_id"], r["generator"], r["condition"]) for r in extractions})
        == len(generations),
        "claim_id_integrity": all(
            row["status"] != "complete"
            or [claim["claim_id"] for claim in row["claims"]]
            == [f"C{i}" for i in range(1, row["claim_count"] + 1)]
            for row in extractions
        ),
        "no_empty_completed_claim_arrays": all(
            row["status"] != "complete" or row["claim_count"] > 0 for row in extractions
        ),
        "all_stage9_output_hashes_preserved": all(
            row["explanation_sha256"]
            == text_sha256(str(generation["explanation"]))
            for row, generation in zip(extractions, generations, strict=True)
            if generation.get("explanation") is not None
        ),
    }
    if not all(integrity.values()):
        raise ValueError(f"Stage 10 integrity checks failed: {integrity}")
    outputs = [extractions_path, summary_path, sample_path]
    if raw_attempts_path.exists():
        outputs.append(raw_attempts_path)
    manifest = {
        "schema_version": 1,
        "stage": 10,
        "stage_name": "atomic_claim_extraction",
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": run_hash,
        "resolved_configuration": resolved,
        "prompt_registry_sha256": sha256_file(args.prompts),
        "input_artifact_hashes": {
            str(args.stage9_manifest): sha256_file(args.stage9_manifest),
            str(generations_path): generations_hash,
            str(args.selection_manifest): sha256_file(args.selection_manifest),
            str(selection_path): selection_hash,
        },
        "output_artifact_hashes": {str(path): sha256_file(path) for path in outputs},
        "model": extractor,
        "row_counts": {
            "stage9_matrix_cells": len(all_stage9_rows),
            "stage9_accepted_explanations": len(generations),
            "extractions": len(extractions),
            "extracted_claims": sum(row["claim_count"] for row in extractions),
            "stratified_manual_review_examples": len(review_sample(extractions)),
            "raw_attempts": len(retry_rows),
        },
        "failure_counts": {
            "extraction_terminal_failures": sum(
                row["status"] == "terminal_failure" for row in extractions
            ),
            "upstream_generation_failures": sum(
                row["status"] == "not_applicable_generation_failure" for row in extractions
            ),
            "retried_extractions": sum(row["retry_count"] > 0 for row in extractions),
            "normalized_duplicate_outputs": sum(
                row["structural_normalization_applied"] for row in extractions
            ),
            "retry_error_types": dict(
                Counter(
                    str(row["validation_error"]).split(":", 1)[0]
                    for row in retry_rows
                    if row["validation_error"]
                )
            ),
        },
        "integrity_checks": integrity,
        "status": "complete_claim_extraction_only_verification_not_started",
        "safeguards": {
            "complete_frozen_accepted_matrix_required": True,
            "claim_ids_consecutive_in_textual_order": True,
            "bounded_structured_retries": maximum_attempts,
            "terminal_failures_retained": True,
            "raw_response_retention": True,
            "no_verification_or_judging_model_calls": True,
        },
        "environment": environment_summary(),
        "command": (
            "python scripts/run_stage10_claim_extraction.py --config configs/experiment.yaml "
            "--models-config configs/models.yaml --prompts configs/prompts.yaml "
            "--stage9-manifest artifacts/manifests/stage9_explanation_generation_manifest.json "
            "--selection-manifest artifacts/manifests/stage9_v3_case_selection_manifest.json"
        ),
    }
    write_new_json(manifest_path, manifest)
    write_json(args.output_manifest, manifest)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "extractions": len(extractions),
                "extracted_claims": manifest["row_counts"]["extracted_claims"],
                "terminal_failures": manifest["failure_counts"]["extraction_terminal_failures"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
