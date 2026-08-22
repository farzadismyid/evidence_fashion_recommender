"""Retry only the final eight short Mistral Stage 9 outputs, then re-freeze Stage 9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from retry_stage9_remaining_length_failures import (
    bound_output,
    canonical_hash,
    canonical_json,
    installed_model_digest,
    read_jsonl,
    rendered_prompt,
    write_jsonl_atomic,
)

from evidence_fashion.explanation import OllamaClient
from evidence_fashion.grounding_contracts import validate_generated_explanation
from evidence_fashion.manifest import sha256_file, utc_timestamp, write_json
from evidence_fashion.prompt_registry import load_prompt_registry, prompt_manifest_fields

MISTRAL = "ministral-3:14b-instruct-2512-q4_K_M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(".runtime/current/explanations/stage9-v3-generation-b691865366b3"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/stage9_explanation_generation_manifest.json"),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("artifacts/manifests/stage9_v3_case_selection_manifest.json"),
    )
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_path = args.run_dir / "explanations.jsonl"
    raw_path = args.run_dir / "raw_generation_attempts.jsonl"
    outputs = read_jsonl(outputs_path)
    raw = read_jsonl(raw_path)
    raw_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw:
        raw_by_key.setdefault((row["case_id"], row["condition"], row["generator"]), []).append(row)
    targets = []
    for row in outputs:
        if row["status"] != "terminal_failure" or row["generator"] != MISTRAL:
            continue
        attempts = raw_by_key[(row["case_id"], row["condition"], row["generator"])]
        if str(attempts[-1].get("validation_error", "")).startswith("Explanation must contain"):
            targets.append(row)
    if len(targets) != 8:
        raise ValueError(
            f"Expected exactly eight short Mistral terminal cells; found {len(targets)}."
        )
    accepted_before = {
        (row["case_id"], row["condition"], row["generator"]): canonical_json(row)
        for row in outputs
        if row["status"] == "success"
    }
    selection_manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    selection_rows = read_jsonl(bound_output(selection_manifest, "condition_inputs.jsonl"))
    selection_by_key = {
        (row["calibration_case_id"], row["condition"]): row for row in selection_rows
    }
    registry = load_prompt_registry(args.prompts)
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    mistral_settings = next(
        row for row in models["generators"]["roster"] if row["model_id"] == MISTRAL
    )
    observed = installed_model_digest(args.ollama_endpoint, MISTRAL)
    if observed != str(mistral_settings["immutable_digest"]):
        raise ValueError("Configured Mistral digest does not match Ollama.")
    client = OllamaClient(models["generation_defaults"], endpoint=args.ollama_endpoint)
    repaired: dict[tuple[str, str, str], dict[str, Any]] = {}
    repair_raw: list[dict[str, Any]] = []
    try:
        for failed in sorted(targets, key=lambda row: (row["condition"], row["case_id"])):
            key = (failed["case_id"], failed["condition"], failed["generator"])
            prior = raw_by_key[key][-1]
            selection = selection_by_key[(failed["case_id"], failed["condition"])]
            if (
                failed["locked_candidate_id"] != selection["locked_candidate_id"]
                or failed["A_sha256"] != canonical_hash(selection["A_common_context"])
                or failed["B_sha256"] != canonical_hash(selection["B_exact_stored_trace"])
            ):
                raise ValueError("Mistral terminal cell differs from its frozen V3 packet.")
            existing = prior.get("raw_text")
            if not isinstance(existing, str) or not existing.strip():
                raise ValueError("Short Mistral output lacks text to rewrite.")
            rendered = rendered_prompt(registry, selection)
            role = str(rendered["role"])
            locked_name = str(selection["A_common_context"]["locked_item_minimal_name"])
            repair = (
                "\n\nFINAL MISTRAL LENGTH REPAIR: Rewrite the existing response to 45–75 words. "
                "Preserve its meaning, the exact locked recommendation, condition, supplied "
                "evidence, and any citations; do not add unsupported detail.\n\n"
                "Existing response:\n"
                + existing
            )
            for retry_index in range(3):
                prompt = str(rendered["user_prompt"]) + repair
                if retry_index:
                    prompt += "\n\n" + str(registry["roles"][role]["retry"]["retry_instruction"])
                try:
                    response = client.generate(
                        MISTRAL,
                        prompt,
                        system_prompt=str(rendered["system_prompt"]),
                        token_limit=int(registry["roles"][role]["token_limit"]),
                        timeout_seconds=float(models["generation_defaults"]["timeout_seconds"])
                        * (2**retry_index),
                    )
                    try:
                        citations = validate_generated_explanation(
                            response.text,
                            locked_item_name=locked_name,
                            target_category=str(failed["target_category"]),
                            trace_rule_ids=list(failed["trace_rule_ids"]),
                            citations_required=failed["condition"] == "rule_rag",
                        )
                        validation_error = None
                    except ValueError as error:
                        citations = []
                        validation_error = str(error)
                    raw_record = {
                        **prompt_manifest_fields(rendered),
                        "case_id": failed["case_id"],
                        "condition": failed["condition"],
                        "generator": MISTRAL,
                        "attempt": int(prior["attempt"]) + retry_index + 1,
                        "completion_repair_round": 5,
                        "repair_scope": "eight_short_mistral_cells_only",
                        "raw_text": response.text,
                        "latency_seconds": response.latency_seconds,
                        "validation_error": validation_error,
                    }
                    repair_raw.append(raw_record)
                    if validation_error is None:
                        repaired[key] = {
                            **failed,
                            "status": "success",
                            "explanation": response.text,
                            "retry_count": int(raw_record["attempt"]),
                            "latency_seconds": response.latency_seconds,
                            "prompt_eval_count": response.prompt_eval_count,
                            "eval_count": response.eval_count,
                            "total_duration_ns": response.total_duration_ns,
                            "citation_occurrences": citations,
                            "accepted_by": "final_three_attempt_mistral_length_repair",
                        }
                        break
                except Exception as error:
                    repair_raw.append(
                        {
                            **prompt_manifest_fields(rendered),
                            "case_id": failed["case_id"],
                            "condition": failed["condition"],
                            "generator": MISTRAL,
                            "attempt": int(prior["attempt"]) + retry_index + 1,
                            "completion_repair_round": 5,
                            "repair_scope": "eight_short_mistral_cells_only",
                            "raw_text": None,
                            "latency_seconds": None,
                            "validation_error": f"{type(error).__name__}: {error}",
                        }
                    )
    finally:
        client.unload(MISTRAL)
    completed = [
        repaired.get((row["case_id"], row["condition"], row["generator"]), row) for row in outputs
    ]
    accepted_after = {
        (row["case_id"], row["condition"], row["generator"]): canonical_json(row)
        for row in completed
        if row["status"] == "success"
        and (row["case_id"], row["condition"], row["generator"]) in accepted_before
    }
    if accepted_after != accepted_before:
        raise ValueError("An existing accepted output would be altered by this repair.")
    write_jsonl_atomic(outputs_path, completed)
    write_jsonl_atomic(raw_path, [*raw, *repair_raw])
    remaining = [row for row in completed if row["status"] == "terminal_failure"]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["output_artifact_hashes"] = {
        str(outputs_path): sha256_file(outputs_path),
        str(raw_path): sha256_file(raw_path),
    }
    manifest["row_counts"]["raw_attempts"] = len(raw) + len(repair_raw)
    manifest["failure_counts"]["terminal_failures"] = len(remaining)
    manifest["stage9_freeze"] = {
        "status": "frozen_after_final_mistral_length_repair",
        "timestamp_utc": utc_timestamp(),
        "targeted_cells": len(targets),
        "maximum_attempts_per_cell": 3,
        "repaired_cells": len(repaired),
        "accepted_matrix_cells": len(completed) - len(remaining),
        "remaining_terminal_failures": len(remaining),
        "accepted_outputs_unchanged": True,
        "no_further_stage9_retry_loops_authorized": True,
    }
    manifest["status"] = (
        "frozen_complete_matrix_with_terminal_failures"
        if remaining
        else "frozen_complete_accepted_matrix"
    )
    write_json(args.run_dir / "manifest.json", manifest)
    write_json(args.manifest, manifest)
    print(json.dumps({"repaired": len(repaired), **manifest["stage9_freeze"]}, indent=2))


if __name__ == "__main__":
    main()
