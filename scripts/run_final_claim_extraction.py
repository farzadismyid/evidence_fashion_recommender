"""Run fresh Qwen atomic-claim extraction over accepted final Stage-2 explanations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from evidence_fashion.assessment import extraction_schema
from evidence_fashion.explanation import OllamaClient, text_sha256
from evidence_fashion.extraction_contracts import validate_atomic_claims
from evidence_fashion.manifest import (
    configuration_hash,
    environment_summary,
    git_commit,
    load_resolved_configuration,
    sha256_file,
    utc_timestamp,
    write_new_json,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def _repair_instruction(error: Exception) -> str:
    return (
        "\n\nReturn only valid JSON. Extract every explicit independently verifiable claim, "
        "with exactly claims: [{claim_id, claim_text, claim_type}]. IDs must be C1, C2, and so "
        "on in textual order; do not include support, evidence, truth, or verdict fields. "
        f"Error: {error}"
    )


def _extract_one(
    *,
    client: OllamaClient,
    model_id: str,
    prompt: str,
    system_prompt: str,
    schema: dict[str, Any],
    claim_types: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    repair = ""
    retries = int(config["extraction"]["bounded_retry_attempts"])
    for attempt_number in range(retries + 1):
        active_prompt = prompt + repair
        result = None
        raw = None
        try:
            result = client.generate(
                model_id,
                active_prompt,
                system_prompt=system_prompt,
                json_format=schema,
                token_limit=int(config["inference"]["structured_token_limit"]),
                timeout_seconds=float(config["inference"]["timeout_seconds"]),
            )
            raw = result.text
            payload = json.loads(raw)
            claims = validate_atomic_claims(payload, claim_types=claim_types)
        except Exception as error:  # Keep every invalid response for the final audit.
            attempt = {
                "attempt_number": attempt_number + 1,
                "prompt_hash": text_sha256(active_prompt),
                "error": f"{type(error).__name__}: {error}",
            }
            if result is not None and raw is not None:
                attempt.update(
                    {
                        "raw_response": raw,
                        "response_hash": text_sha256(raw),
                        "latency_seconds": result.latency_seconds,
                        "prompt_eval_count": result.prompt_eval_count,
                        "eval_count": result.eval_count,
                    }
                )
            attempts.append(attempt)
            repair = _repair_instruction(error)
            continue
        attempts.append(
            {
                "attempt_number": attempt_number + 1,
                "prompt_hash": text_sha256(active_prompt),
                "response_hash": text_sha256(raw),
                "latency_seconds": result.latency_seconds,
                "prompt_eval_count": result.prompt_eval_count,
                "eval_count": result.eval_count,
                "claim_count": len(claims),
            }
        )
        return (
            {"status": "accepted", "claims": claims, "attempts_used": attempt_number + 1},
            attempts,
        )
    return {
        "status": "terminal_failure",
        "claims": [],
        "attempts_used": retries + 1,
    }, attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts-config", type=Path, default=Path("configs/prompts.yaml"))
    args = parser.parse_args()
    experiment = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    prompts = yaml.safe_load(args.prompts_config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    resolved["prompts"] = prompts
    config_hash = configuration_hash(resolved)
    explanation_manifests = sorted(
        Path(experiment["paths"]["explanation_runs"]).glob("final-explanations-*/manifest.json")
    )
    if len(explanation_manifests) != 1:
        raise ValueError("Exactly one frozen final Stage-2 explanation manifest is required.")
    explanation_manifest_path = explanation_manifests[0]
    explanation_manifest = json.loads(explanation_manifest_path.read_text(encoding="utf-8"))
    if explanation_manifest["status"] != "complete":
        raise ValueError("Stage-2 explanations are not frozen.")
    explanation_path = Path(
        next(
            path
            for path in explanation_manifest["output_artifact_hashes"]
            if path.endswith("explanations.jsonl")
        )
    )
    explanations = [row for row in _read_jsonl(explanation_path) if row["status"] == "accepted"]
    if len(explanations) != int(explanation_manifest["row_counts"]["accepted_cells"]):
        raise ValueError("Accepted explanation count differs from the frozen Stage-2 manifest.")
    run_dir = Path(experiment["paths"]["extraction_runs"]) / f"final-extraction-{config_hash[:12]}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen extraction run: {run_dir}")
    run_dir.mkdir(parents=True)
    output_path = run_dir / "extractions.jsonl"
    attempts_path = run_dir / "raw_extraction_attempts.jsonl"
    inference = {**models["inference_defaults"]}
    inference["token_limit"] = inference["generation_token_limit"]
    experiment["inference"] = inference
    client = OllamaClient(inference, endpoint=str(inference["endpoint"]))
    role = prompts["roles"]["claim_extraction"]
    claim_types = list(experiment["extraction"]["claim_types"])
    schema = extraction_schema(claim_types)
    accepted = Counter()
    failures = Counter()
    claim_counts: list[int] = []
    retries = Counter()
    with output_path.open("x", encoding="utf-8", newline="\n") as output_handle, attempts_path.open(
        "x", encoding="utf-8", newline="\n"
    ) as attempts_handle:
        try:
            for position, explanation in enumerate(explanations, start=1):
                prompt = str(role["user_template"]).format(
                    claim_types_json=json.dumps(claim_types), explanation=explanation["explanation"]
                )
                result, attempts = _extract_one(
                    client=client,
                    model_id=str(models["extractor"]["model_id"]),
                    prompt=prompt,
                    system_prompt=str(role["system_prompt"]),
                    schema=schema,
                    claim_types=claim_types,
                    config=experiment,
                )
                record = {
                    "case_id": explanation["case_id"],
                    "target_category": explanation["target_category"],
                    "generator_model_id": explanation["generator_model_id"],
                    "generator_model_digest": explanation["generator_model_digest"],
                    "condition": explanation["condition"],
                    "locked_candidate_id": explanation["locked_candidate_id"],
                    "common_context_A": explanation["common_context_A"],
                    "common_context_A_hash": explanation["common_context_A_hash"],
                    "trace_hash": explanation["trace_hash"],
                    "exact_stored_rule_trace_B": explanation["exact_stored_rule_trace_B"],
                    "explanation": explanation["explanation"],
                    "explanation_hash": text_sha256(explanation["explanation"]),
                    "extractor_model_id": models["extractor"]["model_id"],
                    "extractor_model_digest": models["extractor"]["immutable_digest"],
                    "extraction_prompt_hash": text_sha256(prompt),
                    "extraction_system_prompt_hash": text_sha256(str(role["system_prompt"])),
                    **result,
                }
                _append_jsonl(output_handle, record)
                for attempt in attempts:
                    _append_jsonl(
                        attempts_handle,
                        {
                            "case_id": explanation["case_id"],
                            "generator_model_id": explanation["generator_model_id"],
                            "condition": explanation["condition"],
                            **attempt,
                        },
                    )
                key = f"{explanation['generator_model_id']}:{explanation['condition']}"
                retries[result["attempts_used"]] += 1
                if result["status"] == "accepted":
                    accepted[key] += 1
                    claim_counts.append(len(result["claims"]))
                else:
                    failures[key] += 1
                if position % 50 == 0:
                    print(f"claim extraction: {position}/{len(explanations)}", flush=True)
        finally:
            client.unload(str(models["extractor"]["model_id"]))
    if sum(accepted.values()) + sum(failures.values()) != len(explanations):
        raise RuntimeError("Every accepted Stage-2 explanation must have an extraction record.")
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": 2,
        "stage": 3,
        "stage_name": "fresh_atomic_claim_extraction",
        "status": "complete",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": config_hash,
        "input_artifact_hashes": {
            str(explanation_manifest_path): sha256_file(explanation_manifest_path),
            str(explanation_path): sha256_file(explanation_path),
        },
        "output_artifact_hashes": {
            str(output_path): sha256_file(output_path),
            str(attempts_path): sha256_file(attempts_path),
        },
        "models": {"extractor": models["extractor"]},
        "prompt_hashes": {
            "system": text_sha256(str(role["system_prompt"])),
            "template": text_sha256(str(role["user_template"])),
        },
        "row_counts": {
            "accepted_stage2_explanations": len(explanations),
            "extraction_records": sum(accepted.values()) + sum(failures.values()),
            "accepted_extractions": sum(accepted.values()),
            "terminal_failures": sum(failures.values()),
            "total_claims": sum(claim_counts),
            "raw_attempts": sum(1 for _ in attempts_path.read_text(encoding="utf-8").splitlines()),
        },
        "accepted_by_generator_condition": dict(sorted(accepted.items())),
        "failure_counts_by_generator_condition": dict(sorted(failures.items())),
        "claim_count_summary": {
            "mean": sum(claim_counts) / len(claim_counts) if claim_counts else None,
            "median": median(claim_counts) if claim_counts else None,
            "minimum": min(claim_counts) if claim_counts else None,
            "maximum": max(claim_counts) if claim_counts else None,
        },
        "attempts_used_distribution": dict(sorted(retries.items())),
        "contract": {
            "evidence_independent": True,
            "no_support_judgements": True,
            "consecutive_claim_ids": True,
            "duplicate_claims_rejected": True,
            "no_claim_cap": True,
            "terminal_failures_retained": True,
        },
        "environment": environment_summary(),
    }
    write_new_json(manifest_path, manifest)
    print(json.dumps(manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
