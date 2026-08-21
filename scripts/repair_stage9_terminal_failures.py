"""Boundedly retry only terminal Stage 9 cells under the frozen prompt contract."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from evidence_fashion.explanation import OllamaClient
from evidence_fashion.grounding_contracts import validate_generated_explanation
from evidence_fashion.manifest import sha256_file, write_json
from evidence_fashion.prompt_registry import load_prompt_registry


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
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument("--additional-attempts", type=int, default=3)
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".repairing")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def canonical_json(row: Mapping[str, Any]) -> str:
    return json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def installed_model_digest(endpoint: str, model_id: str) -> str | None:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/api/tags", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return next(
        (
            str(row.get("digest"))
            for row in payload.get("models", [])
            if row.get("name") == model_id
        ),
        None,
    )


def locked_name_from_prompt(prompt: str) -> str:
    prefix = "Required first-sentence wording: include the exact recommended item text, "
    if prefix not in prompt:
        raise ValueError("Stored Stage 9 prompt has no locked-item contract line.")
    return prompt.split(prefix, 1)[1].split(".\n", 1)[0].strip()


def final_failure_reason(attempts: list[Mapping[str, Any]]) -> str:
    return str(attempts[-1].get("validation_error") or "unknown_terminal_failure")


def main() -> None:
    args = parse_args()
    if args.additional_attempts < 1:
        raise ValueError("--additional-attempts must be at least one.")
    output_path = args.run_dir / "explanations.jsonl"
    raw_path = args.run_dir / "raw_generation_attempts.jsonl"
    outputs = read_jsonl(output_path)
    raw = read_jsonl(raw_path)
    failed = [row for row in outputs if row["status"] == "terminal_failure"]
    if not failed:
        print(json.dumps({"initial_terminal_failures": 0, "repaired": 0}, indent=2))
        return
    output_keys = {(r["case_id"], r["condition"], r["generator"]) for r in outputs}
    if len(outputs) != 3000 or len(output_keys) != 3000:
        raise ValueError("Stage 9 output matrix must be complete and unique before repair.")
    raw_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw:
        raw_by_key.setdefault((row["case_id"], row["condition"], row["generator"]), []).append(row)
    for row in failed:
        key = (row["case_id"], row["condition"], row["generator"])
        if key not in raw_by_key:
            raise ValueError(f"Terminal Stage 9 cell has no retained raw attempts: {key}")
    accepted_before = {
        (row["case_id"], row["condition"], row["generator"]): canonical_json(row)
        for row in outputs
        if row["status"] == "success"
    }
    initial_reason_counts = Counter(
        final_failure_reason(raw_by_key[(r["case_id"], r["condition"], r["generator"])])
        for r in failed
    )
    registry = load_prompt_registry(args.prompts)
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    known_models = {str(row["model_id"]): row for row in models["generators"]["roster"]}
    unknown = sorted({str(row["generator"]) for row in failed} - set(known_models))
    if unknown:
        raise ValueError(f"Terminal rows name unknown generator models: {unknown}")
    for model_id in {str(row["generator"]) for row in failed}:
        observed = installed_model_digest(args.ollama_endpoint, model_id)
        expected = str(known_models[model_id]["immutable_digest"])
        if observed != expected:
            raise ValueError(f"Model digest mismatch for {model_id}: {observed} != {expected}")

    defaults = models["generation_defaults"]
    client = OllamaClient(defaults, endpoint=args.ollama_endpoint)
    repaired: dict[tuple[str, str, str], dict[str, Any]] = {}
    repair_raw: list[dict[str, Any]] = []
    active_model: str | None = None
    for failed_row in sorted(
        failed, key=lambda row: (row["generator"], row["condition"], row["case_id"])
    ):
        model = str(failed_row["generator"])
        if active_model is not None and active_model != model:
            client.unload(active_model)
        active_model = model
        key = (failed_row["case_id"], failed_row["condition"], model)
        previous_attempts = raw_by_key[key]
        exemplar = previous_attempts[0]
        role = str(exemplar["role"])
        retry_contract = registry["roles"][role]["retry"]
        # The base and system prompts are the stored, byte-identical Stage 9 prompts.
        base_prompt = str(exemplar["user_prompt"])
        system_prompt = str(exemplar["system_prompt"])
        locked_name = locked_name_from_prompt(base_prompt)
        trace_ids = list(failed_row["trace_rule_ids"])
        for retry_index in range(args.additional_attempts):
            prompt = base_prompt
            if retry_index:
                prompt += "\n\n" + str(retry_contract["retry_instruction"])
            try:
                response = client.generate(
                    model,
                    prompt,
                    system_prompt=system_prompt,
                    token_limit=int(registry["roles"][role]["token_limit"]),
                    timeout_seconds=float(defaults["timeout_seconds"]) * (2**retry_index),
                )
                try:
                    citations = validate_generated_explanation(
                        response.text,
                        locked_item_name=locked_name,
                        target_category=str(failed_row["target_category"]),
                        trace_rule_ids=trace_ids,
                        citations_required=failed_row["condition"] == "rule_rag",
                    )
                    validation_error = None
                except ValueError as error:
                    citations = []
                    validation_error = str(error)
                repair_raw.append(
                    {
                        **exemplar,
                        "attempt": len(previous_attempts) + retry_index,
                        "repair_round": 1,
                        "raw_text": response.text,
                        "latency_seconds": response.latency_seconds,
                        "validation_error": validation_error,
                        "frozen_prompt_reused": True,
                    }
                )
                if validation_error is None:
                    repaired[key] = {
                        **failed_row,
                        "status": "success",
                        "explanation": response.text,
                        "retry_count": len(previous_attempts) + retry_index,
                        "latency_seconds": response.latency_seconds,
                        "prompt_eval_count": response.prompt_eval_count,
                        "eval_count": response.eval_count,
                        "total_duration_ns": response.total_duration_ns,
                        "citation_occurrences": citations,
                        "repair_round": 1,
                    }
                    break
            except Exception as error:  # every failed model call is retained in raw attempts
                repair_raw.append(
                    {
                        **exemplar,
                        "attempt": len(previous_attempts) + retry_index,
                        "repair_round": 1,
                        "raw_text": None,
                        "latency_seconds": None,
                        "validation_error": f"{type(error).__name__}: {error}",
                        "frozen_prompt_reused": True,
                    }
                )
    if active_model is not None:
        client.unload(active_model)

    replaced = [
        repaired.get((row["case_id"], row["condition"], row["generator"]), row)
        for row in outputs
    ]
    accepted_after = {
        (row["case_id"], row["condition"], row["generator"]): canonical_json(row)
        for row in replaced
        if row["status"] == "success"
        and (row["case_id"], row["condition"], row["generator"]) in accepted_before
    }
    if accepted_after != accepted_before:
        raise ValueError("An originally accepted Stage 9 output would be altered by repair.")
    write_jsonl_atomic(output_path, replaced)
    write_jsonl_atomic(raw_path, [*raw, *repair_raw])
    remaining = [row for row in replaced if row["status"] == "terminal_failure"]
    remaining_by_group = Counter((row["generator"], row["condition"]) for row in remaining)
    updated_raw_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in [*raw, *repair_raw]:
        key = (row["case_id"], row["condition"], row["generator"])
        updated_raw_by_key.setdefault(key, []).append(row)
    remaining_reason_counts = Counter(
        final_failure_reason(
            updated_raw_by_key[(row["case_id"], row["condition"], row["generator"])]
        )
        for row in remaining
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["output_artifact_hashes"] = {
        str(output_path): sha256_file(output_path),
        str(raw_path): sha256_file(raw_path),
    }
    manifest["row_counts"]["raw_attempts"] = len(raw) + len(repair_raw)
    manifest["failure_counts"]["terminal_failures"] = len(remaining)
    manifest["recovery"] = {
        "initial_terminal_failures": len(failed),
        "additional_attempts_per_failed_cell": args.additional_attempts,
        "repaired": len(repaired),
        "remaining_terminal_failures": len(remaining),
        "initial_final_failure_reasons": dict(initial_reason_counts),
        "remaining_final_failure_reasons": dict(remaining_reason_counts),
        "remaining_by_model_condition": {
            f"{model} | {condition}": count
            for (model, condition), count in sorted(remaining_by_group.items())
        },
        "mechanism": (
            "retry only former terminal cells with byte-identical frozen Stage 9 base prompts, "
            "the frozen retry instruction, unchanged validators, and raw-attempt retention"
        ),
        "accepted_outputs_unchanged": True,
    }
    write_json(args.run_dir / "manifest.json", manifest)
    write_json(args.manifest, manifest)
    print(json.dumps(manifest["recovery"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
