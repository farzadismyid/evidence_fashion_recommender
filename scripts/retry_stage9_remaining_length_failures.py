"""Run one final current-contract retry for terminal Stage 9 length failures only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from evidence_fashion.explanation import OllamaClient
from evidence_fashion.grounding_contracts import validate_generated_explanation
from evidence_fashion.manifest import sha256_file, write_json
from evidence_fashion.prompt_registry import (
    load_prompt_registry,
    prompt_manifest_fields,
    render_prompt,
)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".retrying")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def canonical_json(row: Mapping[str, Any]) -> str:
    return json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bound_output(manifest: Mapping[str, Any], suffix: str) -> Path:
    matches = [Path(path) for path in manifest["output_artifact_hashes"] if path.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Selection manifest must bind exactly one {suffix} artifact.")
    path = matches[0]
    expected = manifest["output_artifact_hashes"][str(path)]
    if not path.exists() or sha256_file(path) != expected:
        raise ValueError(f"Hash-mismatched V3 selection artifact: {path}")
    return path


def rule_evidence(trace: Mapping[str, Any]) -> str:
    return "\n".join(f"[{rule['rule_id']}] {rule['rule_text']}" for rule in trace["rules"])


def rendered_prompt(registry: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    context = selection["A_common_context"]
    if selection["condition"] == "no_rag":
        return render_prompt(registry, "no_rag_explanation", context)
    return render_prompt(
        registry,
        "rule_rag_explanation",
        {**context, "rule_evidence": rule_evidence(selection["B_exact_stored_trace"])},
    )


def installed_model_digest(endpoint: str, model_id: str) -> str | None:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/api/tags", timeout=10) as response:
        installed = json.loads(response.read().decode("utf-8")).get("models", [])
    return next(
        (str(row.get("digest")) for row in installed if str(row.get("name")) == model_id),
        None,
    )


def last_attempt(
    raw_by_key: Mapping[tuple[str, str, str], list[dict[str, Any]]], row: Mapping[str, Any]
) -> dict[str, Any]:
    key = (str(row["case_id"]), str(row["condition"]), str(row["generator"]))
    attempts = raw_by_key.get(key)
    if not attempts:
        raise ValueError(f"No raw attempts retained for {key}")
    return attempts[-1]


def main() -> None:
    args = parse_args()
    outputs_path = args.run_dir / "explanations.jsonl"
    raw_path = args.run_dir / "raw_generation_attempts.jsonl"
    outputs = read_jsonl(outputs_path)
    raw = read_jsonl(raw_path)
    selection_manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    selection_path = bound_output(selection_manifest, "condition_inputs.jsonl")
    selections = read_jsonl(selection_path)
    selection_by_key = {(r["calibration_case_id"], r["condition"]): r for r in selections}
    if len(outputs) != 3000 or len(selection_by_key) != 1000:
        raise ValueError("Stage 9 output or frozen selection matrix is incomplete.")
    raw_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw:
        raw_by_key.setdefault((row["case_id"], row["condition"], row["generator"]), []).append(row)
    targets = []
    for row in outputs:
        if row["status"] != "terminal_failure":
            continue
        reason = str(last_attempt(raw_by_key, row).get("validation_error") or "")
        if reason.startswith("Explanation must contain"):
            targets.append(row)
    if len(targets) != 44:
        raise ValueError(f"Expected 44 terminal length failures; found {len(targets)}.")
    accepted_before = {
        (r["case_id"], r["condition"], r["generator"]): canonical_json(r)
        for r in outputs
        if r["status"] == "success"
    }
    registry = load_prompt_registry(args.prompts)
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    settings = {str(row["model_id"]): row for row in models["generators"]["roster"]}
    for model in {str(row["generator"]) for row in targets}:
        observed = installed_model_digest(args.ollama_endpoint, model)
        expected = str(settings[model]["immutable_digest"])
        if observed != expected:
            raise ValueError(f"Model digest mismatch for {model}.")
    client = OllamaClient(models["generation_defaults"], endpoint=args.ollama_endpoint)
    repaired: dict[tuple[str, str, str], dict[str, Any]] = {}
    repair_raw: list[dict[str, Any]] = []
    active_model: str | None = None
    for failed in sorted(
        targets, key=lambda row: (row["generator"], row["condition"], row["case_id"])
    ):
        model = str(failed["generator"])
        if active_model is not None and active_model != model:
            client.unload(active_model)
        active_model = model
        selection = selection_by_key.get((failed["case_id"], failed["condition"]))
        if selection is None:
            raise ValueError("A terminal cell is outside the frozen V3 selection.")
        if (
            failed["locked_candidate_id"] != selection["locked_candidate_id"]
            or failed["A_sha256"] != canonical_hash(selection["A_common_context"])
            or failed["B_sha256"] != canonical_hash(selection["B_exact_stored_trace"])
        ):
            raise ValueError("A terminal cell does not preserve its frozen V3 input packet.")
        rendered = rendered_prompt(registry, selection)
        prior = last_attempt(raw_by_key, failed)
        try:
            response = client.generate(
                model,
                str(rendered["user_prompt"]),
                system_prompt=str(rendered["system_prompt"]),
                token_limit=int(registry["roles"][str(rendered["role"])]["token_limit"]),
                timeout_seconds=float(models["generation_defaults"]["timeout_seconds"]),
            )
            try:
                citations = validate_generated_explanation(
                    response.text,
                    locked_item_name=str(selection["A_common_context"]["locked_item_minimal_name"]),
                    target_category=str(failed["target_category"]),
                    trace_rule_ids=list(failed["trace_rule_ids"]),
                    citations_required=failed["condition"] == "rule_rag",
                )
                validation_error = None
            except ValueError as error:
                citations = []
                validation_error = str(error)
            repair_raw.append(
                {
                    **prompt_manifest_fields(rendered),
                    "case_id": failed["case_id"],
                    "condition": failed["condition"],
                    "generator": model,
                    "attempt": int(prior["attempt"]) + 1,
                    "completion_repair_round": 3,
                    "raw_text": response.text,
                    "latency_seconds": response.latency_seconds,
                    "validation_error": validation_error,
                    "repair_scope": "remaining_length_failures_only",
                }
            )
            if validation_error is None:
                key = (failed["case_id"], failed["condition"], model)
                repaired[key] = {
                    **failed,
                    "status": "success",
                    "explanation": response.text,
                    "retry_count": int(prior["attempt"]) + 1,
                    "latency_seconds": response.latency_seconds,
                    "prompt_eval_count": response.prompt_eval_count,
                    "eval_count": response.eval_count,
                    "total_duration_ns": response.total_duration_ns,
                    "citation_occurrences": citations,
                    "accepted_by": "one_final_current_45_75_length_retry",
                }
        except Exception as error:
            repair_raw.append(
                {
                    **prompt_manifest_fields(rendered),
                    "case_id": failed["case_id"],
                    "condition": failed["condition"],
                    "generator": model,
                    "attempt": int(prior["attempt"]) + 1,
                    "completion_repair_round": 3,
                    "raw_text": None,
                    "latency_seconds": None,
                    "validation_error": f"{type(error).__name__}: {error}",
                    "repair_scope": "remaining_length_failures_only",
                }
            )
    if active_model is not None:
        client.unload(active_model)
    completed = [
        repaired.get((row["case_id"], row["condition"], row["generator"]), row) for row in outputs
    ]
    accepted_after = {
        (r["case_id"], r["condition"], r["generator"]): canonical_json(r)
        for r in completed
        if r["status"] == "success"
        and (r["case_id"], r["condition"], r["generator"]) in accepted_before
    }
    if accepted_after != accepted_before:
        raise ValueError("A pre-existing accepted output would be changed by this retry.")
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
    manifest["final_length_retry"] = {
        "targeted_terminal_length_failures": len(targets),
        "one_attempt_per_target": True,
        "accepted": len(repaired),
        "remaining_terminal_failures": len(remaining),
        "prompt_contract": "current_global_45_75",
        "frozen_v3_cases_and_traces_preserved": True,
        "accepted_outputs_unchanged": True,
    }
    write_json(args.run_dir / "manifest.json", manifest)
    write_json(args.manifest, manifest)
    print(json.dumps(manifest["final_length_retry"], indent=2))


if __name__ == "__main__":
    main()
