"""Perform the one authorized final targeted Stage 9 repair, then freeze the matrix."""

from __future__ import annotations

import argparse
import hashlib
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
from evidence_fashion.manifest import sha256_file, utc_timestamp, write_json
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
    temporary = path.with_suffix(path.suffix + ".finalizing")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bound_output(manifest: Mapping[str, Any], suffix: str) -> Path:
    matches = [Path(path) for path in manifest["output_artifact_hashes"] if path.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Selection manifest must bind exactly one {suffix} artifact.")
    path = matches[0]
    if not path.exists() or sha256_file(path) != manifest["output_artifact_hashes"][str(path)]:
        raise ValueError(f"Hash-mismatched frozen selection input: {path}")
    return path


def rule_evidence(trace: Mapping[str, Any]) -> str:
    return "\n".join(f"[{rule['rule_id']}] {rule['rule_text']}" for rule in trace["rules"])


def render_main_prompt(registry: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
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
        rows = json.loads(response.read().decode("utf-8")).get("models", [])
    return next(
        (str(row.get("digest")) for row in rows if str(row.get("name")) == model_id), None
    )


def last_attempt(
    raw_by_key: Mapping[tuple[str, str, str], list[dict[str, Any]]], row: Mapping[str, Any]
) -> dict[str, Any]:
    key = (str(row["case_id"]), str(row["condition"]), str(row["generator"]))
    attempts = raw_by_key.get(key)
    if not attempts:
        raise ValueError(f"No retained raw attempt for terminal cell: {key}")
    return attempts[-1]


def classify(reason: str) -> str:
    if reason.startswith("Explanation must contain"):
        return "length_rewrite"
    if reason == "Explanation does not preserve the exact locked recommendation.":
        return "locked_item_rewrite"
    if reason.startswith("HTTPError: HTTP Error 500"):
        return "normal_http_retry"
    raise ValueError(f"Unexpected terminal failure outside final repair scope: {reason}")


def repair_appendix(kind: str, *, locked_name: str, existing: str) -> str:
    if kind == "length_rewrite":
        directive = (
            "Rewrite the existing response to 45–75 words while preserving its meaning, the "
            "exact locked recommendation, condition, supplied evidence, and any citations."
        )
    elif kind == "locked_item_rewrite":
        directive = (
            "Rewrite the existing response using this exact locked recommendation verbatim: "
            f"{locked_name}. Preserve its original reasoning, condition, supplied evidence, "
            "and any citations, and remain within 45–75 words."
        )
    else:
        raise ValueError(f"No repair appendix is permitted for {kind}.")
    return f"\n\nFINAL TARGETED REPAIR: {directive}\n\nExisting response:\n{existing}"


def main() -> None:
    args = parse_args()
    outputs_path = args.run_dir / "explanations.jsonl"
    raw_path = args.run_dir / "raw_generation_attempts.jsonl"
    outputs = read_jsonl(outputs_path)
    raw = read_jsonl(raw_path)
    if len(outputs) != 3000:
        raise ValueError("Stage 9 matrix must have exactly 3,000 cells before final repair.")
    raw_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw:
        raw_by_key.setdefault((row["case_id"], row["condition"], row["generator"]), []).append(row)
    targets = [row for row in outputs if row["status"] == "terminal_failure"]
    target_kinds = Counter(
        classify(str(last_attempt(raw_by_key, row)["validation_error"])) for row in targets
    )
    if target_kinds != Counter(
        {"length_rewrite": 9, "locked_item_rewrite": 9, "normal_http_retry": 1}
    ):
        raise ValueError(
            f"Final repair must target exactly the authorized 19 cells: {target_kinds}"
        )
    accepted_before = {
        (row["case_id"], row["condition"], row["generator"]): canonical_json(row)
        for row in outputs
        if row["status"] == "success"
    }
    selection_manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    selection_rows = read_jsonl(bound_output(selection_manifest, "condition_inputs.jsonl"))
    selections = {(row["calibration_case_id"], row["condition"]): row for row in selection_rows}
    if len(selections) != 1000:
        raise ValueError("Frozen V3 selection must contain 1,000 condition inputs.")
    registry = load_prompt_registry(args.prompts)
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    model_settings = {str(row["model_id"]): row for row in models["generators"]["roster"]}
    for model in {str(row["generator"]) for row in targets}:
        observed = installed_model_digest(args.ollama_endpoint, model)
        expected = str(model_settings[model]["immutable_digest"])
        if observed != expected:
            raise ValueError(f"Model digest mismatch for {model}: {observed} != {expected}")
    client = OllamaClient(models["generation_defaults"], endpoint=args.ollama_endpoint)
    repaired: dict[tuple[str, str, str], dict[str, Any]] = {}
    final_raw: list[dict[str, Any]] = []
    active_model: str | None = None
    for failed in sorted(
        targets, key=lambda row: (row["generator"], row["condition"], row["case_id"])
    ):
        model = str(failed["generator"])
        if active_model is not None and active_model != model:
            client.unload(active_model)
        active_model = model
        selection = selections.get((failed["case_id"], failed["condition"]))
        if selection is None:
            raise ValueError("Terminal cell is outside the frozen V3 selection.")
        if (
            failed["locked_candidate_id"] != selection["locked_candidate_id"]
            or failed["A_sha256"] != canonical_hash(selection["A_common_context"])
            or failed["B_sha256"] != canonical_hash(selection["B_exact_stored_trace"])
        ):
            raise ValueError("Terminal cell no longer matches its frozen V3 packet.")
        prior = last_attempt(raw_by_key, failed)
        kind = classify(str(prior["validation_error"]))
        rendered = render_main_prompt(registry, selection)
        locked_name = str(selection["A_common_context"]["locked_item_minimal_name"])
        if kind == "normal_http_retry":
            prompt = str(rendered["user_prompt"])
        else:
            existing = prior.get("raw_text")
            if not isinstance(existing, str) or not existing.strip():
                raise ValueError(f"{kind} requires its retained failed response text.")
            prompt = str(rendered["user_prompt"]) + repair_appendix(
                kind, locked_name=locked_name, existing=existing
            )
        try:
            response = client.generate(
                model,
                prompt,
                system_prompt=str(rendered["system_prompt"]),
                token_limit=int(registry["roles"][str(rendered["role"])]["token_limit"]),
                timeout_seconds=float(models["generation_defaults"]["timeout_seconds"]),
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
                "generator": model,
                "attempt": int(prior["attempt"]) + 1,
                "completion_repair_round": 4,
                "repair_kind": kind,
                "repair_prompt": prompt,
                "raw_text": response.text,
                "latency_seconds": response.latency_seconds,
                "validation_error": validation_error,
            }
            final_raw.append(raw_record)
            if validation_error is None:
                repaired[(failed["case_id"], failed["condition"], model)] = {
                    **failed,
                    "status": "success",
                    "explanation": response.text,
                    "retry_count": int(raw_record["attempt"]),
                    "latency_seconds": response.latency_seconds,
                    "prompt_eval_count": response.prompt_eval_count,
                    "eval_count": response.eval_count,
                    "total_duration_ns": response.total_duration_ns,
                    "citation_occurrences": citations,
                    "accepted_by": f"final_targeted_{kind}",
                }
        except Exception as error:
            final_raw.append(
                {
                    **prompt_manifest_fields(rendered),
                    "case_id": failed["case_id"],
                    "condition": failed["condition"],
                    "generator": model,
                    "attempt": int(prior["attempt"]) + 1,
                    "completion_repair_round": 4,
                    "repair_kind": kind,
                    "repair_prompt": prompt,
                    "raw_text": None,
                    "latency_seconds": None,
                    "validation_error": f"{type(error).__name__}: {error}",
                }
            )
    if active_model is not None:
        client.unload(active_model)
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
        raise ValueError("A previously accepted Stage 9 output would be changed by final repair.")
    write_jsonl_atomic(outputs_path, completed)
    write_jsonl_atomic(raw_path, [*raw, *final_raw])
    remaining = [row for row in completed if row["status"] == "terminal_failure"]
    final_raw_by_key = {
        (row["case_id"], row["condition"], row["generator"]): row for row in final_raw
    }
    unresolved = {
        f"{row['case_id']} | {row['generator']} | {row['condition']}": final_raw_by_key[
            (row["case_id"], row["condition"], row["generator"])
        ]["validation_error"]
        for row in remaining
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["output_artifact_hashes"] = {
        str(outputs_path): sha256_file(outputs_path),
        str(raw_path): sha256_file(raw_path),
    }
    manifest["row_counts"]["raw_attempts"] = len(raw) + len(final_raw)
    manifest["failure_counts"]["terminal_failures"] = len(remaining)
    manifest["stage9_freeze"] = {
        "status": "frozen_after_final_targeted_repair",
        "timestamp_utc": utc_timestamp(),
        "final_targeted_calls": dict(target_kinds),
        "final_targeted_repairs_accepted": len(repaired),
        "accepted_matrix_cells": len(completed) - len(remaining),
        "unresolved_cells": unresolved,
        "accepted_outputs_unchanged": True,
        "frozen_regardless_of_remaining_failures": True,
    }
    manifest["status"] = (
        "frozen_complete_matrix_with_terminal_failures"
        if remaining
        else "frozen_complete_accepted_matrix"
    )
    write_json(args.run_dir / "manifest.json", manifest)
    write_json(args.manifest, manifest)
    print(json.dumps(manifest["stage9_freeze"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
