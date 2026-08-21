"""Boundedly repair only failed Stage 9 generation cells without changing valid cells."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from evidence_fashion.explanation import OllamaClient
from evidence_fashion.grounding_contracts import validate_generated_explanation
from evidence_fashion.manifest import sha256_file, write_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".repairing")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _repair_instruction(*, locked_name: str, condition: str, trace_ids: list[str]) -> str:
    citation_clause = (
        " End with at least one separate canonical citation selected only from: "
        + " ".join(f"[{rule_id}]" for rule_id in trace_ids)
        + "."
        if condition == "rule_rag"
        else " Do not add citations."
    )
    return (
        "\n\nCONTRACT REPAIR: Your prior response omitted or changed the locked recommendation. "
        "Return one concise sentence that begins with this exact item text, copied "
        "character-for-character with no markdown or substitute name: "
        f"{locked_name!r}."
        + citation_clause
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(".runtime/current/explanations/stage9-generation-51ea5ff43ce5"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/stage9_explanation_generation_manifest.json"),
    )
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_path = args.run_dir / "explanations.jsonl"
    raw_path = args.run_dir / "raw_generation_attempts.jsonl"
    outputs = _read_jsonl(outputs_path)
    raw = _read_jsonl(raw_path)
    failed = [row for row in outputs if row["status"] == "terminal_failure"]
    if not failed:
        print(json.dumps({"repaired": 0, "remaining_terminal_failures": 0}, indent=2))
        return

    raw_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw:
        key = (row["case_id"], row["condition"], row["generator"])
        raw_by_key.setdefault(key, []).append(row)
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    client = OllamaClient(models["generation_defaults"])
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
        prior_attempts = raw_by_key[key]
        exemplar = prior_attempts[-1]
        locked_prefix = "Required first-sentence wording: include the exact recommended item text, "
        locked_name = str(exemplar["user_prompt"]).split(locked_prefix, 1)[1]
        prompt = str(exemplar["user_prompt"]) + _repair_instruction(
            locked_name=locked_name,
            condition=str(failed_row["condition"]),
            trace_ids=list(failed_row["trace_rule_ids"]),
        )
        for retry_index in range(args.max_attempts):
            try:
                response = client.generate(
                    model,
                    prompt,
                    system_prompt=str(exemplar["system_prompt"]),
                    token_limit=int(models["generation_defaults"]["token_limit"]),
                    timeout_seconds=float(models["generation_defaults"]["timeout_seconds"])
                    * (2**retry_index),
                )
                try:
                    citations = validate_generated_explanation(
                        response.text,
                        locked_item_name=locked_name,
                        target_category=str(failed_row["target_category"]),
                        trace_rule_ids=list(failed_row["trace_rule_ids"]),
                        citations_required=failed_row["condition"] == "rule_rag",
                    )
                    validation_error = None
                except ValueError as error:
                    citations = []
                    validation_error = str(error)
                repair_raw.append(
                    {
                        **exemplar,
                        "attempt": len(prior_attempts) + retry_index,
                        "repair_round": 1,
                        "raw_text": response.text,
                        "latency_seconds": response.latency_seconds,
                        "validation_error": validation_error,
                    }
                )
                if validation_error is None:
                    repaired[key] = {
                        **failed_row,
                        "status": "success",
                        "explanation": response.text,
                        "retry_count": len(prior_attempts) + retry_index,
                        "latency_seconds": response.latency_seconds,
                        "prompt_eval_count": response.prompt_eval_count,
                        "eval_count": response.eval_count,
                        "total_duration_ns": response.total_duration_ns,
                        "citation_occurrences": citations,
                        "repair_round": 1,
                    }
                    break
            except Exception as error:  # retained as raw audit evidence
                repair_raw.append(
                    {
                        **exemplar,
                        "attempt": len(prior_attempts) + retry_index,
                        "repair_round": 1,
                        "raw_text": None,
                        "latency_seconds": None,
                        "validation_error": f"{type(error).__name__}: {error}",
                    }
                )
    if active_model is not None:
        client.unload(active_model)

    replaced = [
        repaired.get((row["case_id"], row["condition"], row["generator"]), row) for row in outputs
    ]
    _write_jsonl_atomic(outputs_path, replaced)
    _write_jsonl_atomic(raw_path, [*raw, *repair_raw])
    remaining = sum(row["status"] == "terminal_failure" for row in replaced)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["output_artifact_hashes"] = {
        str(outputs_path): sha256_file(outputs_path),
        str(raw_path): sha256_file(raw_path),
    }
    manifest["row_counts"]["raw_attempts"] = len(raw) + len(repair_raw)
    manifest["failure_counts"]["terminal_failures"] = remaining
    manifest["recovery"] = {
        "initial_terminal_failures": len(failed),
        "repair_round": 1,
        "repaired": len(repaired),
        "remaining_terminal_failures": remaining,
        "mechanism": "bounded exact-locked-item contract retry; valid generation cells unchanged",
    }
    write_json(args.run_dir / "manifest.json", manifest)
    write_json(args.manifest, manifest)
    print(json.dumps(manifest["recovery"], indent=2))


if __name__ == "__main__":
    main()
