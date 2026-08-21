"""Complete the authorized Stage 9 45–75-word acceptance transition."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".transitioning")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def canonical_json(row: Mapping[str, Any]) -> str:
    return json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def locked_name_from_prompt(prompt: str) -> str:
    prefix = "Required first-sentence wording: include the exact recommended item text, "
    if prefix not in prompt:
        raise ValueError("Stored Stage 9 prompt has no locked-item contract line.")
    return prompt.split(prefix, 1)[1].split(".\n", 1)[0].strip()


def installed_model_digest(endpoint: str, model_id: str) -> str | None:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/api/tags", timeout=10) as response:
        models = json.loads(response.read().decode("utf-8")).get("models", [])
    return next(
        (
            str(row.get("digest"))
            for row in models
            if str(row.get("name")) == model_id
        ),
        None,
    )


def final_attempt(
    raw_by_key: Mapping[tuple[str, str, str], list[dict[str, Any]]],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    key = (str(row["case_id"]), str(row["condition"]), str(row["generator"]))
    attempts = raw_by_key.get(key)
    if not attempts:
        raise ValueError(f"Missing retained raw attempts for terminal cell: {key}")
    return attempts[-1]


def revalidated_row(
    failed_row: Mapping[str, Any], candidate: Mapping[str, Any], citations: list[str]
) -> dict[str, Any]:
    text = str(candidate["raw_text"])
    return {
        **failed_row,
        "status": "success",
        "explanation": text,
        "retry_count": int(candidate["attempt"]),
        "latency_seconds": candidate["latency_seconds"],
        "prompt_eval_count": None,
        "eval_count": None,
        "total_duration_ns": None,
        "citation_occurrences": citations,
        "accepted_by": "global_45_75_revalidation_no_model_call",
    }


def main() -> None:
    args = parse_args()
    outputs_path = args.run_dir / "explanations.jsonl"
    raw_path = args.run_dir / "raw_generation_attempts.jsonl"
    outputs = read_jsonl(outputs_path)
    raw = read_jsonl(raw_path)
    if len(outputs) != 3000:
        raise ValueError("Stage 9 output matrix must contain 3,000 cells.")
    raw_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw:
        raw_by_key.setdefault((row["case_id"], row["condition"], row["generator"]), []).append(row)
    accepted_before = {
        (row["case_id"], row["condition"], row["generator"]): canonical_json(row)
        for row in outputs
        if row["status"] == "success"
    }
    revalidated: dict[tuple[str, str, str], dict[str, Any]] = {}
    locked_failures: list[tuple[dict[str, Any], dict[str, Any]]] = []
    other_failures: list[tuple[dict[str, Any], str]] = []
    for failed in (row for row in outputs if row["status"] == "terminal_failure"):
        candidate = final_attempt(raw_by_key, failed)
        text = candidate.get("raw_text")
        if not isinstance(text, str):
            other_failures.append((failed, "missing_raw_text"))
            continue
        try:
            citations = validate_generated_explanation(
                text,
                locked_item_name=locked_name_from_prompt(str(candidate["user_prompt"])),
                target_category=str(failed["target_category"]),
                trace_rule_ids=list(failed["trace_rule_ids"]),
                citations_required=failed["condition"] == "rule_rag",
            )
            key = (failed["case_id"], failed["condition"], failed["generator"])
            revalidated[key] = revalidated_row(failed, candidate, citations)
        except ValueError as error:
            reason = str(error)
            if reason == "Explanation does not preserve the exact locked recommendation.":
                locked_failures.append((failed, candidate))
            else:
                other_failures.append((failed, reason))
    if len(locked_failures) != 22:
        raise ValueError(
            f"Expected exactly 22 locked-item failures after 45–75 revalidation; found "
            f"{len(locked_failures)}."
        )

    registry = load_prompt_registry(args.prompts)
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    model_settings = {str(row["model_id"]): row for row in models["generators"]["roster"]}
    for model in {str(row["generator"]) for row, _ in locked_failures}:
        expected = str(model_settings[model]["immutable_digest"])
        observed = installed_model_digest(args.ollama_endpoint, model)
        if observed != expected:
            raise ValueError(f"Model digest mismatch for {model}: {observed} != {expected}")
    client = OllamaClient(models["generation_defaults"], endpoint=args.ollama_endpoint)
    repair_raw: list[dict[str, Any]] = []
    repaired: dict[tuple[str, str, str], dict[str, Any]] = {}
    active_model: str | None = None
    for failed, exemplar in sorted(
        locked_failures,
        key=lambda pair: (pair[0]["generator"], pair[0]["condition"], pair[0]["case_id"]),
    ):
        model = str(failed["generator"])
        if active_model is not None and active_model != model:
            client.unload(active_model)
        active_model = model
        locked_name = locked_name_from_prompt(str(exemplar["user_prompt"]))
        appendix = (
            "\n\nCOMPLETION REPAIR: Keep every original evidence and citation requirement. "
            "Use the 45–75 word acceptance range. Name this exact locked recommendation "
            f"verbatim, with no substitute: {locked_name}."
        )
        try:
            response = client.generate(
                model,
                str(exemplar["user_prompt"]) + appendix,
                system_prompt=str(exemplar["system_prompt"]),
                token_limit=int(registry["roles"][str(exemplar["role"])]["token_limit"]),
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
                **exemplar,
                "attempt": int(exemplar["attempt"]) + 1,
                "completion_repair_round": 2,
                "raw_text": response.text,
                "latency_seconds": response.latency_seconds,
                "validation_error": validation_error,
                "repair_appendix": appendix,
            }
            repair_raw.append(raw_record)
            if validation_error is None:
                key = (failed["case_id"], failed["condition"], failed["generator"])
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
                    "accepted_by": "one_bounded_locked_item_completion_repair",
                }
        except Exception as error:
            repair_raw.append(
                {
                    **exemplar,
                    "attempt": int(exemplar["attempt"]) + 1,
                    "completion_repair_round": 2,
                    "raw_text": None,
                    "latency_seconds": None,
                    "validation_error": f"{type(error).__name__}: {error}",
                    "repair_appendix": appendix,
                }
            )
    if active_model is not None:
        client.unload(active_model)

    completed = [
        revalidated.get(
            (row["case_id"], row["condition"], row["generator"]),
            repaired.get((row["case_id"], row["condition"], row["generator"]), row),
        )
        for row in outputs
    ]
    accepted_after = {
        (row["case_id"], row["condition"], row["generator"]): canonical_json(row)
        for row in completed
        if row["status"] == "success"
        and (row["case_id"], row["condition"], row["generator"]) in accepted_before
    }
    if accepted_after != accepted_before:
        raise ValueError("A pre-existing accepted output would be changed by completion repair.")
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
    manifest["completion_length_contract"] = {
        "global_range": "45-75",
        "target_words": 65,
        "revalidated_length_only_without_model_calls": len(revalidated),
        "one_bounded_locked_item_repair_calls": len(locked_failures),
        "locked_item_repairs_accepted": len(repaired),
        "not_retried_non_locked_residual_failures": len(other_failures),
        "accepted_outputs_unchanged": True,
        "active_prompt_registry_sha256": sha256_file(args.prompts),
    }
    write_json(args.run_dir / "manifest.json", manifest)
    write_json(args.manifest, manifest)
    print(
        json.dumps(
            {
                "revalidated_without_model_calls": len(revalidated),
                "locked_repair_attempts": len(locked_failures),
                "locked_repairs_accepted": len(repaired),
                "remaining_terminal_failures": len(remaining),
                "non_locked_residual_reasons": [reason for _, reason in other_failures],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
