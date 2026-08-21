"""Generate the frozen Stage 9 3,000-explanation matrix from Stage 8 inputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from evidence_fashion.explanation import OllamaClient
from evidence_fashion.grounding_contracts import validate_generated_explanation
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument("--runtime-root", type=Path, default=Path(".runtime/current/explanations"))
    return parser.parse_args()


def _bound_output(manifest: Mapping[str, Any], suffix: str) -> tuple[Path, str]:
    matches = [
        (Path(path), str(digest))
        for path, digest in manifest["output_artifact_hashes"].items()
        if path.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Stage 8 manifest must bind exactly one {suffix} artifact.")
    path, digest = matches[0]
    if not path.exists() or sha256_file(path) != digest:
        raise ValueError(f"Stage 8 input is hash-mismatched: {path}")
    return path, digest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rule_evidence(trace: Mapping[str, Any]) -> str:
    rules = trace.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("Rule-RAG Stage 9 input must have a non-empty exact trace.")
    return "\n".join(f"[{rule['rule_id']}] {rule['rule_text']}" for rule in rules)


def _render(
    registry: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[str, ...], bool]:
    context = row["A_common_context"]
    if row["condition"] == "no_rag":
        rendered = render_prompt(registry, "no_rag_explanation", context)
        return rendered, (), False
    variables = {**context, "rule_evidence": _rule_evidence(row["B_exact_stored_trace"])}
    rendered = render_prompt(registry, "rule_rag_explanation", variables)
    return rendered, tuple(rule["rule_id"] for rule in row["B_exact_stored_trace"]["rules"]), True


def _generate(
    client: OllamaClient,
    *,
    model: str,
    rendered: Mapping[str, Any],
    locked_name: str,
    target_category: str,
    trace_ids: Sequence[str],
    citations_required: bool,
    retry: Mapping[str, Any],
    token_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(int(retry["max_attempts"]) + 1):
        prompt = str(rendered["user_prompt"])
        if attempt:
            prompt += "\n\n" + str(retry["retry_instruction"])
        try:
            response = client.generate(
                model,
                prompt,
                system_prompt=str(rendered["system_prompt"]),
                token_limit=token_limit,
                timeout_seconds=float(client.defaults["timeout_seconds"]) * (2**attempt),
            )
            try:
                occurrences = validate_generated_explanation(
                    response.text,
                    locked_item_name=locked_name,
                    target_category=target_category,
                    trace_rule_ids=trace_ids,
                    citations_required=citations_required,
                )
            except ValueError as error:
                attempts.append(
                    {
                        "attempt": attempt,
                        "raw_text": response.text,
                        "validation_error": str(error),
                        "latency_seconds": response.latency_seconds,
                    }
                )
                continue
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_text": response.text,
                    "validation_error": None,
                    "latency_seconds": response.latency_seconds,
                }
            )
            return {
                "status": "success",
                "explanation": response.text,
                "retry_count": attempt,
                "latency_seconds": response.latency_seconds,
                "prompt_eval_count": response.prompt_eval_count,
                "eval_count": response.eval_count,
                "total_duration_ns": response.total_duration_ns,
                "citation_occurrences": occurrences,
            }, attempts
        except Exception as error:
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_text": None,
                    "validation_error": f"{type(error).__name__}: {error}",
                    "latency_seconds": None,
                }
            )
    return {
        "status": "terminal_failure",
        "explanation": None,
        "retry_count": int(retry["max_attempts"]),
        "latency_seconds": None,
        "prompt_eval_count": None,
        "eval_count": None,
        "total_duration_ns": None,
        "citation_occurrences": [],
    }, attempts


def _append(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    registry = load_prompt_registry(args.prompts)
    resolved = load_resolved_configuration(args.config, args.models_config)
    digest = configuration_hash({"resolved": resolved, "prompts_sha256": sha256_file(args.prompts)})
    selection_manifest_path = Path(
        "artifacts/manifests/stage8_explanation_case_selection_manifest.json"
    )
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    inputs_path, inputs_hash = _bound_output(selection_manifest, "condition_inputs.jsonl")
    rows = _read_jsonl(inputs_path)
    expected_inputs = int(config["explanations"]["case_count"]) * len(
        config["explanations"]["conditions"]
    )
    if (
        len(rows) != expected_inputs
        or len({(r["calibration_case_id"], r["condition"]) for r in rows}) != expected_inputs
    ):
        raise ValueError("Frozen Stage 8 condition matrix is incomplete or duplicated.")
    run_id = f"stage9-generation-{digest[:12]}"
    run_dir = args.runtime_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Immutable Stage 9 generation already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    output_path, raw_path = (
        run_dir / "explanations.jsonl",
        run_dir / "raw_generation_attempts.jsonl",
    )
    client = OllamaClient(models["generation_defaults"])
    total = 0
    failures = 0
    with (
        output_path.open("x", encoding="utf-8", newline="\n") as outputs,
        raw_path.open("x", encoding="utf-8", newline="\n") as raw_outputs,
    ):
        for generator in models["generators"]["roster"]:
            model = str(generator["model_id"])
            for row in rows:
                rendered, trace_ids, citations_required = _render(registry, row)
                role = rendered["role"]
                retry = registry["roles"][role]["retry"]
                result, attempts = _generate(
                    client,
                    model=model,
                    rendered=rendered,
                    locked_name=str(row["A_common_context"]["locked_item_minimal_name"]),
                    target_category=str(row["target_category"]),
                    trace_ids=trace_ids,
                    citations_required=citations_required,
                    retry=retry,
                    token_limit=int(registry["roles"][role]["token_limit"]),
                )
                key = {
                    "case_id": row["calibration_case_id"],
                    "condition": row["condition"],
                    "generator": model,
                }
                for attempt in attempts:
                    _append(raw_outputs, {**key, **attempt, **prompt_manifest_fields(rendered)})
                _append(
                    outputs,
                    {
                        **key,
                        "target_category": row["target_category"],
                        "locked_candidate_id": row["locked_candidate_id"],
                        "A_sha256": row["A_sha256"],
                        "B_sha256": row["B_sha256"],
                        "trace_rule_ids": list(trace_ids),
                        **prompt_manifest_fields(rendered),
                        **result,
                    },
                )
                total += 1
                failures += int(result["status"] != "success")
            client.unload(model)
    expected = len(rows) * len(models["generators"]["roster"])
    outputs = _read_jsonl(output_path)
    keys = {(r["case_id"], r["condition"], r["generator"]) for r in outputs}
    if len(outputs) != expected or len(keys) != expected:
        raise ValueError("Stage 9 generation matrix is incomplete or contains duplicate keys.")
    manifest = {
        "schema_version": 1,
        "stage": 9,
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": digest,
        "resolved_configuration": resolved,
        "prompt_registry_sha256": sha256_file(args.prompts),
        "input_artifact_hashes": {str(inputs_path): inputs_hash},
        "output_artifact_hashes": {
            str(output_path): sha256_file(output_path),
            str(raw_path): sha256_file(raw_path),
        },
        "models": models["generators"],
        "row_counts": {
            "condition_inputs": len(rows),
            "generations": total,
            "raw_attempts": len(_read_jsonl(raw_path)),
        },
        "failure_counts": {"terminal_failures": failures},
        "validation": {
            "complete_matrix": True,
            "duplicate_keys": 0,
            "locked_contract_validated_before_persist": True,
        },
        "environment": environment_summary(),
        "command": (
            "python scripts/run_stage9_explanation_generation.py --config configs/experiment.yaml "
            "--models-config configs/models.yaml --prompts configs/prompts.yaml "
            "--runtime-root .runtime/current/explanations"
        ),
    }
    write_new_json(run_dir / "manifest.json", manifest)
    write_json(Path("artifacts/manifests/stage9_explanation_generation_manifest.json"), manifest)
    print(
        json.dumps(
            {"run_id": run_id, "generations": total, "terminal_failures": failures}, indent=2
        )
    )


if __name__ == "__main__":
    main()
