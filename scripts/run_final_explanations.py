"""Generate the frozen Stage-2 explanation matrix from locked recommendation inputs only."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from evidence_fashion.explanation import OllamaClient, text_sha256
from evidence_fashion.final_contracts import canonical_json_sha256
from evidence_fashion.grounding_contracts import validate_generated_explanation
from evidence_fashion.manifest import (
    configuration_hash,
    environment_summary,
    git_commit,
    load_resolved_configuration,
    sha256_file,
    utc_timestamp,
    write_new_json,
)


def _manifest_output(manifest: dict[str, Any], suffix: str) -> Path:
    return Path(next(path for path in manifest["output_artifact_hashes"] if path.endswith(suffix)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def _prompt_for_case(case: dict[str, Any], role: dict[str, Any]) -> str:
    context = dict(case["common_context_A"])
    values = dict(context)
    if "rule_evidence" in role["template_variables"]:
        values["rule_evidence"] = json.dumps(
            case["exact_stored_rule_trace_B"], ensure_ascii=False, sort_keys=True
        )
    return str(role["user_template"]).format(**values)


def _generate_cell(
    *,
    client: OllamaClient,
    model_id: str,
    prompt: str,
    role: dict[str, Any],
    case: dict[str, Any],
    condition: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_ids = [rule["rule_id"] for rule in case["exact_stored_rule_trace_B"]["rules"]]
    needs_citations = bool(role.get("citation_required", False))
    attempts: list[dict[str, Any]] = []
    retry_limit = int(config["explanations"]["bounded_retry_attempts"])
    repair = ""
    for attempt_number in range(retry_limit + 1):
        active_prompt = prompt + repair
        try:
            result = client.generate(
                model_id,
                active_prompt,
                system_prompt=str(role["system_prompt"]),
                token_limit=int(config["inference"]["generation_token_limit"]),
                timeout_seconds=float(config["inference"]["timeout_seconds"]),
            )
            text = result.text
            citations = validate_generated_explanation(
                text,
                locked_item_name=case["common_context_A"]["locked_item_minimal_name"],
                target_category=case["target_category"],
                trace_rule_ids=trace_ids if needs_citations else (),
                citations_required=needs_citations,
                minimum_words=int(config["explanations"]["word_minimum"]),
                maximum_words=int(config["explanations"]["word_maximum"]),
            )
        except Exception as error:  # Retain terminal failure details; do not abandon the batch.
            attempts.append(
                {
                    "attempt_number": attempt_number + 1,
                    "prompt_hash": text_sha256(active_prompt),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            repair = (
                "\n\nRewrite only the explanation. Preserve the exact recommended item, stay "
                "within 45-75 words, and obey the supplied citation format and evidence limits."
            )
            continue
        attempts.append(
            {
                "attempt_number": attempt_number + 1,
                "prompt_hash": text_sha256(active_prompt),
                "response_hash": text_sha256(text),
                "word_count": len(text.split()),
                "citation_ids": citations,
                "latency_seconds": result.latency_seconds,
                "prompt_eval_count": result.prompt_eval_count,
                "eval_count": result.eval_count,
            }
        )
        return (
            {
                "status": "accepted",
                "explanation": text,
                "word_count": len(text.split()),
                "citation_ids": citations,
                "attempts_used": attempt_number + 1,
            },
            attempts,
        )
    return (
        {
            "status": "terminal_failure",
            "explanation": None,
            "word_count": None,
            "citation_ids": [],
            "attempts_used": retry_limit + 1,
        },
        attempts,
    )


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
    recommendation_manifests = sorted(
        Path(experiment["paths"]["recommendation_runs"]).glob(
            "final-recommendations-*/manifest.json"
        )
    )
    if len(recommendation_manifests) != 1:
        raise ValueError("Exactly one frozen Stage-2 recommendation manifest is required.")
    recommendation_manifest_path = recommendation_manifests[0]
    recommendation_manifest = json.loads(recommendation_manifest_path.read_text(encoding="utf-8"))
    if recommendation_manifest["status"] != "frozen_recommendations_complete":
        raise ValueError("The Stage-2 Part-A recommendation output is not frozen.")
    input_path = _manifest_output(recommendation_manifest, "explanation_cases.jsonl")
    cases = _read_jsonl(input_path)
    expected_cases = int(experiment["explanations"]["case_count"])
    if len(cases) != expected_cases:
        raise ValueError("Frozen explanation input count does not match the final configuration.")
    conditions = list(experiment["explanations"]["conditions"])
    roster = list(models["generators"]["roster"])
    expected_cells = expected_cases * len(conditions) * len(roster)
    run_dir = (
        Path(experiment["paths"]["explanation_runs"]) / f"final-explanations-{config_hash[:12]}"
    )
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite a frozen explanation run: {run_dir}")
    run_dir.mkdir(parents=True)
    output_path = run_dir / "explanations.jsonl"
    attempts_path = run_dir / "raw_generation_attempts.jsonl"
    inference = {**models["inference_defaults"]}
    inference["token_limit"] = inference["generation_token_limit"]
    experiment["inference"] = inference
    client = OllamaClient(inference, endpoint=str(inference["endpoint"]))
    accepted = Counter()
    failures = Counter()
    word_counts: dict[str, list[int]] = defaultdict(list)
    attempted_cells = 0
    with (
        output_path.open("x", encoding="utf-8", newline="\n") as output_handle,
        attempts_path.open("x", encoding="utf-8", newline="\n") as attempts_handle,
    ):
        for model in roster:
            model_id = str(model["model_id"])
            try:
                for case in cases:
                    for condition in conditions:
                        role = prompts["roles"][f"{condition}_explanation"]
                        prompt = _prompt_for_case(case, role)
                        result, attempts = _generate_cell(
                            client=client,
                            model_id=model_id,
                            prompt=prompt,
                            role=role,
                            case=case,
                            condition=condition,
                            config=experiment,
                        )
                        cell = {
                            "case_id": case["case_id"],
                            "target_category": case["target_category"],
                            "generator_model_id": model_id,
                            "generator_model_digest": model["immutable_digest"],
                            "condition": condition,
                            "common_context_A": case["common_context_A"],
                            "common_context_A_hash": case["common_context_A_hash"],
                            "locked_candidate_id": case["locked_candidate_id"],
                            "exact_stored_rule_trace_B": (
                                case["exact_stored_rule_trace_B"]
                                if condition == "rule_rag"
                                else None
                            ),
                            "trace_hash": case["trace_hash"],
                            "prompt_hash": text_sha256(prompt),
                            "system_prompt_hash": text_sha256(str(role["system_prompt"])),
                            **result,
                        }
                        if (
                            condition == "rule_rag"
                            and canonical_json_sha256(cell["exact_stored_rule_trace_B"])
                            != cell["trace_hash"]
                        ):
                            raise ValueError("Rule-RAG explanation lost its locked trace identity.")
                        _append_jsonl(output_handle, cell)
                        for attempt in attempts:
                            _append_jsonl(
                                attempts_handle,
                                {
                                    "case_id": case["case_id"],
                                    "generator_model_id": model_id,
                                    "condition": condition,
                                    **attempt,
                                },
                            )
                        attempted_cells += 1
                        key = f"{model_id}:{condition}"
                        if result["status"] == "accepted":
                            accepted[key] += 1
                            word_counts[key].append(int(result["word_count"]))
                        else:
                            failures[key] += 1
            finally:
                client.unload(model_id)
    if attempted_cells != expected_cells:
        raise RuntimeError("Every configured Stage-2 generation cell must be attempted.")
    word_summary = {
        key: {
            "count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "mean": sum(values) / len(values),
        }
        for key, values in sorted(word_counts.items())
    }
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": 2,
        "stage": 2,
        "stage_name": "fresh_explanations",
        "status": "complete",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": config_hash,
        "input_artifact_hashes": {
            str(recommendation_manifest_path): sha256_file(recommendation_manifest_path),
            str(input_path): sha256_file(input_path),
        },
        "output_artifact_hashes": {
            str(output_path): sha256_file(output_path),
            str(attempts_path): sha256_file(attempts_path),
        },
        "models": roster,
        "prompt_hashes": {
            name: {
                "system": text_sha256(str(role["system_prompt"])),
                "template": text_sha256(str(role["user_template"])),
            }
            for name, role in prompts["roles"].items()
            if name in {"no_rag_explanation", "rule_rag_explanation"}
        },
        "row_counts": {
            "frozen_explanation_cases": len(cases),
            "attempted_cells": attempted_cells,
            "accepted_cells": sum(accepted.values()),
            "terminal_failure_cells": sum(failures.values()),
            "raw_generation_attempts": sum(
                1 for _ in attempts_path.read_text(encoding="utf-8").splitlines()
            ),
        },
        "accepted_by_model_condition": dict(sorted(accepted.items())),
        "failure_counts_by_model_condition": dict(sorted(failures.items())),
        "word_count_distributions": word_summary,
        "fixed_contract": {
            "word_minimum": experiment["explanations"]["word_minimum"],
            "word_maximum": experiment["explanations"]["word_maximum"],
            "retry_attempts": experiment["explanations"]["bounded_retry_attempts"],
            "sequential_model_order": [model["model_id"] for model in roster],
            "no_retrieval_after_recommendation_lock": True,
            "rule_rag_trace_hash_preserved": True,
        },
        "environment": environment_summary(),
    }
    write_new_json(manifest_path, manifest)
    print(json.dumps(manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
