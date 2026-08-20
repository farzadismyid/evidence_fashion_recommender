"""Run immutable Stage 7 explanation generation without Stage 8 assessment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from evidence_fashion.explanation import (
    OllamaClient,
    build_no_rag_prompt,
    build_rule_rag_prompt,
    common_context,
    text_sha256,
    word_count,
)
from evidence_fashion.grounding_contracts import (
    require_trace_applicability,
    validate_generated_explanation,
)
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

FROZEN_STAGE2_TO_6_SECTIONS = (
    "dataset",
    "preprocessing",
    "splits",
    "recommendation_evaluation",
    "candidate_pool",
    "retrieval",
    "embedding_validation",
    "rule_retrieval",
    "reranking",
    "reranking_search",
    "stage4_validation",
    "stage6",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_new(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n")


def _locked_input(manifest: Mapping[str, Any]) -> tuple[Path, str]:
    matches = [
        (Path(path), digest)
        for path, digest in manifest["output_artifact_hashes"].items()
        if path.endswith("locked_cases.jsonl")
    ]
    if len(matches) != 1:
        raise ValueError("Stage 6 manifest must bind exactly one locked_cases.jsonl output.")
    path, expected = matches[0]
    if not path.exists() or sha256_file(path) != expected:
        raise ValueError("The frozen Stage 6 locked-case artifact is missing or hash-mismatched.")
    return path, str(expected)


def _manifest_output(manifest: Mapping[str, Any], suffix: str) -> tuple[Path, str]:
    matches = [
        (Path(path), str(digest))
        for path, digest in manifest["output_artifact_hashes"].items()
        if path.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Manifest must bind exactly one {suffix} output.")
    path, expected = matches[0]
    if not path.exists() or sha256_file(path) != expected:
        raise ValueError(f"Reused {suffix} artifact is missing or hash-mismatched.")
    return path, expected


def validate_frozen_stage2_to_6(
    config: Mapping[str, Any], stage6_manifest: Mapping[str, Any]
) -> None:
    frozen = stage6_manifest["resolved_configuration"]["experiment"]
    changed = [
        section
        for section in FROZEN_STAGE2_TO_6_SECTIONS
        if config.get(section) != frozen.get(section)
    ]
    if changed:
        raise ValueError(f"Frozen Stage 2-6 configuration changed: {changed}")


def select_stage7_cases(
    locked: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    settings = config["stage7"]
    if settings["case_selection_method"] != "sha256_order_stratified_by_target_category":
        raise ValueError("Unsupported Stage 7 case-selection method.")
    categories = list(config["preprocessing"]["target_categories"])
    quota = int(settings["cases_per_category"])
    seed = int(settings["case_selection_seed"])
    selected: list[dict[str, Any]] = []
    for category in categories:
        available = [dict(row) for row in locked if row["target_category"] == category]
        available.sort(
            key=lambda row: (
                hashlib.sha256(f"{seed}:{row['case_id']}".encode()).hexdigest(),
                row["case_id"],
            )
        )
        if len(available) < quota:
            raise ValueError(f"Only {len(available)} locked {category} cases; {quota} required.")
        selected.extend(available[:quota])
    if len(selected) != int(config["explanations"]["case_count"]):
        raise ValueError("Stage 7 case selection does not match explanations.case_count.")
    return selected


def _trace_score(trace: Mapping[str, Any]) -> float:
    values = np.asarray(
        [rule["weighted_contribution"] for rule in trace["rules"]], dtype=np.float64
    )
    return float(0.7 * values.max() + 0.3 * values.mean())


def build_case_packets(
    selected: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]
) -> list[dict[str, Any]]:
    packets = []
    for case in selected:
        trace = case["evidence_trace"]
        require_trace_applicability(trace)
        if len(trace["rules"]) != int(settings["rule_count"]):
            raise ValueError("Every Stage 7 B packet must contain exactly five scoring rules.")
        if trace["candidate_id"] != case["locked_candidate_id"]:
            raise ValueError("A locked candidate does not match its exact scoring trace.")
        if not math.isclose(_trace_score(trace), trace["evidence_score"], abs_tol=1e-12):
            raise ValueError("An exact B trace does not reproduce its evidence score.")
        packet_a = common_context(case)
        packet_b = trace
        packets.append(
            {
                "case_id": case["case_id"],
                "query_item_id": case["query_item_id"],
                "query_outfit_id": case["query_outfit_id"],
                "target_category": case["target_category"],
                "locked_candidate_id": case["locked_candidate_id"],
                "A_common_context": packet_a,
                "A_sha256": canonical_hash(packet_a),
                "B_exact_stored_trace": packet_b,
                "B_sha256": canonical_hash(packet_b),
                "stage6_locked_record_sha256": canonical_hash(case),
            }
        )
    return packets


def ollama_models(endpoint: str) -> dict[str, str]:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/api/tags", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {str(row["name"]): str(row["digest"]) for row in payload["models"]}


def validate_generator_digests(models: Mapping[str, Any], endpoint: str) -> None:
    installed = ollama_models(endpoint)
    mismatches = {
        row["model_id"]: (row["immutable_digest"], installed.get(row["model_id"]))
        for row in models["generators"]["roster"]
        if installed.get(row["model_id"]) != row["immutable_digest"]
    }
    if mismatches:
        raise ValueError(f"Configured generator digests do not match Ollama: {mismatches}")


def refusal_markers(text: str, config: Mapping[str, Any]) -> list[str]:
    lowered = text.lower()
    return [
        marker for marker in config["stage7"]["refusal_detection"]["markers"] if marker in lowered
    ]


def generate_with_retries(
    client: OllamaClient,
    *,
    model: str,
    prompt: str,
    token_limit: int,
    retries: int,
    timeout_seconds: float,
    timeout_multiplier: float,
    validator: Any | None = None,
) -> tuple[Any | None, int, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    for attempt in range(retries + 1):
        try:
            result = client.generate(
                model,
                prompt
                + (
                    "\n\nYour prior response violated the locked-recommendation contract. "
                    "Name the exact recommended item verbatim; do not substitute an alternative. "
                    "Use only separate [K###] citations from the supplied trace."
                    if attempt
                    else ""
                ),
                token_limit=token_limit,
                timeout_seconds=timeout_seconds * (timeout_multiplier**attempt),
            )
            if result.text.strip() and (validator is None or validator(result.text) is None):
                return result, attempt, errors
            raise ValueError("empty_or_whitespace_generation")
        except Exception as error:  # the immutable retry log preserves the exact failure class
            errors.append({"error_type": type(error).__name__, "message": str(error)})
    return None, retries, errors


def generation_summary(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for (generator, condition), group in frame.groupby(["generator", "condition"], sort=True):
        groups.append((str(generator), str(condition), group))
    for condition, group in frame.groupby("condition", sort=True):
        groups.append(("all_generators", str(condition), group))
    rows = []
    for generator, condition, group in groups:
        words = group["word_count"].astype(float)
        rows.append(
            {
                "generator": generator,
                "condition": condition,
                "generations": len(group),
                "mean_words": words.mean(),
                "std_words": words.std(ddof=0),
                "min_words": words.min(),
                "median_words": words.median(),
                "max_words": words.max(),
                "p05_words": words.quantile(0.05),
                "p95_words": words.quantile(0.95),
                "mean_latency_seconds": group["latency_seconds"].mean(),
                "total_retries": group["retry_count"].sum(),
                "retried_generations": group["retry_count"].gt(0).sum(),
                "refusals": group["refusal_detected"].sum(),
                "malformed_or_empty": group["malformed_or_empty"].sum(),
                "word_limit_violations": (
                    words.gt(group["requested_word_limit"].astype(float)).sum()
                    if group["requested_word_limit"].notna().all()
                    else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_generation_integrity(
    records: Sequence[Mapping[str, Any]],
    packets: Sequence[Mapping[str, Any]],
    selected_by_id: Mapping[str, Mapping[str, Any]],
    settings: Mapping[str, Any],
    models: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected = (
        len(packets)
        * len(models["generators"]["roster"])
        * len(config["explanations"]["conditions"])
    )
    keys = [(row["case_id"], row["generator"], row["condition"]) for row in records]
    if len(records) != expected or len(set(keys)) != expected:
        raise ValueError("Stage 7 generation matrix is incomplete or contains duplicate keys.")
    packet_by_id = {row["case_id"]: row for row in packets}
    trace_mismatches = 0
    prompt_mismatches = 0
    condition_label_leaks = 0
    output_hash_mismatches = 0
    for row in records:
        case = selected_by_id[row["case_id"]]
        packet = packet_by_id[row["case_id"]]
        if packet["B_exact_stored_trace"] != case["evidence_trace"]:
            trace_mismatches += 1
        if row["condition"] == "no_rag":
            expected_prompt = build_no_rag_prompt(case, int(config["stage7"]["no_rag_word_limit"]))
            expected_ids: list[str] = []
            if "Evidence rules:" in row["prompt"]:
                prompt_mismatches += 1
        else:
            expected_prompt, expected_ids = build_rule_rag_prompt(case, settings)
            if set(expected_ids) != {
                rule["rule_id"] for rule in packet["B_exact_stored_trace"]["rules"]
            }:
                trace_mismatches += 1
        if row["prompt"] != expected_prompt or row["prompt_rule_ids"] != expected_ids:
            prompt_mismatches += 1
        if row["prompt_sha256"] != text_sha256(row["prompt"]):
            prompt_mismatches += 1
        if "no_rag" in row["prompt"].lower() or "rule_rag" in row["prompt"].lower():
            condition_label_leaks += 1
        if row["output_sha256"] != text_sha256(row["output_text"]):
            output_hash_mismatches += 1
    checks = {
        "expected_generations": expected,
        "observed_generations": len(records),
        "unique_generation_keys": len(set(keys)),
        "case_packets": len(packets),
        "five_rule_packets": sum(len(row["B_exact_stored_trace"]["rules"]) == 5 for row in packets),
        "trace_mismatches": trace_mismatches,
        "prompt_mismatches": prompt_mismatches,
        "condition_label_leaks": condition_label_leaks,
        "output_hash_mismatches": output_hash_mismatches,
        "malformed_or_empty": sum(bool(row["malformed_or_empty"]) for row in records),
        "refusals": sum(bool(row["refusal_detected"]) for row in records),
    }
    if any(
        checks[key]
        for key in (
            "trace_mismatches",
            "prompt_mismatches",
            "condition_label_leaks",
            "output_hash_mismatches",
            "malformed_or_empty",
        )
    ):
        raise ValueError(f"Stage 7 integrity checks failed: {checks}")
    return checks


def _update_registry(config_digest: str, summary_path: Path) -> None:
    registry = Path("artifacts/manifests/figure_table_registry.csv")
    rows = pd.read_csv(registry, dtype=str).fillna("")
    rows = rows[~rows["artifact_id"].eq("table_stage7_generation_summary")]
    addition = pd.DataFrame(
        [
            {
                "artifact_id": "table_stage7_generation_summary",
                "artifact_type": "table",
                "title": "Stage 7 frozen explanation generation summary",
                "research_question": "Was the full frozen explanation matrix generated intact?",
                "source_data": ".runtime/stage7",
                "generation_function_or_script": (
                    "scripts/run_stage7_explanation_generation.py:generation_summary"
                ),
                "configuration_hash": config_digest,
                "output_path": str(summary_path),
                "caption": (
                    "Generation counts, lengths, retries, refusals, and empty-output audit "
                    "for the immutable Stage 7 corpus."
                ),
                "intended_thesis_chapter": "Methods and results",
                "intended_paper_section": "Explanation generation",
                "status": "final",
                "notes": "Generation only; extraction, verification, and judging begin in Stage 8.",
            }
        ]
    )
    pd.concat([rows, addition], ignore_index=True).to_csv(registry, index=False)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    config_digest = configuration_hash(resolved)
    settings = config["stage7"]
    stage6_manifest_path = Path(settings["source_manifest"])
    stage6_manifest = json.loads(stage6_manifest_path.read_text(encoding="utf-8"))
    validate_frozen_stage2_to_6(config, stage6_manifest)
    locked_path, locked_hash = _locked_input(stage6_manifest)
    locked = read_jsonl(locked_path)
    selected = select_stage7_cases(locked, config)
    selected_by_id = {row["case_id"]: row for row in selected}
    frozen_path = Path(settings["frozen_prompt_settings"])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["configuration_id"] != settings["rule_rag_prompt_configuration"]:
        raise ValueError("Stage 7 prompt ID does not match the frozen Stage 5 setting.")
    prompt_settings = frozen["settings"]
    configured_prompt = next(
        row
        for row in config["explanation_search"]["candidate_configurations"]
        if row["id"] == config["explanation_search"]["frozen_configuration_id"]
    )
    if prompt_settings != configured_prompt:
        raise ValueError("Frozen Stage 5 prompt settings differ from the current configuration.")
    packets = build_case_packets(selected, prompt_settings)
    expected = (
        len(selected)
        * len(models["generators"]["roster"])
        * len(config["explanations"]["conditions"])
    )
    run_id = f"stage7-generation-{config_digest[:12]}"
    runtime_root = args.runtime_root or Path(config["paths"]["runtime_root"])
    run_dir = runtime_root / "stage7" / run_id
    packet_path = run_dir / "case_evidence_packets.jsonl"
    progress_path = run_dir / "generation_progress.jsonl"
    retry_log_path = run_dir / "retry_failures.jsonl"
    final_path = run_dir / "generations.jsonl"
    runtime_summary_path = run_dir / "generation_summary.csv"
    runtime_manifest_path = run_dir / "manifest.json"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "configuration_hash": config_digest,
                    "locked_input": str(locked_path),
                    "cases": len(selected),
                    "generators": [row["model_id"] for row in models["generators"]["roster"]],
                    "conditions": config["explanations"]["conditions"],
                    "planned_generations": expected,
                    "would_call_local_models": True,
                    "would_run_stage8": False,
                },
                indent=2,
            )
        )
        return
    validate_generator_digests(models, args.ollama_endpoint)
    if final_path.exists() or runtime_manifest_path.exists():
        raise FileExistsError(f"Immutable Stage 7 run already exists: {run_dir}")
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Partial Stage 7 run exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if packet_path.exists():
        if read_jsonl(packet_path) != packets:
            raise ValueError("Existing Stage 7 case packets differ from deterministic selection.")
    else:
        write_jsonl_new(packet_path, packets)
    reuse_manifest_path: Path | None = None
    reuse_generations_path: Path | None = None
    reuse_manifest: dict[str, Any] | None = None
    if settings.get("reuse_rule_rag_source_manifest"):
        reuse_manifest_path = Path(settings["reuse_rule_rag_source_manifest"])
        reuse_manifest = json.loads(reuse_manifest_path.read_text(encoding="utf-8"))
        reuse_generations_path, _ = _manifest_output(reuse_manifest, "generations.jsonl")
    if not progress_path.exists() and reuse_manifest is not None:
        selected_ids = set(selected_by_id)
        generator_digests = {
            row["model_id"]: row["immutable_digest"] for row in models["generators"]["roster"]
        }
        reused = []
        for row in read_jsonl(reuse_generations_path):
            if row["condition"] != "rule_rag":
                continue
            if row["case_id"] not in selected_ids:
                raise ValueError("Reused Rule-RAG row is outside the selected case set.")
            if generator_digests.get(row["generator"]) != row["generator_immutable_digest"]:
                raise ValueError("Reused Rule-RAG generator digest changed.")
            expected_prompt, expected_ids = build_rule_rag_prompt(
                selected_by_id[row["case_id"]], prompt_settings
            )
            if row["prompt"] != expected_prompt or row["prompt_rule_ids"] != expected_ids:
                raise ValueError("Reused Rule-RAG prompt differs from the frozen prompt.")
            if int(row["requested_word_limit"]) != int(prompt_settings["word_limit"]):
                raise ValueError("Reused Rule-RAG row has a different word limit.")
            reused.append(row)
        expected_reused = len(selected) * len(models["generators"]["roster"])
        if len(reused) != expected_reused:
            raise ValueError(
                f"Expected {expected_reused} reusable Rule-RAG rows; found {len(reused)}."
            )
        write_jsonl_new(progress_path, reused)
        print(f"reused {len(reused)} unchanged Rule-RAG generations", flush=True)
    existing = read_jsonl(progress_path) if progress_path.exists() else []
    completed = {(row["case_id"], row["generator"], row["condition"]) for row in existing}
    if len(completed) != len(existing):
        raise ValueError("Stage 7 progress contains duplicate generation keys.")
    client = OllamaClient(models["generation_defaults"], endpoint=args.ollama_endpoint)
    retries = int(settings["generation_retry_attempts"])
    timeout_multiplier = float(settings["retry_timeout_multiplier"])
    token_limit = int(config["explanations"]["generation_token_safety_ceiling"])
    report_interval = int(settings["progress_report_interval"])
    new_failures = 0
    for generator_settings in models["generators"]["roster"]:
        generator = generator_settings["model_id"]
        for case in selected:
            no_rag_word_limit = int(settings["no_rag_word_limit"])
            prompts = {"no_rag": (build_no_rag_prompt(case, no_rag_word_limit), [])}
            rule_prompt, rule_ids = build_rule_rag_prompt(case, prompt_settings)
            prompts["rule_rag"] = (rule_prompt, rule_ids)
            for condition in config["explanations"]["conditions"]:
                key = (case["case_id"], generator, condition)
                if key in completed:
                    continue
                prompt, prompt_rule_ids = prompts[condition]

                def contract_validator(
                    text: str,
                    *,
                    locked_name: str = str(case["locked_candidate_minimal_name"]),
                    target: str = str(case["target_category"]),
                    trace_ids: Sequence[str] = tuple(rule_ids),
                    required: bool = condition == "rule_rag",
                ) -> None:
                    validate_generated_explanation(
                        text,
                        locked_item_name=locked_name,
                        target_category=target,
                        trace_rule_ids=trace_ids,
                        citations_required=required,
                    )

                result, retry_count, errors = generate_with_retries(
                    client,
                    model=generator,
                    prompt=prompt,
                    token_limit=token_limit,
                    retries=retries,
                    timeout_seconds=float(models["generation_defaults"]["timeout_seconds"]),
                    timeout_multiplier=timeout_multiplier,
                    validator=contract_validator,
                )
                for attempt_index, error in enumerate(errors, start=1):
                    append_jsonl(
                        retry_log_path,
                        {
                            "case_id": case["case_id"],
                            "generator": generator,
                            "condition": condition,
                            "attempt": attempt_index,
                            **error,
                        },
                    )
                if result is None:
                    new_failures += 1
                    append_jsonl(
                        retry_log_path,
                        {
                            "case_id": case["case_id"],
                            "generator": generator,
                            "condition": condition,
                            "status": "terminal_contract_failure",
                            "errors": errors,
                        },
                    )
                    print(f"exhausted {key}", flush=True)
                    continue
                markers = refusal_markers(result.text, config)
                record = {
                    "case_id": case["case_id"],
                    "query_outfit_id": case["query_outfit_id"],
                    "target_category": case["target_category"],
                    "generator": generator,
                    "generator_immutable_digest": generator_settings["immutable_digest"],
                    "condition": condition,
                    "A_sha256": canonical_hash(common_context(case)),
                    "B_sha256": canonical_hash(case["evidence_trace"]),
                    "evidence_shown": "A" if condition == "no_rag" else "A_plus_B",
                    "prompt_configuration_id": (
                        settings["no_rag_prompt_template"]
                        if condition == "no_rag"
                        else prompt_settings["id"]
                    ),
                    "prompt": prompt,
                    "prompt_sha256": text_sha256(prompt),
                    "prompt_rule_ids": prompt_rule_ids,
                    "requested_word_limit": (
                        no_rag_word_limit
                        if condition == "no_rag"
                        else prompt_settings["word_limit"]
                    ),
                    "generation_token_safety_ceiling": token_limit,
                    "output_text": result.text,
                    "output_sha256": text_sha256(result.text),
                    "word_count": word_count(result.text),
                    "latency_seconds": result.latency_seconds,
                    "prompt_eval_count": result.prompt_eval_count,
                    "eval_count": result.eval_count,
                    "total_duration_ns": result.total_duration_ns,
                    "retry_count": retry_count,
                    "retry_errors": errors,
                    "malformed_or_empty": not bool(result.text.strip()),
                    "refusal_detected": bool(markers),
                    "refusal_markers": markers,
                }
                append_jsonl(progress_path, record)
                existing.append(record)
                completed.add(key)
                if len(existing) % report_interval == 0:
                    print(
                        f"stage7 {len(existing)}/{expected} generator={generator} "
                        f"condition={condition} retries={retry_count}",
                        flush=True,
                    )
        client.unload(generator)
    if new_failures or len(existing) != expected:
        raise RuntimeError(
            f"Stage 7 remains incomplete: {len(existing)}/{expected}; "
            f"new exhausted keys={new_failures}. Resume after resolving failures."
        )
    integrity = validate_generation_integrity(
        existing, packets, selected_by_id, prompt_settings, models, config
    )
    write_jsonl_new(final_path, existing)
    summary = generation_summary(existing)
    summary.to_csv(runtime_summary_path, index=False)
    tracked_summary_path = Path("artifacts/tables/table_stage7_generation_summary.csv")
    summary.to_csv(tracked_summary_path, index=False)
    _update_registry(config_digest, tracked_summary_path)
    registry_path = Path("artifacts/manifests/figure_table_registry.csv")
    retry_records = read_jsonl(retry_log_path) if retry_log_path.exists() else []
    outputs = [packet_path, final_path, runtime_summary_path, tracked_summary_path, registry_path]
    if retry_log_path.exists():
        outputs.append(retry_log_path)
    failure_types = Counter(row["error_type"] for row in retry_records)
    manifest = {
        "schema_version": 1,
        "stage": 7,
        "stage_name": "full_explanation_generation",
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": config_digest,
        "resolved_configuration": resolved,
        "input_artifact_hashes": {
            str(stage6_manifest_path): sha256_file(stage6_manifest_path),
            str(locked_path): locked_hash,
            str(frozen_path): sha256_file(frozen_path),
            str(args.models_config): sha256_file(args.models_config),
            **(
                {
                    str(reuse_manifest_path): sha256_file(reuse_manifest_path),
                    str(reuse_generations_path): sha256_file(reuse_generations_path),
                }
                if reuse_manifest_path is not None and reuse_generations_path is not None
                else {}
            ),
        },
        "output_artifact_hashes": {str(path): sha256_file(path) for path in outputs},
        "models": {
            "generators": models["generators"],
            "generation_defaults": models["generation_defaults"],
        },
        "selected_settings": prompt_settings,
        "case_selection": {
            "method": settings["case_selection_method"],
            "seed": settings["case_selection_seed"],
            "cases_per_category": settings["cases_per_category"],
            "category_counts": dict(Counter(row["target_category"] for row in selected)),
        },
        "row_counts": {
            "source_locked_cases": len(locked),
            "selected_cases": len(selected),
            "case_evidence_packets": len(packets),
            "generators": len(models["generators"]["roster"]),
            "conditions": len(config["explanations"]["conditions"]),
            "generations": len(existing),
        },
        "failure_counts": {
            "exhausted_generation_keys": 0,
            "retry_attempt_failures": len(retry_records),
            "retried_generations": sum(row["retry_count"] > 0 for row in existing),
            "malformed_or_empty": integrity["malformed_or_empty"],
            "refusals": integrity["refusals"],
            "retry_failure_types": dict(failure_types),
        },
        "reuse": {
            "unchanged_rule_rag_generations": sum(
                row["condition"] == "rule_rag" for row in existing
            ),
            "source_manifest": (
                str(reuse_manifest_path) if reuse_manifest_path is not None else None
            ),
        },
        "integrity_checks": integrity,
        "status": {
            "generation": "complete_immutable",
            "claim_extraction": "not_started_stage8",
            "claim_verification": "not_started_stage8",
            "general_judging": "not_started_stage8",
        },
        "inference_server_version": models["generation_defaults"]["inference_server_version"],
        "device": models["generation_defaults"]["device"],
        "environment": environment_summary(),
        "command": (
            "python scripts/run_stage7_explanation_generation.py --config configs/experiment.yaml"
        ),
    }
    write_new_json(runtime_manifest_path, manifest)
    tracked_manifest = Path("artifacts/manifests/stage7_explanation_generation_manifest.json")
    write_json(tracked_manifest, manifest)
    write_json(Path("artifacts/manifests/explanation_generation_manifest.json"), manifest)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "generations": len(existing),
                "failure_counts": manifest["failure_counts"],
                "integrity_checks": integrity,
                "verification_status": manifest["status"]["claim_verification"],
                "summary": summary.to_dict("records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
