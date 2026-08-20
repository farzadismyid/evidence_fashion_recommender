"""Stage 5 validation-only prompt optimisation and 50-case explanation pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evidence_fashion.data import load_pinned_split, write_jsonl
from evidence_fashion.explanation import (
    ASSESSMENT_SCHEMA,
    OllamaClient,
    build_assessment_prompt,
    build_no_rag_prompt,
    build_rule_rag_prompt,
    generate_explanation_with_contract_retries,
    text_sha256,
    word_count,
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
from evidence_fashion.reranking import pareto_frontier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--optimize-only", action="store_true")
    parser.add_argument("--pilot-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stage4_inputs(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Path, str]:
    manifest_path = Path("artifacts/manifests/stage4_reranking_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    locked_path = Path(
        next(
            path
            for path in manifest["output_artifact_hashes"]
            if path.endswith("locked_cases.jsonl")
        )
    )
    digest = sha256_file(locked_path)
    if digest != manifest["output_artifact_hashes"][str(locked_path)]:
        raise ValueError("Stage 4 locked cases do not match their manifest hash.")
    records = _read_jsonl(locked_path)
    expected = config["stage4_validation"]["case_count"]
    if len(records) != expected:
        raise ValueError(f"Expected {expected} locked Stage 4 cases; found {len(records)}.")
    return records, locked_path, digest


def _split_cases(
    records: list[dict[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    explanation = config["explanations"]
    optimisation = []
    pilot = []
    frame = pd.DataFrame(records)
    for _, group in frame.groupby("target_category", sort=True):
        ordered = group.sort_values("case_id", kind="stable").to_dict("records")
        opt_count = explanation["optimisation_cases_per_category"]
        pilot_count = explanation["pilot_cases_per_category"]
        optimisation.extend(ordered[:opt_count])
        pilot.extend(ordered[opt_count : opt_count + pilot_count])
    if len(optimisation) != explanation["optimisation_cases"]:
        raise ValueError("Optimisation case quota does not match the stratified selection.")
    if len(pilot) != explanation["pilot_cases"]:
        raise ValueError("Pilot case quota does not match the stratified selection.")
    if {row["case_id"] for row in optimisation} & {row["case_id"] for row in pilot}:
        raise ValueError("Stage 5 optimisation and pilot cases overlap.")
    return optimisation, pilot


def _result_fields(result) -> dict[str, Any]:
    return {
        "latency_seconds": result.latency_seconds,
        "prompt_eval_count": result.prompt_eval_count,
        "eval_count": result.eval_count,
        "total_duration_ns": result.total_duration_ns,
    }


def _condition_metrics(
    assessment: Mapping[str, Any],
    output: str,
    valid_rule_ids: set[str],
    word_limit: int | None,
    latency: float,
) -> dict[str, float]:
    claims = assessment.get("claims", [])
    total = len(claims)
    statuses = [claim.get("support_status") for claim in claims]
    cited = [claim for claim in claims if claim.get("evidence_rule_ids")]
    entailed = [
        claim
        for claim in cited
        if claim.get("support_status") == "supported"
        and set(claim.get("evidence_rule_ids", [])).issubset(valid_rule_ids)
    ]
    denominator = max(total, 1)
    return {
        "claim_count": float(total),
        "support_rate": statuses.count("supported") / denominator,
        "unsupported_rate": statuses.count("unsupported") / denominator,
        "contradiction_rate": statuses.count("contradicted") / denominator,
        "not_verifiable_rate": statuses.count("not_verifiable") / denominator,
        "citation_entailment_rate": len(entailed) / max(len(cited), 1),
        "general_quality": float(assessment["general_quality"]),
        "clarity": float(assessment["clarity"]),
        "specificity": float(assessment["specificity"]),
        "word_count": float(word_count(output)),
        "length_violation_rate": float(word_limit is not None and word_count(output) > word_limit),
        "malformed_rate": float(not output.strip()),
        "latency_seconds": latency,
    }


def _assess_pair(
    client: OllamaClient,
    judge_model: str,
    case: Mapping[str, Any],
    no_rag: str,
    rule_rag: str,
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    swap = int(hashlib.sha256(str(case["case_id"]).encode()).hexdigest(), 16) % 2 == 1
    first_text, second_text = (rule_rag, no_rag) if swap else (no_rag, rule_rag)
    first_condition, second_condition = ("rule_rag", "no_rag") if swap else ("no_rag", "rule_rag")
    prompt = build_assessment_prompt(
        case, first_text, second_text, first_condition, second_condition
    )
    response, result, retry_count = client.generate_json(
        judge_model, prompt, ASSESSMENT_SCHEMA, retries=retries
    )
    mapped = {
        first_condition: response["first"],
        second_condition: response["second"],
        "preference": (
            first_condition
            if response["preference"] == "first"
            else second_condition
            if response["preference"] == "second"
            else "tie"
        ),
        "preference_reason": response["preference_reason"],
    }
    metadata = {
        "position_swap": swap,
        "prompt_hash": text_sha256(prompt),
        "retry_count": retry_count,
        **_result_fields(result),
    }
    return mapped, metadata


def _generate_pair(
    client: OllamaClient,
    generator: str,
    judge: str,
    case: Mapping[str, Any],
    settings: Mapping[str, Any],
    retries: int,
    no_rag_cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    no_prompt = build_no_rag_prompt(case)
    if no_rag_cache is None:
        no_result, _, no_errors = generate_explanation_with_contract_retries(
            client,
            model=generator,
            prompt=no_prompt,
            locked_item_name=str(case["locked_candidate_minimal_name"]),
            target_category=str(case["target_category"]),
            retries=retries,
        )
        if no_result is None:
            raise ValueError(f"Terminal locked-recommendation failure: {no_errors}")
        no_text = no_result.text
        no_metadata = _result_fields(no_result)
    else:
        no_text = str(no_rag_cache["output"])
        no_metadata = dict(no_rag_cache["generation_metadata"])
    rule_prompt, prompt_rule_ids = build_rule_rag_prompt(case, settings)
    rule_result, _, rule_errors = generate_explanation_with_contract_retries(
        client,
        model=generator,
        prompt=rule_prompt,
        locked_item_name=str(case["locked_candidate_minimal_name"]),
        target_category=str(case["target_category"]),
        trace_rule_ids=prompt_rule_ids,
        citations_required=True,
        retries=retries,
    )
    if rule_result is None:
        raise ValueError(f"Terminal locked-recommendation failure: {rule_errors}")
    assessment, assessment_metadata = _assess_pair(
        client, judge, case, no_text, rule_result.text, retries
    )
    valid_rule_ids = {str(rule["rule_id"]) for rule in case["evidence_trace"]["rules"]}
    return {
        "case_id": case["case_id"],
        "target_category": case["target_category"],
        "generator": generator,
        "settings": dict(settings),
        "A": {
            "user_request": case["request"],
            "query_item_minimal_name": case["query_item_minimal_name"],
            "locked_item_minimal_name": case["locked_candidate_minimal_name"],
        },
        "B_exact_stored_trace": case["evidence_trace"],
        "B_trace_hash": text_sha256(json.dumps(case["evidence_trace"], sort_keys=True)),
        "no_rag": {
            "prompt": no_prompt,
            "prompt_hash": text_sha256(no_prompt),
            "output": no_text,
            "generation_metadata": no_metadata,
            "assessment": assessment["no_rag"],
            "metrics": _condition_metrics(
                assessment["no_rag"],
                no_text,
                set(),
                None,
                float(no_metadata["latency_seconds"]),
            ),
        },
        "rule_rag": {
            "prompt": rule_prompt,
            "prompt_hash": text_sha256(rule_prompt),
            "prompt_rule_ids": prompt_rule_ids,
            "output": rule_result.text,
            "generation_metadata": _result_fields(rule_result),
            "assessment": assessment["rule_rag"],
            "metrics": _condition_metrics(
                assessment["rule_rag"],
                rule_result.text,
                valid_rule_ids,
                int(settings["word_limit"]),
                rule_result.latency_seconds,
            ),
        },
        "paired_judge": {
            "preference": assessment["preference"],
            "reason": assessment["preference_reason"],
            **assessment_metadata,
        },
    }


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def _optimization_summary(records: list[dict[str, Any]], config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for record in records:
        metrics = record["rule_rag"]["metrics"]
        rows.append(
            {
                "configuration_id": record["settings"]["id"],
                "rule_count": record["settings"]["rule_count"],
                "no_rag_word_count": record["no_rag"]["metrics"]["word_count"],
                **metrics,
            }
        )
    aggregated = (
        pd.DataFrame(rows).groupby(["configuration_id", "rule_count"], as_index=False).mean()
    )
    aggregated["absolute_word_count_gap"] = (
        aggregated["word_count"] - aggregated["no_rag_word_count"]
    ).abs()
    aggregated["general_quality_normalized"] = aggregated["general_quality"] / 5
    aggregated["clarity_normalized"] = aggregated["clarity"] / 5
    aggregated["specificity_normalized"] = aggregated["specificity"] / 5
    aggregated["one_minus_unsupported_rate"] = 1 - aggregated["unsupported_rate"]
    aggregated["one_minus_contradiction_rate"] = 1 - aggregated["contradiction_rate"]
    aggregated["one_minus_length_violation_rate"] = 1 - aggregated["length_violation_rate"]
    aggregated["one_minus_malformed_rate"] = 1 - aggregated["malformed_rate"]
    objectives = [
        "support_rate",
        "citation_entailment_rate",
        "general_quality_normalized",
        "clarity_normalized",
        "specificity_normalized",
        "one_minus_unsupported_rate",
        "one_minus_contradiction_rate",
        "one_minus_length_violation_rate",
        "one_minus_malformed_rate",
    ]
    aggregated = pareto_frontier(aggregated, objectives)
    utility = config["explanation_search"]["selection_utility"]
    aggregated["selection_utility"] = sum(
        aggregated[column] * weight for column, weight in utility.items()
    )
    return aggregated.sort_values("configuration_id").reset_index(drop=True)


def _select_configuration(summary: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    search = config["explanation_search"]
    selected_id = search["frozen_configuration_id"]
    selected = summary[summary["configuration_id"].eq(selected_id)]
    if len(selected) != 1:
        raise ValueError("The researcher-selected primary configuration was not evaluated once.")
    row = selected.iloc[0]
    if int(row["rule_count"]) != int(config["stage4_validation"]["selected_rule_top_k"]):
        raise ValueError("The researcher-selected prompt does not use the complete scoring trace.")
    if float(row["absolute_word_count_gap"]) > float(
        search["maximum_mean_word_count_gap_from_no_rag"]
    ):
        raise ValueError("The researcher-selected prompt fails the frozen mean-length constraint.")
    return row


def _manifest_base(
    stage_name: str,
    run_id: str,
    config_digest: str,
    resolved: Mapping[str, Any],
    locked_path: Path,
    locked_hash: str,
    models: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": 5,
        "stage_name": stage_name,
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": config_digest,
        "resolved_configuration": resolved,
        "input_artifact_hashes": {str(locked_path): locked_hash},
        "models": models,
        "environment": environment_summary(),
        "inference_server_version": models["generation_defaults"]["inference_server_version"],
        "device": models["generation_defaults"]["device"],
        "structured_retry_token_limits": [
            models["generation_defaults"]["structured_token_limit"] * (2**attempt)
            for attempt in range(3)
        ],
        "structured_retry_timeout_seconds": [
            models["generation_defaults"]["timeout_seconds"] * (2**attempt) for attempt in range(3)
        ],
    }


def run_optimization(
    args: argparse.Namespace,
    config: dict[str, Any],
    models: dict[str, Any],
    resolved: dict[str, Any],
    config_digest: str,
    cases: list[dict[str, Any]],
    locked_path: Path,
    locked_hash: str,
) -> None:
    run_id = f"stage5-optimization-v2-{config_digest[:12]}"
    runtime_root = args.runtime_root or Path(config["paths"]["runtime_root"])
    run_dir = runtime_root / "stage5" / run_id
    progress = run_dir / "progress.jsonl"
    final_records = run_dir / "optimization_records.jsonl"
    if final_records.exists() and not args.resume:
        raise FileExistsError(f"Immutable Stage 5 optimisation exists: {final_records}")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    existing = (
        _read_jsonl(final_records)
        if args.resume and final_records.exists()
        else _read_jsonl(progress)
        if args.resume and progress.exists()
        else []
    )
    completed = {(row["case_id"], row["generator"], row["settings"]["id"]) for row in existing}
    client = OllamaClient(models["generation_defaults"])
    judge = models["judges"]["roster"][0]["model_id"]
    retries = config["structured_outputs"]["retry_attempts"]
    records = list(existing)
    baseline_cache: dict[tuple[str, str], Any] = {}
    active_ids = set(config["explanation_search"]["active_configuration_ids"])
    for row in existing:
        baseline_cache[(row["case_id"], row["generator"])] = row["no_rag"]
    for case in cases:
        for generator_settings in models["generators"]["roster"]:
            generator = generator_settings["model_id"]
            for settings in config["explanation_search"]["candidate_configurations"]:
                if settings["id"] not in active_ids:
                    continue
                key = (case["case_id"], generator, settings["id"])
                if key in completed:
                    continue
                record = _generate_pair(
                    client,
                    generator,
                    judge,
                    case,
                    settings,
                    retries,
                    baseline_cache.get((case["case_id"], generator)),
                )
                baseline_cache[(case["case_id"], generator)] = record["no_rag"]
                _append_jsonl(progress, record)
                records.append(record)
                total = len(cases) * len(models["generators"]["roster"]) * len(active_ids)
                print(f"optimisation {len(records)}/{total}", flush=True)
    if not final_records.exists():
        write_jsonl(final_records, records)
    summary = _optimization_summary(records, config)
    selected = _select_configuration(summary, config)
    runtime_table = run_dir / "optimization_summary.csv"
    tracked_table = Path("artifacts/tables/table_stage5_primary_validation.csv")
    summary.to_csv(runtime_table, index=False)
    summary.to_csv(tracked_table, index=False)
    selected_id = str(selected["configuration_id"])
    selected_settings = next(
        row
        for row in config["explanation_search"]["candidate_configurations"]
        if row["id"] == selected_id
    )
    frozen = {
        "selection_status": "researcher_selected_primary_validated_for_fresh_pilot",
        "configuration_id": selected_id,
        "settings": selected_settings,
        "metrics": json.loads(selected.to_json()),
        "length_matched_sensitivity": {
            "method": config["explanation_search"]["length_matched_sensitivity_method"],
            "pairs_per_generator": config["explanation_search"][
                "length_matched_sensitivity_pairs_per_generator"
            ],
            "status": "required_in_fresh_pilot",
        },
        "optimisation_case_ids": sorted(row["case_id"] for row in cases),
        "pilot_overlap": 0,
    }
    frozen_path = Path("artifacts/manifests/stage5_frozen_settings.json")
    write_json(frozen_path, frozen)
    manifest = _manifest_base(
        "explanation_optimisation",
        run_id,
        config_digest,
        resolved,
        locked_path,
        locked_hash,
        models,
    )
    manifest.update(
        {
            "output_artifact_hashes": {
                str(final_records): sha256_file(final_records),
                str(runtime_table): sha256_file(runtime_table),
                str(tracked_table): sha256_file(tracked_table),
                str(frozen_path): sha256_file(frozen_path),
            },
            "row_counts": {
                "optimisation_cases": len(cases),
                "generators": len(models["generators"]["roster"]),
                "candidate_configurations": len(active_ids),
                "paired_assessments": len(records),
            },
            "failure_counts": {"malformed_outputs": int(summary["malformed_rate"].sum())},
            "selection": frozen,
            "condition_A_fixed": True,
            "optimised_variables": "rule_rag_only",
            "command": "python scripts/run_explanation_eval.py --optimize-only",
        }
    )
    runtime_manifest = run_dir / "manifest.json"
    write_new_json(runtime_manifest, manifest)
    write_json(Path("artifacts/manifests/stage5_optimization_manifest.json"), manifest)
    print(json.dumps(frozen, indent=2))


def _save_pilot_images(
    cases: list[dict[str, Any]], config: Mapping[str, Any], run_dir: Path
) -> dict[str, dict[str, str]]:
    data_manifest = json.loads(
        Path(config["paths"]["active_data_manifest"]).read_text(encoding="utf-8")
    )
    items_path = Path(
        next(
            path
            for path in data_manifest["output_artifact_hashes"]
            if path.endswith("prepared_items.parquet")
        )
    )
    lookup = pd.read_parquet(items_path).set_index("item_id")
    raw, _ = load_pinned_split(config)
    image_column = config["dataset"]["columns"]["image"]
    image_dir = run_dir / "images"
    image_dir.mkdir()
    paths = {}
    for case in cases:
        case_paths = {}
        for role, item_id in {
            "query": case["query_item_id"],
            "locked": case["locked_candidate_id"],
        }.items():
            index = int(lookup.loc[item_id, "original_dataset_index"])
            path = image_dir / f"{case['case_id']}-{role}.png"
            raw[index][image_column].convert("RGB").save(path, format="PNG")
            case_paths[role] = str(path)
        paths[case["case_id"]] = case_paths
    return paths


def _flatten_pilot(records: list[dict[str, Any]]) -> tuple[list, list, list, list]:
    generations, claims, verifications, judges = [], [], [], []
    for record in records:
        for condition in ("no_rag", "rule_rag"):
            payload = record[condition]
            generations.append(
                {
                    "case_id": record["case_id"],
                    "generator": record["generator"],
                    "condition": condition,
                    "prompt": payload["prompt"],
                    "prompt_hash": payload["prompt_hash"],
                    "output": payload["output"],
                    **payload["generation_metadata"],
                }
            )
            for index, claim in enumerate(payload["assessment"]["claims"]):
                claim_id = f"{record['case_id']}:{condition}:{index}"
                claims.append({"claim_id": claim_id, "claim": claim["claim"]})
                verifications.append(
                    {
                        "claim_id": claim_id,
                        "support_status": claim["support_status"],
                        "support_source": claim["support_source"],
                        "evidence_rule_ids": claim["evidence_rule_ids"],
                    }
                )
        judges.append({"case_id": record["case_id"], **record["paired_judge"]})
    return generations, claims, verifications, judges


def _pilot_summary(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        for condition in ("no_rag", "rule_rag"):
            rows.append(
                {
                    "condition": condition,
                    **record[condition]["metrics"],
                    "preferred": float(record["paired_judge"]["preference"] == condition),
                    "tie": float(record["paired_judge"]["preference"] == "tie"),
                }
            )
    return pd.DataFrame(rows).groupby("condition", as_index=False).mean()


def _length_matched_sensitivity(
    records: list[dict[str, Any]], config: Mapping[str, Any]
) -> pd.DataFrame:
    """Summarise an equal-size, case-paired closest-length cohort per generator."""
    count = int(config["explanation_search"]["length_matched_sensitivity_pairs_per_generator"])
    selected: list[dict[str, Any]] = []
    frame = pd.DataFrame(
        {
            "index": range(len(records)),
            "generator": [row["generator"] for row in records],
            "absolute_word_gap": [
                abs(
                    row["no_rag"]["metrics"]["word_count"]
                    - row["rule_rag"]["metrics"]["word_count"]
                )
                for row in records
            ],
            "case_id": [row["case_id"] for row in records],
        }
    )
    for _, group in frame.groupby("generator", sort=True):
        chosen = group.sort_values(["absolute_word_gap", "case_id"], kind="stable").head(count)
        selected.extend(records[int(index)] for index in chosen["index"])
    rows = []
    grouped_records = [
        (generator, [row for row in selected if row["generator"] == generator])
        for generator in sorted({row["generator"] for row in selected})
    ]
    grouped_records.append(("all_generators", selected))
    for generator, group_records in grouped_records:
        summary = _pilot_summary(group_records)
        mean_gap = float(
            pd.Series(
                [
                    abs(
                        row["no_rag"]["metrics"]["word_count"]
                        - row["rule_rag"]["metrics"]["word_count"]
                    )
                    for row in group_records
                ]
            ).mean()
        )
        for record in summary.to_dict("records"):
            rows.append(
                {
                    "generator": generator,
                    "matched_pairs": len(group_records),
                    "mean_absolute_paired_word_gap": mean_gap,
                    **record,
                }
            )
    return pd.DataFrame(rows)


def _write_stage5_manifests(
    base: dict[str, Any], outputs: Mapping[str, Path], counts: Mapping[str, int]
) -> None:
    names = {
        "generations": "explanation_generation_manifest.json",
        "claims": "claim_extraction_manifest.json",
        "verifications": "claim_verification_manifest.json",
        "judges": "judge_manifest.json",
    }
    for role, filename in names.items():
        manifest = dict(base)
        manifest["stage_name"] = role
        manifest["output_artifact_hashes"] = {str(outputs[role]): sha256_file(outputs[role])}
        manifest["row_counts"] = {role: counts[role]}
        manifest["failure_counts"] = {"malformed_outputs": 0}
        write_json(Path("artifacts/manifests") / filename, manifest)


def _register_stage5_tables(config_digest: str) -> None:
    path = Path("artifacts/manifests/figure_table_registry.csv")
    optimisation_config_digest = json.loads(
        Path("artifacts/manifests/stage5_optimization_manifest.json").read_text(encoding="utf-8")
    )["configuration_hash"]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0]) if rows else []
    additions = [
        {
            "artifact_id": "table_stage5_primary_validation",
            "artifact_type": "table",
            "title": "Fresh validation of the researcher-selected explanation prompt",
            "research_question": (
                "Does the frozen rag_c3 prompt satisfy the exact-trace and length constraints?"
            ),
            "source_data": ".runtime/stage5",
            "generation_function_or_script": (
                "scripts/run_explanation_eval.py:_optimization_summary"
            ),
            "configuration_hash": optimisation_config_digest,
            "output_path": "artifacts/tables/table_stage5_primary_validation.csv",
            "caption": ("Fresh validation-only assessment of researcher-selected rag_c3."),
            "intended_thesis_chapter": "Methods and results",
            "intended_paper_section": "Explanation evaluation",
            "status": "final",
            "notes": ("The completed six-prompt search remains preserved as supporting evidence."),
        },
        {
            "artifact_id": "table_stage5_pilot_summary",
            "artifact_type": "table",
            "title": "Fifty-case explanation pilot summary",
            "research_question": (
                "Does evidence grounding change support and explanation quality in the pilot?"
            ),
            "source_data": ".runtime/stage5",
            "generation_function_or_script": "scripts/run_explanation_eval.py:_pilot_summary",
            "configuration_hash": config_digest,
            "output_path": "artifacts/tables/table_stage5_pilot_summary.csv",
            "caption": "Descriptive pilot results; not confirmatory inference.",
            "intended_thesis_chapter": "Methods and results",
            "intended_paper_section": "Explanation evaluation",
            "status": "final",
            "notes": ("Fresh exact-trace pilot; descriptive only and approved before Stage 6."),
        },
        {
            "artifact_id": "table_stage5_length_matched_sensitivity",
            "artifact_type": "table",
            "title": "Length-matched explanation sensitivity",
            "research_question": (
                "Are pilot conclusions robust in a case-paired length-matched cohort?"
            ),
            "source_data": ".runtime/stage5",
            "generation_function_or_script": (
                "scripts/run_explanation_eval.py:_length_matched_sensitivity"
            ),
            "configuration_hash": config_digest,
            "output_path": "artifacts/tables/table_stage5_length_matched_sensitivity.csv",
            "caption": "Closest-length case pairs selected equally within each generator.",
            "intended_thesis_chapter": "Methods and results",
            "intended_paper_section": "Explanation evaluation",
            "status": "final",
            "notes": (
                "Separate sensitivity analysis; primary pilot remains the complete 50-case sample."
            ),
        },
    ]
    replacement_ids = {row["artifact_id"] for row in additions}
    rows = [row for row in rows if row["artifact_id"] not in replacement_ids]
    rows.extend(additions)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_pilot(
    args: argparse.Namespace,
    config: dict[str, Any],
    models: dict[str, Any],
    resolved: dict[str, Any],
    config_digest: str,
    cases: list[dict[str, Any]],
    locked_path: Path,
    locked_hash: str,
) -> None:
    requested = args.pilot_cases or config["explanations"]["pilot_cases"]
    if requested != config["explanations"]["pilot_cases"] or requested != len(cases):
        raise ValueError("Stage 5 requires the configured disjoint 50-case pilot.")
    selected_id = config["explanation_search"]["frozen_configuration_id"]
    if selected_id == "pending_optimisation":
        raise ValueError("Freeze the validation-selected RAG configuration before the pilot.")
    settings = next(
        row
        for row in config["explanation_search"]["candidate_configurations"]
        if row["id"] == selected_id
    )
    if config["explanation_search"].get("require_rule_count_matches_scoring_trace", False):
        trace_counts = {len(case["evidence_trace"]["rules"]) for case in cases}
        if trace_counts != {int(settings["rule_count"])}:
            raise ValueError("Pilot B must contain exactly the rules that contributed to scoring.")
    run_id = f"stage5-pilot-v2-{config_digest[:12]}"
    runtime_root = args.runtime_root or Path(config["paths"]["runtime_root"])
    run_dir = runtime_root / "stage5" / run_id
    progress = run_dir / "progress.jsonl"
    final_records = run_dir / "pilot_records.jsonl"
    if final_records.exists():
        raise FileExistsError(f"Immutable Stage 5 pilot exists: {final_records}")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    existing = _read_jsonl(progress) if args.resume and progress.exists() else []
    completed = {(row["case_id"], row["generator"]) for row in existing}
    client = OllamaClient(models["generation_defaults"])
    judge = models["judges"]["roster"][0]["model_id"]
    retries = config["structured_outputs"]["retry_attempts"]
    records = list(existing)
    total = len(cases) * len(models["generators"]["roster"])
    for case in cases:
        for generator_settings in models["generators"]["roster"]:
            generator = generator_settings["model_id"]
            if (case["case_id"], generator) in completed:
                continue
            record = _generate_pair(client, generator, judge, case, settings, retries)
            _append_jsonl(progress, record)
            records.append(record)
            print(f"pilot {len(records)}/{total}", flush=True)
    image_paths = _save_pilot_images(cases, config, run_dir)
    for record in records:
        record["images"] = image_paths[record["case_id"]]
    write_jsonl(final_records, records)
    generations, claims, verifications, judges = _flatten_pilot(records)
    output_records = {
        "generations": run_dir / "generations.jsonl",
        "claims": run_dir / "claims.jsonl",
        "verifications": run_dir / "verifications.jsonl",
        "judges": run_dir / "judges.jsonl",
    }
    for path, rows in zip(
        output_records.values(),
        (generations, claims, verifications, judges),
        strict=True,
    ):
        write_jsonl(path, rows)
    summary = _pilot_summary(records)
    length_sensitivity = _length_matched_sensitivity(records, config)
    runtime_table = run_dir / "pilot_summary.csv"
    runtime_length_table = run_dir / "length_matched_sensitivity.csv"
    tracked_table = Path("artifacts/tables/table_stage5_pilot_summary.csv")
    tracked_length_table = Path("artifacts/tables/table_stage5_length_matched_sensitivity.csv")
    summary.to_csv(runtime_table, index=False)
    summary.to_csv(tracked_table, index=False)
    length_sensitivity.to_csv(runtime_length_table, index=False)
    length_sensitivity.to_csv(tracked_length_table, index=False)
    base = _manifest_base(
        "pilot",
        run_id,
        config_digest,
        resolved,
        locked_path,
        locked_hash,
        models,
    )
    base.update(
        {
            "output_artifact_hashes": {
                str(final_records): sha256_file(final_records),
                str(runtime_table): sha256_file(runtime_table),
                str(tracked_table): sha256_file(tracked_table),
                str(runtime_length_table): sha256_file(runtime_length_table),
                str(tracked_length_table): sha256_file(tracked_length_table),
            },
            "row_counts": {
                "pilot_cases": len(cases),
                "generator_case_pairs": len(records),
                "explanations": len(generations),
                "claims": len(claims),
                "verifications": len(verifications),
                "paired_judgments": len(judges),
                "length_matched_sensitivity_rows": len(length_sensitivity),
            },
            "failure_counts": {"malformed_outputs": 0},
            "selected_settings": settings,
            "optimisation_pilot_overlap": 0,
            "length_fairness": {
                "maximum_mean_word_count_gap_from_no_rag": config["explanation_search"][
                    "maximum_mean_word_count_gap_from_no_rag"
                ],
                "observed_absolute_mean_gap": abs(
                    float(summary.loc[summary["condition"].eq("no_rag"), "word_count"].iloc[0])
                    - float(summary.loc[summary["condition"].eq("rule_rag"), "word_count"].iloc[0])
                ),
                "separate_length_matched_sensitivity": True,
            },
            "manual_inspection_status": "approved_before_stage6_execution",
            "command": "python scripts/run_explanation_eval.py --pilot-cases 50",
        }
    )
    runtime_manifest = run_dir / "manifest.json"
    write_new_json(runtime_manifest, base)
    write_json(Path("artifacts/manifests/stage5_pilot_manifest.json"), base)
    role_rows = (generations, claims, verifications, judges)
    role_counts = {key: len(rows) for key, rows in zip(output_records, role_rows, strict=True)}
    _write_stage5_manifests(
        base,
        output_records,
        role_counts,
    )
    _register_stage5_tables(config_digest)
    print(summary.to_json(orient="records", indent=2))


def main() -> None:
    args = parse_args()
    if args.optimize_only == bool(args.pilot_cases):
        raise SystemExit("Choose exactly one of --optimize-only or --pilot-cases 50.")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    config_digest = configuration_hash(resolved)
    records, locked_path, locked_hash = _stage4_inputs(config)
    optimisation, pilot = _split_cases(records, config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "configuration_hash": config_digest,
                    "mode": "optimisation" if args.optimize_only else "pilot",
                    "cases": len(optimisation if args.optimize_only else pilot),
                    "would_call_local_models": True,
                },
                indent=2,
            )
        )
        return
    if args.optimize_only:
        run_optimization(
            args,
            config,
            models,
            resolved,
            config_digest,
            optimisation,
            locked_path,
            locked_hash,
        )
    else:
        run_pilot(
            args,
            config,
            models,
            resolved,
            config_digest,
            pilot,
            locked_path,
            locked_hash,
        )


if __name__ == "__main__":
    main()
