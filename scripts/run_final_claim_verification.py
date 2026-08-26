"""Run fresh Phi verification over every accepted final Stage-3 claim record."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
import yaml

from evidence_fashion.assessment import citation_occurrences, common_reference_eligibility
from evidence_fashion.explanation import OllamaClient, text_sha256
from evidence_fashion.final_contracts import canonical_json_sha256
from evidence_fashion.manifest import (
    configuration_hash,
    environment_summary,
    git_commit,
    load_resolved_configuration,
    sha256_file,
    utc_timestamp,
    write_new_json,
)
from evidence_fashion.rule_retrieval import full_kb_candidate_retrieval
from evidence_fashion.verification_contracts import validate_verdicts, verification_schema


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def _repair_instruction(error: Exception) -> str:
    return (
        "\n\nReturn only valid JSON with one verdict per supplied claim, in the same order. "
        "Each row must contain exactly claim_id, trace_support, full_kb_support, "
        "common_reference_support, and citation_entailment. Use the frozen labels exactly. "
        "Obey the supplied authoritative common-reference output constraints exactly. "
        f"Validation error: {error}"
    )


def _common_facts(context: dict[str, Any]) -> dict[str, str]:
    return {
        "locked_item_minimal_name": str(context["locked_item_minimal_name"]),
        "query_item_minimal_name": str(context["query_item_minimal_name"]),
        "user_request": str(context["user_request"]),
    }


def _extract_trace_rules(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": str(rule["rule_id"]),
            "rule_text": str(rule["rule_text"]),
            "antecedent_established": bool(rule["antecedent_established"]),
            "antecedent_checks": dict(rule["antecedent_checks"]),
        }
        for rule in trace["rules"]
    ]


def _verify_one(
    *,
    client: OllamaClient,
    model_id: str,
    prompt: str,
    system_prompt: str,
    schema: dict[str, Any],
    claims: list[dict[str, Any]],
    eligible: dict[str, bool],
    valid_citations_present: bool,
    retries: int,
    inference: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    repair = ""
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
                token_limit=int(inference["structured_token_limit"]),
                timeout_seconds=float(inference["timeout_seconds"]),
            )
            raw = result.text
            verified = validate_verdicts(
                json.loads(raw),
                claims=claims,
                common_reference_eligible=eligible,
                valid_citations_present=valid_citations_present,
            )
        except Exception as error:
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
                "raw_response": raw,
                "response_hash": text_sha256(raw),
                "latency_seconds": result.latency_seconds,
                "prompt_eval_count": result.prompt_eval_count,
                "eval_count": result.eval_count,
            }
        )
        return (
            {"status": "accepted", "claims": verified, "attempts_used": attempt_number + 1},
            attempts,
        )
    return (
        {"status": "terminal_failure", "claims": [], "attempts_used": retries + 1},
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

    extraction_manifests = sorted(
        Path(experiment["paths"]["extraction_runs"]).glob("final-extraction-*/manifest.json")
    )
    if len(extraction_manifests) != 1:
        raise ValueError("Exactly one frozen final Stage-3 extraction manifest is required.")
    extraction_manifest_path = extraction_manifests[0]
    extraction_manifest = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
    if extraction_manifest["status"] != "complete":
        raise ValueError("Stage-3 extractions are not frozen.")
    extraction_path = Path(
        next(
            path
            for path in extraction_manifest["output_artifact_hashes"]
            if path.endswith("extractions.jsonl")
        )
    )
    extractions = [row for row in _read_jsonl(extraction_path) if row["status"] == "accepted"]
    if len(extractions) != int(extraction_manifest["row_counts"]["accepted_extractions"]):
        raise ValueError("Accepted extraction count differs from the frozen Stage-3 manifest.")

    recommendation_manifests = sorted(
        Path(experiment["paths"]["recommendation_runs"]).glob("final-recommendations-*/manifest.json")
    )
    if len(recommendation_manifests) != 1:
        raise ValueError("Exactly one frozen final recommendation manifest is required.")
    recommendation_manifest_path = recommendation_manifests[0]
    recommendation_manifest = json.loads(recommendation_manifest_path.read_text(encoding="utf-8"))
    explanation_case_path = Path(
        next(
            path
            for path in recommendation_manifest["output_artifact_hashes"]
            if path.endswith("explanation_cases.jsonl")
        )
    )
    cases = {row["case_id"]: row for row in _read_jsonl(explanation_case_path)}
    if {row["case_id"] for row in extractions}.difference(cases):
        raise ValueError("An extraction record lacks its frozen explanation case.")

    kb_path = Path(experiment["paths"]["knowledge_base"])
    kb = pd.read_csv(kb_path, keep_default_na=False)
    known_rule_ids = set(kb["rule_id"].astype(str))
    run_dir = Path(experiment["paths"]["verification_runs"]) / (
        f"final-verification-{config_hash[:12]}"
    )
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen verification run: {run_dir}")
    run_dir.mkdir(parents=True)
    output_path = run_dir / "verifications.jsonl"
    attempts_path = run_dir / "raw_verification_attempts.jsonl"

    inference = dict(models["inference_defaults"])
    client = OllamaClient(inference, endpoint=str(inference["endpoint"]))
    role = prompts["roles"]["claim_verification"]
    schema = verification_schema()
    accepted = Counter()
    failures = Counter()
    verdict_counts = Counter()
    attempts_used = Counter()
    claims_per_record: list[int] = []
    raw_attempt_count = 0
    with output_path.open("x", encoding="utf-8", newline="\n") as output_handle, attempts_path.open(
        "x", encoding="utf-8", newline="\n"
    ) as attempts_handle:
        try:
            for position, extraction in enumerate(extractions, start=1):
                case = cases[extraction["case_id"]]
                trace = case["exact_stored_rule_trace_B"]
                if canonical_json_sha256(trace) != extraction["trace_hash"]:
                    raise ValueError(
                        "Frozen Stage-3 trace hash does not match the exact case trace."
                    )
                trace_rules = _extract_trace_rules(trace)
                trace_rule_ids = [rule["rule_id"] for rule in trace_rules]
                context = _common_facts(case["common_context_A"])
                full_kb_rules = full_kb_candidate_retrieval(
                    kb,
                    target_category=str(extraction["target_category"]),
                    query_group=str(trace["query_group"]),
                )
                full_kb_rules = [
                    {
                        "rule_id": str(rule["rule_id"]),
                        "rule_text": str(rule["rule_text"]),
                        "applicable_query_categories": str(rule["applicable_query_categories"]),
                        "required_context": str(rule["required_context"]),
                        "query_terms": str(rule["query_terms"]),
                        "candidate_terms": str(rule["candidate_terms"]),
                    }
                    for rule in full_kb_rules
                ]
                occurrences = citation_occurrences(
                    extraction["explanation"],
                    known_rule_ids=known_rule_ids,
                    trace_rule_ids=trace_rule_ids,
                )
                valid_citations = any(item["valid_canonical_occurrence"] for item in occurrences)
                eligible_details = {
                    claim["claim_id"]: common_reference_eligibility(claim, context)
                    for claim in extraction["claims"]
                }
                eligibility = {
                    claim_id: details["eligible"] for claim_id, details in eligible_details.items()
                }
                common_reference_constraints = [
                    {
                        "claim_id": claim_id,
                        "required_common_reference_support": (
                            "supported_or_not_supported" if details["eligible"] else "N/A"
                        ),
                        "reason": details["reason"],
                    }
                    for claim_id, details in eligible_details.items()
                ]
                prompt = str(role["user_template"]).format(
                    full_kb_rules_json=json.dumps(full_kb_rules, ensure_ascii=False),
                    exact_trace_rules_json=json.dumps(trace_rules, ensure_ascii=False),
                    common_reference_facts_json=json.dumps(context, ensure_ascii=False),
                    common_reference_constraints_json=json.dumps(
                        common_reference_constraints, ensure_ascii=False
                    ),
                    citation_occurrences_json=json.dumps(occurrences, ensure_ascii=False),
                    claims_json=json.dumps(extraction["claims"], ensure_ascii=False),
                )
                result, attempts = _verify_one(
                    client=client,
                    model_id=str(models["verifier"]["model_id"]),
                    prompt=prompt,
                    system_prompt=str(role["system_prompt"]),
                    schema=schema,
                    claims=extraction["claims"],
                    eligible=eligibility,
                    valid_citations_present=valid_citations,
                    retries=int(experiment["verification"]["bounded_retry_attempts"]),
                    inference=inference,
                )
                record = {
                    "case_id": extraction["case_id"],
                    "target_category": extraction["target_category"],
                    "generator_model_id": extraction["generator_model_id"],
                    "generator_model_digest": extraction["generator_model_digest"],
                    "condition": extraction["condition"],
                    "locked_candidate_id": extraction["locked_candidate_id"],
                    "explanation_hash": extraction["explanation_hash"],
                    "trace_hash": extraction["trace_hash"],
                    "common_context_A_hash": extraction["common_context_A_hash"],
                    "claims": result["claims"],
                    "input_claim_count": len(extraction["claims"]),
                    "claim_ids": [claim["claim_id"] for claim in extraction["claims"]],
                    "exact_trace_rule_ids": trace_rule_ids,
                    "exact_trace_packet_hash": canonical_json_sha256(trace),
                    "full_kb_candidate_rule_ids": [rule["rule_id"] for rule in full_kb_rules],
                    "full_kb_candidate_packet_hash": canonical_json_sha256(
                        {"rules": full_kb_rules}
                    ),
                    "common_reference_eligibility": eligible_details,
                    "citation_occurrences": occurrences,
                    "valid_citations_present": valid_citations,
                    "verifier_model_id": models["verifier"]["model_id"],
                    "verifier_model_digest": models["verifier"]["immutable_digest"],
                    "verification_prompt_hash": text_sha256(prompt),
                    "verification_system_prompt_hash": text_sha256(str(role["system_prompt"])),
                    "status": result["status"],
                    "attempts_used": result["attempts_used"],
                }
                _append_jsonl(output_handle, record)
                for attempt in attempts:
                    _append_jsonl(
                        attempts_handle,
                        {
                            "case_id": extraction["case_id"],
                            "generator_model_id": extraction["generator_model_id"],
                            "condition": extraction["condition"],
                            **attempt,
                        },
                    )
                    raw_attempt_count += 1
                key = f"{extraction['generator_model_id']}:{extraction['condition']}"
                attempts_used[result["attempts_used"]] += 1
                if result["status"] == "accepted":
                    accepted[key] += 1
                    claims_per_record.append(len(result["claims"]))
                    for claim in result["claims"]:
                        for field in (
                            "trace_support",
                            "full_kb_support",
                            "common_reference_support",
                            "citation_entailment",
                        ):
                            verdict_counts[f"{field}:{claim[field]}"] += 1
                else:
                    failures[key] += 1
                if position % 25 == 0:
                    print(f"claim verification: {position}/{len(extractions)}", flush=True)
        finally:
            client.unload(str(models["verifier"]["model_id"]))
    if sum(accepted.values()) + sum(failures.values()) != len(extractions):
        raise RuntimeError("Every accepted Stage-3 extraction requires one verification record.")
    manifest = {
        "schema_version": 2,
        "stage": 4,
        "stage_name": "fresh_claim_verification",
        "status": "complete",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": config_hash,
        "input_artifact_hashes": {
            str(extraction_manifest_path): sha256_file(extraction_manifest_path),
            str(extraction_path): sha256_file(extraction_path),
            str(recommendation_manifest_path): sha256_file(recommendation_manifest_path),
            str(explanation_case_path): sha256_file(explanation_case_path),
            str(kb_path): sha256_file(kb_path),
        },
        "output_artifact_hashes": {
            str(output_path): sha256_file(output_path),
            str(attempts_path): sha256_file(attempts_path),
        },
        "models": {"verifier": models["verifier"]},
        "prompt_hashes": {
            "system": text_sha256(str(role["system_prompt"])),
            "template": text_sha256(str(role["user_template"])),
        },
        "row_counts": {
            "accepted_stage3_extractions": len(extractions),
            "verification_records": sum(accepted.values()) + sum(failures.values()),
            "accepted_verifications": sum(accepted.values()),
            "terminal_failures": sum(failures.values()),
            "claims_submitted": sum(len(row["claims"]) for row in extractions),
            "claims_verified": sum(claims_per_record),
            "raw_attempts": raw_attempt_count,
        },
        "accepted_by_generator_condition": dict(sorted(accepted.items())),
        "failure_counts_by_generator_condition": dict(sorted(failures.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "claims_per_record": {
            "mean": sum(claims_per_record) / len(claims_per_record) if claims_per_record else None,
            "median": median(claims_per_record) if claims_per_record else None,
            "minimum": min(claims_per_record) if claims_per_record else None,
            "maximum": max(claims_per_record) if claims_per_record else None,
        },
        "attempts_used_distribution": dict(sorted(attempts_used.items())),
        "contract": {
            "exact_stage3_claim_id_preservation": True,
            "exact_stored_trace_only": True,
            "final_200_rule_kb_only": True,
            "deterministic_citation_syntax": True,
            "common_reference_eligibility_deterministic": True,
            "terminal_failures_retained": True,
        },
        "environment": environment_summary(),
    }
    write_new_json(run_dir / "manifest.json", manifest)
    print(json.dumps(manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
