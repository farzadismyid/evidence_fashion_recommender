"""Resumable orchestration for final_eval_v2 claim and general-quality evaluation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..cache import ArtifactCache, file_fingerprint, stable_fingerprint
from ..models.base import Generator
from .claim_evaluation import (
    build_reference_packet,
    claim_extraction_prompt,
    claim_verification_prompt,
    parse_claim_verifications,
    parse_extracted_claims,
)
from .final_judging import (
    GENERAL_DIMENSIONS,
    _parse_scores,
    anchored_general_judge_prompt,
    is_cross_model_judgment,
    model_family,
    primary_and_sensitivity_summaries,
)
from .study import cached_generate

KEY_COLUMNS = ("paper_case_id", "grounding_variant", "generation_model")


def _key(row: pd.Series | dict[str, Any]) -> str:
    return stable_fingerprint({column: str(row[column]) for column in KEY_COLUMNS})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def _write_progress(
    output_dir: Path, *, stage: str, completed: int, expected: int, status: str
) -> None:
    value = {
        "stage": stage,
        "status": status,
        "completed": completed,
        "expected": expected,
        "remaining": expected - completed,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    temporary = output_dir / "progress.json.tmp"
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "progress.json")


def _write_stopped_handoff(
    report_path: Path, *, title: str, completed: int, expected: int, error: Exception
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# {title}\n\n"
        "Status: stopped cleanly on error.\n\n"
        f"Completed: {completed}/{expected}\n\n"
        f"Remaining: {expected - completed}\n\n"
        f"Error: `{error!r}`\n\n"
        "Completed records remain in the append-only checkpoint and will be skipped on resume.\n",
        encoding="utf-8",
    )


def _prepare_stage(
    *, output_dir: Path, stage: str, input_path: Path, models: list[str], expected: int
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "stage_manifest.json"
    inputs = {
        "stage": stage,
        "input": str(input_path),
        "input_hash": file_fingerprint(input_path),
        "models": models,
        "schema_version": "v2",
        "expected": expected,
    }
    fingerprint = stable_fingerprint(inputs)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_fingerprint") != fingerprint:
            raise ValueError(f"Existing {stage} checkpoint used different inputs or models.")
    else:
        manifest_path.write_text(
            json.dumps({**inputs, "input_fingerprint": fingerprint}, indent=2),
            encoding="utf-8",
        )
    return manifest_path


def validate_explanations(explanations: pd.DataFrame) -> None:
    required = {*KEY_COLUMNS, "generated_explanation", "generation_protocol"}
    missing = required - set(explanations.columns)
    if missing:
        raise ValueError(f"Stage 4/5 explanations are missing: {sorted(missing)}")
    if len(explanations) != 3600:
        raise ValueError(f"Stage 4/5 requires all 3,600 explanations; found {len(explanations)}.")
    if set(explanations["generation_protocol"].astype(str)) != {"final_eval_v2"}:
        raise ValueError("Stage 4/5 requires immutable final_eval_v2 generations.")
    if explanations.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Stage 4/5 explanation identities must be unique.")


def length_compliance(explanations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = explanations[list(KEY_COLUMNS)].copy()
    rows["word_count"] = (
        explanations["generated_explanation"].astype(str).map(lambda value: len(value.split()))
    )
    rows["over_35_words"] = rows["word_count"] > 35
    summary = rows.groupby(["generation_model", "grounding_variant"], as_index=False).agg(
        explanations=("word_count", "size"),
        mean_words=("word_count", "mean"),
        over_35_count=("over_35_words", "sum"),
        length_compliance_rate=("over_35_words", lambda values: 1.0 - values.mean()),
    )
    return rows, summary


def _claim_repair_prompt(raw_response: str) -> str:
    return f"""Repair only the JSON syntax of the response below.
Do not add, remove, merge, split, rewrite, or reassess any claim. Preserve every claim and claim
type exactly. Return one JSON object only with the schema
{{"claims":[{{"claim_id":"C1","claim":"...","claim_type":"other"}}]}}.

Malformed response:
{raw_response}"""


def _extract_with_recovery(
    *,
    extractor: Generator,
    prompt: str,
    cache: ArtifactCache,
    context: dict[str, object],
    explanation_key: str,
) -> dict[str, Any]:
    responses: list[str] = []
    errors: list[str] = []
    for attempt in range(3):
        attempt_context = {
            **context,
            "malformed_json_retry_policy": "two_retries_then_repair_v2",
            "explanation_key": explanation_key,
            "attempt": attempt,
        }
        response = cached_generate(
            extractor,
            prompt,
            cache,
            "final_eval_claim_extraction_v2",
            cache_context=attempt_context,
        )
        responses.append(response)
        try:
            return {
                "claims": parse_extracted_claims(response),
                "raw_extraction_response": response,
                "raw_malformed_response": responses[0] if errors else "",
                "retry_count": attempt,
                "repair_attempt_status": "not_needed",
                "repaired_json_response": False,
                "extraction_error": "",
            }
        except json.JSONDecodeError as error:
            errors.append(repr(error))
    malformed = responses[-1]
    repair_prompt = _claim_repair_prompt(malformed)
    repair_response = cached_generate(
        extractor,
        repair_prompt,
        cache,
        "final_eval_claim_extraction_repair_v2",
        cache_context={
            **context,
            "malformed_json_retry_policy": "two_retries_then_repair_v2",
            "explanation_key": explanation_key,
            "repair_attempt": 1,
        },
    )
    try:
        claims = parse_extracted_claims(repair_response)
        return {
            "claims": claims,
            "raw_extraction_response": repair_response,
            "raw_malformed_response": malformed,
            "retry_count": 2,
            "repair_attempt_status": "succeeded",
            "repaired_json_response": True,
            "extraction_error": errors[-1],
        }
    except json.JSONDecodeError as repair_error:
        return {
            "claims": [],
            "raw_extraction_response": repair_response,
            "raw_malformed_response": malformed,
            "retry_count": 2,
            "repair_attempt_status": "failed",
            "repaired_json_response": False,
            "extraction_error": repr(repair_error),
        }


def _verification_repair_prompt(raw_response: str) -> str:
    return f"""Repair only the JSON syntax of the response below.
Do not add, remove, rewrite, or reassess any verification. Preserve every claim ID, support label,
supporting rule ID, entailment value, and reason exactly. Return one JSON object only with the
schema
{{"verifications":[{{"claim_id":"C1","support_label":"unsupported","
supporting_rule_ids":[],"citation_entails_claim":null,"brief_reason":"..."}}]}}.

Malformed response:
{raw_response}"""


def _verify_with_recovery(
    *,
    verifier: Generator,
    prompt: str,
    claim_ids: set[str],
    cache: ArtifactCache,
    context: dict[str, object],
    explanation_key: str,
) -> dict[str, Any]:
    responses: list[str] = []
    errors: list[str] = []
    last_error: json.JSONDecodeError | ValueError | None = None
    for attempt in range(3):
        response = cached_generate(
            verifier,
            prompt,
            cache,
            "final_eval_claim_verification_v2",
            cache_context={
                **context,
                "malformed_json_retry_policy": "two_retries_then_repair_v2",
                "explanation_key": explanation_key,
                "attempt": attempt,
            },
        )
        responses.append(response)
        try:
            return {
                "verifications": parse_claim_verifications(response, claim_ids),
                "raw_verification_response": response,
                "raw_malformed_response": responses[0] if errors else "",
                "retry_count": attempt,
                "repair_attempt_status": "not_needed",
                "repaired_json_response": False,
                "verification_error": "",
            }
        except (json.JSONDecodeError, ValueError) as error:
            errors.append(repr(error))
            last_error = error
    malformed = responses[-1]
    if not isinstance(last_error, json.JSONDecodeError):
        return {
            "verifications": [],
            "raw_verification_response": malformed,
            "raw_malformed_response": "",
            "retry_count": 2,
            "repair_attempt_status": "not_applicable",
            "repaired_json_response": False,
            "verification_error": errors[-1],
        }
    repair_response = cached_generate(
        verifier,
        _verification_repair_prompt(malformed),
        cache,
        "final_eval_claim_verification_repair_v2",
        cache_context={
            **context,
            "malformed_json_retry_policy": "two_retries_then_repair_v2",
            "explanation_key": explanation_key,
            "repair_attempt": 1,
        },
    )
    try:
        return {
            "verifications": parse_claim_verifications(repair_response, claim_ids),
            "raw_verification_response": repair_response,
            "raw_malformed_response": malformed,
            "retry_count": 2,
            "repair_attempt_status": "succeeded",
            "repaired_json_response": True,
            "verification_error": errors[-1],
        }
    except (json.JSONDecodeError, ValueError) as repair_error:
        return {
            "verifications": [],
            "raw_verification_response": repair_response,
            "raw_malformed_response": malformed,
            "retry_count": 2,
            "repair_attempt_status": "failed",
            "repaired_json_response": False,
            "verification_error": repr(repair_error),
        }


def _general_judge_repair_prompt(raw_response: str) -> str:
    return f"""Repair only the JSON syntax of the response below.
Do not add, remove, rewrite, or reassess any score or reason. Preserve every score and the reason
exactly. Return one JSON object only with integer keys input_consistency, general_quality, clarity,
specificity, hallucination_risk, evidence_misuse, plus brief_reason.

Malformed response:
{raw_response}"""


def _judge_with_recovery(
    *,
    judge: Generator,
    prompt: str,
    cache: ArtifactCache,
    context: dict[str, object],
    judgment_key: str,
) -> dict[str, Any]:
    responses: list[str] = []
    errors: list[str] = []
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = cached_generate(
                judge,
                prompt,
                cache,
                "final_eval_general_judge_v2",
                cache_context={
                    **context,
                    "row_error_retry_policy": "two_retries_then_syntax_repair_v2",
                    "judgment_key": judgment_key,
                    "attempt": attempt,
                },
            )
            responses.append(response)
            return {
                **_parse_scores(response),
                "raw_judge_response": response,
                "raw_malformed_response": responses[0] if errors else "",
                "retry_count": attempt,
                "repair_attempt_status": "not_needed",
                "repaired_json_response": False,
                "judging_error": "",
            }
        except Exception as error:
            errors.append(repr(error))
            last_error = error
    raw_response = responses[-1] if responses else ""
    if not isinstance(last_error, json.JSONDecodeError):
        return {
            **{dimension: None for dimension in GENERAL_DIMENSIONS},
            "brief_reason": "",
            "raw_judge_response": raw_response,
            "raw_malformed_response": "",
            "retry_count": 2,
            "repair_attempt_status": "not_applicable",
            "repaired_json_response": False,
            "judging_error": errors[-1],
        }
    try:
        repair_response = cached_generate(
            judge,
            _general_judge_repair_prompt(raw_response),
            cache,
            "final_eval_general_judge_repair_v2",
            cache_context={
                **context,
                "row_error_retry_policy": "two_retries_then_syntax_repair_v2",
                "judgment_key": judgment_key,
                "repair_attempt": 1,
            },
        )
        return {
            **_parse_scores(repair_response),
            "raw_judge_response": repair_response,
            "raw_malformed_response": raw_response,
            "retry_count": 2,
            "repair_attempt_status": "succeeded",
            "repaired_json_response": True,
            "judging_error": errors[-1],
        }
    except Exception as repair_error:
        return {
            **{dimension: None for dimension in GENERAL_DIMENSIONS},
            "brief_reason": "",
            "raw_judge_response": "",
            "raw_malformed_response": raw_response,
            "retry_count": 2,
            "repair_attempt_status": "failed",
            "repaired_json_response": False,
            "judging_error": repr(repair_error),
        }


def run_claim_extraction_v2(
    *,
    explanations: pd.DataFrame,
    extractor: Generator,
    cache: ArtifactCache,
    input_path: Path,
    output_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    validate_explanations(explanations)
    expected = len(explanations)
    checkpoint = output_dir / "extraction_checkpoint.jsonl"
    stage_manifest_path = _prepare_stage(
        output_dir=output_dir,
        stage="extract_claims_v2",
        input_path=input_path,
        models=[extractor.model_id],
        expected=expected,
    )
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    stage_manifest["malformed_json_retry_policy"] = {
        "retries": 2,
        "repair_only_attempts": 1,
        "persistent_failure_treatment": "explicit_failure_N/A",
    }
    stage_manifest_path.write_text(json.dumps(stage_manifest, indent=2), encoding="utf-8")
    existing = _read_jsonl(checkpoint)
    completed = {record["explanation_key"] for record in existing}
    _write_progress(
        output_dir,
        stage="extract_claims_v2",
        completed=len(completed),
        expected=expected,
        status="running",
    )
    context = {"input_hash": file_fingerprint(input_path), "claim_schema_version": "v2"}
    for _, explanation in explanations.iterrows():
        explanation_key = _key(explanation)
        if explanation_key in completed:
            continue
        prompt = claim_extraction_prompt(str(explanation["generated_explanation"]))
        try:
            recovered = _extract_with_recovery(
                extractor=extractor,
                prompt=prompt,
                cache=cache,
                context=context,
                explanation_key=explanation_key,
            )
            claims = recovered["claims"]
            record = {
                "explanation_key": explanation_key,
                **{column: explanation[column] for column in KEY_COLUMNS},
                "explanation_hash": stable_fingerprint(str(explanation["generated_explanation"])),
                "extractor_model": extractor.model_id,
                "claims": claims,
                "claim_extraction_failed": len(claims) == 0,
                **{key: value for key, value in recovered.items() if key != "claims"},
            }
            _append_jsonl(checkpoint, record)
            existing.append(record)
            completed.add(explanation_key)
            _write_progress(
                output_dir,
                stage="extract_claims_v2",
                completed=len(completed),
                expected=expected,
                status="running",
            )
        except Exception as error:
            error_record = {
                "explanation_key": explanation_key,
                **{column: explanation[column] for column in KEY_COLUMNS},
                "error": repr(error),
            }
            _append_jsonl(output_dir / "extraction_errors.jsonl", error_record)
            _write_progress(
                output_dir,
                stage="extract_claims_v2",
                completed=len(completed),
                expected=expected,
                status="stopped_on_error",
            )
            _write_stopped_handoff(
                report_path,
                title="Stage 4A Claim Extraction Handoff",
                completed=len(completed),
                expected=expected,
                error=error,
            )
            raise RuntimeError(
                f"Claim extraction stopped cleanly at {len(completed)}/{expected}: {error!r}"
            ) from error
    claim_rows = []
    for record in existing:
        claims = record["claims"] or [{"claim_id": "", "claim": "", "claim_type": ""}]
        for claim in claims:
            claim_rows.append({key: value for key, value in record.items() if key != "claims"})
            claim_rows[-1].update(claim)
    pd.DataFrame(existing).drop(columns=["claims"]).to_csv(
        output_dir / "extraction_results.csv", index=False
    )
    pd.DataFrame(claim_rows).to_csv(output_dir / "claims.csv", index=False)
    failed_rows = []
    for record in existing:
        if not record["claim_extraction_failed"]:
            continue
        failed_rows.append(
            {
                "paper_case_id": record["paper_case_id"],
                "variant": record["grounding_variant"],
                "generator": record["generation_model"],
                "raw_malformed_response": record.get("raw_malformed_response", ""),
                "error_message": record.get("extraction_error", "empty claim extraction"),
                "retry_count": record.get("retry_count", 0),
                "repair_attempt_status": record.get("repair_attempt_status", "not_applicable"),
            }
        )
    pd.DataFrame(
        failed_rows,
        columns=[
            "paper_case_id",
            "variant",
            "generator",
            "raw_malformed_response",
            "error_message",
            "retry_count",
            "repair_attempt_status",
        ],
    ).to_csv(output_dir / "failed_extractions.csv", index=False)
    lengths, length_summary = length_compliance(explanations)
    lengths.to_csv(output_dir / "length_compliance.csv", index=False)
    length_summary.to_csv(output_dir / "length_compliance_summary.csv", index=False)
    failures = sum(bool(record["claim_extraction_failed"]) for record in existing)
    repaired = sum(bool(record.get("repaired_json_response")) for record in existing)
    manifest = {
        "stage": "extract_claims_v2",
        "completed": expected,
        "expected": expected,
        "extraction_failures": failures,
        "repaired_json_responses": repaired,
        "claims": len(pd.DataFrame(claim_rows).query("claim != ''")),
        "input_hash": file_fingerprint(input_path),
        "extractor_model": extractor.model_id,
    }
    (output_dir / "completion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_progress(
        output_dir,
        stage="extract_claims_v2",
        completed=expected,
        expected=expected,
        status="complete",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Stage 4A Claim Extraction Handoff\n\n"
        f"Completed: {expected}/{expected}\n\n"
        f"Atomic claims: {manifest['claims']}\n\n"
        f"Repaired malformed JSON responses: {repaired}\n\n"
        f"Empty extractions treated as failure/N/A: {failures}\n\n"
        "Original Stage 3 explanations were read only and remain unchanged. "
        "Claim verification and general judging have not run.\n",
        encoding="utf-8",
    )
    return manifest


def _generation_evidence(row: pd.Series) -> str:
    variant = str(row["grounding_variant"])
    parts = []
    if variant in {"item_rag", "hybrid_rag"}:
        parts.append(str(row.get("item_evidence_text", "")))
    if variant in {"rule_rag", "hybrid_rag"}:
        parts.append(str(row.get("rule_evidence_text", "")))
    return "\n".join(parts)


def run_claim_verification_v2(
    *,
    explanations: pd.DataFrame,
    extraction_dir: Path,
    verifier: Generator,
    cache: ArtifactCache,
    input_path: Path,
    output_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    validate_explanations(explanations)
    extracted_records = _read_jsonl(extraction_dir / "extraction_checkpoint.jsonl")
    if len(extracted_records) != len(explanations):
        raise ValueError("Claim extraction must be complete before verification.")
    extracted = {record["explanation_key"]: record for record in extracted_records}
    expected = len(explanations)
    checkpoint = output_dir / "verification_checkpoint.jsonl"
    stage_manifest_path = _prepare_stage(
        output_dir=output_dir,
        stage="verify_claims_v2",
        input_path=input_path,
        models=[verifier.model_id],
        expected=expected,
    )
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    stage_manifest["malformed_json_retry_policy"] = {
        "retries": 2,
        "repair_only_attempts": 1,
        "persistent_failure_treatment": "explicit_failure_N/A",
    }
    stage_manifest_path.write_text(json.dumps(stage_manifest, indent=2), encoding="utf-8")
    records = _read_jsonl(checkpoint)
    completed = {record["explanation_key"] for record in records}
    context = {"input_hash": file_fingerprint(input_path), "claim_schema_version": "v2"}
    for _, explanation in explanations.iterrows():
        explanation_key = _key(explanation)
        if explanation_key in completed:
            continue
        extraction = extracted[explanation_key]
        claims = extraction["claims"]
        if extraction["claim_extraction_failed"]:
            record = {
                "explanation_key": explanation_key,
                **{column: explanation[column] for column in KEY_COLUMNS},
                "claim_extraction_failed": True,
                "claim_verification_failed": False,
                "verifications": [],
                "verification_status": "N/A",
            }
        else:
            packet = build_reference_packet(explanation)
            prompt = claim_verification_prompt(claims, packet)
            try:
                recovered = _verify_with_recovery(
                    verifier=verifier,
                    prompt=prompt,
                    claim_ids={claim["claim_id"] for claim in claims},
                    cache=cache,
                    context=context,
                    explanation_key=explanation_key,
                )
                record = {
                    "explanation_key": explanation_key,
                    **{column: explanation[column] for column in KEY_COLUMNS},
                    "claim_extraction_failed": False,
                    "claim_verification_failed": not bool(recovered["verifications"]),
                    "verifier_model": verifier.model_id,
                    "reference_packet_hash": packet.fingerprint,
                    **recovered,
                    "verification_status": (
                        "complete" if recovered["verifications"] else "N/A"
                    ),
                }
            except Exception as error:
                _append_jsonl(
                    output_dir / "verification_errors.jsonl",
                    {"explanation_key": explanation_key, "error": repr(error)},
                )
                _write_progress(
                    output_dir,
                    stage="verify_claims_v2",
                    completed=len(completed),
                    expected=expected,
                    status="stopped_on_error",
                )
                _write_stopped_handoff(
                    report_path,
                    title="Stage 4B Claim Verification Handoff",
                    completed=len(completed),
                    expected=expected,
                    error=error,
                )
                raise RuntimeError(
                    f"Claim verification stopped cleanly at {len(completed)}/{expected}: {error!r}"
                ) from error
        _append_jsonl(checkpoint, record)
        records.append(record)
        completed.add(explanation_key)
        _write_progress(
            output_dir,
            stage="verify_claims_v2",
            completed=len(completed),
            expected=expected,
            status="running",
        )
    rows = []
    for record in records:
        values = record["verifications"] or [{"claim_id": "", "support_label": pd.NA}]
        for value in values:
            row = {key: item for key, item in record.items() if key != "verifications"}
            row.update(value)
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "verified_claims.csv", index=False)
    failed = [record for record in records if record.get("claim_verification_failed")]
    pd.DataFrame(
        [
            {
                "paper_case_id": record["paper_case_id"],
                "variant": record["grounding_variant"],
                "generator": record["generation_model"],
                "raw_malformed_response": record.get("raw_malformed_response", ""),
                "error_message": record.get("verification_error", ""),
                "retry_count": record.get("retry_count", 0),
                "repair_attempt_status": record.get("repair_attempt_status", "not_applicable"),
            }
            for record in failed
        ]
    ).to_csv(output_dir / "failed_verifications.csv", index=False)
    manifest = {
        "stage": "verify_claims_v2",
        "completed": expected,
        "expected": expected,
        "extraction_failure_na_rows": sum(
            bool(record["claim_extraction_failed"]) for record in records
        ),
        "verification_failure_na_rows": len(failed),
        "repaired_json_responses": sum(
            bool(record.get("repaired_json_response")) for record in records
        ),
    }
    (output_dir / "completion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_progress(
        output_dir,
        stage="verify_claims_v2",
        completed=expected,
        expected=expected,
        status="complete",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# Stage 4B Claim Verification Handoff\n\n"
        f"Completed: {expected}/{expected}\n\n"
        f"Extraction failure/N/A rows: {manifest['extraction_failure_na_rows']}\n\n"
        f"Verification failure/N/A rows: {manifest['verification_failure_na_rows']}\n\n"
        f"Repaired malformed JSON responses: {manifest['repaired_json_responses']}\n\n"
        "Original Stage 3 explanations were read only and remain unchanged. "
        "General judging has not run.\n",
        encoding="utf-8",
    )
    return manifest


def run_general_judging_v2(
    *,
    explanations: pd.DataFrame,
    judges: list[Generator],
    cache: ArtifactCache,
    input_path: Path,
    output_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    validate_explanations(explanations)
    expected = len(explanations) * len(judges)
    checkpoint = output_dir / "general_judging_checkpoint.jsonl"
    stage_manifest_path = _prepare_stage(
        output_dir=output_dir,
        stage="judge_general_quality_v2",
        input_path=input_path,
        models=[judge.model_id for judge in judges],
        expected=expected,
    )
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    stage_manifest["row_error_retry_policy"] = {
        "retries": 2,
        "syntax_repair_attempts": 1,
        "persistent_failure_treatment": "explicit_failure_N/A_continue",
    }
    stage_manifest_path.write_text(json.dumps(stage_manifest, indent=2), encoding="utf-8")
    records = _read_jsonl(checkpoint)
    completed = {record["judgment_key"] for record in records}
    context = {"input_hash": file_fingerprint(input_path), "judge_schema_version": "v2"}
    # Keep one local judge model active at a time. Judgment keys make traversal order irrelevant
    # to resume semantics, while judge-major traversal avoids repeated model swaps.
    for judge in judges:
        for _, explanation in explanations.iterrows():
            enriched = explanation.copy()
            enriched["generation_evidence_text"] = _generation_evidence(explanation)
            prompt = anchored_general_judge_prompt(enriched)
            judgment_key = stable_fingerprint(
                {"explanation_key": _key(explanation), "judge_model": judge.model_id}
            )
            if judgment_key in completed:
                continue
            recovered = _judge_with_recovery(
                judge=judge,
                prompt=prompt,
                cache=cache,
                context=context,
                judgment_key=judgment_key,
            )
            record = {
                "judgment_key": judgment_key,
                **{column: explanation[column] for column in KEY_COLUMNS},
                "judge_model": judge.model_id,
                "generation_model_family": model_family(str(explanation["generation_model"])),
                "judge_model_family": model_family(judge.model_id),
                "cross_model_primary_eligible": is_cross_model_judgment(
                    str(explanation["generation_model"]), judge.model_id
                ),
                "judging_failed": recovered["general_quality"] is None,
                **recovered,
            }
            _append_jsonl(checkpoint, record)
            records.append(record)
            completed.add(judgment_key)
            _write_progress(
                output_dir,
                stage="judge_general_quality_v2",
                completed=len(completed),
                expected=expected,
                status="running",
            )
    judged = pd.DataFrame(records)
    judged.to_csv(output_dir / "judge_results.csv", index=False)
    successful = judged[~judged["judging_failed"].astype(bool)].copy()
    primary, sensitivity = primary_and_sensitivity_summaries(successful)
    primary.to_csv(output_dir / "primary_cross_model_summary.csv", index=False)
    sensitivity.to_csv(output_dir / "sensitivity_all_judges_summary.csv", index=False)
    lengths, length_summary = length_compliance(explanations)
    lengths.to_csv(output_dir / "length_compliance.csv", index=False)
    length_summary.to_csv(output_dir / "length_compliance_summary.csv", index=False)
    failures = judged[judged["judging_failed"].astype(bool)].copy()
    failures[
        [
            "paper_case_id",
            "grounding_variant",
            "generation_model",
            "judge_model",
            "cross_model_primary_eligible",
            "raw_malformed_response",
            "judging_error",
            "retry_count",
            "repair_attempt_status",
        ]
    ].to_csv(output_dir / "failed_general_judgments.csv", index=False)
    manifest = {
        "stage": "judge_general_quality_v2",
        "completed": expected,
        "expected": expected,
        "primary": "cross_model_only",
        "sensitivity": "all_judges_including_self",
        "failed_judgments_na": int(len(failures)),
        "repaired_json_responses": int(judged["repaired_json_response"].astype(bool).sum()),
    }
    (output_dir / "completion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_progress(
        output_dir,
        stage="judge_general_quality_v2",
        completed=expected,
        expected=expected,
        status="complete",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Stage 5 General Judging Handoff\n\n"
        f"Completed: {expected}/{expected}\n\n"
        "Primary results: cross-model-only judgments.\n\n"
        "Sensitivity results: all judges, including self-family judgments.\n\n"
        f"Failed judgment/N/A rows: {manifest['failed_judgments_na']}\n\n"
        f"Repaired malformed JSON responses: {manifest['repaired_json_responses']}\n\n"
        "Length compliance is reported separately and explanations were not modified.\n",
        encoding="utf-8",
    )
    return manifest
