"""Targeted, key-preserving recovery and deterministic Stage 4 merge."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
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
from .final_judging import _parse_scores, anchored_general_judge_prompt
from .stage45_v2 import KEY_COLUMNS, _generation_evidence, _key
from .study import cached_generate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _canonical_hash(record: dict[str, Any]) -> str:
    return stable_fingerprint(record)


def _local_json_candidates(raw: str) -> list[str]:
    value = raw.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
    start = value.find("{")
    value = value[start:] if start >= 0 else value
    candidates = [value]
    # Only close delimiters when the response ended between tokens, never inside a string.
    in_string = False
    escaped = False
    stack: list[str] = []
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}" and stack:
            stack.pop()
    if not in_string and stack:
        suffix = "".join("]" if item == "[" else "}" for item in reversed(stack))
        closed = value.rstrip().rstrip(",") + suffix
        candidates.append(closed)
    return list(dict.fromkeys(candidates))


def _local_parse(raw: str, parser: Callable[[str], Any]) -> Any | None:
    for candidate in _local_json_candidates(raw):
        try:
            json.loads(candidate, strict=False)
            return parser(candidate)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return None


def _call_with_one_retry(
    *,
    model: Generator,
    prompt: str,
    parser: Callable[[str], Any],
    cache: ArtifactCache,
    namespace: str,
    key: str,
    call_counter: dict[str, int],
) -> tuple[Any | None, str, int]:
    last_response = ""
    for attempt in range(2):
        call_counter["llm_calls"] += 1
        response = cached_generate(
            model,
            prompt,
            cache,
            namespace,
            cache_context={"stage": "stage4d_v2", "key": key, "attempt": attempt},
        )
        last_response = response
        local = _local_parse(response, parser)
        if local is not None:
            return local, response, attempt
        # The sole retry is reserved for incomplete/malformed structured output.
        try:
            parser(response)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        break
    return None, last_response, 1


def _audit_row(
    *, key: str, old: dict[str, Any], new: dict[str, Any], method: str, replaced: bool
) -> dict[str, Any]:
    return {
        "key": key,
        "old_status": "N/A",
        "new_status": "complete" if replaced else "N/A",
        "recovery_method": method,
        "original_hash": _canonical_hash(old),
        "recovered_hash": _canonical_hash(new),
        "replaced": replaced,
    }


def _write_extraction_tables(root: Path, records: list[dict[str, Any]]) -> None:
    _write_jsonl(root / "extraction_checkpoint.jsonl", records)
    rows = []
    for record in records:
        claims = record["claims"] or [{"claim_id": "", "claim": "", "claim_type": ""}]
        for claim in claims:
            rows.append({**{k: v for k, v in record.items() if k != "claims"}, **claim})
    pd.DataFrame(rows).to_csv(root / "claims.csv", index=False)
    pd.DataFrame(records).to_csv(root / "extraction_results.csv", index=False)


def _write_verification_tables(root: Path, records: list[dict[str, Any]]) -> None:
    _write_jsonl(root / "verification_checkpoint.jsonl", records)
    rows = []
    for record in records:
        values = record["verifications"] or [{"claim_id": "", "support_label": pd.NA}]
        for value in values:
            rows.append({**{k: v for k, v in record.items() if k != "verifications"}, **value})
    pd.DataFrame(rows).to_csv(root / "verified_claims.csv", index=False)


def run_stage4d_recovery(
    *,
    artifact_root: Path,
    recovery_root: Path,
    post_root: Path,
    extractor: Generator,
    verifier: Generator,
    judges: dict[str, Generator],
    cache: ArtifactCache,
) -> dict[str, Any]:
    """Recover only explicit failures and materialize key-preserving post-recovery tables."""

    explanations_path = artifact_root / "explanations/explanations.csv"
    explanations = pd.read_csv(explanations_path)
    explanation_by_key = {_key(row): row for _, row in explanations.iterrows()}
    original_extractions = _read_jsonl(
        artifact_root / "claims/extraction/extraction_checkpoint.jsonl"
    )
    original_verifications = _read_jsonl(
        artifact_root / "claims/verification/verification_checkpoint.jsonl"
    )
    original_judgments = _read_jsonl(
        artifact_root / "judging/general/general_judging_checkpoint.jsonl"
    )
    if len(original_extractions) != 3600 or len(original_verifications) != 3600:
        raise ValueError("Stage 4A/4B source checkpoints are incomplete.")
    if len(original_judgments) != 10800:
        raise ValueError("Stage 4C source checkpoint is incomplete.")
    baseline = {
        "explanations_hash": file_fingerprint(explanations_path),
        "extraction_checkpoint_hash": file_fingerprint(
            artifact_root / "claims/extraction/extraction_checkpoint.jsonl"
        ),
        "verification_checkpoint_hash": file_fingerprint(
            artifact_root / "claims/verification/verification_checkpoint.jsonl"
        ),
        "judgment_checkpoint_hash": file_fingerprint(
            artifact_root / "judging/general/general_judging_checkpoint.jsonl"
        ),
        "pre_recovery_manifest_hash": file_fingerprint(
            Path("reports/final_eval_v2/pre_recovery/analysis_manifest.json")
        ),
    }
    (recovery_root / "baseline").mkdir(parents=True, exist_ok=True)
    (recovery_root / "baseline/source_hashes.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    calls = {"llm_calls": 0}

    extraction_audit = []
    merged_extractions = []
    for old in original_extractions:
        if not old["claim_extraction_failed"]:
            merged_extractions.append(old)
            continue
        key = old["explanation_key"]
        local = _local_parse(old.get("raw_malformed_response", ""), parse_extracted_claims)
        method = "local_json_repair"
        response = ""
        recovered = local
        if recovered is None:
            row = explanation_by_key[key]
            recovered, response, retry = _call_with_one_retry(
                model=extractor,
                prompt=claim_extraction_prompt(str(row["generated_explanation"])),
                parser=parse_extracted_claims,
                cache=cache,
                namespace="stage4d_claim_extraction_v2",
                key=key,
                call_counter=calls,
            )
            method = f"llm_recovery_retry_{retry}" if recovered is not None else "unresolved"
        new = old
        if recovered:
            new = {
                **old,
                "claims": recovered,
                "claim_extraction_failed": False,
                "raw_extraction_response": response or old.get("raw_malformed_response", ""),
                "extraction_error": "",
                "stage4d_recovered": True,
            }
        merged_extractions.append(new)
        extraction_audit.append(
            _audit_row(key=key, old=old, new=new, method=method, replaced=new is not old)
        )
    extraction_by_key = {row["explanation_key"]: row for row in merged_extractions}

    verification_audit = []
    merged_verifications = []
    for old in original_verifications:
        key = old["explanation_key"]
        extraction = extraction_by_key[key]
        target = old["verification_status"] == "N/A" and not extraction["claim_extraction_failed"]
        if not target:
            merged_verifications.append(old)
            continue
        claim_ids = {claim["claim_id"] for claim in extraction["claims"]}
        def parser(raw: str, ids: set[str] = claim_ids) -> list[dict[str, object]]:
            return parse_claim_verifications(raw, ids)
        local = _local_parse(old.get("raw_malformed_response", ""), parser)
        method = "local_json_repair"
        response = ""
        recovered = local
        if recovered is None:
            row = explanation_by_key[key]
            recovered, response, retry = _call_with_one_retry(
                model=verifier,
                prompt=claim_verification_prompt(
                    extraction["claims"], build_reference_packet(row)
                ),
                parser=parser,
                cache=cache,
                namespace="stage4d_claim_verification_v2",
                key=key,
                call_counter=calls,
            )
            method = f"llm_recovery_retry_{retry}" if recovered is not None else "unresolved"
        new = old
        if recovered:
            new = {
                **old,
                "claim_extraction_failed": False,
                "claim_verification_failed": False,
                "verifications": recovered,
                "verification_status": "complete",
                "raw_verification_response": response or old.get("raw_malformed_response", ""),
                "verification_error": "",
                "stage4d_recovered": True,
            }
        merged_verifications.append(new)
        verification_audit.append(
            _audit_row(key=key, old=old, new=new, method=method, replaced=new is not old)
        )

    judgment_audit = []
    merged_judgments = []
    for old in original_judgments:
        if not old["judging_failed"]:
            merged_judgments.append(old)
            continue
        key = old["judgment_key"]
        parser = _parse_scores
        local = _local_parse(old.get("raw_malformed_response", ""), parser)
        method = "local_json_repair"
        response = ""
        recovered = local
        if recovered is None:
            row = explanation_by_key[stable_fingerprint({c: str(old[c]) for c in KEY_COLUMNS})]
            enriched = row.copy()
            enriched["generation_evidence_text"] = _generation_evidence(row)
            judge = judges[old["judge_model"]]
            recovered, response, retry = _call_with_one_retry(
                model=judge,
                prompt=anchored_general_judge_prompt(enriched),
                parser=parser,
                cache=cache,
                namespace="stage4d_general_judgment_v2",
                key=key,
                call_counter=calls,
            )
            method = f"llm_recovery_retry_{retry}" if recovered is not None else "unresolved"
        new = old
        if recovered:
            new = {
                **old,
                **recovered,
                "judging_failed": False,
                "raw_judge_response": response or old.get("raw_malformed_response", ""),
                "judging_error": "",
                "stage4d_recovered": True,
            }
        merged_judgments.append(new)
        judgment_audit.append(
            _audit_row(key=key, old=old, new=new, method=method, replaced=new is not old)
        )

    audit_root = recovery_root / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(extraction_audit).to_csv(audit_root / "extraction_recovery.csv", index=False)
    pd.DataFrame(verification_audit).to_csv(audit_root / "verification_recovery.csv", index=False)
    pd.DataFrame(judgment_audit).to_csv(audit_root / "judgment_recovery.csv", index=False)

    _write_extraction_tables(post_root / "claims/extraction", merged_extractions)
    _write_verification_tables(post_root / "claims/verification", merged_verifications)
    judge_root = post_root / "judging/general"
    _write_jsonl(judge_root / "general_judging_checkpoint.jsonl", merged_judgments)
    pd.DataFrame(merged_judgments).to_csv(judge_root / "judge_results.csv", index=False)
    for relative in [
        "retrieval/test/test_ranking_results.csv",
        "validation/reranking_tuning/validation_summary.csv",
        "validation/reranking_tuning/selected_weight.json",
        "explanations/explanations.csv",
        "judging/general/length_compliance_summary.csv",
    ]:
        source = artifact_root / relative
        destination = post_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    unchanged_failures = 0
    successful_hash_mismatches = 0
    for originals, merged, success_test in [
        (original_extractions, merged_extractions, lambda row: not row["claim_extraction_failed"]),
        (
            original_verifications,
            merged_verifications,
            lambda row: row["verification_status"] == "complete",
        ),
        (original_judgments, merged_judgments, lambda row: not row["judging_failed"]),
    ]:
        for old, new in zip(originals, merged, strict=True):
            if success_test(old) and _canonical_hash(old) != _canonical_hash(new):
                successful_hash_mismatches += 1
            if not success_test(old) and _canonical_hash(old) == _canonical_hash(new):
                unchanged_failures += 1
    manifest = {
        "stage": "stage4d_targeted_recovery_v2",
        "llm_calls": calls["llm_calls"],
        "strict_call_budget": 1092,
        "extraction_attempted": len(extraction_audit),
        "extraction_recovered": sum(row["replaced"] for row in extraction_audit),
        "verification_attempted": len(verification_audit),
        "verification_recovered": sum(row["replaced"] for row in verification_audit),
        "judgment_attempted": len(judgment_audit),
        "judgment_recovered": sum(row["replaced"] for row in judgment_audit),
        "successful_hash_mismatches": successful_hash_mismatches,
        "unresolved_rows": unchanged_failures,
        "explanations_hash": file_fingerprint(explanations_path),
    }
    if calls["llm_calls"] > 1092 or successful_hash_mismatches:
        raise ValueError(f"Stage 4D integrity failure: {manifest}")
    recovery_root.mkdir(parents=True, exist_ok=True)
    (recovery_root / "completion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
