"""Run Stage 8 extraction, verification, and cross-model general judging."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evidence_fashion.assessment import (
    build_extraction_prompt,
    build_judge_prompt,
    build_verification_prompt,
    cited_rule_ids,
    extraction_schema,
    judge_schema,
    normalize_verification_payload,
    validate_extraction,
    validate_judgment,
    validate_verification,
    verification_schema,
)
from evidence_fashion.explanation import OllamaClient, text_sha256
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

FROZEN_THROUGH_STAGE7 = (
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
    "explanations",
    "explanation_evidence",
    "explanation_search",
    "structured_outputs",
    "stage6",
    "stage7",
)
_RUN_LOCK_HANDLE: Any | None = None


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
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_final(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def reusable_rule_rag_extractions(
    source_manifest_path: Path,
    generations: Sequence[Mapping[str, Any]],
    extractor: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only unchanged Rule-RAG extractions from a completed earlier run."""
    if not source_manifest_path.exists():
        return [], {"status": "source_manifest_missing"}
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    try:
        source_path, source_hash = locate_output(manifest, "extractions.jsonl")
    except (KeyError, FileNotFoundError, ValueError):
        return [], {"status": "source_extractions_unavailable"}
    source_rows = read_jsonl(source_path)
    source_by_key = {
        (row["case_id"], row["generator"], row["condition"]): row
        for row in source_rows
        if row.get("condition") == "rule_rag"
    }
    reusable: list[dict[str, Any]] = []
    for generation in generations:
        if generation["condition"] != "rule_rag":
            continue
        key = (generation["case_id"], generation["generator"], "rule_rag")
        row = source_by_key.get(key)
        if row is None:
            raise ValueError(f"Missing reusable Rule-RAG extraction: {key}")
        if row.get("explanation_sha256") != generation["output_sha256"]:
            raise ValueError(f"Changed Rule-RAG explanation cannot reuse extraction: {key}")
        if row.get("extractor_model_id") != extractor["model_id"] or row.get(
            "extractor_immutable_digest"
        ) != extractor["immutable_digest"]:
            raise ValueError(f"Extractor identity mismatch for reusable row: {key}")
        reusable.append(row)
    if len(reusable) != 1500:
        raise ValueError(f"Expected 1500 reusable Rule-RAG extractions, found {len(reusable)}")
    return reusable, {
        "status": "validated_reuse",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_extractions": str(source_path),
        "source_extractions_sha256": source_hash,
        "reused_rows": len(reusable),
        "validation": "case_generator_condition_output_hash_and_extractor_digest",
    }


def acquire_run_lock(path: Path) -> None:
    """Hold an OS-released exclusive byte lock for the lifetime of a Stage 8 process."""
    global _RUN_LOCK_HANDLE
    handle = path.open("a+b")
    if handle.seek(0, os.SEEK_END) == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError(f"Another Stage 8 process holds {path}.") from error
    _RUN_LOCK_HANDLE = handle

    def release() -> None:
        if not handle.closed:
            handle.close()

    atexit.register(release)


def locate_output(manifest: Mapping[str, Any], suffix: str) -> tuple[Path, str]:
    matches = [
        (Path(path), str(digest))
        for path, digest in manifest["output_artifact_hashes"].items()
        if path.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Source manifest must bind exactly one {suffix} output.")
    path, digest = matches[0]
    if not path.exists() or sha256_file(path) != digest:
        raise ValueError(f"Source {suffix} is missing or hash-mismatched.")
    return path, digest


def validate_frozen_inputs(
    config: Mapping[str, Any], models: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    frozen_config = manifest["resolved_configuration"]["experiment"]
    changed = [
        section
        for section in FROZEN_THROUGH_STAGE7
        if config.get(section) != frozen_config.get(section)
    ]
    if changed:
        raise ValueError(f"Frozen Stage 2-7 configuration changed: {changed}")
    if models != manifest["resolved_configuration"]["models"]:
        raise ValueError("Frozen model configuration changed after Stage 7.")


def ollama_models(endpoint: str) -> dict[str, str]:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/api/tags", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {str(row["name"]): str(row["digest"]) for row in payload["models"]}


def validate_assessment_models(models: Mapping[str, Any], endpoint: str) -> None:
    installed = ollama_models(endpoint)
    configured = [models["extractor"], models["verifier"], *models["judges"]["roster"]]
    mismatches = {
        row["model_id"]: (row["immutable_digest"], installed.get(row["model_id"]))
        for row in configured
        if installed.get(row["model_id"]) != row["immutable_digest"]
    }
    if mismatches:
        raise ValueError(f"Stage 8 model digests do not match Ollama: {mismatches}")


def structured_call(
    client: OllamaClient,
    *,
    model: str,
    prompt: str,
    schema: Mapping[str, Any],
    validator: Callable[[Mapping[str, Any]], Any],
    retries: int,
    defaults: Mapping[str, Any],
) -> tuple[Any | None, Any | None, int, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    for attempt in range(retries + 1):
        suffix = (
            ""
            if attempt == 0
            else (
                "\n\nReturn only a complete JSON object matching the schema and every "
                "stated contract."
            )
        )
        try:
            result = client.generate(
                model,
                prompt + suffix,
                json_format=schema,
                token_limit=int(defaults["structured_token_limit"]) * (2**attempt),
                timeout_seconds=float(defaults["timeout_seconds"]) * (2**attempt),
            )
            payload = json.loads(result.text)
            return validator(payload), result, attempt, errors
        except Exception as error:  # every rejected attempt is retained in the retry audit
            errors.append({"error_type": type(error).__name__, "message": str(error)})
    return None, None, retries, errors


def retry_records(
    path: Path,
    *,
    phase: str,
    key: Mapping[str, str],
    errors: Sequence[Mapping[str, str]],
) -> None:
    for attempt, error in enumerate(errors, start=1):
        append_jsonl(path, {"phase": phase, **key, "attempt": attempt, **error})


def stage8_refusal_markers(
    generation: Mapping[str, Any], settings: Mapping[str, Any]
) -> list[str]:
    lowered = str(generation["output_text"]).lower()
    return [marker for marker in settings["refusal_detection_markers"] if marker in lowered]


def extraction_summary(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    rows = []
    groups = list(frame.groupby(["generator", "condition"], sort=True))
    groups.extend(
        (("all_generators", condition), group)
        for condition, group in frame.groupby("condition", sort=True)
    )
    for key, group in groups:
        generator, condition = key
        rows.append(
            {
                "generator": generator,
                "condition": condition,
                "explanations": len(group),
                "total_claims": group["claim_count"].sum(),
                "mean_claims": group["claim_count"].mean(),
                "median_claims": group["claim_count"].median(),
                "max_claims": group["claim_count"].max(),
                "not_applicable_refusals": group["status"].eq(
                    "not_applicable_refusal"
                ).sum(),
                "not_applicable_failures": group["status"].eq(
                    "not_applicable_failure"
                ).sum(),
                "total_retries": group["retry_count"].sum(),
            }
        )
    return pd.DataFrame(rows)


def verification_summary(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        for claim in record["claims"]:
            rows.append(
                {
                    "generator": record["generator"],
                    "condition": record["condition"],
                    "support_status": claim["support_status"],
                    "multiple_sources": len(claim["support_sources"]) > 1,
                    "rule_supported": "rule_evidence" in claim["support_sources"],
                    "citation_entails_claim": claim["citation_entails_claim"],
                }
            )
    claims = pd.DataFrame(rows)
    records_frame = pd.DataFrame(records)
    output = []
    keys = [
        (generator, condition)
        for generator in sorted(records_frame["generator"].unique())
        for condition in sorted(records_frame["condition"].unique())
    ]
    keys.extend(
        ("all_generators", condition)
        for condition in sorted(records_frame["condition"].unique())
    )
    for generator, condition in keys:
        selected_records = records_frame[records_frame["condition"].eq(condition)]
        selected_claims = claims[claims["condition"].eq(condition)]
        if generator != "all_generators":
            selected_records = selected_records[selected_records["generator"].eq(generator)]
            selected_claims = selected_claims[selected_claims["generator"].eq(generator)]
        total = len(selected_claims)
        cited = selected_claims["citation_entails_claim"].notna().sum() if total else 0
        output.append(
            {
                "generator": generator,
                "condition": condition,
                "explanations": len(selected_records),
                "claims": total,
                "supported_rate": (
                    selected_claims["support_status"].eq("supported").mean() if total else None
                ),
                "unsupported_rate": (
                    selected_claims["support_status"].eq("unsupported").mean() if total else None
                ),
                "contradicted_rate": (
                    selected_claims["support_status"].eq("contradicted").mean() if total else None
                ),
                "not_verifiable_rate": (
                    selected_claims["support_status"].eq("not_verifiable").mean()
                    if total
                    else None
                ),
                "multiple_source_claims": (
                    int(selected_claims["multiple_sources"].sum()) if total else 0
                ),
                "rule_supported_claims": (
                    int(selected_claims["rule_supported"].sum()) if total else 0
                ),
                "citation_entailment_rate": (
                    selected_claims.loc[
                        selected_claims["citation_entails_claim"].notna(),
                        "citation_entails_claim",
                    ].mean()
                    if cited
                    else None
                ),
                "not_applicable_explanations": selected_records["status"].ne("complete").sum(),
                "total_retries": selected_records["retry_count"].sum(),
            }
        )
    return pd.DataFrame(output)


def judge_summary(records: Sequence[Mapping[str, Any]], dimensions: Sequence[str]) -> pd.DataFrame:
    rows = []
    for record in records:
        for condition, judgment in record["judgments"].items():
            rows.append(
                {
                    "generator": record["generator"],
                    "condition": condition,
                    **{dimension: judgment[dimension] for dimension in dimensions},
                    "status": record["status"],
                    "retry_count": record["retry_count"],
                }
            )
    frame = pd.DataFrame(rows)
    output = []
    groups = list(frame.groupby(["generator", "condition"], sort=True))
    groups.extend(
        (("all_generators", condition), group)
        for condition, group in frame.groupby("condition", sort=True)
    )
    for key, group in groups:
        generator, condition = key
        output.append(
            {
                "generator": generator,
                "condition": condition,
                "judgments": len(group),
                **{dimension: group[dimension].mean() for dimension in dimensions},
                "not_applicable_failures": group["status"].ne("complete").sum(),
                "total_retries": group["retry_count"].sum(),
            }
        )
    return pd.DataFrame(output)


def _update_registry(config_digest: str, paths: Mapping[str, Path]) -> None:
    registry = Path("artifacts/manifests/figure_table_registry.csv")
    rows = pd.read_csv(registry, dtype=str).fillna("")
    ids = set(paths)
    rows = rows[~rows["artifact_id"].isin(ids)]
    descriptions = {
        "table_stage8_claim_extraction": (
            "Stage 8 atomic-claim extraction summary",
            "How many complete atomic fashion claims were extracted?",
        ),
        "table_stage8_claim_verification": (
            "Stage 8 multi-source verification summary",
            "How are claims supported under the common A+B evaluation packet?",
        ),
        "table_stage8_judge_summary": (
            "Stage 8 cross-model general-judge summary",
            "How do full-explanation quality scores vary by condition and generator?",
        ),
    }
    additions = []
    for artifact_id, path in paths.items():
        title, question = descriptions[artifact_id]
        additions.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": "table",
                "title": title,
                "research_question": question,
                "source_data": ".runtime/stage8",
                "generation_function_or_script": "scripts/run_stage8_explanation_assessment.py",
                "configuration_hash": config_digest,
                "output_path": str(path),
                "caption": title + ".",
                "intended_thesis_chapter": "Methods and results",
                "intended_paper_section": "Explanation evaluation",
                "status": "final",
                "notes": (
                    "Stage 8 system output; final paired statistics are deterministic "
                    "postprocessing."
                ),
            }
        )
    pd.concat([rows, pd.DataFrame(additions)], ignore_index=True).to_csv(registry, index=False)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    config_digest = configuration_hash(resolved)
    settings = config["stage8"]
    source_manifest_path = Path(settings["source_manifest"])
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    validate_frozen_inputs(config, models, source_manifest)
    generations_path, generations_hash = locate_output(source_manifest, "generations.jsonl")
    packets_path, packets_hash = locate_output(source_manifest, "case_evidence_packets.jsonl")
    generations = read_jsonl(generations_path)
    packets = read_jsonl(packets_path)
    if len(generations) != 3000 or len(packets) != 500:
        raise ValueError("Stage 8 requires the complete frozen Stage 7 corpus.")
    packet_by_id = {row["case_id"]: row for row in packets}
    generation_by_key = {
        (row["case_id"], row["generator"], row["condition"]): row for row in generations
    }
    run_id = f"stage8-assessment-{config_digest[:12]}"
    runtime_root = args.runtime_root or Path(config["paths"]["runtime_root"])
    run_dir = runtime_root / "stage8" / run_id
    extraction_progress = run_dir / "extraction_progress.jsonl"
    verification_progress = run_dir / "verification_progress.jsonl"
    judge_progress = run_dir / "judge_progress.jsonl"
    retry_log = run_dir / "retry_failures.jsonl"
    extractions_path = run_dir / "extractions.jsonl"
    verifications_path = run_dir / "verifications.jsonl"
    judgments_path = run_dir / "judgments.jsonl"
    runtime_manifest_path = run_dir / "manifest.json"
    if args.dry_run:
        refusal_count = sum(bool(stage8_refusal_markers(row, settings)) for row in generations)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "configuration_hash": config_digest,
                    "explanations": len(generations),
                    "planned_extractions": len(generations) - refusal_count,
                    "planned_verifications": len(generations) - refusal_count,
                    "not_applicable_refusals": refusal_count,
                    "planned_paired_judgments": len(generations) // 2,
                    "would_call_local_models": True,
                    "would_run_postprocessing": False,
                },
                indent=2,
            )
        )
        return
    validate_assessment_models(models, args.ollama_endpoint)
    if runtime_manifest_path.exists():
        raise FileExistsError(f"Immutable Stage 8 run already exists: {run_dir}")
    if not args.resume and any(
        path.exists() for path in (extractions_path, verifications_path, judgments_path)
    ):
        raise FileExistsError(f"Provisional Stage 8 outputs exist; use --resume: {run_dir}")
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Partial Stage 8 run exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    acquire_run_lock(run_dir / "active.lock")
    defaults = models["generation_defaults"]
    client = OllamaClient(defaults, endpoint=args.ollama_endpoint)
    retries = int(config["structured_outputs"]["retry_attempts"])
    interval = int(settings["progress_report_interval"])
    claim_types = list(settings["extraction_claim_types"])
    extractor = models["extractor"]
    reuse_info: dict[str, Any] = {"status": "not_requested"}

    reuse_source = settings.get("reuse_rule_rag_extractions_source_manifest")
    if reuse_source:
        reused, reuse_info = reusable_rule_rag_extractions(
            Path(reuse_source), generations, extractor
        )
        if reused and not extraction_progress.exists():
            write_jsonl_final(extraction_progress, reused)
            print(
                f"stage8 extraction reused {len(reused)}/{len(generations)} validated rows",
                flush=True,
            )

    extraction_history = (
        read_jsonl(extraction_progress) if extraction_progress.exists() else []
    )
    extraction_latest = {
        (row["case_id"], row["generator"], row["condition"]): row
        for row in extraction_history
    }
    extraction_keys = {
        key
        for key, row in extraction_latest.items()
        if row["status"] != "not_applicable_failure"
    }
    for generation in generations:
        key = (generation["case_id"], generation["generator"], generation["condition"])
        if key in extraction_keys:
            continue
        base = {
            "case_id": generation["case_id"],
            "generator": generation["generator"],
            "condition": generation["condition"],
            "explanation_sha256": generation["output_sha256"],
            "extractor_model_id": extractor["model_id"],
            "extractor_immutable_digest": extractor["immutable_digest"],
        }
        detected_refusal_markers = stage8_refusal_markers(generation, settings)
        if detected_refusal_markers and settings[
            "skip_refusal_claim_assessment_as_not_applicable"
        ]:
            record = {
                **base,
                "status": "not_applicable_refusal",
                "prompt": None,
                "prompt_sha256": None,
                "claims": [],
                "claim_count": 0,
                "raw_response_text": None,
                "raw_response_sha256": None,
                "latency_seconds": 0.0,
                "retry_count": 0,
                "retry_errors": [],
                "refusal_markers": detected_refusal_markers,
                "structural_normalization_applied": False,
            }
        else:
            prompt = build_extraction_prompt(generation["output_text"], claim_types)
            validated, result, retry_count, errors = structured_call(
                client,
                model=extractor["model_id"],
                prompt=prompt,
                schema=extraction_schema(claim_types),
                validator=lambda payload: validate_extraction(payload, claim_types),
                retries=retries,
                defaults=defaults,
            )
            retry_records(
                retry_log,
                phase="extraction",
                key={"case_id": key[0], "generator": key[1], "condition": key[2]},
                errors=errors,
            )
            record = {
                **base,
                "status": "complete" if validated is not None else "not_applicable_failure",
                "prompt": prompt,
                "prompt_sha256": text_sha256(prompt),
                "claims": validated or [],
                "claim_count": len(validated or []),
                "raw_response_text": result.text if result else None,
                "raw_response_sha256": text_sha256(result.text) if result else None,
                "latency_seconds": result.latency_seconds if result else 0.0,
                "retry_count": retry_count,
                "retry_errors": errors,
                "refusal_markers": detected_refusal_markers,
                "structural_normalization_applied": (
                    canonical_hash(json.loads(result.text)["claims"])
                    != canonical_hash(validated)
                    if result and validated is not None
                    else False
                ),
            }
        append_jsonl(extraction_progress, record)
        extraction_latest[key] = record
        extraction_keys.add(key)
        if len(extraction_keys) % interval == 0:
            print(
                f"stage8 extraction {len(extraction_keys)}/{len(generations)}", flush=True
            )
    if len(extraction_latest) != len(generations):
        raise RuntimeError("Stage 8 extraction matrix is incomplete.")
    extractions = [
        extraction_latest[(row["case_id"], row["generator"], row["condition"])]
        for row in generations
    ]
    client.unload(extractor["model_id"])
    extraction_by_key = {
        (row["case_id"], row["generator"], row["condition"]): row for row in extractions
    }

    verification_history = (
        read_jsonl(verification_progress) if verification_progress.exists() else []
    )
    verification_latest = {
        (row["case_id"], row["generator"], row["condition"]): row
        for row in verification_history
    }
    verifier = models["verifier"]
    for key, record in list(verification_latest.items()):
        extraction = extraction_by_key[key]
        if (
            record["status"] != "not_applicable_failure"
            or extraction["status"] != "complete"
            or not record.get("raw_response_text")
        ):
            continue
        packet = packet_by_id[key[0]]
        allowed_rules = {
            rule["rule_id"] for rule in packet["B_exact_stored_trace"]["rules"]
        }
        citation_ids = record["citation_ids"]
        try:
            raw_payload = json.loads(record["raw_response_text"])
            normalized_payload, actions = normalize_verification_payload(
                raw_payload, allowed_rules, citation_ids
            )
            repaired_claims = validate_verification(
                normalized_payload, extraction["claims"], allowed_rules, citation_ids
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        repaired = {
            **record,
            "status": "complete",
            "claims": repaired_claims,
            "structural_normalization_applied": True,
            "normalization_actions": actions,
            "recovered_from_terminal_failure": True,
        }
        append_jsonl(verification_progress, repaired)
        verification_latest[key] = repaired
    verification_keys = {
        key
        for key, row in verification_latest.items()
        if row["status"] != "not_applicable_failure"
    }
    for generation in generations:
        key = (generation["case_id"], generation["generator"], generation["condition"])
        if key in verification_keys:
            continue
        extraction = extraction_by_key[key]
        packet = packet_by_id[generation["case_id"]]
        citation_ids = cited_rule_ids(generation["output_text"], settings["citation_pattern"])
        allowed_rules = {
            rule["rule_id"] for rule in packet["B_exact_stored_trace"]["rules"]
        }
        base = {
            "case_id": generation["case_id"],
            "generator": generation["generator"],
            "condition": generation["condition"],
            "explanation_sha256": generation["output_sha256"],
            "extraction_sha256": canonical_hash(extraction["claims"]),
            "A_sha256": packet["A_sha256"],
            "B_sha256": packet["B_sha256"],
            "evaluation_packet": settings["verification_evidence_packet"],
            "generation_evidence_shown": generation["evidence_shown"],
            "citation_ids": citation_ids,
            "invalid_citation_ids": sorted(set(citation_ids) - allowed_rules),
            "verifier_model_id": verifier["model_id"],
            "verifier_immutable_digest": verifier["immutable_digest"],
        }
        if extraction["status"] != "complete":
            record = {
                **base,
                "status": extraction["status"],
                "prompt": None,
                "prompt_sha256": None,
                "claims": [],
                "raw_response_text": None,
                "raw_response_sha256": None,
                "latency_seconds": 0.0,
                "retry_count": 0,
                "retry_errors": [],
                "structural_normalization_applied": False,
            }
        else:
            prompt = build_verification_prompt(
                explanation=generation["output_text"],
                claims=extraction["claims"],
                packet_a=packet["A_common_context"],
                packet_b=packet["B_exact_stored_trace"],
                evidence_shown=generation["evidence_shown"],
                citation_ids=citation_ids,
            )
            normalization_actions: list[str] = []

            def verification_validator(
                payload: Mapping[str, Any],
                *,
                rules: set[str] = allowed_rules,
                citations: Sequence[str] = citation_ids,
                claims: Sequence[Mapping[str, Any]] = extraction["claims"],
                action_sink: list[str] = normalization_actions,
            ) -> list[dict[str, Any]]:
                normalized, actions = normalize_verification_payload(
                    payload, rules, citations
                )
                action_sink[:] = actions
                return validate_verification(normalized, claims, rules, citations)

            validated, result, retry_count, errors = structured_call(
                client,
                model=verifier["model_id"],
                prompt=prompt,
                schema=verification_schema(),
                validator=verification_validator,
                retries=retries,
                defaults=defaults,
            )
            retry_records(
                retry_log,
                phase="verification",
                key={"case_id": key[0], "generator": key[1], "condition": key[2]},
                errors=errors,
            )
            record = {
                **base,
                "status": "complete" if validated is not None else "not_applicable_failure",
                "prompt": prompt,
                "prompt_sha256": text_sha256(prompt),
                "claims": validated or [],
                "raw_response_text": result.text if result else None,
                "raw_response_sha256": text_sha256(result.text) if result else None,
                "latency_seconds": result.latency_seconds if result else 0.0,
                "retry_count": retry_count,
                "retry_errors": errors,
                "structural_normalization_applied": (
                    canonical_hash(json.loads(result.text)["claims"])
                    != canonical_hash(validated)
                    if result and validated is not None
                    else False
                ),
                "normalization_actions": normalization_actions,
                "recovered_from_terminal_failure": False,
            }
        append_jsonl(verification_progress, record)
        verification_latest[key] = record
        verification_keys.add(key)
        if len(verification_keys) % interval == 0:
            print(
                f"stage8 verification {len(verification_keys)}/{len(generations)}",
                flush=True,
            )
    if len(verification_latest) != len(generations):
        raise RuntimeError("Stage 8 verification matrix is incomplete.")
    verifications = [
        verification_latest[(row["case_id"], row["generator"], row["condition"])]
        for row in generations
    ]
    client.unload(verifier["model_id"])

    judgment_history = read_jsonl(judge_progress) if judge_progress.exists() else []
    judgment_latest = {
        (row["case_id"], row["generator"]): row for row in judgment_history
    }
    judge_keys = set(judgment_latest)
    judge = models["judges"]["roster"][0]
    dimensions = list(settings["judge_dimensions"])
    minimum = int(settings["judge_score_minimum"])
    maximum = int(settings["judge_score_maximum"])
    pairs = sorted({(row["case_id"], row["generator"]) for row in generations})
    for case_id, generator in pairs:
        if (case_id, generator) in judge_keys:
            continue
        condition_rows = {
            condition: generation_by_key[(case_id, generator, condition)]
            for condition in config["explanations"]["conditions"]
        }
        order_hash = hashlib.sha256(
            f"{settings['paired_position_seed']}:{case_id}:{generator}".encode()
        ).hexdigest()
        ordered = (
            ["no_rag", "rule_rag"]
            if int(order_hash, 16) % 2 == 0
            else ["rule_rag", "no_rag"]
        )
        first, second = (condition_rows[name] for name in ordered)
        packet = packet_by_id[case_id]
        prompt = build_judge_prompt(
            packet_a=packet["A_common_context"],
            packet_b=packet["B_exact_stored_trace"],
            first_text=first["output_text"],
            second_text=second["output_text"],
            first_evidence_shown=(
                "common context A only"
                if first["evidence_shown"] == "A"
                else "common context A plus exact rule trace B"
            ),
            second_evidence_shown=(
                "common context A only"
                if second["evidence_shown"] == "A"
                else "common context A plus exact rule trace B"
            ),
            dimensions=dimensions,
        )
        validated, result, retry_count, errors = structured_call(
            client,
            model=judge["model_id"],
            prompt=prompt,
            schema=judge_schema(dimensions, minimum, maximum),
            validator=lambda payload: validate_judgment(
                payload, dimensions, minimum, maximum
            ),
            retries=retries,
            defaults=defaults,
        )
        retry_records(
            retry_log,
            phase="judging",
            key={"case_id": case_id, "generator": generator, "condition": "paired"},
            errors=errors,
        )
        record = {
            "case_id": case_id,
            "generator": generator,
            "judge_model_id": judge["model_id"],
            "judge_immutable_digest": judge["immutable_digest"],
            "position_assignment": {"first": ordered[0], "second": ordered[1]},
            "explanation_sha256": {
                condition: condition_rows[condition]["output_sha256"]
                for condition in condition_rows
            },
            "A_sha256": packet["A_sha256"],
            "B_sha256": packet["B_sha256"],
            "prompt": prompt,
            "prompt_sha256": text_sha256(prompt),
            "status": "complete" if validated is not None else "not_applicable_failure",
            "judgments": (
                {ordered[0]: validated["first"], ordered[1]: validated["second"]}
                if validated is not None
                else {}
            ),
            "raw_response_text": result.text if result else None,
            "raw_response_sha256": text_sha256(result.text) if result else None,
            "latency_seconds": result.latency_seconds if result else 0.0,
            "retry_count": retry_count,
            "retry_errors": errors,
        }
        append_jsonl(judge_progress, record)
        judgment_latest[(case_id, generator)] = record
        judge_keys.add((case_id, generator))
        if len(judge_keys) % interval == 0:
            print(f"stage8 judging {len(judge_keys)}/{len(pairs)}", flush=True)
    if len(judgment_latest) != len(pairs):
        raise RuntimeError("Stage 8 paired-judgment matrix is incomplete.")
    judgments = [judgment_latest[key] for key in pairs]
    client.unload(judge["model_id"])

    write_jsonl_final(extractions_path, extractions)
    write_jsonl_final(verifications_path, verifications)
    write_jsonl_final(judgments_path, judgments)
    extraction_table = extraction_summary(extractions)
    verification_table = verification_summary(verifications)
    complete_judgments = [row for row in judgments if row["status"] == "complete"]
    judge_table = judge_summary(complete_judgments, dimensions)
    runtime_tables = {
        "extraction": run_dir / "claim_extraction_summary.csv",
        "verification": run_dir / "claim_verification_summary.csv",
        "judging": run_dir / "judge_summary.csv",
    }
    tracked_tables = {
        "table_stage8_claim_extraction": Path(
            "artifacts/tables/table_stage8_claim_extraction.csv"
        ),
        "table_stage8_claim_verification": Path(
            "artifacts/tables/table_stage8_claim_verification.csv"
        ),
        "table_stage8_judge_summary": Path(
            "artifacts/tables/table_stage8_judge_summary.csv"
        ),
    }
    for frame, runtime_path, tracked_path in zip(
        (extraction_table, verification_table, judge_table),
        runtime_tables.values(),
        tracked_tables.values(),
        strict=True,
    ):
        frame.to_csv(runtime_path, index=False)
        frame.to_csv(tracked_path, index=False)
    _update_registry(config_digest, tracked_tables)
    registry = Path("artifacts/manifests/figure_table_registry.csv")

    retry_rows = read_jsonl(retry_log) if retry_log.exists() else []
    extraction_failures = sum(row["status"] == "not_applicable_failure" for row in extractions)
    verification_failures = sum(
        row["status"] == "not_applicable_failure" for row in verifications
    )
    judging_failures = sum(row["status"] == "not_applicable_failure" for row in judgments)
    verification_claims = sum(len(row["claims"]) for row in verifications)
    multiple_sources = sum(
        len(claim["support_sources"]) > 1
        for row in verifications
        for claim in row["claims"]
    )
    invalid_citations = sum(len(row["invalid_citation_ids"]) for row in verifications)
    integrity = {
        "generation_keys": len(generation_by_key),
        "extraction_keys": len(extraction_keys),
        "verification_keys": len(verification_keys),
        "paired_judgment_keys": len(judge_keys),
        "claim_id_coverage_complete": all(
            verification["status"] != "complete"
            or [claim["claim_id"] for claim in verification["claims"]]
            == [claim["claim_id"] for claim in extraction_by_key[key]["claims"]]
            for key, verification in {
                (row["case_id"], row["generator"], row["condition"]): row
                for row in verifications
            }.items()
        ),
        "failed_verifications_empty": all(
            row["status"] == "complete" or not row["claims"] for row in verifications
        ),
        "common_union_packet_used": all(
            row["evaluation_packet"] == "common_union_A_plus_B" for row in verifications
        ),
        "generation_visibility_preserved": all(
            row["generation_evidence_shown"]
            == generation_by_key[(row["case_id"], row["generator"], row["condition"])][
                "evidence_shown"
            ]
            for row in verifications
        ),
        "judge_condition_label_leaks": sum(
            "no_rag" in row["prompt"].lower() or "rule_rag" in row["prompt"].lower()
            for row in judgments
        ),
        "cross_model_judgments": all(
            "qwen" not in row["generator"].lower() for row in judgments
        ),
        "cross_model_verification": (
            verifier["immutable_digest"] != extractor["immutable_digest"]
        ),
    }
    if not all(
        (
            integrity["generation_keys"] == 3000,
            integrity["extraction_keys"] == 3000,
            integrity["verification_keys"] == 3000,
            integrity["paired_judgment_keys"] == 1500,
            integrity["claim_id_coverage_complete"],
            integrity["failed_verifications_empty"],
            integrity["common_union_packet_used"],
            integrity["generation_visibility_preserved"],
            integrity["judge_condition_label_leaks"] == 0,
            integrity["cross_model_judgments"],
            integrity["cross_model_verification"],
        )
    ):
        raise ValueError(f"Stage 8 integrity checks failed: {integrity}")
    outputs = [
        extractions_path,
        verifications_path,
        judgments_path,
        *runtime_tables.values(),
        *tracked_tables.values(),
        registry,
    ]
    if retry_log.exists():
        outputs.append(retry_log)
    base_manifest = {
        "schema_version": 1,
        "stage": 8,
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": config_digest,
        "resolved_configuration": resolved,
        "input_artifact_hashes": {
            str(source_manifest_path): sha256_file(source_manifest_path),
            str(generations_path): generations_hash,
            str(packets_path): packets_hash,
        },
        "reuse": {"rule_rag_extractions": reuse_info},
        "output_artifact_hashes": {str(path): sha256_file(path) for path in outputs},
        "models": {
            "extractor": extractor,
            "verifier": verifier,
            "judges": models["judges"],
            "generation_defaults": defaults,
        },
        "row_counts": {
            "explanations": len(generations),
            "extractions": len(extractions),
            "extracted_claims": sum(row["claim_count"] for row in extractions),
            "verifications": len(verifications),
            "verified_claims": verification_claims,
            "paired_judgments": len(judgments),
            "scored_explanations": 2 * len(complete_judgments),
            "multiple_source_claims": multiple_sources,
            "invalid_citations": invalid_citations,
        },
        "failure_counts": {
            "extraction_failures": extraction_failures,
            "verification_failures": verification_failures,
            "judging_failures": judging_failures,
            "not_applicable_refusals": sum(
                row["status"] == "not_applicable_refusal" for row in extractions
            ),
            "retry_attempt_failures": len(retry_rows),
            "retried_extractions": sum(row["retry_count"] > 0 for row in extractions),
            "normalized_extractor_outputs": sum(
                row.get("structural_normalization_applied", False) for row in extractions
            ),
            "retried_verifications": sum(row["retry_count"] > 0 for row in verifications),
            "normalized_verifier_outputs": sum(
                row.get("structural_normalization_applied", False) for row in verifications
            ),
            "retried_judgments": sum(row["retry_count"] > 0 for row in judgments),
            "retry_failure_types": dict(Counter(row["error_type"] for row in retry_rows)),
        },
        "integrity_checks": integrity,
        "status": {
            "claim_extraction": "complete",
            "claim_verification": "complete",
            "general_judging": "complete_cross_model",
            "study_specific_statistics": "complete_postprocessing",
            "study_scope": "closed_at_stage8_without_external_or_manual_audit",
        },
        "environment": environment_summary(),
        "inference_server_version": defaults["inference_server_version"],
        "device": defaults["device"],
        "command": (
            "python scripts/run_stage8_explanation_assessment.py "
            "--config configs/experiment.yaml"
        ),
    }
    write_new_json(runtime_manifest_path, base_manifest)
    write_json(Path("artifacts/manifests/stage8_assessment_manifest.json"), base_manifest)
    role_specs = {
        "claim_extraction_manifest.json": (
            "atomic_claim_extraction",
            [
                extractions_path,
                runtime_tables["extraction"],
                tracked_tables["table_stage8_claim_extraction"],
            ],
        ),
        "claim_verification_manifest.json": (
            "multi_source_claim_verification",
            [
                verifications_path,
                runtime_tables["verification"],
                tracked_tables["table_stage8_claim_verification"],
            ],
        ),
        "judge_manifest.json": (
            "cross_model_general_judging",
            [
                judgments_path,
                runtime_tables["judging"],
                tracked_tables["table_stage8_judge_summary"],
            ],
        ),
    }
    for filename, (stage_name, role_outputs) in role_specs.items():
        role_manifest = dict(base_manifest)
        role_manifest["stage_name"] = stage_name
        role_manifest["output_artifact_hashes"] = {
            str(path): sha256_file(path) for path in role_outputs
        }
        write_json(Path("artifacts/manifests") / filename, role_manifest)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "row_counts": base_manifest["row_counts"],
                "failure_counts": base_manifest["failure_counts"],
                "integrity_checks": integrity,
                "status": base_manifest["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
