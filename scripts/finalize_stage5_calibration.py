"""Run the user-authorized final Stage 5 calibration attempt and freeze its record."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from prepare_stage5_human_calibration import _run_phi_batch, _run_qwen_batch, _write_jsonl_new

from evidence_fashion.assessment import (
    CLAIM_STATUSES,
    ENTAILMENT_FIELDS,
    common_reference_eligibility,
)
from evidence_fashion.calibration import (
    calibration_alignment_records,
    calibration_gates,
    calibration_metrics,
    validate_human_calibration,
)
from evidence_fashion.explanation import OllamaClient
from evidence_fashion.manifest import sha256_file, utc_timestamp, write_new_json
from evidence_fashion.prompt_registry import load_prompt_registry

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = Path(__file__).resolve()
STAGE3_MANIFEST = ROOT / "artifacts/manifests/stage3_prompt_freeze_manifest.json"
STAGE4_MANIFEST = ROOT / "artifacts/manifests/stage4_sequential_batch_manifest.json"
DEFAULT_HUMAN_GOLD = Path("data/calibration/stage5_annotations_done_v2.jsonl")
OLLAMA_CLIENT_IMPLEMENTATION = ROOT / "src/evidence_fashion/explanation.py"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-gold", type=Path, default=DEFAULT_HUMAN_GOLD)
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts-config", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument("--calibration-config", type=Path, default=Path("configs/calibration.yaml"))
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    return parser.parse_args()


def _phi_human_gold_metrics(
    human_records: Sequence[Mapping[str, Any]], phi_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score Phi directly against human claim IDs, independent of Qwen alignment."""
    by_key = {(str(row["calibration_case_id"]), str(row["condition"])): row for row in phi_records}
    totals = Counter()
    matches = Counter()
    by_verdict = {
        field: {status: Counter() for status in CLAIM_STATUSES} for field in ENTAILMENT_FIELDS
    }
    binary = {field: Counter() for field in ("full_kb_entailment", "exact_trace_entailment")}
    citation_total = citation_syntax_matches = citation_entailment_total = (
        citation_entailment_matches
    ) = 0
    structured = 0
    retry_records = retry_attempts = 0
    for human in human_records:
        key = (str(human["calibration_case_id"]), str(human["condition"]))
        phi = by_key[key]
        structured += phi.get("status") == "complete"
        retries = phi.get("retry_count", {})
        retry_records += any(int(value) > 0 for value in retries.values())
        retry_attempts += sum(int(value) for value in retries.values())
        claims = {str(row["claim_id"]): row for row in human["human_claims"]}
        expected = {str(row["claim_id"]): row for row in human["human_verification"]}
        observed = {str(row["claim_id"]): row for row in phi["entailment"]["claims"]}
        expected_citations = {
            str(row["claim_id"]): row for row in human["human_citation_validation"]
        }
        observed_citations = {
            str(row["claim_id"]): row for row in phi["citation_validation"]["claims"]
        }
        for claim_id, human_verdict in expected.items():
            model_verdict = observed[claim_id]
            for field in ENTAILMENT_FIELDS:
                if (
                    field == "common_reference_item_fact_support"
                    and not common_reference_eligibility(
                        claims[claim_id], human["common_reference_item_facts"]
                    )["eligible"]
                ):
                    continue
                expected_status = str(human_verdict[field])
                observed_status = str(model_verdict[field])
                totals[field] += 1
                matches[field] += expected_status == observed_status
                by_verdict[field][expected_status]["total"] += 1
                by_verdict[field][expected_status]["matched"] += expected_status == observed_status
                if field in binary:
                    binary[field][
                        (
                            "supported" if expected_status == "supported" else "not_supported",
                            "supported" if observed_status == "supported" else "not_supported",
                        )
                    ] += 1
            citation = expected_citations[claim_id]
            if not citation["citation_present"]:
                continue
            citation_total += 1
            model_citation = observed_citations[claim_id]
            citation_syntax_matches += (
                model_citation.get("citation_present") is True
                and model_citation.get("canonical_citation_format")
                == citation["canonical_citation_format"]
            )
            if citation["canonical_citation_format"]:
                citation_entailment_total += 1
                citation_entailment_matches += (
                    model_citation.get("citation_entails_claim")
                    == citation["citation_entails_claim"]
                )
    return {
        "verifier_full_kb_accuracy": matches["full_kb_entailment"]
        / (totals["full_kb_entailment"] or 1),
        "verifier_exact_trace_accuracy": matches["exact_trace_entailment"]
        / (totals["exact_trace_entailment"] or 1),
        "verifier_common_reference_accuracy": (
            matches["common_reference_item_fact_support"]
            / totals["common_reference_item_fact_support"]
            if totals["common_reference_item_fact_support"]
            else None
        ),
        "verifier_common_reference_eligible_claim_count": totals[
            "common_reference_item_fact_support"
        ],
        "verifier_agreement_by_verdict": {
            field: {
                status: {
                    "total": counts["total"],
                    "matched": counts["matched"],
                    "agreement": counts["matched"] / counts["total"] if counts["total"] else None,
                }
                for status, counts in values.items()
            }
            for field, values in by_verdict.items()
        },
        "verifier_binary_supported_agreement": {
            field: {
                "supported_supported": counts[("supported", "supported")],
                "supported_not_supported": counts[("supported", "not_supported")],
                "not_supported_supported": counts[("not_supported", "supported")],
                "not_supported_not_supported": counts[("not_supported", "not_supported")],
                "agreement": (
                    (
                        counts[("supported", "supported")]
                        + counts[("not_supported", "not_supported")]
                    )
                    / sum(counts.values())
                    if counts
                    else None
                ),
            }
            for field, counts in binary.items()
        },
        "citation_syntax_accuracy": citation_syntax_matches / citation_total
        if citation_total
        else 1.0,
        "citation_entailment_accuracy": (
            citation_entailment_matches / citation_entailment_total
            if citation_entailment_total
            else 1.0
        ),
        "citation_syntax_scored_claim_count": citation_total,
        "citation_entailment_scored_claim_count": citation_entailment_total,
        "structured_output_success_rate": structured / (len(human_records) or 1),
        "retry_record_count": retry_records,
        "retry_attempt_count": retry_attempts,
        "terminal_failure_count": len(human_records) - structured,
    }


def main() -> None:
    args = parse_args()
    paths = {
        "human": ROOT / args.human_gold,
        "experiment": ROOT / args.experiment_config,
        "models": ROOT / args.models_config,
        "prompts": ROOT / args.prompts_config,
        "calibration": ROOT / args.calibration_config,
    }
    human_records = _read_jsonl(paths["human"])
    calibration = yaml.safe_load(paths["calibration"].read_text(encoding="utf-8"))
    human_validation = validate_human_calibration(human_records, calibration)
    experiment = yaml.safe_load(paths["experiment"].read_text(encoding="utf-8"))
    models = yaml.safe_load(paths["models"].read_text(encoding="utf-8"))
    registry = load_prompt_registry(paths["prompts"])
    run_identity = {
        "human_gold_sha256": sha256_file(paths["human"]),
        "prompt_sha256": sha256_file(paths["prompts"]),
        "model_sha256": sha256_file(paths["models"]),
        "implementation_sha256": sha256_file(SCRIPT_PATH),
        "ollama_client_sha256": sha256_file(OLLAMA_CLIENT_IMPLEMENTATION),
        "mode": "final_authorized_stage5_calibration",
    }
    run_id = hashlib.sha256(_canonical_json(run_identity).encode()).hexdigest()[:12]
    run_dir = ROOT / ".runtime/current/calibration" / f"stage5-final-{run_id}"
    if run_dir.exists():
        if any(run_dir.iterdir()):
            raise FileExistsError(f"Final Stage 5 run already exists: {run_dir}")
        run_dir = run_dir.with_name(f"{run_dir.name}-recovered")
    run_dir.mkdir(parents=True)
    client = OllamaClient(models["generation_defaults"], endpoint=args.ollama_endpoint)
    qwen_path = run_dir / "qwen_claim_extraction_sealed.jsonl"
    phi_path = run_dir / "phi_verification_human_gold_sealed.jsonl"
    phi_raw_path = run_dir / "phi_raw_responses_sealed.jsonl"
    qwen_records = _run_qwen_batch(
        client, registry, models, human_records, experiment["stage8"]["extraction_claim_types"]
    )
    _write_jsonl_new(qwen_path, qwen_records)
    human_claim_inputs = {
        (str(row["calibration_case_id"]), str(row["condition"])): {"claims": row["human_claims"]}
        for row in human_records
    }
    phi_records, phi_raw_records = _run_phi_batch(
        client, registry, models, human_records, human_claim_inputs
    )
    _write_jsonl_new(phi_path, phi_records)
    _write_jsonl_new(phi_raw_path, phi_raw_records)
    qwen_model_rows = [
        {
            "calibration_case_id": row["calibration_case_id"],
            "condition": row["condition"],
            "status": row["status"],
            "claims": row["claims"],
            "entailment": {"claims": []},
            "citation_validation": {"claims": []},
        }
        for row in qwen_records
    ]
    alignments = calibration_alignment_records(human_records, qwen_model_rows)
    alignment_path = run_dir / "human_qwen_alignment_final.json"
    write_new_json(
        alignment_path,
        {
            "schema_version": 1,
            "alignment_method": "exact_text_then_entity_polarity_semantic_one_to_one",
            "records": alignments,
        },
    )
    qwen_metrics = calibration_metrics(human_records, qwen_model_rows, alignments=alignments)
    phi_metrics = _phi_human_gold_metrics(human_records, phi_records)
    gates = calibration_gates(
        qwen_metrics
        | {
            "verifier_full_kb_accuracy": phi_metrics["verifier_full_kb_accuracy"],
            "verifier_exact_trace_accuracy": phi_metrics["verifier_exact_trace_accuracy"],
            "verifier_common_reference_accuracy": phi_metrics["verifier_common_reference_accuracy"],
            "citation_validity_accuracy": phi_metrics["citation_entailment_accuracy"],
            "structured_output_success_rate": min(
                qwen_metrics["structured_output_success_rate"],
                phi_metrics["structured_output_success_rate"],
            ),
        },
        calibration,
    )
    manifest = {
        "schema_version": 2,
        "stage": 5,
        "status": "frozen",
        "freeze_basis": "user_authorized_final_attempt_regardless_of_predefined_gate_outcome",
        "predefined_gates_passed": gates["stage5_pass"],
        "timestamp_utc": utc_timestamp(),
        "human_gold": str(paths["human"].relative_to(ROOT)),
        "human_annotation_validation": human_validation,
        "qwen_calibration": qwen_metrics,
        "phi_human_gold_calibration": phi_metrics,
        "predefined_gates": gates,
        "limitations": [
            "Calibration remains disjoint from the final 500-case test set.",
            (
                "Any unmet predefined calibration gates are retained as limitations of the "
                "locally available models."
            ),
        ],
        "prompt_versions": {
            "registry_sha256": sha256_file(paths["prompts"]),
            "claim_extraction": registry["roles"]["claim_extraction"],
            "claim_verification": registry["roles"]["claim_verification"],
        },
        "sealed_outputs": {
            "qwen_claim_extraction": str(qwen_path.relative_to(ROOT)),
            "phi_verification_human_gold": str(phi_path.relative_to(ROOT)),
            "phi_raw_responses": str(phi_raw_path.relative_to(ROOT)),
            "human_qwen_alignment": str(alignment_path.relative_to(ROOT)),
        },
        "bound_artifact_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                paths["human"],
                paths["experiment"],
                paths["models"],
                paths["prompts"],
                paths["calibration"],
                STAGE3_MANIFEST,
                STAGE4_MANIFEST,
                SCRIPT_PATH,
                OLLAMA_CLIENT_IMPLEMENTATION,
                qwen_path,
                phi_path,
                phi_raw_path,
                alignment_path,
            )
        },
        "next_stage": "Stage 6 canonical runtime destinations",
    }
    write_new_json(run_dir / "provenance.json", manifest)
    write_new_json(ROOT / "artifacts/manifests/stage5_calibration_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
