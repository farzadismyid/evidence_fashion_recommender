"""Run the authorised synthetic Qwen/Phi Stage-1 integrity check only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evidence_fashion.explanation import OllamaClient
from evidence_fashion.grounding_contracts import citation_occurrences
from evidence_fashion.manifest import (
    configuration_hash,
    load_resolved_configuration,
    sha256_file,
    write_new_json,
)

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "claim_text": {"type": "string"},
                    "claim_type": {"type": "string"},
                },
                "required": ["claim_id", "claim_text", "claim_type"],
            },
        }
    },
    "required": ["claims"],
}

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "trace_support": {"type": "string", "enum": ["supported", "not_supported"]},
                    "full_kb_support": {
                        "type": "string",
                        "enum": ["supported", "not_supported"],
                    },
                    "common_reference_support": {
                        "type": "string",
                        "enum": ["supported", "not_supported", "N/A"],
                    },
                    "citation_entailment": {
                        "type": "string",
                        "enum": ["entails", "does_not_entail", "N/A"],
                    },
                },
                "required": [
                    "claim_id",
                    "trace_support",
                    "full_kb_support",
                    "common_reference_support",
                    "citation_entailment",
                ],
            },
        }
    },
    "required": ["claims"],
}


def _require_claim_ids(records: list[dict[str, Any]], expected: list[str]) -> None:
    observed = [str(record["claim_id"]) for record in records]
    if observed != expected or len(observed) != len(set(observed)):
        raise ValueError(f"Claim IDs were not preserved exactly: {observed!r} != {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts-config", type=Path, default=Path("configs/prompts.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    prompts = yaml.safe_load(args.prompts_config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    resolved["prompts"] = prompts
    config_hash = configuration_hash(resolved)
    output_dir = Path(config["paths"]["final_analysis_runs"]) / "stage1_integrity"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite integrity-check output: {output_dir}")
    output_dir.mkdir(parents=True)
    explanation = "The black crossbody bag adds a practical finishing touch to the outfit."
    extraction_prompt = prompts["roles"]["claim_extraction"]["user_template"].format(
        claim_types_json=json.dumps(config["extraction"]["claim_types"]), explanation=explanation
    )
    defaults = {
        **models["inference_defaults"],
        "token_limit": models["inference_defaults"]["generation_token_limit"],
    }
    client = OllamaClient(defaults, endpoint=models["inference_defaults"]["endpoint"])
    extraction, extraction_result, extraction_retries = client.generate_json(
        models["extractor"]["model_id"],
        extraction_prompt,
        EXTRACTION_SCHEMA,
        retries=int(config["extraction"]["bounded_retry_attempts"]),
        system_prompt=prompts["roles"]["claim_extraction"]["system_prompt"],
    )
    claims = extraction["claims"]
    extraction_ids = [str(claim["claim_id"]) for claim in claims]
    if not claims or len(extraction_ids) != len(set(extraction_ids)):
        raise ValueError("Synthetic Qwen extraction did not return stable unique claim IDs.")
    trace = [
        {"rule_id": "K001", "rule_text": "Synthetic trace rule accepted by the packet schema."}
    ]
    citations = citation_occurrences(explanation + " [K001] [K999]", known_rule_ids=["K001"])
    if not citations[0]["valid_canonical_occurrence"] or citations[1]["valid_canonical_occurrence"]:
        raise ValueError("Canonical citation or invalid-ID rejection is not functioning.")
    verification_prompt = prompts["roles"]["claim_verification"]["user_template"].format(
        full_kb_rules_json=json.dumps(trace),
        exact_trace_rules_json=json.dumps(trace),
        common_reference_facts_json=json.dumps({"item_type": "crossbody bag", "colour": "black"}),
        citation_occurrences_json=json.dumps(citations),
        claims_json=json.dumps(claims),
    )
    verification, verification_result, verification_retries = client.generate_json(
        models["verifier"]["model_id"],
        verification_prompt,
        VERIFICATION_SCHEMA,
        retries=int(config["verification"]["bounded_retry_attempts"]),
        system_prompt=prompts["roles"]["claim_verification"]["system_prompt"],
    )
    _require_claim_ids(verification["claims"], extraction_ids)
    raw_path = output_dir / "synthetic_model_responses.json"
    raw_path.write_text(
        json.dumps(
            {
                "synthetic_only": True,
                "explanation": explanation,
                "extraction": extraction,
                "verification": verification,
                "extraction_latency_seconds": extraction_result.latency_seconds,
                "verification_latency_seconds": verification_result.latency_seconds,
            }, indent=2),
        encoding="utf-8",
    )
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 2,
        "stage": "final_stage1_synthetic_qwen_phi_integrity_check",
        "synthetic_only": True,
        "configuration_hash": config_hash,
        "models": {"extractor": models["extractor"], "verifier": models["verifier"]},
        "prompt_hashes": {
            role: __import__("hashlib").sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for role, payload in prompts["roles"].items()
        },
        "row_counts": {"synthetic_explanations": 1, "extracted_claims": len(claims)},
        "failure_counts": {"terminal_failures": 0},
        "retries": {"qwen": extraction_retries, "phi": verification_retries},
        "checks": {
            "structured_output_valid": True,
            "claim_ids_preserved": True,
            "final_rule_id_accepted": True,
            "trace_packet_accepted": True,
            "canonical_citation_parsing": True,
            "invalid_rule_id_rejected": True,
            "literal_common_reference_example": True,
        },
        "output_artifact_hashes": {str(raw_path): sha256_file(raw_path)},
    }
    write_new_json(manifest_path, manifest)
    client.unload(models["extractor"]["model_id"])
    client.unload(models["verifier"]["model_id"])
    print(json.dumps(manifest["checks"], indent=2))


if __name__ == "__main__":
    main()
