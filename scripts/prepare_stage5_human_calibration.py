"""Prepare the sealed Stage 5 calibration packet and evaluator outputs before annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evidence_fashion.assessment import (
    citation_occurrences,
    citation_validation_schema,
    separated_entailment_schema,
)
from evidence_fashion.explanation import OllamaClient, text_sha256
from evidence_fashion.kb_audit import declared_values, matches_declared_terms
from evidence_fashion.manifest import sha256_file, utc_timestamp, write_new_json
from evidence_fashion.prompt_registry import (
    load_prompt_registry,
    prompt_manifest_fields,
    render_prompt,
)

ROOT = Path(__file__).parents[1]
STAGE3_MANIFEST = ROOT / "artifacts/manifests/stage3_prompt_freeze_manifest.json"
STAGE4_MANIFEST = ROOT / "artifacts/manifests/stage4_sequential_batch_manifest.json"
CATEGORIES = ("tops", "bottoms", "shoes", "outerwear", "bags")
QUERY_CATEGORY = {
    "tops": "bottoms",
    "bottoms": "tops",
    "shoes": "bottoms",
    "outerwear": "tops",
    "bags": "tops",
}
COVERAGE_TAGS = (
    "compound_claim",
    "explicit_attribute",
    "styling_inference",
    "functional_inference",
    "unsupported_plausible_statement",
    "partial_entailment",
    "contradiction",
    "negation",
    "invalid_citation",
    "bag_example",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts-config", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument("--calibration-config", type=Path, default=Path("configs/calibration.yaml"))
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    return parser.parse_args()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable calibration output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")


def _candidate_rules(rules: pd.DataFrame, target: str, query: str) -> list[dict[str, Any]]:
    filtered = rules[
        rules["recommended_category"].eq(target)
        & rules["audit_status"].eq("retain")
        & rules["applicable_query_categories"].str.split("|").apply(lambda values: query in values)
    ].copy()
    return filtered.sort_values("rule_id").to_dict(orient="records")


def _antecedent_matched_rules(
    candidates: Sequence[Mapping[str, Any]],
    *,
    query: str,
    query_text: str,
    outfit_context_text: str,
    candidate_text: str,
) -> list[dict[str, Any]]:
    observed_context: set[str] = set()
    matched = []
    permitted_query_text = f"{query} | {query_text} | {outfit_context_text}"
    for rule in candidates:
        if query not in declared_values(rule["applicable_query_categories"]):
            continue
        if not matches_declared_terms(rule["required_context"], " | ".join(observed_context)):
            continue
        if not matches_declared_terms(rule["query_terms"], permitted_query_text):
            continue
        if not matches_declared_terms(rule["candidate_terms"], candidate_text):
            continue
        matched.append(dict(rule))
    return matched


def _select_cases(
    items: pd.DataFrame,
    rules: pd.DataFrame,
    request_templates: Mapping[str, str],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    validation = items[items["research_split"].eq("validation")]
    selected: list[dict[str, Any]] = []
    for target in CATEGORIES:
        query = QUERY_CATEGORY[target]
        candidates_for_pair = _candidate_rules(rules, target, query)
        candidates: list[tuple[str, pd.Series, pd.Series, str, list[dict[str, Any]]]] = []
        for outfit_id, outfit in validation.groupby("outfit_id", sort=False):
            targets = outfit[outfit["broad_category"].eq(target)].sort_values("item_id")
            queries = outfit[outfit["broad_category"].eq(query)].sort_values("item_id")
            if not targets.empty and not queries.empty:
                query_item, locked_item = queries.iloc[0], targets.iloc[0]
                outfit_context = " | ".join(
                    f"{row.category}: {row.text}" for row in outfit.itertuples(index=False)
                )
                matched = _antecedent_matched_rules(
                    candidates_for_pair,
                    query=query,
                    query_text=str(query_item["text"]),
                    outfit_context_text=outfit_context,
                    candidate_text=f"{locked_item['category']} | {locked_item['text']}",
                )
                if matched:
                    candidates.append(
                        (str(outfit_id), query_item, locked_item, outfit_context, matched)
                    )
        ranked = sorted(
            candidates,
            key=lambda row: hashlib.sha256(f"{seed}:{target}:{row[0]}".encode()).hexdigest(),
        )
        if len(ranked) < 2:
            raise ValueError(
                f"Insufficient validation outfits for {query} to {target} calibration."
            )
        for index, (outfit_id, query_item, locked_item, outfit_context, matched) in enumerate(
            ranked[:2], start=1
        ):
            calibration_case_id = f"s5-{target}-{index}-{outfit_id}"
            trace = sorted(
                matched,
                key=lambda rule: hashlib.sha256(
                    f"{seed}:{calibration_case_id}:{rule['rule_id']}".encode()
                ).hexdigest(),
            )[:5]
            tags = [COVERAGE_TAGS[len(selected)]]
            if target == "bags" and "bag_example" not in tags:
                tags.append("bag_example")
            selected.append(
                {
                    "calibration_case_id": calibration_case_id,
                    "source_split": "validation",
                    "source_outfit_id": outfit_id,
                    "target_category": target,
                    "query_category": query,
                    "coverage_tags": tags,
                    "common_reference_item_facts": {
                        "user_request": request_templates[target],
                        "query_item_id": str(query_item["item_id"]),
                        "query_item_category": str(query_item["category"]),
                        "query_item_minimal_name": str(query_item["text"]),
                        "locked_item_id": str(locked_item["item_id"]),
                        "locked_item_category": str(locked_item["category"]),
                        "locked_item_minimal_name": str(locked_item["text"]),
                        "outfit_context_text": outfit_context,
                    },
                    "full_kb_candidate_rules": [
                        {"rule_id": rule["rule_id"], "rule_text": rule["rule_text"]}
                        for rule in candidates_for_pair
                    ],
                    "exact_trace_rules": [
                        {"rule_id": rule["rule_id"], "rule_text": rule["rule_text"]}
                        for rule in trace
                    ],
                }
            )
    if len(selected) != 10 or set(tag for row in selected for tag in row["coverage_tags"]) != set(
        COVERAGE_TAGS
    ):
        raise ValueError(
            "Calibration selection does not meet the required 10-case coverage design."
        )
    return selected


def _render_explanation(
    client: OllamaClient,
    registry: Mapping[str, Any],
    generator: Mapping[str, Any],
    case: Mapping[str, Any],
    condition: str,
) -> dict[str, Any]:
    context = case["common_reference_item_facts"]
    role = "no_rag_explanation" if condition == "no_rag" else "rule_rag_explanation"
    variables: dict[str, Any] = {
        "user_request": context["user_request"],
        "query_item_minimal_name": context["query_item_minimal_name"],
        "locked_item_minimal_name": context["locked_item_minimal_name"],
    }
    if role == "rule_rag_explanation":
        variables["rule_evidence"] = "\n".join(
            f"[{rule['rule_id']}] {rule['rule_text']}" for rule in case["exact_trace_rules"]
        )
    rendered = render_prompt(registry, role, variables)
    result = client.generate(
        generator["model_id"],
        rendered["user_prompt"],
        system_prompt=rendered["system_prompt"],
        token_limit=int(registry["roles"][role]["token_limit"]),
    )
    noncanonical_rule_citations = re.findall(r"\[[A-Za-z]\d{3}[^\]]*\]", result.text)
    if any(not citation.startswith("[K") for citation in noncanonical_rule_citations):
        raise ValueError(
            "Generated explanation contains a non-K canonical rule citation; "
            "discard this calibration run and regenerate it."
        )
    return {
        "condition": condition,
        "explanation": result.text,
        "explanation_sha256": text_sha256(result.text),
        "generator_model_id": generator["model_id"],
        "generator_immutable_digest": generator["immutable_digest"],
        "generation": {
            "latency_seconds": result.latency_seconds,
            "prompt_eval_count": result.prompt_eval_count,
            "eval_count": result.eval_count,
        },
        "prompt_provenance": prompt_manifest_fields(rendered),
    }


def _run_qwen_batch(
    client: OllamaClient,
    registry: Mapping[str, Any],
    models: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    claim_types: Sequence[str],
) -> list[dict[str, Any]]:
    from evidence_fashion.assessment import extraction_schema, validate_extraction

    extractor = models["extractor"]
    output = []
    for record in records:
        rendered = render_prompt(
            registry,
            "claim_extraction",
            {
                "claim_types_json": _canonical_json(list(claim_types)),
                "explanation": record["explanation"],
            },
        )
        payload, result, retries = client.generate_json(
            extractor["model_id"],
            rendered["user_prompt"],
            extraction_schema(claim_types),
            retries=int(registry["roles"]["claim_extraction"]["retry"]["max_attempts"]),
            system_prompt=rendered["system_prompt"],
            repair_instruction=registry["roles"]["claim_extraction"]["retry"]["retry_instruction"],
        )
        output.append(
            {
                "calibration_case_id": record["calibration_case_id"],
                "condition": record["condition"],
                "status": "complete",
                "claims": validate_extraction(payload, claim_types, require_claims=False),
                "model_id": extractor["model_id"],
                "immutable_digest": extractor["immutable_digest"],
                "raw_response_sha256": text_sha256(result.text),
                "retry_count": retries,
                "prompt_provenance": prompt_manifest_fields(rendered),
            }
        )
    client.unload(extractor["model_id"])
    return output


def _run_phi_batch(
    client: OllamaClient,
    registry: Mapping[str, Any],
    models: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    extractions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    verifier = models["verifier"]
    output = []
    for record in records:
        key = (record["calibration_case_id"], record["condition"])
        claims = extractions[key]["claims"]
        rendered = render_prompt(
            registry,
            "claim_verification",
            {
                "full_kb_rules_json": _canonical_json(record["full_kb_candidate_rules"]),
                "exact_trace_rules_json": _canonical_json(record["exact_trace_rules"]),
                "common_reference_facts_json": _canonical_json(
                    record["common_reference_item_facts"]
                ),
                "explanation": record["explanation"],
                "claims_json": _canonical_json(claims),
            },
        )
        entailment, result, retries = client.generate_json(
            verifier["model_id"],
            rendered["user_prompt"],
            separated_entailment_schema(),
            retries=int(registry["roles"]["claim_verification"]["retry"]["max_attempts"]),
            system_prompt=rendered["system_prompt"],
            repair_instruction=registry["roles"]["claim_verification"]["retry"][
                "retry_instruction"
            ],
        )
        citations = citation_occurrences(record["explanation"])
        citation_rendered = render_prompt(
            registry,
            "citation_validation",
            {
                "exact_trace_rules_json": _canonical_json(record["exact_trace_rules"]),
                "citation_occurrences_json": _canonical_json(citations),
                "claims_json": _canonical_json(claims),
            },
        )
        citation_payload, citation_result, citation_retries = client.generate_json(
            verifier["model_id"],
            citation_rendered["user_prompt"],
            citation_validation_schema(),
            retries=int(registry["roles"]["citation_validation"]["retry"]["max_attempts"]),
            system_prompt=citation_rendered["system_prompt"],
            repair_instruction=registry["roles"]["citation_validation"]["retry"][
                "retry_instruction"
            ],
        )
        output.append(
            {
                "calibration_case_id": record["calibration_case_id"],
                "condition": record["condition"],
                "status": "complete",
                "entailment": entailment,
                "citation_validation": citation_payload,
                "model_id": verifier["model_id"],
                "immutable_digest": verifier["immutable_digest"],
                "raw_response_sha256": {
                    "entailment": text_sha256(result.text),
                    "citation_validation": text_sha256(citation_result.text),
                },
                "retry_count": {"entailment": retries, "citation_validation": citation_retries},
                "prompt_provenance": {
                    "entailment": prompt_manifest_fields(rendered),
                    "citation_validation": prompt_manifest_fields(citation_rendered),
                },
            }
        )
    client.unload(verifier["model_id"])
    return output


def _write_annotation_view(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Stage 5 human calibration packet", ""]
    for record in records:
        lines.extend(
            [
                f"## {record['calibration_case_id']} — {record['condition']}",
                "",
                f"- Category: {record['target_category']} (query: {record['query_category']})",
                f"- Coverage tags: {', '.join(record['coverage_tags'])}",
                f"- Request: {record['common_reference_item_facts']['user_request']}",
                f"- Query item: {record['common_reference_item_facts']['query_item_minimal_name']}",
                "- Recommended item: "
                f"{record['common_reference_item_facts']['locked_item_minimal_name']}",
                "- Full outfit context: "
                f"{record['common_reference_item_facts']['outfit_context_text']}",
                "",
                "### Explanation",
                "",
                record["explanation"],
                "",
                "### Exact trace (antecedent-matched rules)",
                "",
            ]
        )
        lines.extend(
            f"- `{rule['rule_id']}`: {rule['rule_text']}" for rule in record["exact_trace_rules"]
        )
        lines.extend(["", "### Full-KB candidate rules (assess antecedent applicability)", ""])
        lines.extend(
            f"- `{rule['rule_id']}`: {rule['rule_text']}"
            for rule in record["full_kb_candidate_rules"]
        )
        lines.extend(
            [
                "",
                f"- Observed citations: {', '.join(record['observed_citations']) or 'none'}",
                "- Annotator ID: ____________________",
                "- Completion time (UTC): ____________________",
                "- Human claims / verification / citation validation: complete in JSONL.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    experiment = yaml.safe_load((ROOT / args.experiment_config).read_text(encoding="utf-8"))
    models = yaml.safe_load((ROOT / args.models_config).read_text(encoding="utf-8"))
    calibration = yaml.safe_load((ROOT / args.calibration_config).read_text(encoding="utf-8"))
    registry = load_prompt_registry(ROOT / args.prompts_config)
    for stage, path in ((3, STAGE3_MANIFEST), (4, STAGE4_MANIFEST)):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen":
            raise ValueError(f"Stage {stage} must be frozen before calibration preparation.")
    packet_path = ROOT / calibration["calibration"]["annotation_path"]
    view_path = packet_path.with_name("stage5_annotation_view.md")
    if packet_path.exists() or view_path.exists():
        raise FileExistsError("Refusing to overwrite an existing human calibration packet.")
    data_manifest = json.loads(
        (ROOT / experiment["paths"]["active_data_manifest"]).read_text(encoding="utf-8")
    )
    items_path = ROOT / next(
        path
        for path in data_manifest["output_artifact_hashes"]
        if path.endswith("prepared_items.parquet")
    )
    rules_path = ROOT / experiment["paths"]["knowledge_base"]
    cases = _select_cases(
        pd.read_parquet(items_path),
        pd.read_csv(rules_path),
        experiment["preprocessing"]["category_taxonomy"]["broad_request_templates"],
        seed=42,
    )
    run_id = hashlib.sha256(_canonical_json(cases).encode()).hexdigest()[:12]
    run_dir = ROOT / ".runtime/current/calibration" / f"stage5-{run_id}"
    if run_dir.exists():
        raise FileExistsError(f"Sealed Stage 5 calibration run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    client = OllamaClient(models["generation_defaults"], endpoint=args.ollama_endpoint)
    generator = models["generators"]["roster"][0]
    records = []
    for case in cases:
        for condition in calibration["calibration"]["conditions"]:
            generation = _render_explanation(client, registry, generator, case, condition)
            records.append(
                {
                    **case,
                    **generation,
                    "observed_citations": [
                        entry["raw"] for entry in citation_occurrences(generation["explanation"])
                    ],
                    "annotator_id": "",
                    "completed_at_utc": "",
                    "human_claims": [],
                    "human_verification": [],
                    "human_citation_validation": [],
                }
            )
    client.unload(generator["model_id"])
    qwen = _run_qwen_batch(
        client,
        registry,
        models,
        records,
        experiment["stage8"]["extraction_claim_types"],
    )
    phi = _run_phi_batch(
        client,
        registry,
        models,
        records,
        {(row["calibration_case_id"], row["condition"]): row for row in qwen},
    )
    qwen_path = run_dir / "qwen_claim_extraction_sealed.jsonl"
    phi_path = run_dir / "phi_verification_sealed.jsonl"
    _write_jsonl_new(qwen_path, qwen)
    _write_jsonl_new(phi_path, phi)
    _write_jsonl_new(packet_path, records)
    _write_annotation_view(view_path, records)
    manifest = {
        "schema_version": 1,
        "stage": 5,
        "status": "prepared_pending_human_annotation",
        "timestamp_utc": utc_timestamp(),
        "run_id": run_id,
        "selection": {
            "source_split": "validation",
            "final_explanation_split": "test",
            "paired_cases": len(cases),
            "records": len(records),
            "counts_by_category": {
                category: sum(row["target_category"] == category for row in records)
                for category in CATEGORIES
            },
            "coverage_tags": sorted({tag for row in records for tag in row["coverage_tags"]}),
            "exact_trace_rule_counts": sorted({len(case["exact_trace_rules"]) for case in cases}),
        },
        "sealed_internal_outputs": {
            "qwen_claim_extraction": str(qwen_path.relative_to(ROOT)),
            "phi_verification": str(phi_path.relative_to(ROOT)),
        },
        "human_packet": {
            "canonical_jsonl": str(packet_path.relative_to(ROOT)),
            "annotation_view": str(view_path.relative_to(ROOT)),
            "contains_model_predictions": False,
        },
        "bound_artifact_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                ROOT / args.experiment_config,
                ROOT / args.models_config,
                ROOT / args.prompts_config,
                ROOT / args.calibration_config,
                STAGE3_MANIFEST,
                STAGE4_MANIFEST,
                items_path,
                rules_path,
                Path(__file__),
                qwen_path,
                phi_path,
                packet_path,
                view_path,
            )
        },
        "next_gate": "Await human annotations; do not calculate agreement or freeze Stage 5.",
    }
    provenance_path = run_dir / "provenance.json"
    write_new_json(provenance_path, manifest)
    tracked_manifest = ROOT / "artifacts/manifests/stage5_calibration_packet_manifest.json"
    write_new_json(tracked_manifest, manifest)
    print(
        json.dumps(
            {
                "human_packet": manifest["human_packet"],
                "sealed_internal_outputs": manifest["sealed_internal_outputs"],
                "manifests": [
                    str(provenance_path.relative_to(ROOT)),
                    str(tracked_manifest.relative_to(ROOT)),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
