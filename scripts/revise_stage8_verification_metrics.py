from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evidence_fashion.manifest import git_commit, sha256_file, utc_timestamp, write_json
from evidence_fashion.verification_analysis import (
    aggregate_claims,
    claim_role,
    classify_refusal,
    normalization_reason_codes,
    source_bucket,
    visible_status,
)

ROOT = Path(__file__).parents[1]
DEFAULT_STAGE8_MANIFEST = ROOT / "artifacts/manifests/stage8_assessment_manifest.json"
DEFAULT_STAGE7_MANIFEST = ROOT / "artifacts/manifests/stage7_explanation_generation_manifest.json"
DEFAULT_TABLE = ROOT / "artifacts/tables/table_stage8_grounding_revision.csv"
DEFAULT_MANIFEST = ROOT / "artifacts/manifests/stage8_metric_revision_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically revise Stage 8 verification aggregations; makes no model calls."
        )
    )
    parser.add_argument("--stage8-manifest", type=Path, default=DEFAULT_STAGE8_MANIFEST)
    parser.add_argument("--stage7-manifest", type=Path, default=DEFAULT_STAGE7_MANIFEST)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def locate_output(manifest: dict[str, Any], suffix: str) -> Path:
    matches = [
        (ROOT / raw_path, digest)
        for raw_path, digest in manifest["output_artifact_hashes"].items()
        if raw_path.replace("\\", "/").endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Manifest must bind exactly one {suffix} artifact.")
    path, expected = matches[0]
    if not path.exists() or sha256_file(path) != expected:
        raise ValueError(f"Missing or hash-mismatched input: {path}")
    return path


def paired_difference_rows(
    explanations: list[dict[str, Any]], scope: str, scope_value: str
) -> dict[str, Any] | None:
    selected = [
        row
        for row in explanations
        if scope == "overall"
        or (scope == "generator" and row["generator"] == scope_value)
        or (scope == "category" and row["category"] == scope_value)
    ]
    by_key = {(row["case_id"], row["generator"], row["condition"]): row for row in selected}
    pairs = []
    for case_id, generator, condition in by_key:
        if condition != "no_rag":
            continue
        first = by_key[(case_id, generator, condition)]
        second = by_key.get((case_id, generator, "rule_rag"))
        if second is None:
            continue
        first_claims = [c for c in first["claims"] if c["claim_role"] == "substantive"]
        second_claims = [c for c in second["claims"] if c["claim_role"] == "substantive"]
        if first_claims and second_claims:
            pairs.append((first_claims, second_claims))
    if not pairs:
        return None

    def difference(field: str, value: str) -> float:
        values = []
        for first, second in pairs:
            first_rate = sum(claim[field] == value for claim in first) / len(first)
            second_rate = sum(claim[field] == value for claim in second) / len(second)
            values.append(second_rate - first_rate)
        return sum(values) / len(values)

    result: dict[str, Any] = {
        "analysis_subset": "all_records",
        "scope": scope,
        "scope_value": scope_value,
        "aggregation": "paired_macro_difference",
        "condition": "rule_rag_minus_no_rag",
        "explanations": len(pairs),
        "eligible_explanations": len(pairs),
        "total_claims": "",
        "identity_context_claims": "",
        "substantive_explanatory_claims": "",
    }
    for status in ("supported", "unsupported", "contradicted", "not_verifiable"):
        result[f"visible_{status}_rate"] = difference("visible_status", status)
        result[f"shared_ab_{status}_rate"] = difference("shared_ab_status", status)
    for source in ("a_only", "b_only", "both_a_b", "neither"):
        result[f"source_{source}_claims"] = ""
        result[f"source_{source}_rate"] = difference("source_bucket", source)
    result["posthoc_b_aligned_claims"] = ""
    result["posthoc_b_aligned_rate"] = difference_for_predicate(
        pairs, lambda claim: claim["source_bucket"] in {"b_only", "both_a_b"}
    )
    return result


def difference_for_predicate(
    pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]], predicate: Any
) -> float:
    values = []
    for first, second in pairs:
        values.append(
            sum(predicate(claim) for claim in second) / len(second)
            - sum(predicate(claim) for claim in first) / len(first)
        )
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    stage8_manifest = json.loads(args.stage8_manifest.read_text(encoding="utf-8"))
    stage7_manifest = json.loads(args.stage7_manifest.read_text(encoding="utf-8"))
    extractions_path = locate_output(stage8_manifest, "extractions.jsonl")
    verifications_path = locate_output(stage8_manifest, "verifications.jsonl")
    generations_path = locate_output(stage7_manifest, "generations.jsonl")
    packets_path = locate_output(stage7_manifest, "case_evidence_packets.jsonl")
    extractions = read_jsonl(extractions_path)
    verifications = read_jsonl(verifications_path)
    generations = read_jsonl(generations_path)
    packets = read_jsonl(packets_path)
    generation_by_key = {
        (row["case_id"], row["generator"], row["condition"]): row for row in generations
    }
    extraction_by_key = {
        (row["case_id"], row["generator"], row["condition"]): row for row in extractions
    }
    packet_by_case = {row["case_id"]: row for row in packets}

    explanations: list[dict[str, Any]] = []
    normalization_reasons: Counter[str] = Counter()
    normalized_distribution: Counter[tuple[str, str]] = Counter()
    verifier_outputs = 0
    for verification in verifications:
        key = (verification["case_id"], verification["generator"], verification["condition"])
        extraction = extraction_by_key[key]
        generation = generation_by_key[key]
        extracted_by_id = {claim["claim_id"]: claim for claim in extraction["claims"]}
        claims = []
        for verified in verification["claims"]:
            extracted = extracted_by_id[verified["claim_id"]]
            combined = {
                **verified,
                "claim_text": extracted["claim_text"],
                "claim_type": extracted["claim_type"],
            }
            combined["claim_role"] = claim_role(combined["claim_type"])
            combined["source_bucket"] = source_bucket(combined)
            combined["shared_ab_status"] = combined["support_status"]
            combined["visible_status"] = visible_status(verification["condition"], combined)
            claims.append(combined)
        normalized = bool(verification.get("structural_normalization_applied"))
        if verification.get("raw_response_text"):
            verifier_outputs += 1
            if normalized:
                normalized_distribution[(verification["generator"], verification["condition"])] += 1
                allowed = {
                    rule["rule_id"]
                    for rule in packet_by_case[verification["case_id"]][
                        "B_exact_stored_trace"
                    ]["rules"]
                }
                codes = normalization_reason_codes(
                    verification["raw_response_text"],
                    verification["claims"],
                    allowed_rule_ids=allowed,
                    citation_ids=verification["citation_ids"],
                )
                if not codes:
                    codes = {"unresolved_structural_difference"}
                normalization_reasons.update(codes)
        explanations.append(
            {
                "case_id": verification["case_id"],
                "generator": verification["generator"],
                "condition": verification["condition"],
                "category": generation["target_category"],
                "normalized": normalized,
                "status": verification["status"],
                "claims": claims,
            }
        )

    rows: list[dict[str, Any]] = []
    subset_specs = [
        ("all_records", lambda row: True),
        (
            "no_structural_normalization",
            lambda row: not row["normalized"] and row["status"] == "complete",
        ),
        ("structural_normalization", lambda row: row["normalized"]),
    ]
    for subset_name, subset_predicate in subset_specs:
        subset = [row for row in explanations if subset_predicate(row)]
        scopes: list[tuple[str, str]] = [("overall", "all")]
        scopes.extend(("generator", value) for value in sorted({r["generator"] for r in subset}))
        if subset_name == "all_records":
            scopes.extend(("category", value) for value in sorted({r["category"] for r in subset}))
        for scope, scope_value in scopes:
            scoped = [
                row
                for row in subset
                if scope == "overall"
                or (scope == "generator" and row["generator"] == scope_value)
                or (scope == "category" and row["category"] == scope_value)
            ]
            for condition in ("no_rag", "rule_rag"):
                selected = [row for row in scoped if row["condition"] == condition]
                for aggregation in ("micro_claim", "macro_explanation"):
                    rows.append(
                        {
                            "analysis_subset": subset_name,
                            "scope": scope,
                            "scope_value": scope_value,
                            "aggregation": aggregation,
                            "condition": condition,
                            **aggregate_claims(selected, aggregation),
                        }
                    )
    for scope, scope_value in [
        ("overall", "all"),
        *[("generator", value) for value in sorted({r["generator"] for r in explanations})],
        *[("category", value) for value in sorted({r["category"] for r in explanations})],
    ]:
        row = paired_difference_rows(explanations, scope, scope_value)
        if row:
            rows.append(row)

    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    with args.output_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    stage8_refusals = [row for row in extractions if row["status"] == "not_applicable_refusal"]
    refusal_records = []
    stage7_detected = 0
    for extraction in stage8_refusals:
        key = (extraction["case_id"], extraction["generator"], extraction["condition"])
        generation = generation_by_key[key]
        classification, reason = classify_refusal(generation["output_text"])
        stage7_detected += int(generation["refusal_detected"])
        refusal_records.append(
            {
                "case_id": extraction["case_id"],
                "classification": classification,
                "condition": extraction["condition"],
                "generator": extraction["generator"],
                "reason": reason,
                "stage7_detected": generation["refusal_detected"],
                "stage7_markers": generation["refusal_markers"],
                "stage8_markers": extraction["refusal_markers"],
            }
        )
    normalized_count = sum(normalized_distribution.values())
    manifest = {
        "schema_version": 1,
        "stage": 8,
        "stage_name": "verification_metric_revision",
        "status": "complete_deterministic_reaggregation",
        "timestamp_utc": utc_timestamp(),
        "git_commit_at_derivation": git_commit(),
        "baseline_stage8_commit": "c9ecae4",
        "model_calls": 0,
        "input_artifact_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                args.stage8_manifest,
                extractions_path,
                verifications_path,
                generations_path,
                packets_path,
            )
        },
        "output_artifact_hashes": {
            str(args.output_table.relative_to(ROOT)): sha256_file(args.output_table)
        },
        "claim_role_rule": {
            "identity_context_types": ["item_type"],
            "substantive_types": sorted(
                {
                    claim["claim_type"]
                    for extraction in extractions
                    for claim in extraction["claims"]
                    if claim["claim_type"] != "item_type"
                }
            ),
            "limitation": (
                "The saved schema has no claim-role label. Colour, material, and other mix simple "
                "facts with explanatory reasoning; treating all non-item_type claims as "
                "substantive avoids undocumented lexical or post-hoc semantic classification but "
                "overcounts some "
                "simple facts as substantive."
            ),
        },
        "visible_evidence_rule": {
            "no_rag": "A only; supported only when A is an attributed support source",
            "rule_rag": "A+B; preserved shared-reference status",
            "no_rag_b_only_treatment": "visible-evidence unsupported and post-hoc B alignment",
            "no_rag_both_treatment": "A-grounded and additionally post-hoc B aligned",
        },
        "refusal_audit": {
            "stage7_detected": stage7_detected,
            "stage8_detected": len(stage8_refusals),
            "classification_counts": dict(
                Counter(row["classification"] for row in refusal_records)
            ),
            "reason_counts": dict(Counter(row["reason"] for row in refusal_records)),
            "records": refusal_records,
        },
        "normalization_sensitivity": {
            "verifier_outputs": verifier_outputs,
            "normalized_outputs": normalized_count,
            "normalized_percentage_of_verifier_outputs": normalized_count / verifier_outputs * 100,
            "distribution": {
                f"{generator}|{condition}": count
                for (generator, condition), count in sorted(normalized_distribution.items())
            },
            "reason_record_counts": dict(sorted(normalization_reasons.items())),
        },
        "row_counts": {
            "canonical_table_rows": len(rows),
            "explanations": len(explanations),
            "verified_claims": sum(len(row["claims"]) for row in explanations),
        },
    }
    write_json(args.output_manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
