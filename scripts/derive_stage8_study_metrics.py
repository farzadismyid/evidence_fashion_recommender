from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from evidence_fashion.evaluation.statistics import (
    clustered_bootstrap_mean,
    two_sided_bootstrap_pvalue,
)
from evidence_fashion.manifest import git_commit, sha256_file, utc_timestamp, write_json
from evidence_fashion.study_metrics import (
    citation_observation_available,
    classify_item_attribute,
    dta_entailed,
    exact_b_entails_attribute,
    explicit_a_entails_attribute,
    requires_rule_support,
    valid_rule_citation,
)
from evidence_fashion.verification_analysis import claim_role, source_bucket, visible_status

ROOT = Path(__file__).parents[1]
STAGE7_MANIFEST = ROOT / "artifacts/manifests/stage7_explanation_generation_manifest.json"
STAGE8_MANIFEST = ROOT / "artifacts/manifests/stage8_assessment_manifest.json"
STAGE8_REVISION_MANIFEST = ROOT / "artifacts/manifests/stage8_metric_revision_manifest.json"
OUTPUT_TABLE = ROOT / "artifacts/tables/table_stage8_study_specific_metrics.csv"
OUTPUT_MANIFEST = ROOT / "artifacts/manifests/stage8_study_metrics_manifest.json"
BOOTSTRAP_REPLICATES = 5_000
CONFIDENCE_LEVEL = 0.95
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive additive study-specific Stage 8 metrics with zero model calls."
    )
    parser.add_argument("--output-table", type=Path, default=OUTPUT_TABLE)
    parser.add_argument("--output-manifest", type=Path, default=OUTPUT_MANIFEST)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def locate_output(manifest: Mapping[str, Any], suffix: str) -> Path:
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


def mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def rate(
    claims: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]
) -> float | None:
    return sum(predicate(claim) for claim in claims) / len(claims) if claims else None


def select_length_matched(explanations: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """Apply the frozen 10-pairs-per-generator closest-length method to Stage 7."""
    by_pair: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in explanations:
        by_pair[(row["case_id"], row["generator"])][row["condition"]] = row
    selected: set[tuple[str, str]] = set()
    generators = sorted({row["generator"] for row in explanations})
    for generator in generators:
        candidates = []
        for (case_id, pair_generator), conditions in by_pair.items():
            if pair_generator != generator or set(conditions) != {"no_rag", "rule_rag"}:
                continue
            gap = abs(conditions["no_rag"]["word_count"] - conditions["rule_rag"]["word_count"])
            candidates.append((gap, case_id, generator))
        candidates.sort()
        selected.update((case_id, generator) for _, case_id, generator in candidates[:10])
    if len(selected) != 30:
        raise ValueError(f"Expected 30 length-matched pairs, found {len(selected)}.")
    return selected


def combine_records(
    generations: Sequence[Mapping[str, Any]],
    packets: Sequence[Mapping[str, Any]],
    extractions: Sequence[Mapping[str, Any]],
    verifications: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    packet_by_case = {row["case_id"]: row for row in packets}
    extraction_by_key = {
        (row["case_id"], row["generator"], row["condition"]): row for row in extractions
    }
    verification_by_key = {
        (row["case_id"], row["generator"], row["condition"]): row for row in verifications
    }
    explanations: list[dict[str, Any]] = []
    for generation in generations:
        key = (generation["case_id"], generation["generator"], generation["condition"])
        packet = packet_by_case[generation["case_id"]]
        compact_a = packet["A_common_context"]
        complete_a = {
            "user_request": compact_a["user_request"],
            "query_item_id": packet["query_item_id"],
            "query_item_category": packet["B_exact_stored_trace"]["query_group"],
            "query_item_text": compact_a["query_item_minimal_name"],
            "locked_candidate_id": packet["locked_candidate_id"],
            "locked_item_category": packet["target_category"],
            "locked_item_text": compact_a["locked_item_minimal_name"],
        }
        extraction = extraction_by_key[key]
        verification = verification_by_key[key]
        extracted_by_id = {claim["claim_id"]: claim for claim in extraction["claims"]}
        claims = []
        for verified in verification["claims"]:
            extracted = extracted_by_id[verified["claim_id"]]
            claim = {**verified, **extracted}
            claim["claim_role"] = claim_role(claim["claim_type"])
            claim["dta_entailed"] = dta_entailed(claim)
            claim["source_bucket"] = source_bucket(claim)
            claim["visible_status"] = visible_status(generation["condition"], claim)
            classification = classify_item_attribute(claim, complete_a)
            claim["attribute_bucket"] = classification.bucket
            claim["attribute_reason"] = classification.reason
            if classification.bucket == "item_attribute":
                supported_a = explicit_a_entails_attribute(claim["claim_text"], complete_a)
                supported_b = generation["condition"] == "rule_rag" and exact_b_entails_attribute(
                    claim["claim_text"],
                    complete_a,
                    packet["B_exact_stored_trace"]["rules"],
                )
                claim["uiar_unsupported"] = not (supported_a or supported_b)
                claim["uiar_support_a_literal"] = supported_a
                claim["uiar_support_b_strict"] = supported_b
            else:
                claim["uiar_unsupported"] = None
                claim["uiar_support_a_literal"] = None
                claim["uiar_support_b_strict"] = None
            claim["citation_observation_available"] = citation_observation_available(claim)
            claim["valid_rule_citation"] = valid_rule_citation(claim, verification["citation_ids"])
            claim["requires_rule_support"] = requires_rule_support(claim)
            claims.append(claim)
        explanations.append(
            {
                "case_id": generation["case_id"],
                "generator": generation["generator"],
                "condition": generation["condition"],
                "category": generation["target_category"],
                "word_count": generation["word_count"],
                "output_text": generation["output_text"],
                "status": verification["status"],
                "citation_ids": verification["citation_ids"],
                "context_a": complete_a,
                "trace_b": packet["B_exact_stored_trace"],
                "claims": claims,
            }
        )
    return explanations


def aggregate(explanations: Sequence[Mapping[str, Any]], aggregation: str) -> dict[str, Any]:
    substantive = [
        claim
        for explanation in explanations
        for claim in explanation["claims"]
        if claim["claim_role"] == "substantive"
    ]
    attributes = [
        claim
        for explanation in explanations
        for claim in explanation["claims"]
        if claim["attribute_bucket"] == "item_attribute"
    ]
    ambiguous = [
        claim
        for explanation in explanations
        for claim in explanation["claims"]
        if claim["attribute_bucket"] == "ambiguous"
    ]
    condition = explanations[0]["condition"] if explanations else ""

    if aggregation == "micro_claim":
        dta = rate(substantive, lambda claim: claim["dta_entailed"])
        uiar = rate(attributes, lambda claim: claim["uiar_unsupported"])
        unsupported = sum(claim["uiar_unsupported"] for claim in attributes)
        words = sum(explanation["word_count"] for explanation in explanations)
        density = unsupported / words * 100 if words else None
        visible = rate(substantive, lambda claim: claim["visible_status"] == "supported")
        shared_ab = rate(substantive, lambda claim: claim["support_status"] == "supported")
        citation_pool = [claim for claim in substantive if claim["citation_observation_available"]]
        requires = [claim for claim in substantive if claim["requires_rule_support"]]
        citation_precision = (
            rate(citation_pool, lambda claim: claim["valid_rule_citation"])
            if condition == "rule_rag"
            else None
        )
        citation_coverage = (
            rate(requires, lambda claim: claim["valid_rule_citation"])
            if condition == "rule_rag"
            else None
        )
    else:
        dta_rates = []
        uiar_rates = []
        density_rates = []
        visible_rates = []
        shared_rates = []
        precision_rates = []
        coverage_rates = []
        for explanation in explanations:
            exp_substantive = [
                claim for claim in explanation["claims"] if claim["claim_role"] == "substantive"
            ]
            exp_attributes = [
                claim
                for claim in explanation["claims"]
                if claim["attribute_bucket"] == "item_attribute"
            ]
            citation_pool = [
                claim for claim in exp_substantive if claim["citation_observation_available"]
            ]
            requires = [claim for claim in exp_substantive if claim["requires_rule_support"]]
            if exp_substantive:
                dta_rates.append(rate(exp_substantive, lambda claim: claim["dta_entailed"]))
                visible_rates.append(
                    rate(exp_substantive, lambda claim: claim["visible_status"] == "supported")
                )
                shared_rates.append(
                    rate(exp_substantive, lambda claim: claim["support_status"] == "supported")
                )
            if exp_attributes:
                uiar_rates.append(rate(exp_attributes, lambda claim: claim["uiar_unsupported"]))
            density_rates.append(
                sum(claim["uiar_unsupported"] for claim in exp_attributes)
                / explanation["word_count"]
                * 100
            )
            if condition == "rule_rag" and citation_pool:
                precision_rates.append(
                    rate(citation_pool, lambda claim: claim["valid_rule_citation"])
                )
            if condition == "rule_rag" and requires:
                coverage_rates.append(rate(requires, lambda claim: claim["valid_rule_citation"]))
        dta = mean(dta_rates)
        uiar = mean(uiar_rates)
        density = mean(density_rates)
        visible = mean(visible_rates)
        shared_ab = mean(shared_rates)
        citation_precision = mean(precision_rates) if condition == "rule_rag" else None
        citation_coverage = mean(coverage_rates) if condition == "rule_rag" else None

    return {
        "explanations": len(explanations),
        "substantive_claims": len(substantive),
        "item_attribute_claims": len(attributes),
        "uiar_eligible_explanations": sum(
            any(claim["attribute_bucket"] == "item_attribute" for claim in row["claims"])
            for row in explanations
        ),
        "ambiguous_attribute_claims": len(ambiguous),
        "dta_rate": dta,
        "uiar_rate": uiar,
        "unsupported_attribute_density_per_100_words": density,
        "citation_precision": citation_precision,
        "citation_coverage": citation_coverage,
        "citation_evaluated_claims": (
            sum(claim["citation_observation_available"] for claim in substantive)
            if condition == "rule_rag"
            else ""
        ),
        "validly_cited_claims": (
            sum(claim["valid_rule_citation"] for claim in substantive)
            if condition == "rule_rag"
            else ""
        ),
        "validly_cited_rule_required_claims": (
            sum(
                claim["valid_rule_citation"] and claim["requires_rule_support"]
                for claim in substantive
            )
            if condition == "rule_rag"
            else ""
        ),
        "claims_requiring_rule_support": (
            sum(claim["requires_rule_support"] for claim in substantive)
            if condition == "rule_rag"
            else ""
        ),
        "visible_evidence_support_rate_secondary": visible,
        "shared_ab_support_rate_secondary": shared_ab,
    }


def paired_metric_rows(
    explanations: Sequence[Mapping[str, Any]], subset: str, scope: str, scope_value: str
) -> dict[str, Any] | None:
    selected = [
        row
        for row in explanations
        if scope == "overall"
        or (scope == "generator" and row["generator"] == scope_value)
        or (scope == "category" and row["category"] == scope_value)
    ]
    by_key = {(row["case_id"], row["generator"], row["condition"]): row for row in selected}
    dta_values: list[float] = []
    dta_clusters: list[str] = []
    uiar_values: list[float] = []
    uiar_clusters: list[str] = []
    for (case_id, generator, condition), no_rag in sorted(by_key.items()):
        if condition != "no_rag":
            continue
        rule_rag = by_key.get((case_id, generator, "rule_rag"))
        if not rule_rag:
            continue
        no_sub = [c for c in no_rag["claims"] if c["claim_role"] == "substantive"]
        rule_sub = [c for c in rule_rag["claims"] if c["claim_role"] == "substantive"]
        if no_sub and rule_sub:
            dta_values.append(
                rate(rule_sub, lambda claim: claim["dta_entailed"])
                - rate(no_sub, lambda claim: claim["dta_entailed"])
            )
            dta_clusters.append(case_id)
        no_attr = [c for c in no_rag["claims"] if c["attribute_bucket"] == "item_attribute"]
        rule_attr = [c for c in rule_rag["claims"] if c["attribute_bucket"] == "item_attribute"]
        if no_attr and rule_attr:
            uiar_values.append(
                rate(rule_attr, lambda claim: claim["uiar_unsupported"])
                - rate(no_attr, lambda claim: claim["uiar_unsupported"])
            )
            uiar_clusters.append(case_id)
    if not dta_values:
        return None
    seed_offset = int(
        hashlib.sha256(f"{subset}|{scope}|{scope_value}".encode()).hexdigest()[:8], 16
    )
    dta_mean, dta_low, dta_high, dta_draws = clustered_bootstrap_mean(
        dta_values,
        dta_clusters,
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=CONFIDENCE_LEVEL,
        seed=SEED + seed_offset,
    )
    result = {
        "analysis_subset": subset,
        "scope": scope,
        "scope_value": scope_value,
        "aggregation": "paired_macro_difference",
        "condition": "rule_rag_minus_no_rag",
        "explanations": len(dta_values),
        "substantive_claims": "",
        "item_attribute_claims": "",
        "uiar_eligible_explanations": "",
        "ambiguous_attribute_claims": "",
        "dta_rate": dta_mean,
        "uiar_rate": "",
        "unsupported_attribute_density_per_100_words": "",
        "citation_precision": "",
        "citation_coverage": "",
        "citation_evaluated_claims": "",
        "validly_cited_claims": "",
        "validly_cited_rule_required_claims": "",
        "claims_requiring_rule_support": "",
        "visible_evidence_support_rate_secondary": "",
        "shared_ab_support_rate_secondary": "",
        "dta_ci_lower": dta_low,
        "dta_ci_upper": dta_high,
        "dta_p_value": two_sided_bootstrap_pvalue(dta_draws),
        "uiar_ci_lower": "",
        "uiar_ci_upper": "",
        "uiar_p_value": "",
        "dta_paired_units": len(dta_values),
        "uiar_paired_units": len(uiar_values),
    }
    if len(uiar_values) >= 30:
        uiar_mean, uiar_low, uiar_high, uiar_draws = clustered_bootstrap_mean(
            uiar_values,
            uiar_clusters,
            replicates=BOOTSTRAP_REPLICATES,
            confidence_level=CONFIDENCE_LEVEL,
            seed=SEED + seed_offset + 1,
        )
        result.update(
            {
                "uiar_rate": uiar_mean,
                "uiar_ci_lower": uiar_low,
                "uiar_ci_upper": uiar_high,
                "uiar_p_value": two_sided_bootstrap_pvalue(uiar_draws),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    stage7_manifest = read_json(STAGE7_MANIFEST)
    stage8_manifest = read_json(STAGE8_MANIFEST)
    generations_path = locate_output(stage7_manifest, "generations.jsonl")
    packets_path = locate_output(stage7_manifest, "case_evidence_packets.jsonl")
    extractions_path = locate_output(stage8_manifest, "extractions.jsonl")
    verifications_path = locate_output(stage8_manifest, "verifications.jsonl")
    explanations = combine_records(
        read_jsonl(generations_path),
        read_jsonl(packets_path),
        read_jsonl(extractions_path),
        read_jsonl(verifications_path),
    )
    length_pairs = select_length_matched(explanations)
    length_rows = [
        row for row in explanations if (row["case_id"], row["generator"]) in length_pairs
    ]
    subset_specs = [
        ("full_corpus", explanations),
        (
            "length_matched_10_pairs_per_generator",
            length_rows,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for subset_name, subset in subset_specs:
        scopes: list[tuple[str, str]] = [("overall", "all")]
        if subset_name == "full_corpus":
            scopes.extend(
                ("generator", value) for value in sorted({r["generator"] for r in subset})
            )
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
                conditioned = [row for row in scoped if row["condition"] == condition]
                for aggregation in ("micro_claim", "macro_explanation"):
                    rows.append(
                        {
                            "analysis_subset": subset_name,
                            "scope": scope,
                            "scope_value": scope_value,
                            "aggregation": aggregation,
                            "condition": condition,
                            **aggregate(conditioned, aggregation),
                            "dta_ci_lower": "",
                            "dta_ci_upper": "",
                            "dta_p_value": "",
                            "uiar_ci_lower": "",
                            "uiar_ci_upper": "",
                            "uiar_p_value": "",
                            "dta_paired_units": "",
                            "uiar_paired_units": "",
                        }
                    )
            paired = paired_metric_rows(scoped, subset_name, scope, scope_value)
            if paired:
                rows.append(paired)

    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    with args.output_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    attribute_counts: dict[str, int] = defaultdict(int)
    attribute_reason_counts: dict[str, int] = defaultdict(int)
    for explanation in explanations:
        for claim in explanation["claims"]:
            attribute_counts[claim["attribute_bucket"]] += 1
            attribute_reason_counts[f"{claim['attribute_bucket']}|{claim['attribute_reason']}"] += 1
    input_paths = [
        STAGE7_MANIFEST,
        STAGE8_MANIFEST,
        STAGE8_REVISION_MANIFEST,
        generations_path,
        packets_path,
        extractions_path,
        verifications_path,
    ]
    manifest = {
        "schema_version": 1,
        "stage": 8,
        "stage_name": "study_specific_explanation_metrics",
        "status": "complete_zero_inference_additive_analysis",
        "timestamp_utc": utc_timestamp(),
        "git_commit_at_derivation": git_commit(),
        "model_calls": 0,
        "preserved_baselines": {
            "stage8_assessment_manifest": sha256_file(STAGE8_MANIFEST),
            "stage8_metric_revision_manifest": sha256_file(STAGE8_REVISION_MANIFEST),
            "canonical_baselines_modified": False,
        },
        "input_artifact_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths
        },
        "output_artifact_hashes": {
            str(args.output_table.relative_to(ROOT)): sha256_file(args.output_table)
        },
        "metric_definitions": {
            "dta": (
                "substantive claims saved as supported by rule_evidence with at least one exact "
                "supporting Stage 6 trace rule ID / all substantive claims"
            ),
            "uiar": (
                "conservatively classified concrete actual-item attribute claims unsupported by "
                "explicit complete frozen A identity/category/text context or a strictly item-"
                "specific B rule / all classified actual-item attribute claims"
            ),
            "unsupported_attribute_density": (
                "unsupported classified item-attribute claims / explanation words * 100"
            ),
            "citation_precision": (
                "substantive claims with saved citation entailment true, supported B attribution, "
                "and overlap between supporting and observed rule IDs / substantive claims with a "
                "non-null saved citation-entailment decision"
            ),
            "citation_coverage": (
                "validly cited substantive claims / substantive claims not already attributed to A"
            ),
        },
        "condition_interpretation": {
            "no_rag_dta": "post-hoc decision-trace alignment; not grounding",
            "rule_rag_dta": "decision-trace faithfulness",
            "citation_metrics_no_rag": "not_applicable",
        },
        "length_matched_sensitivity": {
            "role": "sensitivity_only_not_primary_confirmatory_estimate",
            "method": "smallest_case_paired_absolute_word_gaps_stratified_by_generator",
            "pairs_per_generator": 10,
            "pairs": len(length_pairs),
            "mean_words_by_condition": {
                condition: mean(
                    [row["word_count"] for row in length_rows if row["condition"] == condition]
                )
                for condition in ("no_rag", "rule_rag")
            },
            "mean_absolute_paired_word_gap": mean(
                [
                    abs(
                        next(
                            row["word_count"]
                            for row in length_rows
                            if (row["case_id"], row["generator"]) == pair
                            and row["condition"] == "no_rag"
                        )
                        - next(
                            row["word_count"]
                            for row in length_rows
                            if (row["case_id"], row["generator"]) == pair
                            and row["condition"] == "rule_rag"
                        )
                    )
                    for pair in sorted(length_pairs)
                ]
            ),
            "pair_keys_sha256": hashlib.sha256(
                json.dumps(sorted(length_pairs), separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "uiar_classifier": {
            "type": "conservative_deterministic_text_and_schema_classifier",
            "bucket_counts": dict(sorted(attribute_counts.items())),
            "reason_counts": dict(sorted(attribute_reason_counts.items())),
            "generic_B_rules_establish_instance_attributes": False,
            "ambiguous_bucket_excluded_from_uiar": True,
        },
        "row_counts": {
            "metric_table_rows": len(rows),
            "explanations": len(explanations),
            "claims": sum(len(row["claims"]) for row in explanations),
        },
    }
    write_json(args.output_manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
