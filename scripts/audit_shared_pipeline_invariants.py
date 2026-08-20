"""Audit shared grounding contracts across a large validation-only structural sample."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evidence_fashion.assessment import validate_extraction
from evidence_fashion.grounding_contracts import (
    require_trace_applicability,
    rule_applicability_gate,
    validate_generated_explanation,
)
from evidence_fashion.manifest import sha256_file, utc_timestamp, write_new_json
from evidence_fashion.rule_retrieval import full_kb_candidate_retrieval

ROOT = Path(__file__).parents[1]
QUERY_CATEGORY = {
    "tops": "bottoms",
    "bottoms": "tops",
    "shoes": "bottoms",
    "outerwear": "tops",
    "bags": "tops",
}
# The stricter production gate leaves fewer trace-valid validation examples for
# shoes and outerwear.  These fixed quotas retain every available case for
# those targets and still make one 100-case, cross-category audit.
SAMPLE_QUOTAS = {"tops": 26, "bottoms": 30, "shoes": 9, "outerwear": 9, "bags": 26}


def _load_items(experiment: dict[str, Any]) -> pd.DataFrame:
    manifest = json.loads(
        (ROOT / experiment["paths"]["active_data_manifest"]).read_text(encoding="utf-8")
    )
    path = ROOT / next(
        item
        for item in manifest["output_artifact_hashes"]
        if item.endswith("prepared_items.parquet")
    )
    return pd.read_parquet(path)


def _sample(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    items = _load_items(experiment)
    rules = pd.read_csv(ROOT / experiment["paths"]["knowledge_base"])
    validation = items[items["research_split"].eq("validation")]
    selected: list[dict[str, Any]] = []
    for target, query in QUERY_CATEGORY.items():
        rows: list[dict[str, Any]] = []
        for outfit_id, outfit in validation.groupby("outfit_id", sort=False):
            targets = outfit[outfit["broad_category"].eq(target)].sort_values("item_id")
            queries = outfit[outfit["broad_category"].eq(query)].sort_values("item_id")
            if targets.empty or queries.empty:
                continue
            locked, query_item = targets.iloc[0], queries.iloc[0]
            context = " | ".join(f"{row.category}: {row.text}" for row in outfit.itertuples())
            case = {
                "target_category": target,
                "query_category": query,
                "query_group": query,
                "query_text": str(query_item["text"]),
                "outfit_context_text": context,
                "user_request": experiment["preprocessing"]["category_taxonomy"][
                    "broad_request_templates"
                ][target],
                "applicability_contexts": [],
            }
            candidate = {"category": str(locked["category"]), "text": str(locked["text"])}
            trace = []
            for rule in full_kb_candidate_retrieval(
                rules, target_category=target, query_group=query
            ):
                decision = rule_applicability_gate(rule, case=case, candidate=candidate)
                if decision.established:
                    trace.append(
                        {
                            "rule_id": rule["rule_id"],
                            "rule_text": rule["rule_text"],
                            **decision.trace_metadata(),
                        }
                    )
            if trace:
                rows.append(
                    {
                        "outfit_id": str(outfit_id),
                        "target_category": target,
                        "locked_item": str(locked["text"]),
                        "trace": trace[:5],
                    }
                )
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"shared-invariant-audit:{target}:{row['outfit_id']}".encode()
            ).hexdigest()
        )
        quota = SAMPLE_QUOTAS[target]
        if len(rows) < quota:
            raise ValueError(f"Only {len(rows)} trace-valid validation cases for {target}.")
        selected.extend(rows[:quota])
    return selected


def main() -> None:
    experiment = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    sample = _sample(experiment)
    checks = {
        "trace_antecedent_validity": 0,
        "locked_item_preservation": 0,
        "citation_in_trace_validity": 0,
        "canonical_citation_syntax": 0,
        "claim_id_integrity": 0,
        "duplicate_control": 0,
    }
    for row in sample:
        require_trace_applicability({"rules": row["trace"]})
        checks["trace_antecedent_validity"] += 1
        citation = row["trace"][0]["rule_id"]
        probe = f"{row['locked_item']} is the exact recommended item. [{citation}]"
        validate_generated_explanation(
            probe,
            locked_item_name=row["locked_item"],
            target_category=row["target_category"],
            trace_rule_ids=[rule["rule_id"] for rule in row["trace"]],
            citations_required=True,
        )
        checks["locked_item_preservation"] += 1
        checks["citation_in_trace_validity"] += 1
        checks["canonical_citation_syntax"] += 1
        claims = validate_extraction(
            {
                "claims": [
                    {
                        "claim_id": "C1",
                        "claim_text": "The exact item is recommended.",
                        "claim_type": "other",
                    },
                    {
                        "claim_id": "C2",
                        "claim_text": "The exact item is recommended.",
                        "claim_type": "other",
                    },
                ]
            },
            ["other"],
        )
        if [claim["claim_id"] for claim in claims] != ["C1"]:
            raise ValueError("Claim duplicate control failed.")
        checks["claim_id_integrity"] += 1
        checks["duplicate_control"] += 1
    run_identity = {
        "sample": sample,
        "grounding_contracts_sha256": sha256_file(
            ROOT / "src/evidence_fashion/grounding_contracts.py"
        ),
    }
    run_id = hashlib.sha256(json.dumps(run_identity, sort_keys=True).encode()).hexdigest()[:12]
    runtime = ROOT / ".runtime/current/audits" / f"shared-invariants-{run_id}"
    runtime.mkdir(parents=True, exist_ok=False)
    sample_path = runtime / "validation_structural_sample.jsonl"
    sample_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sample), encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "audit_type": "shared_pipeline_structural_contracts",
        "sample_origin": "validation",
        "sample_size": len(sample),
        "sample_size_by_category": SAMPLE_QUOTAS,
        "generation_execution": "structural contract probes only; no production outputs changed",
        "checks": checks,
        "timestamp_utc": utc_timestamp(),
        "sample_path": str(sample_path.relative_to(ROOT)),
        "bound_artifact_hashes": {
            "configs/experiment.yaml": sha256_file(ROOT / "configs/experiment.yaml"),
            "data/kb/fashion_rules.csv": sha256_file(ROOT / "data/kb/fashion_rules.csv"),
            str(sample_path.relative_to(ROOT)): sha256_file(sample_path),
            "src/evidence_fashion/grounding_contracts.py": sha256_file(
                ROOT / "src/evidence_fashion/grounding_contracts.py"
            ),
        },
    }
    write_new_json(runtime / "report.json", report)
    write_new_json(ROOT / "artifacts/manifests/shared_pipeline_structural_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
