from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from evidence_fashion.data import build_evaluation_cases, map_broad_category, prepare_metadata
from evidence_fashion.kb_audit import (
    audit_bag_case_packets,
    coverage_matrix,
    load_canonical_rules,
    load_legacy_audit,
)
from evidence_fashion.rule_retrieval import RuleRetriever

ROOT = Path(__file__).parents[1]


def _config() -> dict:
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    config["dataset"]["expected_counts"] = {}
    config["splits"].pop("exact_outfit_counts", None)
    config["recommendation_evaluation"].update(
        {"case_count": 5, "cases_per_category": 1, "case_split": "test"}
    )
    return config


def _rows() -> list[dict[str, str]]:
    raw_categories = {
        "tops": "Blouses",
        "bottoms": "Skinny Jeans",
        "shoes": "Ankle Booties",
        "outerwear": "Blazers",
        "bags": "Shoulder Bags",
    }
    return [
        {
            "item_ID": f"outfit-{outfit}_{position}",
            "category": category,
            "text": broad,
        }
        for outfit in range(30)
        for position, (broad, category) in enumerate(raw_categories.items())
    ]


def test_stage1_taxonomy_and_cases_are_bag_native() -> None:
    config = _config()
    expected = ["tops", "bottoms", "shoes", "outerwear", "bags"]
    assert config["preprocessing"]["target_categories"] == expected
    assert config["recommendation_evaluation"]["target_category_order"] == expected
    assert map_broad_category("Shoulder Bags", config) == "bags"
    for excluded in ("Backpacks", "Briefcases", "Luggage", "Sunglasses"):
        assert map_broad_category(excluded, config) == "other"

    frame = prepare_metadata(_rows(), config)
    cases = build_evaluation_cases(frame, config)
    assert cases["target_category"].value_counts().to_dict() == {
        category: 1 for category in expected
    }
    assert "target_accessory_subcategory" not in cases.columns
    allowed = set(config["preprocessing"]["category_taxonomy"]["bag_allowlist"])
    bag_case = cases[cases["target_category"].eq("bags")].iloc[0]
    bag_items = frame.set_index("item_id").loc[bag_case["positive_item_ids"], "category"]
    assert set(bag_items).issubset(allowed)


def test_five_category_kb_and_legacy_audit_are_complete() -> None:
    config = _config()
    rules = load_canonical_rules(ROOT / config["paths"]["knowledge_base"])
    audit = load_legacy_audit(ROOT / config["paths"]["legacy_kb_audit"])
    assert len(rules) >= 100
    assert (rules["recommended_category"].value_counts() >= 20).all()
    assert set(rules["input_category"]) <= {"tops", "bottoms", "shoes", "outerwear", "bags"}
    assert rules["audit_status"].eq("retain").all()
    assert rules["source_locator"].str.strip().ne("").all()
    assert rules["source_validation_status"].str.fullmatch(
        r"verified_reachable_and_direct_\d{4}-\d{2}-\d{2}"
    ).all()
    assert rules["rule_limitations"].str.strip().ne("").all()
    assert audit["audited_asset"]["row_count"] == 126
    assert audit["result"]["legacy_rows_carried_forward_verbatim"] == 0
    assert audit["result"]["experimental_results_inspected"] is False
    matrix = coverage_matrix(rules)
    off_diagonal = matrix.to_numpy()[~np.eye(len(matrix), dtype=bool)]
    assert (off_diagonal >= 3).all()
    source_registry = pd.read_csv(ROOT / config["paths"]["kb_source_registry"])
    assert len(source_registry) >= 43
    assert source_registry["rule_count"].sum() == len(rules)
    similarity_audit = pd.read_csv(ROOT / config["paths"]["kb_rule_similarity_audit"])
    assert len(similarity_audit) >= 63
    assert similarity_audit["audit_decision"].str.startswith("retain_distinct_").all()
    report = (ROOT / config["paths"]["kb_audit_report"]).read_text(encoding="utf-8")
    assert "126 / 126" in report
    assert "experimental condition results were not inspected" in report
    assert f"{len(rules)}/{len(rules)} direct HTTPS citations" in report


def test_stage2_static_bag_audit_passes_coverage_and_diversity_gates() -> None:
    audit = json.loads(
        (ROOT / "reports/stage2_bag_case_applicability_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["supported_case_count"] == 200
    assert audit["unsupported_case_count"] == 0
    assert audit["maximum_rule_prevalence"] <= 0.30
    assert audit["duplicate_packet_case_fraction"] <= 0.70
    assert audit["unique_nonempty_packets"] >= 100
    assert audit["stage2_pass"] is True
    assert audit["experimental_condition_results_inspected"] is False


def test_stage2_freeze_manifest_binds_the_reviewed_kb_artifacts() -> None:
    manifest = json.loads(
        (ROOT / "artifacts/manifests/stage2_kb_freeze_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "frozen"
    assert manifest["rule_count"] >= 100
    assert min(manifest["rule_counts_by_target"].values()) >= 20
    assert manifest["audit_gates"]["stage2_pass"] is True
    assert manifest["experimental_condition_results_inspected"] is False
    for relative_path, expected_hash in manifest["bound_artifact_hashes"].items():
        artifact = ROOT / relative_path
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected_hash


def _settings() -> dict:
    return {
        "candidate_top_k": 5,
        "category_filter_field": "recommended_category",
        "applicability_filter_field": "applicable_query_categories",
        "required_context_field": "required_context",
        "query_terms_field": "query_terms",
        "candidate_terms_field": "candidate_terms",
        "audit_status_field": "audit_status",
        "approved_audit_status": "retain",
        "reliability_weights": {"high": 1.0, "medium": 0.85, "low": 0.65},
        "query_group_bonus": 0.0,
        "score_max_weight": 0.7,
        "score_mean_weight": 0.3,
    }


def _case(query_group: str) -> dict:
    return {
        "query_category": "Blouses",
        "query_group": query_group,
        "query_text": "simple blouse",
        "user_request": "Recommend a bag that completes this outfit.",
        "target_category": "bags",
    }


def test_bag_retrieval_records_empty_trace_when_no_rule_is_approved() -> None:
    rules = pd.DataFrame(
        [{
            "rule_id": "R1",
            "rule_text": "Recommend a tote.",
            "input_category": "tops",
            "recommended_category": "bags",
            "source_reliability": "high",
            "audit_status": "rewrite_required",
            "applicable_query_categories": "tops",
            "required_context": "none",
            "query_terms": "none",
            "candidate_terms": "none",
        }]
    )
    retriever = RuleRetriever(rules, np.array([[1.0, 0.0]]), _settings())
    trace = retriever.retrieve_and_score(
        case=_case("tops"),
        candidate={"item_id": "b1", "category": "Tote Bags", "text": "black tote"},
        representation_embedding=np.array([1.0, 0.0]),
    )
    assert trace.rules == ()
    assert trace.evidence_score == 0.0


def test_bag_retrieval_enforces_query_applicability_before_top_k() -> None:
    rules = pd.DataFrame(
        [
            {
                "rule_id": "R1",
                "rule_text": "Recommend a directly supported tote.",
                "input_category": "tops",
                "recommended_category": "bags",
                "source_reliability": "high",
                "audit_status": "retain",
                "applicable_query_categories": "tops",
                "required_context": "none",
                "query_terms": "none",
                "candidate_terms": "none",
            },
            {
                "rule_id": "R2",
                "rule_text": "Outerwear-only bag rule.",
                "input_category": "outerwear",
                "recommended_category": "bags",
                "source_reliability": "high",
                "audit_status": "retain",
                "applicable_query_categories": "outerwear",
                "required_context": "none",
                "query_terms": "none",
                "candidate_terms": "none",
            },
        ]
    )
    retriever = RuleRetriever(rules, np.array([[1.0, 0.0], [1.0, 0.0]]), _settings())
    trace = retriever.retrieve_and_score(
        case=_case("tops"),
        candidate={"item_id": "b1", "category": "Tote Bags", "text": "black tote"},
        representation_embedding=np.array([1.0, 0.0]),
    )
    assert [rule.rule_id for rule in trace.rules] == ["R1"]
    assert trace.filtering["rules_excluded_by_applicability"] == 1


def test_case_packet_audit_reports_coverage_duplication_and_overlap() -> None:
    cases = [
        {
            "case_id": "b1",
            "target_category": "bags",
            "query_group": "tops",
            "evidence_trace": {"rules": [{"rule_id": "R1"}, {"rule_id": "R2"}]},
        },
        {
            "case_id": "b2",
            "target_category": "bags",
            "query_group": "bottoms",
            "evidence_trace": {"rules": [{"rule_id": "R1"}, {"rule_id": "R2"}]},
        },
        {
            "case_id": "b3",
            "target_category": "bags",
            "query_group": "shoes",
            "evidence_trace": {"rules": []},
        },
    ]
    audit = audit_bag_case_packets(cases)
    assert audit["supported_case_count"] == 2
    assert audit["unsupported_case_ids"] == ["b3"]
    assert audit["duplicate_packet_cases"] == 2
    assert audit["maximum_rule_prevalence"] == pytest.approx(2 / 3)
    assert audit["coverage_pass"] is False
