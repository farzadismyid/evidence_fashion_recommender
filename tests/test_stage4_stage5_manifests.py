import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.skip(reason="Legacy Stage 4-5 output contracts await regeneration")

ROOT = Path(__file__).parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_output_hashes(manifest: dict) -> None:
    for raw_path, expected in manifest["output_artifact_hashes"].items():
        path = ROOT / raw_path
        assert path.exists(), raw_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def _assert_supporting_hashes(manifest: dict) -> None:
    for raw_path, expected in manifest["supporting_diagnostic_artifact_hashes"].items():
        path = ROOT / raw_path
        assert path.exists(), raw_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_category_audit_covers_every_raw_category_once() -> None:
    audit = pd.read_csv(ROOT / "artifacts" / "tables" / "table_category_audit.csv")
    assert len(audit) == 377
    assert audit["raw_category"].is_unique
    assert audit["item_count"].sum() == 94096
    assert audit["decision"].value_counts().to_dict() == {
        "exclude": 225,
        "keep": 135,
        "review": 17,
    }


def test_accessory_cases_filter_every_candidate_to_requested_subcategory() -> None:
    manifest = json.loads(
        (
            ROOT
            / "artifacts"
            / "manifests"
            / "data_preparation_leakage_resolved_manifest.json"
        ).read_text(encoding="utf-8")
    )
    items_path = ROOT / next(
        path
        for path in manifest["output_artifact_hashes"]
        if path.endswith("prepared_items.parquet")
    )
    cases_path = ROOT / next(
        path
        for path in manifest["output_artifact_hashes"]
        if path.endswith("evaluation_cases.jsonl")
    )
    subtype = pd.read_parquet(items_path).set_index("item_id")["accessory_subcategory"]
    cases = _read_jsonl(cases_path)
    accessory = [case for case in cases if case["target_category"] == "accessories"]
    assert len(accessory) == 200
    for case in accessory:
        assert case["target_accessory_subcategory"]
        assert set(subtype.loc[case["candidate_item_ids"]]) == {
            case["target_accessory_subcategory"]
        }


def test_stage4_manifest_and_locked_trace_contract() -> None:
    manifest = json.loads(
        (ROOT / "artifacts" / "manifests" / "stage4_reranking_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    _assert_output_hashes(manifest)
    _assert_supporting_hashes(manifest)
    assert manifest["row_counts"]["validation_cases"] == 300
    assert manifest["row_counts"]["candidate_traces"] == 29955
    assert manifest["row_counts"]["pareto_configurations"] == 16
    assert manifest["row_counts"]["pool_sensitivity_rows"] == 15
    assert manifest["failure_counts"] == {"ranking_failures": 0, "trace_failures": 0}
    assert manifest["trace_validation"] == {
        "complete_trace_matches_selected_rule_count": True,
        "locked_traces_checked": 300,
        "maximum_absolute_score_reproduction_error": 0.0,
    }
    frozen = manifest["selection"]["frozen_stage5_operating_point"]
    assert (frozen["rule_top_k"], frozen["clip_weight"], frozen["evidence_weight"]) == (
        5,
        0.75,
        0.25,
    )
    locked_path = ROOT / next(
        path for path in manifest["output_artifact_hashes"] if path.endswith("locked_cases.jsonl")
    )
    locked = _read_jsonl(locked_path)
    accessory = [row for row in locked if row["target_category"] == "accessories"]
    assert len(accessory) == 60
    assert all(row["target_accessory_subcategory"] for row in accessory)
    assert all(len(row["evidence_trace"]["rules"]) == 5 for row in locked)
    for row in locked:
        contributions = [
            rule["weighted_contribution"] for rule in row["evidence_trace"]["rules"]
        ]
        reproduced = 0.7 * max(contributions) + 0.3 * sum(contributions) / len(contributions)
        assert abs(reproduced - row["evidence_trace"]["evidence_score"]) <= 1e-12

    sensitivity = pd.read_csv(ROOT / "artifacts" / "tables" / "table_stage4_pool_sensitivity.csv")
    assert set(sensitivity["pool_target"]) == {100, 500, 1000}
    assert set(sensitivity["method"]) == {
        "minilm_text",
        "clip_image",
        "clip_text",
        "fused_clip",
        "evidence_rerank_fixed_075_025",
    }
    main = pd.read_csv(ROOT / "artifacts" / "tables" / "table_stage4_main_results.csv")
    assert set(main["candidate_max_negatives"]) == {99}
    assert set(main["image_weight"]) == {0.4}
    assert set(main["text_weight"]) == {0.6}


def test_stage5_researcher_selected_prompt_passes_validation_constraint() -> None:
    config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    assert config["explanation_search"]["frozen_configuration_id"] == "rag_c3"
    assert config["explanation_search"]["maximum_mean_word_count_gap_from_no_rag"] == 5
    frozen = json.loads(
        (ROOT / "artifacts" / "manifests" / "stage5_frozen_settings.json").read_text(
            encoding="utf-8"
        )
    )
    assert frozen["configuration_id"] == "rag_c3"
    assert frozen["settings"]["rule_count"] == 5
    assert frozen["metrics"]["absolute_word_count_gap"] < 5
    assert frozen["selection_status"] == (
        "researcher_selected_primary_validated_for_fresh_pilot"
    )


def test_fresh_stage5_pilot_is_disjoint_exact_trace_and_hash_bound() -> None:
    pilot_manifest = json.loads(
        (ROOT / "artifacts" / "manifests" / "stage5_pilot_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    _assert_output_hashes(pilot_manifest)
    assert pilot_manifest["device"] == "cuda"
    assert pilot_manifest["row_counts"]["pilot_cases"] == 50
    assert pilot_manifest["row_counts"]["generator_case_pairs"] == 150
    assert pilot_manifest["row_counts"]["explanations"] == 300
    assert pilot_manifest["row_counts"]["length_matched_sensitivity_rows"] == 8
    assert pilot_manifest["failure_counts"] == {"malformed_outputs": 0}
    assert pilot_manifest["selected_settings"]["id"] == "rag_c3"
    assert pilot_manifest["length_fairness"]["maximum_mean_word_count_gap_from_no_rag"] == 5
    assert pilot_manifest["length_fairness"]["observed_absolute_mean_gap"] > 5
    pilot_path = ROOT / next(
        path
        for path in pilot_manifest["output_artifact_hashes"]
        if path.endswith("pilot_records.jsonl")
    )
    pilot = _read_jsonl(pilot_path)
    optimisation = json.loads(
        (ROOT / "artifacts" / "manifests" / "stage5_optimization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    optimisation_ids = set(optimisation["selection"]["optimisation_case_ids"])
    assert not optimisation_ids.intersection(row["case_id"] for row in pilot)
    assert all("Evidence rules:" not in row["no_rag"]["prompt"] for row in pilot)
    assert all(len(row["rule_rag"]["prompt_rule_ids"]) == 5 for row in pilot)
    assert all(
        set(row["rule_rag"]["prompt_rule_ids"])
        == {rule["rule_id"] for rule in row["B_exact_stored_trace"]["rules"]}
        for row in pilot
    )
    assert any(
        len(claim["support_source"]) > 1
        for row in pilot
        for condition in ("no_rag", "rule_rag")
        for claim in row[condition]["assessment"]["claims"]
    )
    sensitivity = pd.read_csv(
        ROOT / "artifacts" / "tables" / "table_stage5_length_matched_sensitivity.csv"
    )
    overall = sensitivity[sensitivity["generator"].eq("all_generators")]
    assert set(overall["matched_pairs"]) == {30}
    assert set(overall["condition"]) == {"no_rag", "rule_rag"}
