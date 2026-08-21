import csv
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_canonical_five_category_kb_has_auditable_baseline_and_expansion() -> None:
    path = ROOT / "data" / "kb" / "fashion_rules.csv"
    assert len(hashlib.sha256(path.read_bytes()).hexdigest()) == 64
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 100
    assert len({row["rule_id"] for row in rows if row["rule_id"]}) == len(rows)
    assert {f"K{index:03d}" for index in range(1, 101)}.issubset(
        {row["rule_id"] for row in rows}
    )
    assert {row["recommended_category"] for row in rows} == {
        "tops", "bottoms", "shoes", "outerwear", "bags"
    }


def test_configuration_pins_authoritative_dataset_and_evidence_boundary() -> None:
    config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    assert config["dataset"]["revision"] == "8c782ee447faf2d2a0402ac883cf07d3b3f43e1c"
    assert config["candidate_pool"]["max_negatives"] == 99
    assert config["stage4_validation"]["completed_diagnostic_candidate_max_negatives"] == 999
    assert config["retrieval"]["fusion"]["image_weight"] == 0.40
    assert config["retrieval"]["fusion"]["text_weight"] == 0.60
    assert config["retrieval"]["fusion"] == {
        "image_weight": 0.40,
        "text_weight": 0.60,
        "normalize_inputs": True,
        "normalize_output": True,
        "selection_split": "validation",
        "validation_grid_image_weights": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "selection_metric": "ndcg_at_10",
        "tie_breaking": [
            "higher_mrr",
            "higher_hr_at_10",
            "lower_distance_from_reference",
            "lower_image_weight",
        ],
        "frozen": True,
        "operating_point_policy": "researcher_selected_multimodal_design_constraint",
        "validation_optimum_is_diagnostic_only": True,
    }
    assert config["explanation_evidence"]["forbid_image_derived_text"] is True
    assert config["explanation_evidence"]["B_source"] == "exact_stored_rule_scoring_trace"


def test_clean_top_level_directories_match_proposal() -> None:
    actual = {
        path.name for path in ROOT.iterdir() if path.is_dir() and not path.name.startswith(".")
    }
    assert actual == {
        "artifacts",
        "archive",
        "configs",
        "data",
        "notebooks",
        "reports",
        "scripts",
        "src",
        "tests",
        "thesis",
    }
