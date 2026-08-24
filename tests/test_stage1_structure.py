import csv
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_final_kb_is_exactly_200_rules_with_40_per_target() -> None:
    path = ROOT / "data" / "kb" / "fashion_rules.csv"
    assert len(hashlib.sha256(path.read_bytes()).hexdigest()) == 64
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 200
    assert len({row["rule_id"] for row in rows if row["rule_id"]}) == 200
    counts = {}
    for row in rows:
        counts[row["recommended_category"]] = counts.get(row["recommended_category"], 0) + 1
    assert counts == {"tops": 40, "bottoms": 40, "shoes": 40, "outerwear": 40, "bags": 40}


def test_final_configuration_freezes_not_selects_the_confirmatory_defaults() -> None:
    config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    assert config["paths"]["knowledge_base"] == "data/kb/fashion_rules.csv"
    assert config["retrieval"]["fusion"]["image_weight"] == 0.40
    assert config["retrieval"]["fusion"]["text_weight"] == 0.60
    assert config["reranking"] == {
        "clip_weight": 0.75,
        "evidence_weight": 0.25,
        "normalization": "min_max_within_candidate_pool",
        "confirmatory_policy": "fixed_075_clip_025_evidence_top_k_5",
    }
    assert config["rule_retrieval"]["rule_top_k"] == 5
    assert config["validation_sensitivity"]["evidence_weights"][0] == 0.0
    assert config["validation_sensitivity"]["rule_top_k_values"] == [1, 3, 5]
    assert config["explanation_evidence"]["forbid_second_rule_retrieval"] is True
