import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(reason="Superseded by the five-category Stage 1-2 contracts")

ROOT = Path(__file__).parents[1]


def test_data_preparation_manifest_records_full_validation() -> None:
    path = ROOT / "artifacts" / "manifests" / "data_preparation_leakage_resolved_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["stage"] == 2
    assert manifest["row_counts"] == {
        "candidate_rows": 879134,
        "development_outfits": 14157,
        "evaluation_cases": 1000,
        "items": 69725,
        "outfits": 20225,
        "target_category_items": 69725,
        "test_outfits": 3035,
        "validation_outfits": 3033,
    }
    assert manifest["validation"]["split_outfit_overlap"] == 0
    assert manifest["validation"]["duplicate_item_ids"] == 0
    assert manifest["validation"]["cross_split_exact_image_groups"] == 0
    assert manifest["validation"]["candidate_pool_min_size"] == 60
    assert manifest["validation"]["candidate_pool_max_size"] == 1002
    assert manifest["validation"]["category_audit"] == {
        "excluded_categories": 225,
        "kept_categories": 135,
        "kept_items": 69725,
        "raw_categories": 377,
        "raw_items": 94096,
        "review_categories": 17,
    }
    assert manifest["validation"]["case_counts"] == {
        "accessories": 200,
        "bottoms": 200,
        "outerwear": 200,
        "shoes": 200,
        "tops": 200,
    }
