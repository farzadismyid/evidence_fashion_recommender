import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(reason="Legacy-run manifests are outside the fresh Stage 1-2 run")

ROOT = Path(__file__).parents[1]


def test_active_data_manifest_proves_zero_exact_image_leakage() -> None:
    path = ROOT / "artifacts/manifests/data_preparation_leakage_resolved_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    resolution = manifest["validation"]["duplicate_resolution"]
    assert resolution == {
        "changed_outfits": 13,
        "component_reassignments": 10,
        "duplicate_groups": 19,
        "final_cross_split_groups": 0,
        "initial_cross_split_groups": 10,
        "rebalance_reassignments": 3,
    }
    assert manifest["validation"]["cross_split_exact_image_groups"] == 0
    assert manifest["row_counts"]["candidate_rows"] == 879134
    assert manifest["row_counts"]["development_outfits"] == 14157
    assert manifest["row_counts"]["validation_outfits"] == 3033
    assert manifest["row_counts"]["test_outfits"] == 3035


def test_active_embedding_manifest_uses_repaired_data() -> None:
    path = ROOT / "artifacts/manifests/embedding_leakage_resolved_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["stage"] == 3
    data_manifest = json.loads(
        (ROOT / "artifacts/manifests/data_preparation_leakage_resolved_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["configuration_hash"] == data_manifest["configuration_hash"]
    assert manifest["row_counts"] == {"embedded_items": 8}
    assert manifest["validation"]["deterministic_ranking"] is True
