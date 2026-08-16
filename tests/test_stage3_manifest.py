import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_embedding_manifest_records_pinned_models_and_validation() -> None:
    path = ROOT / "artifacts" / "manifests" / "embedding_leakage_resolved_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["stage"] == 3
    assert manifest["scope"] == "validation"
    assert manifest["row_counts"] == {"embedded_items": 8}
    assert manifest["models"]["minilm"]["immutable_digest"] == (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    assert manifest["models"]["clip"]["immutable_digest"] == (
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    )
    assert manifest["models"]["minilm"]["device"] == "cuda"
    assert manifest["models"]["clip"]["device"] == "cuda"
    assert manifest["validation"]["dimensions"] == {
        "clip_fused": 512,
        "clip_image": 512,
        "clip_text": 512,
        "minilm_text": 384,
    }
    assert manifest["validation"]["deterministic_ranking"] is True
    assert manifest["validation"]["category_filtering"] is True
    assert manifest["explanation_evidence_boundary"] == "no_image_derived_text"
