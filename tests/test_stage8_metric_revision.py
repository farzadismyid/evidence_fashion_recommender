import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_stage8_metric_revision_manifest_and_table_when_present() -> None:
    manifest_path = ROOT / "artifacts/manifests/stage8_metric_revision_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["baseline_stage8_commit"] == "c9ecae4"
    assert manifest["model_calls"] == 0
    assert manifest["row_counts"]["verified_claims"] == 20618
    assert manifest["refusal_audit"]["stage7_detected"] == 0
    assert manifest["refusal_audit"]["stage8_detected"] == 0
    assert manifest["refusal_audit"]["classification_counts"] == {}
    assert manifest["normalization_sensitivity"]["verifier_outputs"] == 2980
    assert manifest["normalization_sensitivity"]["normalized_outputs"] == 1280
    for raw_path, expected in {
        **manifest["input_artifact_hashes"],
        **manifest["output_artifact_hashes"],
    }.items():
        artifact = ROOT / raw_path
        assert artifact.exists(), raw_path
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected

    table_path = ROOT / "artifacts/tables/table_stage8_grounding_revision.csv"
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    overall = {
        (row["aggregation"], row["condition"]): row
        for row in rows
        if row["analysis_subset"] == "all_records" and row["scope"] == "overall"
    }
    no_rag = overall[("micro_claim", "no_rag")]
    rule_rag = overall[("micro_claim", "rule_rag")]
    assert int(no_rag["substantive_explanatory_claims"]) == 10703
    assert int(rule_rag["substantive_explanatory_claims"]) == 8666
    assert float(no_rag["visible_supported_rate"]) < 0.03
    assert float(rule_rag["visible_supported_rate"]) > 0.85
    assert int(no_rag["source_b_only_claims"]) == 9632
