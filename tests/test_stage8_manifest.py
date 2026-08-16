import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_stage8_manifest_contract_when_present() -> None:
    path = ROOT / "artifacts/manifests/stage8_assessment_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    assert manifest["stage"] == 8
    assert manifest["row_counts"]["extractions"] == 3000
    assert manifest["row_counts"]["verifications"] == 3000
    assert manifest["row_counts"]["paired_judgments"] == 1500
    assert manifest["integrity_checks"]["claim_id_coverage_complete"] is True
    assert manifest["integrity_checks"]["common_union_packet_used"] is True
    assert manifest["status"]["study_specific_statistics"] == "complete_postprocessing"
    assert manifest["status"]["study_scope"] == "closed_at_stage8_without_external_or_manual_audit"
    for raw_path, expected in manifest["output_artifact_hashes"].items():
        # Later additive stages append rows to this shared cross-stage registry.
        if Path(raw_path).name == "figure_table_registry.csv":
            continue
        artifact = ROOT / raw_path
        assert artifact.exists(), raw_path
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected
