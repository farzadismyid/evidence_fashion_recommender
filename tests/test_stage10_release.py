from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(reason="Stage 10 belongs to the archived experiment")

ROOT = Path(__file__).parents[1]


def test_stage10_release_manifest_and_readiness_hash() -> None:
    manifest_path = ROOT / "artifacts/manifests/stage10_release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == 10
    assert manifest["model_calls"] == 0
    assert manifest["status"] == "complete_publication_ready"
    assert manifest["checks"]["removed_evaluation_audit_paths"] is True
    for raw_path, expected_hash in manifest["output_artifact_hashes"].items():
        path = ROOT / raw_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash


def test_release_readiness_records_stage10_completion() -> None:
    path = ROOT / "artifacts/tables/table_stage10_release_readiness.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    statuses = {row["requirement"]: row["status"] for row in rows}
    assert statuses["Final cleanup and release review"] == "complete_stage10"
    assert statuses["Stage 8 system evaluation is canonical"] == "complete"


def test_evaluation_audit_paths_are_absent() -> None:
    assert not (ROOT / "artifacts/audits").exists()
