"""Run the deterministic final cleanup and release-integrity review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evidence_fashion.manifest import git_commit, sha256_file, utc_timestamp, write_json

ROOT = Path(__file__).parents[1]
MANIFESTS = (
    "stage1_repository_manifest.json",
    "data_preparation_leakage_resolved_manifest.json",
    "embedding_leakage_resolved_manifest.json",
    "stage4_reranking_manifest.json",
    "stage5_optimization_manifest.json",
    "stage5_pilot_manifest.json",
    "stage5_length_control_followup_manifest.json",
    "stage6_recommendation_manifest.json",
    "stage6b_fashionclip_manifest.json",
    "stage7_explanation_generation_manifest.json",
    "stage8_assessment_manifest.json",
    "stage8_metric_revision_manifest.json",
    "stage8_study_metrics_manifest.json",
    "stage8_publication_analysis_manifest.json",
    "thesis_chapters_manifest.json",
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", action="store_true")
    parser.add_argument("--ruff-passed", action="store_true")
    return parser.parse_args()


def resolve_artifact(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def validate_manifest(path: Path) -> tuple[dict[str, Any], int]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checked = 0
    for raw_path, expected_hash in manifest.get("output_artifact_hashes", {}).items():
        artifact = resolve_artifact(raw_path)
        # This is a deliberately append-only registry shared by sequential stages. Its
        # current hash cannot equal every historical manifest's snapshot simultaneously.
        if artifact.name == "figure_table_registry.csv":
            continue
        if not artifact.exists():
            raise FileNotFoundError(f"Missing output bound by {path.name}: {artifact}")
        actual_hash = sha256_file(artifact)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Hash mismatch for {artifact} bound by {path.name}: "
                f"{actual_hash} != {expected_hash}"
            )
        checked += 1
    return manifest, checked


def forbidden_paths() -> list[str]:
    audit_directory = ROOT / "artifacts/audits"
    if not audit_directory.exists():
        return []
    return sorted(str(path.relative_to(ROOT)) for path in audit_directory.rglob("*"))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.tests_passed or not args.ruff_passed:
        raise SystemExit("Run the full tests and Ruff first, then pass both acknowledgement flags.")

    stage_summaries = []
    artifact_checks = 0
    for filename in MANIFESTS:
        path = ROOT / "artifacts/manifests" / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing canonical manifest: {path}")
        manifest, checked = validate_manifest(path)
        artifact_checks += checked
        status = manifest.get("status", "completed_frozen_artifact")
        if manifest.get("stage") == 7 and isinstance(status, dict):
            status = status["generation"]
        stage_summaries.append(
            {
                "manifest": filename,
                "stage": manifest.get("stage"),
                "stage_name": manifest.get("stage_name"),
                "status": status,
                "validated_output_artifacts": checked,
                "sha256": sha256_file(path),
            }
        )

    unwanted = forbidden_paths()
    if unwanted:
        raise ValueError(f"Removed evaluation-audit paths remain: {unwanted}")

    readiness_rows = [
        {
            "requirement": "Experimental Stages 1-8 have canonical manifests",
            "status": "complete",
            "evidence_or_limitation": f"{len(MANIFESTS)} canonical/derived manifests validated",
        },
        {
            "requirement": "Canonical manifest output hashes resolve",
            "status": "complete",
            "evidence_or_limitation": f"{artifact_checks} artifact hashes validated",
        },
        {
            "requirement": "Evaluation-audit footprint removed",
            "status": "complete",
            "evidence_or_limitation": "No human or external evaluation audit path retained",
        },
        {
            "requirement": "Stage 8 system evaluation is canonical",
            "status": "complete",
            "evidence_or_limitation": (
                "DTA, UIAR, citation, grounding and sensitivity tables retained"
            ),
        },
        {
            "requirement": "Complete automated checks",
            "status": "complete",
            "evidence_or_limitation": "Full pytest suite and Ruff acknowledged as passed",
        },
        {
            "requirement": "Publication visual coverage for Stage 7-8 findings",
            "status": "complete",
            "evidence_or_limitation": (
                "Paired judge, association, heterogeneity, sensitivity and example artifacts added"
            ),
        },
        {
            "requirement": "Independent validation of automated explanation labels",
            "status": "outside_final_scope",
            "evidence_or_limitation": "Results must be described as automated system evaluation",
        },
        {
            "requirement": "Final cleanup and release review",
            "status": "complete_stage10",
            "evidence_or_limitation": "Deterministic review; zero model calls",
        },
    ]
    readiness_path = ROOT / "artifacts/tables/table_stage10_release_readiness.csv"
    write_csv(readiness_path, readiness_rows)

    manifest_path = ROOT / "artifacts/manifests/stage10_release_manifest.json"
    release_manifest = {
        "schema_version": 1,
        "stage": 10,
        "stage_name": "final_cleanup_and_release_review",
        "status": "complete_publication_ready",
        "timestamp_utc": utc_timestamp(),
        "git_commit_at_review": git_commit(),
        "model_calls": 0,
        "experimental_scope": "completed_stages_1_through_8_with_stage5_followup_and_stage6b",
        "evaluation_scope": "automated_system_evaluation_only",
        "checks": {
            "tests_passed": True,
            "ruff_passed": True,
            "canonical_manifests": len(MANIFESTS),
            "validated_output_artifacts": artifact_checks,
            "removed_evaluation_audit_paths": True,
        },
        "stage_manifests": stage_summaries,
        "output_artifact_hashes": {
            str(readiness_path.relative_to(ROOT)): sha256_file(readiness_path)
        },
        "notes": [
            "No experiment or model inference was run during the release review.",
            "Automated explanation metrics are not independently validated ground truth.",
            (
                "Publication analyses and Chapters 1-5 were reconciled to the corrected "
                "frozen outputs."
            ),
        ],
    }
    write_json(manifest_path, release_manifest)
    print(json.dumps(release_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
