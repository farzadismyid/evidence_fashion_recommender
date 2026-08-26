"""Complete Stage-5 table/figure registration and release hashing without model calls."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from evidence_fashion.manifest import sha256_file, utc_timestamp, write_json, write_new_json

ROOT = Path(".runtime/current/final_analysis/final-stage5")
TABLES = Path("artifacts/tables")
FIGURES = Path("artifacts/figures")
RELEASE = Path("artifacts/release")
STAGE5 = Path("artifacts/manifests/final_stage5_manifest.json")


def main() -> None:
    tables = (
        "explanation_record_metrics.csv",
        "explanation_paired_contrasts.csv",
        "recommendation_metrics_with_ci.csv",
        "terminal_failures.csv",
    )
    for name in tables:
        source = ROOT / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, TABLES / f"final_{name}")
    release_manifest = RELEASE / "release_manifest.json"
    if release_manifest.exists():
        raise FileExistsError(release_manifest)
    release_files = sorted(path for path in RELEASE.iterdir() if path.is_file())
    write_new_json(
        release_manifest,
        {
            "schema_version": 1,
            "stage": 5,
            "created_at_utc": utc_timestamp(),
            "files": {str(path): sha256_file(path) for path in release_files},
            "source_tables": {str(ROOT / name): sha256_file(ROOT / name) for name in tables},
            "figures": {
                str(path): sha256_file(path)
                for path in sorted(FIGURES.glob("final_explanation_paired_contrasts.*"))
            },
        },
    )
    stage5 = json.loads(STAGE5.read_text(encoding="utf-8"))
    stage5["release_manifest"] = {
        "path": str(release_manifest),
        "sha256": sha256_file(release_manifest),
    }
    stage5["table_registry"] = {
        str(TABLES / f"final_{name}"): sha256_file(TABLES / f"final_{name}") for name in tables
    }
    stage5["figure_registry"] = {
        str(path): sha256_file(path)
        for path in sorted(FIGURES.glob("final_explanation_paired_contrasts.*"))
    }
    write_json(STAGE5, stage5)


if __name__ == "__main__":
    main()
