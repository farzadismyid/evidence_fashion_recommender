"""Complete Stage-5 table/figure registration and release hashing without model calls."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from evidence_fashion.manifest import sha256_file, utc_timestamp, write_json, write_new_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    root = Path(paths["final_analysis_runs"]) / "final-stage5"
    tables_dir = Path(paths["tables"])
    figures_dir = Path(paths["figures"])
    release_dir = Path(paths["release"])
    stage5_path = Path(paths["stage1_manifest"]).with_name("final_stage5_manifest.json")
    tables = (
        "explanation_record_metrics.csv",
        "explanation_paired_contrasts.csv",
        "recommendation_metrics_with_ci.csv",
        "terminal_failures.csv",
    )
    for name in tables:
        source = root / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, tables_dir / f"final_{name}")
    release_manifest = release_dir / "release_manifest.json"
    if release_manifest.exists():
        raise FileExistsError(release_manifest)
    release_files = sorted(path for path in release_dir.iterdir() if path.is_file())
    write_new_json(
        release_manifest,
        {
            "schema_version": 1,
            "stage": 5,
            "created_at_utc": utc_timestamp(),
            "files": {str(path): sha256_file(path) for path in release_files},
            "source_tables": {str(root / name): sha256_file(root / name) for name in tables},
            "figures": {
                str(path): sha256_file(path)
                for path in sorted(figures_dir.glob("final_explanation_paired_contrasts.*"))
            },
        },
    )
    stage5 = json.loads(stage5_path.read_text(encoding="utf-8"))
    stage5["release_manifest"] = {
        "path": str(release_manifest),
        "sha256": sha256_file(release_manifest),
    }
    stage5["table_registry"] = {
        str(tables_dir / f"final_{name}"): sha256_file(tables_dir / f"final_{name}")
        for name in tables
    }
    stage5["figure_registry"] = {
        str(path): sha256_file(path)
        for path in sorted(figures_dir.glob("final_explanation_paired_contrasts.*"))
    }
    write_json(stage5_path, stage5)


if __name__ == "__main__":
    main()
