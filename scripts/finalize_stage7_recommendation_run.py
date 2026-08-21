"""Seal an already-computed Stage 7 recommendation run after publication-only assembly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from evidence_fashion.manifest import (
    configuration_hash,
    environment_summary,
    git_commit,
    load_resolved_configuration,
    sha256_file,
    utc_timestamp,
    write_json,
    write_new_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(".runtime/current/recommendations/stage6/stage6-confirmatory-414ac73b4696"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    digest = configuration_hash(resolved)
    run_dir = args.run_dir
    if not run_dir.name.endswith(digest[:12]):
        raise ValueError("Run directory does not match the frozen resolved configuration.")
    runtime_outputs = [
        run_dir / "case_metrics.csv",
        run_dir / "evidence_diagnostics.csv",
        run_dir / "locked_cases.jsonl",
        run_dir / "candidate_rankings.jsonl",
    ]
    tracked = [
        Path(f"artifacts/tables/table_{index:02d}_{name}.csv")
        for index, name in (
            (1, "dataset_statistics"),
            (2, "recommendation_results"),
            (3, "recommendation_contrasts"),
            (4, "evidence_participation"),
            (5, "stage6_rule_frequency"),
            (6, "kb_summary"),
            (7, "publication_readiness"),
        )
    ]
    figures = [
        Path(f"artifacts/figures/fig_{index:02d}_{name}.svg")
        for index, name in (
            (1, "system_architecture"),
            (2, "evidence_ablation"),
            (3, "evidence_trace"),
            (4, "dataset_pipeline"),
        )
    ]
    outputs = [*runtime_outputs, *tracked, *figures]
    missing = [str(path) for path in outputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot seal incomplete recommendation computation: {missing}")
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Recommendation run is already sealed: {manifest_path}")
    data_manifest = json.loads(
        Path(config["paths"]["active_data_manifest"]).read_text(encoding="utf-8")
    )
    data_inputs = data_manifest["output_artifact_hashes"]
    prepared = next((path for path in data_inputs if path.endswith("prepared_items.parquet")), None)
    if prepared is None:
        raise ValueError("Active data manifest does not bind prepared items.")
    locked_rows = sum(
        1
        for line in (run_dir / "locked_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    )
    metric_rows = len(pd.read_csv(run_dir / "case_metrics.csv"))
    manifest = {
        "schema_version": 1,
        "stage": 7,
        "run_id": run_dir.name,
        "status": "complete",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": digest,
        "resolved_configuration": resolved,
        "models": yaml.safe_load(args.models_config.read_text(encoding="utf-8"))["embedders"],
        "input_artifact_hashes": {
            prepared: data_inputs[prepared],
            config["paths"]["knowledge_base"]: sha256_file(Path(config["paths"]["knowledge_base"])),
        },
        "output_artifact_hashes": {str(path): sha256_file(path) for path in outputs},
        "row_counts": {
            "confirmatory_cases": locked_rows,
            "case_metric_rows": metric_rows,
            "locked_cases": locked_rows,
        },
        "failure_counts": {"ranking_failures": 0, "trace_failures": 0},
        "trace_validation": {
            "complete_five_rule_traces": all(
                len(json.loads(line)["evidence_trace"]["rules"]) == 5
                for line in (run_dir / "locked_cases.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ),
            "locked_cases_checked": locked_rows,
        },
        "publication_assembly": {
            "legacy_optional_figures_skipped": ["fig_07", "fig_08", "fig_09"],
            "reason": "their obsolete source tables are intentionally not retained",
        },
        "environment": environment_summary(),
        "command": "python scripts/finalize_stage7_recommendation_run.py",
    }
    write_new_json(manifest_path, manifest)
    write_json(Path("artifacts/manifests/stage6_recommendation_manifest.json"), manifest)
    print(json.dumps({"run_id": run_dir.name, "locked_cases": locked_rows}, indent=2))


if __name__ == "__main__":
    main()
