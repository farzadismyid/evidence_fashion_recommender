"""Pilot the shared 75-word instruction without modifying the frozen original Stage 5."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evidence_fashion.explanation import (
    OllamaClient,
    build_no_rag_prompt,
    text_sha256,
    word_count,
)
from evidence_fashion.manifest import (
    configuration_hash,
    environment_summary,
    git_commit,
    load_resolved_configuration,
    sha256_file,
    utc_timestamp,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()


def pilot_records(manifest: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    paths = [
        (Path(path), digest)
        for path, digest in manifest["output_artifact_hashes"].items()
        if path.endswith("pilot_records.jsonl")
    ]
    if len(paths) != 1:
        raise ValueError("Original Stage 5 manifest must bind one pilot_records.jsonl.")
    path, digest = paths[0]
    if not path.exists() or sha256_file(path) != digest:
        raise ValueError("Original Stage 5 pilot records are missing or hash-mismatched.")
    return path, read_jsonl(path)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    digest = configuration_hash(resolved)
    limit = int(config["stage7"]["no_rag_word_limit"])
    original_manifest_path = Path("artifacts/manifests/stage5_pilot_manifest.json")
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    original_records_path, original = pilot_records(original_manifest)
    if len(original) != 150:
        raise ValueError("Length-control follow-up expects the frozen 50-case x 3-generator pilot.")

    runtime_root = args.runtime_root or Path(config["paths"]["runtime_root"])
    run_dir = runtime_root / "stage5" / f"stage5-length-control-{digest[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    progress = run_dir / "progress.jsonl"
    final = run_dir / "length_control_records.jsonl"
    if final.exists() and not args.resume:
        raise FileExistsError(f"Immutable length-control pilot exists: {final}")
    existing_history = read_jsonl(progress) if progress.exists() else []
    existing_by_key = {
        (row["case_id"], row["generator"]): row for row in existing_history
    }
    existing = list(existing_by_key.values())
    completed = {(row["case_id"], row["generator"]) for row in existing}
    client = OllamaClient(models["generation_defaults"])
    output = list(existing)
    for old in original:
        key = (old["case_id"], old["generator"])
        if key in completed:
            continue
        case = {
            "request": old["A"]["user_request"],
            "query_item_minimal_name": old["A"]["query_item_minimal_name"],
            "locked_candidate_minimal_name": old["A"]["locked_item_minimal_name"],
        }
        prompt = build_no_rag_prompt(case, limit)
        result = client.generate(old["generator"], prompt)
        record = {
            "case_id": old["case_id"],
            "generator": old["generator"],
            "prompt": prompt,
            "prompt_sha256": text_sha256(prompt),
            "requested_word_limit": limit,
            "output_text": result.text,
            "output_sha256": text_sha256(result.text),
            "word_count": word_count(result.text),
            "word_limit_violation": word_count(result.text) > limit,
            "latency_seconds": result.latency_seconds,
            "comparison_rule_rag_word_count": old["rule_rag"]["metrics"]["word_count"],
            "comparison_rule_rag_output_sha256": text_sha256(old["rule_rag"]["output"]),
        }
        append_jsonl(progress, record)
        output.append(record)
        completed.add(key)
        if len(output) % 10 == 0:
            print(f"length-control pilot {len(output)}/{len(original)}", flush=True)
    if len(output) != len(original):
        raise RuntimeError("Length-control pilot is incomplete.")
    if not final.exists():
        with final.open("x", encoding="utf-8", newline="\n") as handle:
            for record in output:
                handle.write(
                    json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
                )

    frame = pd.DataFrame(output)
    rows = []
    for generator, group in frame.groupby("generator", sort=True):
        rows.append(
            {
                "generator": generator,
                "pairs": len(group),
                "no_rag_mean_words": group["word_count"].mean(),
                "rule_rag_mean_words": group["comparison_rule_rag_word_count"].mean(),
                "mean_paired_word_difference": (
                    group["word_count"] - group["comparison_rule_rag_word_count"]
                ).mean(),
                "no_rag_word_limit_violations": int(group["word_limit_violation"].sum()),
                "mean_latency_seconds": group["latency_seconds"].mean(),
            }
        )
    rows.append(
        {
            "generator": "all_generators",
            "pairs": len(frame),
            "no_rag_mean_words": frame["word_count"].mean(),
            "rule_rag_mean_words": frame["comparison_rule_rag_word_count"].mean(),
            "mean_paired_word_difference": (
                frame["word_count"] - frame["comparison_rule_rag_word_count"]
            ).mean(),
            "no_rag_word_limit_violations": int(frame["word_limit_violation"].sum()),
            "mean_latency_seconds": frame["latency_seconds"].mean(),
        }
    )
    summary = pd.DataFrame(rows)
    runtime_table = run_dir / "length_control_summary.csv"
    tracked_table = Path("artifacts/tables/table_stage5_length_control_followup.csv")
    summary.to_csv(runtime_table, index=False)
    summary.to_csv(tracked_table, index=False)
    manifest = {
        "schema_version": 1,
        "stage": "5_followup_length_control",
        "status": "complete",
        "run_id": run_dir.name,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": digest,
        "resolved_followup": {
            "no_rag_prompt_template": config["stage7"]["no_rag_prompt_template"],
            "shared_word_limit": limit,
            "rule_rag_generation_reused": True,
        },
        "input_artifact_hashes": {
            str(original_manifest_path): sha256_file(original_manifest_path),
            str(original_records_path): sha256_file(original_records_path),
        },
        "output_artifact_hashes": {
            str(final): sha256_file(final),
            str(runtime_table): sha256_file(runtime_table),
            str(tracked_table): sha256_file(tracked_table),
        },
        "row_counts": {"cases": 50, "generator_case_pairs": len(output)},
        "models": {"generators": models["generators"]},
        "environment": environment_summary(),
        "notes": (
            "Additive follow-up only; the original Stage 5 optimisation and pilot remain frozen."
        ),
    }
    runtime_manifest = run_dir / "manifest.json"
    write_json(runtime_manifest, manifest)
    write_json(
        Path("artifacts/manifests/stage5_length_control_followup_manifest.json"),
        manifest,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
