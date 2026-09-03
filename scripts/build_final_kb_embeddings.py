"""Embed the frozen final knowledge base with its pinned local semantic model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from evidence_fashion.kb_audit import load_canonical_rules
from evidence_fashion.manifest import (
    configuration_hash,
    environment_summary,
    git_commit,
    load_resolved_configuration,
    sha256_file,
    utc_timestamp,
    write_new_json,
)
from evidence_fashion.retrieval import OllamaEmbedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    resolved = load_resolved_configuration(args.config, args.models_config)
    config_hash = configuration_hash(resolved)
    kb_path = Path(config["paths"]["knowledge_base"])
    rules = load_canonical_rules(kb_path)
    run_dir = Path(config["paths"]["embedding_runs"]) / f"final-kb-{config_hash[:12]}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable KB embedding run: {run_dir}")
    embedder = OllamaEmbedder(
        models["embedders"]["qwen3_embedding"],
        endpoint=str(models["inference_defaults"]["endpoint"]),
    )
    vectors = embedder.encode(
        rules["rule_text"].astype(str).tolist(),
        batch_size=int(config["embeddings"]["batch_size"]),
    )
    run_dir.mkdir(parents=True)
    vector_path = run_dir / "rule_embeddings.npy"
    with vector_path.open("xb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    metadata_path = run_dir / "rule_metadata.jsonl"
    with metadata_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in rules.to_dict("records"):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": 2,
        "stage": "final_stage1",
        "run_id": run_dir.name,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "configuration_hash": config_hash,
        "input_artifact_hashes": {str(kb_path): sha256_file(kb_path)},
        "output_artifact_hashes": {
            str(vector_path): sha256_file(vector_path),
            str(metadata_path): sha256_file(metadata_path),
        },
        "models": {"qwen3_embedding": models["embedders"]["qwen3_embedding"]},
        "row_counts": {"rules": len(rules), "embedding_dimension": int(vectors.shape[1])},
        "failure_counts": {"embedding_failures": 0},
        "seed": config["project"]["random_seed"],
        "command": "python scripts/build_final_kb_embeddings.py",
        "environment": environment_summary(),
    }
    write_new_json(manifest_path, manifest)
    print(json.dumps({"run_id": run_dir.name, **manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
