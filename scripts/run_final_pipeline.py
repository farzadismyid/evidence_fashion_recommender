"""Check prerequisites and reproduce the final pipeline in an isolated output directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="Validate local model prerequisites.")
    action.add_argument("--run", action="store_true", help="Run every final stage in order.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts-config", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument(
        "--run-root",
        type=Path,
        help=(
            "New output directory; defaults to a timestamped directory under "
            ".runtime/reproductions/."
        ),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def required_ollama_models(models: dict[str, Any]) -> set[str]:
    return {
        str(models["embedders"]["qwen3_embedding"]["model_id"]),
        *(str(row["model_id"]) for row in models["generators"]["roster"]),
        str(models["extractor"]["model_id"]),
        str(models["verifier"]["model_id"]),
    }


def check_prerequisites(models: dict[str, Any]) -> None:
    endpoint = str(models["inference_defaults"]["endpoint"]).rstrip("/")
    try:
        with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except OSError as error:
        raise RuntimeError(f"Cannot reach Ollama at {endpoint}.") from error
    available = {str(row.get("name")) for row in payload.get("models", [])}
    missing = sorted(required_ollama_models(models).difference(available))
    if missing:
        raise RuntimeError(f"Ollama is missing required models: {', '.join(missing)}")

    from evidence_fashion.retrieval import CLIPEmbedder, MiniLMEmbedder

    try:
        MiniLMEmbedder(models["embedders"]["minilm"])
        CLIPEmbedder(models["embedders"]["clip"])
    except OSError as error:
        raise RuntimeError(
            "Pinned Hugging Face model files are not available in the local cache."
        ) from error


def isolated_config(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    result = dict(config)
    paths = dict(config["paths"])
    runtime = run_root / "runtime"
    outputs = run_root / "artifacts"
    manifest_dir = outputs / "manifests"
    paths.update(
        {
            "runtime_root": str(runtime),
            "data_runs": str(runtime / "data"),
            "embedding_runs": str(runtime / "embeddings"),
            "recommendation_runs": str(runtime / "recommendations"),
            "explanation_runs": str(runtime / "explanations"),
            "extraction_runs": str(runtime / "extraction"),
            "verification_runs": str(runtime / "verification"),
            "final_analysis_runs": str(runtime / "final_analysis"),
            "manifests": str(manifest_dir),
            "tables": str(outputs / "tables"),
            "figures": str(outputs / "figures"),
            "reports": str(run_root / "reports"),
            "release": str(outputs / "release"),
            "category_audit_table": str(outputs / "tables" / "final_stage1_category_audit.csv"),
            "active_data_manifest": str(manifest_dir / "final_data_preparation_manifest.json"),
            "active_embedding_manifest": str(manifest_dir / "final_embedding_manifest.json"),
            "stage1_manifest": str(manifest_dir / "final_stage1_preflight_manifest.json"),
        }
    )
    result["paths"] = paths
    return result


def run_stage(script: str, config: Path, models: Path, prompts: Path) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), "--config", str(config)]
    if script != "finalize_stage5_release.py":
        command.extend(["--models-config", str(models)])
    if script in {
        "finalize_stage1_preflight.py",
        "run_final_recommendations.py",
        "run_final_explanations.py",
        "run_final_claim_extraction.py",
        "run_final_claim_verification.py",
        "run_final_integrity_check.py",
        "run_final_stage5_analysis.py",
    }:
        command.extend(["--prompts-config", str(prompts)])
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    models_path = args.models_config.resolve()
    prompts_path = args.prompts_config.resolve()
    config = load_yaml(config_path)
    models = load_yaml(models_path)
    check_prerequisites(models)
    print("Prerequisite check passed.")
    if args.check:
        return

    run_root = args.run_root or (
        ROOT
        / ".runtime"
        / "reproductions"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    run_root = run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite reproduction output: {run_root}")
    run_root.mkdir(parents=True)
    isolated_path = run_root / "experiment.yaml"
    isolated_path.write_text(
        yaml.safe_dump(isolated_config(config, run_root), sort_keys=False), encoding="utf-8"
    )
    for script in (
        "prepare_data.py",
        "build_embeddings.py",
        "build_final_kb_embeddings.py",
        "run_final_validation_sensitivity.py",
        "run_final_integrity_check.py",
        "finalize_stage1_preflight.py",
        "run_final_recommendations.py",
        "run_final_explanations.py",
        "run_final_claim_extraction.py",
        "run_final_claim_verification.py",
        "run_final_stage5_analysis.py",
        "finalize_stage5_release.py",
    ):
        run_stage(script, isolated_path, models_path, prompts_path)
    print(f"Reproduction complete: {run_root}")


if __name__ == "__main__":
    main()
