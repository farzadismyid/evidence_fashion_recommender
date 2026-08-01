"""Deterministic setup and auditable run manifests."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def git_revision() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": revision, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git": git_revision(),
    }
    try:
        import torch

        manifest["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    except ImportError:
        manifest["torch"] = None
    return manifest


def ollama_manifest() -> list[dict[str, str]]:
    try:
        output = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    models = []
    for line in output[1:]:
        parts = line.split()
        if len(parts) >= 2:
            models.append({"name": parts[0], "digest": parts[1]})
    return models


def create_run_directory(config: AppConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.outputs_dir / "runs" / f"{config.run.experiment_name}_{timestamp}"
    for name in ["logs", "embeddings", "indexes", "predictions", "metrics", "figures", "reports"]:
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_manifest(config: AppConfig, run_dir: Path) -> None:
    import yaml

    resolved = config.model_dump(mode="json")
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    manifest = environment_manifest()
    manifest["ollama_models"] = ollama_manifest()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
