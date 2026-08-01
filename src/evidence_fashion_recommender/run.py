"""Run context shared by commands and experiment stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cache import ArtifactCache
from .config import AppConfig
from .logging import configure_logging
from .reproducibility import create_run_directory, seed_everything, write_run_manifest


@dataclass
class RunContext:
    config: AppConfig
    run_dir: Path
    cache: ArtifactCache


def start_run(config: AppConfig) -> RunContext:
    seed_everything(config.project.seed, config.project.deterministic)
    run_dir = create_run_directory(config)
    configure_logging(config.run.log_level, run_dir / "logs" / "run.log")
    write_run_manifest(config, run_dir)
    cache = ArtifactCache(config.paths.cache_dir, config.cache.policy)
    return RunContext(config, run_dir, cache)
