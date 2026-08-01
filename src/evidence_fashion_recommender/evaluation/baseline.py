"""Immutable before-baseline snapshot for robustness comparisons."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from ..cache import file_fingerprint


def freeze_baseline(source: Path, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(
            f"Baseline snapshot already exists and will not be overwritten: {destination}"
        )
    shutil.copytree(source, destination)
    artifacts = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "sha256": file_fingerprint(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": str(source),
        "immutable_by_convention": True,
        "artifacts": artifacts,
    }
    manifest_path = destination / "BASELINE_SNAPSHOT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
