"""Content-addressed cache for expensive research artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def stable_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_fingerprint(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheRecord:
    namespace: str
    fingerprint: str
    path: Path
    hit: bool


class ArtifactCache:
    """Store artifacts under keys derived from every material input."""

    def __init__(self, root: Path, policy: str = "reuse") -> None:
        self.root = root
        self.policy = policy
        self.root.mkdir(parents=True, exist_ok=True)

    def location(self, namespace: str, inputs: Any, suffix: str) -> CacheRecord:
        fingerprint = stable_fingerprint(inputs)
        path = self.root / namespace / f"{fingerprint}{suffix}"
        hit = self.policy == "reuse" and path.exists()
        return CacheRecord(namespace, fingerprint, path, hit)

    def get_or_create(
        self,
        namespace: str,
        inputs: Any,
        suffix: str,
        builder: Callable[[Path], T],
    ) -> tuple[T | Path, CacheRecord]:
        record = self.location(namespace, inputs, suffix)
        if self.policy == "disabled":
            disabled_record = CacheRecord(namespace, record.fingerprint, record.path, False)
            return builder(record.path), disabled_record
        if record.hit:
            return record.path, record
        record.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = record.path.with_suffix(record.path.suffix + ".partial")
        if temporary.exists():
            if temporary.is_dir():
                shutil.rmtree(temporary)
            else:
                temporary.unlink()
        result = builder(temporary)
        temporary.replace(record.path)
        metadata = {
            "namespace": namespace,
            "fingerprint": record.fingerprint,
            "inputs": inputs,
        }
        record.path.with_suffix(record.path.suffix + ".meta.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
        return result if result is not None else record.path, record
