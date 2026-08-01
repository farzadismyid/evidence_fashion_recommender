"""Fail-closed provenance freeze for final_eval_v2 after validation selection."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .cache import file_fingerprint, stable_fingerprint
from .reproducibility import environment_manifest, git_revision, ollama_manifest


def _load_selection(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing {name} selection artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("selected_on") != "validation":
        raise ValueError(f"{name} selection must record selected_on=validation.")
    return value


def create_final_eval_v2_freeze(
    *,
    destination: Path,
    resolved_config: Path,
    fusion_selection: Path,
    reranking_selection: Path,
    hybrid_selection: Path,
    schedules: list[Path],
    cases: list[Path],
    knowledge_base: Path,
    dependency_lock: Path,
    prompt_files: list[Path],
    command_list: list[str],
    expected_stage1_packet_hash: str,
    gate_definition: dict[str, Any],
    source_state: dict[str, Any] | None = None,
) -> Path:
    """Create the immutable pre-test manifest only after all selections exist."""

    if destination != Path("outputs/final_eval_v2/freeze"):
        raise ValueError("Freeze destination must be outputs/final_eval_v2/freeze.")
    source = source_state or git_revision()
    if not source.get("commit") or source.get("dirty") is not False:
        raise ValueError("Final freeze requires a clean committed Git source state.")
    fusion = _load_selection(fusion_selection, "fusion")
    reranking = _load_selection(reranking_selection, "reranking")
    hybrid = _load_selection(hybrid_selection, "Hybrid")
    if hybrid.get("candidate_type") != "hybrid" or int(hybrid.get("item_count", 0)) <= 0:
        raise ValueError("Final Hybrid selection must contain item evidence.")
    if hybrid.get("stage1_packet_protocol") != "final_eval_v2_selected":
        raise ValueError("Final Hybrid selection cannot use legacy-only evidence packets.")
    if hybrid.get("stage1_packet_hash") != expected_stage1_packet_hash:
        raise ValueError("Hybrid selection does not match the selected Stage 1 packet hash.")
    inputs = [resolved_config, *schedules, *cases, knowledge_base, dependency_lock, *prompt_files]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise ValueError(f"Freeze inputs do not exist: {missing}")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "protocol": "final_eval_v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": source,
        "config": {"path": str(resolved_config), "sha256": file_fingerprint(resolved_config)},
        "selections": {
            "fusion": {
                "path": str(fusion_selection),
                "sha256": file_fingerprint(fusion_selection),
                "value": fusion,
            },
            "reranking": {
                "path": str(reranking_selection),
                "sha256": file_fingerprint(reranking_selection),
                "value": reranking,
            },
            "hybrid": {
                "path": str(hybrid_selection),
                "sha256": file_fingerprint(hybrid_selection),
                "value": hybrid,
            },
        },
        "stage1_packet_hash": expected_stage1_packet_hash,
        "gate_definition": gate_definition,
        "input_hashes": {str(path): file_fingerprint(path) for path in inputs},
        "prompt_hashes": {str(path): file_fingerprint(path) for path in prompt_files},
        "commands": command_list,
        "commands_hash": stable_fingerprint(command_list),
        "environment": environment_manifest(),
        "ollama_models": ollama_manifest(),
    }
    path = destination / "FINAL_FREEZE_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
