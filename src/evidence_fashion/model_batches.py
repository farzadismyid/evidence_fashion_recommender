"""Stage 4 sequential model-batch policy and runtime guard."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

EXPECTED_BATCHES = (
    ("gemma_generation", "gemma4:12b", "generator"),
    ("llama_generation", "llama3.1:8b-instruct-q8_0", "generator"),
    ("ministral_generation", "ministral-3:14b-instruct-2512-q4_K_M", "generator"),
    ("qwen_claim_extraction", "qwen3.5:9b", "claim_extraction"),
    ("phi_claim_verification", "phi4:14b", "claim_verification"),
)
OPTIONAL_JUDGE = ("deepseek_blind_judging", "deepseek-r1:14b", "blind_judge")


@dataclass(frozen=True)
class Batch:
    batch_id: str
    model_id: str
    role: str
    batch_scope: str


def stage4_batches(
    policy: Mapping[str, Any], *, approve_optional_judge: bool = False
) -> list[Batch]:
    """Validate the frozen declaration and return only the allowed complete-batch schedule."""
    if policy.get("schema_version") != 1:
        raise ValueError("Stage 4 batch policy must use schema_version 1.")
    if policy.get("execution_mode") != "complete_batch_then_unload":
        raise ValueError("Stage 4 requires complete_batch_then_unload execution.")
    if policy.get("max_loaded_models") != 1:
        raise ValueError("Stage 4 permits exactly one loaded model.")
    declared = policy.get("batches")
    expected_ids = [item[0] for item in EXPECTED_BATCHES]
    if not isinstance(declared, list) or [row.get("id") for row in declared] != expected_ids:
        raise ValueError("Stage 4 batches do not match the frozen order.")
    expected_models = [item[1] for item in EXPECTED_BATCHES]
    if [row.get("model_id") for row in declared] != expected_models:
        raise ValueError("Stage 4 batch model identities do not match the frozen order.")
    if [row.get("role") for row in declared] != [item[2] for item in EXPECTED_BATCHES]:
        raise ValueError("Stage 4 batch roles do not match the frozen order.")
    if any(row.get("batch_scope") != "all_records_for_model" for row in declared):
        raise ValueError("Stage 4 prohibits record-by-record model switching.")
    if policy.get("stop_after") != "phi_claim_verification":
        raise ValueError("Stage 4 must stop after claim verification for inspection.")
    optional = policy.get("optional_blind_judge")
    if not isinstance(optional, Mapping) or optional.get("model_id") != OPTIONAL_JUDGE[1]:
        raise ValueError("Stage 4 optional judge declaration is incomplete.")
    if optional.get("requires_separate_approval") is not True:
        raise ValueError("Stage 4 blind judging requires separate approval.")
    batches = [
        Batch(
            batch_id=row["id"],
            model_id=row["model_id"],
            role=row["role"],
            batch_scope=row["batch_scope"],
        )
        for row in declared
    ]
    if approve_optional_judge:
        batches.append(Batch(*OPTIONAL_JUDGE, "all_records_for_model"))
    return batches


class SequentialBatchExecutor:
    """Execute whole-model batches with mandatory unload before a later model may start."""

    def __init__(self, batches: Sequence[Batch]) -> None:
        self.batches = list(batches)
        self.events: list[dict[str, Any]] = []
        self._loaded_model: str | None = None

    def execute(
        self,
        run_batch: Callable[[Batch], Mapping[str, Any] | None],
        unload_model: Callable[[str], None],
    ) -> list[dict[str, Any]]:
        for batch in self.batches:
            if self._loaded_model is not None:
                raise RuntimeError("Cannot begin a batch while another model remains loaded.")
            self._loaded_model = batch.model_id
            self.events.append(
                {
                    "event": "batch_started",
                    "batch_id": batch.batch_id,
                    "model_id": batch.model_id,
                }
            )
            try:
                summary = dict(run_batch(batch) or {})
                self.events.append(
                    {
                        "event": "batch_completed",
                        "batch_id": batch.batch_id,
                        "model_id": batch.model_id,
                        "summary": summary,
                    }
                )
            except Exception:
                self.events.append(
                    {
                        "event": "batch_failed",
                        "batch_id": batch.batch_id,
                        "model_id": batch.model_id,
                    }
                )
                try:
                    unload_model(batch.model_id)
                    self.events.append(
                        {"event": "model_unloaded_after_failure", "model_id": batch.model_id}
                    )
                finally:
                    self._loaded_model = None
                raise
            unload_model(batch.model_id)
            self._loaded_model = None
            self.events.append({"event": "model_unloaded", "model_id": batch.model_id})
        return list(self.events)
