import pytest

from evidence_fashion.verification_corrections import (
    enforce_trace_implies_full_kb,
    trace_support_implies_full_kb,
)


def _record(*, full_ids: list[str] | None = None) -> dict:
    return {
        "exact_trace_rule_ids": ["K001"],
        "full_kb_candidate_rule_ids": full_ids or ["K001", "K002"],
        "claims": [
            {"claim_id": "C1", "trace_support": "supported", "full_kb_support": "not_supported"}
        ],
    }


def test_trace_support_implies_full_kb_is_corrected_when_packet_contains_trace() -> None:
    record = _record()
    assert enforce_trace_implies_full_kb([record]) == 1
    assert record["claims"][0]["full_kb_support"] == "supported"
    assert trace_support_implies_full_kb(record)


def test_trace_support_correction_rejects_a_noncontained_trace_packet() -> None:
    with pytest.raises(ValueError, match="exact-trace rule"):
        enforce_trace_implies_full_kb([_record(full_ids=["K002"])])
