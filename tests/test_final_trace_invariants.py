from evidence_fashion.final_contracts import (
    build_rule_rag_evidence_packet,
    canonical_json_sha256,
    reproduce_evidence_score,
)


def _trace() -> dict:
    return {
        "candidate_id": "candidate-1",
        "evidence_score": 0.885,
        "rules": [
            {"rule_id": "K001", "weighted_contribution": 0.8},
            {"rule_id": "K002", "weighted_contribution": 0.9},
        ],
    }


def test_stored_trace_reproduces_its_score() -> None:
    assert reproduce_evidence_score(
        _trace(), {"score_max_weight": 0.7, "score_mean_weight": 0.3}
    ) == 0.885


def test_rule_rag_packet_is_the_exact_locked_trace_without_retrieval() -> None:
    trace = _trace()
    packet, packet_hash = build_rule_rag_evidence_packet(trace)
    assert packet == trace
    assert packet_hash == canonical_json_sha256(trace)
    assert packet["rules"] is not trace["rules"]


def test_empty_trace_requires_zero_evidence_score() -> None:
    trace = {"candidate_id": "candidate-1", "evidence_score": 0.0, "rules": []}
    assert reproduce_evidence_score(
        trace, {"score_max_weight": 0.7, "score_mean_weight": 0.3}
    ) == 0.0
