from evidence_fashion_recommender.evaluation.claim_evaluation import (
    parse_claim_verifications,
)
from evidence_fashion_recommender.evaluation.stage4d_v2 import (
    _canonical_hash,
    _local_parse,
)


def test_local_repair_only_closes_complete_json_tokens() -> None:
    raw = '{"verifications":[{"claim_id":"c1","support_label":"unsupported"}]'

    parsed = _local_parse(raw, lambda value: parse_claim_verifications(value, {"c1"}))

    assert parsed is not None
    assert parsed[0]["claim_id"] == "c1"
    assert parsed[0]["support_label"] == "unsupported"
    assert _local_parse('{"verifications":[{"claim_id":"c1', lambda value: value) is None


def test_canonical_hash_is_order_independent_and_change_sensitive() -> None:
    original = {"key": "row-1", "status": "complete"}

    assert _canonical_hash(original) == _canonical_hash(
        {"status": "complete", "key": "row-1"}
    )
    assert _canonical_hash(original) != _canonical_hash({**original, "status": "N/A"})
