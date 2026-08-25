import pytest

from evidence_fashion.extraction_contracts import validate_atomic_claims


def _payload() -> dict:
    return {
        "claims": [
            {
                "claim_id": "C1",
                "claim_text": "The top balances the trousers.",
                "claim_type": "styling_relation",
            },
            {"claim_id": "C2", "claim_text": "The look is polished.", "claim_type": "formality"},
        ]
    }


def test_atomic_claim_contract_accepts_ordered_evidence_independent_claims() -> None:
    assert [claim["claim_id"] for claim in validate_atomic_claims(
        _payload(), claim_types=["styling_relation", "formality"]
    )] == ["C1", "C2"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload["claims"][1].update({"claim_id": "C3"}),
        lambda payload: payload["claims"][1].update(
            {"claim_text": "The top balances the trousers."}
        ),
        lambda payload: payload["claims"][1].update({"support": "supported"}),
    ],
)
def test_atomic_claim_contract_rejects_id_drift_duplicates_and_support_judgement(mutator) -> None:
    payload = _payload()
    mutator(payload)
    with pytest.raises(ValueError):
        validate_atomic_claims(payload, claim_types=["styling_relation", "formality"])
