import pytest

from evidence_fashion.verification_contracts import validate_verdicts


def _payload() -> dict:
    return {
        "claims": [
            {
                "claim_id": "C1",
                "trace_support": "supported",
                "full_kb_support": "supported",
                "common_reference_support": "supported",
                "citation_entailment": "N/A",
            }
        ]
    }


def test_verdict_contract_accepts_frozen_labels() -> None:
    verdicts = validate_verdicts(
        _payload(),
        claims=[{"claim_id": "C1"}],
        common_reference_eligible={"C1": True},
        valid_citations_present=False,
    )
    assert verdicts[0]["trace_support"] == "supported"


@pytest.mark.parametrize(
    "field,value",
    [("claim_id", "C2"), ("common_reference_support", "N/A"), ("citation_entailment", "entails")],
)
def test_verdict_contract_rejects_id_and_na_violations(field: str, value: str) -> None:
    payload = _payload()
    payload["claims"][0][field] = value
    with pytest.raises(ValueError):
        validate_verdicts(
            payload,
            claims=[{"claim_id": "C1"}],
            common_reference_eligible={"C1": True},
            valid_citations_present=False,
        )
