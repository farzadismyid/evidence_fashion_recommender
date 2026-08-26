from evidence_fashion.final_analysis import (
    bootstrap_paired_difference,
    paired_complete_rows,
    record_metrics,
)


def _record(case_id: str, condition: str, support: str) -> dict:
    return {
        "case_id": case_id,
        "generator_model_id": "model",
        "condition": condition,
        "claims": [
            {
                "trace_support": support,
                "full_kb_support": support,
                "common_reference_support": "N/A",
                "citation_entailment": "N/A",
            }
        ],
    }


def test_final_metrics_use_final_stage4_fields() -> None:
    values = record_metrics(
        _record("a", "no_rag", "supported"), {"explanation": "one two three four"}
    )
    assert values["trace_support_rate"] == 1.0
    assert values["full_kb_support_rate"] == 1.0
    assert values["unsupported_item_fact_rate"] is None


def test_pairing_requires_both_conditions() -> None:
    rows = [
        _record("a", "no_rag", "not_supported"),
        _record("a", "rule_rag", "supported"),
        _record("b", "no_rag", "supported"),
    ]
    pairs = paired_complete_rows(rows)
    for pair in pairs:
        pair["no_rag"]["trace_support_rate"] = 0.0
        pair["rule_rag"]["trace_support_rate"] = 1.0
    assert len(pairs) == 1
    assert (
        bootstrap_paired_difference(
            pairs,
            "trace_support_rate",
            replicates=20,
            confidence_level=0.95,
            seed=42,
            aggregate_generators_by_case=True,
        )["n"]
        == 1
    )
