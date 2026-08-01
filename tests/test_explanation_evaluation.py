from evidence_fashion_recommender.evaluation.explanations import evaluate_explanation


def test_citation_validation() -> None:
    result = evaluate_explanation(
        "The polished shape complements the dress [R001], unlike [R999].",
        {"R001"},
        "A polished shoe complements a formal dress.",
    )
    assert result["citation_presence"] == 1
    assert result["citation_correctness"] == 0
    assert result["invalid_citation_count"] == 1
