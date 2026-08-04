from pathlib import Path

import pandas as pd

from evidence_fashion_recommender.cache import ArtifactCache
from evidence_fashion_recommender.evaluation.stage45_v2 import (
    _extract_with_recovery,
    _judge_with_recovery,
    _key,
    _verify_with_recovery,
    length_compliance,
    run_claim_extraction_v2,
    run_claim_verification_v2,
)


class _Extractor:
    model_id = "extractor@v2"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return '{"claims":[{"claim_id":"C1","claim":"A styling claim","claim_type":"other"}]}'


class _MalformedThenRepairExtractor:
    model_id = "repair-extractor@v2"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls <= 3:
            return '{"claims":[{"claim_id":"C1"'
        return '{"claims":[{"claim_id":"C1","claim":"Preserved claim","claim_type":"other"}]}'


class _VerifierThatMustNotRun:
    model_id = "verifier@v2"

    def generate(self, prompt: str) -> str:
        raise AssertionError("failed extraction rows must not call the verifier")


class _MalformedThenRepairVerifier:
    model_id = "repair-verifier@v2"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls <= 3:
            return '{"verifications":[{"claim_id":"C1"'
        return (
            '{"verifications":[{"claim_id":"C1","support_label":"unsupported",'
            '"supporting_rule_ids":[],"citation_entails_claim":null,'
            '"brief_reason":"Preserved reason"}]}'
        )


class _MismatchedClaimVerifier:
    model_id = "mismatched-verifier@v2"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return (
            '{"verifications":[{"claim_id":"WRONG","support_label":"unsupported",'
            '"supporting_rule_ids":[],"citation_entails_claim":null,'
            '"brief_reason":"Wrong ID"}]}'
        )


class _MalformedThenRepairJudge:
    model_id = "repair-judge@v2"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls <= 3:
            return '{"input_consistency":5'
        return (
            '{"input_consistency":5,"general_quality":4,"clarity":4,"specificity":3,'
            '"hallucination_risk":5,"evidence_misuse":5,"brief_reason":"Preserved"}'
        )


class _InvalidScoreJudge:
    model_id = "invalid-score-judge@v2"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return (
            '{"input_consistency":9,"general_quality":4,"clarity":4,"specificity":3,'
            '"hallucination_risk":5,"evidence_misuse":5,"brief_reason":"Invalid"}'
        )


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_case_id": f"T{index}",
                "grounding_variant": "no_rag",
                "generation_model": "generator@v2",
                "generation_protocol": "final_eval_v2",
                "generated_explanation": "word " * words,
            }
            for index, words in enumerate((35, 36))
        ]
    )


def test_length_compliance_is_separate_and_does_not_change_text() -> None:
    explanations = _rows()
    original = explanations["generated_explanation"].copy()
    rows, summary = length_compliance(explanations)
    assert rows["word_count"].tolist() == [35, 36]
    assert rows["over_35_words"].tolist() == [False, True]
    assert summary.loc[0, "length_compliance_rate"] == 0.5
    assert explanations["generated_explanation"].equals(original)


def test_claim_extraction_resumes_completed_explanations(tmp_path: Path, monkeypatch) -> None:
    explanations = _rows()
    input_path = tmp_path / "explanations.csv"
    explanations.to_csv(input_path, index=False)
    output = tmp_path / "output"
    report = tmp_path / "handoff.md"
    extractor = _Extractor()
    monkeypatch.setattr(
        "evidence_fashion_recommender.evaluation.stage45_v2.validate_explanations",
        lambda frame: None,
    )
    first = run_claim_extraction_v2(
        explanations=explanations,
        extractor=extractor,
        cache=ArtifactCache(tmp_path / "cache"),
        input_path=input_path,
        output_dir=output,
        report_path=report,
    )
    assert first["completed"] == 2
    assert extractor.calls == 2
    second = run_claim_extraction_v2(
        explanations=explanations,
        extractor=extractor,
        cache=ArtifactCache(tmp_path / "cache"),
        input_path=input_path,
        output_dir=output,
        report_path=report,
    )
    assert second["completed"] == 2
    assert extractor.calls == 2


def test_malformed_json_retries_twice_then_uses_repair_prompt(tmp_path: Path) -> None:
    extractor = _MalformedThenRepairExtractor()
    result = _extract_with_recovery(
        extractor=extractor,
        prompt="extract",
        cache=ArtifactCache(tmp_path / "cache"),
        context={"input_hash": "abc"},
        explanation_key="row-1",
    )
    assert extractor.calls == 4
    assert result["repaired_json_response"] is True
    assert result["repair_attempt_status"] == "succeeded"
    assert result["retry_count"] == 2
    assert result["claims"][0]["claim"] == "Preserved claim"


def test_verification_marks_failed_extraction_na_without_calling_model(
    tmp_path: Path, monkeypatch
) -> None:
    explanations = _rows().iloc[[0]].copy()
    input_path = tmp_path / "explanations.csv"
    explanations.to_csv(input_path, index=False)
    extraction_dir = tmp_path / "extraction"
    extraction_dir.mkdir()
    extraction_key = _key(explanations.iloc[0])
    (extraction_dir / "extraction_checkpoint.jsonl").write_text(
        '{"explanation_key":"'
        + extraction_key
        + '","claim_extraction_failed":true,"claims":[]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evidence_fashion_recommender.evaluation.stage45_v2.validate_explanations",
        lambda frame: None,
    )

    output = tmp_path / "verification"
    result = run_claim_verification_v2(
        explanations=explanations,
        extraction_dir=extraction_dir,
        verifier=_VerifierThatMustNotRun(),
        cache=ArtifactCache(tmp_path / "cache"),
        input_path=input_path,
        output_dir=output,
        report_path=tmp_path / "handoff.md",
    )

    verified = pd.read_csv(output / "verified_claims.csv", keep_default_na=False)
    assert result["completed"] == 1
    assert verified.loc[0, "verification_status"] == "N/A"
    assert verified.loc[0, "support_label"] == ""
    assert not bool(verified.loc[0, "claim_verification_failed"])


def test_malformed_verification_retries_twice_then_uses_repair_prompt(
    tmp_path: Path,
) -> None:
    verifier = _MalformedThenRepairVerifier()
    result = _verify_with_recovery(
        verifier=verifier,
        prompt="verify",
        claim_ids={"C1"},
        cache=ArtifactCache(tmp_path / "cache"),
        context={"input_hash": "abc"},
        explanation_key="row-1",
    )
    assert verifier.calls == 4
    assert result["repaired_json_response"] is True
    assert result["repair_attempt_status"] == "succeeded"
    assert result["retry_count"] == 2
    assert result["verifications"][0]["support_label"] == "unsupported"


def test_schema_mismatch_retries_then_becomes_na_without_semantic_repair(
    tmp_path: Path,
) -> None:
    verifier = _MismatchedClaimVerifier()
    result = _verify_with_recovery(
        verifier=verifier,
        prompt="verify",
        claim_ids={"C1"},
        cache=ArtifactCache(tmp_path / "cache"),
        context={"input_hash": "abc"},
        explanation_key="row-1",
    )
    assert verifier.calls == 3
    assert result["verifications"] == []
    assert result["repair_attempt_status"] == "not_applicable"
    assert result["repaired_json_response"] is False


def test_malformed_general_judgment_retries_then_repairs(tmp_path: Path) -> None:
    judge = _MalformedThenRepairJudge()
    result = _judge_with_recovery(
        judge=judge,
        prompt="judge",
        cache=ArtifactCache(tmp_path / "cache"),
        context={"input_hash": "abc"},
        judgment_key="judgment-1",
    )
    assert judge.calls == 4
    assert result["general_quality"] == 4
    assert result["repair_attempt_status"] == "succeeded"
    assert result["repaired_json_response"] is True


def test_invalid_general_judgment_retries_then_becomes_na(tmp_path: Path) -> None:
    judge = _InvalidScoreJudge()
    result = _judge_with_recovery(
        judge=judge,
        prompt="judge",
        cache=ArtifactCache(tmp_path / "cache"),
        context={"input_hash": "abc"},
        judgment_key="judgment-1",
    )
    assert judge.calls == 3
    assert result["general_quality"] is None
    assert result["repair_attempt_status"] == "not_applicable"
