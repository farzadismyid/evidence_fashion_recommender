import pandas as pd

from evidence_fashion_recommender.cache import ArtifactCache
from evidence_fashion_recommender.evaluation.claim_evaluation import (
    build_reference_packet,
    extract_atomic_claims,
    verify_atomic_claims,
)


class _Extractor:
    model_id = "extractor@1"

    def generate(self, prompt: str) -> str:
        return (
            '{"claims":['
            '{"claim_id":"C1","claim":"The shoes suit formal occasions",'
            '"claim_type":"formality"},'
            '{"claim_id":"C2","claim":"The shoes visually match the dress",'
            '"claim_type":"visual_match"}]}'
        )


class _EmptyExtractor:
    model_id = "empty@1"

    def generate(self, prompt: str) -> str:
        return '{"claims":[]}'


class _Verifier:
    model_id = "verifier@1"

    def generate(self, prompt: str) -> str:
        return (
            '{"verifications":['
            '{"claim_id":"C1","support_label":"supported_by_rule_evidence",'
            '"supporting_rule_ids":["R001"],"citation_entails_claim":true},'
            '{"claim_id":"C2","support_label":"supported_by_query_or_locked_item",'
            '"supporting_rule_ids":[],"citation_entails_claim":null}]}'
        )


def _explanation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_case_id": "c1",
                "grounding_variant": "hybrid_rag",
                "generation_model": "generator@1",
                "query_text": "black dress",
                "user_request": "formal shoes",
                "recommended_text": "black pumps",
                "item_evidence_text": "ITEM-1: pumps",
                "rule_evidence_text": "R001: match formality",
                "generated_explanation": "These pumps suit formal occasions and match the dress.",
            }
        ]
    )


def test_reference_packet_distinguishes_generation_evidence() -> None:
    packet = build_reference_packet(_explanation().iloc[0])
    assert packet.item_evidence_shown_to_generator
    assert packet.rule_evidence_shown_to_generator
    assert packet.retrieved_rule_evidence.startswith("R001")


def test_extracts_all_claims_then_verifies_support_source(tmp_path) -> None:
    frame = _explanation()
    extracted = extract_atomic_claims(frame, _Extractor(), ArtifactCache(tmp_path))
    assert len(extracted) == 2
    verified = verify_atomic_claims(
        frame.iloc[0], extracted, _Verifier(), ArtifactCache(tmp_path)
    )
    assert set(verified["support_label"]) == {
        "supported_by_rule_evidence",
        "supported_by_query_or_locked_item",
    }
    assert verified["evaluation_reference_packet_hash"].nunique() == 1


def test_empty_claim_extraction_is_failure_not_perfect_support(tmp_path) -> None:
    frame = _explanation()
    extracted = extract_atomic_claims(frame, _EmptyExtractor(), ArtifactCache(tmp_path))
    assert extracted.loc[0, "claim_extraction_failed"]
    verified = verify_atomic_claims(
        frame.iloc[0], extracted, _Verifier(), ArtifactCache(tmp_path)
    )
    assert verified.loc[0, "claim_extraction_failed"]
    assert pd.isna(verified.loc[0, "claim_support"])
