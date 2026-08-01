"""Two-stage atomic claim extraction and structured-reference verification."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import pandas as pd

from ..cache import ArtifactCache, stable_fingerprint
from ..models.base import Generator
from .study import cached_generate

CLAIM_TYPES = {
    "item_type",
    "colour",
    "material",
    "occasion",
    "formality",
    "season",
    "comfort",
    "trend",
    "body_fit",
    "styling_relation",
    "visual_match",
    "other",
}
SUPPORT_LABELS = {
    "supported_by_query_or_locked_item",
    "supported_by_item_evidence",
    "supported_by_rule_evidence",
    "unsupported",
    "contradicted",
    "not_verifiable",
}


def _json_object(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


@dataclass(frozen=True)
class ReferencePacket:
    query_item_metadata: str
    user_request: str
    locked_recommended_item_metadata: str
    retrieved_item_evidence: str
    retrieved_rule_evidence: str
    item_evidence_shown_to_generator: bool
    rule_evidence_shown_to_generator: bool

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(asdict(self))


def build_reference_packet(row: pd.Series) -> ReferencePacket:
    variant = str(row["grounding_variant"])
    return ReferencePacket(
        query_item_metadata=str(row.get("query_text", "")),
        user_request=str(row.get("user_request", "")),
        locked_recommended_item_metadata=str(row.get("recommended_text", "")),
        retrieved_item_evidence=str(row.get("item_evidence_text", "")),
        retrieved_rule_evidence=str(row.get("rule_evidence_text", "")),
        item_evidence_shown_to_generator=variant in {"item_rag", "hybrid_rag"},
        rule_evidence_shown_to_generator=variant in {"rule_rag", "hybrid_rag"},
    )


def claim_extraction_prompt(explanation: str) -> str:
    types = ", ".join(sorted(CLAIM_TYPES))
    return f"""Extract every atomic fashion or styling claim from the explanation.
Do not assess whether claims are true. Do not omit claims and do not cap their number.
Allowed claim types: {types}.

Explanation:
{explanation}

Return one JSON object only:
{{"claims":[{{"claim_id":"C1","claim":"...","claim_type":"styling_relation"}}]}}"""


def parse_extracted_claims(response: str) -> list[dict[str, str]]:
    claims = _json_object(response).get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("claims must be a list")
    parsed = []
    for index, claim in enumerate(claims, 1):
        text = str(claim.get("claim", "")).strip()
        claim_type = str(claim.get("claim_type", "other")).strip().lower()
        if not text:
            raise ValueError("extracted claims must contain non-empty claim text")
        if claim_type not in CLAIM_TYPES:
            claim_type = "other"
        parsed.append(
            {
                "claim_id": str(claim.get("claim_id") or f"C{index}"),
                "claim": text,
                "claim_type": claim_type,
            }
        )
    return parsed


def extract_atomic_claims(
    explanations: pd.DataFrame,
    extractor: Generator,
    cache: ArtifactCache,
) -> pd.DataFrame:
    rows = []
    for _, explanation in explanations.iterrows():
        prompt = claim_extraction_prompt(str(explanation["generated_explanation"]))
        response = ""
        try:
            response = cached_generate(extractor, prompt, cache, "final_eval_claim_extraction_v2")
            claims = parse_extracted_claims(response)
            failed = len(claims) == 0
            if claims:
                for claim in claims:
                    rows.append(
                        {
                            "paper_case_id": explanation["paper_case_id"],
                            "grounding_variant": explanation["grounding_variant"],
                            "generation_model": explanation.get("generation_model", ""),
                            "extractor_model": extractor.model_id,
                            **claim,
                            "claim_extraction_failed": False,
                            "raw_extraction_response": response,
                        }
                    )
            else:
                rows.append(
                    {
                        "paper_case_id": explanation["paper_case_id"],
                        "grounding_variant": explanation["grounding_variant"],
                        "generation_model": explanation.get("generation_model", ""),
                        "extractor_model": extractor.model_id,
                        "claim_id": "",
                        "claim": "",
                        "claim_type": "",
                        "claim_extraction_failed": failed,
                        "raw_extraction_response": response,
                    }
                )
        except Exception as error:
            rows.append(
                {
                    "paper_case_id": explanation["paper_case_id"],
                    "grounding_variant": explanation["grounding_variant"],
                    "generation_model": explanation.get("generation_model", ""),
                    "extractor_model": extractor.model_id,
                    "claim_id": "",
                    "claim": "",
                    "claim_type": "",
                    "claim_extraction_failed": True,
                    "extraction_error": repr(error),
                    "raw_extraction_response": response,
                }
            )
    return pd.DataFrame(rows)


def claim_verification_prompt(claims: list[dict[str, str]], packet: ReferencePacket) -> str:
    labels = ", ".join(sorted(SUPPORT_LABELS))
    packet_json = json.dumps(asdict(packet), ensure_ascii=False, indent=2)
    claims_json = json.dumps(claims, ensure_ascii=False, indent=2)
    return f"""Verify every extracted claim using only the structured reference packet.
Generation-evidence flags identify what the generator actually saw; other packet fields are
evaluation references only. Assign exactly one support label per claim.
Allowed labels: {labels}.
For rule support, include supporting_rule_ids. If a cited rule is present, state whether it
entails the claim in citation_entails_claim (true, false, or null).

Structured reference packet:
{packet_json}

Claims:
{claims_json}

Return one JSON object only:
{{"verifications":[{{"claim_id":"C1","support_label":"unsupported","
supporting_rule_ids":[],"citation_entails_claim":null,"brief_reason":"..."}}]}}"""


def parse_claim_verifications(response: str, claim_ids: set[str]) -> list[dict[str, object]]:
    values = _json_object(response).get("verifications", [])
    if not isinstance(values, list):
        raise ValueError("verifications must be a list")
    parsed = []
    seen = set()
    for value in values:
        claim_id = str(value.get("claim_id", ""))
        label = str(value.get("support_label", "")).lower()
        if claim_id not in claim_ids or claim_id in seen:
            raise ValueError("verification claim IDs must match extracted claims exactly")
        if label not in SUPPORT_LABELS:
            raise ValueError(f"unsupported support label: {label}")
        seen.add(claim_id)
        parsed.append(
            {
                "claim_id": claim_id,
                "support_label": label,
                "supporting_rule_ids": value.get("supporting_rule_ids", []),
                "citation_entails_claim": value.get("citation_entails_claim"),
                "brief_reason": str(value.get("brief_reason", "")),
            }
        )
    if seen != claim_ids:
        raise ValueError("every extracted claim must be verified")
    return parsed


def verify_atomic_claims(
    explanation: pd.Series,
    extracted: pd.DataFrame,
    verifier: Generator,
    cache: ArtifactCache,
) -> pd.DataFrame:
    valid = extracted[~extracted["claim_extraction_failed"].astype(bool)]
    if valid.empty:
        return pd.DataFrame(
            [
                {
                    "paper_case_id": explanation["paper_case_id"],
                    "claim_extraction_failed": True,
                    "claim_support": pd.NA,
                }
            ]
        )
    claims = valid[["claim_id", "claim", "claim_type"]].to_dict("records")
    packet = build_reference_packet(explanation)
    prompt = claim_verification_prompt(claims, packet)
    response = cached_generate(verifier, prompt, cache, "final_eval_claim_verification_v2")
    verified = parse_claim_verifications(response, {claim["claim_id"] for claim in claims})
    output = valid.merge(pd.DataFrame(verified), on="claim_id", validate="one_to_one")
    output["verifier_model"] = verifier.model_id
    output["evaluation_reference_packet_hash"] = packet.fingerprint
    output["item_evidence_shown_to_generator"] = packet.item_evidence_shown_to_generator
    output["rule_evidence_shown_to_generator"] = packet.rule_evidence_shown_to_generator
    output["raw_verification_response"] = response
    return output
