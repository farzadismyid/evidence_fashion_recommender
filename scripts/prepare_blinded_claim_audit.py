"""Prepare, but never score, the blinded 360-claim researcher audit."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("outputs/final_eval_v2/post_recovery")
OUT = Path("outputs/final_eval_v2/manual_audit")
SEED = 42
LABELS = (
    "supported_by_rule_evidence",
    "supported_by_item_evidence",
    "supported_by_query_or_locked_item",
    "unsupported",
    "contradicted",
    "not_verifiable",
)
VARIANTS = ("no_rag", "item_rag", "rule_rag", "hybrid_rag")
BLIND_COLUMNS = (
    "anonymous_audit_id",
    "atomic_claim",
    "query_context",
    "user_request",
    "locked_recommended_item",
    "retrieved_item_evidence",
    "retrieved_expert_rule_evidence",
    "human_label",
    "human_notes",
)


def model_family(value: str) -> str:
    return str(value).split("@", 1)[0]


def _audit_id(row: pd.Series) -> str:
    value = "|".join(str(row[column]) for column in ("explanation_key", "claim_id"))
    return "A" + hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()[:12].upper()


def load_frame() -> pd.DataFrame:
    claims = pd.read_csv(ROOT / "claims/extraction/claims.csv")
    verified = pd.read_csv(ROOT / "claims/verification/verified_claims.csv")
    explanations = pd.read_csv(ROOT / "explanations/explanations.csv")
    keys = ["explanation_key", "claim_id"]
    frame = claims.merge(
        verified[keys + ["verification_status", "support_label", "brief_reason"]],
        on=keys,
        validate="one_to_one",
    ).merge(
        explanations[
            [
                "paper_case_id",
                "grounding_variant",
                "generation_model",
                "query_text",
                "user_request",
                "recommended_text",
                "item_evidence_text",
                "rule_evidence_text",
            ]
        ],
        on=["paper_case_id", "grounding_variant", "generation_model"],
        validate="many_to_one",
    )
    frame = frame[
        (frame.verification_status == "complete") & frame.support_label.isin(LABELS)
    ].copy()
    frame["generator_family"] = frame.generation_model.map(model_family)
    return frame


def sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministically select one claim/explanation with fixed variant/model quotas."""
    families = sorted(frame.generator_family.unique())
    if len(families) != 3:
        raise ValueError(f"Expected three generator families, found {families}")
    rng = np.random.default_rng(SEED)
    working = frame.copy()
    working["_random"] = rng.random(len(working))
    selected: list[int] = []
    used_explanations: set[str] = set()
    paper_counts: dict[str, int] = {}
    quotas = {(variant, family): 30 for variant in VARIANTS for family in families}

    def take(index: int) -> bool:
        row = working.loc[index]
        key, case = str(row.explanation_key), str(row.paper_case_id)
        group = (str(row.grounding_variant), str(row.generator_family))
        if quotas[group] <= 0 or key in used_explanations or paper_counts.get(case, 0) >= 2:
            return False
        selected.append(index)
        used_explanations.add(key)
        paper_counts[case] = paper_counts.get(case, 0) + 1
        quotas[group] -= 1
        return True

    # Guarantee representation of every automatic label when the label is available.
    for label in LABELS:
        candidates = working[working.support_label == label].sort_values("_random").index
        if len(candidates) and not any(take(int(index)) for index in candidates):
            raise ValueError(f"Cannot include available label {label!r} under audit constraints.")
    # Fill exact 30-per-generator, 90-per-variant quotas. Random order makes selection reproducible.
    for group in quotas:
        candidates = (
            working[
                (working.grounding_variant == group[0]) & (working.generator_family == group[1])
            ]
            .sort_values("_random")
            .index
        )
        for index in candidates:
            if quotas[group] == 0:
                break
            take(int(index))
        if quotas[group]:
            raise ValueError(
                f"Unable to fill deterministic quota for {group}: {quotas[group]} remaining"
            )
    result = working.loc[selected].copy()
    if len(result) != 360 or result.explanation_key.nunique() != 360:
        raise ValueError("Audit must contain exactly 360 claims from distinct explanations.")
    if (result.groupby("grounding_variant").size() != 90).any():
        raise ValueError("Variant quota failure")
    if (result.groupby(["grounding_variant", "generator_family"]).size() != 30).any():
        raise ValueError("Generator balance failure")
    if result.paper_case_id.value_counts().max() > 2:
        raise ValueError("Paper-case concentration limit failure")
    # Nominal, recorded design probability: simple random explanation draw within v/model,
    # times uniform claim selection within explanation. Label-priority selection is recorded in key.
    counts = frame.groupby(["grounding_variant", "generator_family"])["explanation_key"].nunique()
    claim_counts = frame.groupby("explanation_key").size()
    result["nominal_inclusion_probability"] = [
        30
        / counts[(row.grounding_variant, row.generator_family)]
        / claim_counts[row.explanation_key]
        for _, row in result.iterrows()
    ]
    result["sampling_weight"] = 1 / result.nominal_inclusion_probability
    result["anonymous_audit_id"] = result.apply(_audit_id, axis=1)
    return result.sort_values("anonymous_audit_id").reset_index(drop=True)


def write_documents() -> None:
    guide = """# Blinded claim-verification annotation guide

Annotate each atomic claim using only the displayed query/request, locked item, retrieved item evidence, and retrieved expert-rule evidence. Do not infer the generation condition or automatic decision.

Use exactly one label: `supported_by_rule_evidence` when rule evidence semantically entails the claim; `supported_by_item_evidence` when retrieved item evidence entails it; `supported_by_query_or_locked_item` when query/request/locked-item information entails it; `unsupported` when available evidence does not support it; `contradicted` when available evidence conflicts with it; `not_verifiable` when evidence is insufficient, ambiguous, or cannot settle it.

Apply semantic support, not word overlap. A related rule is not support unless it entails the specific claim. If more than one source supports a claim, choose the most direct source (rule, item, then query/locked item) and explain the additional support in notes. Use `contradicted` only for affirmative conflict, not absence. Use `not_verifiable` for genuinely indeterminate claims and `unsupported` for claims that are not entailed despite adequate relevant context. Record concise reasoning in `human_notes`.
"""
    protocol = """# Blinded 360-claim researcher-audit protocol

Purpose: assess agreement between independent human labels and the frozen automatic verifier without revealing variant, generator, automatic label, verifier reason, scores, or model metadata.

Population: complete post-recovery atomic-claim verification rows. Seed: 42. The deterministic design samples 360 distinct explanations: 90 per grounding variant and 30 per generator family within each variant. It first secures one occurrence of every available automatic label, including contradicted and not-verifiable, then fills quotas while capping sampled claims at two per paper case. The sealed key records automatic label, provenance, nominal inclusion probability, and inverse-probability weight. Nominal probabilities are the pre-label-priority within variant/generator explanation draw probability times one-over-claims-per-explanation; label-priority selection makes the weights design documentation rather than a claim of exact inclusion probabilities under every constraint.

Researchers must annotate only `blinded_360_claims.csv`, retain `human_label` and `human_notes`, and not open the key until annotation is complete. The scoring script is intentionally not run during preparation.
"""
    OUT.joinpath("ANNOTATION_GUIDE.md").write_text(guide, encoding="utf-8")
    OUT.joinpath("MANUAL_AUDIT_PROTOCOL.md").write_text(protocol, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = sample(load_frame())
    blind = pd.DataFrame(
        {
            "anonymous_audit_id": selected.anonymous_audit_id,
            "atomic_claim": selected.claim,
            "query_context": selected.query_text,
            "user_request": selected.user_request,
            "locked_recommended_item": selected.recommended_text,
            "retrieved_item_evidence": selected.item_evidence_text,
            "retrieved_expert_rule_evidence": selected.rule_evidence_text,
            "human_label": "",
            "human_notes": "",
        }
    )
    if tuple(blind.columns) != BLIND_COLUMNS:
        raise ValueError("Blinded schema changed")
    forbidden = {
        "grounding_variant",
        "generation_model",
        "support_label",
        "brief_reason",
        "verifier_model",
    }
    if (
        forbidden & set(blind.columns)
        or len(blind) != 360
        or blind.anonymous_audit_id.duplicated().any()
    ):
        raise ValueError("Blinding validation failed")
    key = selected[
        [
            "anonymous_audit_id",
            "explanation_key",
            "paper_case_id",
            "grounding_variant",
            "generation_model",
            "claim_id",
            "claim",
            "support_label",
            "brief_reason",
            "nominal_inclusion_probability",
            "sampling_weight",
        ]
    ]
    blind.to_csv(OUT / "blinded_360_claims.csv", index=False)
    key.to_csv(OUT / "audit_key_DO_NOT_OPEN_UNTIL_ANNOTATION_COMPLETE.csv", index=False)
    write_documents()
    (OUT / "preparation_manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "claims": 360,
                "distinct_explanations": int(selected.explanation_key.nunique()),
                "variant_counts": selected.grounding_variant.value_counts().to_dict(),
                "generator_counts": {
                    f"{variant}|{family}": int(count)
                    for (variant, family), count in selected.groupby(
                        ["grounding_variant", "generator_family"]
                    )
                    .size()
                    .items()
                },
                "label_counts": selected.support_label.value_counts().to_dict(),
                "source_tables": [
                    "claims/extraction/claims.csv",
                    "claims/verification/verified_claims.csv",
                    "explanations/explanations.csv",
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
