"""Claim-level faithfulness and independent rule-relevance verification."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ..cache import ArtifactCache
from ..models.base import Generator
from .study import cached_generate, evidence_for_variant


def _json_object(response: str) -> dict:
    for candidate in reversed(re.findall(r"\{.*\}", response, re.DOTALL)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return json.loads(response)


def claim_verification_prompt(row: pd.Series) -> str:
    evidence = evidence_for_variant(row)
    if not evidence:
        evidence = (
            f"Query: {row['query_text']}\n"
            f"Request: {row['user_request']}\n"
            f"Locked item: {row['recommended_text']}"
        )
    return f"""Verify an explanation claim by claim using only the supplied evidence.
Do not use outside fashion knowledge. Split the explanation into atomic factual or
compatibility claims. For every claim label support as supported, unsupported, or
not_verifiable. A stylistic recommendation is supported only when its stated reason is
present in the evidence.

Evidence:
{evidence}

Explanation:
{row["generated_explanation"]}

Return only JSON:
{{"claims":[{{"claim":"...", "support":"supported|unsupported|not_verifiable"}}]}}"""


def verify_claims(
    explanations: pd.DataFrame,
    verifiers: list[Generator],
    cache: ArtifactCache,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, errors = [], []
    for _, explanation in explanations.iterrows():
        prompt = claim_verification_prompt(explanation)
        for verifier in verifiers:
            try:
                response = cached_generate(verifier, prompt, cache, "claim_verification")
                claims = _json_object(response).get("claims", [])
                labels = [str(claim.get("support", "")).lower() for claim in claims]
                total = len(labels)
                supported = sum(label == "supported" for label in labels)
                unsupported = sum(label == "unsupported" for label in labels)
                unverifiable = sum(label == "not_verifiable" for label in labels)
                rows.append(
                    {
                        "paper_case_id": explanation["paper_case_id"],
                        "grounding_variant": explanation["grounding_variant"],
                        "generation_model": explanation.get("generation_model", ""),
                        "verifier_model": verifier.model_id,
                        "claim_count": total,
                        "supported_claim_count": supported,
                        "unsupported_claim_count_model": unsupported,
                        "not_verifiable_claim_count": unverifiable,
                        "claim_support_rate": supported / total if total else 1.0,
                        "raw_verification": response,
                    }
                )
            except Exception as error:
                errors.append(
                    {
                        "paper_case_id": explanation["paper_case_id"],
                        "grounding_variant": explanation["grounding_variant"],
                        "verifier_model": verifier.model_id,
                        "error": repr(error),
                    }
                )
    results = pd.DataFrame(rows)
    if results.empty:
        return results, pd.DataFrame(errors)
    consensus = (
        results.groupby(["paper_case_id", "grounding_variant", "generation_model"], dropna=False)
        .agg(
            verifier_count=("verifier_model", "nunique"),
            mean_claim_support_rate=("claim_support_rate", "mean"),
            conservative_claim_support_rate=("claim_support_rate", "min"),
            mean_model_unsupported_claims=("unsupported_claim_count_model", "mean"),
        )
        .reset_index()
    )
    return results.merge(
        consensus,
        on=["paper_case_id", "grounding_variant", "generation_model"],
        how="left",
    ), pd.DataFrame(errors)


def rule_relevance_prompt(case: pd.Series) -> str:
    return f"""Independently judge which retrieved fashion rules apply to this exact
recommendation. Do not assume a rule is relevant merely because it was retrieved.

Query item: {case["query_text"]}
User request: {case["user_request"]}
Recommended item: {case["recommended_text"]}

Retrieved rules:
{case["rule_evidence_text"]}

Return only JSON with every supplied rule ID:
{{"relevance":{{"R001":0,"R002":1}}}}
Use 1 for relevant and 0 for not relevant."""


def verify_rule_relevance(
    cases: pd.DataFrame,
    judges: list[Generator],
    cache: ArtifactCache,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, errors = [], []
    for _, case in cases.iterrows():
        prompt = rule_relevance_prompt(case)
        supplied = re.findall(r"\bR\d+\b", str(case["rule_evidence_ids"]))
        for judge in judges:
            try:
                response = cached_generate(judge, prompt, cache, "rule_relevance_judge")
                relevance = _json_object(response).get("relevance", {})
                for rank, rule_id in enumerate(supplied, 1):
                    rows.append(
                        {
                            "paper_case_id": case["paper_case_id"],
                            "judge_model": judge.model_id,
                            "rule_id": rule_id,
                            "rank": rank,
                            "relevant": int(bool(relevance.get(rule_id, 0))),
                        }
                    )
            except Exception as error:
                errors.append(
                    {
                        "paper_case_id": case["paper_case_id"],
                        "judge_model": judge.model_id,
                        "error": repr(error),
                    }
                )
    results = pd.DataFrame(rows)
    if not results.empty:
        consensus = (
            results.groupby(["paper_case_id", "rule_id", "rank"])["relevant"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "consensus_probability", "count": "judge_count"})
        )
        consensus["consensus_relevant"] = (consensus["consensus_probability"] >= 0.5).astype(int)
        results = results.merge(consensus, on=["paper_case_id", "rule_id", "rank"], how="left")
    return results, pd.DataFrame(
        errors,
        columns=["paper_case_id", "judge_model", "error"],
    )


def consensus_retrieval_metrics(results: pd.DataFrame) -> pd.DataFrame:
    unique = results.drop_duplicates(["paper_case_id", "rule_id", "rank"])
    rows = []
    for case_id, group in unique.groupby("paper_case_id"):
        group = group.sort_values("rank")
        labels = group["consensus_relevant"].to_numpy()
        positive = np.flatnonzero(labels) + 1
        rows.append(
            {
                "paper_case_id": case_id,
                "consensus_precision_at_1": float(labels[:1].mean()),
                "consensus_precision_at_3": float(labels[:3].mean()),
                "consensus_precision_at_5": float(labels[:5].mean()),
                "consensus_hit_rate_at_5": float(labels[:5].any()),
                "consensus_mrr": 1.0 / positive[0] if len(positive) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def rule_relevance_agreement(results: pd.DataFrame) -> pd.DataFrame:
    """Pairwise agreement and Cohen's kappa for binary rule relevance."""

    rows = []
    pivot = results.pivot_table(
        index=["paper_case_id", "rule_id", "rank"],
        columns="judge_model",
        values="relevant",
    )
    judges = list(pivot.columns)
    for first_index, first in enumerate(judges):
        for second in judges[first_index + 1 :]:
            paired = pivot[[first, second]].dropna()
            observed = float((paired[first] == paired[second]).mean())
            first_positive = float(paired[first].mean())
            second_positive = float(paired[second].mean())
            expected = first_positive * second_positive + (1 - first_positive) * (
                1 - second_positive
            )
            kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
            rows.append(
                {
                    "judge_a": first,
                    "judge_b": second,
                    "n": len(paired),
                    "percent_agreement": observed,
                    "cohen_kappa": kappa,
                }
            )
    return pd.DataFrame(rows)


def counterfactual_category_test(cases: pd.DataFrame, knowledge_base: pd.DataFrame) -> pd.DataFrame:
    """Rotate target categories and measure false applicability of retrieved rules."""

    categories = sorted(cases["target_category"].astype(str).unique())
    rotated = {
        category: categories[(index + 1) % len(categories)]
        for index, category in enumerate(categories)
    }
    kb = knowledge_base.set_index("rule_id")
    rows = []
    for _, case in cases.iterrows():
        ids = re.findall(r"\bR\d+\b", str(case["rule_evidence_ids"]))
        counterfactual = rotated[str(case["target_category"])]
        false_matches = sum(
            rule_id in kb.index and str(kb.loc[rule_id, "recommended_category"]) == counterfactual
            for rule_id in ids
        )
        rows.append(
            {
                "paper_case_id": case["paper_case_id"],
                "actual_target_category": case["target_category"],
                "counterfactual_target_category": counterfactual,
                "retrieved_rule_count": len(ids),
                "counterfactual_false_match_count": false_matches,
                "counterfactual_false_match_rate": (false_matches / len(ids) if ids else 0.0),
            }
        )
    return pd.DataFrame(rows)
