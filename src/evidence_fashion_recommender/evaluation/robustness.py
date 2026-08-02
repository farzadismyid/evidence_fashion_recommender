"""Prompt ablations and multi-model robustness aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..cache import ArtifactCache, stable_fingerprint
from ..generation import build_explanation_prompt
from ..models.base import Generator
from .study import (
    JUDGE_DIMENSIONS,
    _parse_judge,
    _rule_frame,
    cached_generate,
    evaluate_explanations,
    evidence_for_variant,
)


@dataclass(frozen=True)
class HybridPromptSpec:
    max_words: int
    rule_limit: int
    prompt_order: str
    item_limit: int = 5

    @property
    def name(self) -> str:
        prefix = "hybrid" if self.item_limit > 0 else "rule_only_candidate"
        return (
            f"{prefix}_w{self.max_words}_r{self.rule_limit}_i{self.item_limit}_"
            f"{self.prompt_order}"
        )

    @property
    def final_hybrid_eligible(self) -> bool:
        return self.item_limit > 0


def full_hybrid_specs(
    word_budgets: list[int],
    rule_counts: list[int],
    item_counts: list[int],
    evidence_orders: list[str],
) -> list[HybridPromptSpec]:
    """Build the complete v2 factorial grid, including labelled rule-only candidates."""

    return [
        HybridPromptSpec(words, rules, order, items)
        for words in word_budgets
        for rules in rule_counts
        for items in item_counts
        for order in evidence_orders
    ]


def validate_stage1_validation_packets(cases: pd.DataFrame) -> str:
    """Require Hybrid v2 tuning packets produced by selected Stage 1 validation retrieval."""

    required = {"research_split", "stage1_packet_hash", "stage1_packet_protocol"}
    missing = required - set(cases.columns)
    if missing:
        raise ValueError(f"Hybrid v2 cases are missing Stage 1 packet fields: {sorted(missing)}")
    if set(cases["research_split"].astype(str)) != {"validation"}:
        raise ValueError("Hybrid v2 selection requires validation cases only.")
    if set(cases["stage1_packet_protocol"].astype(str)) != {"final_eval_v2_selected"}:
        raise ValueError("Legacy-only evidence packets are ineligible for Hybrid v2 selection.")
    hashes = set(cases["stage1_packet_hash"].astype(str))
    if len(hashes) != 1 or not next(iter(hashes)).strip():
        raise ValueError("Hybrid v2 cases must share one non-empty Stage 1 packet hash.")
    return next(iter(hashes))


def select_hybrid_finalists(
    summary: pd.DataFrame,
    *,
    practical_tie: float,
    finalist_count: int,
) -> pd.DataFrame:
    """Priority-select eligible Hybrid finalists without a weighted composite."""

    required = {
        "grounding_variant",
        "item_limit",
        "hallucinated_claim_rate",
        "rule_supported_claim_rate",
        "evidence_misuse_rate",
        "candidate_substitution_rate",
        "rule_evidence_overlap",
        "item_evidence_overlap",
        "general_clarity",
        "max_words",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Hybrid selection summary is missing columns: {sorted(missing)}")
    eligible = summary[summary["item_limit"] > 0].copy()
    if eligible.empty:
        raise ValueError("No final-Hybrid-eligible configurations contain item evidence.")
    priorities = [
        ("hallucinated_claim_rate", True),
        ("rule_supported_claim_rate", False),
        ("evidence_misuse_rate", True),
        ("candidate_substitution_rate", True),
        ("rule_evidence_overlap", False),
        ("item_evidence_overlap", False),
        ("general_clarity", False),
    ]
    pool = eligible
    for metric, minimize in priorities:
        best = pool[metric].min() if minimize else pool[metric].max()
        if minimize:
            tied = pool[metric] <= best + practical_tie
        else:
            tied = pool[metric] >= best - practical_tie
        narrowed = pool[tied]
        if len(narrowed) >= finalist_count:
            pool = narrowed
        else:
            break
    return pool.sort_values("max_words", kind="stable").head(finalist_count).reset_index(drop=True)


def one_factor_hybrid_specs(
    word_limits: list[int],
    rule_counts: list[int],
    prompt_orders: list[str],
) -> list[HybridPromptSpec]:
    """Build an efficient one-factor grid around a readable central configuration."""

    central_words = 75 if 75 in word_limits else word_limits[0]
    central_rules = 3 if 3 in rule_counts else rule_counts[0]
    central_order = "candidate_first" if "candidate_first" in prompt_orders else prompt_orders[0]
    specs = {HybridPromptSpec(words, central_rules, central_order) for words in word_limits}
    if 55 in word_limits and 5 in rule_counts:
        specs.add(HybridPromptSpec(55, 5, "candidate_first"))
    specs.update(HybridPromptSpec(central_words, count, central_order) for count in rule_counts)
    specs.update(HybridPromptSpec(central_words, central_rules, order) for order in prompt_orders)
    return sorted(specs, key=lambda value: value.name)


def generate_hybrid_ablations(
    cases: pd.DataFrame,
    specs: list[HybridPromptSpec],
    generator: Generator,
    cache: ArtifactCache,
    cache_context: dict[str, object] | None = None,
) -> pd.DataFrame:
    rows = []
    for _, case in cases.iterrows():
        for spec in specs:
            prompt = build_explanation_prompt(
                query_text=str(case["query_text"]),
                user_request=str(case["user_request"]),
                recommended_text=str(case["recommended_text"]),
                variant="hybrid_rag",
                item_evidence=str(case.get("item_evidence_text", "")).splitlines(),
                rule_evidence=_rule_frame(case),
                max_words=spec.max_words,
                rule_limit=spec.rule_limit,
                item_limit=spec.item_limit,
                prompt_order=spec.prompt_order,
            )
            rows.append(
                {
                    **case.to_dict(),
                    "grounding_variant": spec.name,
                    "base_grounding_variant": "hybrid_rag",
                    "max_words": spec.max_words,
                    "rule_limit": spec.rule_limit,
                    "item_limit": spec.item_limit,
                    "prompt_order": spec.prompt_order,
                    "candidate_type": (
                        "hybrid" if spec.final_hybrid_eligible else "rule_only_candidate"
                    ),
                    "generated_explanation": cached_generate(
                        generator,
                        prompt,
                        cache,
                        "robustness_generations",
                        cache_context=cache_context,
                    ),
                    "generation_model": generator.model_id,
                    "generation_prompt_fingerprint": stable_fingerprint(prompt),
                }
            )
    return pd.DataFrame(rows)


def generate_robustness_study(
    cases: pd.DataFrame,
    variants: list[str],
    generators: list[Generator],
    selected_hybrid: HybridPromptSpec,
    cache: ArtifactCache,
) -> pd.DataFrame:
    rows = []
    for generator in generators:
        for _, case in cases.iterrows():
            for variant in variants:
                options = (
                    {
                        "max_words": selected_hybrid.max_words,
                        "rule_limit": selected_hybrid.rule_limit,
                        "item_limit": selected_hybrid.item_limit,
                        "prompt_order": selected_hybrid.prompt_order,
                    }
                    if variant == "hybrid_rag"
                    else {}
                )
                prompt = build_explanation_prompt(
                    query_text=str(case["query_text"]),
                    user_request=str(case["user_request"]),
                    recommended_text=str(case["recommended_text"]),
                    variant=variant,
                    item_evidence=str(case.get("item_evidence_text", "")).splitlines(),
                    rule_evidence=_rule_frame(case),
                    **options,
                )
                rows.append(
                    {
                        **case.to_dict(),
                        "grounding_variant": variant,
                        "generated_explanation": cached_generate(
                            generator, prompt, cache, "robustness_final_generations"
                        ),
                        "generation_model": generator.model_id,
                        "generation_prompt_fingerprint": stable_fingerprint(prompt),
                    }
                )
    return pd.DataFrame(rows)


def _robustness_judge_prompt(row: pd.Series) -> str:
    evidence = evidence_for_variant(row) or (
        f"Query: {row['query_text']}\nRequest: {row['user_request']}\n"
        f"Locked item: {row['recommended_text']}"
    )
    return f"""Independently evaluate this fashion explanation using only the supplied
information. The generator is a different model. Do not identify the experimental method.
Assess each dimension separately before choosing its score. Do not default all dimensions
to the same value.

Query: {row["query_text"]}
Request: {row["user_request"]}
Locked recommended item: {row["recommended_text"]}
Available evidence:
{evidence}
Explanation:
{row["generated_explanation"]}

Use integer scores: 1=very poor, 2=poor, 3=mixed, 4=strong, 5=excellent.
Required score keys: faithfulness_to_available_information, usefulness_to_user,
specificity, style_appropriateness, grounding_safety.
Also return a claims array with at most three atomic claims. Each claim object must have
the keys claim and support; support must be supported, unsupported, or not_verifiable.
Keep each claim under 12 words and brief_reason under 20 words.
Return one compact JSON object only, with the five score keys, claims, and brief_reason."""


def judge_robustness_study(
    explanations: pd.DataFrame,
    judges: list[Generator],
    cache: ArtifactCache,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, errors = [], []
    for _, explanation in explanations.iterrows():
        prompt = _robustness_judge_prompt(explanation)
        for judge in judges:
            try:
                response = cached_generate(judge, prompt, cache, "robustness_final_judges")
                scores = _parse_judge(response)
                claims = scores.get("claims", [])
                labels = [str(claim.get("support", "")).lower() for claim in claims]
                compliant_labels = sum(
                    bool(claim.get("support_label_compliant", False)) for claim in claims
                )
                supported = sum(label == "supported" for label in labels)
                unsupported = sum(label == "unsupported" for label in labels)
                rows.append(
                    {
                        **explanation.to_dict(),
                        **scores,
                        "overall_judge_score": float(
                            np.mean([scores[dimension] for dimension in JUDGE_DIMENSIONS])
                        ),
                        "judge_model": judge.model_id,
                        "self_judge": (
                            judge.model_id.split("@", 1)[0]
                            == str(explanation["generation_model"]).split("@", 1)[0]
                        ),
                        "claim_count": len(labels),
                        "supported_claim_count": supported,
                        "unsupported_claim_count_model": unsupported,
                        "not_verifiable_claim_count": sum(
                            label == "not_verifiable" for label in labels
                        ),
                        "claim_support_rate": (supported / len(labels) if labels else 1.0),
                        "claim_label_compliant_count": compliant_labels,
                        "claim_label_compliance_rate": (
                            compliant_labels / len(labels) if labels else 1.0
                        ),
                        "raw_judge_response": response,
                    }
                )
            except Exception as error:
                errors.append(
                    {
                        "paper_case_id": explanation["paper_case_id"],
                        "grounding_variant": explanation["grounding_variant"],
                        "generation_model": explanation["generation_model"],
                        "judge_model": judge.model_id,
                        "error": repr(error),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(
        errors,
        columns=[
            "paper_case_id",
            "grounding_variant",
            "generation_model",
            "judge_model",
            "error",
        ],
    )


def evaluate_hybrid_ablations(
    explanations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation_input = explanations.copy()
    evaluation_input["grounding_variant"] = "hybrid_rag"
    evaluated, _ = evaluate_explanations(evaluation_input)
    evaluated["grounding_variant"] = explanations["grounding_variant"].to_numpy()
    evaluated["length_violation"] = evaluated["explanation_length_words"] > evaluated["max_words"]
    summary = (
        evaluated.groupby(["grounding_variant", "max_words", "rule_limit", "prompt_order"])
        .agg(
            explanations=("generated_explanation", "count"),
            unsupported_claims=("unsupported_claim_count_evidence_aware", "mean"),
            evidence_overlap=("evidence_overlap", "mean"),
            substitution_rate=("candidate_substitution_flag", "mean"),
            prompt_leakage_rate=("prompt_leakage_flag", "mean"),
            citation_presence=("citation_presence", "mean"),
            length_violation_rate=("length_violation", "mean"),
        )
        .reset_index()
    )
    # Deterministic selection score; weights are declared here before test evaluation.
    summary["automatic_selection_score"] = (
        summary["evidence_overlap"]
        + 0.25 * summary["citation_presence"]
        - 0.25 * summary["unsupported_claims"]
        - 0.50 * summary["substitution_rate"]
        - 0.25 * summary["length_violation_rate"]
    )
    return evaluated, summary.sort_values("automatic_selection_score", ascending=False).reset_index(
        drop=True
    )


def judge_agreement(judged: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    """Pairwise rank agreement between independent judges."""

    rows = []
    for dimension in dimensions:
        pivot = judged.pivot_table(
            index=["paper_case_id", "grounding_variant", "generation_model"],
            columns="judge_model",
            values=dimension,
        )
        judges = list(pivot.columns)
        for first_index, first in enumerate(judges):
            for second in judges[first_index + 1 :]:
                paired = pivot[[first, second]].dropna()
                rows.append(
                    {
                        "dimension": dimension,
                        "judge_a": first,
                        "judge_b": second,
                        "n": len(paired),
                        "spearman": float(paired[first].corr(paired[second], method="spearman")),
                        "kendall": float(paired[first].corr(paired[second], method="kendall")),
                        "mean_absolute_difference": float(
                            np.abs(paired[first] - paired[second]).mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)
