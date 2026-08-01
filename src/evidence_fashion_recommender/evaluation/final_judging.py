"""Anchored v2 general-quality judging and cross-model primary filtering."""

from __future__ import annotations

import json
import re

import pandas as pd

from ..cache import ArtifactCache
from ..models.base import Generator
from .study import cached_generate

GENERAL_DIMENSIONS = (
    "input_consistency",
    "general_quality",
    "clarity",
    "specificity",
    "hallucination_risk",
    "evidence_misuse",
)


def model_family(model_id: str) -> str:
    """Normalize tags/digests/settings while retaining the actual base model family."""

    base = str(model_id).split("@", 1)[0].strip().lower()
    base = re.sub(r":latest$", "", base)
    return base


def is_cross_model_judgment(generation_model: str, judge_model: str) -> bool:
    return model_family(generation_model) != model_family(judge_model)


def anchored_general_judge_prompt(row: pd.Series) -> str:
    return f"""Evaluate this explanation using the anchored dimensions below.
Do not infer unavailable attributes and do not identify the experimental variant.

input_consistency: 1=contradicts query/request/locked item; 3=mostly consistent but vague or
partially unsupported; 5=fully consistent with query/request/locked item.
general_quality: 1=unclear or unhelpful; 3=understandable but generic; 5=clear, concise and
useful-looking.
clarity: 1=hard to understand; 3=mostly understandable; 5=unambiguous and concise.
specificity: 1=generic; 3=some relevant detail; 5=specific to the supplied items/request.
hallucination_risk: 1=many unsupported/invented fashion claims; 3=some unsupported claims;
5=no unsupported fashion claims found.
evidence_misuse: 1=wrong rule/item/category or serious misapplication; 3=minor evidence-use
issues; 5=no evidence misuse found. Use 5 when no external evidence was supplied and none is
claimed or misused; this is not an external-grounding score.

Query item: {row.get('query_text', '')}
User request: {row.get('user_request', '')}
Locked recommended item: {row.get('recommended_text', '')}
Generation evidence actually supplied:
{row.get('generation_evidence_text', '') or 'No external generation evidence.'}
Explanation:
{row.get('generated_explanation', '')}

Return one JSON object with integer keys input_consistency, general_quality, clarity,
specificity, hallucination_risk, evidence_misuse, plus brief_reason."""


def _parse_scores(response: str) -> dict[str, object]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    scores = {}
    for dimension in GENERAL_DIMENSIONS:
        score = int(value[dimension])
        if score < 1 or score > 5:
            raise ValueError(f"{dimension} must be an integer from 1 to 5")
        scores[dimension] = score
    scores["brief_reason"] = str(value.get("brief_reason", ""))
    return scores


def judge_general_quality_v2(
    explanations: pd.DataFrame,
    judges: list[Generator],
    cache: ArtifactCache,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, errors = [], []
    for _, explanation in explanations.iterrows():
        prompt = anchored_general_judge_prompt(explanation)
        for judge in judges:
            cross_model = is_cross_model_judgment(
                str(explanation.get("generation_model", "")), judge.model_id
            )
            try:
                response = cached_generate(judge, prompt, cache, "final_eval_general_judge_v2")
                scores = _parse_scores(response)
                rows.append(
                    {
                        **explanation.to_dict(),
                        **scores,
                        "judge_model": judge.model_id,
                        "generation_model_family": model_family(
                            str(explanation.get("generation_model", ""))
                        ),
                        "judge_model_family": model_family(judge.model_id),
                        "cross_model_primary_eligible": cross_model,
                        "raw_judge_response": response,
                    }
                )
            except Exception as error:
                errors.append(
                    {
                        "paper_case_id": explanation.get("paper_case_id", ""),
                        "grounding_variant": explanation.get("grounding_variant", ""),
                        "generation_model": explanation.get("generation_model", ""),
                        "judge_model": judge.model_id,
                        "error": repr(error),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(errors)


def primary_and_sensitivity_summaries(judged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouping = ["grounding_variant"]
    dimensions = list(GENERAL_DIMENSIONS)
    primary = (
        judged[judged["cross_model_primary_eligible"]]
        .groupby(grouping, as_index=False)[dimensions]
        .mean()
    )
    sensitivity = judged.groupby(grouping, as_index=False)[dimensions].mean()
    sensitivity["analysis_role"] = "all_judges_sensitivity"
    primary["analysis_role"] = "cross_model_only_primary"
    return primary, sensitivity
