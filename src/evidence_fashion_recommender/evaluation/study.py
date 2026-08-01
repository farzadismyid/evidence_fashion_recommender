"""Systematic explanation, faithfulness, RAG, and independent-judge study."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..cache import ArtifactCache, stable_fingerprint
from ..config import AppConfig
from ..evidence import detect_item_types
from ..generation import build_explanation_prompt
from ..models.base import Generator
from .explanations import candidate_substitution_types, cited_rule_ids, evidence_overlap
from .statistics import compare_variants

UNSUPPORTED_CLAIM_TERMS = [
    "comfortable",
    "comfort",
    "high-quality",
    "high quality",
    "luxurious",
    "luxury",
    "must-have",
    "perfect",
    "breathtaking",
    "fashionista",
    "body shape",
    "slimming",
    "flattering",
    "guaranteed",
]
OCCASION_DRIFT_TERMS = [
    "travel",
    "cocktail",
    "evening",
    "summer",
    "daytime",
    "everyday",
    "night out",
    "date night",
]
JUDGE_DIMENSIONS = [
    "faithfulness_to_available_information",
    "usefulness_to_user",
    "specificity",
    "style_appropriateness",
    "grounding_safety",
]


def _input_category_contains(value: str, query_group: str) -> bool:
    return query_group in {part.strip() for part in str(value).split(",")}


def evidence_for_variant(row: pd.Series) -> str:
    return {
        "no_rag": "",
        "item_rag": str(row.get("item_evidence_text", "")),
        "rule_rag": str(row.get("rule_evidence_text", "")),
        "hybrid_rag": str(row.get("hybrid_evidence_text", "")),
    }.get(str(row["grounding_variant"]), "")


def _rule_frame(row: pd.Series) -> pd.DataFrame:
    ids = re.findall(r"\bR\d+\b", str(row.get("rule_evidence_ids", "")))
    lines = str(row.get("rule_evidence_text", "")).splitlines()
    texts = {}
    for line in lines:
        match = re.match(r"\s*(R\d+)\s*:\s*(.*)", line)
        if match:
            texts[match.group(1)] = match.group(2)
    return pd.DataFrame(
        [{"rule_id": rule_id, "rule_text": texts.get(rule_id, "")} for rule_id in ids]
    )


def cached_generate(
    generator: Generator,
    prompt: str,
    cache: ArtifactCache,
    namespace: str,
) -> str:
    record = cache.location(
        namespace,
        {"model": generator.model_id, "prompt": prompt, "schema_version": 1},
        ".json",
    )
    if record.hit:
        return json.loads(record.path.read_text(encoding="utf-8"))["response"]
    response = generator.generate(prompt)
    if cache.policy != "disabled":
        record.path.parent.mkdir(parents=True, exist_ok=True)
        record.path.write_text(
            json.dumps({"response": response}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return response


def generate_study(
    fixed_cases: pd.DataFrame,
    variants: list[str],
    generator: Generator,
    cache: ArtifactCache,
) -> pd.DataFrame:
    rows = []
    for _, case in fixed_cases.iterrows():
        for variant in variants:
            rule_frame = _rule_frame(case)
            item_evidence = str(case.get("item_evidence_text", "")).splitlines()
            prompt = build_explanation_prompt(
                query_text=str(case["query_text"]),
                user_request=str(case["user_request"]),
                recommended_text=str(case["recommended_text"]),
                variant=variant,
                item_evidence=item_evidence,
                rule_evidence=rule_frame,
            )
            rows.append(
                {
                    **case.to_dict(),
                    "grounding_variant": variant,
                    "generated_explanation": cached_generate(
                        generator, prompt, cache, "explanation_generations"
                    ),
                    "generation_prompt_fingerprint": stable_fingerprint(prompt),
                }
            )
    return pd.DataFrame(rows)


def _unsupported_terms(text: str, evidence: str, terms: list[str]) -> list[str]:
    text_lower = str(text).lower()
    evidence_lower = str(evidence).lower()
    return [term for term in terms if term in text_lower and term not in evidence_lower]


def evaluate_explanations(explanations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in explanations.iterrows():
        explanation = str(row["generated_explanation"])
        evidence = evidence_for_variant(row)
        retrieved = set(re.findall(r"\bR\d+\b", str(row.get("rule_evidence_ids", ""))))
        cited = cited_rule_ids(explanation)
        grounded = row["grounding_variant"] in {"rule_rag", "hybrid_rag"}
        unsupported = _unsupported_terms(explanation, evidence, UNSUPPORTED_CLAIM_TERMS)
        occasion = _unsupported_terms(explanation, evidence, OCCASION_DRIFT_TERMS)
        candidate_types = detect_item_types(
            f"{row.get('recommended_category', '')} {row.get('recommended_text', '')}"
        )
        query_types = detect_item_types(
            f"{row.get('query_category', '')} {row.get('query_text', '')}"
        )
        extra_types = candidate_substitution_types(explanation, candidate_types | query_types)
        rows.append(
            {
                **row.to_dict(),
                "citation_presence": float(bool(cited)),
                "citation_correctness": float(cited.issubset(retrieved)),
                "citation_precision": (
                    len(cited & retrieved) / len(cited) if cited else float(not grounded)
                ),
                "unsupported_claim_count_evidence_aware": len(unsupported),
                "unsupported_claim_terms_evidence_aware": ", ".join(unsupported),
                "occasion_drift_count_evidence_aware": len(occasion),
                "occasion_drift_terms_evidence_aware": ", ".join(occasion),
                "evidence_overlap": evidence_overlap(explanation, evidence),
                "explanation_length_words": len(explanation.split()),
                "candidate_substitution_flag": bool(extra_types),
                "extra_item_types": ", ".join(sorted(extra_types)),
                "prompt_leakage_flag": bool(
                    re.search(
                        r"\b(prompt|retrieval|dataset|grounding variant)\b", explanation, re.I
                    )
                ),
            }
        )
    evaluated = pd.DataFrame(rows)
    summary = (
        evaluated.groupby("grounding_variant")
        .agg(
            num_explanations=("generated_explanation", "count"),
            citation_presence_rate=("citation_presence", "mean"),
            citation_correctness_rate=("citation_correctness", "mean"),
            mean_citation_precision=("citation_precision", "mean"),
            mean_unsupported_claim_count=(
                "unsupported_claim_count_evidence_aware",
                "mean",
            ),
            mean_occasion_drift_count=("occasion_drift_count_evidence_aware", "mean"),
            mean_evidence_overlap=("evidence_overlap", "mean"),
            mean_explanation_length_words=("explanation_length_words", "mean"),
            candidate_substitution_rate=("candidate_substitution_flag", "mean"),
            prompt_leakage_rate=("prompt_leakage_flag", "mean"),
        )
        .reset_index()
    )
    return evaluated, summary


def build_judge_prompt(row: pd.Series) -> str:
    evidence = evidence_for_variant(row) or (
        "No external evidence was provided. Judge consistency using only the query, "
        "request, and locked recommended item."
    )
    return f"""You are an independent evaluator of a fashion recommendation explanation.
The generator model is different from you. Do not infer which experimental method was used.

Query category: {row["query_category"]}
Query description: {row["query_text"]}
User request: {row["user_request"]}
Locked recommended category: {row["recommended_category"]}
Locked recommended item: {row["recommended_text"]}

Available grounding information:
{evidence}

Explanation:
{row["generated_explanation"]}

Score every dimension from 1 (poor) to 5 (excellent):
- faithfulness_to_available_information
- usefulness_to_user
- specificity
- style_appropriateness
- grounding_safety

Return only JSON. The values below are placeholders; replace every score after evaluation:
{{"faithfulness_to_available_information": 3, "usefulness_to_user": 3,
"specificity": 3, "style_appropriateness": 3, "grounding_safety": 3,
"brief_reason": "one short sentence"}}"""


def _parse_judge(response: str) -> dict[str, object]:
    def repair_json(candidate: str) -> str:
        # Local models occasionally emit a trailing comma or omit the comma
        # between two object properties, or add a standalone // explanation.
        # These repairs are deliberately narrow and leave JSON values unchanged.
        repaired = re.sub(r"(?m)^\s*//.*$", "", candidate)
        if '"claims"' in repaired:
            repaired = re.sub(
                r'(})\s*,?\s*("brief_reason"\s*:)',
                r"\1\n  ],\n\2",
                repaired,
                count=1,
            )
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r'("(?:[^"\\]|\\.)*")\s*\n(\s*")', r"\1,\n\2", repaired)
        return repaired

    parsed = None
    decoder = json.JSONDecoder()
    for start in (index for index, value in enumerate(response) if value == "{"):
        for candidate in (response[start:], repair_json(response[start:])):
            try:
                possible, _ = decoder.raw_decode(candidate)
                if isinstance(possible, dict) and all(
                    dimension in possible for dimension in JUDGE_DIMENSIONS
                ):
                    parsed = possible
                    break
            except json.JSONDecodeError:
                continue
        if parsed is not None:
            break
    if parsed is None:
        parsed = json.loads(repair_json(response))
    for dimension in JUDGE_DIMENSIONS:
        parsed[dimension] = max(1, min(5, int(parsed[dimension])))
    parsed["brief_reason"] = str(parsed.get("brief_reason", ""))
    claims = parsed.get("claims", [])
    if not isinstance(claims, list):
        claims = []
    normalized_claims = []
    for claim in claims[:3]:
        if not isinstance(claim, dict):
            continue
        raw_support = str(claim.get("support", "")).strip()
        folded = re.sub(r"[\s-]+", "_", raw_support.lower())
        if folded.startswith("unsupported"):
            support = "unsupported"
            compliant = folded == "unsupported"
        elif folded.startswith("supported") or folded == "verified":
            support = "supported"
            compliant = folded == "supported"
        elif folded.startswith(("not_verifiable", "unverifiable", "unverified")):
            support = "not_verifiable"
            compliant = folded == "not_verifiable"
        else:
            # A descriptive answer is not a valid categorical support verdict.
            # Treat it conservatively rather than inferring a favorable label.
            support = "not_verifiable"
            compliant = False
        normalized_claims.append(
            {
                **claim,
                "support_raw": raw_support,
                "support": support,
                "support_label_compliant": compliant,
            }
        )
    parsed["claims"] = normalized_claims
    return parsed


def judge_explanations(
    explanations: pd.DataFrame,
    judge: Generator,
    cache: ArtifactCache,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results, errors = [], []
    for index, row in explanations.iterrows():
        prompt = build_judge_prompt(row)
        try:
            response = cached_generate(judge, prompt, cache, "independent_judge")
            parsed = _parse_judge(response)
            results.append(
                {
                    **row.to_dict(),
                    **parsed,
                    "overall_judge_score": float(
                        np.mean([parsed[dimension] for dimension in JUDGE_DIMENSIONS])
                    ),
                    "judge_model": judge.model_id,
                    "raw_judge_response": response,
                }
            )
        except Exception as error:
            errors.append({"row_index": index, "error": repr(error)})
    result_frame = pd.DataFrame(results)
    error_frame = pd.DataFrame(errors, columns=["row_index", "error"])
    if result_frame.empty:
        return result_frame, error_frame, pd.DataFrame()
    summary = (
        result_frame.groupby("grounding_variant")[[*JUDGE_DIMENSIONS, "overall_judge_score"]]
        .mean()
        .reset_index()
    )
    return result_frame, error_frame, summary


def evaluate_rag_retrieval(
    fixed_cases: pd.DataFrame,
    knowledge_base: pd.DataFrame,
    cutoffs: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kb = knowledge_base.set_index("rule_id", drop=False)
    rows = []
    for _, case in fixed_cases.iterrows():
        ids = re.findall(r"\bR\d+\b", str(case.get("rule_evidence_ids", "")))
        query_group = str(case["query_group"])
        target_category = str(case["target_category"])
        applicable = knowledge_base[
            (knowledge_base["recommended_category"].astype(str) == target_category)
            & knowledge_base["input_category"]
            .astype(str)
            .apply(_input_category_contains, args=(query_group,))
        ]
        total_relevant = len(applicable)
        relevance = []
        reliability = []
        for rule_id in ids:
            if rule_id not in kb.index:
                relevance.append(0)
                continue
            rule = kb.loc[rule_id]
            inputs = {part.strip() for part in str(rule["input_category"]).split(",")}
            relevant = (
                query_group in inputs and str(rule["recommended_category"]) == target_category
            )
            relevance.append(int(relevant))
            reliability.append(str(rule["source_reliability"]))
        metric = {}
        labels = np.asarray(relevance, dtype=int)
        for k in [value for value in cutoffs if value <= len(relevance)]:
            top = labels[:k]
            hits = int(top.sum())
            discounts = 1.0 / np.log2(np.arange(2, k + 2))
            dcg = float((top * discounts).sum())
            ideal_hits = min(total_relevant, k)
            idcg = float(discounts[:ideal_hits].sum()) if ideal_hits else 0.0
            metric.update(
                {
                    f"precision_at_{k}": hits / k,
                    f"recall_at_{k}": hits / total_relevant if total_relevant else 0.0,
                    f"hit_rate_at_{k}": float(hits > 0),
                    f"ndcg_at_{k}": dcg / idcg if idcg else 0.0,
                }
            )
        positive_ranks = np.flatnonzero(labels) + 1
        metric["reciprocal_rank"] = 1.0 / positive_ranks[0] if len(positive_ranks) else 0.0
        rows.append(
            {
                "paper_case_id": case["paper_case_id"],
                "target_category": case["target_category"],
                "retrieved_rule_count": len(ids),
                "evidence_coverage": float(bool(ids)),
                "category_compatible_rate": float(np.mean(relevance)) if relevance else 0.0,
                "high_reliability_rule_count": sum(value == "high" for value in reliability),
                "unique_rule_count": len(set(ids)),
                "total_applicable_rules": total_relevant,
                **metric,
            }
        )
    results = pd.DataFrame(rows)
    numeric = results.select_dtypes(include=np.number).columns
    summary = results.groupby("target_category")[numeric].mean().reset_index()
    return results, summary


def explanation_statistics(
    automatic: pd.DataFrame,
    judged: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    automatic_tests = compare_variants(
        automatic,
        "paper_case_id",
        "grounding_variant",
        [
            "unsupported_claim_count_evidence_aware",
            "occasion_drift_count_evidence_aware",
            "evidence_overlap",
            "candidate_substitution_flag",
        ],
        config.evaluation.bootstrap_samples,
        config.evaluation.confidence_level,
        config.project.seed,
    ).assign(metric_family="automatic")
    judge_tests = compare_variants(
        judged,
        "paper_case_id",
        "grounding_variant",
        [*JUDGE_DIMENSIONS, "overall_judge_score"],
        config.evaluation.bootstrap_samples,
        config.evaluation.confidence_level,
        config.project.seed,
    ).assign(metric_family="independent_llm_judge")
    return pd.concat([automatic_tests, judge_tests], ignore_index=True)


def write_study_outputs(
    output_dir: Path,
    **frames: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
