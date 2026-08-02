"""Resumable, validation-only Hybrid-RAG v2 screening and selection."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ..cache import ArtifactCache, file_fingerprint, stable_fingerprint
from ..models.base import Generator
from .explanations import evidence_overlap
from .robustness import (
    HybridPromptSpec,
    generate_hybrid_ablations,
    select_hybrid_finalists,
    validate_stage1_validation_packets,
)
from .study import cached_generate, evaluate_explanations


def balanced_screening_subset(cases: pd.DataFrame, per_category: int, seed: int) -> pd.DataFrame:
    """Freeze a deterministic category-balanced subset without depending on input order."""

    parts = []
    for _, group in cases.groupby("target_category", sort=True):
        count = min(per_category, len(group))
        parts.append(group.sample(n=count, random_state=seed).sort_values("paper_case_id"))
    return pd.concat(parts, ignore_index=True).sort_values("paper_case_id").reset_index(drop=True)


def _extract_json(response: str) -> dict[str, object]:
    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if not match:
        raise ValueError("Hybrid validation judge did not return a JSON object.")
    value = json.loads(match.group(0))
    required = {
        "fashion_claim_count",
        "hallucinated_fashion_claim_count",
        "styling_claim_count",
        "rule_supported_styling_claim_count",
        "evidence_misuse",
        "candidate_substitution",
        "general_clarity",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"Hybrid validation judge response is missing: {sorted(missing)}")
    for key in (
        "fashion_claim_count",
        "hallucinated_fashion_claim_count",
        "styling_claim_count",
        "rule_supported_styling_claim_count",
    ):
        value[key] = max(0, int(value[key]))
    for key in ("evidence_misuse", "candidate_substitution"):
        raw = value[key]
        value[key] = raw if isinstance(raw, bool) else str(raw).lower() == "true"
    clarity = float(value["general_clarity"])
    if not 1 <= clarity <= 5:
        raise ValueError("Hybrid validation general_clarity must be between 1 and 5.")
    value["general_clarity"] = clarity
    return value


def _judge_prompt(row: pd.Series) -> str:
    return f"""Evaluate one Hybrid-RAG validation explanation using only the supplied data.
Count atomic fashion claims and styling claims. A hallucinated fashion claim is not supported
by the query, locked item, catalogue evidence, or expert rules. A rule-supported styling claim
must be entailed by an expert rule. Evidence misuse means evidence is applied to the wrong item
or treated as an attribute of the locked item. Candidate substitution means another product is
recommended or presented as the locked recommendation.

Query: {row['query_text']}
Request: {row['user_request']}
Locked item: {row['recommended_text']}
Catalogue evidence:
{row.get('item_evidence_text', '')}
Expert rules:
{row.get('rule_evidence_text', '')}
Explanation:
{row['generated_explanation']}

Return one JSON object only with nonnegative integer fashion_claim_count,
hallucinated_fashion_claim_count, styling_claim_count, rule_supported_styling_claim_count;
boolean evidence_misuse and candidate_substitution; and general_clarity from 1 to 5."""


def judge_hybrid_explanations(
    explanations: pd.DataFrame,
    judge: Generator,
    cache: ArtifactCache,
    *,
    packet_hash: str,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for _, explanation in explanations.iterrows():
        prompt = _judge_prompt(explanation)
        identity = {
            "stage1_packet_hash": packet_hash,
            "phase": phase,
            "judge_schema_version": "hybrid_v2",
        }
        try:
            response = cached_generate(
                judge, prompt, cache, "hybrid_v2_judges", cache_context=identity
            )
            rows.append(
                {
                    **explanation.to_dict(),
                    **_extract_json(response),
                    "judge_model": judge.model_id,
                    "raw_judge_response": response,
                    "judge_prompt_fingerprint": stable_fingerprint(
                        {"prompt": prompt, **identity}
                    ),
                }
            )
        except Exception as error:
            errors.append(
                {
                    "paper_case_id": explanation["paper_case_id"],
                    "grounding_variant": explanation["grounding_variant"],
                    "error": repr(error),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(
        errors, columns=["paper_case_id", "grounding_variant", "error"]
    )


def summarize_hybrid_phase(
    explanations: pd.DataFrame, judged: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    automatic_input = explanations.assign(grounding_variant="hybrid_rag")
    automatic, _ = evaluate_explanations(automatic_input)
    automatic["grounding_variant"] = explanations["grounding_variant"].to_numpy()
    automatic["rule_evidence_overlap"] = automatic.apply(
        lambda row: evidence_overlap(row["generated_explanation"], row["rule_evidence_text"]),
        axis=1,
    )
    automatic["item_evidence_overlap"] = automatic.apply(
        lambda row: evidence_overlap(row["generated_explanation"], row["item_evidence_text"]),
        axis=1,
    )
    keys = ["paper_case_id", "grounding_variant"]
    metrics = judged[
        keys
        + [
            "fashion_claim_count",
            "hallucinated_fashion_claim_count",
            "styling_claim_count",
            "rule_supported_styling_claim_count",
            "evidence_misuse",
            "candidate_substitution",
            "general_clarity",
        ]
    ]
    per_case = automatic.merge(metrics, on=keys, validate="one_to_one")
    rows = []
    group_columns = [
        "grounding_variant",
        "max_words",
        "rule_limit",
        "item_limit",
        "prompt_order",
        "candidate_type",
    ]
    for values, group in per_case.groupby(group_columns, sort=False):
        fashion_claims = int(group["fashion_claim_count"].sum())
        styling_claims = int(group["styling_claim_count"].sum())
        rows.append(
            {
                **dict(zip(group_columns, values, strict=True)),
                "cases": len(group),
                "hallucinated_claim_rate": (
                    group["hallucinated_fashion_claim_count"].sum() / fashion_claims
                    if fashion_claims
                    else 0.0
                ),
                "rule_supported_claim_rate": (
                    group["rule_supported_styling_claim_count"].sum() / styling_claims
                    if styling_claims
                    else 0.0
                ),
                "evidence_misuse_rate": group["evidence_misuse"].mean(),
                "candidate_substitution_rate": group["candidate_substitution"].mean(),
                "rule_evidence_overlap": group["rule_evidence_overlap"].mean(),
                "item_evidence_overlap": group["item_evidence_overlap"].mean(),
                "general_clarity": group["general_clarity"].mean(),
                "mean_words": group["explanation_length_words"].mean(),
            }
        )
    return per_case, pd.DataFrame(rows)


def run_hybrid_validation_v2(
    *,
    cases: pd.DataFrame,
    specs: list[HybridPromptSpec],
    generator: Generator,
    judge: Generator,
    cache: ArtifactCache,
    output_dir: Path,
    report_path: Path,
    screening_cases_per_category: int,
    finalist_count: int,
    practical_tie: float,
    seed: int,
    input_path: Path,
) -> dict[str, object]:
    packet_hash = validate_stage1_validation_packets(cases)
    subset = balanced_screening_subset(cases, screening_cases_per_category, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def execute_phase(
        phase_cases: pd.DataFrame, phase_specs: list[HybridPromptSpec], phase: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        generated = generate_hybrid_ablations(
            phase_cases,
            phase_specs,
            generator,
            cache,
            cache_context={"stage1_packet_hash": packet_hash, "phase": phase},
        )
        judged, errors = judge_hybrid_explanations(
            generated, judge, cache, packet_hash=packet_hash, phase=phase
        )
        if not errors.empty:
            errors.to_csv(output_dir / f"{phase}_judge_errors.csv", index=False)
            raise RuntimeError(f"{len(errors)} {phase} Hybrid judge calls failed; rerun to resume.")
        per_case, summary = summarize_hybrid_phase(generated, judged)
        generated.to_csv(output_dir / f"{phase}_explanations.csv", index=False)
        judged.to_csv(output_dir / f"{phase}_judge_results.csv", index=False)
        per_case.to_csv(output_dir / f"{phase}_per_case.csv", index=False)
        summary.to_csv(output_dir / f"{phase}_grid_results.csv", index=False)
        return per_case, summary

    _, screening = execute_phase(subset, specs, "screening")
    finalists = select_hybrid_finalists(
        screening, practical_tie=practical_tie, finalist_count=finalist_count
    )
    finalists.to_csv(output_dir / "screening_finalists.csv", index=False)
    spec_by_name = {spec.name: spec for spec in specs}
    finalist_specs = [spec_by_name[name] for name in finalists["grounding_variant"]]
    _, final_results = execute_phase(cases, finalist_specs, "finalist")
    selection = select_hybrid_finalists(
        final_results, practical_tie=practical_tie, finalist_count=1
    ).iloc[0]
    selected = {
        "name": selection["grounding_variant"],
        "max_words": int(selection["max_words"]),
        "rule_limit": int(selection["rule_limit"]),
        "item_count": int(selection["item_limit"]),
        "item_limit": int(selection["item_limit"]),
        "prompt_order": selection["prompt_order"],
        "candidate_type": selection["candidate_type"],
        "selected_on": "validation",
        "selection_protocol": "priority_v2_no_weighted_composite",
        "stage1_packet_hash": packet_hash,
        "packet_source_protocol": "final_eval_v2_selected",
        "generator_model": generator.model_id,
        "judge_model": judge.model_id,
        "practical_tie": practical_tie,
    }
    (output_dir / "selected_hybrid_config.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    final_results.to_csv(output_dir / "grid_results.csv", index=False)
    manifest = {
        "protocol": "final_eval_v2_hybrid_validation",
        "input": str(input_path),
        "input_hash": file_fingerprint(input_path),
        "stage1_packet_hash": packet_hash,
        "screening_cases": len(subset),
        "validation_cases": len(cases),
        "grid_size": len(specs),
        "finalist_count": len(finalist_specs),
        "selected_hybrid_config": selected,
    }
    manifest["artifact_fingerprint"] = stable_fingerprint(manifest)
    (output_dir / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report_path.write_text(
        "# Final Evaluation v2 Hybrid Validation\n\n"
        f"Stage 1 packet hash: `{packet_hash}`\n\n"
        f"Screened {len(specs)} configurations on {len(subset)} balanced validation cases; "
        f"evaluated {len(finalist_specs)} finalists on all {len(cases)} validation cases.\n\n"
        "Selection used the frozen priority hierarchy with no weighted composite.\n\n"
        "## Selected configuration\n\n```json\n"
        f"{json.dumps(selected, indent=2)}\n```\n",
        encoding="utf-8",
    )
    return manifest
