"""Decision-gated final_eval_v2 explanation generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..cache import ArtifactCache, file_fingerprint, stable_fingerprint
from ..generation import build_explanation_prompt
from ..models.base import Generator
from ..reproducibility import git_revision
from .study import _rule_frame, cached_generate

VARIANTS = ("no_rag", "item_rag", "rule_rag", "hybrid_rag")


def _load_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object.")
    return value


def validate_generation_inputs(
    cases: pd.DataFrame,
    *,
    input_path: Path,
    reranking_selection_path: Path,
    hybrid_selection_path: Path,
    decision_path: Path,
    freeze_path: Path,
) -> tuple[dict[str, object], str]:
    required = {
        "paper_case_id",
        "research_split",
        "stage1_packet_protocol",
        "stage1_packet_hash",
        "query_text",
        "user_request",
        "recommended_text",
        "item_evidence_text",
        "rule_evidence_ids",
        "rule_evidence_text",
    }
    missing = required - set(cases.columns)
    if missing:
        raise ValueError(f"Final v2 generation packets are missing: {sorted(missing)}")
    if set(cases["research_split"].astype(str)) != {"test"}:
        raise ValueError("Final v2 explanations require test packets only.")
    if set(cases["stage1_packet_protocol"].astype(str)) != {"final_eval_v2_selected"}:
        raise ValueError("Final v2 explanations cannot use legacy packets.")
    packet_hashes = set(cases["stage1_packet_hash"].astype(str))
    if len(packet_hashes) != 1 or not next(iter(packet_hashes)).strip():
        raise ValueError("Final v2 test packets must share one non-empty packet hash.")
    if cases["paper_case_id"].duplicated().any():
        raise ValueError("Final v2 test packet case IDs must be unique.")
    decision = _load_json(decision_path, "Stage 1 decision")
    if decision.get("decision") != "regenerate_all_variants":
        raise ValueError("Stage 3 generation requires Gate A regenerate_all_variants.")
    reranking = _load_json(reranking_selection_path, "reranking selection")
    if (
        reranking.get("selection_policy") != "evidence_in_loop_pareto_v2"
        or float(reranking.get("clip_weight", -1)) != 0.75
        or float(reranking.get("evidence_weight", -1)) != 0.25
    ):
        raise ValueError("Stage 3 requires the frozen 0.75/0.25 evidence-in-loop reranker.")
    hybrid = _load_json(hybrid_selection_path, "Stage 2 Hybrid selection")
    if hybrid.get("selected_on") != "validation" or hybrid.get("candidate_type") != "hybrid":
        raise ValueError("Stage 3 requires an eligible validation-selected Hybrid config.")
    freeze = _load_json(freeze_path, "final v2 freeze manifest")
    source = git_revision()
    if source.get("dirty") or source.get("commit") != freeze.get("source", {}).get("commit"):
        raise ValueError("Current clean source commit must match the final freeze manifest.")
    frozen_inputs = freeze.get("input_hashes", {})
    if frozen_inputs.get(str(input_path)) != file_fingerprint(input_path):
        raise ValueError("Test packets do not match the final freeze manifest.")
    selections = freeze.get("selections", {})
    if selections.get("reranking", {}).get("sha256") != file_fingerprint(reranking_selection_path):
        raise ValueError("Reranking selection does not match the final freeze manifest.")
    if selections.get("hybrid", {}).get("sha256") != file_fingerprint(hybrid_selection_path):
        raise ValueError("Hybrid selection does not match the final freeze manifest.")
    if freeze.get("gate_definition", {}).get("decision") != "regenerate_all_variants":
        raise ValueError("Frozen decision gate does not authorize all-variant generation.")
    return hybrid, next(iter(packet_hashes))


def run_final_explanations_v2(
    *,
    cases: pd.DataFrame,
    generators: list[Generator],
    cache: ArtifactCache,
    output_dir: Path,
    report_path: Path,
    input_path: Path,
    reranking_selection_path: Path,
    hybrid_selection_path: Path,
    decision_path: Path,
    freeze_path: Path,
) -> dict[str, object]:
    hybrid, test_packet_hash = validate_generation_inputs(
        cases,
        input_path=input_path,
        reranking_selection_path=reranking_selection_path,
        hybrid_selection_path=hybrid_selection_path,
        decision_path=decision_path,
        freeze_path=freeze_path,
    )
    freeze_hash = file_fingerprint(freeze_path)
    word_budget = int(hybrid["max_words"])
    rule_limit = int(hybrid["rule_limit"])
    item_limit = int(hybrid["item_count"])
    prompt_order = str(hybrid["prompt_order"])
    rows, errors = [], []
    for generator in generators:
        for _, case in cases.iterrows():
            for variant in VARIANTS:
                prompt = build_explanation_prompt(
                    query_text=str(case["query_text"]),
                    user_request=str(case["user_request"]),
                    recommended_text=str(case["recommended_text"]),
                    variant=variant,
                    item_evidence=str(case.get("item_evidence_text", "")).splitlines(),
                    rule_evidence=_rule_frame(case),
                    max_words=word_budget,
                    rule_limit=rule_limit,
                    item_limit=item_limit,
                    prompt_order=prompt_order,
                )
                context = {
                    "protocol": "final_eval_v2_generation",
                    "freeze_hash": freeze_hash,
                    "test_packet_hash": test_packet_hash,
                    "variant": variant,
                    "word_budget": word_budget,
                    "rule_limit": rule_limit,
                    "item_limit": item_limit,
                    "prompt_order": prompt_order,
                }
                try:
                    response = cached_generate(
                        generator,
                        prompt,
                        cache,
                        "final_eval_v2_generations",
                        cache_context=context,
                    )
                    rows.append(
                        {
                            **case.to_dict(),
                            "recommended_category": case.get(
                                "recommended_category", case.get("target_category", "")
                            ),
                            "grounding_variant": variant,
                            "generated_explanation": response,
                            "generation_model": generator.model_id,
                            "generation_protocol": "final_eval_v2",
                            "evaluation_protocol": "v2",
                            "max_words": word_budget,
                            "rule_limit": rule_limit,
                            "item_limit": item_limit,
                            "prompt_order": prompt_order,
                            "generation_prompt_fingerprint": stable_fingerprint(
                                {"prompt": prompt, **context}
                            ),
                            "freeze_manifest_hash": freeze_hash,
                        }
                    )
                except Exception as error:
                    errors.append(
                        {
                            "paper_case_id": case["paper_case_id"],
                            "grounding_variant": variant,
                            "generation_model": generator.model_id,
                            "error": repr(error),
                        }
                    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    explanations = pd.DataFrame(rows)
    error_frame = pd.DataFrame(
        errors, columns=["paper_case_id", "grounding_variant", "generation_model", "error"]
    )
    explanations.to_csv(output_dir / "explanations.csv", index=False)
    error_frame.to_csv(output_dir / "generation_errors.csv", index=False)
    expected = len(cases) * len(VARIANTS) * len(generators)
    if errors or len(explanations) != expected:
        raise RuntimeError(
            f"Stage 3 incomplete: expected {expected}, wrote {len(explanations)}, "
            f"errors={len(errors)}. Rerun to resume."
        )
    manifest = {
        "protocol": "final_eval_v2_generation",
        "decision": "regenerate_all_variants",
        "input": str(input_path),
        "input_hash": file_fingerprint(input_path),
        "test_packet_hash": test_packet_hash,
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_hash": freeze_hash,
        "hybrid_selection": hybrid,
        "variants": list(VARIANTS),
        "generators": [value.model_id for value in generators],
        "cases": len(cases),
        "rows": len(explanations),
        "errors": 0,
        "shared_word_budget": word_budget,
        "rule_limit": rule_limit,
        "item_limit": item_limit,
        "prompt_order": prompt_order,
    }
    manifest["artifact_fingerprint"] = stable_fingerprint(manifest)
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report_path.write_text(
        "# Final Evaluation v2 Generation Summary\n\n"
        f"Generated {len(explanations)} explanations for {len(cases)} frozen test cases, "
        f"four variants, and {len(generators)} generators.\n\n"
        f"Shared word budget: {word_budget}. Generation errors: 0.\n",
        encoding="utf-8",
    )
    return manifest
