"""Command-line interface for reproducible project operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="efr", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a validated configuration value using dotted notation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-config", help="Validate and print the resolved config.")
    subparsers.add_parser("doctor", help="Check the local runtime and configured services.")
    subparsers.add_parser("prepare-data", help="Load and cache processed dataset metadata.")
    subparsers.add_parser("audit-kb", help="Audit the configured knowledge base.")
    subparsers.add_parser(
        "import-legacy",
        help="Verify and import archived notebook embeddings into the new cache.",
    )
    subparsers.add_parser(
        "evaluate-ranking",
        help="Run the controlled text-vs-CLIP ranking evaluation.",
    )
    recommend_parser = subparsers.add_parser(
        "recommend",
        help="Run one end-to-end recommendation and optionally generate explanations.",
    )
    recommend_parser.add_argument("--query-item-id")
    recommend_parser.add_argument("--query-image")
    recommend_parser.add_argument("--query-text")
    recommend_parser.add_argument("--query-category")
    recommend_parser.add_argument("--target-category", required=True)
    recommend_parser.add_argument("--request", required=True)
    recommend_parser.add_argument("--generate", action="store_true")
    study_parser = subparsers.add_parser(
        "run-explanation-study",
        help="Run generation, RAG, faithfulness, independent judging, and statistics.",
    )
    study_parser.add_argument(
        "--input",
        default=(
            "archive/original_notebook_outputs/results/paper_experiments/"
            "kb_v3_experiment/fixed_paper_explanation_dataset_v3_candidate_type_filtered.csv"
        ),
    )
    study_parser.add_argument("--limit", type=int, default=None)
    study_parser.add_argument("--skip-judge", action="store_true")
    report_parser = subparsers.add_parser(
        "build-final-report",
        help="Build the final systematic tables, figures, manifests, and report.",
    )
    report_parser.add_argument("--baseline-run", required=True)
    report_parser.add_argument("--improved-run", required=True)
    report_parser.add_argument("--study-run", required=True)
    report_parser.add_argument("--output", default="outputs/final")
    robustness_report_parser = subparsers.add_parser(
        "build-robustness-report",
        help="Build the before-versus-after systematic robustness report.",
    )
    robustness_report_parser.add_argument(
        "--baseline", default="outputs/robustness/before_baseline"
    )
    robustness_report_parser.add_argument(
        "--robustness-study", default="outputs/robustness/final_study"
    )
    robustness_report_parser.add_argument(
        "--heldout-ranking", default="outputs/robustness/heldout_ranking"
    )
    robustness_report_parser.add_argument("--output", default="outputs/robustness/final_report")
    cases_parser = subparsers.add_parser(
        "build-study-cases",
        help="Rebuild the 100 fixed explanation rows through the modular pipeline.",
    )
    cases_parser.add_argument(
        "--schedule",
        default=(
            "archive/original_notebook_outputs/results/paper_experiments/"
            "kb_v3_experiment/fixed_paper_explanation_dataset_v3_candidate_type_filtered.csv"
        ),
    )
    cases_parser.add_argument("--output", default="outputs/modular_study_cases.csv")
    schedule_parser = subparsers.add_parser(
        "build-robustness-schedules",
        help="Create deterministic, outfit-disjoint balanced research schedules.",
    )
    schedule_parser.add_argument("--output-dir", default="outputs/robustness/schedules")
    ablation_parser = subparsers.add_parser(
        "run-hybrid-ablations",
        help="Run validation-only Hybrid-RAG prompt and evidence-budget ablations.",
    )
    ablation_parser.add_argument("--input", default="outputs/robustness/validation_cases.csv")
    ablation_parser.add_argument("--output-dir", default="outputs/robustness/hybrid_ablations")
    ablation_parser.add_argument("--limit", type=int, default=None)
    ablation_parser.add_argument("--skip-judge", action="store_true")
    hybrid_v2_parser = subparsers.add_parser(
        "run-hybrid-validation-v2",
        help="Run resumable staged Hybrid-RAG selection on frozen v2 validation packets.",
    )
    hybrid_v2_parser.add_argument(
        "--input", default="outputs/final_eval_v2/prepared/validation/locked_packets.csv"
    )
    hybrid_v2_parser.add_argument("--output-dir", default="outputs/final_eval_v2/hybrid_validation")
    freeze_v2_parser = subparsers.add_parser(
        "freeze-final-eval-v2", help="Create the immutable clean-source v2 freeze."
    )
    freeze_v2_parser.add_argument("--destination", default="outputs/final_eval_v2/freeze")
    generation_v2_parser = subparsers.add_parser(
        "run-final-explanations-v2", help="Generate all four final variants under frozen Gate A."
    )
    generation_v2_parser.add_argument(
        "--input", default="outputs/final_eval_v2/prepared/test/locked_packets.csv"
    )
    generation_v2_parser.add_argument(
        "--reranking-selection",
        default="outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json",
    )
    generation_v2_parser.add_argument(
        "--hybrid-selection",
        default="outputs/final_eval_v2/hybrid_validation/selected_hybrid_config.json",
    )
    generation_v2_parser.add_argument(
        "--decision", default="outputs/final_eval_v2/decision_gate/decision.json"
    )
    generation_v2_parser.add_argument(
        "--freeze", default="outputs/final_eval_v2/freeze/FINAL_FREEZE_MANIFEST.json"
    )
    generation_v2_parser.add_argument("--output-dir", default="outputs/final_eval_v2/explanations")
    tuning_parser = subparsers.add_parser(
        "tune-reranking",
        help="Select evidence-reranking weight using validation outfits only.",
    )
    tuning_parser.add_argument(
        "--input", default="outputs/robustness/schedules/validation_schedule.csv"
    )
    tuning_parser.add_argument("--output-dir", default="outputs/robustness/reranking_tuning")
    heldout_parser = subparsers.add_parser(
        "evaluate-heldout-ranking",
        help="Evaluate the validation-selected reranker on frozen test outfits.",
    )
    heldout_parser.add_argument("--input", default="outputs/robustness/schedules/test_schedule.csv")
    heldout_parser.add_argument("--output-dir", default="outputs/robustness/heldout_ranking")
    heldout_parser.add_argument(
        "--selection",
        default="outputs/robustness/reranking_tuning/selected_weight.json",
        help="Frozen validation-selected reranking artifact.",
    )
    fusion_v2_parser = subparsers.add_parser(
        "tune-clip-fusion",
        help="Select final_eval_v2 CLIP fusion weights from a prepared validation bundle.",
    )
    fusion_v2_parser.add_argument("--bundle", default="outputs/final_eval_v2/prepared/validation")
    fusion_v2_parser.add_argument(
        "--output-dir", default="outputs/final_eval_v2/validation/fusion_tuning"
    )
    retrieval_v2_parser = subparsers.add_parser(
        "evaluate-final-retrieval-v2",
        help="Evaluate frozen v2 fusion/reranking settings on a prepared test bundle.",
    )
    retrieval_v2_parser.add_argument("--bundle", default="outputs/final_eval_v2/prepared/test")
    retrieval_v2_parser.add_argument(
        "--fusion-selection",
        default="outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json",
    )
    retrieval_v2_parser.add_argument(
        "--reranking-selection",
        default="outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json",
    )
    retrieval_v2_parser.add_argument(
        "--locked-packets", default="outputs/final_eval_v2/prepared/test/locked_packets.csv"
    )
    retrieval_v2_parser.add_argument("--output-dir", default="outputs/final_eval_v2/retrieval/test")
    gate_v2_parser = subparsers.add_parser(
        "compare-locked-artifacts-v2",
        help="Compare legacy and v2 locked recommendation/evidence packets.",
    )
    gate_v2_parser.add_argument("--legacy-packets", required=True)
    gate_v2_parser.add_argument(
        "--v2-packets",
        default=("outputs/final_eval_v2/retrieval/test/locked_recommendation_evidence_packets.csv"),
    )
    gate_v2_parser.add_argument("--output-dir", default="outputs/final_eval_v2/decision_gate")
    prepare_v2_parser = subparsers.add_parser(
        "prepare-final-retrieval-v2-bundle",
        help="Materialize a v2 Stage 1 bundle from frozen files without rebuilding embeddings.",
    )
    prepare_v2_parser.add_argument("--split", choices=["validation", "test"], required=True)
    prepare_v2_parser.add_argument("--schedule", required=True)
    prepare_v2_parser.add_argument("--candidate-sets", required=True)
    prepare_v2_parser.add_argument("--target-embedding-dir", required=True)
    prepare_v2_parser.add_argument("--query-embedding-dir", required=True)
    prepare_v2_parser.add_argument("--output-dir", required=True)
    rerank_v2_parser = subparsers.add_parser(
        "tune-reranking-v2",
        help="Select v2 evidence-reranking weights from a prepared validation bundle.",
    )
    rerank_v2_parser.add_argument("--bundle", default="outputs/final_eval_v2/prepared/validation")
    rerank_v2_parser.add_argument(
        "--fusion-selection",
        default="outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json",
    )
    rerank_v2_parser.add_argument(
        "--output-dir", default="outputs/final_eval_v2/validation/reranking_tuning"
    )
    evidence_selection_parser = subparsers.add_parser(
        "select-evidence-in-loop-reranking-v2",
        help="Freeze the validation Pareto/knee reranker and retain CLIP-only as baseline.",
    )
    evidence_selection_parser.add_argument(
        "--summary",
        default="outputs/final_eval_v2/validation/reranking_tuning/validation_summary.csv",
    )
    evidence_selection_parser.add_argument(
        "--output",
        default="outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json",
    )
    packets_v2_parser = subparsers.add_parser(
        "create-locked-packets-v2",
        help="Freeze locked packets produced by selected v2 retrieval settings.",
    )
    packets_v2_parser.add_argument("--split", choices=["validation", "test"], required=True)
    packets_v2_parser.add_argument("--source-cases", required=True)
    packets_v2_parser.add_argument(
        "--fusion-selection",
        default="outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json",
    )
    packets_v2_parser.add_argument(
        "--reranking-selection",
        default="outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json",
    )
    packets_v2_parser.add_argument("--output", required=True)
    materialize_v2_parser = subparsers.add_parser(
        "materialize-final-retrieval-v2-inputs",
        help="Resolve cached embeddings and materialize fresh v2 retrieval inputs.",
    )
    materialize_v2_parser.add_argument("--split", choices=["validation", "test"], required=True)
    materialize_v2_parser.add_argument("--schedule", required=True)
    materialize_v2_parser.add_argument("--target-items", required=True)
    materialize_v2_parser.add_argument("--candidate-source", required=True)
    materialize_v2_parser.add_argument(
        "--output-root", default="outputs/final_eval_v2/materialized"
    )
    query_v2_parser = subparsers.add_parser(
        "materialize-final-retrieval-v2-query-embeddings",
        help="Explicitly compute only missing v2 query embeddings; never target embeddings.",
    )
    query_v2_parser.add_argument("--split", choices=["validation", "test"], required=True)
    query_v2_parser.add_argument("--schedule", required=True)
    query_v2_parser.add_argument(
        "--approve-compute-query-embeddings",
        action="store_true",
        help="Required acknowledgement before query model execution.",
    )
    selected_cases_v2_parser = subparsers.add_parser(
        "materialize-final-retrieval-v2-selected-cases",
        help="Materialize fresh selected-v2 cases after fusion/reranking selection.",
    )
    selected_cases_v2_parser.add_argument("--split", choices=["validation", "test"], required=True)
    selected_cases_v2_parser.add_argument("--schedule", required=True)
    selected_cases_v2_parser.add_argument("--source-cases", required=True)
    selected_cases_v2_parser.add_argument(
        "--fusion-selection",
        default="outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json",
    )
    selected_cases_v2_parser.add_argument(
        "--reranking-selection",
        default="outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json",
    )
    selected_cases_v2_parser.add_argument("--output", required=True)
    target_source_parser = subparsers.add_parser(
        "materialize-final-eval-v2-target-items",
        help="Freeze the fresh v2 target-item row order against cached target embeddings.",
    )
    target_source_parser.add_argument(
        "--output", default="outputs/final_eval_v2/sources/target_items.parquet"
    )
    candidate_source_parser = subparsers.add_parser(
        "produce-final-eval-v2-candidates",
        help="Build deterministic candidate pools with fresh v2 evidence scores.",
    )
    candidate_source_parser.add_argument("--split", choices=["validation", "test"], required=True)
    candidate_source_parser.add_argument("--schedule", required=True)
    candidate_source_parser.add_argument(
        "--target-items", default="outputs/final_eval_v2/sources/target_items.parquet"
    )
    candidate_source_parser.add_argument("--output", required=True)
    selected_source_parser = subparsers.add_parser(
        "produce-final-eval-v2-selected-cases",
        help="Produce fresh locked recommendations/evidence using validation-selected settings.",
    )
    selected_source_parser.add_argument("--split", choices=["validation", "test"], required=True)
    selected_source_parser.add_argument("--schedule", required=True)
    selected_source_parser.add_argument(
        "--target-items", default="outputs/final_eval_v2/sources/target_items.parquet"
    )
    selected_source_parser.add_argument("--candidate-sets", required=True)
    selected_source_parser.add_argument("--bundle", required=True)
    selected_source_parser.add_argument(
        "--fusion-selection",
        default="outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json",
    )
    selected_source_parser.add_argument(
        "--reranking-selection",
        default="outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json",
    )
    selected_source_parser.add_argument("--output", required=True)
    inspect_v2_parser = subparsers.add_parser(
        "inspect-final-eval-v2-readiness",
        help="Read-only manifest/hash/protocol preflight for v2 stages.",
    )
    inspect_v2_parser.add_argument(
        "--validation-schedule", default="outputs/robustness/schedules/validation_schedule.csv"
    )
    inspect_v2_parser.add_argument(
        "--test-schedule", default="outputs/robustness/schedules/test_schedule.csv"
    )
    robustness_parser = subparsers.add_parser(
        "run-robustness-study",
        help="Run the frozen multi-generator, multi-judge held-out study.",
    )
    robustness_parser.add_argument("--input", default="outputs/robustness/test_cases.csv")
    robustness_parser.add_argument(
        "--selection",
        default="outputs/robustness/hybrid_ablations/validation_selection.csv",
    )
    robustness_parser.add_argument("--output-dir", default="outputs/robustness/final_study")
    robustness_parser.add_argument("--limit", type=int, default=None)
    freeze_parser = subparsers.add_parser(
        "freeze-baseline",
        help="Copy and fingerprint the before-baseline without allowing overwrite.",
    )
    freeze_parser.add_argument("--source", default="outputs/final")
    freeze_parser.add_argument("--destination", default="outputs/robustness/before_baseline")
    subparsers.add_parser(
        "build-embeddings",
        help="Build or reuse all target text and multimodal embeddings.",
    )
    subparsers.add_parser(
        "build-indexes",
        help="Build or reuse persistent category-aware FAISS indexes.",
    )
    caption_parser = subparsers.add_parser(
        "caption-image",
        help="Caption a user image with the configured optional caption model.",
    )
    caption_parser.add_argument("--image", required=True)
    subparsers.add_parser("show-plan", help="Show the stages selected by the config.")
    return parser


def command_validate(config_path: str, overrides: list[str]) -> int:
    config = load_config(config_path, overrides)
    print(json.dumps(config.model_dump(mode="json"), indent=2))
    return 0


def command_doctor(config_path: str, overrides: list[str]) -> int:
    from .reproducibility import environment_manifest, ollama_manifest

    config = load_config(config_path, overrides)
    report = environment_manifest()
    report["config_valid"] = True
    report["knowledge_base_exists"] = config.paths.knowledge_base.exists()
    report["configured_models"] = {name: model.name for name, model in config.models}
    installed_ollama = ollama_manifest()
    report["ollama_models"] = installed_ollama
    installed_by_name = {
        model["name"].split(":", 1)[0]: model["digest"] for model in installed_ollama
    }
    llm_checks = {}
    for role in ("generator", "judge"):
        model = getattr(config.models, role)
        installed_digest = installed_by_name.get(model.name.split(":", 1)[0])
        llm_checks[role] = {
            "available": installed_digest is not None,
            "installed_digest": installed_digest,
            "expected_digest": model.expected_digest,
            "digest_matches": (
                model.expected_digest is None
                or (
                    installed_digest is not None
                    and installed_digest.startswith(model.expected_digest)
                )
            ),
        }
    report["llm_checks"] = llm_checks
    robustness_checks = []
    if config.robustness.enabled:
        configured = [
            *[("generator", model) for model in config.robustness.generators],
            *[("judge", model) for model in config.robustness.judges],
        ]
        for role, model in configured:
            installed_digest = installed_by_name.get(model.name.split(":", 1)[0])
            robustness_checks.append(
                {
                    "role": role,
                    "name": model.name,
                    "available": installed_digest is not None,
                    "installed_digest": installed_digest,
                    "expected_digest": model.expected_digest,
                    "digest_matches": (
                        model.expected_digest is None
                        or (
                            installed_digest is not None
                            and installed_digest.startswith(model.expected_digest)
                        )
                    ),
                }
            )
    report["robustness_llm_checks"] = robustness_checks
    print(json.dumps(report, indent=2))
    healthy = (
        report["knowledge_base_exists"]
        and all(check["available"] and check["digest_matches"] for check in llm_checks.values())
        and all(check["available"] and check["digest_matches"] for check in robustness_checks)
    )
    return int(not healthy)


def command_prepare_data(config_path: str, overrides: list[str]) -> int:
    from .data.dataset import load_prepared_dataset
    from .run import start_run

    config = load_config(config_path, overrides)
    context = start_run(config)
    prepared = load_prepared_dataset(config, context.cache)
    summary = {
        "rows": len(prepared.items),
        "outfits": int(prepared.items["outfit_ID"].nunique()),
        "categories": int(prepared.items["category"].nunique()),
        "cache_hit": prepared.cache_hit,
        "cache_path": str(prepared.metadata_cache_path),
        "run_dir": str(context.run_dir),
    }
    (context.run_dir / "reports" / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def command_audit_kb(config_path: str, overrides: list[str]) -> int:
    from .evidence import load_knowledge_base
    from .kb_audit import audit_knowledge_base

    config = load_config(config_path, overrides)
    kb = load_knowledge_base(config.paths.knowledge_base)
    report = audit_knowledge_base(kb, config.dataset.target_categories)
    print(json.dumps(report, indent=2))
    has_errors = bool(
        report["missing_required_columns"]
        or report["duplicate_rule_ids"]
        or report["blank_rule_count"]
        or report["missing_target_categories"]
    )
    return int(has_errors)


def command_import_legacy(config_path: str, overrides: list[str]) -> int:
    from .artifacts import import_legacy_embeddings
    from .cache import ArtifactCache
    from .data.dataset import load_prepared_dataset, target_items

    config = load_config(config_path, overrides)
    cache = ArtifactCache(config.paths.cache_dir, config.cache.policy)
    prepared = load_prepared_dataset(config, cache)
    items = target_items(prepared.items, config)
    imported = import_legacy_embeddings(
        config,
        cache,
        items,
        Path("data/processed/target_items.parquet"),
        Path("archive/original_notebook_outputs/embeddings"),
    )
    print(
        json.dumps(
            [
                {
                    "modality": item.modality,
                    "source": str(item.source),
                    "destination": str(item.destination),
                    "shape": item.shape,
                }
                for item in imported
            ],
            indent=2,
        )
    )
    return 0


def command_evaluate_ranking(config_path: str, overrides: list[str]) -> int:
    import numpy as np

    from .artifacts import load_embedding_set
    from .cache import file_fingerprint
    from .data.dataset import (
        load_huggingface_split,
        load_prepared_dataset,
        target_items,
    )
    from .embeddings import cached_text_embeddings
    from .evaluation.controlled import (
        QueryEmbeddings,
        build_evaluation_cases,
        encode_evaluation_queries,
        evaluate_controlled,
    )
    from .evaluation.evidence_ranking import CandidateEvidenceScorer
    from .evaluation.ranking import aggregate_ranking_results
    from .evidence import build_evidence_text, load_knowledge_base
    from .models.multimodal import CLIPEmbedder
    from .models.text import SentenceTransformerEmbedder
    from .run import start_run

    config = load_config(config_path, overrides)
    context = start_run(config)
    prepared = load_prepared_dataset(config, context.cache)
    targets = target_items(prepared.items, config)
    embeddings = load_embedding_set(config, context.cache, targets)
    cases = build_evaluation_cases(
        prepared.items,
        targets,
        config.dataset.target_categories,
        max_cases_per_target=config.evaluation.controlled_case_pool_per_target,
        seed=config.project.seed,
    )
    if config.evaluation.controlled_cases > len(cases):
        raise ValueError(
            "evaluation.controlled_cases exceeds the generated case pool; increase "
            "evaluation.controlled_case_pool_per_target."
        )
    cases_path = context.run_dir / "predictions" / "evaluation_cases.parquet"
    cases.to_parquet(cases_path, index=False)

    query_inputs = {
        "cases": cases.head(config.evaluation.controlled_cases)[
            ["query_item_id", "target_category", "user_request"]
        ].to_dict("records"),
        "text_model": config.models.text_embedding.model_dump(mode="json"),
        "clip_model": config.models.multimodal_embedding.model_dump(mode="json"),
        "schema_version": 1,
    }
    minilm_record = context.cache.location(
        "evaluation_queries", {**query_inputs, "modality": "minilm"}, ".npy"
    )
    clip_record = context.cache.location(
        "evaluation_queries", {**query_inputs, "modality": "clip_fused"}, ".npy"
    )
    if minilm_record.hit and clip_record.hit:
        query_embeddings = QueryEmbeddings(np.load(minilm_record.path), np.load(clip_record.path))
        text_model = None
    else:
        text_model = SentenceTransformerEmbedder(
            config.models.text_embedding, config.project.device
        )
        clip_model = CLIPEmbedder(config.models.multimodal_embedding, config.project.device)
        split = load_huggingface_split(config)
        query_embeddings = encode_evaluation_queries(
            cases.head(config.evaluation.controlled_cases),
            prepared.items,
            split,
            text_model,
            clip_model,
        )
        for record, array in [
            (minilm_record, query_embeddings.minilm),
            (clip_record, query_embeddings.clip_fused),
        ]:
            record.path.parent.mkdir(parents=True, exist_ok=True)
            np.save(record.path, array)

    evidence_scorer = None
    if config.evidence.enabled and config.reranking.enabled:
        if text_model is None:
            text_model = SentenceTransformerEmbedder(
                config.models.text_embedding, config.project.device
            )
        kb = load_knowledge_base(config.paths.knowledge_base)
        kb_embeddings, _, _ = cached_text_embeddings(
            build_evidence_text(kb),
            text_model,
            context.cache,
            "knowledge_base_embeddings",
            file_fingerprint(config.paths.knowledge_base),
        )
        evidence_scorer = CandidateEvidenceScorer(
            kb,
            kb_embeddings,
            text_model,
            config.evidence.candidate_top_k,
            config.evidence.candidate_type_filtering,
        )

    results = evaluate_controlled(
        config,
        cases,
        targets,
        embeddings["minilm_text"],
        embeddings["clip_fused"],
        query_embeddings,
        evidence_scorer,
    )
    summary = aggregate_ranking_results(results)
    results.to_csv(context.run_dir / "metrics" / "controlled_ranking_results.csv", index=False)
    summary.to_csv(context.run_dir / "metrics" / "controlled_ranking_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nRun directory: {context.run_dir}")
    return 0


def command_build_embeddings(config_path: str, overrides: list[str]) -> int:
    from .artifacts import build_target_embeddings
    from .data.dataset import (
        load_huggingface_split,
        load_prepared_dataset,
        target_items,
    )
    from .run import start_run

    config = load_config(config_path, overrides)
    context = start_run(config)
    prepared = load_prepared_dataset(config, context.cache)
    targets = target_items(prepared.items, config)
    split = load_huggingface_split(config)
    artifacts = build_target_embeddings(
        config,
        context.cache,
        targets,
        split,
    )
    report = {name: str(path) for name, path in artifacts.items()}
    (context.run_dir / "reports" / "embedding_artifacts.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


def command_build_indexes(config_path: str, overrides: list[str]) -> int:
    from .artifacts import load_embedding_set
    from .cache import stable_fingerprint
    from .data.dataset import load_prepared_dataset, target_items
    from .indexes import build_or_load_category_indexes
    from .run import start_run

    config = load_config(config_path, overrides)
    context = start_run(config)
    prepared = load_prepared_dataset(config, context.cache)
    targets = target_items(prepared.items, config)
    embeddings = load_embedding_set(config, context.cache, targets)
    fingerprint = stable_fingerprint(
        {
            "model": config.models.multimodal_embedding.model_dump(mode="json"),
            "shape": list(embeddings["clip_fused"].shape),
            "item_ids": targets["item_ID"].astype(str).tolist(),
        }
    )
    indexes, path, hit = build_or_load_category_indexes(
        context.cache, targets, embeddings["clip_fused"], fingerprint
    )
    report = {
        "cache_hit": hit,
        "path": str(path),
        "categories": sorted(indexes.indexes),
        "rows": len(targets),
        "run_dir": str(context.run_dir),
    }
    (context.run_dir / "reports" / "faiss_indexes.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


def command_caption_image(config_path: str, overrides: list[str], args: argparse.Namespace) -> int:
    from PIL import Image

    from .models.caption import FlorenceCaptioner

    config = load_config(config_path, overrides)
    if not config.models.captioning.enabled:
        raise ValueError("Captioning is disabled in the resolved configuration.")
    model = FlorenceCaptioner(config.models.captioning, config.project.device)
    print(model.caption([Image.open(args.image)])[0])
    return 0


def command_recommend(config_path: str, overrides: list[str], args: argparse.Namespace) -> int:
    import pandas as pd

    from .artifacts import load_embedding_set
    from .cache import file_fingerprint, stable_fingerprint
    from .data.dataset import (
        load_huggingface_split,
        load_prepared_dataset,
        target_items,
    )
    from .embeddings import cached_text_embeddings
    from .evaluation.evidence_ranking import CandidateEvidenceScorer
    from .evidence import build_evidence_text, load_knowledge_base
    from .generation import generate_explanation
    from .indexes import build_or_load_category_indexes
    from .models.llm import OllamaGenerator
    from .models.multimodal import CLIPEmbedder
    from .models.text import SentenceTransformerEmbedder
    from .pipeline import recommend, recommend_from_query
    from .run import start_run

    config = load_config(config_path, overrides)
    context = start_run(config)
    prepared = load_prepared_dataset(config, context.cache)
    targets = target_items(prepared.items, config)
    embeddings = load_embedding_set(config, context.cache, targets)
    embedding_fingerprint = stable_fingerprint(
        {
            "model": config.models.multimodal_embedding.model_dump(mode="json"),
            "shape": list(embeddings["clip_fused"].shape),
            "item_ids": targets["item_ID"].astype(str).tolist(),
        }
    )
    category_indexes, _, _ = build_or_load_category_indexes(
        context.cache, targets, embeddings["clip_fused"], embedding_fingerprint
    )
    split = load_huggingface_split(config)
    clip_model = CLIPEmbedder(config.models.multimodal_embedding, config.project.device)

    evidence_scorer = None
    if config.evidence.enabled:
        text_model = SentenceTransformerEmbedder(
            config.models.text_embedding, config.project.device
        )
        kb = load_knowledge_base(config.paths.knowledge_base)
        kb_embeddings, _, _ = cached_text_embeddings(
            build_evidence_text(kb),
            text_model,
            context.cache,
            "knowledge_base_embeddings",
            file_fingerprint(config.paths.knowledge_base),
        )
        evidence_scorer = CandidateEvidenceScorer(
            kb,
            kb_embeddings,
            text_model,
            config.evidence.candidate_top_k,
            config.evidence.candidate_type_filtering,
        )

    if args.query_item_id:
        result = recommend(
            config,
            prepared.items,
            targets,
            split,
            embeddings["clip_fused"],
            clip_model,
            evidence_scorer,
            args.query_item_id,
            args.target_category,
            args.request,
            category_indexes,
        )
    else:
        from PIL import Image

        if not (args.query_image and args.query_text and args.query_category):
            raise ValueError(
                "Provide --query-item-id, or provide --query-image, --query-text, "
                "and --query-category together."
            )
        query = pd.Series(
            {
                "item_ID": "",
                "outfit_ID": "",
                "category": args.query_category,
                "query_category": args.query_category,
                "text": args.query_text,
            }
        )
        result = recommend_from_query(
            config,
            query,
            Image.open(args.query_image).convert("RGB"),
            targets,
            embeddings["clip_fused"],
            clip_model,
            evidence_scorer,
            args.target_category,
            args.request,
            category_indexes,
        )
    result.recommendations.to_csv(
        context.run_dir / "predictions" / "recommendations.csv", index=False
    )
    evidence_rows = []
    for item_id, evidence in result.evidence.items():
        table = evidence.copy()
        table.insert(0, "recommended_item_id", item_id)
        evidence_rows.append(table)
    if evidence_rows:
        pd.concat(evidence_rows, ignore_index=True).to_csv(
            context.run_dir / "predictions" / "retrieved_evidence.csv", index=False
        )

    if args.generate:
        generator = OllamaGenerator(config.models.generator)
        generation_rows = []
        for _, candidate in result.recommendations.iterrows():
            evidence = result.evidence.get(str(candidate["item_ID"]), pd.DataFrame())
            for variant in config.generation.variants:
                explanation = generate_explanation(
                    generator,
                    query_text=str(result.query["text"]),
                    user_request=args.request,
                    recommended_text=str(candidate["text"]),
                    variant=variant,
                    item_evidence=[],
                    rule_evidence=evidence,
                )
                generation_rows.append(
                    {
                        "item_ID": candidate["item_ID"],
                        "variant": variant,
                        "generated_explanation": explanation,
                    }
                )
        pd.DataFrame(generation_rows).to_csv(
            context.run_dir / "predictions" / "explanations.csv", index=False
        )
    display_columns = [
        "rank",
        "item_ID",
        "category",
        "text",
        "clip_score",
        "evidence_score",
        "final_score",
    ]
    print(result.recommendations[display_columns].to_string(index=False))
    print(f"\nRun directory: {context.run_dir}")
    return 0


def command_explanation_study(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    import pandas as pd

    from .evaluation.study import (
        evaluate_explanations,
        evaluate_rag_retrieval,
        explanation_statistics,
        generate_study,
        judge_explanations,
        write_study_outputs,
    )
    from .evidence import load_knowledge_base
    from .models.llm import OllamaGenerator
    from .run import start_run

    config = load_config(config_path, overrides)
    context = start_run(config)
    fixed_cases = pd.read_csv(args.input)
    if args.limit is not None:
        fixed_cases = fixed_cases.head(args.limit)
    study_dir = context.run_dir / "metrics" / "explanation_study"

    generator = OllamaGenerator(config.models.generator)
    explanations = generate_study(
        fixed_cases,
        config.generation.variants,
        generator,
        context.cache,
    )
    automatic, automatic_summary = evaluate_explanations(explanations)
    kb = load_knowledge_base(config.paths.knowledge_base)
    rag_results, rag_summary = evaluate_rag_retrieval(
        fixed_cases,
        kb,
        [k for k in config.evaluation.cutoffs if k <= config.evidence.candidate_top_k],
    )

    if args.skip_judge:
        judged = pd.DataFrame()
        judge_errors = pd.DataFrame()
        judge_summary = pd.DataFrame()
        statistics = pd.DataFrame()
    else:
        judge = OllamaGenerator(config.models.judge)
        judged, judge_errors, judge_summary = judge_explanations(explanations, judge, context.cache)
        statistics = (
            explanation_statistics(automatic, judged, config)
            if not judged.empty
            else pd.DataFrame()
        )

    write_study_outputs(
        study_dir,
        fixed_cases=fixed_cases,
        explanations=explanations,
        automatic_evaluation=automatic,
        automatic_summary=automatic_summary,
        rag_retrieval_evaluation=rag_results,
        rag_retrieval_summary=rag_summary,
        independent_judge_results=judged,
        independent_judge_errors=judge_errors,
        independent_judge_summary=judge_summary,
        statistical_tests=statistics,
    )
    print("\nAutomatic explanation summary")
    print(automatic_summary.to_string(index=False))
    print("\nRAG retrieval summary")
    print(rag_summary.to_string(index=False))
    if not judge_summary.empty:
        print("\nIndependent judge summary")
        print(judge_summary.to_string(index=False))
    print(f"\nRun directory: {context.run_dir}")
    return 0


def command_final_report(args: argparse.Namespace) -> int:
    from .reporting import build_final_report

    report = build_final_report(
        Path(args.baseline_run),
        Path(args.improved_run),
        Path(args.study_run),
        Path(args.output),
    )
    print(f"Final report: {report}")
    return 0


def command_robustness_report(args: argparse.Namespace) -> int:
    from .robustness_reporting import build_robustness_report

    report = build_robustness_report(
        Path(args.baseline),
        Path(args.robustness_study),
        Path(args.heldout_ranking),
        Path(args.output),
    )
    print(f"Robustness report: {report}")
    return 0


def command_build_study_cases(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    import pandas as pd

    from .artifacts import load_embedding_set
    from .cache import file_fingerprint
    from .data.dataset import (
        load_huggingface_split,
        load_prepared_dataset,
        target_items,
    )
    from .embeddings import cached_text_embeddings
    from .evaluation.evidence_ranking import CandidateEvidenceScorer
    from .evaluation.study_cases import build_modular_study_cases
    from .evidence import build_evidence_text, load_knowledge_base
    from .models.multimodal import CLIPEmbedder
    from .models.text import SentenceTransformerEmbedder
    from .run import start_run

    config = load_config(config_path, overrides)
    context = start_run(config)
    prepared = load_prepared_dataset(config, context.cache)
    targets = target_items(prepared.items, config)
    embeddings = load_embedding_set(config, context.cache, targets)
    split = load_huggingface_split(config)
    clip_model = CLIPEmbedder(config.models.multimodal_embedding, config.project.device)
    text_model = SentenceTransformerEmbedder(config.models.text_embedding, config.project.device)
    kb = load_knowledge_base(config.paths.knowledge_base)
    kb_embeddings, _, _ = cached_text_embeddings(
        build_evidence_text(kb),
        text_model,
        context.cache,
        "knowledge_base_embeddings",
        file_fingerprint(config.paths.knowledge_base),
    )
    evidence_scorer = CandidateEvidenceScorer(
        kb,
        kb_embeddings,
        text_model,
        config.evidence.candidate_top_k,
        config.evidence.candidate_type_filtering,
    )
    cases = build_modular_study_cases(
        config,
        pd.read_csv(args.schedule),
        prepared.items,
        targets,
        split,
        embeddings["clip_fused"],
        clip_model,
        evidence_scorer,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output, index=False)
    cases.to_csv(context.run_dir / "predictions" / "modular_study_cases.csv", index=False)
    print(f"Built {len(cases)} modular explanation-study rows: {output}")
    print(f"Run directory: {context.run_dir}")
    return 0


def command_build_robustness_schedules(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    import pandas as pd

    from .data.dataset import load_prepared_dataset, target_items
    from .evaluation.controlled import build_evaluation_cases
    from .evaluation.splits import (
        assert_disjoint_outfits,
        assign_outfit_splits,
        balanced_sample,
    )
    from .run import start_run

    config = load_config(config_path, overrides)
    if not config.robustness.enabled:
        raise ValueError("Robustness evaluation is disabled in this configuration.")
    context = start_run(config)
    prepared = load_prepared_dataset(config, context.cache)
    targets = target_items(prepared.items, config)
    requested = config.robustness.expanded_cases_per_category
    # Oversample before hashing because the smallest 20% partition must still contain
    # the requested balanced count in every target category.
    pool_per_category = max(requested * 10, requested)
    cases = build_evaluation_cases(
        prepared.items,
        targets,
        config.dataset.target_categories,
        max_cases_per_target=pool_per_category,
        seed=config.project.seed,
    )
    cases = assign_outfit_splits(
        cases,
        outfit_column="query_outfit_id",
        seed=config.project.seed,
        development_fraction=config.robustness.development_fraction,
        validation_fraction=config.robustness.validation_fraction,
    )
    assert_disjoint_outfits(cases, "query_outfit_id")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for split in ("development", "validation", "test"):
        selected = balanced_sample(
            cases,
            split=split,
            category_column="target_category",
            cases_per_category=requested,
            seed=config.project.seed,
        )
        selected.insert(0, "case_index", range(len(selected)))
        selected.to_csv(output_dir / f"{split}_schedule.csv", index=False)
        summary_rows.extend(
            {
                "research_split": split,
                "target_category": category,
                "cases": len(group),
                "unique_outfits": group["query_outfit_id"].nunique(),
            }
            for category, group in selected.groupby("target_category")
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    manifest = {
        "seed": config.project.seed,
        "split_fractions": {
            "development": config.robustness.development_fraction,
            "validation": config.robustness.validation_fraction,
            "test": config.robustness.test_fraction,
        },
        "cases_per_category_per_split": requested,
        "outfit_overlap": 0,
        "run_dir": str(context.run_dir),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nSchedules: {output_dir}")
    return 0


def command_hybrid_ablations(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    import pandas as pd

    from .evaluation.robustness import (
        evaluate_hybrid_ablations,
        generate_hybrid_ablations,
        one_factor_hybrid_specs,
    )
    from .evaluation.study import JUDGE_DIMENSIONS, judge_explanations
    from .models.llm import OllamaGenerator
    from .run import start_run

    config = load_config(config_path, overrides)
    if not config.robustness.enabled or not config.robustness.generators:
        raise ValueError("Configured robustness generators are required.")
    context = start_run(config)
    cases = pd.read_csv(args.input)
    if set(cases.get("research_split", [])) not in (set(), {"validation"}):
        raise ValueError("Prompt selection may only use validation cases.")
    if args.limit:
        cases = cases.head(args.limit)
    specs = one_factor_hybrid_specs(
        config.robustness.hybrid_word_limits,
        config.robustness.hybrid_rule_counts,
        config.robustness.hybrid_prompt_orders,
    )
    generator = OllamaGenerator(config.robustness.generators[0])
    explanations = generate_hybrid_ablations(cases, specs, generator, context.cache)
    evaluated, summary = evaluate_hybrid_ablations(explanations)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    explanations.to_csv(output_dir / "explanations.csv", index=False)
    evaluated.to_csv(output_dir / "automatic_evaluation.csv", index=False)
    summary.to_csv(output_dir / "automatic_summary.csv", index=False)
    if not args.skip_judge:
        judge_input = explanations.copy()
        names = judge_input["grounding_variant"].copy()
        judge_input["grounding_variant"] = "hybrid_rag"
        judged, errors, _ = judge_explanations(
            judge_input,
            OllamaGenerator(config.robustness.judges[0]),
            context.cache,
        )
        judged["grounding_variant"] = names.iloc[judged.index].to_numpy()
        judge_summary = (
            judged.groupby("grounding_variant")[[*JUDGE_DIMENSIONS, "overall_judge_score"]]
            .mean()
            .reset_index()
        )
        selected = summary.merge(judge_summary, on="grounding_variant")
        selected["validation_selection_score"] = (
            selected["automatic_selection_score"] + selected["overall_judge_score"] / 5.0
        )
        selected = selected.sort_values("validation_selection_score", ascending=False)
        judged.to_csv(output_dir / "judge_results.csv", index=False)
        errors.to_csv(output_dir / "judge_errors.csv", index=False)
        selected.to_csv(output_dir / "validation_selection.csv", index=False)
        print(selected.to_string(index=False))
    else:
        print(summary.to_string(index=False))
    print(f"\nRun directory: {context.run_dir}")
    return 0


def command_hybrid_validation_v2(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    import pandas as pd

    from .evaluation.hybrid_v2 import run_hybrid_validation_v2
    from .evaluation.robustness import full_hybrid_specs
    from .models.llm import OllamaGenerator
    from .run import start_run

    config = load_config(config_path, overrides)
    if not config.final_evaluation.enabled:
        raise ValueError("final_evaluation must be enabled for Hybrid v2 validation.")
    if not config.robustness.generators or not config.robustness.judges:
        raise ValueError("Configured robustness generator and judge models are required.")
    input_path = Path(args.input)
    output_dir = _require_v2_output(config, args.output_dir)
    cases = pd.read_csv(input_path)
    context = start_run(config)
    specs = full_hybrid_specs(
        config.final_evaluation.hybrid_word_budgets,
        config.final_evaluation.hybrid_rule_counts,
        config.final_evaluation.hybrid_item_counts,
        config.final_evaluation.hybrid_evidence_orders,
    )
    manifest = run_hybrid_validation_v2(
        cases=cases,
        specs=specs,
        generator=OllamaGenerator(config.robustness.generators[0]),
        judge=OllamaGenerator(config.robustness.judges[0]),
        cache=context.cache,
        output_dir=output_dir,
        report_path=config.final_evaluation.report_root / "hybrid_validation_report.md",
        screening_cases_per_category=(config.final_evaluation.hybrid_screening_cases_per_category),
        finalist_count=config.final_evaluation.hybrid_finalist_count,
        practical_tie=config.final_evaluation.hybrid_practical_tie,
        seed=config.project.seed,
        input_path=input_path,
    )
    print(json.dumps(manifest, indent=2))
    print(f"Run directory: {context.run_dir}")
    return 0


def command_freeze_final_eval_v2(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    import pandas as pd

    from .final_freeze import create_final_eval_v2_freeze

    config = load_config(config_path, overrides)
    destination = _require_v2_output(config, args.destination)
    freeze_inputs = config.final_evaluation.output_root / "freeze_inputs"
    freeze_inputs.mkdir(parents=True, exist_ok=True)
    resolved_config = freeze_inputs / "resolved_config.json"
    resolved_config.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    decision_path = config.final_evaluation.output_root / "decision_gate" / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    validation_packets = (
        config.final_evaluation.output_root / "prepared" / "validation" / "locked_packets.csv"
    )
    hashes = set(pd.read_csv(validation_packets)["stage1_packet_hash"].astype(str))
    if len(hashes) != 1:
        raise ValueError("Validation packets must share one Stage 1 packet hash.")
    path = create_final_eval_v2_freeze(
        destination=destination,
        resolved_config=resolved_config,
        fusion_selection=config.final_evaluation.output_root
        / "validation/fusion_tuning/selected_fusion.json",
        reranking_selection=config.final_evaluation.output_root
        / "validation/reranking_tuning/selected_weight.json",
        hybrid_selection=config.final_evaluation.output_root
        / "hybrid_validation/selected_hybrid_config.json",
        schedules=[
            Path("outputs/robustness/schedules/validation_schedule.csv"),
            Path("outputs/robustness/schedules/test_schedule.csv"),
        ],
        cases=[
            validation_packets,
            config.final_evaluation.output_root / "prepared/test/locked_packets.csv",
        ],
        knowledge_base=config.paths.knowledge_base,
        dependency_lock=Path("uv.lock"),
        prompt_files=[Path("src/evidence_fashion_recommender/generation.py")],
        command_list=[
            "freeze-final-eval-v2",
            "run-final-explanations-v2",
            "extract-claims-v2",
            "verify-claims-v2",
            "judge-general-quality-v2",
            "analyze-final-eval-v2",
            "build-final-report-v2",
        ],
        expected_stage1_packet_hash=next(iter(hashes)),
        gate_definition=decision,
        additional_inputs=[decision_path],
    )
    print(f"Final freeze manifest: {path}")
    return 0


def command_final_explanations_v2(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    import pandas as pd

    from .evaluation.generation_v2 import run_final_explanations_v2
    from .models.llm import OllamaGenerator
    from .run import start_run

    config = load_config(config_path, overrides)
    context = start_run(config)
    manifest = run_final_explanations_v2(
        cases=pd.read_csv(args.input),
        generators=[OllamaGenerator(value) for value in config.robustness.generators],
        cache=context.cache,
        output_dir=_require_v2_output(config, args.output_dir),
        report_path=config.final_evaluation.report_root / "generation_summary.md",
        input_path=Path(args.input),
        reranking_selection_path=Path(args.reranking_selection),
        hybrid_selection_path=Path(args.hybrid_selection),
        decision_path=Path(args.decision),
        freeze_path=Path(args.freeze),
    )
    print(json.dumps(manifest, indent=2))
    return 0


def command_tune_reranking(config_path: str, overrides: list[str], args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    from .artifacts import load_embedding_set
    from .cache import file_fingerprint
    from .data.dataset import (
        load_huggingface_split,
        load_prepared_dataset,
        target_items,
    )
    from .embeddings import cached_text_embeddings
    from .evaluation.controlled import QueryEmbeddings, encode_evaluation_queries
    from .evaluation.evidence_ranking import CandidateEvidenceScorer
    from .evaluation.tuning import evaluate_reranking_grid, select_reranking_weight
    from .evidence import build_evidence_text, load_knowledge_base
    from .models.multimodal import CLIPEmbedder
    from .models.text import SentenceTransformerEmbedder
    from .run import start_run

    config = load_config(config_path, overrides)
    if not config.robustness.enabled:
        raise ValueError("Robustness evaluation is disabled.")
    cases = pd.read_csv(args.input)
    expected_split = getattr(args, "expected_split", "validation")
    if set(cases["research_split"]) != {expected_split}:
        raise ValueError(f"Expected only {expected_split} cases.")
    context = start_run(config)
    prepared = load_prepared_dataset(config, context.cache)
    targets = target_items(prepared.items, config)
    embeddings = load_embedding_set(config, context.cache, targets)
    query_inputs = {
        "cases": cases[["query_item_id", "target_category", "user_request"]].to_dict("records"),
        "model": config.models.multimodal_embedding.model_dump(mode="json"),
        "schema_version": 1,
    }
    query_record = context.cache.location("robustness_query_embeddings", query_inputs, ".npy")
    clip_model = CLIPEmbedder(config.models.multimodal_embedding, config.project.device)
    if query_record.hit:
        query_embeddings = QueryEmbeddings(
            minilm=np.empty((len(cases), 0)),
            clip_fused=np.load(query_record.path),
        )
    else:
        split = load_huggingface_split(config)
        text_model_for_queries = SentenceTransformerEmbedder(
            config.models.text_embedding, config.project.device
        )
        query_embeddings = encode_evaluation_queries(
            cases,
            prepared.items,
            split,
            text_model_for_queries,
            clip_model,
        )
        query_record.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(query_record.path, query_embeddings.clip_fused)
    text_model = SentenceTransformerEmbedder(config.models.text_embedding, config.project.device)
    kb = load_knowledge_base(config.paths.knowledge_base)
    kb_embeddings, _, _ = cached_text_embeddings(
        build_evidence_text(kb),
        text_model,
        context.cache,
        "knowledge_base_embeddings",
        file_fingerprint(config.paths.knowledge_base),
    )
    scorer = CandidateEvidenceScorer(
        kb,
        kb_embeddings,
        text_model,
        config.evidence.candidate_top_k,
        config.evidence.candidate_type_filtering,
    )
    if expected_split == "validation":
        weights = config.robustness.rerank_clip_weights
    else:
        selection_path = Path(args.selection)
        if not selection_path.is_file():
            raise ValueError(f"Missing frozen reranking selection: {selection_path}")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_clip_weight = float(selection["clip_weight"])
        weights = [selected_clip_weight, 1.0]
    results = evaluate_reranking_grid(
        config,
        cases,
        targets,
        embeddings["clip_fused"],
        query_embeddings,
        scorer,
        weights,
    )
    summary = select_reranking_weight(results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / f"{expected_split}_results.csv", index=False)
    summary.to_csv(output_dir / f"{expected_split}_summary.csv", index=False)
    if expected_split == "validation":
        (output_dir / "selected_weight.json").write_text(
            json.dumps(summary.iloc[0].to_dict(), indent=2), encoding="utf-8"
        )
    print(summary.to_string(index=False))
    if expected_split == "validation":
        print(f"\nSelected clip weight: {summary.iloc[0]['clip_weight']}")
    else:
        print("\nHeld-out comparison only; no configuration was selected on test.")
    return 0


def command_heldout_ranking(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    args.expected_split = "test"
    return command_tune_reranking(config_path, overrides, args)


def _require_v2_output(config, output_dir: str) -> Path:
    root = config.final_evaluation.output_root.resolve()
    output = Path(output_dir).resolve()
    if output != root and root not in output.parents:
        raise ValueError(f"final_eval_v2 outputs must stay under {root}")
    return output


def command_tune_clip_fusion_v2(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .evaluation.stage1 import load_stage1_bundle, tune_clip_fusion_artifacts

    config = load_config(config_path, overrides)
    if not config.final_evaluation.enabled:
        raise ValueError("final_eval_v2 is disabled in this configuration.")
    bundle = load_stage1_bundle(Path(args.bundle), "validation")
    output = _require_v2_output(config, args.output_dir)
    selected = tune_clip_fusion_artifacts(
        bundle,
        output_dir=output,
        image_weights=config.final_evaluation.fusion_image_weights,
        cutoffs=config.evaluation.cutoffs,
    )
    print(json.dumps(selected, indent=2))
    return 0


def command_evaluate_final_retrieval_v2(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .evaluation.stage1 import evaluate_final_retrieval_artifacts, load_stage1_bundle

    config = load_config(config_path, overrides)
    if not config.final_evaluation.enabled:
        raise ValueError("final_eval_v2 is disabled in this configuration.")
    bundle = load_stage1_bundle(Path(args.bundle), "test")
    output = _require_v2_output(config, args.output_dir)
    evaluate_final_retrieval_artifacts(
        bundle,
        output_dir=output,
        fusion_selection=Path(args.fusion_selection),
        reranking_selection=Path(args.reranking_selection),
        locked_packets=Path(args.locked_packets),
        cutoffs=config.evaluation.cutoffs,
    )
    print(f"Final retrieval v2 artifacts: {output}")
    return 0


def command_compare_locked_artifacts_v2(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .evaluation.stage1 import compare_locked_artifact_outputs

    config = load_config(config_path, overrides)
    if not config.final_evaluation.enabled:
        raise ValueError("final_eval_v2 is disabled in this configuration.")
    output = _require_v2_output(config, args.output_dir)
    decision = compare_locked_artifact_outputs(
        legacy_packets=Path(args.legacy_packets),
        v2_packets=Path(args.v2_packets),
        output_dir=output,
    )
    print(json.dumps(decision, indent=2))
    return 0


def command_prepare_final_retrieval_v2_bundle(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .evaluation.stage1_preparation import prepare_stage1_bundle

    config = load_config(config_path, overrides)
    if not config.final_evaluation.enabled:
        raise ValueError("final_eval_v2 is disabled in this configuration.")
    output = _require_v2_output(config, args.output_dir)
    fingerprint = prepare_stage1_bundle(
        split=args.split,
        schedule_path=Path(args.schedule),
        candidate_sets_path=Path(args.candidate_sets),
        target_embedding_dir=Path(args.target_embedding_dir),
        query_embedding_dir=Path(args.query_embedding_dir),
        output_dir=output,
    )
    print(f"Prepared {args.split} bundle: {output} ({fingerprint[:12]})")
    return 0


def command_tune_reranking_v2(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .evaluation.stage1_preparation import tune_reranking_v2_artifacts

    config = load_config(config_path, overrides)
    if not config.final_evaluation.enabled:
        raise ValueError("final_eval_v2 is disabled in this configuration.")
    output = _require_v2_output(config, args.output_dir)
    selected = tune_reranking_v2_artifacts(
        bundle_dir=Path(args.bundle),
        fusion_selection_path=Path(args.fusion_selection),
        resolved_config_path=Path(config_path),
        output_dir=output,
        clip_weights=config.robustness.rerank_clip_weights,
        cutoffs=config.evaluation.cutoffs,
    )
    print(json.dumps(selected, indent=2))
    return 0


def command_select_evidence_in_loop_reranking_v2(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .evaluation.stage1_preparation import freeze_evidence_in_loop_reranking_selection

    config = load_config(config_path, overrides)
    if not config.final_evaluation.enabled:
        raise ValueError("final_eval_v2 is disabled in this configuration.")
    output = _require_v2_output(config, args.output)
    selected = freeze_evidence_in_loop_reranking_selection(
        summary_path=Path(args.summary),
        selected_path=output,
        clip_weight=config.final_evaluation.proposed_reranking_clip_weight,
        selection_policy=config.final_evaluation.proposed_reranking_selection_policy,
    )
    print(json.dumps(selected, indent=2))
    return 0


def command_create_locked_packets_v2(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .evaluation.stage1_preparation import create_locked_packets_v2

    config = load_config(config_path, overrides)
    if not config.final_evaluation.enabled:
        raise ValueError("final_eval_v2 is disabled in this configuration.")
    output = _require_v2_output(config, args.output)
    packet_hash = create_locked_packets_v2(
        source_cases_path=Path(args.source_cases),
        fusion_selection_path=Path(args.fusion_selection),
        reranking_selection_path=Path(args.reranking_selection),
        output_path=output,
        expected_split=args.split,
    )
    print(f"Locked {args.split} packet hash: {packet_hash}")
    return 0


def _read_table(path: Path):
    import pandas as pd

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def command_materialize_final_retrieval_v2_inputs(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .evaluation.materialization import materialize_retrieval_inputs

    config = load_config(config_path, overrides)
    if not config.final_evaluation.enabled:
        raise ValueError("final_eval_v2 is disabled in this configuration.")
    output_root = _require_v2_output(config, args.output_root)
    manifest = materialize_retrieval_inputs(
        config=config,
        split=args.split,
        schedule_path=Path(args.schedule),
        target_items=_read_table(Path(args.target_items)),
        candidate_source=Path(args.candidate_source),
        output_root=output_root,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def command_materialize_final_retrieval_v2_query_embeddings(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .data.dataset import load_huggingface_split, load_prepared_dataset
    from .evaluation.controlled import encode_evaluation_queries
    from .evaluation.materialization import materialize_query_embeddings, normalize_schedule
    from .models.multimodal import CLIPEmbedder
    from .models.text import SentenceTransformerEmbedder
    from .run import start_run

    config = load_config(config_path, overrides)
    if not config.final_evaluation.enabled:
        raise ValueError("final_eval_v2 is disabled in this configuration.")
    if not args.approve_compute_query_embeddings:
        raise PermissionError("Pass --approve-compute-query-embeddings after explicit approval.")

    def build(schedule):
        context = start_run(config)
        prepared = load_prepared_dataset(config, context.cache)
        dataset_split = load_huggingface_split(config)
        text_model = SentenceTransformerEmbedder(
            config.models.text_embedding, config.project.device
        )
        clip_model = CLIPEmbedder(config.models.multimodal_embedding, config.project.device)
        encoded = encode_evaluation_queries(
            normalize_schedule(schedule, args.split),
            prepared.items,
            dataset_split,
            text_model,
            clip_model,
        )
        return {
            "query_minilm": encoded.minilm,
            "query_clip_image": encoded.clip_image,
            "query_clip_text": encoded.clip_text,
        }

    cache_dir = materialize_query_embeddings(
        config=config,
        split=args.split,
        schedule_path=Path(args.schedule),
        builder=build,
        approved=True,
    )
    print(f"Query-only embeddings: {cache_dir}")
    return 0


def command_materialize_final_retrieval_v2_selected_cases(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .evaluation.materialization import materialize_selected_cases

    config = load_config(config_path, overrides)
    if not config.final_evaluation.enabled:
        raise ValueError("final_eval_v2 is disabled in this configuration.")
    output = _require_v2_output(config, args.output)
    manifest = materialize_selected_cases(
        split=args.split,
        schedule_path=Path(args.schedule),
        source_cases=Path(args.source_cases),
        fusion_selection=Path(args.fusion_selection),
        reranking_selection=Path(args.reranking_selection),
        output_path=output,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _build_v2_evidence_scorer(config):
    from .cache import ArtifactCache, file_fingerprint, stable_fingerprint
    from .embeddings import cached_text_embeddings
    from .evaluation.evidence_ranking import CandidateEvidenceScorer
    from .evidence import build_evidence_text, load_knowledge_base
    from .models.text import SentenceTransformerEmbedder

    cache = ArtifactCache(config.paths.cache_dir, config.cache.policy)
    knowledge_base = load_knowledge_base(config.paths.knowledge_base)
    embedder = SentenceTransformerEmbedder(config.models.text_embedding, config.project.device)
    kb_hash = file_fingerprint(config.paths.knowledge_base)
    embeddings, embedding_path, _ = cached_text_embeddings(
        build_evidence_text(knowledge_base),
        embedder,
        cache,
        "knowledge_base_embeddings",
        kb_hash,
    )
    scorer = CandidateEvidenceScorer(
        knowledge_base,
        embeddings,
        embedder,
        config.evidence.candidate_top_k,
        config.evidence.candidate_type_filtering,
    )
    evidence_hash = stable_fingerprint(
        {"knowledge_base": kb_hash, "embeddings": file_fingerprint(embedding_path)}
    )
    return scorer, evidence_hash


def command_materialize_final_eval_v2_target_items(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .cache import ArtifactCache
    from .data.dataset import load_prepared_dataset, target_items
    from .evaluation.v2_sources import materialize_target_item_table

    config = load_config(config_path, overrides)
    output = _require_v2_output(config, args.output)
    prepared = load_prepared_dataset(
        config, ArtifactCache(config.paths.cache_dir, config.cache.policy)
    )
    manifest = materialize_target_item_table(
        config=config,
        items=target_items(prepared.items, config),
        source_paths=[prepared.metadata_cache_path],
        output_path=output,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def command_produce_final_eval_v2_candidates(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .evaluation.v2_sources import produce_candidate_sets

    config = load_config(config_path, overrides)
    output = _require_v2_output(config, args.output)
    scorer, evidence_hash = _build_v2_evidence_scorer(config)
    manifest = produce_candidate_sets(
        config=config,
        split=args.split,
        schedule_path=Path(args.schedule),
        target_items_path=Path(args.target_items),
        scorer=scorer,
        evidence_hash=evidence_hash,
        output_path=output,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def command_produce_final_eval_v2_selected_cases(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .evaluation.v2_sources import produce_selected_cases

    config = load_config(config_path, overrides)
    output = _require_v2_output(config, args.output)
    scorer, _ = _build_v2_evidence_scorer(config)
    bundle = Path(args.bundle)
    manifest = produce_selected_cases(
        split=args.split,
        schedule_path=Path(args.schedule),
        target_items_path=Path(args.target_items),
        candidate_sets_path=Path(args.candidate_sets),
        target_clip_image_path=bundle / "target_clip_image.npy",
        target_clip_text_path=bundle / "target_clip_text.npy",
        query_clip_image_path=bundle / "query_clip_image.npy",
        query_clip_text_path=bundle / "query_clip_text.npy",
        fusion_selection_path=Path(args.fusion_selection),
        reranking_selection_path=Path(args.reranking_selection),
        scorer=scorer,
        output_path=output,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def command_inspect_final_eval_v2_readiness(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    from .evaluation.v2_preflight import inspect_readiness

    config = load_config(config_path, overrides)
    result = inspect_readiness(
        config,
        {
            "validation": Path(args.validation_schedule),
            "test": Path(args.test_schedule),
        },
    )
    print(json.dumps(result, indent=2))
    return 0


def command_robustness_study(
    config_path: str, overrides: list[str], args: argparse.Namespace
) -> int:
    import pandas as pd

    from .evaluation.explanations import substitution_detector_benchmark
    from .evaluation.robustness import (
        HybridPromptSpec,
        generate_robustness_study,
        judge_agreement,
        judge_robustness_study,
    )
    from .evaluation.statistics import compare_variants
    from .evaluation.study import (
        JUDGE_DIMENSIONS,
        evaluate_explanations,
        evaluate_rag_retrieval,
    )
    from .evaluation.verification import (
        consensus_retrieval_metrics,
        counterfactual_category_test,
        rule_relevance_agreement,
        verify_rule_relevance,
    )
    from .evidence import load_knowledge_base
    from .models.llm import OllamaGenerator
    from .run import start_run

    config = load_config(config_path, overrides)
    cases = pd.read_csv(args.input)
    if set(cases["research_split"]) != {"test"}:
        raise ValueError("The final robustness study requires frozen test cases.")
    if args.limit:
        cases = cases.head(args.limit)
    selection = pd.read_csv(args.selection).iloc[0]
    selected_spec = HybridPromptSpec(
        max_words=int(selection["max_words"]),
        rule_limit=int(selection["rule_limit"]),
        prompt_order=str(selection["prompt_order"]),
    )
    context = start_run(config)
    generators = [OllamaGenerator(model) for model in config.robustness.generators]
    judges = [OllamaGenerator(model) for model in config.robustness.judges]
    explanations = generate_robustness_study(
        cases,
        config.generation.variants,
        generators,
        selected_spec,
        context.cache,
    )
    automatic_parts, automatic_summaries = [], []
    for model, group in explanations.groupby("generation_model"):
        evaluated, summary = evaluate_explanations(group)
        evaluated["generation_model"] = model
        summary["generation_model"] = model
        automatic_parts.append(evaluated)
        automatic_summaries.append(summary)
    automatic = pd.concat(automatic_parts, ignore_index=True)
    automatic_summary = pd.concat(automatic_summaries, ignore_index=True)
    judged, judge_errors = judge_robustness_study(explanations, judges, context.cache)
    dimensions = [
        *JUDGE_DIMENSIONS,
        "overall_judge_score",
        "claim_support_rate",
        "claim_label_compliance_rate",
    ]
    judge_summary = (
        judged.groupby(["generation_model", "judge_model", "grounding_variant"])[dimensions]
        .mean()
        .reset_index()
    )
    judge_variability = (
        judged.groupby(["generation_model", "judge_model", "grounding_variant"])[dimensions]
        .std()
        .reset_index()
    )
    ensemble_summary = (
        judged.groupby(["generation_model", "grounding_variant"])[dimensions].mean().reset_index()
    )
    cross_model_summary = (
        judged[~judged["self_judge"]]
        .groupby(["generation_model", "grounding_variant"])[dimensions]
        .mean()
        .reset_index()
    )
    agreement = judge_agreement(judged, dimensions)
    statistical_parts = []
    for (generator, judge), group in judged.groupby(["generation_model", "judge_model"]):
        tests = compare_variants(
            group,
            "paper_case_id",
            "grounding_variant",
            dimensions,
            config.evaluation.bootstrap_samples,
            config.evaluation.confidence_level,
            config.project.seed,
        )
        tests["generation_model"] = generator
        tests["judge_model"] = judge
        statistical_parts.append(tests)
    statistics = pd.concat(statistical_parts, ignore_index=True)
    knowledge_base = load_knowledge_base(config.paths.knowledge_base)
    relevance, relevance_errors = verify_rule_relevance(cases, judges, context.cache)
    consensus_metrics = consensus_retrieval_metrics(relevance)
    relevance_agreement = rule_relevance_agreement(relevance)
    counterfactual = counterfactual_category_test(cases, knowledge_base)
    kb_results, kb_summary = evaluate_rag_retrieval(
        cases,
        knowledge_base,
        [cutoff for cutoff in config.evaluation.cutoffs if cutoff <= 5],
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "explanations": explanations,
        "automatic_evaluation": automatic,
        "automatic_summary": automatic_summary,
        "judge_results": judged,
        "judge_errors": judge_errors,
        "judge_summary": judge_summary,
        "judge_score_standard_deviations": judge_variability,
        "judge_ensemble_summary": ensemble_summary,
        "cross_model_judge_summary": cross_model_summary,
        "judge_agreement": agreement,
        "statistical_tests": statistics,
        "rule_relevance_judgments": relevance,
        "rule_relevance_errors": relevance_errors,
        "rule_relevance_agreement": relevance_agreement,
        "consensus_retrieval_metrics": consensus_metrics,
        "counterfactual_retrieval_test": counterfactual,
        "kb_proxy_retrieval_results": kb_results,
        "kb_proxy_retrieval_summary": kb_summary,
    }
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    substitution_metrics, substitution_examples = substitution_detector_benchmark()
    pd.DataFrame(substitution_examples).to_csv(
        output_dir / "substitution_detector_validation.csv", index=False
    )
    (output_dir / "substitution_detector_metrics.json").write_text(
        json.dumps(substitution_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "selected_hybrid_spec.json").write_text(
        json.dumps(
            {
                "name": selected_spec.name,
                "max_words": selected_spec.max_words,
                "rule_limit": selected_spec.rule_limit,
                "prompt_order": selected_spec.prompt_order,
                "selected_on": "validation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(ensemble_summary.to_string(index=False))
    print(f"\nJudge errors: {len(judge_errors)}")
    print(f"Rule relevance errors: {len(relevance_errors)}")
    print(f"Run directory: {context.run_dir}")
    return 0


def command_freeze_baseline(args: argparse.Namespace) -> int:
    from .evaluation.baseline import freeze_baseline

    manifest = freeze_baseline(Path(args.source), Path(args.destination))
    print(f"Frozen baseline manifest: {manifest}")
    return 0


def command_show_plan(config_path: str, overrides: list[str]) -> int:
    config = load_config(config_path, overrides)
    stages = [
        "prepare-data",
        f"embed-text ({config.models.text_embedding.name})",
        f"embed-multimodal ({config.models.multimodal_embedding.name})",
        f"retrieve ({config.retrieval.query_mode}, top {config.retrieval.candidate_pool_size})",
    ]
    if config.evidence.enabled:
        stages.append(f"retrieve-evidence ({config.evidence.kb_version})")
    if config.reranking.enabled:
        stages.append(f"rerank ({config.reranking.method})")
    stages.extend(
        [
            f"generate ({', '.join(config.generation.variants)})",
            f"evaluate ({config.evaluation.controlled_cases} controlled cases)",
        ]
    )
    print("\n".join(f"{index}. {stage}" for index, stage in enumerate(stages, 1)))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "validate-config": command_validate,
        "doctor": command_doctor,
        "prepare-data": command_prepare_data,
        "audit-kb": command_audit_kb,
        "import-legacy": command_import_legacy,
        "evaluate-ranking": command_evaluate_ranking,
        "build-embeddings": command_build_embeddings,
        "build-indexes": command_build_indexes,
        "show-plan": command_show_plan,
    }
    if args.command == "recommend":
        return command_recommend(args.config, args.set, args)
    if args.command == "run-explanation-study":
        return command_explanation_study(args.config, args.set, args)
    if args.command == "build-final-report":
        return command_final_report(args)
    if args.command == "build-robustness-report":
        return command_robustness_report(args)
    if args.command == "build-study-cases":
        return command_build_study_cases(args.config, args.set, args)
    if args.command == "build-robustness-schedules":
        return command_build_robustness_schedules(args.config, args.set, args)
    if args.command == "run-hybrid-ablations":
        return command_hybrid_ablations(args.config, args.set, args)
    if args.command == "run-hybrid-validation-v2":
        return command_hybrid_validation_v2(args.config, args.set, args)
    if args.command == "freeze-final-eval-v2":
        return command_freeze_final_eval_v2(args.config, args.set, args)
    if args.command == "run-final-explanations-v2":
        return command_final_explanations_v2(args.config, args.set, args)
    if args.command == "tune-reranking":
        return command_tune_reranking(args.config, args.set, args)
    if args.command == "evaluate-heldout-ranking":
        return command_heldout_ranking(args.config, args.set, args)
    if args.command == "tune-clip-fusion":
        return command_tune_clip_fusion_v2(args.config, args.set, args)
    if args.command == "evaluate-final-retrieval-v2":
        return command_evaluate_final_retrieval_v2(args.config, args.set, args)
    if args.command == "compare-locked-artifacts-v2":
        return command_compare_locked_artifacts_v2(args.config, args.set, args)
    if args.command == "prepare-final-retrieval-v2-bundle":
        return command_prepare_final_retrieval_v2_bundle(args.config, args.set, args)
    if args.command == "tune-reranking-v2":
        return command_tune_reranking_v2(args.config, args.set, args)
    if args.command == "select-evidence-in-loop-reranking-v2":
        return command_select_evidence_in_loop_reranking_v2(args.config, args.set, args)
    if args.command == "create-locked-packets-v2":
        return command_create_locked_packets_v2(args.config, args.set, args)
    if args.command == "materialize-final-retrieval-v2-inputs":
        return command_materialize_final_retrieval_v2_inputs(args.config, args.set, args)
    if args.command == "materialize-final-retrieval-v2-query-embeddings":
        return command_materialize_final_retrieval_v2_query_embeddings(args.config, args.set, args)
    if args.command == "materialize-final-retrieval-v2-selected-cases":
        return command_materialize_final_retrieval_v2_selected_cases(args.config, args.set, args)
    if args.command == "materialize-final-eval-v2-target-items":
        return command_materialize_final_eval_v2_target_items(args.config, args.set, args)
    if args.command == "produce-final-eval-v2-candidates":
        return command_produce_final_eval_v2_candidates(args.config, args.set, args)
    if args.command == "produce-final-eval-v2-selected-cases":
        return command_produce_final_eval_v2_selected_cases(args.config, args.set, args)
    if args.command == "inspect-final-eval-v2-readiness":
        return command_inspect_final_eval_v2_readiness(args.config, args.set, args)
    if args.command == "run-robustness-study":
        return command_robustness_study(args.config, args.set, args)
    if args.command == "freeze-baseline":
        return command_freeze_baseline(args)
    if args.command == "caption-image":
        return command_caption_image(args.config, args.set, args)
    return commands[args.command](args.config, args.set)


if __name__ == "__main__":
    raise SystemExit(main())
