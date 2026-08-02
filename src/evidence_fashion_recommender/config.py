"""Typed YAML configuration with inheritance and dotted overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    name: str
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"
    precision: Literal["float16", "bfloat16", "float32"] = "float32"


class PathsConfig(StrictModel):
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    knowledge_base: Path
    outputs_dir: Path
    cache_dir: Path


class DatasetConfig(StrictModel):
    provider: Literal["huggingface"] = "huggingface"
    name: str
    revision: str | None = None
    prefer_local_cache: bool = True
    split: str
    image_column: str
    category_column: str
    text_column: str
    item_id_column: str
    outfit_id_separator: str = "_"
    target_categories: list[str]
    query_categories: list[str] = Field(default_factory=list)


class EmbeddingModelConfig(StrictModel):
    provider: Literal["sentence_transformers", "huggingface"]
    name: str
    revision: str | None = None
    local_files_only: bool = False
    batch_size: int = Field(gt=0)
    normalize: bool = True


class MultimodalModelConfig(EmbeddingModelConfig):
    image_weight: float = Field(ge=0)
    text_weight: float = Field(ge=0)

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> MultimodalModelConfig:
        if abs(self.image_weight + self.text_weight - 1.0) > 1e-8:
            raise ValueError("Multimodal image_weight and text_weight must sum to 1.")
        return self


class CaptionModelConfig(StrictModel):
    enabled: bool = True
    provider: Literal["huggingface"] = "huggingface"
    name: str
    revision: str | None = None
    local_files_only: bool = False
    batch_size: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0)


class LLMConfig(StrictModel):
    provider: Literal["ollama"] = "ollama"
    name: str
    expected_digest: str | None = None
    think: bool = False
    endpoint: str
    temperature: float = Field(ge=0)
    max_tokens: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)


class ModelsConfig(StrictModel):
    text_embedding: EmbeddingModelConfig
    multimodal_embedding: MultimodalModelConfig
    captioning: CaptionModelConfig
    generator: LLMConfig
    judge: LLMConfig


class PreprocessingConfig(StrictModel):
    broad_category_mapping: str = "default"
    remove_suspicious_text_mismatches: bool = True
    remove_low_information_images: bool = True
    low_information_std_threshold: float = Field(ge=0)


class RetrievalConfig(StrictModel):
    backend: Literal["faiss"] = "faiss"
    metric: Literal["cosine", "inner_product", "l2"] = "cosine"
    query_mode: Literal["text_only", "image_only", "clip_text_only", "fused_multimodal"] = (
        "fused_multimodal"
    )
    candidate_pool_size: int = Field(gt=0)
    final_top_k: int = Field(gt=0)
    exclude_query_outfit: bool = False

    @model_validator(mode="after")
    def pool_must_cover_top_k(self) -> RetrievalConfig:
        if self.candidate_pool_size < self.final_top_k:
            raise ValueError("candidate_pool_size must be at least final_top_k.")
        return self


class EvidenceConfig(StrictModel):
    enabled: bool = True
    kb_version: str
    retrieval_model_role: Literal["text_embedding"] = "text_embedding"
    query_top_k: int = Field(gt=0)
    candidate_top_k: int = Field(gt=0)
    reliability_threshold: float = Field(ge=0)
    candidate_type_filtering: bool = True


class RerankingConfig(StrictModel):
    enabled: bool = True
    method: Literal["weighted_sum"] = "weighted_sum"
    clip_weight: float = Field(ge=0)
    evidence_weight: float = Field(ge=0)
    normalize_scores: bool = True

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> RerankingConfig:
        if self.enabled and abs(self.clip_weight + self.evidence_weight - 1.0) > 1e-8:
            raise ValueError("Reranking clip_weight and evidence_weight must sum to 1.")
        return self


class GenerationConfig(StrictModel):
    variants: list[Literal["no_rag", "item_rag", "rule_rag", "hybrid_rag"]]
    candidate_locked: bool = True
    leakage_safe: bool = True
    explanations_per_recommendation: int = Field(gt=0)
    prompt_version: str


class EvaluationConfig(StrictModel):
    controlled_cases: int = Field(gt=0)
    controlled_case_pool_per_target: int = Field(gt=0)
    negatives_per_case: int = Field(gt=0)
    cutoffs: list[int]
    explanation_cases_per_category: int = Field(gt=0)
    recommendations_per_case: int = Field(gt=0)
    bootstrap_samples: int = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)
    multiple_comparison_methods: list[Literal["holm", "benjamini_hochberg"]]
    human_review_examples_per_variant_category: int = Field(gt=0)


class RobustnessConfig(StrictModel):
    enabled: bool = False
    development_fraction: float = Field(gt=0, lt=1)
    validation_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(gt=0, lt=1)
    expanded_cases_per_category: int = Field(gt=0)
    generators: list[LLMConfig] = Field(default_factory=list)
    judges: list[LLMConfig] = Field(default_factory=list)
    hybrid_word_limits: list[int] = Field(default_factory=lambda: [55])
    hybrid_rule_counts: list[int] = Field(default_factory=lambda: [5])
    hybrid_prompt_orders: list[Literal["candidate_first", "rules_first"]] = Field(
        default_factory=lambda: ["candidate_first"]
    )
    rerank_clip_weights: list[float] = Field(default_factory=lambda: [0.9])

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> RobustnessConfig:
        total = self.development_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-8:
            raise ValueError("Robustness split fractions must sum to 1.")
        if any(value < 0 or value > 1 for value in self.rerank_clip_weights):
            raise ValueError("Robustness rerank weights must be between 0 and 1.")
        return self


class FinalEvaluationConfig(StrictModel):
    """Versioned settings for the final v2 evaluation protocol."""

    enabled: bool = False
    output_root: Path = Path("outputs/final_eval_v2")
    report_root: Path = Path("reports/final_eval_v2")
    fusion_image_weights: list[float] = Field(
        default_factory=lambda: [value / 10 for value in range(10, -1, -1)]
    )
    hybrid_word_budgets: list[int] = Field(default_factory=lambda: [35, 55, 75])
    hybrid_rule_counts: list[int] = Field(default_factory=lambda: [3, 5])
    hybrid_item_counts: list[int] = Field(default_factory=lambda: [0, 2, 5])
    hybrid_evidence_orders: list[Literal["rules_first", "item_first"]] = Field(
        default_factory=lambda: ["rules_first", "item_first"]
    )
    hybrid_practical_tie: float = Field(default=0.01, ge=0, le=1)
    hybrid_screening_cases_per_category: int = Field(default=10, gt=0)
    hybrid_finalist_count: int = Field(default=6, ge=4, le=6)
    proposed_reranking_selection_policy: Literal["evidence_in_loop_pareto_v2"] = (
        "evidence_in_loop_pareto_v2"
    )
    proposed_reranking_clip_weight: float = Field(default=0.75, ge=0, lt=1)
    require_stage1_validation_packets: bool = True
    claim_schema_version: str = "v2"
    judge_schema_version: str = "v2"
    primary_explanation_comparisons: list[tuple[str, str]] = Field(
        default_factory=lambda: [
            ("rule_rag", "no_rag"),
            ("hybrid_rag", "no_rag"),
            ("rule_rag", "item_rag"),
            ("hybrid_rag", "rule_rag"),
        ]
    )
    primary_retrieval_comparisons: list[tuple[str, str]] = Field(
        default_factory=lambda: [
            ("clip_fused", "minilm_text"),
            ("clip_fused", "clip_image"),
            ("clip_fused", "clip_text"),
            ("evidence_reranked", "clip_fused"),
        ]
    )

    @model_validator(mode="after")
    def validate_final_evaluation(self) -> FinalEvaluationConfig:
        if self.output_root != Path("outputs/final_eval_v2"):
            raise ValueError("Final evaluation output_root must be outputs/final_eval_v2.")
        if self.report_root != Path("reports/final_eval_v2"):
            raise ValueError("Final evaluation report_root must be reports/final_eval_v2.")
        if any(value < 0 or value > 1 for value in self.fusion_image_weights):
            raise ValueError("Fusion image weights must be between 0 and 1.")
        if len(set(self.fusion_image_weights)) != len(self.fusion_image_weights):
            raise ValueError("Fusion image weights must be unique.")
        if self.proposed_reranking_clip_weight != 0.75:
            raise ValueError("The frozen evidence-in-loop Pareto v2 CLIP weight must be 0.75.")
        return self


class CacheConfig(StrictModel):
    policy: Literal["reuse", "refresh", "disabled"] = "reuse"
    hash_algorithm: Literal["sha256"] = "sha256"
    include_code_version: bool = True


class RunConfig(StrictModel):
    experiment_name: str
    save_intermediates: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AppConfig(StrictModel):
    project: ProjectConfig
    paths: PathsConfig
    dataset: DatasetConfig
    models: ModelsConfig
    preprocessing: PreprocessingConfig
    retrieval: RetrievalConfig
    evidence: EvidenceConfig
    reranking: RerankingConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    robustness: RobustnessConfig
    final_evaluation: FinalEvaluationConfig = Field(default_factory=FinalEvaluationConfig)
    cache: CacheConfig
    run: RunConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Circular config inheritance detected at {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (path.parent / parent).resolve()
    return _deep_merge(_read_yaml(parent_path, seen), raw)


def _parse_override(value: str) -> Any:
    return yaml.safe_load(value)


def _apply_override(data: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must use key=value syntax: {expression}")
    dotted_key, raw_value = expression.split("=", 1)
    keys = dotted_key.split(".")
    cursor: dict[str, Any] = data
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            raise KeyError(f"Unknown configuration path: {dotted_key}")
        cursor = cursor[key]
    if keys[-1] not in cursor:
        raise KeyError(f"Unknown configuration key: {dotted_key}")
    cursor[keys[-1]] = _parse_override(raw_value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> AppConfig:
    """Load, inherit, override, and validate a project configuration."""
    data = _read_yaml(Path(path))
    for expression in overrides or []:
        _apply_override(data, expression)
    return AppConfig.model_validate(data)
