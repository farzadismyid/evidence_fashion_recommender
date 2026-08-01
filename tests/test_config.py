from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_fashion_recommender.config import load_config


def test_default_config_loads() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.retrieval.final_top_k == 5
    assert config.models.generator.name == "llama3.2"


def test_config_inheritance_and_override() -> None:
    config = load_config(
        "configs/paper_baseline.yaml",
        ["models.generator.name=llama3.3", "evaluation.controlled_cases=10"],
    )
    assert config.run.experiment_name == "paper-baseline-v3"
    assert config.models.generator.name == "llama3.3"
    assert config.evaluation.controlled_cases == 10


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(KeyError):
        load_config("configs/default.yaml", ["retrieval.unknown=1"])


def test_invalid_weights_are_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(
            "configs/default.yaml",
            ["reranking.clip_weight=0.8", "reranking.evidence_weight=0.3"],
        )
