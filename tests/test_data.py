from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
import yaml

from evidence_fashion.data import (
    assign_research_split,
    attach_candidate_pools,
    build_candidate_pool,
    build_evaluation_cases,
    map_broad_category,
    map_query_category,
    prepare_metadata,
    resolve_exact_image_duplicate_splits,
)


@pytest.fixture
def config() -> dict:
    loaded = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    loaded["dataset"]["expected_counts"] = {}
    loaded["splits"].pop("exact_outfit_counts", None)
    loaded["recommendation_evaluation"].update(
        {
            "case_count": 5,
            "cases_per_category": 1,
            "case_split": "test",
        }
    )
    loaded["candidate_pool"].update({"max_negatives": 2, "negative_source_split": "case_split"})
    return loaded


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("Day Dresses", "other"),
        ("Skinny Jeans", "bottoms"),
        ("Ankle Booties", "shoes"),
        ("Shoulder Bags", "bags"),
        ("Backpacks", "other"),
        ("Briefcases", "other"),
        ("Blazers", "outerwear"),
        ("Dining Tables", "other"),
    ],
)
def test_validated_category_mapping(category: str, expected: str, config: dict) -> None:
    assert map_broad_category(category, config) == expected


def test_categories_outside_five_group_ontology_are_excluded(config: dict) -> None:
    assert map_query_category("Day Dresses", config) == "other"


def test_bag_allowlist_is_exact_and_exclusions_are_enforced(config: dict) -> None:
    taxonomy = config["preprocessing"]["category_taxonomy"]
    assert set(taxonomy["broad_category_mapping"]["bags"]) == set(taxonomy["bag_allowlist"])
    for category in taxonomy["bag_excluded_categories"]:
        assert map_broad_category(category, config) == "other"


def test_split_is_deterministic_and_outfit_grouped() -> None:
    first = assign_research_split(
        "outfit-1", seed=42, development_fraction=0.7, validation_fraction=0.15
    )
    second = assign_research_split(
        "outfit-1", seed=42, development_fraction=0.7, validation_fraction=0.15
    )
    assert first == second


def _synthetic_rows(config: dict) -> list[dict[str, str]]:
    rows = []
    categories = {
        "bags": "Shoulder Bags",
        "bottoms": "Skinny Jeans",
        "outerwear": "Blazers",
        "shoes": "Ankle Booties",
        "tops": "Blouses",
    }
    outfit_number = 0
    while len({row["outfit"] for row in rows}) < 20:
        outfit = f"o{outfit_number}"
        for position, (broad, category) in enumerate(categories.items()):
            rows.append(
                {
                    "outfit": outfit,
                    "item_ID": f"{outfit}_{position}",
                    "category": category,
                    "text": f"{broad} item",
                }
            )
        outfit_number += 1
    return [{key: value for key, value in row.items() if key != "outfit"} for row in rows]


def test_preparation_cases_and_candidate_pools_are_deterministic_and_valid(config: dict) -> None:
    frame = prepare_metadata(_synthetic_rows(config), config)
    first_cases = build_evaluation_cases(frame, config)
    second_cases = build_evaluation_cases(frame, config)
    pd.testing.assert_frame_equal(first_cases, second_cases)
    attached = attach_candidate_pools(frame, first_cases, config)
    assert attached["target_category"].value_counts().to_dict() == {
        category: 1 for category in config["recommendation_evaluation"]["target_category_order"]
    }
    for case in attached.to_dict("records"):
        assert case["query_item_id"] not in case["candidate_item_ids"]
        assert any(case["candidate_relevance"])
        assert case["candidate_pool_size"] <= 3


def test_pool_excludes_query_outfit_from_negatives(config: dict) -> None:
    frame = prepare_metadata(_synthetic_rows(config), config)
    case = build_evaluation_cases(frame, config).iloc[0].to_dict()
    pool = build_candidate_pool(frame, case, config)
    negatives = pool[~pool["is_positive"]]
    assert set(negatives["outfit_id"].astype(str)) != {case["query_outfit_id"]}
    assert case["query_outfit_id"] not in set(negatives["outfit_id"].astype(str))


def test_missing_schema_is_rejected(config: dict) -> None:
    broken = deepcopy(_synthetic_rows(config))
    for row in broken:
        row.pop("text")
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_metadata(broken, config)


def test_exact_image_duplicate_components_are_colocated_and_quotas_preserved(
    config: dict,
) -> None:
    rows = _synthetic_rows(config)
    hashes = [f"hash-{index}" for index in range(len(rows))]
    frame = prepare_metadata(rows, config, exact_image_hashes=hashes)
    outfits = frame.drop_duplicates("outfit_id").reset_index(drop=True)
    first_outfit = str(outfits.iloc[0]["outfit_id"])
    other_split = outfits[outfits["research_split"] != outfits.iloc[0]["research_split"]].iloc[0]
    second_outfit = str(other_split["outfit_id"])
    linked_hash = "linked-exact-image"
    frame.loc[frame["outfit_id"] == first_outfit, "exact_image_sha256"] = linked_hash
    frame.loc[frame["outfit_id"] == second_outfit, "exact_image_sha256"] = linked_hash
    before_counts = frame.groupby("research_split")["outfit_id"].nunique().to_dict()
    config["splits"]["exact_outfit_counts"] = before_counts

    resolved, audit = resolve_exact_image_duplicate_splits(frame, config)

    assert audit.initial_cross_split_groups == 1
    assert audit.final_cross_split_groups == 0
    assert resolved.groupby("research_split")["outfit_id"].nunique().to_dict() == before_counts
    linked_splits = resolved.loc[
        resolved["exact_image_sha256"] == linked_hash, "research_split"
    ].unique()
    assert len(linked_splits) == 1
