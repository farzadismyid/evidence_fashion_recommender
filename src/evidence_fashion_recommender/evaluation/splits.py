"""Deterministic, outfit-grouped research partitions."""

from __future__ import annotations

import hashlib

import pandas as pd


def outfit_from_item_id(item_id: str, separator: str = "_") -> str:
    return str(item_id).rsplit(separator, 1)[0]


def _unit_interval(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_outfit_splits(
    frame: pd.DataFrame,
    *,
    outfit_column: str,
    seed: int,
    development_fraction: float,
    validation_fraction: float,
) -> pd.DataFrame:
    """Assign all rows from an outfit to one stable split."""

    result = frame.copy()
    boundary_validation = development_fraction + validation_fraction

    def assign(outfit_id: object) -> str:
        value = _unit_interval(str(outfit_id), seed)
        if value < development_fraction:
            return "development"
        if value < boundary_validation:
            return "validation"
        return "test"

    result["research_split"] = result[outfit_column].map(assign)
    return result


def assert_disjoint_outfits(frame: pd.DataFrame, outfit_column: str) -> None:
    memberships = frame.groupby(outfit_column)["research_split"].nunique()
    if int(memberships.max()) != 1:
        raise ValueError("Research partitions contain overlapping outfits.")


def balanced_sample(
    frame: pd.DataFrame,
    *,
    split: str,
    category_column: str,
    cases_per_category: int,
    seed: int,
) -> pd.DataFrame:
    selected = []
    subset = frame[frame["research_split"] == split]
    for _, group in subset.groupby(category_column, sort=True):
        if len(group) < cases_per_category:
            raise ValueError(
                f"Split {split!r} has only {len(group)} rows in a category; "
                f"{cases_per_category} requested."
            )
        selected.append(group.sample(cases_per_category, random_state=seed))
    return pd.concat(selected, ignore_index=True)
