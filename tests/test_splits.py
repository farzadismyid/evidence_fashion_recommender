import pandas as pd

from evidence_fashion_recommender.evaluation.splits import (
    assert_disjoint_outfits,
    assign_outfit_splits,
)


def test_outfit_splits_are_stable_and_grouped() -> None:
    frame = pd.DataFrame({"outfit": ["a", "a", "b", "c", "c"], "value": range(5)})
    first = assign_outfit_splits(
        frame,
        outfit_column="outfit",
        seed=42,
        development_fraction=0.6,
        validation_fraction=0.2,
    )
    second = assign_outfit_splits(
        frame,
        outfit_column="outfit",
        seed=42,
        development_fraction=0.6,
        validation_fraction=0.2,
    )
    assert first["research_split"].tolist() == second["research_split"].tolist()
    assert_disjoint_outfits(first, "outfit")
