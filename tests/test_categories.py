import pytest

from evidence_fashion_recommender.data.categories import (
    map_broad_category,
    map_query_category,
)


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("Day Dresses", "other"),
        ("Skinny Jeans", "bottoms"),
        ("Ankle Booties", "shoes"),
        ("Shoulder Bags", "accessories"),
        ("Blazers", "outerwear"),
        ("Dining Tables", "other"),
    ],
)
def test_broad_category_mapping(category: str, expected: str) -> None:
    assert map_broad_category(category) == expected


def test_dresses_are_query_only() -> None:
    assert map_query_category("Day Dresses") == "dresses"
