from pathlib import Path

from evidence_fashion_recommender.cache import ArtifactCache, stable_fingerprint


def test_fingerprint_is_order_independent() -> None:
    assert stable_fingerprint({"a": 1, "b": 2}) == stable_fingerprint({"b": 2, "a": 1})


def test_cache_location_changes_with_inputs(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    first = cache.location("embeddings", {"model": "a"}, ".npy")
    second = cache.location("embeddings", {"model": "b"}, ".npy")
    assert first.path != second.path
