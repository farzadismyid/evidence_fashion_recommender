"""Pinned Polyvore adaptation and deterministic controlled-pool case construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tops": (
        "top",
        "tops",
        "shirt",
        "shirts",
        "blouse",
        "blouses",
        "tee",
        "t-shirt",
        "tank",
        "camisole",
        "camisoles",
        "sweater",
        "sweaters",
        "sweatshirt",
        "sweatshirts",
        "cardigan",
        "cardigans",
        "hoodie",
        "hoodies",
        "tunic",
        "tunics",
        "polo",
    ),
    "bottoms": (
        "jean",
        "jeans",
        "trouser",
        "trousers",
        "pants",
        "leggings",
        "shorts",
        "skirt",
        "skirts",
        "culottes",
        "joggers",
    ),
    "shoes": (
        "shoe",
        "shoes",
        "boot",
        "boots",
        "sneaker",
        "sneakers",
        "sandal",
        "sandals",
        "pump",
        "pumps",
        "heel",
        "heels",
        "flat",
        "flats",
        "loafer",
        "loafers",
        "oxford",
        "oxfords",
        "mule",
        "mules",
        "slipper",
        "slippers",
        "espadrille",
        "espadrilles",
    ),
    "outerwear": (
        "coat",
        "coats",
        "jacket",
        "jackets",
        "blazer",
        "blazers",
        "parka",
        "parkas",
        "poncho",
        "ponchos",
        "vest",
        "vests",
        "cape",
        "capes",
        "trench",
    ),
    "bags": (
        "bag",
        "bags",
        "handbag",
        "handbags",
        "clutch",
        "clutches",
        "tote",
        "totes",
        "messenger bag",
    ),
}
PREPARED_COLUMNS = (
    "original_dataset_index",
    "item_id",
    "outfit_id",
    "item_position",
    "category",
    "text",
    "broad_category",
    "query_category",
    "research_split",
    "exact_image_sha256",
)


@dataclass(frozen=True)
class DataValidation:
    counts: dict[str, int]
    split_outfit_overlap: int
    duplicate_item_ids: int
    invalid_item_ids: int
    case_counts: dict[str, int]
    candidate_pool_min_size: int
    candidate_pool_max_size: int
    exact_image_duplicate_groups: int
    cross_split_exact_image_groups: int
    reassigned_outfits: int


@dataclass(frozen=True)
class DuplicateResolution:
    duplicate_groups: int
    initial_cross_split_groups: int
    final_cross_split_groups: int
    component_reassignments: int
    rebalance_reassignments: int
    changed_outfits: int


def build_category_audit(
    rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]
) -> pd.DataFrame:
    """Audit every observed raw category against legacy and exact cleaned mappings."""
    taxonomy = _taxonomy(config)
    broad_values = [
        category
        for categories in taxonomy["broad_category_mapping"].values()
        for category in categories
    ]
    if len(broad_values) != len(set(broad_values)):
        raise ValueError("A raw category appears in more than one configured broad category.")
    configured_bags = set(taxonomy["broad_category_mapping"]["bags"])
    if configured_bags != set(taxonomy["bag_allowlist"]):
        raise ValueError("The bags mapping must exactly match the frozen bag allowlist.")
    forbidden_bags = set(taxonomy["bag_excluded_categories"])
    if configured_bags & forbidden_bags:
        raise ValueError("Excluded bag-like categories cannot appear in the bag allowlist.")
    if set(taxonomy["review_categories"]) & set(broad_values):
        raise ValueError("Review categories cannot also be kept categories.")
    category_column = config["dataset"]["columns"]["category"]
    raw = pd.DataFrame.from_records(rows, columns=[category_column])
    counts = raw[category_column].fillna("").astype(str).value_counts().sort_index()
    records = []
    for category, count in counts.items():
        proposed = map_broad_category(category, config)
        decision = category_decision(category, config)
        current = map_legacy_broad_category(category)
        if decision == "keep":
            reason = f"Explicit wearable {proposed} category"
            reason += "."
        elif decision == "review":
            reason = "Ambiguous aggregate category; excluded pending researcher review."
        elif current != "other":
            reason = "Rejected legacy substring match; category was not explicitly approved."
        else:
            reason = "Outside the configured five-group wearable outfit taxonomy."
        records.append(
            {
                "raw_category": category,
                "item_count": int(count),
                "current_broad_category": current,
                "proposed_broad_category": proposed if decision == "keep" else "",
                "decision": decision,
                "reason": reason,
            }
        )
    return pd.DataFrame(records)


def map_legacy_broad_category(category: object) -> str:
    """Return the pre-audit substring mapping for before/after reporting only."""
    normalized = str(category).lower()
    for broad_category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return broad_category
    return "other"


def _taxonomy(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["preprocessing"]["category_taxonomy"]


def map_broad_category(category: object, config: Mapping[str, Any]) -> str:
    raw = str(category)
    for broad_category, raw_categories in _taxonomy(config)["broad_category_mapping"].items():
        if raw in raw_categories:
            return str(broad_category)
    return "other"


def category_decision(category: object, config: Mapping[str, Any]) -> str:
    raw = str(category)
    if map_broad_category(raw, config) != "other":
        return "keep"
    if raw in _taxonomy(config)["review_categories"]:
        return "review"
    return str(_taxonomy(config)["unlisted_category_decision"])


def map_query_category(category: object, config: Mapping[str, Any]) -> str:
    return map_broad_category(category, config)


def outfit_from_item_id(item_id: object, separator: str) -> tuple[str, int | None]:
    parts = str(item_id).rsplit(separator, 1)
    if len(parts) != 2:
        return "", None
    try:
        return parts[0], int(parts[1])
    except ValueError:
        return "", None


def _unit_interval(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_research_split(
    outfit_id: object,
    *,
    seed: int,
    development_fraction: float,
    validation_fraction: float,
) -> str:
    value = _unit_interval(str(outfit_id), seed)
    if value < development_fraction:
        return "development"
    if value < development_fraction + validation_fraction:
        return "validation"
    return "test"


def assign_research_splits(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    """Assign pinned outfits by seeded SHA-256 order and configured exact split quotas."""

    split = config["splits"]
    outfits = sorted(frame["outfit_id"].astype(str).unique())
    exact_counts = split.get("exact_outfit_counts")
    if exact_counts and sum(exact_counts.values()) == len(outfits):
        counts = exact_counts
    else:
        fractions = split["requested_fractions"]
        development = int(len(outfits) * fractions["development"])
        validation = int(len(outfits) * fractions["validation"])
        counts = {
            "development": development,
            "validation": validation,
            "test": len(outfits) - development - validation,
        }
    ordered = sorted(outfits, key=lambda value: (_hash_order(value, split["seed"]), value))
    development_end = counts["development"]
    validation_end = development_end + counts["validation"]
    assignments = {
        outfit: (
            "development"
            if index < development_end
            else "validation"
            if index < validation_end
            else "test"
        )
        for index, outfit in enumerate(ordered)
    }
    return frame["outfit_id"].astype(str).map(assignments).astype("string")


def prepare_metadata(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    exact_image_hashes: Sequence[str] | None = None,
) -> pd.DataFrame:
    dataset = config["dataset"]
    columns = dataset["columns"]
    raw = pd.DataFrame.from_records(rows)
    required = {columns["item_id"], columns["category"], columns["text"]}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = pd.DataFrame(
        {
            "original_dataset_index": raw.index.astype("int64"),
            "item_id": raw[columns["item_id"]].astype("string"),
            "category": raw[columns["category"]].fillna("").astype("string"),
            "text": raw[columns["text"]].fillna("").astype("string"),
        }
    )
    parsed = frame["item_id"].map(
        lambda value: outfit_from_item_id(value, dataset["outfit_id_separator"])
    )
    frame["outfit_id"] = parsed.map(lambda value: value[0]).astype("string")
    frame["item_position"] = pd.array(parsed.map(lambda value: value[1]), dtype="Int64")
    if config["preprocessing"]["drop_missing_item_id"]:
        frame = frame[frame["item_id"].notna() & frame["item_id"].ne("")]
    if config["preprocessing"]["require_outfit_position"]:
        frame = frame[frame["outfit_id"].ne("") & frame["item_position"].notna()]
    if config["preprocessing"]["duplicate_item_policy"] == "reject":
        duplicates = int(frame["item_id"].duplicated().sum())
        if duplicates:
            raise ValueError(f"Dataset contains {duplicates} duplicate item IDs.")

    frame["broad_category"] = frame["category"].map(
        lambda value: map_broad_category(value, config)
    ).astype("string")
    frame["query_category"] = frame["category"].map(
        lambda value: map_query_category(value, config)
    ).astype("string")
    frame = frame[
        frame["category"].map(lambda value: category_decision(value, config)).eq("keep")
    ].copy()
    frame["research_split"] = assign_research_splits(frame, config)
    if exact_image_hashes is None:
        frame["exact_image_sha256"] = ""
    else:
        if len(exact_image_hashes) != len(raw):
            raise ValueError("Image hashes must align with raw dataset rows.")
        hash_by_index = dict(enumerate(exact_image_hashes))
        frame["exact_image_sha256"] = frame["original_dataset_index"].map(hash_by_index)
    return frame.loc[:, PREPARED_COLUMNS].reset_index(drop=True)


def resolve_exact_image_duplicate_splits(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, DuplicateResolution]:
    """Co-locate exact-image-linked outfits and deterministically restore split quotas."""

    policy = config["splits"]["exact_image_duplicate_policy"]
    if not policy["enabled"]:
        return frame.copy(), DuplicateResolution(0, 0, 0, 0, 0, 0)
    if frame["exact_image_sha256"].eq("").any():
        raise ValueError("Exact-image leakage resolution requires a hash for every item.")

    parent = {outfit: outfit for outfit in frame["outfit_id"].astype(str).unique()}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    duplicate_groups = 0
    for _, group in frame.groupby("exact_image_sha256", sort=True):
        outfits = sorted(group["outfit_id"].astype(str).unique())
        if len(group) > 1:
            duplicate_groups += 1
        for outfit in outfits[1:]:
            union(outfits[0], outfit)

    components: dict[str, list[str]] = {}
    for outfit in parent:
        components.setdefault(find(outfit), []).append(outfit)
    baseline = (
        frame.drop_duplicates("outfit_id")
        .set_index("outfit_id")["research_split"]
        .astype(str)
    )
    assignments = baseline.to_dict()
    initial_cross = 0
    component_moves = 0
    seed = policy["rebalance_seed"]
    for outfits in components.values():
        present = {assignments[outfit] for outfit in outfits}
        if len(present) <= 1:
            continue
        initial_cross += 1
        anchor = min(outfits, key=lambda value: (_hash_order(f"anchor:{value}", seed), value))
        destination = assignments[anchor]
        for outfit in outfits:
            if assignments[outfit] != destination:
                assignments[outfit] = destination
                component_moves += 1

    targets = config["splits"]["exact_outfit_counts"]
    split_order = policy["split_tie_order"]

    def counts() -> dict[str, int]:
        return {
            split: sum(value == split for value in assignments.values()) for split in split_order
        }

    singleton_outfits = {values[0] for values in components.values() if len(values) == 1}
    rebalance_moves = 0
    while counts() != targets:
        current = counts()
        donors = [split for split in split_order if current[split] > targets[split]]
        recipients = [split for split in split_order if current[split] < targets[split]]
        if not donors or not recipients:
            raise ValueError(
                "Unable to rebalance exact-image components to configured split quotas."
            )
        donor = donors[0]
        recipient = recipients[0]
        candidates = sorted(
            [outfit for outfit in singleton_outfits if assignments[outfit] == donor],
            key=lambda value: (
                _hash_order(f"rebalance:{donor}:{recipient}:{value}", seed),
                value,
            ),
        )
        if not candidates:
            raise ValueError(f"No singleton outfit can rebalance {donor} to {recipient}.")
        assignments[candidates[0]] = recipient
        rebalance_moves += 1

    result = frame.copy()
    result["research_split"] = result["outfit_id"].astype(str).map(assignments).astype("string")
    cross_after = int(
        (
            result.groupby("exact_image_sha256")["research_split"].nunique()
            > 1
        ).sum()
    )
    if policy["require_zero_cross_split_exact_hash_groups"] and cross_after:
        raise ValueError(f"{cross_after} exact-image groups still cross research splits.")
    changed = sum(assignments[outfit] != baseline[outfit] for outfit in assignments)
    return result, DuplicateResolution(
        duplicate_groups=duplicate_groups,
        initial_cross_split_groups=initial_cross,
        final_cross_split_groups=cross_after,
        component_reassignments=component_moves,
        rebalance_reassignments=rebalance_moves,
        changed_outfits=changed,
    )


def _hash_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def build_evaluation_cases(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    evaluation = config["recommendation_evaluation"]
    split_name = evaluation["case_split"]
    categories = evaluation["target_category_order"]
    per_category = evaluation["cases_per_category"]
    seed = evaluation["case_seed"]
    split_frame = frame[frame["research_split"] == split_name]
    target = split_frame[split_frame["broad_category"].isin(categories)]
    queries = split_frame[split_frame["query_category"].isin(categories)]

    positive_lookup = {
        (str(outfit), str(category)): sorted(group["item_id"].astype(str).tolist())
        for (outfit, category), group in target.groupby(["outfit_id", "broad_category"])
    }
    buckets: dict[str, list[dict[str, Any]]] = {category: [] for category in categories}
    taxonomy = _taxonomy(config)
    for query in queries.to_dict("records"):
        for target_category in categories:
            if (
                config["preprocessing"].get("exclude_same_effective_type_query_target", False)
                and str(query["query_category"]) == target_category
            ):
                continue
            source = positive_lookup.get((str(query["outfit_id"]), target_category), [])
            positives = [item_id for item_id in source if item_id != str(query["item_id"])]
            if not positives:
                continue
            case_key = f"{query['item_id']}::{target_category}"
            outfit_items = split_frame[
                split_frame["outfit_id"].astype(str).eq(str(query["outfit_id"]))
                & split_frame["item_id"].astype(str).ne(str(query["item_id"]))
            ].sort_values(["broad_category", "item_id"], kind="stable")
            outfit_context = " | ".join(
                f"{row['category']} {row['text']}"
                for row in outfit_items.to_dict("records")
            )
            record = {
                "case_id": "case-" + hashlib.sha256(case_key.encode()).hexdigest()[:16],
                "query_item_id": str(query["item_id"]),
                "query_outfit_id": str(query["outfit_id"]),
                "query_category": str(query["category"]),
                "query_group": str(query["query_category"]),
                "query_text": str(query["text"]),
                "outfit_context_text": outfit_context,
                "target_category": target_category,
                "user_request": taxonomy["broad_request_templates"][target_category],
                "positive_item_ids": positives,
                "num_positives": len(positives),
                "research_split": split_name,
                "_order": _hash_order(case_key, seed),
            }
            buckets[target_category].append(record)

    selected: list[dict[str, Any]] = []
    for category in categories:
        available = sorted(buckets[category], key=lambda row: (row["_order"], row["case_id"]))
        if len(available) < per_category:
            raise ValueError(
                f"Only {len(available)} {category!r} cases are available; {per_category} required."
            )
        selected.extend(available[:per_category])
    for row in selected:
        row.pop("_order")
    result = pd.DataFrame(selected)
    if len(result) != evaluation["case_count"]:
        raise ValueError("Configured case_count does not match category quota total.")
    return result


def build_candidate_pool(
    frame: pd.DataFrame,
    case: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    settings = config["candidate_pool"]
    category_items = frame[frame["broad_category"] == case["target_category"]]
    if settings["negative_source_split"] == "case_split":
        category_items = category_items[
            category_items["research_split"] == case["research_split"]
        ]
    query_item_id = str(case["query_item_id"])
    query_outfit_id = str(case["query_outfit_id"])
    positives = category_items[
        (category_items["outfit_id"].astype(str) == query_outfit_id)
        & (category_items["item_id"].astype(str) != query_item_id)
    ].copy()
    if positives.empty:
        raise ValueError("No same-outfit positive exists for the requested target category.")
    negative_pool = category_items[
        (category_items["outfit_id"].astype(str) != query_outfit_id)
        & (category_items["item_id"].astype(str) != query_item_id)
    ].copy()
    negative_pool["_order"] = negative_pool["item_id"].map(
        lambda value: _hash_order(f"{case['case_id']}:{value}", config["project"]["random_seed"])
    )
    negatives = negative_pool.sort_values(["_order", "item_id"], kind="stable").head(
        settings["max_negatives"]
    )
    candidates = pd.concat(
        [positives.assign(is_positive=True), negatives.assign(is_positive=False)],
        ignore_index=True,
    )
    candidates["_candidate_order"] = candidates["item_id"].map(
        lambda value: _hash_order(
            f"pool:{case['case_id']}:{value}", config["project"]["random_seed"]
        )
    )
    return (
        candidates.sort_values(["_candidate_order", "item_id"], kind="stable")
        .drop(columns=["_order", "_candidate_order"], errors="ignore")
        .reset_index(drop=True)
    )


def attach_candidate_pools(
    frame: pd.DataFrame, cases: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for case in cases.to_dict("records"):
        pool = build_candidate_pool(frame, case, config)
        case["candidate_item_ids"] = pool["item_id"].astype(str).tolist()
        case["candidate_relevance"] = pool["is_positive"].astype(bool).tolist()
        case["candidate_pool_size"] = len(pool)
        records.append(case)
    return pd.DataFrame(records)


def validate_prepared_data(
    frame: pd.DataFrame,
    cases: pd.DataFrame,
    config: Mapping[str, Any],
    duplicate_resolution: DuplicateResolution | None = None,
) -> DataValidation:
    target_categories = config["preprocessing"]["target_categories"]
    outfit_memberships = frame.groupby("outfit_id")["research_split"].nunique()
    overlap = int((outfit_memberships > 1).sum())
    counts = {
        "items": len(frame),
        "outfits": int(frame["outfit_id"].nunique()),
        "target_category_items": int(frame["broad_category"].isin(target_categories).sum()),
    }
    split_counts = frame.groupby("research_split")["outfit_id"].nunique().to_dict()
    counts.update({f"{name}_outfits": int(value) for name, value in split_counts.items()})
    expected = config["dataset"]["expected_counts"]
    mismatches = {
        key: (counts.get(key), value)
        for key, value in expected.items()
        if counts.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Pinned processed counts do not match: {mismatches}")
    if overlap:
        raise ValueError(f"{overlap} outfits occur in more than one research split.")
    cross_image_groups = int(
        (frame.groupby("exact_image_sha256")["research_split"].nunique() > 1).sum()
    )
    if config["splits"]["exact_image_duplicate_policy"][
        "require_zero_cross_split_exact_hash_groups"
    ] and cross_image_groups:
        raise ValueError(f"{cross_image_groups} exact-image groups cross research splits.")

    invalid_pool_cases = []
    for case in cases.to_dict("records"):
        ids = case["candidate_item_ids"]
        labels = case["candidate_relevance"]
        if str(case["query_item_id"]) in ids or not any(labels):
            invalid_pool_cases.append(case["case_id"])
        candidate_outfits = frame.set_index("item_id").loc[ids, "outfit_id"].astype(str).tolist()
        for outfit, positive in zip(candidate_outfits, labels, strict=True):
            if not positive and outfit == str(case["query_outfit_id"]):
                invalid_pool_cases.append(case["case_id"])
    if invalid_pool_cases:
        raise ValueError(f"Invalid controlled candidate pools: {sorted(set(invalid_pool_cases))}")
    return DataValidation(
        counts=counts,
        split_outfit_overlap=overlap,
        duplicate_item_ids=int(frame["item_id"].duplicated().sum()),
        invalid_item_ids=int((frame["outfit_id"] == "").sum()),
        case_counts={
            str(key): int(value) for key, value in cases["target_category"].value_counts().items()
        },
        candidate_pool_min_size=int(cases["candidate_pool_size"].min()),
        candidate_pool_max_size=int(cases["candidate_pool_size"].max()),
        exact_image_duplicate_groups=int(frame["exact_image_sha256"].duplicated(keep=False).groupby(frame["exact_image_sha256"]).any().sum()),
        cross_split_exact_image_groups=cross_image_groups,
        reassigned_outfits=duplicate_resolution.changed_outfits if duplicate_resolution else 0,
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def load_pinned_split(config: Mapping[str, Any], selected_columns: list[str] | None = None):
    """Load the exact pinned split, preferring immutable local Arrow shards when present."""

    from datasets import Dataset, concatenate_datasets, load_dataset

    dataset = config["dataset"]
    cached_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "datasets"
        / dataset["name"].replace("/", "___")
        / dataset["configuration"]
        / "0.0.0"
        / dataset["revision"]
    )
    cached_shards = sorted(cached_root.glob(f"*-{dataset['split']}-*.arrow"))
    if cached_shards:
        parts = [Dataset.from_file(str(path)) for path in cached_shards]
        split = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        observed_fingerprint = dataset["fingerprint"]
    else:
        split = load_dataset(
            dataset["name"],
            dataset["configuration"],
            split=dataset["split"],
            revision=dataset["revision"],
        )
        observed_fingerprint = split._fingerprint  # noqa: SLF001
    if observed_fingerprint != dataset["fingerprint"]:
        raise ValueError(
            f"Dataset fingerprint {observed_fingerprint!r} does not match pinned "
            f"{dataset['fingerprint']!r}."
        )
    return (
        split.select_columns(selected_columns) if selected_columns is not None else split,
        observed_fingerprint,
    )
