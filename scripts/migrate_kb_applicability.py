"""Apply the reviewed AND-of-OR query-term declarations to the canonical KB."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
KB_PATH = ROOT / "data/kb/fashion_rules.csv"
EXPANSION_PATH = ROOT / "data/kb/fashion_rules_expansion.csv"

QUERY_TERMS = {
    "K002": "chino|chinos & smart casual|smart-casual",
    "K003": "chino|chinos & trainer|trainers|sneaker|sneakers & smart casual|smart-casual",
    "K004": "loafer|loafers & jean|jeans|denim",
    "K006": "blazer|tailored jacket|sportcoat|sport coat & business casual|business-casual",
    "K009": (
        "button down|button-down|dress shirt|business shirt & business casual|business-casual"
    ),
    "K012": "trainer|trainers|sneaker|sneakers & smart casual|smart-casual",
    "K013": "blazer|tailored coat|tailored jacket & work|office|workwear",
    "K019": "bold colour|bold color|bright|vibrant & restraint|restrained|subtle",
    "K020": "blazer|sportcoat|sport coat|tailored jacket & business casual|business-casual",
    "K021": "bag|handbag|clutch|tote & wedding|wedding guest",
    "K022": "jean|jeans|denim & polish|polished|dress up|dressed up|elevate",
    "K023": "chino|chinos & smart casual|smart-casual",
    "K024": (
        "tailored trouser|tailored trousers|dress pants|smart trousers & "
        "business casual|business-casual"
    ),
    "K025": (
        "work shirt|business shirt|button up|button-up|button down|button-down & "
        "business casual|business-casual"
    ),
    "K028": "structured tote|leather tote|structured bag & smart casual|smart-casual",
    "K031": "minimal|minimalist|simple & date night|date-night|evening date",
    "K032": "festive|party & dark toned|dark-toned|black outfit|dark outfit",
    "K033": "blazer|tailored jacket & smart casual|smart-casual",
    "K034": (
        "straight leg jean|straight-leg jean|jeans|denim & "
        "blazer|tailoring|tailored & heel|heels"
    ),
    "K036": "baggy jean|baggy jeans|wide leg jean|wide-leg jean & bomber|menswear",
}

CANONICAL_SOURCE_METADATA = {
    "https://www.gq-magazine.co.uk/article/black-tie-guide": {
        "source_title": "Black tie dress code explained",
        "source_year": "2024",
    },
    "https://www.gq.com/story/business-casual-attire-for-men-explained": {
        "source_title": "What Is Business Casual? The GQ Guide to Dressing for Work and Beyond",
        "source_year": "2025",
    },
    "https://www.vogue.com/article/baggy-jean-outfits": {
        "source_title": "Baggy Jeans Are Still Trending—7 Ways to Style the Model Off-Duty Staple",
        "source_year": "2026",
    },
    "https://www.vogue.com/article/how-to-style-wide-leg-trousers": {
        "source_title": (
            "Wide-Leg Trousers Are a Vogue Office Favorite—7 Street Style-Approved Ways "
            "to Wear Them"
        ),
        "source_year": "2026",
    },
}


def main() -> None:
    rules = pd.read_csv(KB_PATH, dtype=str, keep_default_na=False)
    expansion = pd.read_csv(EXPANSION_PATH, dtype=str, keep_default_na=False)
    if list(rules.columns) != list(expansion.columns):
        raise ValueError("Expansion schema must exactly match the canonical KB schema.")
    if expansion["rule_id"].duplicated().any() or len(expansion) != 38:
        raise ValueError("Expansion must contain exactly 38 unique rules.")
    rules = pd.concat(
        [rules.loc[~rules["rule_id"].isin(expansion["rule_id"])], expansion],
        ignore_index=True,
    )
    if set(QUERY_TERMS) - set(rules["rule_id"]):
        raise ValueError("Applicability migration references an unknown rule ID.")
    rules.loc[rules["rule_id"].isin(QUERY_TERMS), "query_terms"] = rules.loc[
        rules["rule_id"].isin(QUERY_TERMS), "rule_id"
    ].map(QUERY_TERMS)
    rules.loc[rules["rule_id"].eq("K001"), "source_title"] = (
        "Berner Kühl Copenhagen Fall 2025 Collection"
    )
    k016 = rules["rule_id"].eq("K016")
    rules.loc[k016, "rule_text"] = (
        "For explicit cropped wide-leg trousers, pumps such as slingbacks or mules are a "
        "directly documented shoe direction that keeps the footwear visible."
    )
    rules.loc[k016, "source_type"] = "fashion_editorial"
    rules.loc[k016, "source_title"] = (
        "Wide-Leg Trousers Are a Vogue Office Favorite—7 Street Style-Approved Ways to Wear Them"
    )
    rules.loc[k016, "source_author_or_org"] = "Vogue"
    rules.loc[k016, "source_year"] = "2026"
    rules.loc[k016, "source_url_or_reference"] = (
        "https://www.vogue.com/article/how-to-style-wide-leg-trousers"
    )
    rules.loc[k016, "source_locator"] = "The Classic Black section"
    rules.loc[k016, "source_reliability"] = "high"
    rules.loc[k016, "evidence_summary"] = (
        "The article explicitly pairs cropped wide-leg black trousers with visible pumps, "
        "including slingbacks and mules."
    )
    rules.loc[k016, "rule_scope"] = "Cropped wide-leg trousers to pumps."
    rules.loc[k016, "rule_limitations"] = (
        "An observed editorial outfit direction; do not extend it to all wide-leg lengths."
    )
    rules["source_validation_status"] = "verified_reachable_and_direct_2026-08-16"
    for source_url, metadata in CANONICAL_SOURCE_METADATA.items():
        source_rows = rules["source_url_or_reference"].eq(source_url)
        for field, value in metadata.items():
            rules.loc[source_rows, field] = value
    status = rules.pop("source_validation_status")
    rules.insert(rules.columns.get_loc("source_reliability"), "source_validation_status", status)
    rules.to_csv(KB_PATH, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
