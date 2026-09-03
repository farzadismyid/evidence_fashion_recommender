"""Study-specific deterministic explanation metrics derived from saved Stage 8 records."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from evidence_fashion.grounding_contracts import BRACKETED_RE, CANONICAL_CITATION_RE
from evidence_fashion.verification_analysis import SUPPORT_A, SUPPORT_B, claim_role

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
CITATION_RE = CANONICAL_CITATION_RE
INLINE_RULE_RE = re.compile(r"\b(?:rule|rules)\s+K\d{3}(?:\s*(?:,|and)\s*K\d{3})*", re.IGNORECASE)
BARE_RULE_RE = re.compile(r"\bK\d{3}\b", re.IGNORECASE)
ANY_BRACKET_RE = BRACKETED_RE
EVIDENCE_META_RE = re.compile(r"\b(?:rules?|evidence)\b", re.IGNORECASE)
RULE_FRAMING_RE = re.compile(
    r"(?:based on|according to)\s+(?:the\s+)?(?:provided\s+|supplied\s+|"
    r"high-reliability\s+)?rules?\s*[:,]?\s*",
    re.IGNORECASE,
)

DIRECT_ATTRIBUTE_TYPES = frozenset({"body_fit", "colour", "comfort", "material", "season"})
RELATIONAL_TYPES = frozenset({"occasion", "styling_relation", "visual_match"})

RELATIONAL_TERMS = frozenset(
    {
        "balance",
        "balances",
        "balanced",
        "clash",
        "clashes",
        "cohesive",
        "complement",
        "complements",
        "compatible",
        "contrast",
        "contrasts",
        "coordinate",
        "coordinates",
        "harmonious",
        "match",
        "matches",
        "matching",
        "palette",
        "pair",
        "pairs",
        "pairing",
    }
)

PHYSICAL_TERMS = frozenset(
    {
        "ankle",
        "breathable",
        "button",
        "buttons",
        "chain",
        "collar",
        "comfortable",
        "cushion",
        "cushioned",
        "drape",
        "embroidered",
        "embroidery",
        "fringe",
        "fringed",
        "hardware",
        "heel",
        "heels",
        "lace",
        "laces",
        "lightweight",
        "lining",
        "logo",
        "logos",
        "pattern",
        "patterned",
        "pocket",
        "pockets",
        "print",
        "printed",
        "ribbed",
        "seam",
        "seams",
        "sleeve",
        "sleeves",
        "strap",
        "straps",
        "striped",
        "texture",
        "textured",
        "toe",
        "zip",
        "zipper",
        "zippers",
    }
)

ATTRIBUTE_VALUE_TERMS = frozenset(
    {
        # Common colours and colour modifiers.
        "beige",
        "black",
        "blue",
        "brown",
        "burgundy",
        "charcoal",
        "coral",
        "cream",
        "creamy",
        "dark",
        "earthy",
        "gold",
        "golden",
        "gray",
        "green",
        "grey",
        "ivory",
        "khaki",
        "light",
        "metallic",
        "multicolour",
        "navy",
        "neutral",
        "nude",
        "olive",
        "orange",
        "pastel",
        "pink",
        "purple",
        "red",
        "silver",
        "tan",
        "taupe",
        "teal",
        "white",
        "yellow",
        # Materials.
        "canvas",
        "cashmere",
        "chiffon",
        "corduroy",
        "cotton",
        "denim",
        "fabric",
        "faux",
        "fur",
        "lace",
        "leather",
        "linen",
        "mesh",
        "metal",
        "nylon",
        "polyester",
        "satin",
        "silk",
        "suede",
        "velvet",
        "wool",
        # Fit, silhouette, and concrete design properties.
        "bootcut",
        "boyfriend",
        "cropped",
        "fitted",
        "flare",
        "flared",
        "flowing",
        "high",
        "loose",
        "oversized",
        "pointed",
        "relaxed",
        "skinny",
        "slim",
        "straight",
        "structured",
        "tailored",
        "wide",
        # Comfort and season properties.
        "autumn",
        "breathable",
        "comfortable",
        "fall",
        "spring",
        "summer",
        "warm",
        "winter",
        *PHYSICAL_TERMS,
    }
)

COLOUR_TERMS = frozenset(
    {
        "beige",
        "black",
        "blue",
        "brown",
        "burgundy",
        "charcoal",
        "coral",
        "cream",
        "creamy",
        "dark",
        "earthy",
        "gold",
        "golden",
        "gray",
        "green",
        "grey",
        "ivory",
        "khaki",
        "light",
        "metallic",
        "multicolour",
        "navy",
        "neutral",
        "nude",
        "olive",
        "orange",
        "pastel",
        "pink",
        "purple",
        "red",
        "silver",
        "tan",
        "taupe",
        "teal",
        "white",
        "yellow",
    }
)
MATERIAL_TERMS = frozenset(
    {
        "canvas",
        "cashmere",
        "chiffon",
        "corduroy",
        "cotton",
        "denim",
        "fabric",
        "faux",
        "fur",
        "lace",
        "leather",
        "linen",
        "mesh",
        "metal",
        "nylon",
        "polyester",
        "satin",
        "silk",
        "suede",
        "velvet",
        "wool",
    }
)
FIT_TERMS = frozenset(
    {
        "bootcut",
        "boyfriend",
        "cropped",
        "fitted",
        "flare",
        "flared",
        "flowing",
        "loose",
        "oversized",
        "pointed",
        "relaxed",
        "skinny",
        "slim",
        "straight",
        "structured",
        "tailored",
        "wide",
    }
)
COMFORT_TERMS = frozenset(
    {"breathable", "comfortable", "cushion", "cushioned", "lightweight", "soft", "warm"}
)
SEASON_TERMS = frozenset({"autumn", "fall", "spring", "summer", "winter"})
DIRECT_TYPE_TERMS = {
    "body_fit": FIT_TERMS,
    "colour": COLOUR_TERMS,
    "comfort": COMFORT_TERMS,
    "material": MATERIAL_TERMS,
    "season": SEASON_TERMS,
}

UNSUPPORTED_QUALIFIERS = frozenset(
    {
        "durable",
        "high-quality",
        "luxurious",
        "premium",
        "quality",
        "soft",
        "sturdy",
    }
)

ITEM_TERMS = frozenset(
    {
        "accessory",
        "accessories",
        "bag",
        "belt",
        "blazer",
        "boot",
        "boots",
        "bracelet",
        "clutch",
        "coat",
        "earrings",
        "glasses",
        "handbag",
        "hat",
        "heel",
        "heels",
        "jacket",
        "jean",
        "jeans",
        "jewellery",
        "jewelry",
        "loafer",
        "loafers",
        "necklace",
        "pant",
        "pants",
        "pump",
        "pumps",
        "ring",
        "sandal",
        "sandals",
        "shirt",
        "shoe",
        "shoes",
        "shorts",
        "skirt",
        "sneaker",
        "sneakers",
        "sweater",
        "top",
        "tops",
        "trouser",
        "trousers",
        "vest",
        "watch",
    }
)


@dataclass(frozen=True)
class AttributeClassification:
    """Conservative classification of an extracted claim for UIAR."""

    bucket: str
    reason: str


def tokens(text: str) -> set[str]:
    """Return normalized lexical tokens used only by the deterministic UIAR audit."""
    return set(TOKEN_RE.findall(text.casefold().replace("high quality", "high-quality")))


def _item_texts(context_a: Mapping[str, Any]) -> list[str]:
    """Group every explicit frozen A identity/category/text field by actual item."""
    query = " ".join(
        str(context_a[key])
        for key in (
            "query_item_id",
            "query_item_category",
            "query_item_text",
            "query_item_minimal_name",
        )
        if context_a.get(key) is not None
    )
    locked = " ".join(
        str(context_a[key])
        for key in (
            "locked_candidate_id",
            "locked_item_category",
            "locked_item_text",
            "locked_item_minimal_name",
        )
        if context_a.get(key) is not None
    )
    return [value for value in (query, locked) if value]


def _matching_item_texts(claim_text: str, context_a: Mapping[str, Any]) -> list[str]:
    claim_tokens = tokens(claim_text)
    records = []
    for item_text in _item_texts(context_a):
        item_tokens = tokens(item_text)
        item_overlap = claim_tokens & item_tokens & ITEM_TERMS
        distinctive_overlap = {
            token
            for token in claim_tokens & item_tokens
            if len(token) >= 4
            and token
            not in {
                "design",
                "fashion",
                "item",
                "mens",
                "style",
                "that",
                "this",
                "with",
                "women",
                "womens",
            }
        }
        records.append((item_text, bool(item_overlap), bool(distinctive_overlap)))
    strong = [item_text for item_text, item_overlap, _ in records if item_overlap]
    return strong or [item_text for item_text, _, distinctive in records if distinctive]


def refers_to_actual_item(claim_text: str, context_a: Mapping[str, Any]) -> bool:
    """Require lexical evidence that a claim concerns the query or locked item."""
    return bool(_matching_item_texts(claim_text, context_a))


def classify_item_attribute(
    claim: Mapping[str, Any], context_a: Mapping[str, Any]
) -> AttributeClassification:
    """Classify concrete item assertions without treating styling opinions as attributes."""
    claim_type = str(claim["claim_type"])
    text = str(claim["claim_text"])
    claim_tokens = tokens(text)
    if not refers_to_actual_item(text, context_a):
        return AttributeClassification("outside", "does_not_identify_actual_item")

    has_relation = bool(claim_tokens & RELATIONAL_TERMS)
    has_physical = bool(claim_tokens & (PHYSICAL_TERMS | ATTRIBUTE_VALUE_TERMS))
    if claim_type in DIRECT_ATTRIBUTE_TYPES:
        if not claim_tokens & DIRECT_TYPE_TERMS[claim_type]:
            return AttributeClassification("ambiguous", f"{claim_type}_schema_without_marker")
        if has_relation and not has_physical:
            return AttributeClassification("outside", "subjective_or_relational_styling_claim")
        if has_relation and claim_type in {"colour", "material"}:
            return AttributeClassification("ambiguous", "mixed_attribute_and_relation")
        return AttributeClassification("item_attribute", f"direct_{claim_type}_claim")

    if claim_type in RELATIONAL_TYPES:
        if has_physical:
            return AttributeClassification("ambiguous", "relational_schema_with_physical_terms")
        return AttributeClassification("outside", "subjective_or_relational_styling_claim")

    if claim_type in {"formality", "trend", "other"} and has_physical:
        if text.casefold().lstrip().startswith(("recommend ", "recommended ", "suggest ")):
            return AttributeClassification("outside", "recommendation_identity_not_attribute")
        predicate = re.search(
            r"\b(?:has|have|features?|is|are|made from|uses|includes|contains|"
            r"constructed with)\b(.+)$",
            text.casefold(),
        )
        if predicate and tokens(predicate.group(1)) & ATTRIBUTE_VALUE_TERMS:
            return AttributeClassification("item_attribute", "explicit_physical_predicate")
        return AttributeClassification("ambiguous", "physical_term_without_attribute_predicate")
    if claim_type in {"formality", "trend", "other"}:
        return AttributeClassification("ambiguous", "mixed_schema_without_concrete_marker")
    return AttributeClassification("outside", "non_attribute_schema")


def explicit_a_entails_attribute(claim_text: str, context_a: Mapping[str, Any]) -> bool:
    """Use all explicit frozen A identity/category/text fields, without outside inference."""
    claim_tokens = tokens(claim_text)
    values = claim_tokens & ATTRIBUTE_VALUE_TERMS
    matching_items = _matching_item_texts(claim_text, context_a)
    if not values or not matching_items:
        return False
    required = values | (claim_tokens & UNSUPPORTED_QUALIFIERS)
    item_support = [required.issubset(tokens(item_text)) for item_text in matching_items]
    if {"both", "each"} & claim_tokens:
        return len(item_support) >= 2 and all(item_support)
    if not any(item_support):
        return False
    return True


def exact_b_entails_attribute(
    claim_text: str, context_a: Mapping[str, Any], rules: Sequence[Mapping[str, Any]]
) -> bool:
    """Require a rule to name the actual item and its asserted value; generic rules fail."""
    claim_tokens = tokens(claim_text)
    asserted_values = claim_tokens & ATTRIBUTE_VALUE_TERMS
    if not asserted_values:
        return False
    for item_text in _item_texts(context_a):
        normalized_item = " ".join(TOKEN_RE.findall(item_text.casefold()))
        for rule in rules:
            normalized_rule = " ".join(TOKEN_RE.findall(str(rule["rule_text"]).casefold()))
            if (
                normalized_item
                and normalized_item in normalized_rule
                and asserted_values.issubset(tokens(normalized_rule))
            ):
                return True
    return False


def dta_entailed(claim: Mapping[str, Any]) -> bool:
    """Use the saved strict B attribution and an exact supporting rule ID."""
    return (
        claim.get("support_status") == "supported"
        and SUPPORT_B in claim.get("support_sources", [])
        and bool(claim.get("supporting_rule_ids"))
    )


def valid_rule_citation(claim: Mapping[str, Any], citation_ids: Sequence[str]) -> bool:
    """Require entailment, B support, and an attributed rule actually cited in the output."""
    cited = set(citation_ids)
    supporting = set(claim.get("supporting_rule_ids", []))
    return (
        claim.get("citation_entails_claim") is True
        and dta_entailed(claim)
        and bool(cited & supporting)
    )


def citation_observation_available(claim: Mapping[str, Any]) -> bool:
    """Stage 8 uses non-null to indicate evaluation against observed output citations."""
    return claim.get("citation_entails_claim") is not None


def requires_rule_support(claim: Mapping[str, Any]) -> bool:
    """Exclude substantive claims already entailed by visible item/context evidence A."""
    return claim_role(str(claim["claim_type"])) == "substantive" and SUPPORT_A not in claim.get(
        "support_sources", []
    )


def strip_condition_revealing_citations(text: str) -> str:
    """Remove citation markers and rule IDs from a blinded human-preference rendering."""
    stripped = RULE_FRAMING_RE.sub("", text)
    stripped = CITATION_RE.sub("", stripped)
    stripped = INLINE_RULE_RE.sub("", stripped)
    stripped = BARE_RULE_RE.sub("", stripped)
    stripped = ANY_BRACKET_RE.sub("", stripped)
    stripped = EVIDENCE_META_RE.sub("", stripped)
    stripped = re.sub(
        r"\b(?:based on|according to|grounded in)\s*[,.:;]?", "", stripped, flags=re.I
    )
    stripped = re.sub(r"[ \t]+([,.;:!?])", r"\1", stripped)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    return stripped.strip()
