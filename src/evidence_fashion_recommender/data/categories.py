"""Transparent category mapping preserved from the original notebook."""

from __future__ import annotations

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
    "accessories": (
        "bag",
        "bags",
        "handbag",
        "handbags",
        "clutch",
        "clutches",
        "tote",
        "totes",
        "backpack",
        "backpacks",
        "satchel",
        "satchels",
        "earring",
        "earrings",
        "necklace",
        "necklaces",
        "bracelet",
        "bracelets",
        "bangle",
        "bangles",
        "ring",
        "rings",
        "watch",
        "watches",
        "sunglasses",
        "eyewear",
        "glasses",
        "belt",
        "belts",
        "scarf",
        "scarves",
        "hat",
        "hats",
        "beanie",
        "beanies",
        "wallet",
        "wallets",
    ),
}

DRESS_KEYWORDS = ("dress", "dresses", "gown", "gowns")


def map_broad_category(category: str) -> str:
    normalized = str(category).lower()
    for broad_category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return broad_category
    return "other"


def map_query_category(category: str) -> str:
    normalized = str(category).lower()
    if any(keyword in normalized for keyword in DRESS_KEYWORDS):
        return "dresses"
    return map_broad_category(category)
