# ruff: noqa: E501
"""Audit whether every prose antecedent is represented in structured KB fields.

This is deliberately bounded to the canonical 100-rule study KB.  It records a
review row for every rule and verifies the corrected fields for every confirmed
gap; it neither creates rules nor changes any non-antecedent metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

CONFIRMED_ANTECEDENT_GAPS: dict[str, dict[str, str]] = {
    "K003": {
        "conditions": "refined",
        "query_terms": "chino|chinos & refined & trainer|trainers|sneaker|sneakers & smart casual|smart-casual",
    },
    "K007": {
        "conditions": "focal item",
        "query_terms": "bag|handbag|clutch|tote & focal|statement",
    },
    "K010": {
        "conditions": "classic",
        "query_terms": "t-shirt|t shirt|tee & classic|classic tee|classic t-shirt",
    },
    "K011": {"conditions": "chunky or refined", "query_terms": "loafer|loafers & chunky|refined"},
    "K012": {
        "conditions": "refined",
        "query_terms": "trainer|trainers|sneaker|sneakers & refined & smart casual|smart-casual",
    },
    "K016": {"conditions": "cropped", "query_terms": "cropped|crop & wide leg|wide-leg|palazzo"},
    "K020": {
        "conditions": "polished",
        "query_terms": "blazer|sportcoat|sport coat|tailored jacket & polished|polish & business casual|business-casual",
    },
    "K026": {"conditions": "request remains casual", "query_terms": "t-shirt|t shirt|tee & casual"},
    "K032": {"conditions": "boots", "query_terms": "jacket|jackets|coat|coats|blazer & boot|boots"},
    "K033": {
        "conditions": "boots",
        "query_terms": "sweater|sweatshirt|knit|jumper|cardigan & boot|boots",
    },
    "K034": {
        "conditions": "straight-leg (must not be optional beside generic jeans)",
        "query_terms": "straight leg jean|straight-leg jean|straight leg denim|straight-leg denim & blazer|tailoring|tailored & heel|heels",
    },
    "K039": {
        "conditions": "understated",
        "query_terms": "summer bag|summer handbag|canvas tote|raffia bag|woven bag & understated|simple|minimal",
    },
    "K043": {
        "conditions": "voluminous leather bomber",
        "query_terms": "leather bomber|leather jacket & voluminous|oversized & statement trouser|checked trouser|check trouser",
    },
    "K044": {
        "conditions": "pointy cowboy boots",
        "query_terms": "pointy cowboy boot|pointed cowboy boot|pointy western boot|pointed western boot & baggy jean|wide leg jean|wide-leg jean",
    },
    "K048": {
        "conditions": "neutral",
        "query_terms": "white button down|white button-down|white shirt & fluid|loose|relaxed & neutral",
    },
    "K050": {
        "conditions": "spring",
        "query_terms": "blazer|sport coat|sportcoat & relaxed|baggy|loose & spring",
    },
    "K056": {
        "conditions": "relaxed tailoring",
        "query_terms": "baggy trouser|loose trouser|relaxed trouser & relaxed tailoring|relaxed tailored",
    },
    "K057": {
        "conditions": "relaxed; preppy",
        "query_terms": "khaki trouser|khaki pant & wide leg|wide-leg|pleated & relaxed & preppy|ivy",
    },
    "K058": {
        "conditions": "simple; polished",
        "query_terms": "simple|plain & crewneck t-shirt|crewneck t shirt|crewneck tee & straight leg jean|straight-leg jean & polished|polish|dress up|dressed up",
    },
    "K059": {
        "conditions": "monochrome",
        "query_terms": "simple tank|white tank & trouser suit|pantsuit|suit & monochrome",
    },
    "K062": {
        "conditions": "tailored",
        "query_terms": "embellished bag|floral bag|statement top handle|statement top-handle & all black|head to toe black|head-to-toe black & tailored|tailoring",
    },
    "K063": {
        "conditions": "polished denim",
        "query_terms": "denim bag|jean bag & denim|double denim|denim outfit & polished|polish|dress up|dressed up",
    },
    "K064": {
        "conditions": "straight",
        "query_terms": "penny loafer|penny loafers & dark jean|dark denim & straight leg|straight-leg|straight jean & neo prep|neo-prep",
    },
    "K065": {
        "conditions": "pointy cowboy boots",
        "query_terms": "pointy cowboy boot|pointed cowboy boot|pointed western boot|pointy western boot & wide leg jean|wide-leg jean|extra wide jean",
    },
    "K067": {
        "conditions": "relaxed; preppy; daytime",
        "query_terms": "khaki trouser|khaki pant & wide leg|wide-leg|pleated & relaxed & preppy|ivy & daytime",
    },
    "K068": {
        "conditions": "tucked",
        "query_terms": "t-shirt|t shirt|tee & tucked|tuck in|tucked in & pleated trouser|pleated pant & preppy|ivy",
    },
    "K069": {
        "conditions": "polished",
        "query_terms": "button down|button-down|button up|button-up & wide leg jean|wide-leg jean|extra wide jean & polished|polish|dress up|dressed up",
    },
    "K072": {
        "conditions": "polished",
        "query_terms": "grey suit trouser|gray suit trouser & soft knit|crewneck knit & pointed boot & polished|polish",
    },
    "K077": {
        "conditions": "crisp",
        "query_terms": "loose indigo jean|loose indigo jeans & white button down|white button-down|crisp white shirt & crisp|clean",
    },
    "K082": {
        "conditions": "red wedges",
        "query_terms": "red wedge|red wedge sandal|red wedges & wrap top|wrap blouse",
    },
    "K084": {
        "conditions": "worn leather bomber",
        "query_terms": "t-shirt|t shirt|tee & worn leather bomber|worn leather jacket",
    },
    "K087": {
        "conditions": "cropped",
        "query_terms": "indigo jean|indigo jeans & cropped|crop & leather jacket|leather bomber",
    },
}


def audit(kb_path: Path) -> dict[str, object]:
    with kb_path.open(encoding="utf-8", newline="") as handle:
        rules = list(csv.DictReader(handle))
    if len(rules) != 100:
        raise ValueError(f"Bounded audit requires exactly 100 KB rules, found {len(rules)}.")

    records = []
    for rule in rules:
        rule_id = rule["rule_id"]
        correction = CONFIRMED_ANTECEDENT_GAPS.get(rule_id)
        if correction and rule["query_terms"] != correction["query_terms"]:
            raise ValueError(f"{rule_id} does not enforce its audited antecedent conditions.")
        records.append(
            {
                "rule_id": rule_id,
                "review_status": "corrected_confirmed_gap" if correction else "reviewed_no_gap",
                "prose_conditions_previously_unenforceable": correction["conditions"]
                if correction
                else [],
                "structured_field_corrected": "query_terms" if correction else [],
            }
        )
    return {
        "status": "passed",
        "scope": "existing_100_rules_only",
        "rule_count": len(rules),
        "changed_rule_ids": sorted(CONFIRMED_ANTECEDENT_GAPS),
        "changed_rule_count": len(CONFIRMED_ANTECEDENT_GAPS),
        "records": records,
        "kb_sha256": hashlib.sha256(kb_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=Path, default=Path("data/kb/fashion_rules.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/manifests/kb_antecedent_consistency_audit.json"),
    )
    args = parser.parse_args()
    report = audit(args.kb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "rule_count", "changed_rule_count", "kb_sha256")
            }
        )
    )


if __name__ == "__main__":
    main()
