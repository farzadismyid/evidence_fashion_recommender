"""Create the bounded, source-justified Stage 5 human-gold v2 revision."""

# ruff: noqa: E501

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
REVISION = "stage5_human_gold_v2_source_only_cleanup"


def _copy_row(row: dict[str, Any], claim_id: str) -> dict[str, Any]:
    result = copy.deepcopy(row)
    result["claim_id"] = claim_id
    return result


def _replace_claims(
    record: dict[str, Any],
    claims: list[dict[str, Any]],
    verification: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> None:
    expected_ids = [f"C{index}" for index in range(1, len(claims) + 1)]
    assert [claim["claim_id"] for claim in claims] == expected_ids
    assert [row["claim_id"] for row in verification] == expected_ids
    assert [row["claim_id"] for row in citations] == expected_ids
    record["human_claims"] = claims
    record["human_verification"] = verification
    record["human_citation_validation"] = citations
    record["human_gold_revision"] = REVISION


def revise(record: dict[str, Any]) -> list[dict[str, str]]:
    key = (record["calibration_case_id"], record["condition"])
    claims = record["human_claims"]
    verification = record["human_verification"]
    citations = record["human_citation_validation"]
    changes: list[dict[str, str]] = []

    if key in {
        ("s5-tops-1-199670397", "no_rag"),
        ("s5-tops-1-199670397", "rule_rag"),
    }:
        old = claims[2]
        old_verification = verification[2]
        old_citation = citations[2]
        first = {
            **copy.deepcopy(old),
            "claim_id": "C3",
            "claim_text": "The cashmere top works with this outfit.",
        }
        second = {
            **copy.deepcopy(old),
            "claim_id": "C4",
            "claim_text": "The cashmere top keeps the overall look balanced.",
        }
        _replace_claims(
            record,
            [claims[0], claims[1], first, second],
            [
                verification[0],
                verification[1],
                _copy_row(old_verification, "C3"),
                _copy_row(old_verification, "C4"),
            ],
            [
                citations[0],
                citations[1],
                _copy_row(old_citation, "C3"),
                _copy_row(old_citation, "C4"),
            ],
        )
        changes.append(
            {
                "change": "split",
                "from": old["claim_text"],
                "to": "The cashmere top works with this outfit. / The cashmere top keeps the overall look balanced.",
                "justification": "The source asserts suitability and visual balance as independently verifiable propositions.",
            }
        )
        if key[1] == "rule_rag":
            changes.append(
                {
                    "change": "retain_single_conditional_rule_claim",
                    "from": claims[1]["claim_text"],
                    "to": claims[1]["claim_text"],
                    "justification": "The alternatives 'fitted top or cardigan' form one conditional rule proposition, not two independent claims.",
                }
            )

    elif key == ("s5-shoes-1-216844457", "rule_rag"):
        changes.append(
            {
                "change": "retain_single_conditional_rule_claim",
                "from": claims[0]["claim_text"],
                "to": claims[0]["claim_text"],
                "justification": "The shoe direction and visibility rationale are one rule consequent, not independently asserted claims.",
            }
        )

    elif key == ("s5-shoes-2-216832929", "no_rag"):
        old_relation = claims[1]
        relation_verification = verification[1]
        relation_citation = citations[1]
        transition_claim = {
            **copy.deepcopy(old_relation),
            "claim_id": "C1",
            "claim_text": "The sock booties provide a streamlined transition from the hemline to the foot.",
        }
        silhouette_claim = {
            **copy.deepcopy(old_relation),
            "claim_id": "C2",
            "claim_text": "The sock booties complement the silhouette of the cropped wide-leg pants.",
        }
        transition_verification = _copy_row(relation_verification, "C1")
        transition_verification.update(
            {
                "common_reference_item_fact_support": "unsupported",
                "common_reference_fields": [
                    "locked_item_minimal_name",
                    "query_item_minimal_name",
                    "outfit_context_text",
                ],
                "common_reference_reason": "The concrete case facts name the items but do not supply a streamlined-transition property.",
            }
        )
        silhouette_verification = _copy_row(relation_verification, "C2")
        remaining_claims = [
            {**copy.deepcopy(claims[2]), "claim_id": "C3"},
            {**copy.deepcopy(claims[3]), "claim_id": "C4"},
        ]
        _replace_claims(
            record,
            [transition_claim, silhouette_claim, *remaining_claims],
            [
                transition_verification,
                silhouette_verification,
                _copy_row(verification[2], "C3"),
                _copy_row(verification[3], "C4"),
            ],
            [
                _copy_row(relation_citation, "C1"),
                _copy_row(relation_citation, "C2"),
                _copy_row(citations[2], "C3"),
                _copy_row(citations[3], "C4"),
            ],
        )
        changes.extend(
            [
                {
                    "change": "remove",
                    "from": claims[0]["claim_text"],
                    "to": "",
                    "justification": "The explanation asserts a streamlined transition, not footwear visibility.",
                },
                {
                    "change": "split",
                    "from": old_relation["claim_text"],
                    "to": "The sock booties provide a streamlined transition from the hemline to the foot. / The sock booties complement the silhouette of the cropped wide-leg pants.",
                    "justification": "The source expresses a concrete transition claim and a separate silhouette-relation claim.",
                },
            ]
        )

    return changes


def main() -> None:
    source = ROOT / "data/calibration/stage5_annotations_done.jsonl"
    target = ROOT / "data/calibration/stage5_annotations_done_v2.jsonl"
    changelog = ROOT / "reports/stage5_human_gold_v2_changes.md"
    if target.exists() or changelog.exists():
        raise FileExistsError("Refusing to overwrite the versioned Stage 5 human-gold revision.")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    all_changes = []
    for record in records:
        for change in revise(record):
            all_changes.append(
                {
                    "calibration_case_id": record["calibration_case_id"],
                    "condition": record["condition"],
                    **change,
                }
            )
    target.write_text(
        "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    lines = ["# Stage 5 human-gold v2 source-only cleanup", "", f"Revision: `{REVISION}`", ""]
    for change in all_changes:
        lines.extend(
            [
                f"## {change['calibration_case_id']} — {change['condition']} — {change['change']}",
                "",
                f"- Before: {change['from'] or 'Removed'}",
                f"- After: {change['to'] or 'Removed'}",
                f"- Source-only justification: {change['justification']}",
                "",
            ]
        )
    changelog.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"records": len(records), "changes": len(all_changes), "output": str(target)}))


if __name__ == "__main__":
    main()
