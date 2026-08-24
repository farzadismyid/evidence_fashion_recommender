"""Build the final 200-rule main KB from V3 using full-pool interaction counts.

The source V3 KB remains untouched.  Only categories above the final quota are
trimmed, with rules ordered by observed full-pool top-1 trace participation and
then rule ID for a deterministic fair tie-break.  The previous main KB is
archived before the new main file is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from evidence_fashion.kb_audit import load_canonical_rules


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/kb/fashion_rules_v3.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/kb/fashion_rules.csv"))
    parser.add_argument(
        "--interaction-data",
        type=Path,
        default=Path(
            ".runtime/current/experiments/v3-full-pool-counterfactual-5f2d3b7ecaf1/"
            "per_case_top1_comparison.jsonl"
        ),
    )
    parser.add_argument(
        "--archive", type=Path, default=Path("data/kb/fashion_rules_legacy_120.csv")
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("artifacts/manifests/fashion_rules_final_200_selection.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_rules = load_canonical_rules(args.source)
    interaction_rows = _read_jsonl(args.interaction_data)
    if len(interaction_rows) != 1000 or len({row["case_id"] for row in interaction_rows}) != 1000:
        raise ValueError("Interaction input must be the completed 1,000-case full-pool reranking.")
    counts = Counter(
        rule_id for row in interaction_rows for rule_id in row["new_top1_rule_ids"]
    )
    if not set(counts).issubset(set(source_rules["rule_id"])):
        raise ValueError("Interaction data refers to rule IDs absent from the V3 source KB.")

    removed: list[dict[str, object]] = []
    removed_rule_ids: set[str] = set()
    target_counts: dict[str, int] = {}
    for category in ("bags", "bottoms", "outerwear", "shoes", "tops"):
        category_rules = source_rules[
            source_rules["recommended_category"].astype(str).eq(category)
        ].copy()
        target_counts[category] = len(category_rules)
        excess = len(category_rules) - 40
        if excess < 0:
            raise ValueError(f"V3 has fewer than 40 rules for {category}.")
        ordered = category_rules.assign(
            interaction_count=category_rules["rule_id"].map(counts).fillna(0).astype(int)
        ).sort_values(["interaction_count", "rule_id"], kind="stable")
        removable = ordered[
            ordered["rule_id"].str.removeprefix("K").astype(int).gt(100)
        ]
        if len(removable) < excess:
            raise ValueError(
                f"{category} lacks enough post-baseline rules to meet the 40-rule quota."
            )
        for _, row in removable.head(excess).iterrows():
            removed.append(
                {
                    "rule_id": str(row["rule_id"]),
                    "recommended_category": category,
                    "interaction_count": int(row["interaction_count"]),
                }
            )
            removed_rule_ids.add(str(row["rule_id"]))

    final_rules = source_rules[
        ~source_rules["rule_id"].astype(str).isin(removed_rule_ids)
    ].copy()
    final_rules = final_rules.sort_values("rule_id", kind="stable")
    final_counts = (
        final_rules.groupby("recommended_category").size().reindex(
            ["bags", "bottoms", "outerwear", "shoes", "tops"], fill_value=0
        )
    )
    if len(source_rules) != 209 or len(removed) != 9 or len(final_rules) != 200:
        raise ValueError("Expected a 209-to-200 V3 reduction with exactly nine removals.")
    if not final_counts.eq(40).all():
        raise ValueError("Final KB must have exactly 40 rules in each target category.")
    required_baseline = {f"K{index:03d}" for index in range(1, 101)}
    if not required_baseline.issubset(set(final_rules["rule_id"].astype(str))):
        raise ValueError("The audited K001-K100 baseline must remain intact.")
    if args.archive.exists():
        raise FileExistsError(f"Refusing to replace an existing archive: {args.archive}")
    if args.audit_output.exists():
        raise FileExistsError(
            f"Refusing to replace an existing selection audit: {args.audit_output}"
        )
    archived_main: dict[str, str] | None = None
    if args.output.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, args.archive)
        archived_main = {"path": str(args.archive), "sha256": _sha256(args.archive)}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    final_rules.to_csv(args.output, index=False, encoding="utf-8", lineterminator="\n")
    load_canonical_rules(args.output)
    audit = {
        "schema_version": 1,
        "purpose": "final_200_rule_main_kb_selection",
        "source_kb": {"path": str(args.source), "sha256": _sha256(args.source), "rules": 209},
        "interaction_evidence": {
            "path": str(args.interaction_data),
            "sha256": _sha256(args.interaction_data),
            "definition": (
                "number of 1,000-case full-pool V3 reranking top-1 traces containing rule"
            ),
            "cases": 1000,
        },
        "selection_policy": {
            "target_rules_per_category": 40,
            "fairness": (
                "trim only categories above quota; choose lowest interaction count within each "
                "such category after preserving the audited K001-K100 baseline"
            ),
            "tie_break": "ascending canonical rule_id",
            "protected_rules": "K001-K100 audited baseline",
        },
        "source_counts_by_category": target_counts,
        "removed_rules": removed,
        "final_counts_by_category": {key: int(value) for key, value in final_counts.items()},
        "previous_main_kb_archive": archived_main,
        "output_main_kb": {"path": str(args.output), "sha256": _sha256(args.output), "rules": 200},
    }
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
