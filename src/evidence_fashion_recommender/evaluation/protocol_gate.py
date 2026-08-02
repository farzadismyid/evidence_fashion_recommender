"""Stage 1 comparison gate between legacy and v2 locked evaluation packets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from ..cache import stable_fingerprint

PACKET_COLUMNS = ("item_evidence_text", "rule_evidence_ids", "rule_evidence_text")
SEMANTIC_CASE_COLUMNS = ("query_item_id", "target_category")


@dataclass(frozen=True)
class MaterialChangePolicy:
    max_changed_recommendation_rate: float = 0.0
    max_changed_evidence_packet_rate: float = 0.0


def _normalized(value: object) -> str:
    return "\n".join(line.rstrip() for line in str(value or "").strip().splitlines())


def generation_packet_hash(row: pd.Series) -> str:
    return stable_fingerprint(
        {
            "recommended_item_id": str(row.get("recommended_item_id", "")),
            **{column: _normalized(row.get(column, "")) for column in PACKET_COLUMNS},
        }
    )


def compare_locked_packets(
    legacy: pd.DataFrame,
    v2: pd.DataFrame,
    *,
    policy: MaterialChangePolicy | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare paired locked recommendations and evidence without scoring test quality."""

    policy = policy or MaterialChangePolicy()
    required = {"paper_case_id", "recommended_item_id", *PACKET_COLUMNS}
    for name, frame in (("legacy", legacy), ("v2", v2)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} packets are missing columns: {sorted(missing)}")
        if frame["paper_case_id"].duplicated().any():
            raise ValueError(f"{name} packets contain duplicate paper_case_id values.")
    join_columns = ["paper_case_id"]
    if set(legacy["paper_case_id"]) != set(v2["paper_case_id"]):
        semantic = set(SEMANTIC_CASE_COLUMNS)
        for name, frame in (("legacy", legacy), ("v2", v2)):
            missing = semantic - set(frame.columns)
            if missing:
                raise ValueError(
                    "Protocol-specific case IDs differ and semantic alignment columns are "
                    f"missing from {name}: {sorted(missing)}"
                )
            if frame[list(SEMANTIC_CASE_COLUMNS)].duplicated().any():
                raise ValueError(f"{name} packets contain duplicate semantic case keys.")
        join_columns = list(SEMANTIC_CASE_COLUMNS)
    selected = [*required, *[column for column in join_columns if column not in required]]
    paired = legacy[selected].merge(
        v2[selected],
        on=join_columns,
        how="outer",
        suffixes=("_legacy", "_v2"),
        indicator=True,
    )
    paired["case_missing"] = paired["_merge"] != "both"
    paired["recommendation_changed"] = (
        paired["recommended_item_id_legacy"].astype(str)
        != paired["recommended_item_id_v2"].astype(str)
    ) | paired["case_missing"]
    packet_change_columns = []
    for column in PACKET_COLUMNS:
        changed = f"{column}_changed"
        paired[changed] = (
            paired[f"{column}_legacy"].map(_normalized)
            != paired[f"{column}_v2"].map(_normalized)
        ) | paired["case_missing"]
        packet_change_columns.append(changed)
    paired["evidence_packet_changed"] = paired[packet_change_columns].any(axis=1)
    paired["legacy_packet_hash"] = paired.apply(
        lambda row: stable_fingerprint(
            {
                "recommended_item_id": row.get("recommended_item_id_legacy", ""),
                **{
                    column: _normalized(row.get(f"{column}_legacy", ""))
                    for column in PACKET_COLUMNS
                },
            }
        ),
        axis=1,
    )
    paired["v2_packet_hash"] = paired.apply(
        lambda row: stable_fingerprint(
            {
                "recommended_item_id": row.get("recommended_item_id_v2", ""),
                **{
                    column: _normalized(row.get(f"{column}_v2", ""))
                    for column in PACKET_COLUMNS
                },
            }
        ),
        axis=1,
    )
    recommendation_rate = float(paired["recommendation_changed"].mean())
    evidence_rate = float(paired["evidence_packet_changed"].mean())
    material = (
        recommendation_rate > policy.max_changed_recommendation_rate
        or evidence_rate > policy.max_changed_evidence_packet_rate
    )
    summary: dict[str, object] = {
        "cases": len(paired),
        "changed_recommendation_rate": recommendation_rate,
        "changed_evidence_packet_rate": evidence_rate,
        "material_change": material,
        "decision": "regenerate_all_variants" if material else "legacy_generation_v2_judging",
        "policy": asdict(policy),
        "alignment_key": join_columns,
    }
    return paired.drop(columns="_merge"), summary
