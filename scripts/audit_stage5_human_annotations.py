"""Source-only QA for completed Stage 5 human claims; never edits annotations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
STOPWORDS = frozenset(
    "a an and are as at be because for from in is it of on or that the this to with".split()
)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def _resolve_source_coreference(claim: str, facts: dict[str, Any]) -> str:
    locked = str(facts.get("locked_item_minimal_name", ""))
    query = str(facts.get("query_item_minimal_name", ""))
    result = re.sub(r"\b(the )?(exact|locked|recommended) item\b", locked, claim, flags=re.I)
    result = re.sub(r"\b(the )?query item\b", query, result, flags=re.I)
    return re.sub(r"\b(it|this item|that item)\b", locked, result, flags=re.I)


def audit(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues = []
    for record in records:
        explanation = str(record["explanation"])
        explanation_tokens = _tokens(explanation)
        seen: dict[str, str] = {}
        for claim in record["human_claims"]:
            claim_id = str(claim["claim_id"])
            text = str(claim["claim_text"])
            resolved = _resolve_source_coreference(text, record["common_reference_item_facts"])
            tokens = _tokens(resolved)
            coverage = len(tokens & explanation_tokens) / len(tokens) if tokens else 0.0
            flags = []
            if coverage < 0.45:
                flags.append("possible_not_asserted_by_source")
            if re.search(r"\b(and|or)\b|;", text, flags=re.I):
                flags.append("possible_compound_claim")
            normalized = " ".join(text.lower().split())
            if normalized in seen:
                flags.append(f"duplicate_of_{seen[normalized]}")
            else:
                seen[normalized] = claim_id
            claim_negative = bool(re.search(r"\b(no|not|never|without|cannot|can't)\b", text, re.I))
            source_negative = bool(
                re.search(r"\b(no|not|never|without|cannot|can't)\b", explanation, re.I)
            )
            if claim_negative and not source_negative:
                flags.append("possible_polarity_error")
            if flags:
                issues.append(
                    {
                        "calibration_case_id": record["calibration_case_id"],
                        "condition": record["condition"],
                        "claim_id": claim_id,
                        "claim_text": text,
                        "source_token_coverage": round(coverage, 3),
                        "flags": flags,
                        "source_explanation": explanation,
                    }
                )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/calibration/stage5_annotations_done.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/stage5_annotation_errata.md"))
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in (ROOT / args.annotations).read_text(encoding="utf-8").splitlines()
        if line
    ]
    issues = audit(records)
    counts = Counter(flag for issue in issues for flag in issue["flags"])
    lines = [
        "# Stage 5 annotation errata (source-only QA)",
        "",
        "This report compares human claims only with each record's original explanation and "
        "packet-grounded item coreference. It does not use Qwen or Phi outputs and does not "
        "alter the annotation file. Flags are review candidates, not silent corrections.",
        "",
        f"- Records reviewed: {len(records)}",
        f"- Flagged claims: {len(issues)}",
        f"- Flags: {dict(sorted(counts.items()))}",
        "",
    ]
    for issue in issues:
        lines.extend(
            [
                f"## {issue['calibration_case_id']} — {issue['condition']} — {issue['claim_id']}",
                "",
                f"- Claim: {issue['claim_text']}",
                f"- Source token coverage: {issue['source_token_coverage']}",
                f"- Flags: {', '.join(issue['flags'])}",
                f"- Original explanation: {issue['source_explanation']}",
                "",
            ]
        )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(
        json.dumps({"records": len(records), "flagged_claims": len(issues), "output": str(output)})
    )


if __name__ == "__main__":
    main()
