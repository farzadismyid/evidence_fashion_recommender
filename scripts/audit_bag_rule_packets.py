"""Audit frozen bag evidence packets after Stage 1 cases and retrieval exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_fashion.kb_audit import audit_bag_case_packets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path, help="JSONL cases containing stored evidence traces")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines()]
    result = audit_bag_case_packets(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["coverage_pass"]:
        raise SystemExit("Stage 2 coverage gate failed; see the audit output.")


if __name__ == "__main__":
    main()
