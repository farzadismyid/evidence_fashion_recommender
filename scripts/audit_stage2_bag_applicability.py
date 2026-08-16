"""Run the pre-experiment Stage 2 bag-case applicability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_fashion.kb_audit import audit_static_case_applicability, load_canonical_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--kb", type=Path, default=Path("data/kb/fashion_rules.csv"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines()]
    result = audit_static_case_applicability(cases, load_canonical_rules(args.kb), "bags")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "unsupported_cases"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not result["coverage_pass"]:
        raise SystemExit("Stage 2 applicability gate failed; Stage 2 must not be frozen.")


if __name__ == "__main__":
    main()
