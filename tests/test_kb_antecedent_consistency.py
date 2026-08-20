import csv
from pathlib import Path

from scripts.audit_kb_antecedent_consistency import audit


def test_bounded_kb_antecedent_audit_passes_and_requires_cropped_k016():
    report = audit(Path("data/kb/fashion_rules.csv"))

    assert report["status"] == "passed"
    assert report["rule_count"] == 100
    assert "K016" in report["changed_rule_ids"]
    with Path("data/kb/fashion_rules.csv").open(encoding="utf-8", newline="") as handle:
        k016 = next(row for row in csv.DictReader(handle) if row["rule_id"] == "K016")
    assert "cropped|crop" in k016["query_terms"]
    assert "wide leg|wide-leg|palazzo" in k016["query_terms"]
