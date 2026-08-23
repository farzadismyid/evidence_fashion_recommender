"""Create the read-only Stage 13 release audit from frozen canonical artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parents[1]
MANIFESTS = ROOT / "artifacts" / "manifests"
TABLES = ROOT / "artifacts" / "tables"
FIGURES = ROOT / "artifacts" / "figures"
REPORTS = ROOT / "reports"
RUNTIME = ROOT / ".runtime" / "current"

STAGE9 = MANIFESTS / "stage9_explanation_generation_manifest.json"
STAGE6 = MANIFESTS / "stage6_recommendation_manifest.json"
STAGE10 = MANIFESTS / "stage10_claim_extraction_manifest.json"
STAGE11 = MANIFESTS / "stage11_claim_verification_manifest.json"
STAGE12 = MANIFESTS / "stage12_evaluation_manifest.json"
SELECTION = MANIFESTS / "stage9_v3_case_selection_manifest.json"
EXPLANATIONS = RUNTIME / "explanations" / "stage9-v3-generation-b691865366b3" / "explanations.jsonl"
EXTRACTIONS = RUNTIME / "extraction" / "stage10-claim-extraction-80c0a1d2df6c" / "extractions.jsonl"
VERIFICATIONS = RUNTIME / "verification" / "stage11-claim-verification-d92e64c77a85" / "verifications.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_figures() -> list[dict]:
    findings: list[dict] = []
    for stem in ("stage12_support_rates", "stage12_uifr", "stage12_supported_per_100"):
        png = FIGURES / f"{stem}.png"
        svg = FIGURES / f"{stem}.svg"
        image = Image.open(png)
        dpi = image.info.get("dpi", (0, 0))
        findings.append(
            {
                "figure": stem,
                "svg_present": svg.exists(),
                "png_size_px": f"{image.width}x{image.height}",
                "png_dpi": f"{dpi[0]:.0f}x{dpi[1]:.0f}",
                "status": "pass" if svg.exists() and min(dpi) >= 299 else "fail",
            }
        )
    return findings


def command_status(command: list[str]) -> dict:
    """Run one required non-model validation command and retain its concise outcome."""
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "fail",
        "summary": completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "no stdout",
    }


def main() -> None:
    manifests = {name: load_json(path) for name, path in {
        "stage9": STAGE9,
        "stage6": STAGE6,
        "stage10": STAGE10,
        "stage11": STAGE11,
        "stage12": STAGE12,
        "selection": SELECTION,
    }.items()}
    explanations = load_jsonl(EXPLANATIONS)
    extractions = load_jsonl(EXTRACTIONS)
    verifications = load_jsonl(VERIFICATIONS)
    selected = load_jsonl(
        RUNTIME / "explanations" / "stage9-v3-selection-8e0dedea27a1" / "selected_locked_cases.jsonl"
    )

    category_counts = Counter(row["target_category"] for row in selected)
    explanation_keys = {(row["case_id"], row["generator"], row["condition"]) for row in explanations}
    qualitative = list(csv.DictReader((TABLES / "table_stage12_qualitative_examples_expanded.csv").open(encoding="utf-8")))
    qual_keys = {(row["case_id"], row["generator"], row["condition"]) for row in qualitative}

    accepted_explanations = sum(row["status"] == "success" for row in explanations)
    complete_verifications = sum(row["status"] == "complete" for row in verifications)
    current_artifact_paths = (EXPLANATIONS, EXTRACTIONS, VERIFICATIONS)
    thesis_sources = [ROOT / "thesis" / f"thesis_chapter_{chapter}.md" for chapter in range(1, 6)]
    obsolete_terms = ("DTA", "UIAR", "stage 8", "stage8")
    obsolete_uses = {
        str(path.relative_to(ROOT)): [term for term in obsolete_terms if term.lower() in path.read_text(encoding="utf-8").lower()]
        for path in thesis_sources
    }
    test_checks = [
        command_status([sys.executable, "-m", "pytest", "-q"]),
        command_status(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "src",
                "tests",
                "scripts/run_stage13_release_audit.py",
                "scripts/build_thesis_chapters.py",
            ]
        ),
    ]
    checks = [
        ("Frozen 1,000 recommendation cases", 1000, manifests["stage6"]["row_counts"]["locked_cases"]),
        ("Frozen 500 explanation cases", 500, len(selected)),
        ("Five-way 100-case category balance", True, all(count == 100 for count in category_counts.values()) and len(category_counts) == 5),
        ("Stage 9 matrix cells", 3000, len(explanation_keys)),
        ("Stage 9 accepted explanations", 2987, accepted_explanations),
        ("Stage 9 terminal failures", 13, manifests["stage9"]["failure_counts"]["terminal_failures"]),
        ("Stage 10 extraction records", 2987, len(extractions)),
        ("Stage 10 extracted claims", 17396, manifests["stage10"]["row_counts"]["extracted_claims"]),
        ("Stage 11 complete verification records", 2986, complete_verifications),
        ("Stage 11 verified claims", 17389, manifests["stage11"]["row_counts"]["verified_claims"]),
        ("Stage 11 terminal failures", 1, manifests["stage11"]["failure_counts"]["verification_terminal_failures"]),
        ("Qualitative rows use frozen explanation IDs", len(qual_keys), len(qual_keys & explanation_keys)),
        ("Canonical live output paths are unique", 3, len(set(current_artifact_paths))),
        ("Rebuilt DOCX chapters", 5, sum((ROOT / "thesis" / f"CHPT{chapter}.docx").exists() for chapter in range(1, 6))),
        ("Obsolete explanation metrics in thesis", 0, sum(len(value) for value in obsolete_uses.values())),
    ]
    figures = audit_figures()
    inputs = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (EXPLANATIONS, EXTRACTIONS, VERIFICATIONS, ROOT / "data" / "kb" / "fashion_rules_v3.csv",
                     ROOT / "configs" / "experiment.yaml", ROOT / "configs" / "models.yaml", ROOT / "configs" / "prompts.yaml",
                     MANIFESTS / "data_preparation_leakage_resolved_manifest.json")
    }
    all_pass = all(expected == observed for _, expected, observed in checks)
    all_pass = all_pass and all(item["status"] == "pass" for item in figures)
    all_pass = all_pass and all(item["status"] == "pass" for item in test_checks)
    status = "release_frozen_with_recorded_terminal_failures" if all_pass else "audit_requires_review"

    audit = {
        "stage": 13,
        "status": status,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_calls": 0,
        "canonical_inputs_sha256": inputs,
        "checks": [
            {"check": name, "expected": expected, "observed": observed, "status": "pass" if expected == observed else "fail"}
            for name, expected, observed in checks
        ],
        "figure_checks": figures,
        "test_checks": test_checks,
        "recorded_limitations": [
            "Stage 9 retains 13 terminal generation failures; all accepted outputs remain frozen.",
            "Stage 11 retains one terminal verification record (seven claims); analyses use paired available records.",
            "Stage 12 metrics are frozen automated-evaluator measures, not human preference or world-factual truth.",
        ],
        "current_artifacts_only": {
            "selection": str(SELECTION.relative_to(ROOT)),
            "stage9": str(EXPLANATIONS.relative_to(ROOT)),
            "stage10": str(EXTRACTIONS.relative_to(ROOT)),
            "stage11": str(VERIFICATIONS.relative_to(ROOT)),
            "stage12_primary_table": str((TABLES / "table_stage12_primary_secondary_metrics.csv").relative_to(ROOT)),
        },
    }
    write_json(MANIFESTS / "stage13_release_manifest.json", audit)
    with (TABLES / "table_stage13_release_readiness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "expected", "observed", "status"])
        writer.writeheader()
        writer.writerows(audit["checks"])
    report = ["# Stage 13 Release Audit", "", f"Status: **{status}**.", "", "## Canonical-output checks", ""]
    report.extend(f"- **{item['check']}** — {item['status']}; expected `{item['expected']}`, observed `{item['observed']}`." for item in audit["checks"])
    report.extend(["", "## Figure checks", ""])
    report.extend(f"- **{item['figure']}** — {item['status']}; SVG={item['svg_present']}, PNG={item['png_size_px']} at {item['png_dpi']} DPI." for item in figures)
    report.extend(["", "## Test checks", ""])
    report.extend(f"- **{item['command']}** — {item['status']}; {item['summary']}" for item in test_checks)
    report.extend(["", "## Recorded limitations", ""])
    report.extend(f"- {item}" for item in audit["recorded_limitations"])
    report.extend(["", "## Canonical provenance", ""])
    report.extend(f"- `{path}` — `{digest}`" for path, digest in inputs.items())
    (REPORTS / "stage13_release_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
