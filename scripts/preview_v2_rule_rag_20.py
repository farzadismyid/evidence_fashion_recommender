"""Produce a temporary 20-case V2-KB Rule-RAG comparison without changing frozen stages."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evidence_fashion.explanation import OllamaClient
from evidence_fashion.grounding_contracts import validate_generated_explanation
from evidence_fashion.prompt_registry import load_prompt_registry, render_prompt
from evidence_fashion.retrieval import OllamaEmbedder
from evidence_fashion.rule_retrieval import RuleRetriever, candidate_rule_representation


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rule_evidence(trace: Mapping[str, Any]) -> str:
    return "\n".join(f"[{rule['rule_id']}] {rule['rule_text']}" for rule in trace["rules"])


def _render_markdown(rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "# Temporary V2 KB Rule-RAG comparison",
        "",
        "This preview keeps the same 20 locked cases and the same `gemma4:12b` generator ",
        "as `TEMP_STAGE9_PAIRED_EXPLANATION_REVIEW.md`. It does not alter the frozen KB, ",
        "Stage 7/8/9 outputs, or the final evaluation. V2 traces are retrieved in memory ",
        "with the same strict antecedent gate and the locked candidate preserved.",
        "",
        "The V2 file marks all rules `rebuilt_v2_substantive_expert_rule`; that label is ",
        "accepted only for this isolated preview instead of the frozen production `retain` label.",
        "",
    ]
    for number, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {number:02d} — {row['target_category']} — {row['case_id']}",
                "",
                f"- Request: {row['request']}",
                f"- Query item: {row['query_item']}",
                f"- Locked recommendation: {row['locked_item']}",
                f"- Original V1 trace: {', '.join(row['v1_rule_ids'])}",
                "- V2 exact trace: "
                f"{', '.join(row['v2_rule_ids']) or 'NO STRICTLY APPLICABLE RULE'}",
                f"- No-RAG (existing): {row['no_rag']}",
                f"- V1 Rule-RAG (existing): {row['v1_rule_rag']}",
                f"- V2 Rule-RAG: {row['v2_rule_rag']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _append_v3_rows(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Add one V3 trace/response below each retained V2 comparison pair."""
    by_case = {str(row["case_id"]): row for row in rows}
    rendered: list[str] = []
    current_case: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.search(r"\b(case-[0-9a-f]+)\Z", line)
        if heading:
            current_case = heading.group(1)
        if line.startswith("- V3 exact trace:") or line.startswith("- V3 Rule-RAG:"):
            continue
        rendered.append(line)
        if line.startswith("- V2 Rule-RAG:") and current_case in by_case:
            row = by_case[current_case]
            rendered.extend(
                [
                    "- V3 exact trace: "
                    f"{', '.join(row['v2_rule_ids']) or 'NO STRICTLY APPLICABLE RULE'}",
                    f"- V3 Rule-RAG: {row['v2_rule_rag']}",
                ]
            )
    path.write_text("\n".join(rendered).rstrip() + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=Path, default=Path("data/kb/fashion_rules_v2.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--prompts", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            ".runtime/current/explanations/stage8-selection-414ac73b4696/selected_locked_cases.jsonl"
        ),
    )
    parser.add_argument(
        "--existing-generations",
        type=Path,
        default=Path(
            ".runtime/current/explanations/stage9-generation-51ea5ff43ce5/explanations.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TEMP_STAGE9_KB_V2_COMPARISON.md"),
    )
    parser.add_argument(
        "--append-v3-to",
        type=Path,
        help="Insert V3 trace/response lines into an existing V1/V2 comparison Markdown file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    registry = load_prompt_registry(args.prompts)
    kb = pd.read_csv(args.kb, keep_default_na=False)
    required = {
        "rule_id",
        "rule_text",
        "recommended_category",
        "applicable_query_categories",
        "required_context",
        "query_terms",
        "candidate_terms",
        "source_reliability",
        "audit_status",
    }
    missing = required.difference(kb.columns)
    if missing or kb["rule_id"].duplicated().any() or kb["rule_text"].eq("").any():
        raise ValueError(f"V2 KB fails preview structural checks; missing={sorted(missing)}")
    statuses = set(kb["audit_status"].astype(str))
    if len(statuses) != 1:
        raise ValueError("A preview KB must use one audited status for every rule.")
    allowed_status = next(iter(statuses))

    selected = _read_jsonl(args.selection)
    categories = ["bags", "bottoms", "outerwear", "shoes", "tops"]
    cases = []
    for category in categories:
        cases.extend(
            sorted(
                (row for row in selected if row["target_category"] == category),
                key=lambda row: row["case_id"],
            )[:4]
        )
    if len(cases) != 20:
        raise ValueError("The fixed comparison must contain four selected cases per category.")
    existing = {
        (row["case_id"], row["condition"]): row
        for row in _read_jsonl(args.existing_generations)
        if row["generator"] == "gemma4:12b" and row["status"] == "success"
    }

    settings = dict(config["rule_retrieval"])
    settings["approved_audit_status"] = allowed_status
    embedder = OllamaEmbedder(models["embedders"]["qwen3_embedding"])
    rule_vectors = embedder.encode(kb["rule_text"].astype(str).tolist())
    retriever = RuleRetriever(kb, rule_vectors, settings)
    prepared: list[dict[str, Any]] = []
    representations = []
    for row in cases:
        query_group = str(row["evidence_trace"]["query_group"])
        case = {
            "query_category": query_group,
            "query_group": query_group,
            "query_text": row["query_item_minimal_name"],
            "outfit_context_text": "",
            "user_request": row["request"],
            "applicability_contexts": [],
            "target_category": row["target_category"],
        }
        candidate = {
            "item_id": row["locked_candidate_id"],
            "category": row["target_category"],
            "text": row["locked_candidate_minimal_name"],
        }
        prepared.append({"selected": row, "case": case, "candidate": candidate})
        representations.append(candidate_rule_representation(case, candidate))
    vectors = embedder.encode(representations)
    for row, vector in zip(prepared, vectors, strict=True):
        row["trace"] = retriever.retrieve_and_score(
            case=row["case"], candidate=row["candidate"], representation_embedding=vector
        ).to_dict()

    client = OllamaClient(models["generation_defaults"])
    comparison: list[dict[str, Any]] = []
    for row in prepared:
        selected, trace = row["selected"], row["trace"]
        trace_ids = [rule["rule_id"] for rule in trace["rules"]]
        if not trace_ids:
            response_text = (
                "NOT GENERATED: no preview-KB rule passed the strict antecedent gate for "
                "this locked case, so a Rule-RAG response would be unsupported."
            )
        else:
            variables = {
                "user_request": selected["request"],
                "query_item_minimal_name": selected["query_item_minimal_name"],
                "locked_item_minimal_name": selected["locked_candidate_minimal_name"],
                "rule_evidence": _rule_evidence(trace),
            }
            rendered = render_prompt(registry, "rule_rag_explanation", variables)
            response_text = None
            errors: list[str] = []
            for attempt in range(
                int(registry["roles"][rendered["role"]]["retry"]["max_attempts"]) + 1
            ):
                prompt = str(rendered["user_prompt"])
                if attempt:
                    prompt += "\n\n" + str(
                        registry["roles"][rendered["role"]]["retry"]["retry_instruction"]
                    )
                response = client.generate(
                    "gemma4:12b",
                    prompt,
                    system_prompt=str(rendered["system_prompt"]),
                    token_limit=int(registry["roles"][rendered["role"]]["token_limit"]),
                    timeout_seconds=float(models["generation_defaults"]["timeout_seconds"])
                    * (2**attempt),
                )
                try:
                    validate_generated_explanation(
                        response.text,
                        locked_item_name=str(selected["locked_candidate_minimal_name"]),
                        target_category=str(selected["target_category"]),
                        trace_rule_ids=trace_ids,
                        citations_required=True,
                    )
                    response_text = response.text
                    break
                except ValueError as error:
                    errors.append(str(error))
            if response_text is None:
                response_text = "TERMINAL FAILURE: " + " | ".join(errors)
        old_no_rag = existing[(selected["case_id"], "no_rag")]["explanation"]
        old_rule_rag = existing[(selected["case_id"], "rule_rag")]["explanation"]
        comparison.append(
            {
                "case_id": selected["case_id"],
                "target_category": selected["target_category"],
                "request": selected["request"],
                "query_item": selected["query_item_minimal_name"],
                "locked_item": selected["locked_candidate_minimal_name"],
                "v1_rule_ids": [rule["rule_id"] for rule in selected["evidence_trace"]["rules"]],
                "v2_rule_ids": trace_ids,
                "no_rag": old_no_rag,
                "v1_rule_rag": old_rule_rag,
                "v2_rule_rag": response_text,
            }
        )
    client.unload("gemma4:12b")
    if args.append_v3_to:
        _append_v3_rows(args.append_v3_to, comparison)
        output = args.append_v3_to
    else:
        args.output.write_text(_render_markdown(comparison), encoding="utf-8", newline="\n")
        output = args.output
    print(json.dumps({"cases": len(comparison), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
