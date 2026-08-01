"""Candidate-locked, leakage-safe explanation prompts."""

from __future__ import annotations

import pandas as pd

from .models.base import Generator


def build_explanation_prompt(
    query_text: str,
    user_request: str,
    recommended_text: str,
    variant: str,
    item_evidence: list[str] | None = None,
    rule_evidence: pd.DataFrame | None = None,
    max_words: int = 55,
    rule_limit: int | None = None,
    prompt_order: str = "candidate_first",
) -> str:
    rules = []
    if rule_evidence is not None:
        rule_rows = rule_evidence.to_dict("records")
        if rule_limit is not None:
            rule_rows = rule_rows[:rule_limit]
        for row in rule_rows:
            rules.append(f"[{row['rule_id']}] {row['rule_text']}")
    sections = [
        "Write one concise fashion recommendation explanation.",
        f"Query item: {query_text}",
        f"User request: {user_request}",
        f"Locked recommended item: {recommended_text}",
        f"Grounding variant: {variant}",
        (
            "You must explain why the locked recommended item works. Never recommend, "
            "substitute, or name an alternative product type, even if the evidence mentions one."
        ),
    ]
    item_section = "Retrieved catalogue context:\n" + "\n".join(item_evidence or [])
    rule_sections = [
        "Retrieved expert rules:\n" + "\n".join(rules),
        "The explanation must cite at least one provided rule ID exactly in square brackets.",
    ]
    if variant == "hybrid_rag" and prompt_order == "rules_first":
        sections.extend(rule_sections)
        sections.append(item_section)
    else:
        if variant in {"item_rag", "hybrid_rag"}:
            sections.append(item_section)
        if variant in {"rule_rag", "hybrid_rag"}:
            sections.extend(rule_sections)
    sections.extend(
        [
            "Do not mention prompts, retrieval, datasets, or unavailable attributes.",
            "Do not invent colour, material, occasion, or styling claims.",
            f"Your explanation must refer to this exact item: {recommended_text}.",
            (
                "Use evidence only to justify the locked item. Ignore alternative products "
                "mentioned inside evidence and never repeat their names."
            ),
            f"Return only one or two sentences, no more than {max_words} words.",
        ]
    )
    return "\n\n".join(sections)


def generate_explanation(generator: Generator, **prompt_inputs: object) -> str:
    return generator.generate(build_explanation_prompt(**prompt_inputs))
