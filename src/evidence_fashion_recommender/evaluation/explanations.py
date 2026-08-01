"""Transparent automatic checks for explanation grounding."""

from __future__ import annotations

import re

from ..evidence import detect_item_types

SUBSTITUTION_CUES = (
    r"\binstead(?:\s+of)?\b",
    r"\brather\s+than\b",
    r"\b(?:choose|try|select|recommend|swap(?:\s+for)?|replace(?:\s+with)?|opt\s+for)\b",
    r"\b(?:another|alternative|different)\b",
)


def candidate_substitution_types(
    explanation: str,
    allowed_types: set[str],
) -> set[str]:
    """Return product types explicitly proposed as alternatives to the locked item."""

    text = str(explanation).lower()
    substitutions: set[str] = set()
    for item_type in detect_item_types(text) - allowed_types:
        match = re.search(rf"\b{re.escape(item_type)}\b", text)
        if match is None:
            continue
        window = text[max(0, match.start() - 45) : match.end() + 45]
        if any(re.search(cue, window) for cue in SUBSTITUTION_CUES):
            substitutions.add(item_type)
    return substitutions


def substitution_detector_benchmark() -> tuple[dict[str, float], list[dict[str, object]]]:
    """Evaluate the detector on frozen positive and adversarial negative examples."""

    examples = [
        ("Choose a blazer instead of this jacket.", {"jacket"}, True),
        ("Opt for sneakers rather than the locked pumps.", {"pumps"}, True),
        ("Try a handbag as an alternative.", {"necklace"}, True),
        ("Replace it with a skirt.", {"trousers"}, True),
        ("Select another coat.", {"shoes"}, True),
        ("Swap for sandals.", {"boots"}, True),
        ("This jacket balances the skirt.", {"jacket"}, False),
        ("The shoes complement these trousers.", {"shoes"}, False),
        ("A handbag works with the query blouse.", {"handbag"}, False),
        ("The coat echoes the top's clean lines.", {"coat"}, False),
        ("These pumps pair naturally with jeans.", {"pumps"}, False),
        ("The necklace adds focus above the blazer.", {"necklace"}, False),
    ]
    rows = []
    for text, allowed, expected in examples:
        detected = bool(candidate_substitution_types(text, allowed))
        rows.append(
            {
                "text": text,
                "expected_substitution": expected,
                "detected_substitution": detected,
            }
        )
    true_positive = sum(
        row["expected_substitution"] and row["detected_substitution"] for row in rows
    )
    false_positive = sum(
        not row["expected_substitution"] and row["detected_substitution"] for row in rows
    )
    false_negative = sum(
        row["expected_substitution"] and not row["detected_substitution"] for row in rows
    )
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return {
        "examples": float(len(rows)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }, rows


def cited_rule_ids(explanation: str) -> set[str]:
    return set(re.findall(r"\[([A-Za-z]+\d+)\]", str(explanation)))


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", str(text).lower()))


def evidence_overlap(explanation: str, evidence_text: str) -> float:
    explanation_tokens = token_set(explanation)
    if not explanation_tokens:
        return 0.0
    return len(explanation_tokens & token_set(evidence_text)) / len(explanation_tokens)


def evaluate_explanation(
    explanation: str,
    available_rule_ids: set[str],
    evidence_text: str,
) -> dict[str, float | int]:
    citations = cited_rule_ids(explanation)
    return {
        "citation_presence": int(bool(citations)),
        "citation_correctness": int(citations.issubset(available_rule_ids)),
        "invalid_citation_count": len(citations - available_rule_ids),
        "evidence_overlap": evidence_overlap(explanation, evidence_text),
        "explanation_length_words": len(str(explanation).split()),
    }
