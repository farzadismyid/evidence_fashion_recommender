"""Stage 5 human-calibration validation and model-quality gate calculations."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .assessment import common_reference_eligibility

CLAIM_STATUSES = ("supported", "unsupported", "contradicted", "not_verifiable")
ENTAILMENT_FIELDS = (
    "full_kb_entailment",
    "exact_trace_entailment",
    "common_reference_item_fact_support",
)
VERIFICATION_RULE_ID_FIELDS = (
    "full_kb_candidate_applicable_rule_ids",
    "full_kb_rule_ids",
    "exact_trace_rule_ids",
)
VERIFICATION_REQUIRED_FIELDS = (
    "claim_id",
    *VERIFICATION_RULE_ID_FIELDS,
    *ENTAILMENT_FIELDS,
    "full_kb_reason",
    "exact_trace_reason",
    "common_reference_fields",
    "common_reference_reason",
)
CITATION_REQUIRED_FIELDS = (
    "claim_id",
    "citation_present",
    "canonical_citation_format",
    "cited_rule_ids",
    "invalid_rule_ids",
    "citation_entails_claim",
    "brief_reason",
)
CANONICAL_RULE_ID_RE = re.compile(r"K\d{3}\Z")


def _normalized_claim(text: str) -> str:
    return " ".join(text.lower().split())


_SEMANTIC_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "because",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
        "your",
        "has",
        "have",
        "make",
        "makes",
        "provide",
        "provides",
    }
)
_SEMANTIC_SYNONYMS = {
    "choice": "suitable",
    "complements": "complement",
    "documented": "supported",
    "good": "suitable",
    "option": "suitable",
    "recommendation": "suitable",
    "suitable": "suitable",
    "works": "suitable",
}
_GENERIC_SEMANTIC_TOKENS = frozenset({"claim", "exact", "item", "look", "outfit", "supported"})


def _semantic_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {
        _SEMANTIC_SYNONYMS.get(token, token) for token in tokens if token not in _SEMANTIC_STOPWORDS
    }


def _coreference_normalized(text: str, context: Mapping[str, Any] | None) -> str:
    """Resolve only packet-grounded item references for calibration alignment."""
    resolved = text.lower()
    if not context:
        return resolved
    aliases = {
        "LOCKED_ITEM": str(context.get("locked_item_minimal_name", "")).lower(),
        "QUERY_ITEM": str(context.get("query_item_minimal_name", "")).lower(),
    }
    for marker, alias in aliases.items():
        if not alias:
            continue
        resolved = re.sub(rf"\b{re.escape(alias)}\b", marker.lower(), resolved)
        words = re.findall(r"[a-z0-9]+", alias)
        if len(words) > 1:
            resolved = re.sub(rf"\b{re.escape(words[-1])}\b", marker.lower(), resolved)
    resolved = re.sub(r"\b(the )?(exact|locked|recommended) item\b", "locked_item", resolved)
    resolved = re.sub(r"\b(the )?query item\b", "query_item", resolved)
    resolved = re.sub(r"\b(it|this item|that item)\b", "locked_item", resolved)
    return resolved


def _has_negative_polarity(text: str) -> bool:
    return bool(re.search(r"\b(no|not|never|without|cannot|can't)\b", text.lower()))


def _semantic_alignment_score(
    human_text: str, qwen_text: str, context: Mapping[str, Any] | None
) -> tuple[float | None, str]:
    """Small deterministic calibration-only proposition score with polarity/entity guards."""
    if _has_negative_polarity(human_text) != _has_negative_polarity(qwen_text):
        return None, "polarity_mismatch"
    human_tokens = _semantic_tokens(_coreference_normalized(human_text, context))
    qwen_tokens = _semantic_tokens(_coreference_normalized(qwen_text, context))
    if not human_tokens or not qwen_tokens:
        return None, "empty_content_tokens"
    shared = human_tokens & qwen_tokens
    entity_shared = shared - _GENERIC_SEMANTIC_TOKENS
    if not entity_shared:
        return None, "no_shared_entity"
    human_only = human_tokens - qwen_tokens - _GENERIC_SEMANTIC_TOKENS
    qwen_only = qwen_tokens - human_tokens - _GENERIC_SEMANTIC_TOKENS
    if (human_only and not qwen_only) or (qwen_only and not human_only):
        return None, "split_or_merge_atomization"
    return len(shared) / len(human_tokens | qwen_tokens), "eligible"


def _proposition_relation(
    human_text: str, qwen_text: str, context: Mapping[str, Any] | None
) -> tuple[float | None, str]:
    """Calibration-only many-to-many content relation, separate from scorer alignment."""
    if _has_negative_polarity(human_text) != _has_negative_polarity(qwen_text):
        return None, "polarity_mismatch"
    human_tokens = _semantic_tokens(_coreference_normalized(human_text, context))
    qwen_tokens = _semantic_tokens(_coreference_normalized(qwen_text, context))
    if not human_tokens or not qwen_tokens:
        return None, "empty_content_tokens"
    shared = human_tokens & qwen_tokens
    if not shared - _GENERIC_SEMANTIC_TOKENS:
        return None, "no_shared_entity"
    score = len(shared) / len(human_tokens | qwen_tokens)
    if score < 0.5:
        return None, "below_semantic_coverage_threshold"
    human_only = human_tokens - qwen_tokens - _GENERIC_SEMANTIC_TOKENS
    qwen_only = qwen_tokens - human_tokens - _GENERIC_SEMANTIC_TOKENS
    return score, "semantic_equivalent" if not (
        human_only or qwen_only
    ) else "atomized_or_specificity_variant"


def align_calibration_claims(
    human_claims: Sequence[Mapping[str, Any]],
    qwen_claims: Sequence[Mapping[str, Any]],
    *,
    common_reference_item_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, one-to-one calibration-only proposition alignment."""
    pairs: list[dict[str, Any]] = []
    unmatched_human = set(range(len(human_claims)))
    unmatched_qwen = set(range(len(qwen_claims)))
    candidate_scores: list[dict[str, Any]] = []
    qwen_by_text: dict[str, list[int]] = {}
    for index, claim in enumerate(qwen_claims):
        qwen_by_text.setdefault(
            _normalized_claim(
                _coreference_normalized(
                    str(claim.get("claim_text", "")), common_reference_item_facts
                )
            ),
            [],
        ).append(index)
    for human_index, claim in enumerate(human_claims):
        normalized = _normalized_claim(
            _coreference_normalized(str(claim["claim_text"]), common_reference_item_facts)
        )
        options = qwen_by_text.get(normalized, [])
        qwen_index = next((index for index in options if index in unmatched_qwen), None)
        if qwen_index is None:
            continue
        pairs.append(
            {
                "human_claim_id": str(claim["claim_id"]),
                "qwen_claim_id": str(qwen_claims[qwen_index]["claim_id"]),
                "method": "exact_normalized_text",
                "similarity": 1.0,
            }
        )
        unmatched_human.discard(human_index)
        unmatched_qwen.discard(qwen_index)
        candidate_scores.append(
            {
                "human_claim_id": str(claim["claim_id"]),
                "qwen_claim_id": str(qwen_claims[qwen_index]["claim_id"]),
                "score": 1.0,
                "decision": "matched_exact_normalized_or_coreference",
            }
        )
    coreference_keys = (
        {
            index: " ".join(
                sorted(
                    _semantic_tokens(
                        _coreference_normalized(
                            str(claim.get("claim_text", "")), common_reference_item_facts
                        )
                    )
                )
            )
            for index, claim in enumerate(qwen_claims)
        }
        if common_reference_item_facts
        else {}
    )
    for human_index in sorted(unmatched_human) if common_reference_item_facts else []:
        human_key = " ".join(
            sorted(
                _semantic_tokens(
                    _coreference_normalized(
                        str(human_claims[human_index]["claim_text"]),
                        common_reference_item_facts,
                    )
                )
            )
        )
        qwen_index = next(
            (
                index
                for index in sorted(unmatched_qwen)
                if human_key and coreference_keys[index] == human_key
            ),
            None,
        )
        if qwen_index is None:
            continue
        pairs.append(
            {
                "human_claim_id": str(human_claims[human_index]["claim_id"]),
                "qwen_claim_id": str(qwen_claims[qwen_index]["claim_id"]),
                "method": "coreference_normalized_text",
                "similarity": 1.0,
            }
        )
        unmatched_human.remove(human_index)
        unmatched_qwen.remove(qwen_index)
        candidate_scores.append(
            {
                "human_claim_id": str(human_claims[human_index]["claim_id"]),
                "qwen_claim_id": str(qwen_claims[qwen_index]["claim_id"]),
                "score": 1.0,
                "decision": "matched_coreference_normalized_text",
            }
        )
    candidates = []
    for human_index in sorted(unmatched_human):
        for qwen_index in sorted(unmatched_qwen):
            score, reason = _semantic_alignment_score(
                str(human_claims[human_index]["claim_text"]),
                str(qwen_claims[qwen_index].get("claim_text", "")),
                common_reference_item_facts,
            )
            candidate_scores.append(
                {
                    "human_claim_id": str(human_claims[human_index]["claim_id"]),
                    "qwen_claim_id": str(qwen_claims[qwen_index]["claim_id"]),
                    "score": round(score, 6) if score is not None else None,
                    "decision": "candidate" if score is not None and score >= 0.5 else reason,
                }
            )
            if score is not None and score >= 0.5:
                candidates.append((-score, human_index, qwen_index))
    for negative_score, human_index, qwen_index in sorted(candidates):
        if human_index not in unmatched_human or qwen_index not in unmatched_qwen:
            continue
        pairs.append(
            {
                "human_claim_id": str(human_claims[human_index]["claim_id"]),
                "qwen_claim_id": str(qwen_claims[qwen_index]["claim_id"]),
                "method": "semantic_proposition",
                "similarity": round(-negative_score, 6),
            }
        )
        unmatched_human.remove(human_index)
        unmatched_qwen.remove(qwen_index)
    proposition_relations = []
    for human_claim in human_claims:
        for qwen_claim in qwen_claims:
            score, relation = _proposition_relation(
                str(human_claim["claim_text"]),
                str(qwen_claim.get("claim_text", "")),
                common_reference_item_facts,
            )
            if score is not None:
                proposition_relations.append(
                    {
                        "human_claim_id": str(human_claim["claim_id"]),
                        "qwen_claim_id": str(qwen_claim["claim_id"]),
                        "similarity": round(score, 6),
                        "relation": relation,
                    }
                )
    return {
        "pairs": pairs,
        "candidate_scores": candidate_scores,
        "proposition_relations": proposition_relations,
        "unmatched_human_claim_ids": [
            str(human_claims[index]["claim_id"]) for index in sorted(unmatched_human)
        ],
        "unmatched_qwen_claim_ids": [
            str(qwen_claims[index]["claim_id"]) for index in sorted(unmatched_qwen)
        ],
    }


def calibration_alignment_records(
    human_records: Sequence[Mapping[str, Any]], model_records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    model_by_key = {
        (str(row["calibration_case_id"]), str(row["condition"])): row for row in model_records
    }
    records = []
    for human in human_records:
        key = (str(human["calibration_case_id"]), str(human["condition"]))
        model = model_by_key.get(key)
        if model is None:
            continue
        records.append(
            {
                "calibration_case_id": key[0],
                "condition": key[1],
                **align_calibration_claims(
                    human["human_claims"],
                    model.get("claims", []),
                    common_reference_item_facts=human.get("common_reference_item_facts"),
                ),
            }
        )
    return records


def _require_mapping(record: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = record.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping.")
    return value


def validate_human_calibration(
    records: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]
) -> dict[str, Any]:
    """Refuse anything that cannot serve as disjoint human calibration evidence."""
    calibration = _require_mapping(settings, "calibration")
    if not records:
        raise ValueError("Stage 5 has no human calibration annotations.")
    required_fields = set(calibration["required_human_fields"])
    required_tags = set(calibration["required_coverage_tags"])
    categories = set(calibration["target_categories"])
    conditions = set(calibration["conditions"])
    pairs: dict[str, set[str]] = {}
    observed_tags: set[str] = set()
    counts = Counter()
    for record in records:
        missing = required_fields - set(record)
        if missing:
            raise ValueError(f"Human annotation is missing fields: {sorted(missing)}.")
        if record.get("source_split") != calibration["source_split"]:
            raise ValueError(
                "Calibration records must come only from the reserved validation split."
            )
        if record.get("target_category") not in categories:
            raise ValueError("Calibration record has an unsupported target category.")
        condition = record.get("condition")
        if condition not in conditions:
            raise ValueError("Calibration record has an unsupported condition.")
        case_id = str(record.get("calibration_case_id", "")).strip()
        if not case_id:
            raise ValueError("Calibration records need a calibration_case_id.")
        pairs.setdefault(case_id, set()).add(str(condition))
        counts[str(record["target_category"])] += 1
        observed_tags.update(record.get("coverage_tags", []))
        if not str(record["annotator_id"]).strip() or not str(record["completed_at_utc"]).strip():
            raise ValueError(
                "Calibration annotations require a human annotator ID and completion time."
            )
        claims = record["human_claims"]
        if not isinstance(claims, list) or not claims:
            raise ValueError("Human annotation needs at least one atomic claim.")
        claim_ids = [claim.get("claim_id") for claim in claims if isinstance(claim, Mapping)]
        expected_ids = [f"C{index}" for index in range(1, len(claims) + 1)]
        if claim_ids != expected_ids:
            raise ValueError("Human claims must use consecutive C IDs in textual order.")
        verification = record["human_verification"]
        citations = record["human_citation_validation"]
        if not isinstance(verification, list) or not isinstance(citations, list):
            raise ValueError("Human verification and citation validation must be arrays.")
        for rows, label in ((verification, "verification"), (citations, "citation validation")):
            if [row.get("claim_id") for row in rows] != claim_ids:
                raise ValueError(f"Human {label} must cover each human claim exactly once.")
        for row in verification:
            if any(field not in row for field in VERIFICATION_REQUIRED_FIELDS):
                raise ValueError(
                    "Human verification must contain the independent dimensions "
                    "and evidence fields."
                )
            for field in VERIFICATION_RULE_ID_FIELDS:
                rule_ids = row[field]
                if not isinstance(rule_ids, list) or any(
                    not CANONICAL_RULE_ID_RE.fullmatch(str(rule_id)) for rule_id in rule_ids
                ):
                    raise ValueError("Human verification must use canonical K### rule IDs.")
        for row in citations:
            if any(field not in row for field in CITATION_REQUIRED_FIELDS):
                raise ValueError("Human citation validation is missing a required field.")
            for field in ("cited_rule_ids", "invalid_rule_ids"):
                rule_ids = row.get(field, [])
                if not isinstance(rule_ids, list) or any(
                    not CANONICAL_RULE_ID_RE.fullmatch(str(rule_id)) for rule_id in rule_ids
                ):
                    raise ValueError("Human citation validation must use canonical K### rule IDs.")
    incomplete_pairs = sorted(case_id for case_id, values in pairs.items() if values != conditions)
    if incomplete_pairs:
        raise ValueError(f"Calibration cases must include both conditions: {incomplete_pairs}.")
    if len(pairs) < int(calibration["minimum_paired_cases"]):
        raise ValueError("Stage 5 has fewer than the required paired calibration cases.")
    missing_tags = required_tags - observed_tags
    if missing_tags:
        raise ValueError(f"Calibration coverage is incomplete: {sorted(missing_tags)}.")
    if "bags" not in counts:
        raise ValueError("Stage 5 calibration must include a bag example.")
    return {
        "paired_case_count": len(pairs),
        "record_count": len(records),
        "counts_by_category": dict(sorted(counts.items())),
        "coverage_tags": sorted(observed_tags),
        "disjointness": {
            "calibration_source_split": calibration["source_split"],
            "final_explanation_split": calibration["final_explanation_split"],
            "disjoint": calibration["source_split"] != calibration["final_explanation_split"],
        },
    }


def calibration_metrics(
    human_records: Sequence[Mapping[str, Any]],
    model_records: Sequence[Mapping[str, Any]],
    *,
    alignments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calculate transparent extractor/verifier quality metrics against human annotations."""
    model_by_key = {
        (str(row["calibration_case_id"]), str(row["condition"])): row for row in model_records
    }
    human_claim_count = model_claim_count = strict_matched_claim_count = duplicate_claim_count = 0
    semantic_human_covered: set[tuple[str, str, str]] = set()
    semantic_qwen_covered: set[tuple[str, str, str]] = set()
    atomization_relations = 0
    alignment_by_key = {
        (str(row["calibration_case_id"]), str(row["condition"])): row
        for row in (alignments or calibration_alignment_records(human_records, model_records))
    }
    claim_id_matches = citation_total = citation_entailment_total = 0
    verification_total = Counter()
    verification_matches = Counter()
    verification_by_verdict = {
        field: {status: Counter() for status in CLAIM_STATUSES} for field in ENTAILMENT_FIELDS
    }
    verification_confusion = {
        field: {status: Counter() for status in CLAIM_STATUSES} for field in ENTAILMENT_FIELDS
    }
    citation_syntax_matches = citation_entailment_matches = 0
    binary_confusion = {
        field: Counter() for field in ("full_kb_entailment", "exact_trace_entailment")
    }
    structured_success = 0
    for human in human_records:
        key = (str(human["calibration_case_id"]), str(human["condition"]))
        model = model_by_key.get(key)
        if model is None:
            continue
        if model.get("status") == "complete":
            structured_success += 1
        human_claims = list(human["human_claims"])
        model_claims = list(model.get("claims", []))
        human_claim_count += len(human_claims)
        model_claim_count += len(model_claims)
        model_texts = [_normalized_claim(str(row.get("claim_text", ""))) for row in model_claims]
        alignment = alignment_by_key.get(key, {"pairs": []})
        strict_matched_claim_count += len(alignment["pairs"])
        for relation in alignment.get("proposition_relations", []):
            semantic_human_covered.add((key[0], key[1], relation["human_claim_id"]))
            semantic_qwen_covered.add((key[0], key[1], relation["qwen_claim_id"]))
            atomization_relations += relation["relation"] == "atomized_or_specificity_variant"
        duplicate_claim_count += len(model_texts) - len(set(model_texts))
        expected_ids = [f"C{index}" for index in range(1, len(model_claims) + 1)]
        if [row.get("claim_id") for row in model_claims] == expected_ids:
            claim_id_matches += 1
        human_verification = {row["claim_id"]: row for row in human["human_verification"]}
        raw_entailment = model.get("verification", model.get("entailment", []))
        if isinstance(raw_entailment, Mapping):
            raw_entailment = raw_entailment.get("claims", [])
        model_verification = {row["claim_id"]: row for row in raw_entailment}
        raw_citations = model.get("citation_validation", [])
        if isinstance(raw_citations, Mapping):
            raw_citations = raw_citations.get("claims", [])
        human_citations = {row["claim_id"]: row for row in human["human_citation_validation"]}
        model_citations = {row["claim_id"]: row for row in raw_citations}
        for pair in alignment["pairs"]:
            claim_id = pair["human_claim_id"]
            expected = human_verification[claim_id]
            observed = model_verification.get(pair["qwen_claim_id"], {})
            for field in ENTAILMENT_FIELDS:
                human_claim = next(claim for claim in human_claims if claim["claim_id"] == claim_id)
                if (
                    field == "common_reference_item_fact_support"
                    and not common_reference_eligibility(
                        human_claim, human.get("common_reference_item_facts", {})
                    )["eligible"]
                ):
                    continue
                verification_total[field] += 1
                verification_matches[field] += observed.get(field) == expected.get(field)
                expected_status = str(expected.get(field))
                observed_status = str(observed.get(field))
                verification_by_verdict[field][expected_status]["total"] += 1
                verification_by_verdict[field][expected_status]["matched"] += (
                    observed_status == expected_status
                )
                verification_confusion[field][expected_status][observed_status] += 1
                if field in binary_confusion:
                    binary_confusion[field][
                        (
                            "supported" if expected_status == "supported" else "not_supported",
                            "supported" if observed_status == "supported" else "not_supported",
                        )
                    ] += 1
            citation = human_citations[claim_id]
            if not citation["citation_present"]:
                continue
            citation_total += 1
            observed = model_citations.get(pair["qwen_claim_id"], {})
            citation_syntax_matches += observed.get("citation_present") is True and observed.get(
                "canonical_citation_format"
            ) == citation.get("canonical_citation_format")
            if citation.get("canonical_citation_format") is True:
                citation_entailment_total += 1
                citation_entailment_matches += observed.get(
                    "citation_entails_claim"
                ) == citation.get("citation_entails_claim")
    record_count = len(human_records)
    human_claim_denominator = human_claim_count or 1
    model_claim_denominator = model_claim_count or 1
    record_denominator = record_count or 1
    verifier_by_verdict_metrics = {
        field: {
            status: {
                "total": counts["total"],
                "matched": counts["matched"],
                "agreement": counts["matched"] / counts["total"] if counts["total"] else None,
            }
            for status, counts in values.items()
        }
        for field, values in verification_by_verdict.items()
    }
    confusion_metrics = {
        field: {status: dict(counts) for status, counts in values.items()}
        for field, values in verification_confusion.items()
    }
    binary_metrics = {
        field: {
            "supported_supported": counts[("supported", "supported")],
            "supported_not_supported": counts[("supported", "not_supported")],
            "not_supported_supported": counts[("not_supported", "supported")],
            "not_supported_not_supported": counts[("not_supported", "not_supported")],
            "agreement": (
                (counts[("supported", "supported")] + counts[("not_supported", "not_supported")])
                / sum(counts.values())
                if counts
                else None
            ),
        }
        for field, counts in binary_confusion.items()
    }
    common_reference_eligible = verification_total["common_reference_item_fact_support"]
    return {
        "extractor_claim_recall": len(semantic_human_covered) / human_claim_denominator,
        "extractor_claim_precision": len(semantic_qwen_covered) / model_claim_denominator,
        "extractor_duplicate_rate": duplicate_claim_count / model_claim_denominator,
        "extractor_claim_id_preservation_rate": claim_id_matches / record_denominator,
        "verifier_full_kb_accuracy": verification_matches["full_kb_entailment"]
        / (verification_total["full_kb_entailment"] or 1),
        "verifier_exact_trace_accuracy": verification_matches["exact_trace_entailment"]
        / (verification_total["exact_trace_entailment"] or 1),
        "verifier_common_reference_accuracy": (
            verification_matches["common_reference_item_fact_support"] / common_reference_eligible
            if common_reference_eligible
            else None
        ),
        "citation_validity_accuracy": citation_entailment_matches / citation_entailment_total
        if citation_entailment_total
        else 1.0,
        "citation_syntax_accuracy": citation_syntax_matches / citation_total
        if citation_total
        else 1.0,
        "citation_entailment_accuracy": citation_entailment_matches / citation_entailment_total
        if citation_entailment_total
        else 1.0,
        "structured_output_success_rate": structured_success / record_denominator,
        "aligned_claim_pair_count": float(strict_matched_claim_count),
        "semantic_covered_human_claim_count": float(len(semantic_human_covered)),
        "semantic_covered_qwen_claim_count": float(len(semantic_qwen_covered)),
        "atomization_disagreement_count": float(atomization_relations),
        "genuine_human_missed_claim_count": float(human_claim_count - len(semantic_human_covered)),
        "genuine_qwen_extra_claim_count": float(model_claim_count - len(semantic_qwen_covered)),
        "citation_scored_claim_count": float(citation_total),
        "citation_entailment_scored_claim_count": float(citation_entailment_total),
        "verifier_agreement_by_verdict": verifier_by_verdict_metrics,
        "verifier_confusion_matrix": confusion_metrics,
        "verifier_binary_supported_agreement": binary_metrics,
        "common_reference_eligible_claim_count": float(common_reference_eligible),
    }


def calibration_gates(metrics: Mapping[str, float], settings: Mapping[str, Any]) -> dict[str, bool]:
    criteria = _require_mapping(_require_mapping(settings, "calibration"), "pass_criteria")
    outcomes: dict[str, bool] = {}
    for metric, threshold in criteria.items():
        metric_name = metric.removesuffix("_minimum").removesuffix("_maximum")
        value = metrics.get(metric_name)
        if value is None:
            outcomes[metric_name] = False
        elif metric.endswith("_minimum"):
            outcomes[metric_name] = value >= float(threshold)
        else:
            outcomes[metric_name] = value <= float(threshold)
    outcomes["stage5_pass"] = all(outcomes.values())
    return outcomes
