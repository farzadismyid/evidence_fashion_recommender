"""Stage 5 prompt construction and local deterministic Ollama inference."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .grounding_contracts import require_trace_applicability, validate_generated_explanation


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def common_context(case: Mapping[str, Any]) -> dict[str, str]:
    """Construct common input A from only the three approved fields."""
    return {
        "user_request": str(case["request"]),
        "query_item_minimal_name": str(case["query_item_minimal_name"]),
        "locked_item_minimal_name": str(case["locked_candidate_minimal_name"]),
    }


def build_no_rag_prompt(case: Mapping[str, Any], word_limit: int | None = None) -> str:
    """Render the A-only baseline, optionally under the shared length instruction."""
    context = common_context(case)
    instruction = (
        "Write a 55–75 word explanation, aiming for about 65 words in 2–3 sentences. Explain "
        "why the exact recommended item works with the outfit using only supplied context; do "
        "not pad it with unsupported details. Repeat the recommended item verbatim and do not "
        "substitute another recommendation."
    )
    if word_limit is not None:
        instruction = (
            f"Write a recommendation explanation within the 55–75 word shared contract. "
            f"{instruction}"
        )
    return (
        f"{instruction}\n\n"
        f"User request: {context['user_request']}\n"
        f"Query item: {context['query_item_minimal_name']}\n"
        f"Recommended item: {context['locked_item_minimal_name']}\n"
        f"Required first-sentence wording: include {context['locked_item_minimal_name']} exactly."
    )


def _selected_rules(trace: Mapping[str, Any], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    require_trace_applicability(trace)
    rules = [dict(rule) for rule in trace["rules"]]
    expected = int(settings["rule_count"])
    if len(rules) != expected:
        raise ValueError("Explanation rule count must equal the complete stored reranking trace.")
    if settings["evidence_ordering"] == "weighted_score_descending":
        rules.sort(key=lambda rule: (-float(rule["weighted_contribution"]), rule["rule_id"]))
    return rules


def _rule_projection(rule: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {"rule_id": rule["rule_id"], "rule_text": rule["rule_text"]}
    if settings["include_rule_scores"]:
        projected["weighted_contribution"] = rule["weighted_contribution"]
    if settings["include_reliability_labels"]:
        projected["reliability_label"] = rule["reliability_label"]
    return projected


def build_rule_rag_prompt(
    case: Mapping[str, Any], settings: Mapping[str, Any]
) -> tuple[str, list[str]]:
    """Build the evidence prompt from a projection of the exact stored trace B."""
    context = common_context(case)
    selected = _selected_rules(case["evidence_trace"], settings)
    projected = [_rule_projection(rule, settings) for rule in selected]
    if settings["evidence_format"] == "compact_json":
        evidence = json.dumps(projected, sort_keys=True, separators=(",", ":"))
    else:
        evidence = "\n".join(
            " | ".join(f"{key}={value}" for key, value in rule.items()) for rule in projected
        )
    citation_instruction = (
        "Cite supporting rules using separate square-bracket K IDs such as [K012] [K099]."
        if settings["citation_mode"] == "required"
        else "Rule-ID citations are optional."
    )
    grounding = (
        "Use only the item names and supplied rules; omit any claim the evidence does not support."
        if settings["grounding_prompt_variant"] == "concise"
        else (
            "Use only the supplied item names and rules. Connect each factual compatibility "
            "claim to evidence, distinguish general styling guidance from item facts, and omit "
            "unsupported visual, material, brand, fit, or occasion details."
        )
    )
    prompt = (
        "Write an evidence-grounded recommendation explanation in 55–75 words, aiming for "
        f"about 65 words in 2–3 sentences. {grounding} {citation_instruction} Develop why the "
        "recommendation works rather than merely restating a rule, but do not add unsupported "
        "detail. Name the exact recommended item verbatim and do not substitute another "
        "recommendation.\n\n"
        f"User request: {context['user_request']}\n"
        f"Query item: {context['query_item_minimal_name']}\n"
        f"Recommended item: {context['locked_item_minimal_name']}\n"
        f"Required first-sentence wording: include {context['locked_item_minimal_name']} exactly.\n"
        f"Evidence rules:\n{evidence}"
    )
    return prompt, [str(rule["rule_id"]) for rule in selected]


def word_count(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_seconds: float
    prompt_eval_count: int
    eval_count: int
    total_duration_ns: int


class OllamaClient:
    def __init__(self, defaults: Mapping[str, Any], endpoint: str = "http://127.0.0.1:11434"):
        self.defaults = defaults
        self.endpoint = endpoint.rstrip("/")

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        json_format: Mapping[str, Any] | str | None = None,
        token_limit: int | None = None,
        timeout_seconds: float | None = None,
    ) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {
                "num_ctx": self.defaults["context_length"],
                "temperature": self.defaults["temperature"],
                "top_p": self.defaults["top_p"],
                "top_k": self.defaults["top_k"],
                "seed": self.defaults["seed"],
                "num_predict": token_limit or self.defaults["token_limit"],
            },
        }
        if system_prompt is not None:
            payload["system"] = system_prompt
        if json_format is not None:
            payload["format"] = json_format
        request = urllib.request.Request(
            f"{self.endpoint}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(
            request,
            timeout=(
                float(timeout_seconds)
                if timeout_seconds is not None
                else float(self.defaults["timeout_seconds"])
            ),
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
        return GenerationResult(
            text=str(result["response"]).strip(),
            latency_seconds=time.perf_counter() - started,
            prompt_eval_count=int(result.get("prompt_eval_count", 0)),
            eval_count=int(result.get("eval_count", 0)),
            total_duration_ns=int(result.get("total_duration", 0)),
        )

    def unload(self, model: str) -> None:
        """Unload one completed model batch before the next model is started."""
        payload = {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        request = urllib.request.Request(
            f"{self.endpoint}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            # Ollama may complete the unload but fail to close the acknowledgement socket.
            # A short bounded wait preserves sequential release without discarding a batch.
            with urllib.request.urlopen(
                request, timeout=min(10.0, float(self.defaults["timeout_seconds"]))
            ) as response:
                response.read()
        except TimeoutError:
            return

    def generate_json(
        self,
        model: str,
        prompt: str,
        schema: Mapping[str, Any],
        *,
        retries: int,
        system_prompt: str | None = None,
        repair_instruction: str | None = None,
    ) -> tuple[dict[str, Any], GenerationResult, int]:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            suffix = (
                ""
                if attempt == 0
                else "\n"
                + (
                    repair_instruction
                    or "Return only valid JSON matching the schema. Extract each independent "
                    "claim exactly once; never repeat, duplicate, or paraphrase a claim already "
                    "listed."
                )
            )
            retry_token_limit = self.defaults["structured_token_limit"] * (2**attempt)
            try:
                result = self.generate(
                    model,
                    prompt + suffix,
                    system_prompt=system_prompt,
                    json_format=schema,
                    token_limit=retry_token_limit,
                    timeout_seconds=self.defaults["timeout_seconds"] * (2**attempt),
                )
            except TimeoutError as error:
                last_error = error
                continue
            try:
                return json.loads(result.text), result, attempt
            except (json.JSONDecodeError, TypeError) as error:
                last_error = error
        raise ValueError("Local model did not return valid structured JSON.") from last_error


def generate_explanation_with_contract_retries(
    client: OllamaClient,
    *,
    model: str,
    prompt: str,
    locked_item_name: str,
    target_category: str,
    trace_rule_ids: Sequence[str] = (),
    citations_required: bool = False,
    retries: int = 2,
    system_prompt: str | None = None,
    token_limit: int | None = None,
) -> tuple[GenerationResult | None, int, list[str]]:
    """Generate only a response that preserves the locked item and citation contract."""
    errors: list[str] = []
    for attempt in range(retries + 1):
        repair = (
            ""
            if attempt == 0
            else (
                "\n\nYour prior response violated the locked-recommendation contract. Name the "
                "exact recommended item verbatim; do not substitute an alternative. Use only "
                "separate [K###] citations from supplied rule evidence."
            )
        )
        result = client.generate(
            model,
            prompt + repair,
            system_prompt=system_prompt,
            token_limit=token_limit,
        )
        try:
            validate_generated_explanation(
                result.text,
                locked_item_name=locked_item_name,
                target_category=target_category,
                trace_rule_ids=trace_rule_ids,
                citations_required=citations_required,
            )
        except ValueError as error:
            errors.append(str(error))
            continue
        return result, attempt, errors
    return None, retries, errors


ASSESSMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "first": {"$ref": "#/$defs/condition"},
        "second": {"$ref": "#/$defs/condition"},
        "preference": {"type": "string", "enum": ["first", "second", "tie"]},
        "preference_reason": {"type": "string"},
    },
    "required": ["first", "second", "preference", "preference_reason"],
    "$defs": {
        "claim": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "support_status": {
                    "type": "string",
                    "enum": ["supported", "unsupported", "contradicted", "not_verifiable"],
                },
                "support_source": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["query_or_locked_item", "rule_evidence", "none"],
                    },
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "evidence_rule_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["claim", "support_status", "support_source", "evidence_rule_ids"],
        },
        "condition": {
            "type": "object",
            "properties": {
                "claims": {"type": "array", "items": {"$ref": "#/$defs/claim"}},
                "general_quality": {"type": "integer", "minimum": 1, "maximum": 5},
                "clarity": {"type": "integer", "minimum": 1, "maximum": 5},
                "specificity": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["claims", "general_quality", "clarity", "specificity"],
        },
    },
}


def build_assessment_prompt(
    case: Mapping[str, Any],
    first_text: str,
    second_text: str,
    first_condition: str,
    second_condition: str,
) -> str:
    """Create a position-randomized paired extraction/verification/judging prompt."""
    context = common_context(case)
    trace = case["evidence_trace"]
    allowed = {
        "common_context": context,
        "exact_rule_trace_for_rule_condition_only": trace,
        "first_allowed_sources": (
            "common_context_only" if first_condition == "no_rag" else "common_context_and_trace"
        ),
        "second_allowed_sources": (
            "common_context_only" if second_condition == "no_rag" else "common_context_and_trace"
        ),
    }
    return (
        "Act as a strict fashion-explanation evaluator. Extract every atomic factual or "
        "advisory claim "
        "from each explanation, label each claim against only that explanation's allowed sources, "
        "and assign every applicable support source; the source array may contain both common "
        "context and rule evidence when both support the claim. Use [none] only when no supplied "
        "source supports the claim, "
        "and score quality, clarity, and specificity from 1 to 5. A rule citation supports a claim "
        "only when the cited rule ID exists and its text entails the claim. Do not reward fluency "
        "for unsupported details. Compare the two explanations without guessing their experimental "
        "condition.\n\n"
        f"Evaluation evidence: {json.dumps(allowed, sort_keys=True)}\n\n"
        f"First explanation:\n{first_text}\n\nSecond explanation:\n{second_text}"
    )
