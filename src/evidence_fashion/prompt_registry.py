"""Canonical prompt-registry loading, validation, rendering, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from string import Formatter
from typing import Any

import yaml

REQUIRED_ROLE_FIELDS = frozenset(
    {
        "system_prompt",
        "user_template",
        "template_variables",
        "permitted_evidence",
        "prohibited_inference",
        "output_schema",
        "token_limit",
        "temperature",
        "seed",
        "retry",
    }
)
REQUIRED_RETRY_FIELDS = frozenset(
    {"max_attempts", "repair_attempts", "retry_instruction", "terminal_failure"}
)
REQUIRED_ROLES = frozenset(
    {
        "no_rag_explanation",
        "rule_rag_explanation",
        "claim_extraction",
        "claim_verification",
        "citation_validation",
        "blind_judge",
    }
)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text_sha256(encoded)


def _template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name is not None:
            if not field_name or any(marker in field_name for marker in ".["):
                raise ValueError("Prompt templates permit only simple named placeholders.")
            fields.add(field_name)
    return fields


def validate_prompt_registry(registry: Mapping[str, Any]) -> None:
    if registry.get("schema_version") != 1:
        raise ValueError("Prompt registry must use schema_version 1.")
    roles = registry.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != REQUIRED_ROLES:
        raise ValueError(f"Prompt registry roles must be exactly {sorted(REQUIRED_ROLES)}.")
    for role_name, role in roles.items():
        if not isinstance(role, Mapping):
            raise ValueError(f"Prompt role {role_name} must be a mapping.")
        missing = REQUIRED_ROLE_FIELDS - set(role)
        if missing:
            raise ValueError(f"Prompt role {role_name} is missing fields: {sorted(missing)}.")
        if not isinstance(role["system_prompt"], str) or not role["system_prompt"].strip():
            raise ValueError(f"Prompt role {role_name} needs a non-empty system prompt.")
        if not isinstance(role["user_template"], str) or not role["user_template"].strip():
            raise ValueError(f"Prompt role {role_name} needs a non-empty user template.")
        variables = role["template_variables"]
        if not isinstance(variables, list) or not all(isinstance(item, str) for item in variables):
            raise ValueError(f"Prompt role {role_name} has invalid template_variables.")
        if len(set(variables)) != len(variables) or set(variables) != _template_fields(
            role["user_template"]
        ):
            raise ValueError(
                f"Prompt role {role_name} template variables do not match its template."
            )
        if not isinstance(role["permitted_evidence"], list) or not role["permitted_evidence"]:
            raise ValueError(f"Prompt role {role_name} must declare permitted evidence.")
        if not isinstance(role["prohibited_inference"], list) or not role["prohibited_inference"]:
            raise ValueError(f"Prompt role {role_name} must declare prohibited inference.")
        if not isinstance(role["output_schema"], Mapping) or not role["output_schema"]:
            raise ValueError(f"Prompt role {role_name} must declare an output schema.")
        if not isinstance(role["token_limit"], int) or role["token_limit"] <= 0:
            raise ValueError(f"Prompt role {role_name} has an invalid token limit.")
        if not isinstance(role["temperature"], (int, float)) or role["temperature"] < 0:
            raise ValueError(f"Prompt role {role_name} has an invalid temperature.")
        if not isinstance(role["seed"], int):
            raise ValueError(f"Prompt role {role_name} has an invalid seed.")
        retry = role["retry"]
        if not isinstance(retry, Mapping) or REQUIRED_RETRY_FIELDS - set(retry):
            raise ValueError(f"Prompt role {role_name} has an incomplete retry contract.")
        retry_counts_valid = all(
            isinstance(retry[key], int) and retry[key] >= 0
            for key in ("max_attempts", "repair_attempts")
        )
        if not retry_counts_valid:
            raise ValueError(f"Prompt role {role_name} has invalid retry counts.")
        if (
            not isinstance(retry["retry_instruction"], str)
            or not retry["retry_instruction"].strip()
        ):
            raise ValueError(f"Prompt role {role_name} has no retry instruction.")
    if roles["blind_judge"].get("enabled") is not False:
        raise ValueError("The optional blind judge must be disabled until separately approved.")


def load_prompt_registry(path: Path) -> dict[str, Any]:
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("Prompt registry must contain a YAML mapping.")
    validate_prompt_registry(registry)
    return registry


def render_prompt(
    registry: Mapping[str, Any], role_name: str, variables: Mapping[str, Any]
) -> dict[str, Any]:
    """Render one exact system/user pair and preserve hashes for a completed manifest."""
    validate_prompt_registry(registry)
    try:
        role = registry["roles"][role_name]
    except KeyError as error:
        raise ValueError(f"Unknown prompt role: {role_name}.") from error
    expected = set(role["template_variables"])
    observed = set(variables)
    if observed != expected:
        raise ValueError(
            f"Prompt role {role_name} variables differ; expected {sorted(expected)}, "
            f"got {sorted(observed)}."
        )
    rendered_variables = {key: str(value) for key, value in variables.items()}
    user_prompt = str(role["user_template"]).format(**rendered_variables)
    system_prompt = str(role["system_prompt"])
    combined = f"{system_prompt}\n\n--- USER ---\n{user_prompt}"
    return {
        "role": role_name,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "system_prompt_sha256": text_sha256(system_prompt),
        "user_prompt_sha256": text_sha256(user_prompt),
        "rendered_prompt_sha256": text_sha256(combined),
        "role_contract_sha256": canonical_sha256(dict(role)),
    }


def prompt_manifest_fields(rendered: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact fields completed stage manifests must retain for each model call."""
    required = {
        "role",
        "system_prompt",
        "user_prompt",
        "system_prompt_sha256",
        "user_prompt_sha256",
        "rendered_prompt_sha256",
        "role_contract_sha256",
    }
    if required - set(rendered):
        raise ValueError("Rendered prompt is missing required provenance fields.")
    return {key: str(rendered[key]) for key in required}
