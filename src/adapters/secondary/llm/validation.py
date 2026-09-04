from __future__ import annotations

import json
from typing import Any, Optional


def _try_parse_and_validate(
    raw_text: str, schema: dict[str, Any]
) -> tuple[Optional[dict[str, Any]], Optional[Exception]]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, exc
    try:
        _validate_against_schema(parsed, schema)
    except ValueError as exc:
        return None, exc
    return parsed, None


def _validate_against_schema(instance: object, schema: dict[str, Any]) -> None:
    """Hand-rolled structural check over the JSON Schema subset used by
    src.features.query.prompts.JSON_SCHEMA (object/array/string/boolean,
    properties, required, additionalProperties, items) — not a
    general-purpose validator. Raises ValueError with a human-readable
    reason on mismatch."""
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(instance, dict):
            raise ValueError(f"expected object, got {type(instance).__name__}")
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                raise ValueError(f"missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            extra = set(instance) - set(properties)
            if extra:
                raise ValueError(f"unexpected properties: {sorted(extra)}")
        for key, subschema in properties.items():
            if key in instance:
                _validate_against_schema(instance[key], subschema)
    elif schema_type == "array":
        if not isinstance(instance, list):
            raise ValueError(f"expected array, got {type(instance).__name__}")
        item_schema = schema.get("items")
        if item_schema:
            for item in instance:
                _validate_against_schema(item, item_schema)
    elif schema_type == "string":
        if not isinstance(instance, str):
            raise ValueError(f"expected string, got {type(instance).__name__}")
    elif schema_type == "boolean":
        if not isinstance(instance, bool):
            raise ValueError(f"expected boolean, got {type(instance).__name__}")


def _build_repair_system_prompt(original_system_prompt: str, previous_response: str, error: Exception) -> str:
    return (
        f"{original_system_prompt}\n\n"
        f"Your previous response was invalid and could not be used: {error}\n"
        f"Your previous response was:\n{previous_response}\n\n"
        "Return ONLY the JSON object matching the schema above, with no prose, "
        "explanation, or markdown code fences."
    )
