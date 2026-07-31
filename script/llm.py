"""OpenAI fallback logic for uncertain ABSA predictions."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import OpenAI

from config import ALLOWED_POLARITIES


def create_openai_client(disable_llm: bool) -> OpenAI | None:
    """Create an OpenAI client, or return None when LLM fallback is disabled."""
    if disable_llm:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it or run with --disable-llm."
        )
    return OpenAI(api_key=api_key)


def safe_json_loads(raw: str) -> dict[str, Any] | None:
    """Parse JSON, including responses containing text around the JSON object."""
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def validate_llm_items(
    parsed: dict[str, Any] | None,
    expected_ids: set[int],
) -> dict[int, dict[str, Any]]:
    """Validate LLM corrections and return them indexed by item_id."""
    if not parsed or not isinstance(parsed.get("items"), list):
        return {}

    validated: dict[int, dict[str, Any]] = {}
    for item in parsed["items"]:
        if not isinstance(item, dict):
            continue

        item_id = item.get("item_id")
        aspect = item.get("corrected_aspect")
        polarity = item.get("corrected_polarity")

        if (
            not isinstance(item_id, int)
            or item_id not in expected_ids
            or item_id in validated
        ):
            continue
        if not isinstance(aspect, str) or not aspect.strip():
            continue
        if polarity not in ALLOWED_POLARITIES:
            continue

        validated[item_id] = {
            "item_id": item_id,
            "corrected_aspect": aspect.strip(),
            "corrected_polarity": polarity,
            "change_needed": bool(item.get("change_needed", False)),
            "reason": str(item.get("reason", "")).strip(),
        }

    return validated


def ask_llm_for_uncertain_aspects(
    client: OpenAI,
    llm_model: str,
    text: str,
    stars: Any,
    uncertain_aspects: list[dict[str, Any]],
    max_attempts: int = 3,
) -> tuple[dict[int, dict[str, Any]], str | None]:
    """Ask the LLM to check only low-confidence ABSA predictions."""
    prompt = f"""
You are an ABSA correction expert for restaurant reviews.

Correct only the uncertain ABSA predictions below.
Each input item must have exactly one output item with the same item_id.
Do not return multiple aspects in one field.

Review:
{text}

Stars:
{stars}

Uncertain predictions:
{json.dumps(uncertain_aspects, indent=2, ensure_ascii=False)}

Return only valid JSON:
{{
  "items": [
    {{
      "item_id": 0,
      "corrected_aspect": "single aspect",
      "corrected_polarity": "positive",
      "change_needed": true,
      "reason": "short explanation"
    }}
  ]
}}

Allowed polarities: conflict, negative, neutral, positive.
If the original aspect or polarity is acceptable, keep it unchanged.
""".strip()

    expected_ids = {int(item["item_id"]) for item in uncertain_aspects}
    last_raw: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.responses.create(model=llm_model, input=prompt)
            last_raw = response.output_text
            validated = validate_llm_items(
                safe_json_loads(last_raw), expected_ids
            )
            if set(validated) == expected_ids:
                return validated, last_raw
        except Exception as exc:
            last_raw = f"LLM request failed: {type(exc).__name__}: {exc}"

        if attempt < max_attempts:
            time.sleep(2 ** (attempt - 1))

    return {}, last_raw
