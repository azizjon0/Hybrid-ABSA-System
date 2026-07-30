"""Hybrid ABSA inference pipeline.

Expected project setup:
- A Python module named ``model.py`` that exposes a trained ABSA object as ``model``.
- Environment variable ``OPENAI_API_KEY`` when LLM fallback is enabled.

Example:
    export OPENAI_API_KEY="..."
    python hybrid_absa_pipeline_fixed.py \
        --input restaurant_reviews_30.csv \
        --output hybrid_predictions.csv \
        --model-module model \
        --threshold 0.60
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

ALLOWED_POLARITIES = ("conflict", "negative", "neutral", "positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid ABSA pipeline")
    parser.add_argument("--input", required=True, help="Input CSV containing a text column")
    parser.add_argument("--output", default="hybrid_absa_predictions.csv")
    parser.add_argument("--model-module", default="model", help="Module exposing a variable named model")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def load_absa_model(module_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import model module '{module_name}'. "
            "Run the script from the project root or pass --model-module."
        ) from exc

    if not hasattr(module, "model"):
        raise RuntimeError(f"Module '{module_name}' does not expose a variable named 'model'.")
    return module.model


def create_openai_client(disable_llm: bool) -> OpenAI | None:
    if disable_llm:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it or run with --disable-llm."
        )
    return OpenAI(api_key=api_key)


def safe_json_loads(raw: str) -> dict[str, Any] | None:
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
    if not parsed or not isinstance(parsed.get("items"), list):
        return {}

    validated: dict[int, dict[str, Any]] = {}
    for item in parsed["items"]:
        if not isinstance(item, dict):
            continue

        item_id = item.get("item_id")
        aspect = item.get("corrected_aspect")
        polarity = item.get("corrected_polarity")

        if not isinstance(item_id, int) or item_id not in expected_ids or item_id in validated:
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
            validated = validate_llm_items(safe_json_loads(last_raw), expected_ids)
            if set(validated) == expected_ids:
                return validated, last_raw
        except Exception as exc:
            last_raw = f"LLM request failed: {type(exc).__name__}: {exc}"

        if attempt < max_attempts:
            time.sleep(2 ** (attempt - 1))

    return {}, last_raw


def get_class_labels(model: Any, probability_count: int) -> list[str]:
    """Prefer labels stored by the classifier; otherwise validate the fallback order."""
    candidates = [
        getattr(getattr(model, "polarity_model", None), "classes_", None),
        getattr(getattr(getattr(model, "polarity_model", None), "model_head", None), "classes_", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            labels = [str(value) for value in candidate]
            if len(labels) == probability_count:
                return labels

    if probability_count != len(ALLOWED_POLARITIES):
        raise RuntimeError(
            f"Expected {len(ALLOWED_POLARITIES)} polarity probabilities, got {probability_count}."
        )
    return list(ALLOWED_POLARITIES)


def save_checkpoint(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column")

    df = df[df["text"].notna()].copy()
    df["text"] = df["text"].astype(str)
    texts = df["text"].tolist()
    stars = df["stars"].tolist() if "stars" in df.columns else [None] * len(df)

    model = load_absa_model(args.model_module)
    client = create_openai_client(args.disable_llm)

    print(f"Loaded {len(df)} reviews")
    print("Running local ABSA model...")
    predictions = model.predict(texts)

    rows: list[dict[str, Any]] = []

    for review_idx, (text, prediction) in enumerate(
        tqdm(zip(texts, predictions), total=len(texts), desc="Reviews")
    ):
        star = stars[review_idx]
        local_predictions: list[dict[str, Any]] = []
        uncertain: list[dict[str, Any]] = []

        if not prediction:
            if client is None:
                rows.append(build_row(
                    review_idx, text, star, "", "", "", 0.0, {}, False, None,
                    None, None, None, None, "", "", args.threshold,
                    "No local aspect; LLM disabled",
                ))
            else:
                uncertain = [{"item_id": 0, "aspect": "review", "polarity": "neutral", "confidence": 0.0}]
                llm_items, raw = ask_llm_for_uncertain_aspects(
                    client, args.llm_model, text, star, uncertain
                )
                item = llm_items.get(0)
                final_aspect = item["corrected_aspect"] if item else ""
                final_polarity = item["corrected_polarity"] if item else ""
                rows.append(build_row(
                    review_idx, text, star, "", "", "", 0.0, {}, True, raw,
                    final_aspect or None, final_polarity or None,
                    item.get("change_needed") if item else None,
                    item.get("reason") if item else None,
                    final_aspect, final_polarity, args.threshold,
                    "No local aspect; sent to LLM" if item else "No local aspect; LLM failed",
                ))
        else:
            polarity_inputs = [f"{item['span']} {text}" for item in prediction]
            probabilities = model.polarity_model.predict_proba(polarity_inputs)

            for aspect_idx, (item, probability) in enumerate(zip(prediction, probabilities)):
                prob_list = [float(value) for value in probability.tolist()]
                class_labels = get_class_labels(model, len(prob_list))
                confidence = max(prob_list)
                predicted_label = class_labels[prob_list.index(confidence)]
                probability_map = dict(zip(class_labels, prob_list))

                local = {
                    "local_item_id": aspect_idx,
                    "aspect": str(item["span"]),
                    "polarity": str(item["polarity"]),
                    "proba_predicted_label": predicted_label,
                    "confidence": confidence,
                    "probabilities": probability_map,
                }
                local_predictions.append(local)

                if confidence < args.threshold:
                    uncertain.append({
                        "item_id": aspect_idx,
                        "aspect": local["aspect"],
                        "polarity": local["polarity"],
                        "confidence": round(confidence, 4),
                        "probabilities": {k: round(v, 4) for k, v in probability_map.items()},
                    })

            llm_items: dict[int, dict[str, Any]] = {}
            raw: str | None = None
            if uncertain and client is not None:
                llm_items, raw = ask_llm_for_uncertain_aspects(
                    client, args.llm_model, text, star, uncertain
                )

            for local in local_predictions:
                item = llm_items.get(local["local_item_id"])
                final_aspect = item["corrected_aspect"] if item else local["aspect"]
                final_polarity = item["corrected_polarity"] if item else local["polarity"]
                probabilities = local["probabilities"]

                rows.append(build_row(
                    review_idx,
                    text,
                    star,
                    local["aspect"],
                    local["polarity"],
                    local["proba_predicted_label"],
                    local["confidence"],
                    probabilities,
                    item is not None,
                    raw if item is not None else None,
                    item.get("corrected_aspect") if item else None,
                    item.get("corrected_polarity") if item else None,
                    item.get("change_needed") if item else None,
                    item.get("reason") if item else None,
                    final_aspect,
                    final_polarity,
                    args.threshold,
                    "Low confidence; corrected by LLM" if item else (
                        "Low confidence; LLM unavailable or invalid" if local["confidence"] < args.threshold else ""
                    ),
                ))

        if args.checkpoint_every > 0 and (review_idx + 1) % args.checkpoint_every == 0:
            save_checkpoint(rows, output_path)

    result = pd.DataFrame(rows)
    save_checkpoint(rows, output_path)
    return result


def build_row(
    review_id: int,
    text: str,
    stars: Any,
    aspect: str,
    polarity: str,
    predicted_label: str,
    confidence: float,
    probabilities: dict[str, float],
    llm_called: bool,
    llm_raw_response: str | None,
    llm_aspect: str | None,
    llm_polarity: str | None,
    llm_change_needed: bool | None,
    llm_reason: str | None,
    final_aspect: str,
    final_polarity: str,
    threshold: float,
    comment: str,
) -> dict[str, Any]:
    return {
        "id": review_id,
        "text": text,
        "stars": stars,
        "aspect": aspect,
        "polarity": polarity,
        "proba_predicted_label": predicted_label,
        "confidence": confidence,
        "proba_conflict": probabilities.get("conflict"),
        "proba_negative": probabilities.get("negative"),
        "proba_neutral": probabilities.get("neutral"),
        "proba_positive": probabilities.get("positive"),
        "llm_called": llm_called,
        "llm_raw_response": llm_raw_response,
        "llm_aspect": llm_aspect,
        "llm_polarity": llm_polarity,
        "llm_change_needed": llm_change_needed,
        "llm_reason": llm_reason,
        "final_aspect": final_aspect,
        "final_polarity": final_polarity,
        "review_status": "unreviewed",
        "prediction_was_correct": None,
        "corrected_aspect": final_aspect,
        "corrected_polarity": final_polarity,
        "usable_for_training": True,
        "comment": comment,
        "threshold": threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    result = run_pipeline(args)
    print("\nFinished")
    print(f"Output rows: {len(result)}")
    print(f"LLM-corrected rows: {int(result['llm_called'].sum()) if not result.empty else 0}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()