"""Core Hybrid ABSA inference pipeline."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

from llm import ask_llm_for_uncertain_aspects
from utils import build_row, get_class_labels, save_checkpoint


def load_absa_model(module_name: str) -> Any:
    """Import a module that exposes the trained ABSA object as `model`."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import model module '{module_name}'. "
            "Run the script from the scripts directory or pass --model-module."
        ) from exc

    if not hasattr(module, "model"):
        raise RuntimeError(
            f"Module '{module_name}' does not expose a variable named 'model'."
        )
    return module.model


def load_input_data(input_path: Path) -> tuple[pd.DataFrame, list[str], list[Any]]:
    """Read and validate the input CSV."""
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
    return df, texts, stars


def process_empty_prediction(
    review_idx: int,
    text: str,
    star: Any,
    client: OpenAI | None,
    llm_model: str,
    threshold: float,
) -> dict[str, Any]:
    """Handle a review for which the local model found no aspect."""
    if client is None:
        return build_row(
            review_idx, text, star, "", "", "", 0.0, {}, False, None,
            None, None, None, None, "", "", threshold,
            "No local aspect; LLM disabled",
        )

    uncertain = [
        {
            "item_id": 0,
            "aspect": "review",
            "polarity": "neutral",
            "confidence": 0.0,
        }
    ]
    llm_items, raw = ask_llm_for_uncertain_aspects(
        client, llm_model, text, star, uncertain
    )
    item = llm_items.get(0)
    final_aspect = item["corrected_aspect"] if item else ""
    final_polarity = item["corrected_polarity"] if item else ""

    return build_row(
        review_idx, text, star, "", "", "", 0.0, {}, item is not None, raw,
        final_aspect or None, final_polarity or None,
        item.get("change_needed") if item else None,
        item.get("reason") if item else None,
        final_aspect, final_polarity, threshold,
        "No local aspect; sent to LLM" if item else "No local aspect; LLM failed",
    )


def process_local_predictions(
    review_idx: int,
    text: str,
    star: Any,
    prediction: list[dict[str, Any]],
    model: Any,
    client: OpenAI | None,
    llm_model: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Score local predictions and route uncertain items to the LLM."""
    polarity_inputs = [f"{item['span']} {text}" for item in prediction]
    probability_rows = model.polarity_model.predict_proba(polarity_inputs)

    local_predictions: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []

    for aspect_idx, (item, probability) in enumerate(
        zip(prediction, probability_rows)
    ):
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

        if confidence < threshold:
            uncertain.append(
                {
                    "item_id": aspect_idx,
                    "aspect": local["aspect"],
                    "polarity": local["polarity"],
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        key: round(value, 4)
                        for key, value in probability_map.items()
                    },
                }
            )

    llm_items: dict[int, dict[str, Any]] = {}
    raw: str | None = None
    if uncertain and client is not None:
        llm_items, raw = ask_llm_for_uncertain_aspects(
            client, llm_model, text, star, uncertain
        )

    rows: list[dict[str, Any]] = []
    for local in local_predictions:
        item = llm_items.get(local["local_item_id"])
        final_aspect = item["corrected_aspect"] if item else local["aspect"]
        final_polarity = item["corrected_polarity"] if item else local["polarity"]

        if item:
            comment = "Low confidence; corrected by LLM"
        elif local["confidence"] < threshold:
            comment = "Low confidence; LLM unavailable or invalid"
        else:
            comment = ""

        rows.append(
            build_row(
                review_idx,
                text,
                star,
                local["aspect"],
                local["polarity"],
                local["proba_predicted_label"],
                local["confidence"],
                local["probabilities"],
                item is not None,
                raw if item is not None else None,
                item.get("corrected_aspect") if item else None,
                item.get("corrected_polarity") if item else None,
                item.get("change_needed") if item else None,
                item.get("reason") if item else None,
                final_aspect,
                final_polarity,
                threshold,
                comment,
            )
        )

    return rows


def run_pipeline(
    input_path: Path,
    output_path: Path,
    model_module: str,
    threshold: float,
    llm_model: str,
    checkpoint_every: int,
    client: OpenAI | None,
) -> pd.DataFrame:
    """Run local ABSA inference, optional LLM correction, and CSV export."""
    if not 0 <= threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")

    df, texts, stars = load_input_data(input_path)
    model = load_absa_model(model_module)

    print(f"Loaded {len(df)} reviews")
    print("Running local ABSA model...")
    predictions = model.predict(texts)

    rows: list[dict[str, Any]] = []

    for review_idx, (text, prediction) in enumerate(
        tqdm(zip(texts, predictions), total=len(texts), desc="Reviews")
    ):
        star = stars[review_idx]

        if not prediction:
            rows.append(
                process_empty_prediction(
                    review_idx, text, star, client, llm_model, threshold
                )
            )
        else:
            rows.extend(
                process_local_predictions(
                    review_idx,
                    text,
                    star,
                    prediction,
                    model,
                    client,
                    llm_model,
                    threshold,
                )
            )

        if checkpoint_every > 0 and (review_idx + 1) % checkpoint_every == 0:
            save_checkpoint(rows, output_path)

    result = pd.DataFrame(rows)
    save_checkpoint(rows, output_path)
    return result
