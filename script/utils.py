"""Shared utilities for model labels, output rows, and checkpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import ALLOWED_POLARITIES


def get_class_labels(model: Any, probability_count: int) -> list[str]:
    """Use classifier labels when available; otherwise validate fallback order."""
    candidates = [
        getattr(getattr(model, "polarity_model", None), "classes_", None),
        getattr(
            getattr(getattr(model, "polarity_model", None), "model_head", None),
            "classes_",
            None,
        ),
    ]

    for candidate in candidates:
        if candidate is not None:
            labels = [str(value) for value in candidate]
            if len(labels) == probability_count:
                return labels

    if probability_count != len(ALLOWED_POLARITIES):
        raise RuntimeError(
            f"Expected {len(ALLOWED_POLARITIES)} polarity probabilities, "
            f"got {probability_count}."
        )
    return list(ALLOWED_POLARITIES)


def save_checkpoint(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Atomically save current output rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


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
    """Build one output record using the original dataset schema."""
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
