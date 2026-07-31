"""Load the pretrained SetFit ABSA model."""

from setfit import AbsaModel

from config import ASPECT_MODEL_ID, POLARITY_MODEL_ID


def load_absa_model() -> AbsaModel:
    """Load the pretrained restaurant ABSA aspect and polarity models."""
    return AbsaModel.from_pretrained(
        ASPECT_MODEL_ID,
        POLARITY_MODEL_ID,
    )


# Kept for compatibility with --model-module model.
model = load_absa_model()
