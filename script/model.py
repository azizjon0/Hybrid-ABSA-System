import os

from openai import OpenAI
from setfit import AbsaModel


ASPECT_MODEL_ID = (
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect"
)
POLARITY_MODEL_ID = (
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity"
)


def load_absa_model() -> AbsaModel:
    """Load the pretrained restaurant ABSA model."""
    return AbsaModel.from_pretrained(
        ASPECT_MODEL_ID,
        POLARITY_MODEL_ID,
    )


def create_openai_client() -> OpenAI:
    """Create an OpenAI client using OPENAI_API_KEY."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Configure it before running the pipeline."
        )

    return OpenAI()


model = load_absa_model()

