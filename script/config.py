"""Default configuration for the Hybrid ABSA pipeline."""

ALLOWED_POLARITIES = ("conflict", "negative", "neutral", "positive")

ASPECT_MODEL_ID = (
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect"
)
POLARITY_MODEL_ID = (
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity"
)

DEFAULT_THRESHOLD = 0.60
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_OUTPUT = "hybrid_absa_predictions.csv"
DEFAULT_CHECKPOINT_EVERY = 10
