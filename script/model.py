from setfit import AbsaModel
import pandas as pd
from tqdm.auto import tqdm

model = AbsaModel.from_pretrained(
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity",
)

client = OpenAI(api_key="API")


