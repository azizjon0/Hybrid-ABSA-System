# Scripts

This folder contains the executable Hybrid ABSA inference pipeline.

## Files

- `run.py` — command-line entry point.
- `pipeline.py` — main processing logic.
- `llm.py` — OpenAI fallback, JSON parsing, validation, and retries.
- `model.py` — loads the pretrained SetFit ABSA model.
- `utils.py` — label handling, output row creation, and checkpoints.
- `config.py` — model IDs and default settings.

## Run

From inside the `scripts` folder:

```bash
export OPENAI_API_KEY="your-key"
python run.py \
  --input ../data/restaurant_reviews.csv \
  --output ../outputs/hybrid_predictions.csv \
  --threshold 0.60
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
python run.py `
  --input ../data/restaurant_reviews.csv `
  --output ../outputs/hybrid_predictions.csv `
  --threshold 0.60
```

To run only the local SetFit model:

```bash
python run.py --input ../data/restaurant_reviews.csv --disable-llm
```

The input CSV must contain a `text` column. A `stars` column is optional.
