import pandas as pd
import json
from datetime import datetime
from openai import OpenAI
from tqdm.auto import tqdm

# =========================
# SETTINGS
# =========================



CONFIDENCE_THRESHOLD = 0.60
LLM_MODEL = "gpt-4.1-mini"

DATA_PATH = "/kaggle/working/restaurant_reviews_30.csv"
OUTPUT_PATH = "hybrid_absa_predictions_with_llm_corrected.csv"

labels = ["conflict", "negative", "neutral", "positive"]

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

texts = df["text"].tolist()
stars = df["stars"].tolist() if "stars" in df.columns else [None] * len(df)

print(f"Loaded reviews: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("=" * 100)

# =========================
# RUN ABSA MODEL
# =========================

print("Running ABSA model...")
preds = model.predict(texts)
print("ABSA predictions finished.")
print("=" * 100)

# =========================
# LLM FUNCTION
# =========================

def safe_json_loads(raw):
    try:
        return json.loads(raw)
    except Exception:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            return None


def ask_llm_for_uncertain_aspects(text, stars, uncertain_aspects):
    prompt = f"""
You are an ABSA correction expert for restaurant reviews.

Your job:
Correct ONLY the uncertain ABSA predictions listed below.

Do NOT create new aspects unless the given aspect is clearly wrong.
Do NOT return multiple aspects in one field.
Each input item must have exactly one output item with the same item_id.

Review:
{text}

Stars:
{stars}

Uncertain ABSA predictions:
{json.dumps(uncertain_aspects, indent=2, ensure_ascii=False)}

Return ONLY valid JSON in this exact format:
{{
  "items": [
    {{
      "item_id": 0,
      "corrected_aspect": "single aspect here",
      "corrected_polarity": "positive/negative/neutral/conflict",
      "change_needed": true,
      "reason": "short explanation"
    }}
  ]
}}

Rules:
- polarity must be only one of: positive, negative, neutral, conflict
- corrected_aspect must be one short aspect, not a list
- if original aspect is acceptable, keep the same aspect
- if original polarity is acceptable, keep the same polarity
"""

    response = client.responses.create(
        model=LLM_MODEL,
        input=prompt
    )

    raw = response.output_text
    parsed = safe_json_loads(raw)

    return parsed, raw

# =========================
# MAIN LOOP
# =========================

rows = []
total = len(texts)

for review_idx, (text, pred) in enumerate(tqdm(zip(texts, preds), total=total)):

    star = stars[review_idx]

    print("\n" + "=" * 100)
    print(f"REVIEW {review_idx + 1}/{total}")
    print("-" * 100)
    print(f"Text: {text[:300]}")
    print(f"Stars: {star}")

    review_predictions = []
    uncertain_aspects = []

    # =========================
    # CASE: NO ABSA PREDICTION
    # =========================

    if len(pred) == 0:
        print("No ABSA aspects found.")
        print("Sending whole review to LLM as one uncertain item.")

        uncertain_aspects = [{
            "item_id": 0,
            "aspect": "",
            "polarity": "",
            "confidence": 0.0,
            "proba_conflict": None,
            "proba_negative": None,
            "proba_neutral": None,
            "proba_positive": None
        }]

        parsed, raw = ask_llm_for_uncertain_aspects(
            text=text,
            stars=star,
            uncertain_aspects=uncertain_aspects
        )

        llm_items = {}

        if parsed and "items" in parsed:
            llm_items = {item["item_id"]: item for item in parsed["items"]}

        llm_item = llm_items.get(0, {})

        rows.append({
            "id": review_idx,
            "text": text,
            "stars": star,

            "aspect": "",
            "polarity": "",
            "proba_predicted_label": "",
            "confidence": 0.0,

            "proba_conflict": None,
            "proba_negative": None,
            "proba_neutral": None,
            "proba_positive": None,

            "llm_called": True,
            "llm_raw_response": raw,

            "llm_aspect": llm_item.get("corrected_aspect"),
            "llm_polarity": llm_item.get("corrected_polarity"),
            "llm_change_needed": llm_item.get("change_needed"),
            "llm_reason": llm_item.get("reason"),

            "final_aspect": llm_item.get("corrected_aspect"),
            "final_polarity": llm_item.get("corrected_polarity"),

            "human_checked": False,
            "corrected_aspect": llm_item.get("corrected_aspect"),
            "corrected_polarity": llm_item.get("corrected_polarity"),
            "keep": 1,
            "comment": "No ABSA prediction, sent to LLM",

            "threshold": CONFIDENCE_THRESHOLD,
            "timestamp": datetime.now()
        })

        continue

    # =========================
    # GET PROBABILITIES
    # =========================

    inputs = [f"{item['span']} {text}" for item in pred]
    probs = model.polarity_model.predict_proba(inputs)

    for aspect_idx, (item, prob) in enumerate(zip(pred, probs)):

        prob_list = prob.tolist()

        confidence = max(prob_list)
        predicted_label = labels[prob_list.index(confidence)]

        absa_aspect = item["span"]
        absa_polarity = item["polarity"]

        one_pred = {
            "local_item_id": aspect_idx,

            "aspect": absa_aspect,
            "polarity": absa_polarity,
            "proba_predicted_label": predicted_label,
            "confidence": confidence,

            "proba_conflict": prob_list[0],
            "proba_negative": prob_list[1],
            "proba_neutral": prob_list[2],
            "proba_positive": prob_list[3],
        }

        review_predictions.append(one_pred)

        print("-" * 100)
        print(f"Aspect {aspect_idx + 1}/{len(pred)}")
        print(f"ABSA aspect: {absa_aspect}")
        print(f"ABSA polarity: {absa_polarity}")
        print(f"Confidence: {confidence:.4f}")

        if confidence < CONFIDENCE_THRESHOLD:
            print("LOW CONFIDENCE -> marked for LLM")
            uncertain_aspects.append({
                "item_id": aspect_idx,
                "aspect": absa_aspect,
                "polarity": absa_polarity,
                "confidence": round(float(confidence), 4),
                "proba_conflict": round(float(prob_list[0]), 4),
                "proba_negative": round(float(prob_list[1]), 4),
                "proba_neutral": round(float(prob_list[2]), 4),
                "proba_positive": round(float(prob_list[3]), 4),
            })
        else:
            print("HIGH CONFIDENCE -> ABSA accepted")

    # =========================
    # ONE LLM CALL PER REVIEW
    # =========================

    llm_called_for_review = False
    llm_raw_response = None
    llm_items_by_id = {}

    if len(uncertain_aspects) > 0:
        print(f"\nSending {len(uncertain_aspects)} uncertain aspects to LLM in ONE request...")

        parsed, raw = ask_llm_for_uncertain_aspects(
            text=text,
            stars=star,
            uncertain_aspects=uncertain_aspects
        )

        llm_called_for_review = True
        llm_raw_response = raw

        if parsed and "items" in parsed:
            llm_items_by_id = {
                item.get("item_id"): item
                for item in parsed["items"]
            }
            print("LLM answered successfully.")
        else:
            print("LLM JSON parsing failed. Using ABSA predictions.")

    else:
        print("\nNo uncertain aspects. No LLM call.")

    # =========================
    # BUILD FINAL ROWS
    # =========================

    for one_pred in review_predictions:

        aspect_idx = one_pred["local_item_id"]

        absa_aspect = one_pred["aspect"]
        absa_polarity = one_pred["polarity"]
        confidence = one_pred["confidence"]

        llm_item = llm_items_by_id.get(aspect_idx)

        if llm_item is not None:
            llm_called = True

            llm_aspect = llm_item.get("corrected_aspect", absa_aspect)
            llm_polarity = llm_item.get("corrected_polarity", absa_polarity)
            llm_change_needed = llm_item.get("change_needed")
            llm_reason = llm_item.get("reason")

            final_aspect = llm_aspect
            final_polarity = llm_polarity

            comment = "Low confidence, corrected by LLM"

        else:
            llm_called = False

            llm_aspect = None
            llm_polarity = None
            llm_change_needed = None
            llm_reason = None

            final_aspect = absa_aspect
            final_polarity = absa_polarity

            comment = ""

        rows.append({
            "id": review_idx,
            "text": text,
            "stars": star,

            "aspect": absa_aspect,
            "polarity": absa_polarity,
            "proba_predicted_label": one_pred["proba_predicted_label"],
            "confidence": confidence,

            "proba_conflict": one_pred["proba_conflict"],
            "proba_negative": one_pred["proba_negative"],
            "proba_neutral": one_pred["proba_neutral"],
            "proba_positive": one_pred["proba_positive"],

            "llm_called": llm_called,
            "llm_raw_response": llm_raw_response if llm_called else None,

            "llm_aspect": llm_aspect,
            "llm_polarity": llm_polarity,
            "llm_change_needed": llm_change_needed,
            "llm_reason": llm_reason,

            "final_aspect": final_aspect,
            "final_polarity": final_polarity,

            "human_checked": False,
            "corrected_aspect": final_aspect,
            "corrected_polarity": final_polarity,
            "keep": 1,
            "comment": comment,

            "threshold": CONFIDENCE_THRESHOLD,
            "timestamp": datetime.now()
        })

    print(f"Saved rows for review: {len(review_predictions)}")

# =========================
# SAVE RESULT
# =========================

result = pd.DataFrame(rows)
result.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 100)
print("FINISHED")
print("=" * 100)
print(f"Reviews processed: {len(df)}")
print(f"Total output rows: {len(result)}")
print(f"Rows corrected by LLM: {result['llm_called'].sum()}")
print(f"Rows ABSA only: {(result['llm_called'] == False).sum()}")
print(f"Saved to: {OUTPUT_PATH}")

display(result.head())