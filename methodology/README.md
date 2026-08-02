# Methodology

## 1. Problem Statement

Traditional sentiment analysis assigns a single sentiment label to an entire review, which loses information in cases where a customer simultaneously praises certain aspects (e.g. food) while criticizing others (e.g. service). Aspect-Based Sentiment Analysis (ABSA) addresses this by extracting individual aspects from the text and determining sentiment for each one independently.

The goal of this project is to build a hybrid ABSA system that combines a lightweight local model for high-volume processing with an LLM fallback for difficult, low-confidence cases, and that improves iteratively through human review and fine-tuning on "weak" data.

## 2. System Architecture

The system consists of the following components:

1. **Aspect candidate extraction** — spaCy is used to identify potential noun spans that may represent aspects.
2. **Aspect model (SetFit)** — `tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect` filters candidates, keeping only valid aspects.
3. **Polarity model (SetFit)** — `tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity` classifies sentiment (positive / negative / neutral / conflict) for each extracted aspect.
4. **Confidence-based routing** — if the polarity model's confidence falls below a set threshold, the prediction is flagged as low-confidence and routed to the LLM fallback.
5. **LLM fallback** — for low-confidence cases, the request is processed by an LLM, which produces the final sentiment prediction.
6. **Human review** — a Streamlit application (`absa_review_app.py`) allows manual inspection and correction of predictions (both baseline and LLM-fallback), with probability visualization, filtering by case type, and progress tracking.
7. **Feedback logging** — manual corrections are logged and form the dataset used for subsequent fine-tuning (active learning loop).

Both models (aspect and polarity) are built on the `BAAI/bge-small-en-v1.5` Sentence Transformer backbone with a Logistic Regression classification head (SetFit architecture).

## 3. Confidence-Based Routing Logic

The core idea of the hybrid architecture is not to send every request to the LLM (which is costly and slow), but to use it selectively, only where the local model is actually likely to be wrong.

- For each prediction, the polarity model returns a confidence score.
- If `confidence < threshold`, the case is routed to the LLM.
- In the current implementation, approximately 16% of cases are routed through the LLM (699 out of 4,406 aspect-level samples).

The confidence threshold is a hyperparameter that directly affects the trade-off between system cost/latency and final accuracy. It therefore needs to be studied separately (see Section 5).

## 4. Fine-Tuning on Weak Data (Active Learning Loop)

Beyond the static pipeline, the system is designed to be self-improving:

1. In the Streamlit application, a human corrects erroneous predictions (from both the baseline model and the LLM).
2. Corrected examples — particularly those where the baseline was wrong ("weak" cases) — are accumulated into a separate dataset.
3. This dataset is used to fine-tune the polarity model.
4. After fine-tuning, the pipeline is re-evaluated on the same metrics as the baseline to measure the improvement.

Hypothesis: since the baseline model's main weakness is concentrated in the neutral and conflict classes (see Per-Class F1 in the Metrics section), fine-tuning specifically on these "weak" examples should selectively improve recall/F1 on the problematic classes without degrading accuracy on the "easy" positive/negative cases.

This differs from ordinary fine-tuning on a random subsample: the fine-tuning dataset is not formed randomly but through the routing mechanism plus human-in-the-loop correction, meaning the model is deliberately trained on its own errors.

## 5. Confidence Threshold Sensitivity Analysis

In addition to the fixed threshold used in the current pipeline, a separate experiment is planned to evaluate the hybrid system's accuracy at different routing thresholds:

- **Confidence < 40%** — only the most uncertain cases are routed to the LLM (minimal LLM load).
- **Confidence < 50%** — current/baseline scenario.
- **Confidence < 70%** — significantly more cases are routed to the LLM (more conservative approach).

For each threshold, the following are measured:

- The proportion of cases routed to the LLM (a proxy for cost/latency).
- The pipeline's resulting metrics (accuracy, weighted F1, F2, recall) — both on the full dataset and on the low-confidence subsample.
- The improvement in metrics relative to the baseline at each threshold.

The goal is to build a "LLM call cost vs. quality gain" curve and justify the choice of threshold empirically rather than arbitrarily, identifying the point with the best cost-quality trade-off.

## 6. Evaluation Metrics

The system is evaluated on four key metrics, each capturing a different aspect of quality:

| Metric | Rationale |
|---|---|
| **Accuracy** | Overall proportion of correct predictions — an intuitive baseline measure, but sensitive to class imbalance (positive occurs far more often than conflict). |
| **Weighted F1** | Accounts for class imbalance by weighting each class's F1 score by its share of the dataset — gives a more honest picture than accuracy under an uneven sentiment distribution. |
| **Recall (per-class / macro)** | Critical for this task, since missing a negative review or a conflict case is more costly than a false positive — low recall on the negative/conflict classes means the business fails to see real problems. |
| **F2-score** | Weights recall more heavily than precision (unlike F1, which weights them equally). Rationale: in the context of extracting business insights, missing a genuine negative signal (false negative) is generally more costly than mislabeling a neutral review as negative (false positive) — so recall should be prioritized over precision, and F2 reflects this priority better than F1. |

Additionally (as already implemented), **Macro Precision/Recall/F1** and **Per-Class F1** are tracked to make the model's weak points visible at the class level, rather than relying only on aggregate metrics.

## 7. Experimental Design (Summary)

Evaluation is conducted across three slices:
1. **Raw baseline** — predictions from the local SetFit model only, without fallback.
2. **Final pipeline output** — the hybrid system's final predictions (baseline + LLM for low-confidence cases).
3. **LLM-fallback subset** — predictions restricted to the subsample that was routed to the LLM (an isolated evaluation of fallback effectiveness).

This three-tier comparison separates the claim "the system as a whole became more accurate" from the more specific question "does the LLM actually resolve the cases where the baseline is wrong" — which is confirmed by the accuracy increase from 47.4% to 96.0% specifically on the low-confidence subsample.

## 8. Limitations and Next Steps

- The current models are trained on restaurant reviews (SemEval 2014 Restaurant) and are being adapted to French-language reviews — the cross-domain/cross-lingual transfer aspect should be explicitly acknowledged.
- Near-zero conflict-class polarity in some slices may indicate a pipeline bug (already noted) and requires a separate investigation before the methodology is finalized.
- After fine-tuning on weak data, the full set of experiments (including the threshold analysis) should be repeated to check whether the model has overfit to specific error patterns present in the review sample.
