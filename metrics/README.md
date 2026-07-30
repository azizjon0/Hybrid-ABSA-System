# Metrics

This directory contains the evaluation results of the proposed Hybrid ABSA Pipeline.

The pipeline was evaluated against the baseline SetFit ABSA model under two scenarios:

- **Full Dataset** (4,406 aspect-level samples)
- **Low-Confidence Samples** (699 samples routed to the LLM)

## Full Dataset (4,406 Samples)

| Metric | Baseline ABSA | Hybrid Pipeline |
|--------|--------------:|----------------:|
| Accuracy | **0.8700** | **0.9467** |
| Macro Precision | 0.5675 | **0.9590** |
| Macro Recall | 0.5275 | **0.9063** |
| Macro F1 | 0.5400 | **0.9288** |
| Weighted F1 | 0.8600 | **0.9450** |

### Per-Class F1 Score

| Class | Baseline ABSA | Hybrid Pipeline |
|------|--------------:|----------------:|
| Conflict | 0.0000 | **1.0000** |
| Negative | 0.7400 | **0.9034** |
| Neutral | 0.4900 | **0.8426** |
| Positive | 0.9400 | **0.9691** |

---

## Low-Confidence Samples (699)

These samples were automatically routed to the LLM because the baseline ABSA model produced low-confidence predictions.

| Metric | Baseline ABSA | Hybrid Pipeline |
|--------|--------------:|----------------:|
| Accuracy | **0.4735** | **0.9599** |
| Macro Precision | 0.3410 | **0.8729** |
| Macro Recall | 0.3477 | **0.9687** |
| Macro F1 | 0.3406 | **0.9080** |
| Weighted F1 | 0.4631 | **0.9601** |

### Per-Class F1 Score

| Class | Baseline ABSA | Hybrid Pipeline |
|------|--------------:|----------------:|
| Conflict | 0.0000 | **0.7500** |
| Negative | 0.5768 | **0.9682** |
| Neutral | 0.3501 | **0.9514** |
| Positive | 0.4353 | **0.9625** |

---

## Summary

The baseline ABSA model achieved **87.0% accuracy** on the complete dataset. By selectively routing only low-confidence predictions to an LLM, the proposed Hybrid Pipeline improved overall accuracy to **94.7%**.

The improvement was even more pronounced on low-confidence samples, where accuracy increased from **47.4%** to **96.0%**, demonstrating that the LLM effectively resolves cases where the baseline model is uncertain.