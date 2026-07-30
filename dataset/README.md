# Dataset

This directory contains a manually validated restaurant review dataset used to evaluate the Hybrid ABSA Pipeline.

The dataset consists of **200 Yelp reviews** collected from the **Farmicia** restaurant. Every review and every extracted aspect was manually inspected to produce high-quality ground truth labels.

The manual review process included:

- verifying extracted aspects;
- correcting incorrect aspect spans;
- correcting sentiment labels;
- removing invalid aspect predictions;
- documenting the reason for every correction.

The entire annotation and validation process required approximately **one week** of manual work.

## Dataset Columns

| Column | Description |
|---------|-------------|
| `id` | Review identifier |
| `text` | Original Yelp review |
| `stars` | Yelp star rating |
| `aspect` | Aspect predicted by the baseline ABSA model |
| `polarity` | Sentiment predicted by the baseline ABSA model |
| `proba_predicted_label` | Probability of the predicted sentiment |
| `confidence` | Model confidence score |
| `proba_conflict` | Probability of the conflict class |
| `proba_negative` | Probability of the negative class |
| `proba_neutral` | Probability of the neutral class |
| `proba_positive` | Probability of the positive class |
| `llm_called` | Whether the sample was routed to the LLM |
| `llm_raw_response` | Raw LLM response |
| `llm_aspect` | Aspect proposed by the LLM |
| `llm_polarity` | Sentiment proposed by the LLM |
| `llm_change_needed` | Whether the LLM suggested a correction |
| `llm_reason` | LLM explanation |
| `final_aspect` | Final aspect after the hybrid pipeline |
| `final_polarity` | Final sentiment after the hybrid pipeline |
| `human_checked` | Whether the sample was manually reviewed |
| `corrected_aspect` | Final manually validated aspect (ground truth) |
| `corrected_polarity` | Final manually validated sentiment (ground truth) |
| `keep` | Whether the prediction was kept |
| `comment` | Annotator comments |
| `threshold` | Confidence threshold used by the pipeline |
| `timestamp` | Processing timestamp |

## Purpose

This dataset serves as the evaluation benchmark for the Hybrid ABSA Pipeline. The manually validated labels (`corrected_aspect` and `corrected_polarity`) are used as the ground truth for measuring both the baseline ABSA model and the proposed hybrid system.