# ML Report - YouTube Antisemitism Sentiment Analysis

## Dataset Overview
| Field | Value |
|-------|-------|
| Total comments | 615 |
| Videos processed | 9 |
| Train / Test | 492 / 123 |

## Label Distribution
| Label | Count | % |
|-------|-------|---|
| negative | 252 | 41.0% |
| neutral | 174 | 28.3% |
| positive | 189 | 30.7% |

## Model Results
### Logistic Regression
| Metric | Score |
|--------|-------|
| Accuracy | 0.3740 |
| Precision | 0.4021 |
| Recall | 0.3840 |
| F1 (macro) | 0.3745 |
| CV F1 | 0.4255 +/- 0.0796 |
### Linear SVM
| Metric | Score |
|--------|-------|
| Accuracy | 0.4309 |
| Precision | 0.4474 |
| Recall | 0.4342 |
| F1 (macro) | 0.4290 |
| CV F1 | 0.4672 +/- 0.0427 |

## Best Model
**Linear SVM** - F1 (macro): `0.4290`

## Output Files
| File | Description |
|------|-------------|
| trained_model.pkl | Best model (Linear SVM) |
| vectorizer.pkl | TF-IDF vectorizer |
| processed_dataset.csv | Clean training dataset |
| predictions_sample.csv | 50 sample predictions |
| evaluation_report.txt | Full metrics |
| label_distribution.png | Label chart |
| confusion_matrix.png | Confusion matrix |
| word_frequency_top20.png | Top 20 words |
| model_comparison.png | Model comparison chart |
