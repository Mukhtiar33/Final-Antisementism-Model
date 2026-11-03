# -*- coding: utf-8 -*-
"""
train_pipeline.py - Full ML Pipeline with Charts, Reports and Evaluation
"""
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

print("\n" + "="*60)
print("  STEP 1 - LOADING DATA")
print("="*60)

df = pd.read_csv(BASE / "final_clean_dataset.csv")
df = df[df["clean_text"].notna() & (df["clean_text"].str.strip() != "")]
df = df[df["sentiment_label"].isin([-1, 0, 1])]
df.drop_duplicates(subset=["video_id", "clean_text"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  Rows loaded        : {len(df)}")
print(f"  Videos             : {df['video_id'].nunique()}")
print(f"  Missing values     : {df.isnull().sum().sum()}")
print(f"\n  Sentiment distribution:")
dist = df["sentiment_label"].value_counts().sort_index()
label_map = {-1: "negative", 0: "neutral", 1: "positive"}
for lbl, cnt in dist.items():
    print(f"    {label_map[lbl]:10s}: {cnt} ({cnt/len(df)*100:.1f}%)")

print("\n" + "="*60)
print("  STEP 2 - FEATURE ENGINEERING")
print("="*60)

df["final_input_text"] = (
    df["clean_text"].fillna("") + " " + df["transcript_context"].fillna("")
).str.strip()

X = df["final_input_text"].values
y = df["sentiment_label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True)
X_train_v = vec.fit_transform(X_train)
X_test_v  = vec.transform(X_test)

print(f"  TF-IDF features    : {X_train_v.shape[1]}")
print(f"  Train size         : {len(X_train)}")
print(f"  Test size          : {len(X_test)}")

print("\n" + "="*60)
print("  STEP 3 - TRAINING MODELS")
print("="*60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0,
                                               class_weight="balanced",
                                               random_state=42),
    "Linear SVM":          LinearSVC(max_iter=2000, C=1.0,
                                      class_weight="balanced",
                                      random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_v, y_train)
    y_pred = model.predict(X_test_v)
    cv = cross_val_score(model, X_train_v, y_train, cv=5, scoring="f1_macro")
    results[name] = {
        "model":      model,
        "y_pred":     y_pred,
        "accuracy":   accuracy_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall":     recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1":         f1_score(y_test, y_pred, average="macro", zero_division=0),
        "cv_mean":    cv.mean(),
        "cv_std":     cv.std(),
        "report":     classification_report(y_test, y_pred,
                          target_names=["negative","neutral","positive"],
                          zero_division=0),
        "cm":         confusion_matrix(y_test, y_pred),
    }
    print(f"\n  [{name}]")
    print(f"    Accuracy  : {results[name]['accuracy']:.4f}")
    print(f"    F1 (macro): {results[name]['f1']:.4f}")
    print(f"    CV F1     : {results[name]['cv_mean']:.4f} +/- {results[name]['cv_std']:.4f}")

best_name = max(results, key=lambda k: results[k]["f1"])
best = results[best_name]
print(f"\n  Best model: {best_name} (F1={best['f1']:.4f})")

# Save models
joblib.dump(best["model"], OUT / "trained_model.pkl")
joblib.dump(vec,           OUT / "vectorizer.pkl")
print(f"  Saved -> trained_model.pkl")
print(f"  Saved -> vectorizer.pkl")

print("\n" + "="*60)
print("  STEP 4 - EVALUATION REPORT")
print("="*60)

report_lines = [
    "=" * 60,
    "  EVALUATION REPORT - YouTube Sentiment Analysis",
    "=" * 60,
    f"\n  Dataset Overview",
    f"  Total comments : {len(df)}",
    f"  Videos         : {df['video_id'].nunique()}",
    f"  Train / Test   : {len(X_train)} / {len(X_test)}",
    f"\n  Label Distribution",
]
for lbl, cnt in dist.items():
    report_lines.append(f"  {label_map[lbl]:10s}: {cnt} ({cnt/len(df)*100:.1f}%)")

for name, res in results.items():
    report_lines += [
        f"\n  {'='*50}",
        f"  Model: {name}",
        f"  {'='*50}",
        f"  Accuracy          : {res['accuracy']:.4f}",
        f"  Precision (macro) : {res['precision']:.4f}",
        f"  Recall (macro)    : {res['recall']:.4f}",
        f"  F1 (macro)        : {res['f1']:.4f}",
        f"  CV F1             : {res['cv_mean']:.4f} +/- {res['cv_std']:.4f}",
        f"\n  Per-class Report:",
        res["report"],
    ]

report_lines += [
    "=" * 60,
    f"  BEST MODEL: {best_name}",
    f"  Final F1 (macro): {best['f1']:.4f}",
    "=" * 60,
]
(OUT / "evaluation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
print("  Saved -> evaluation_report.txt")

# Sample predictions
sample_idx = np.random.choice(len(X_test), min(50, len(X_test)), replace=False)
sample_df = pd.DataFrame({
    "text":            [X_test[i] for i in sample_idx],
    "actual_label":    [y_test[i] for i in sample_idx],
    "predicted_label": [best["y_pred"][i] for i in sample_idx],
})
sample_df["actual_name"]    = sample_df["actual_label"].map(label_map)
sample_df["predicted_name"] = sample_df["predicted_label"].map(label_map)
sample_df["correct"]        = sample_df["actual_label"] == sample_df["predicted_label"]
sample_df.to_csv(OUT / "predictions_sample.csv", index=False)
print("  Saved -> predictions_sample.csv")

df.to_csv(OUT / "processed_dataset.csv", index=False)
print("  Saved -> processed_dataset.csv")

print("\n" + "="*60)
print("  STEP 5 - VISUALIZATIONS")
print("="*60)

sns.set_style("whitegrid")
COLORS = {"negative": "#e74c3c", "neutral": "#f39c12", "positive": "#2ecc71"}

# (1) Label distribution
fig, ax = plt.subplots(figsize=(7, 4))
labels_sorted = ["negative", "neutral", "positive"]
counts = [dist.get(k, 0) for k in [-1, 0, 1]]
bars = ax.bar(labels_sorted, counts,
              color=[COLORS[l] for l in labels_sorted],
              edgecolor="white", width=0.5)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(cnt), ha="center", va="bottom", fontweight="bold")
ax.set_title("Sentiment Label Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUT / "label_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> label_distribution.png")

# (2) Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(best["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["negative","neutral","positive"],
            yticklabels=["negative","neutral","positive"],
            ax=ax, linewidths=0.5, linecolor="white")
ax.set_title(f"Confusion Matrix - {best_name}", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> confusion_matrix.png")

# (3) Top 20 word frequencies
from collections import Counter
all_words = " ".join(df["clean_text"].dropna()).split()
word_freq = Counter(w for w in all_words if len(w) > 3 and w.isalpha())
top20 = word_freq.most_common(20)
words, freqs = zip(*top20)
fig, ax = plt.subplots(figsize=(9, 5))
palette = sns.color_palette("Blues_d", len(words))
ax.barh(list(reversed(words)), list(reversed(freqs)),
        color=list(reversed(palette)))
ax.set_title("Top 20 Word Frequencies", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Frequency")
plt.tight_layout()
plt.savefig(OUT / "word_frequency_top20.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> word_frequency_top20.png")

# (4) Model comparison
fig, ax = plt.subplots(figsize=(7, 4))
metric_names = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metric_names))
width = 0.35
for i, (mname, mres) in enumerate(results.items()):
    vals = [mres[m] for m in metric_names]
    bars = ax.bar(x + i*width, vals, width, label=mname,
                  color=["#3498db","#e67e22"][i], alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
ax.set_xticks(x + width/2)
ax.set_xticklabels([m.capitalize() for m in metric_names])
ax.set_ylim(0, 1.15)
ax.set_title("Model Comparison", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Score")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> model_comparison.png")

# ML Report
md = f"""# ML Report - YouTube Antisemitism Sentiment Analysis

## Dataset Overview
| Field | Value |
|-------|-------|
| Total comments | {len(df)} |
| Videos processed | {df['video_id'].nunique()} |
| Train / Test | {len(X_train)} / {len(X_test)} |

## Label Distribution
| Label | Count | % |
|-------|-------|---|
""" + "\n".join(
    f"| {label_map[k]} | {dist.get(k,0)} | {dist.get(k,0)/len(df)*100:.1f}% |"
    for k in [-1, 0, 1]
) + f"""

## Model Results
""" + "\n".join(
    f"### {name}\n"
    f"| Metric | Score |\n|--------|-------|\n"
    f"| Accuracy | {res['accuracy']:.4f} |\n"
    f"| Precision | {res['precision']:.4f} |\n"
    f"| Recall | {res['recall']:.4f} |\n"
    f"| F1 (macro) | {res['f1']:.4f} |\n"
    f"| CV F1 | {res['cv_mean']:.4f} +/- {res['cv_std']:.4f} |"
    for name, res in results.items()
) + f"""

## Best Model
**{best_name}** - F1 (macro): `{best['f1']:.4f}`

## Output Files
| File | Description |
|------|-------------|
| trained_model.pkl | Best model ({best_name}) |
| vectorizer.pkl | TF-IDF vectorizer |
| processed_dataset.csv | Clean training dataset |
| predictions_sample.csv | 50 sample predictions |
| evaluation_report.txt | Full metrics |
| label_distribution.png | Label chart |
| confusion_matrix.png | Confusion matrix |
| word_frequency_top20.png | Top 20 words |
| model_comparison.png | Model comparison chart |
"""
(OUT / "ML_REPORT.md").write_text(md, encoding="utf-8")
print("  Saved -> ML_REPORT.md")

print("\n" + "="*60)
print("  ALL DONE - Check the outputs/ folder")
print("="*60)
print(f"\n  outputs/ contains:")
for f in sorted(OUT.iterdir()):
    print(f"    {f.name}")
print()
