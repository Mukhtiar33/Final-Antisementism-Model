# -*- coding: utf-8 -*-
import sys, joblib
from pathlib import Path

OUT = Path(__file__).parent

def predict(text):
    model      = joblib.load(OUT / "trained_model.pkl")
    vectorizer = joblib.load(OUT / "vectorizer.pkl")
    threshold  = joblib.load(OUT / "threshold.pkl")

    tv    = vectorizer.transform([text.lower()])
    prob  = model.predict_proba(tv)[0][1]
    pred  = int(prob >= threshold)
    label = "ANTISEMITIC" if pred == 1 else "NOT ANTISEMITIC"
    return {"text": text, "label": label, "confidence": round(prob, 3)}

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "This is a test comment."
    result = predict(text)
    print()
    print("Text       :", result["text"])
    print("Label      :", result["label"])
    print("Confidence :", result["confidence"])
