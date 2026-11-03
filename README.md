# Antisemitism Sentiment Analysis Model

YouTube comment sentiment analysis pipeline that classifies comments as negative/neutral/positive, with an antisemitism-focused prediction wrapper.

## Project structure

- `preprocess.py` — Production-ready NLP preprocessing pipeline. Loads scraped Excel/CSV comment & transcript files per video, cleans text, detects header rows, removes garbage rows, and assigns VADER sentiment labels. Outputs `final_clean_dataset.csv` and `clean_transcripts.csv`.
- `train_pipeline.py` — Full ML training pipeline. Trains Logistic Regression and Linear SVM models with TF-IDF features, evaluates with cross-validation, saves the best model + vectorizer to `outputs/`, and writes charts, an evaluation report, and `ML_REPORT.md`.
- `predict.py` — Inference script. Loads the trained model, vectorizer, and threshold to classify a single text as `ANTISEMITIC` or `NOT ANTISEMITIC`.

## Data

Video source data (`video1` … `video9`) contains `Comments.xlsx` and `Transcript.xlsx` files. Large CSVs, pickled models/vectorizers, and generated outputs are gitignored.

Run preprocessing:

```bash
python preprocess.py /path/to/base_dir
```

Train:

```bash
python train_pipeline.py
```

Predict:

```bash
python predict.py "your comment text here"
```

## License

MIT
