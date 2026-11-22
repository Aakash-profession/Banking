"""
Step 3: Baseline model, TF-IDF + Logistic Regression (and start using MLflow)
=============================================================================

Goal: build the simplest model that works well, measure it honestly, and record it.
Every fancier model later (embeddings, DistilBERT, LLMs) must BEAT this number to be
worth its extra cost.

What this script does:
    1. Builds a scikit-learn Pipeline: stemming -> TfidfVectorizer -> LogisticRegression.
    2. Trains on train, evaluates on validation (accuracy, macro-F1, weighted-F1).
    3. Saves a per-intent report and finds the 10 most-confused intent pairs.
    4. Measures model size and prediction latency (time per customer message).
    5. Logs everything to MLflow and saves the model to models/tfidf_lr.joblib.

Run:  python src/03_baseline_tfidf.py

Definitions:
    - Bag of words: describe a text by WHICH words it contains and how often,
      ignoring word order. "card not arrived" and "arrived not card" look the same.
    - TF-IDF (Term Frequency x Inverse Document Frequency): a score per word.
        TF  = how often the word appears in THIS message      (more = more important)
        IDF = how rare the word is across ALL messages        (rarer = more important)
      So "arrived" gets a high score, while "my" (in almost every message) gets a low one.
      Everyday example: in a pile of bank complaints, "my" tells you nothing,
      but "swallowed" tells you a lot (card_swallowed).
    - Logistic Regression (multiclass): learns one weight per (word, intent).
      For a new message it adds up the weights for each intent and turns the totals
      into probabilities that sum to 1 (the "softmax" function). Highest = prediction.
    - Pipeline: glues steps together, so .fit() and .predict() run all of them in
      order. It also prevents leakage: the TF-IDF vocabulary is learned from train only.
    - FunctionTransformer: wraps a normal Python function (our stemming) so it can be a
      step inside a Pipeline. Then the SAME cleaning runs in training and in the API.
    - Baseline: a simple first model used as the reference point.
"""

import sys
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from common import (
    METRICS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    ROOT_DIR,
    evaluate_predictions,
    load_splits,
    log_mlflow_run,
    measure_latency_ms,
    print_header,
    save_json,
    top_confused_pairs,
)

sys.path.append(str(ROOT_DIR))   # lets Python find the app/ folder
from app.preprocessing import stem_texts

# ---------------------------------------------------------------------------
# Settings ("hyperparameters" = settings WE choose; the model learns the weights)
# ---------------------------------------------------------------------------
# ngram_range=(1, 2): use single words AND pairs of words ("not received").
# sublinear_tf=True : use log(count) instead of count, so a word repeated 5 times
#                     does not count 5x more. Usually a small gain on short texts.
# min_df=1          : keep words that appear at least once in train.
# C=10              : how much the model trusts the training data. Smaller C = stronger
#                     "regularization" (simpler model, less overfitting). I tried
#                     C = 1, 10, 30 and 100 on validation: C=1 was clearly worse (the model
#                     was too simple: underfitting); 10, 30 and 100 were within 0.1 points of
#                     each other, so I kept the simplest choice, 10 (see LEARNING_NOTES).
NGRAM_RANGE = (1, 2)
SUBLINEAR_TF = True
MIN_DF = 1
C_VALUE = 10.0
MAX_ITER = 2000   # how many steps the optimizer may take to find the best weights

MODEL_NAME = "tfidf_lr"

train_df, val_df, _ = load_splits()

# ---------------------------------------------------------------------------
# 1. Build and train the pipeline
# ---------------------------------------------------------------------------
print_header("1. TRAIN STEMMING + TF-IDF + LOGISTIC REGRESSION")
# Step 2 chose NLTK stemming (almost as good as spaCy lemmatization, much lighter).
# It is the FIRST step of the pipeline, so the saved model always stems its input,
# in training and in the API alike.
pipeline = Pipeline([
    ("stem", FunctionTransformer(stem_texts)),
    ("tfidf", TfidfVectorizer(ngram_range=NGRAM_RANGE, sublinear_tf=SUBLINEAR_TF, min_df=MIN_DF)),
    ("clf", LogisticRegression(C=C_VALUE, max_iter=MAX_ITER)),
])

start = time.perf_counter()
pipeline.fit(train_df["text"], train_df["intent"])
train_time_s = time.perf_counter() - start

vocabulary_size = len(pipeline.named_steps["tfidf"].vocabulary_)
print(f"Training time   : {train_time_s:.1f} s")
print(f"Vocabulary size : {vocabulary_size:,} words and word-pairs")

# ---------------------------------------------------------------------------
# 2. Evaluate on validation
# ---------------------------------------------------------------------------
print_header("2. VALIDATION RESULTS")
val_predictions = pipeline.predict(val_df["text"])
metrics = evaluate_predictions(val_df["intent"], val_predictions)
for name, value in metrics.items():
    print(f"  {name:<12}: {value:.4f}")

# Per-intent report: precision, recall and F1 for EACH of the 77 intents.
report_text = classification_report(val_df["intent"], val_predictions, digits=3, zero_division=0)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
report_path = REPORTS_DIR / "step3_classification_report.txt"
report_path.write_text(report_text, encoding="utf-8")
print(f"\nFull per-intent report saved: {report_path.relative_to(REPORTS_DIR.parent)}")

# The 5 intents the model finds hardest (lowest F1).
labels = sorted(val_df["intent"].unique())
per_intent_f1 = f1_score(val_df["intent"], val_predictions, labels=labels, average=None)
worst = sorted(zip(labels, per_intent_f1), key=lambda pair: pair[1])[:5]
print("\n5 hardest intents (lowest F1):")
for intent, score in worst:
    print(f"  {intent:<45} F1 = {score:.3f}")

# ---------------------------------------------------------------------------
# 3. Most-confused intent pairs
# ---------------------------------------------------------------------------
print_header("3. TOP 10 CONFUSED PAIRS (true -> predicted)")
confused = top_confused_pairs(val_df["intent"].tolist(), val_predictions.tolist(), n=10)
for item in confused:
    print(f"  {item['count']:>2} x  {item['true']:<42} -> {item['predicted']}")

# ---------------------------------------------------------------------------
# 4. Size and latency
# ---------------------------------------------------------------------------
print_header("4. MODEL SIZE + LATENCY")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODELS_DIR / f"{MODEL_NAME}.joblib"
# joblib saves a Python object (our whole pipeline) to a file, so we can load it later
# without retraining. (It uses "pickle" inside, which is why library versions must match
# when loading: see Step 10.)
joblib.dump(pipeline, model_path)
model_size_mb = model_path.stat().st_size / (1024 * 1024)


def predict_one(text: str) -> str:
    """Predict the intent of ONE message, like during a live call."""
    return pipeline.predict([text])[0]


latency = measure_latency_ms(predict_one, val_df["text"].tolist()[:200])
print(f"Model file size : {model_size_mb:.2f} MB")
print(f"Latency (1 msg) : avg {latency['latency_avg_ms']:.2f} ms, p95 {latency['latency_p95_ms']:.2f} ms")

# ---------------------------------------------------------------------------
# 5. Save results + log to MLflow
# ---------------------------------------------------------------------------
print_header("5. SAVE + LOG TO MLFLOW")
results = {
    "model": MODEL_NAME,
    "split": "validation",
    **metrics,                       # ** copies all keys of the metrics dict in here
    "train_time_s": round(train_time_s, 2),
    **latency,
    "model_size_mb": round(model_size_mb, 2),
}
save_json(results, METRICS_DIR / f"{MODEL_NAME}.json")
save_json({"confused_pairs": confused}, REPORTS_DIR / "step3_confused_pairs.json")

log_mlflow_run(
    run_name=MODEL_NAME,
    # params = what we CHOSE
    params={
        "model": MODEL_NAME,
        "preprocessing": "nltk_porter_stem",
        "ngram_range": str(NGRAM_RANGE),
        "sublinear_tf": SUBLINEAR_TF,
        "min_df": MIN_DF,
        "C": C_VALUE,
        "vocabulary_size": vocabulary_size,
    },
    # metrics = what we MEASURED
    metrics={
        "val_accuracy": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "val_weighted_f1": metrics["weighted_f1"],
        "train_time_s": train_time_s,
        "latency_avg_ms": latency["latency_avg_ms"],
        "model_size_mb": model_size_mb,
    },
    # artifacts = files we want to keep next to the run
    artifact_paths=[report_path],
)

print(f"Saved model   : models/{MODEL_NAME}.joblib")
print(f"Saved metrics : reports/metrics/{MODEL_NAME}.json")
print(f"MLflow        : run '{MODEL_NAME}' logged (open with: mlflow ui)")
