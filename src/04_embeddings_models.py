"""
Step 4: Sentence embeddings + Logistic Regression / XGBoost
============================================================

Goal: replace TF-IDF word counts with "meaning vectors" (embeddings), train two models on
them, and compare all models on accuracy AND speed AND size, the way you would choose a
model for production.

What this script does:
    1. Turns every message into a 384-number embedding with all-MiniLM-L6-v2 and caches
       the result to disk (encoding is the slow part, so we do it only once).
    2. Shows cosine similarity between a few sentences (meaning, not shared words).
    3. Trains Logistic Regression and XGBoost on the embeddings.
    4. Measures latency per message INCLUDING the embedding step (what a live call pays).
    5. Prints a comparison table with the Step 3 baseline and logs to MLflow.

Run:  python src/04_embeddings_models.py
(First run downloads the MiniLM model, about 90 MB.)

Definitions:
    - Embedding: a list of numbers (a vector) that represents the MEANING of a text.
      Texts with similar meaning get vectors that point in similar directions.
      "my card hasn't come" and "card still not delivered" share almost no words,
      but their embeddings are close. TF-IDF would see them as unrelated.
    - Sentence-Transformers: a library of models that turn whole sentences into
      embeddings. all-MiniLM-L6-v2 is small (6 layers, 22M parameters) and fast.
    - Cosine similarity: how similar two vectors' directions are.
      1 = same meaning direction, 0 = unrelated. If vectors are "normalized"
      (length 1), cosine similarity is simply the dot product.
    - XGBoost (gradient boosting): builds many small decision trees ONE AFTER ANOTHER.
      Each new tree focuses on fixing the mistakes the previous trees still make.
      Everyday example: a team of students where each one only studies the
      questions the previous students got wrong.
    - Latency: how long one prediction takes. In an IVR the caller is waiting on the
      line, so latency matters as much as accuracy.
"""

import time

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from common import (
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    evaluate_predictions,
    load_json,
    load_splits,
    log_mlflow_run,
    measure_latency_ms,
    print_header,
    save_json,
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

# XGBoost settings (kept modest so training finishes in a few minutes on a laptop CPU).
# n_estimators  = number of boosting rounds (each round adds one tree PER intent)
# max_depth     = how deep each tree may grow (deeper = more complex = overfits sooner)
# learning_rate = how much each new tree is allowed to change the prediction
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.1

train_df, val_df, test_df = load_splits()

# ---------------------------------------------------------------------------
# 1. Encode all texts once and cache them
# ---------------------------------------------------------------------------
print_header("1. ENCODE TEXTS WITH all-MiniLM-L6-v2")
encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
# Size of the MiniLM model in memory: number of weights x 4 bytes (each weight is a
# 32-bit float). A deployed embedding model must ship this PLUS the classifier.
encoder_size_mb = sum(parameter.numel() for parameter in encoder.parameters()) * 4 / (1024 * 1024)
print(f"MiniLM size: {encoder_size_mb:.0f} MB")


def get_embeddings(split_name: str, texts: list[str]) -> np.ndarray:
    """Load cached embeddings if they exist; otherwise encode and save them.

    normalize_embeddings=True makes every vector length 1, so dot product = cosine.
    """
    cache_path = EMBEDDINGS_DIR / f"{split_name}.npy"
    if cache_path.exists():
        print(f"  {split_name:<5}: loaded from cache ({cache_path.name})")
        return np.load(cache_path)

    print(f"  {split_name:<5}: encoding {len(texts):,} texts...")
    vectors = encoder.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    return vectors


start = time.perf_counter()
X_train = get_embeddings("train", train_df["text"].tolist())
X_val = get_embeddings("val", val_df["text"].tolist())
# We also encode the test set now (cached for Step 9), but we do NOT evaluate on it here.
get_embeddings("test", test_df["text"].tolist())
print(f"Embedding shape (train): {X_train.shape}  -> {X_train.shape[0]:,} texts x {X_train.shape[1]} numbers")
print(f"Time: {time.perf_counter() - start:.1f} s")

# ---------------------------------------------------------------------------
# 2. Cosine similarity demo
# ---------------------------------------------------------------------------
print_header("2. COSINE SIMILARITY (meaning, not shared words)")
sentence_pairs = [
    ("my card hasn't come yet", "card still not delivered"),        # same meaning, few shared words
    ("my card hasn't come yet", "how do I activate my card"),       # shares "my card", different intent
    ("my card hasn't come yet", "what is the weather in Bangalore"),  # unrelated
]
for sentence_a, sentence_b in sentence_pairs:
    vector_a, vector_b = encoder.encode([sentence_a, sentence_b], normalize_embeddings=True)
    similarity = float(np.dot(vector_a, vector_b))   # normalized vectors -> dot = cosine
    print(f"  {similarity:.3f}   '{sentence_a}'  vs  '{sentence_b}'")

y_train = train_df["intent"].to_numpy()
y_val = val_df["intent"].to_numpy()
all_results = {}

# ---------------------------------------------------------------------------
# 3a. Logistic Regression on embeddings
# ---------------------------------------------------------------------------
print_header("3a. LOGISTIC REGRESSION ON EMBEDDINGS")
lr_model = LogisticRegression(C=10.0, max_iter=2000)
start = time.perf_counter()
lr_model.fit(X_train, y_train)
lr_train_time = time.perf_counter() - start

lr_metrics = evaluate_predictions(y_val, lr_model.predict(X_val))
print(f"Training time: {lr_train_time:.1f} s   |   {lr_metrics}")


def predict_one_lr(text: str) -> str:
    """Full live-call path: encode ONE message, then classify it."""
    vector = encoder.encode([text], normalize_embeddings=True)
    return lr_model.predict(vector)[0]


lr_latency = measure_latency_ms(predict_one_lr, val_df["text"].tolist()[:200])
joblib.dump(lr_model, MODELS_DIR / "minilm_lr.joblib")
lr_size_mb = encoder_size_mb + (MODELS_DIR / "minilm_lr.joblib").stat().st_size / (1024 * 1024)
all_results["minilm_lr"] = {"train_time_s": lr_train_time, **lr_metrics, **lr_latency,
                            "model_size_mb": round(lr_size_mb, 1)}

# ---------------------------------------------------------------------------
# 3b. XGBoost on embeddings
# ---------------------------------------------------------------------------
print_header("3b. XGBOOST ON EMBEDDINGS")
# XGBoost needs NUMBER labels (0, 1, 2, ...), not text labels.
# LabelEncoder makes the mapping: "activate_my_card" -> 0, "age_limit" -> 1, ...
label_encoder = LabelEncoder()
y_train_numbers = label_encoder.fit_transform(y_train)

xgb_model = XGBClassifier(
    n_estimators=XGB_N_ESTIMATORS,
    max_depth=XGB_MAX_DEPTH,
    learning_rate=XGB_LEARNING_RATE,
    tree_method="hist",       # fast histogram-based tree building
    n_jobs=-1,                # use all CPU cores
    random_state=RANDOM_SEED,
)
print(f"Training {XGB_N_ESTIMATORS} rounds x 77 intents = {XGB_N_ESTIMATORS * 77:,} small trees "
      "(a few minutes on a laptop)...")
start = time.perf_counter()
xgb_model.fit(X_train, y_train_numbers)
xgb_train_time = time.perf_counter() - start

# Turn number predictions back into intent names with inverse_transform.
xgb_val_predictions = label_encoder.inverse_transform(xgb_model.predict(X_val))
xgb_metrics = evaluate_predictions(y_val, xgb_val_predictions)
print(f"Training time: {xgb_train_time:.1f} s   |   {xgb_metrics}")


def predict_one_xgb(text: str) -> str:
    vector = encoder.encode([text], normalize_embeddings=True)
    return label_encoder.inverse_transform(xgb_model.predict(vector))[0]


xgb_latency = measure_latency_ms(predict_one_xgb, val_df["text"].tolist()[:200])
joblib.dump(xgb_model, MODELS_DIR / "minilm_xgb.joblib")
joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
xgb_size_mb = encoder_size_mb + (MODELS_DIR / "minilm_xgb.joblib").stat().st_size / (1024 * 1024)
all_results["minilm_xgb"] = {"train_time_s": xgb_train_time, **xgb_metrics, **xgb_latency,
                             "model_size_mb": round(xgb_size_mb, 1)}

# ---------------------------------------------------------------------------
# 4. Save, log to MLflow, and compare with the baseline
# ---------------------------------------------------------------------------
print_header("4. COMPARISON ON VALIDATION")
for model_name, result in all_results.items():
    result = {"model": model_name, "split": "validation", **result}
    result["train_time_s"] = round(result["train_time_s"], 2)
    save_json(result, METRICS_DIR / f"{model_name}.json")

    params = {"model": model_name, "embedding_model": EMBEDDING_MODEL_NAME}
    if model_name == "minilm_xgb":
        params["n_estimators"] = XGB_N_ESTIMATORS
        params["max_depth"] = XGB_MAX_DEPTH
        params["learning_rate"] = XGB_LEARNING_RATE
    log_mlflow_run(
        run_name=model_name,
        params=params,
        metrics={
            "val_accuracy": result["accuracy"],
            "val_macro_f1": result["macro_f1"],
            "val_weighted_f1": result["weighted_f1"],
            "train_time_s": result["train_time_s"],
            "latency_avg_ms": result["latency_avg_ms"],
            "model_size_mb": result["model_size_mb"],
        },
    )

print(f"{'model':<12}{'accuracy':>10}{'macro_f1':>10}{'train_s':>10}{'latency_ms':>12}{'size_MB':>9}")
for model_name in ["tfidf_lr", "minilm_lr", "minilm_xgb"]:
    metrics_path = METRICS_DIR / f"{model_name}.json"
    if not metrics_path.exists():
        continue   # e.g. Step 3 was not run yet
    m = load_json(metrics_path)
    print(f"{model_name:<12}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}"
          f"{m['train_time_s']:>10.1f}{m['latency_avg_ms']:>12.2f}{m['model_size_mb']:>9.1f}")
print("\nlatency_ms = time for ONE message on this laptop's CPU (embedding models include encoding).")
