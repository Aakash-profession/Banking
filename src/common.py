"""
common.py: settings and small helper functions shared by all step scripts.

WHY a shared file: every step needs the same folder paths, the same data splits and the
same way of measuring results. Writing them ONCE here means:
    - one place to change a setting
    - no copy-paste mistakes between steps
    - every model is measured in exactly the same way (a fair comparison)

The step scripts use it like this:
    from common import load_splits, evaluate_predictions
(This works because Python looks for imports in the folder of the script being run, src/.)
"""

import json
import os
import time
from pathlib import Path
from typing import Callable

# MLflow 3.x prints a hint meant for AI coding assistants every time it is imported.
# This setting hides it so our output stays readable. (It must be set BEFORE "import mlflow".)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# ---------------------------------------------------------------------------
# Folder paths
# ---------------------------------------------------------------------------
# Path(__file__)          = this file:            .../ivr-reference/src/common.py
# .resolve().parent       = the folder it is in:  .../ivr-reference/src
# .parent again           = the repo root:        .../ivr-reference
# Building paths from the repo root means the scripts work no matter which folder
# you run them from.
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                # the original downloaded CSV files
PROCESSED_DIR = DATA_DIR / "processed"    # our train / val / test splits
MODELS_DIR = ROOT_DIR / "models"          # trained models (not committed to Git)
REPORTS_DIR = ROOT_DIR / "reports"        # charts and result files (committed)
METRICS_DIR = REPORTS_DIR / "metrics"     # one JSON file of results per model
EVAL_DIR = ROOT_DIR / "eval"              # hand-written test sets (out-of-scope, FAQ golden set)

# A "random seed" makes random things repeatable. With the same seed, the "random"
# validation split is the same every time, so results can be compared fairly.
RANDOM_SEED = 42

# All model runs go into ONE MLflow experiment, so we can compare them side by side.
MLFLOW_EXPERIMENT = "ivr-intent-router"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the train / validation / test CSV files created by Step 1.

    Each DataFrame has two columns:
        text   -> the customer message, e.g. "my card hasn't arrived"
        intent -> the correct label,    e.g. "card_arrival"
    """
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            "data/processed/train.csv not found. Run Step 1 first: python src/01_load_and_eda.py"
        )

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Measuring results
# ---------------------------------------------------------------------------
def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Return the 3 headline metrics we use for every model.

    accuracy    = correct predictions / all predictions.
    macro_f1    = F1 score of EACH intent, then the simple average.
                  Every intent counts equally, so a model cannot hide bad performance
                  on a few intents behind good performance on the rest.
    weighted_f1 = like macro_f1, but intents with more examples count more.

    F1 for one intent = the balance between
        precision ("when the model says card_arrival, how often is it right?") and
        recall    ("of all real card_arrival messages, how many did the model find?").
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
    }


def top_confused_pairs(y_true: list[str], y_pred: list[str], n: int = 10) -> list[dict]:
    """Find the n (true intent -> predicted intent) mistakes that happen most often.

    A confusion matrix is a table: row = the TRUE intent, column = the PREDICTED intent,
    cell = how many messages. The diagonal holds the correct predictions; everything
    off the diagonal is a mistake. We list the biggest off-diagonal cells.
    """
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    mistakes = []
    for row in range(len(labels)):
        for col in range(len(labels)):
            if row != col and matrix[row, col] > 0:
                mistakes.append(
                    {"true": labels[row], "predicted": labels[col], "count": int(matrix[row, col])}
                )

    # Sort by count, biggest first, and keep the first n.
    mistakes.sort(key=lambda item: item["count"], reverse=True)
    return mistakes[:n]


def measure_latency_ms(predict_one: Callable[[str], object], texts: list[str]) -> dict[str, float]:
    """Time how long ONE prediction takes, the way a live IVR call would use the model.

    predict_one = a function that takes ONE text and returns a prediction.
    We call it once per text and record each time. We report:
        avg = the average time
        p95 = 95% of calls were faster than this (shows the slow "worst normal" case,
              which matters more to callers than the average).
    """
    predict_one(texts[0])  # warm-up call: the first call is often slow (loading, caching)

    timings_ms = []
    for text in texts:
        start = time.perf_counter()          # a precise stopwatch
        predict_one(text)
        timings_ms.append((time.perf_counter() - start) * 1000)

    return {
        "latency_avg_ms": round(float(np.mean(timings_ms)), 3),
        "latency_p95_ms": round(float(np.percentile(timings_ms, 95)), 3),
    }


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------
def save_json(data: dict, path: Path) -> None:
    """Save a Python dict as a readable JSON file (creating the folder if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def setup_mlflow() -> None:
    """Point MLflow at <repo root>/mlflow.db and select our experiment.

    WHY set the location explicitly: by default MLflow saves to mlflow.db in whatever
    folder you happen to run from. Fixing it to the repo root means every script
    writes to the same place, and `mlflow ui` (run from the repo root) finds all runs.
    """
    db_path = (ROOT_DIR / "mlflow.db").as_posix()   # as_posix = forward slashes, needed in a URI
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def print_header(title: str) -> None:
    """Print a clear section title so the output is easy to read."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
