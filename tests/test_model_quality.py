"""
Regression test: FAIL if the deployed model's quality drops below a set level.

WHY: if someone retrains the model with a bug (wrong data, wrong settings), this test
fails BEFORE the bad model is deployed. In a team, this runs automatically on every change
(continuous integration, "CI").

The minimum values are set a little BELOW the scores we measured in Step 9, so normal
small changes pass, but a real drop fails.
"""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from app.router import MODEL

TEST_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "test.csv"

# Measured on the test set in Step 9: macro-F1 0.8965, accuracy 0.8961
# (see reports/step9_final_results.json), minus a margin of ~1.5 points.
MIN_MACRO_F1 = 0.88
MIN_ACCURACY = 0.88


@pytest.mark.skipif(not TEST_CSV.exists(), reason="test data not found: run src/01_load_and_eda.py")
def test_model_quality_on_test_set():
    test_df = pd.read_csv(TEST_CSV)
    predictions = MODEL.predict(test_df["text"])

    macro_f1 = f1_score(test_df["intent"], predictions, average="macro")
    accuracy = accuracy_score(test_df["intent"], predictions)

    assert macro_f1 >= MIN_MACRO_F1, f"macro-F1 dropped to {macro_f1:.4f}"
    assert accuracy >= MIN_ACCURACY, f"accuracy dropped to {accuracy:.4f}"
