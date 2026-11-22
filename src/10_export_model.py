"""
Step 10 (part 1): Export ("promote") the chosen model into the API folder
=========================================================================

Goal: copy the model we decided to deploy into app/artifacts/, together with a small
"model card" file (model_info.json) that records exactly what is being deployed.

Why a separate export step?
    - models/ holds EVERY experiment and is not committed to Git.
    - app/artifacts/ holds ONLY the model that is live, and IS committed, because Render
      builds the app from the GitHub repo.
    This is a tiny version of a "model registry": Candidate (in models/) -> promoted
    (Active, in app/artifacts/) with a version and its scores recorded.

DECISION (Claude's default, review it): deploy TF-IDF + Logistic Regression.
    - It fits Render's 512 MB free tier easily (sentence-transformers + PyTorch do not).
    - It answers in about a millisecond.
    - Step 9 shows how much accuracy we give up compared with bigger models.

Run:  python src/10_export_model.py   (after Steps 3, 6 and 9)
"""

import shutil
from datetime import date

import sklearn

from common import METRICS_DIR, MODELS_DIR, REPORTS_DIR, ROOT_DIR, load_json, print_header, save_json

APP_ARTIFACTS_DIR = ROOT_DIR / "app" / "artifacts"
MODEL_NAME = "tfidf_lr"

print_header("EXPORT MODEL FOR DEPLOYMENT")
source_path = MODELS_DIR / f"{MODEL_NAME}.joblib"
target_path = APP_ARTIFACTS_DIR / "intent_model.joblib"

APP_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source_path, target_path)

threshold_info = load_json(REPORTS_DIR / "threshold.json")
val_metrics = load_json(METRICS_DIR / f"{MODEL_NAME}.json")

# The test score comes from Step 9 (if it has been run).
test_macro_f1 = None
final_results_path = REPORTS_DIR / "step9_final_results.json"
if final_results_path.exists():
    for row in load_json(final_results_path)["intent_models"]:
        if row["model"] == MODEL_NAME:
            test_macro_f1 = row["test_macro_f1"]

model_info = {
    "model_name": MODEL_NAME,
    "model_version": f"{MODEL_NAME}-{date.today().isoformat()}",
    "confidence_threshold": threshold_info["chosen_threshold"],
    # joblib files can only be loaded safely with the SAME scikit-learn version.
    # That is why requirements.txt pins scikit-learn to exactly this version.
    "sklearn_version": sklearn.__version__,
    "trained_on": "Banking77 train split (90% of the official train set)",
    "val_macro_f1": val_metrics["macro_f1"],
    "test_macro_f1": test_macro_f1,
    "exported_on": date.today().isoformat(),
}
save_json(model_info, APP_ARTIFACTS_DIR / "model_info.json")

size_mb = target_path.stat().st_size / (1024 * 1024)
print(f"Copied  : models/{MODEL_NAME}.joblib -> app/artifacts/intent_model.joblib ({size_mb:.2f} MB)")
print("Written : app/artifacts/model_info.json")
for key, value in model_info.items():
    print(f"  {key:<22}: {value}")
print("\nCommit app/artifacts/ so Render can load the model.")
