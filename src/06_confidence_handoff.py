"""
Step 6: Confidence threshold + human handoff + out-of-scope queries
===================================================================

Goal: decide WHEN the bot should handle a call by itself and when it should transfer
the caller to a human agent. A wrong automatic route is worse than an honest handoff.

What this script does:
    1. Uses the TF-IDF model (our deployable model, see Step 4/10) to predict on validation.
    2. For thresholds 0.00 ... 0.95 it measures:
         coverage      = % of calls the bot handles itself (confidence >= threshold)
         auto_accuracy = accuracy on ONLY those automated calls
         cost          = a simple business cost (wrong routes are expensive, handoffs cost less)
    3. Picks the threshold with the LOWEST cost, and plots the trade-off.
    4. Checks calibration: when the model says 80% sure, is it right ~80% of the time?
    5. Sends 30 off-topic questions ("what's the weather?") and checks they go to a human.

Run:  python src/06_confidence_handoff.py   (needs Step 3's model)

Definitions:
    - Confidence: the model's predicted probability for its top intent.
      predict_proba gives 77 probabilities that sum to 1; confidence = the biggest one.
    - Threshold: the minimum confidence needed to let the bot act alone.
    - Coverage vs accuracy trade-off: a HIGHER threshold means fewer automated calls
      (lower coverage) but those calls are more often right (higher accuracy).
    - Calibration: whether confidence matches reality. A calibrated model that says
      "70% sure" on 100 calls is right on about 70 of them.
    - Out-of-scope (OOS): a message that matches NONE of our 77 intents. The model still
      has to pick one of the 77, so the only defence is low confidence -> human.
"""

import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import EVAL_DIR, MODELS_DIR, REPORTS_DIR, ROOT_DIR, load_splits, print_header, save_json

# The saved pipeline's first step calls app.preprocessing.stem_texts. When joblib loads the
# model, Python must be able to import that function, so the repo root must be on the path.
# (Without this line: "ModuleNotFoundError: No module named 'app'".)
sys.path.append(str(ROOT_DIR))

# DECISION (Claude's default, review it): the business costs.
# Assumption: a WRONG automatic route costs 4x more than sending the call to a human,
# because the caller gets the wrong answer, calls back or gets transferred twice,
# and is more likely to complain. A human handoff costs about 1 unit (a few agent minutes).
# A correct automatic route costs ~0. Change these numbers and the threshold moves.
COST_HUMAN_HANDOFF = 1.0
COST_WRONG_ROUTE = 4.0

THRESHOLDS = [round(value, 2) for value in np.arange(0.0, 0.96, 0.05)]

_, val_df, _ = load_splits()
model = joblib.load(MODELS_DIR / "tfidf_lr.joblib")

# ---------------------------------------------------------------------------
# 1. Confidence for every validation message
# ---------------------------------------------------------------------------
probabilities = model.predict_proba(val_df["text"])        # shape: (rows, 77)
confidence = probabilities.max(axis=1)                      # biggest probability per row
predicted = model.classes_[probabilities.argmax(axis=1)]    # the intent with that probability
is_correct = predicted == val_df["intent"].to_numpy()

print_header("1. CONFIDENCE ON VALIDATION")
print(f"Overall accuracy (no threshold)        : {is_correct.mean():.4f}")
print(f"Average confidence when RIGHT          : {confidence[is_correct].mean():.3f}")
print(f"Average confidence when WRONG          : {confidence[~is_correct].mean():.3f}")
print("(The gap between these two is what makes a threshold useful.)")

# ---------------------------------------------------------------------------
# 2. Coverage vs accuracy vs cost for each threshold
# ---------------------------------------------------------------------------
print_header("2. THRESHOLD TABLE")
print(f"{'threshold':>9}{'coverage':>10}{'auto_acc':>10}{'handoff':>9}{'wrong':>8}{'cost/100':>10}")

rows = []
for threshold in THRESHOLDS:
    automated = confidence >= threshold          # True = the bot handles this call
    coverage = automated.mean()                  # share of calls automated
    handoff_rate = 1 - coverage                  # share of calls sent to a human
    # Accuracy on the automated calls only. (If nothing is automated, call it 1.0.)
    auto_accuracy = is_correct[automated].mean() if automated.any() else 1.0
    # Wrong routes = automated AND incorrect, as a share of ALL calls.
    wrong_rate = (automated & ~is_correct).mean()
    cost_per_100_calls = 100 * (handoff_rate * COST_HUMAN_HANDOFF + wrong_rate * COST_WRONG_ROUTE)

    rows.append({
        "threshold": threshold,
        "coverage": round(float(coverage), 4),
        "auto_accuracy": round(float(auto_accuracy), 4),
        "handoff_rate": round(float(handoff_rate), 4),
        "wrong_route_rate": round(float(wrong_rate), 4),
        "cost_per_100_calls": round(float(cost_per_100_calls), 2),
    })
    print(f"{threshold:>9.2f}{coverage:>10.3f}{auto_accuracy:>10.3f}{handoff_rate:>9.3f}"
          f"{wrong_rate:>8.3f}{cost_per_100_calls:>10.1f}")

best_row = min(rows, key=lambda row: row["cost_per_100_calls"])
chosen_threshold = best_row["threshold"]
print(f"\nChosen threshold (lowest cost): {chosen_threshold}")
print(f"  -> bot handles {best_row['coverage'] * 100:.1f}% of calls with "
      f"{best_row['auto_accuracy'] * 100:.1f}% accuracy; "
      f"{best_row['handoff_rate'] * 100:.1f}% go to a human.")
# Theory check: with costs 1 (human) vs 4 (wrong), automating is worth it when
# P(correct) > 1 - 1/4 = 0.75. If the model were perfectly calibrated, the best
# threshold would be near 0.75. Real models are not perfectly calibrated, which is
# why we choose the threshold from measured data instead of theory.

# ---------------------------------------------------------------------------
# 3. Plot the trade-off
# ---------------------------------------------------------------------------
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(THRESHOLDS, [row["coverage"] for row in rows], marker="o", label="coverage (% automated)")
plt.plot(THRESHOLDS, [row["auto_accuracy"] for row in rows], marker="o", label="accuracy on automated calls")
plt.axvline(chosen_threshold, color="gray", linestyle="--", label=f"chosen threshold = {chosen_threshold}")
plt.xlabel("Confidence threshold")
plt.ylabel("Rate")
plt.title("Coverage vs accuracy (validation, TF-IDF + LR)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "step6_threshold_curve.png", dpi=120)
plt.close()
print("Saved chart: reports/step6_threshold_curve.png")

# ---------------------------------------------------------------------------
# 4. Calibration check (reliability table)
# ---------------------------------------------------------------------------
print_header("4. CALIBRATION: does confidence match reality?")
print(f"{'confidence bin':<16}{'calls':>7}{'avg confidence':>16}{'actual accuracy':>17}")
bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
for low, high in zip(bin_edges[:-1], bin_edges[1:]):
    in_bin = (confidence >= low) & (confidence < high)
    if not in_bin.any():
        continue
    print(f"{low:.1f} - {min(high, 1.0):.1f}       {in_bin.sum():>7}"
          f"{confidence[in_bin].mean():>16.3f}{is_correct[in_bin].mean():>17.3f}")
print("If 'actual accuracy' is HIGHER than 'avg confidence', the model is UNDER-confident")
print("(common for regularized Logistic Regression with many classes).")

# ---------------------------------------------------------------------------
# 5. Out-of-scope queries
# ---------------------------------------------------------------------------
print_header("5. OUT-OF-SCOPE QUERIES (should go to a human)")
oos_texts = (EVAL_DIR / "out_of_scope.txt").read_text(encoding="utf-8").strip().splitlines()
oos_probabilities = model.predict_proba(oos_texts)
oos_confidence = oos_probabilities.max(axis=1)
oos_predicted = model.classes_[oos_probabilities.argmax(axis=1)]
oos_to_human = oos_confidence < chosen_threshold

print(f"Off-topic queries sent to a human: {oos_to_human.sum()} of {len(oos_texts)} "
      f"({oos_to_human.mean() * 100:.0f}%)")
print(f"Highest confidence on an off-topic query: {oos_confidence.max():.3f}")
slipped = [
    (text, intent, conf)
    for text, intent, conf, to_human in zip(oos_texts, oos_predicted, oos_confidence, oos_to_human)
    if not to_human
]
if slipped:
    print("\nOff-topic queries that the bot would WRONGLY handle:")
    for text, intent, conf in slipped:
        print(f"  {conf:.2f}  '{text}'  -> {intent}")

save_json(
    {
        "model": "tfidf_lr",
        "chosen_threshold": chosen_threshold,
        "cost_human_handoff": COST_HUMAN_HANDOFF,
        "cost_wrong_route": COST_WRONG_ROUTE,
        "val_at_threshold": best_row,
        "oos_handoff_rate": round(float(oos_to_human.mean()), 4),
        "oos_max_confidence": round(float(oos_confidence.max()), 4),
        "table": rows,
    },
    REPORTS_DIR / "threshold.json",
)
print("\nSaved: reports/threshold.json (the threshold is used by Step 7 and the API)")
