"""
Step 9: Evaluation framework
============================

Goal: measure everything HONESTLY, once, on data no model or setting was tuned on.

What this script does:
    A. Final TEST-set scores for every intent model (the first and only time we use test).
       Plus the deployed model WITH its confidence threshold (coverage + accuracy).
    B. RAG evaluation on a hand-written golden set (eval/faq_golden_set.csv):
         - retrieval hit@1 / hit@3 for TF-IDF retrieval (deployed) vs MiniLM embeddings
         - answer correctness, correct refusals, wrong refusals
         - the score gap between answerable and unanswerable questions (to set the refusal threshold)
    C. Saves a results table (reports/results_table.md) for the README.
    The automatic regression tests live in tests/ and run with: pytest

Run:  python src/09_evaluate.py

Definitions:
    - Offline evaluation: measuring the system on a fixed saved dataset before release
      (as opposed to online monitoring of real traffic after release).
    - Golden test set: a small, carefully hand-checked set of questions with known correct
      answers. It is the "answer key" you re-run after every change.
    - hit@k: the share of questions where the correct document is among the top k
      retrieved chunks. hit@1 = the best chunk is right; hit@3 = right one is in the top 3.
    - Faithfulness: does the answer only say things that are in the retrieved text?
    - Answer relevance: does the answer actually address the question?
    - LLM-as-judge: using an LLM to grade answers for faithfulness/relevance. It scales,
      but the judge can be biased (prefers long answers, its own style) and can be wrong,
      so you check a sample by hand. Tools like RAGAS and DeepEval package these metrics.
      Here we use a simpler, deterministic check: the expected key phrase must appear.
    - Regression test: an automatic test that fails if quality drops below a set level,
      so a bad change is caught before it is deployed.
"""

import os
import sys
import time
from collections import Counter

import faiss
import joblib
import numpy as np
import pandas as pd

from common import (
    EVAL_DIR,
    METRICS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    ROOT_DIR,
    evaluate_predictions,
    load_json,
    load_splits,
    log_mlflow_run,
    print_header,
    save_json,
)

sys.path.append(str(ROOT_DIR))   # lets Python find the app/ folder
from app.rag import CHUNKS, MIN_RETRIEVAL_SCORE, answer_question, retrieve

# Rate-limit pacing for the 30 RAG answers (same idea as Step 8). One grounded answer uses
# roughly 1,000 tokens (prompt + context + thinking + answer). Without pacing, Groq would
# answer "429 Too Many Requests" and the chain would switch to the slow local fallback,
# mixing two different models in one evaluation.
TOKENS_PER_MINUTE_LIMIT = int(os.getenv("LLM_TOKENS_PER_MINUTE", "0"))
ESTIMATED_TOKENS_PER_ANSWER = 1000

_, _, test_df = load_splits()
y_test = test_df["intent"].to_numpy()

# ---------------------------------------------------------------------------
# A. Final test-set scores for every model
# ---------------------------------------------------------------------------
print_header("A. FINAL TEST-SET RESULTS (3,080 queries, 40 per intent)")
test_results = {}

# 1) TF-IDF + Logistic Regression
tfidf_model = joblib.load(MODELS_DIR / "tfidf_lr.joblib")
test_results["tfidf_lr"] = evaluate_predictions(y_test, tfidf_model.predict(test_df["text"]))

# 2) and 3) MiniLM embeddings + LR / XGBoost (embeddings were cached in Step 4)
test_embeddings = np.load(MODELS_DIR / "embeddings" / "test.npy")
minilm_lr = joblib.load(MODELS_DIR / "minilm_lr.joblib")
test_results["minilm_lr"] = evaluate_predictions(y_test, minilm_lr.predict(test_embeddings))

xgb_model = joblib.load(MODELS_DIR / "minilm_xgb.joblib")
label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
xgb_predictions = label_encoder.inverse_transform(xgb_model.predict(test_embeddings))
test_results["minilm_xgb"] = evaluate_predictions(y_test, xgb_predictions)

# 4) DistilBERT (only if Step 5 was run)
distilbert_dir = MODELS_DIR / "distilbert"
if distilbert_dir.exists():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(distilbert_dir)
    bert_model = AutoModelForSequenceClassification.from_pretrained(distilbert_dir)
    bert_model.eval()   # evaluation mode: switches off dropout (random "forgetting" used in training)

    texts = test_df["text"].tolist()
    bert_predictions = []
    print("Running DistilBERT on the test set (about a minute on CPU)...")
    with torch.no_grad():   # no gradients needed for prediction -> faster, less memory
        for start in range(0, len(texts), 64):   # 64 texts at a time ("batches")
            batch = tokenizer(texts[start:start + 64], padding=True, truncation=True,
                              max_length=64, return_tensors="pt")
            logits = bert_model(**batch).logits
            for label_id in logits.argmax(dim=1).tolist():
                bert_predictions.append(bert_model.config.id2label[label_id])
    test_results["distilbert"] = evaluate_predictions(y_test, bert_predictions)
else:
    print("(DistilBERT not found in models/distilbert: run Step 5 to include it.)")

# Add latency, size and validation score from the step metrics files.
print(f"\n{'model':<12}{'test_acc':>10}{'test_F1':>9}{'val_F1':>8}{'latency_ms':>12}{'size_MB':>9}")
table_rows = []
for model_name, metrics in test_results.items():
    step_metrics = load_json(METRICS_DIR / f"{model_name}.json")
    size = step_metrics.get("model_size_mb", "")
    row = {
        "model": model_name,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "val_macro_f1": step_metrics["macro_f1"],
        "latency_avg_ms": step_metrics["latency_avg_ms"],
        "model_size_mb": size,
    }
    table_rows.append(row)
    print(f"{model_name:<12}{row['test_accuracy']:>10.4f}{row['test_macro_f1']:>9.4f}"
          f"{row['val_macro_f1']:>8.4f}{row['latency_avg_ms']:>12.2f}{str(size):>9}")
print("(If test_F1 is close to val_F1, our validation set gave honest estimates.)")

# The deployed model WITH the threshold from Step 6.
threshold = load_json(REPORTS_DIR / "threshold.json")["chosen_threshold"]
test_probabilities = tfidf_model.predict_proba(test_df["text"])
test_confidence = test_probabilities.max(axis=1)
test_predicted = tfidf_model.classes_[test_probabilities.argmax(axis=1)]
automated = test_confidence >= threshold
deployed_summary = {
    "threshold": threshold,
    "coverage": round(float(automated.mean()), 4),
    "auto_accuracy": round(float((test_predicted[automated] == y_test[automated]).mean()), 4),
    "handoff_rate": round(float(1 - automated.mean()), 4),
}
print(f"\nDeployed model (tfidf_lr) with threshold {threshold} on TEST:")
print(f"  bot handles {deployed_summary['coverage'] * 100:.1f}% of calls, "
      f"{deployed_summary['auto_accuracy'] * 100:.1f}% of those correctly; "
      f"{deployed_summary['handoff_rate'] * 100:.1f}% go to a human")

# ---------------------------------------------------------------------------
# B. RAG evaluation on the golden set
# ---------------------------------------------------------------------------
print_header("B. RAG EVALUATION (eval/faq_golden_set.csv)")
golden_df = pd.read_csv(EVAL_DIR / "faq_golden_set.csv", keep_default_na=False)
answerable_df = golden_df[golden_df["answerable"] == "yes"]
unanswerable_df = golden_df[golden_df["answerable"] == "no"]
print(f"{len(golden_df)} questions: {len(answerable_df)} answerable, {len(unanswerable_df)} not in the docs")


def normalize_text(text: str) -> str:
    """Lowercase and unify look-alike characters before comparing text.

    LLMs sometimes write special Unicode characters that LOOK the same but are different
    characters: "6‑digit" (non-breaking hyphen U+2011) is not equal to "6-digit".
    Without this, a CORRECT answer would be counted as wrong: exact-match metrics are brittle.
    """
    text = text.lower()
    for dash in ["‐", "‑", "‒", "–", "—"]:   # different hyphens/dashes
        text = text.replace(dash, "-")
    text = text.replace("’", "'").replace(" ", " ")          # curly apostrophe, no-break space
    return text


def hit_at_k(retrieved_docs: list[list[str]], expected_docs: list[str], k: int) -> float:
    """Share of questions whose expected document is in the top k retrieved docs."""
    hits = 0
    for docs, expected in zip(retrieved_docs, expected_docs):
        if expected in docs[:k]:
            hits += 1
    return hits / len(expected_docs)


# B1. TF-IDF retrieval (what the deployed API uses)
tfidf_retrieved = [[chunk["doc"] for chunk in retrieve(question, top_k=3)]
                   for question in answerable_df["question"]]
tfidf_hit1 = hit_at_k(tfidf_retrieved, answerable_df["expected_doc"].tolist(), 1)
tfidf_hit3 = hit_at_k(tfidf_retrieved, answerable_df["expected_doc"].tolist(), 3)

# B2. MiniLM embedding retrieval (better at meaning, but needs PyTorch: too big for 512 MB)
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
chunk_vectors = encoder.encode([chunk["text"] for chunk in CHUNKS], normalize_embeddings=True)
minilm_index = faiss.IndexFlatIP(chunk_vectors.shape[1])
minilm_index.add(np.asarray(chunk_vectors, dtype="float32"))
question_vectors = encoder.encode(answerable_df["question"].tolist(), normalize_embeddings=True)
_, minilm_ids = minilm_index.search(np.asarray(question_vectors, dtype="float32"), 3)
minilm_retrieved = [[CHUNKS[i]["doc"] for i in row] for row in minilm_ids]
minilm_hit1 = hit_at_k(minilm_retrieved, answerable_df["expected_doc"].tolist(), 1)
minilm_hit3 = hit_at_k(minilm_retrieved, answerable_df["expected_doc"].tolist(), 3)

print(f"\n{'retriever':<22}{'hit@1':>8}{'hit@3':>8}")
print(f"{'TF-IDF (deployed)':<22}{tfidf_hit1:>8.2f}{tfidf_hit3:>8.2f}")
print(f"{'MiniLM embeddings':<22}{minilm_hit1:>8.2f}{minilm_hit3:>8.2f}")

# B3. Full answers: correctness and refusals
print(f"\nAnswering all {len(golden_df)} questions (refusal threshold = {MIN_RETRIEVAL_SCORE})...")
rows = []
start = time.perf_counter()
for _, item in golden_df.iterrows():
    result = answer_question(item["question"])
    phrase_found = False
    if item["expected_phrase"]:
        phrase_found = normalize_text(item["expected_phrase"]) in normalize_text(result["answer"])
    rows.append({
        "question": item["question"],
        "answerable": item["answerable"] == "yes",
        "refused": result["refused"],
        "phrase_found": phrase_found,
        "top_score": result["top_score"],
        "used_llm": result["used_llm"],
        "llm_provider": result["llm_provider"],
    })
    if result["used_llm"] and TOKENS_PER_MINUTE_LIMIT > 0:
        time.sleep(ESTIMATED_TOKENS_PER_ANSWER / TOKENS_PER_MINUTE_LIMIT * 60)
answer_time_s = time.perf_counter() - start
results_df = pd.DataFrame(rows)
provider_counts = dict(Counter(results_df["llm_provider"]))

answerable_results = results_df[results_df["answerable"]]
unanswerable_results = results_df[~results_df["answerable"]]
correct_answers = (answerable_results["phrase_found"] & ~answerable_results["refused"]).mean()
wrong_refusals = answerable_results["refused"].mean()
correct_refusals = unanswerable_results["refused"].mean()

print(f"Answered by                  : {provider_counts}  (primary=Groq, fallback=Ollama, none=no LLM)")
print(f"Correct answers (answerable) : {correct_answers * 100:.0f}%  (expected key phrase found)")
print(f"Wrong refusals (answerable)  : {wrong_refusals * 100:.0f}%")
print(f"Correct refusals (not in docs): {correct_refusals * 100:.0f}%")
print(f"Total time for {len(golden_df)} answers  : {answer_time_s:.0f} s (includes rate-limit pauses)")

# B4. Score gap -> is the refusal threshold in a good place?
print("\nTop retrieval score (TF-IDF cosine):")
print(f"  answerable questions   : min {answerable_results['top_score'].min():.3f}, "
      f"median {answerable_results['top_score'].median():.3f}")
print(f"  unanswerable questions : max {unanswerable_results['top_score'].max():.3f}, "
      f"median {unanswerable_results['top_score'].median():.3f}")
print("A good threshold sits between the unanswerable max and the answerable min.")
print("Caution: we look at the SAME 30 questions we report on, so this is slightly optimistic;")
print("a bigger golden set split into 'tune' and 'report' halves would be cleaner.")

for _, row in results_df.iterrows():
    if (row["answerable"] and (row["refused"] or not row["phrase_found"])) or (not row["answerable"] and not row["refused"]):
        print(f"  MISS: '{row['question']}'  refused={row['refused']}  "
              f"phrase_found={row['phrase_found']}  score={row['top_score']:.3f}")

# ---------------------------------------------------------------------------
# C. Save everything
# ---------------------------------------------------------------------------
rag_summary = {
    "golden_questions": len(golden_df),
    "tfidf_hit_at_1": round(tfidf_hit1, 4),
    "tfidf_hit_at_3": round(tfidf_hit3, 4),
    "minilm_hit_at_1": round(minilm_hit1, 4),
    "minilm_hit_at_3": round(minilm_hit3, 4),
    "correct_answer_rate": round(float(correct_answers), 4),
    "wrong_refusal_rate": round(float(wrong_refusals), 4),
    "correct_refusal_rate": round(float(correct_refusals), 4),
    "answered_by": provider_counts,
    "refusal_threshold": MIN_RETRIEVAL_SCORE,
}
save_json({"intent_models": table_rows, "deployed_with_threshold": deployed_summary, "rag": rag_summary},
          REPORTS_DIR / "step9_final_results.json")

markdown_lines = [
    "| Model | Test accuracy | Test macro-F1 | Val macro-F1 | Latency / message (ms, laptop CPU) | Size (MB) |",
    "|---|---|---|---|---|---|",
]
for row in table_rows:
    markdown_lines.append(
        f"| {row['model']} | {row['test_accuracy']:.4f} | {row['test_macro_f1']:.4f} | "
        f"{row['val_macro_f1']:.4f} | {row['latency_avg_ms']:.2f} | {row['model_size_mb']} |"
    )
(REPORTS_DIR / "results_table.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

final_metrics = {}
for row in table_rows:
    final_metrics[f"test_macro_f1_{row['model']}"] = row["test_macro_f1"]
for key, value in deployed_summary.items():
    final_metrics[f"deployed_{key}"] = value
for key, value in rag_summary.items():
    if isinstance(value, float):   # only numbers can be MLflow metrics
        final_metrics[key] = value
log_mlflow_run(
    run_name="final_test_evaluation",
    params={"golden_questions": len(golden_df), "refusal_threshold": MIN_RETRIEVAL_SCORE},
    metrics=final_metrics,
)

print("\nSaved: reports/step9_final_results.json, reports/results_table.md (+ MLflow run)")
print("Next: run the regression tests with:  pytest")
