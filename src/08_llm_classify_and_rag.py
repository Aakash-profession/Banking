"""
Step 8: The LLM layer
=====================
    (a) Can an LLM classify intents with just a prompt (zero-shot / few-shot)?
        Compare its accuracy, latency and cost with our trained models.
    (b) RAG: answer FAQ questions from docs/faq/ with sources, and refuse when the
        answer is not in the documents.

Run:  python src/08_llm_classify_and_rag.py        (uses 200 test queries)
      python src/08_llm_classify_and_rag.py 20     (only 20 queries: for slow local models)

Part (a) needs an LLM configured in .env (see .env.example). Without one it is skipped.
Part (b) always runs: without an LLM it uses the "extractive" fallback.

Definitions:
    - Prompt engineering: writing the instructions for an LLM carefully, so it gives the
      output you need in the format you need.
    - Zero-shot: the prompt only DESCRIBES the task ("pick one of these 77 intents").
    - Few-shot: the prompt also includes EXAMPLES ("'I am still waiting on my card'
      -> card_arrival"). Examples usually improve accuracy, but make the prompt longer
      (more tokens = more cost and latency).
    - Structured output: asking for a fixed machine-readable format (here JSON like
      {"intent": "card_arrival"}) and VALIDATING it in code. Never trust the format blindly.
    - Temperature 0: the model always picks its most likely answer -> repeatable results.
    - Tokens: LLMs read and write "tokens" (pieces of words). You pay per token, so a long
      few-shot prompt sent on every call costs real money at IVR scale.
"""

import json
import os
import sys
import time

import joblib
import numpy as np

from common import (
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
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
from app.llm import ask_llm, batch_model_name, is_fallback_configured, is_llm_configured, is_primary_configured
from app.rag import answer_question

# sys.argv = the words typed after "python": ["src/08_...py", "20"]. Default: 200 queries.
N_QUERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 200

# Rate-limit pacing. Free tiers limit tokens PER MINUTE (my Groq key: 8,000).
# Instead of hitting the limit and getting "429 Too Many Requests" (and wasting requests
# from the daily quota), we wait after each call in proportion to the tokens it used:
#   a call that used 2,000 tokens with a limit of 8,000/minute -> wait 2000/8000 * 60 = 15 s
# 0 (the default) = no pacing, e.g. for a local Ollama model.
TOKENS_PER_MINUTE_LIMIT = int(os.getenv("LLM_TOKENS_PER_MINUTE", "0"))

train_df, _, test_df = load_splits()
all_intents = sorted(train_df["intent"].unique())

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an intent classifier for a bank's phone line. Choose exactly ONE intent from "
    "the list for the customer's message. Reply with JSON only, in this exact format: "
    '{"intent": "<intent_name>"}'
)

# Zero-shot: only the list of intent names.
ZERO_SHOT_INTENT_LIST = "\n".join(all_intents)

# Few-shot: each intent name plus ONE real example message from the training set.
few_shot_lines = []
for intent in all_intents:
    example = train_df[train_df["intent"] == intent]["text"].iloc[0]
    few_shot_lines.append(f'{intent}: e.g. "{example}"')
FEW_SHOT_INTENT_LIST = "\n".join(few_shot_lines)


def build_user_prompt(message: str, few_shot: bool) -> str:
    if few_shot:
        return f"Intents, each with one example:\n{FEW_SHOT_INTENT_LIST}\n\nCustomer message: {message}\nJSON:"
    return f"Intents:\n{ZERO_SHOT_INTENT_LIST}\n\nCustomer message: {message}\nJSON:"


def parse_intent(reply_text: str) -> str:
    """Pull the intent out of the LLM's reply and VALIDATE it.

    Returns the intent name, or "INVALID" if the reply is not JSON or names an intent
    that does not exist. (In the live system, INVALID would go to a human agent.)
    """
    start = reply_text.find("{")
    end = reply_text.rfind("}")
    if start == -1 or end <= start:
        return "INVALID"
    try:
        data = json.loads(reply_text[start:end + 1])
    except json.JSONDecodeError:
        return "INVALID"
    intent = str(data.get("intent", "")).strip()
    return intent if intent in all_intents else "INVALID"


# ---------------------------------------------------------------------------
# (a) LLM classification vs trained models
# ---------------------------------------------------------------------------
print_header(f"(a) LLM INTENT CLASSIFICATION ON {N_QUERIES} TEST QUERIES")

if not is_llm_configured():
    print("SKIPPED: no LLM configured. Copy .env.example to .env and fill in LLM_API_KEY + LLM_MODEL.")
    llm_results = {}
else:
    llm_model = batch_model_name()
    # batch_mode=True below: ONE model for the whole experiment (no switching to the
    # fallback halfway), with patient retries when the free tier's rate limit is hit.
    print(f"LLM model: {llm_model}")
    # Random but repeatable sample of test queries (same seed = same sample every run).
    sample_df = test_df.sample(n=N_QUERIES, random_state=RANDOM_SEED)
    y_true = sample_df["intent"].tolist()

    llm_results = {}
    for mode_name, few_shot in [("llm_zero_shot", False), ("llm_few_shot", True)]:
        predictions, latencies, prompt_tokens, completion_tokens, costs = [], [], [], [], []
        for number, message in enumerate(sample_df["text"], start=1):
            try:
                # max_tokens=300: the JSON answer is ~10 tokens, but reasoning models
                # (gpt-oss) need room to "think" first, or they return an empty answer.
                reply = ask_llm(SYSTEM_PROMPT, build_user_prompt(message, few_shot),
                                max_tokens=300, batch_mode=True)
                predictions.append(parse_intent(reply["text"]))
                latencies.append(reply["latency_ms"])
                prompt_tokens.append(reply["prompt_tokens"])
                completion_tokens.append(reply["completion_tokens"])
                costs.append(reply["cost_usd"])
                if TOKENS_PER_MINUTE_LIMIT > 0:
                    tokens_used = reply["prompt_tokens"] + reply["completion_tokens"]
                    time.sleep(tokens_used / TOKENS_PER_MINUTE_LIMIT * 60)
            except Exception as error:
                print(f"  call {number} failed: {type(error).__name__}")
                predictions.append("ERROR")
            if number % 10 == 0:
                print(f"  {mode_name}: {number}/{N_QUERIES} done")

        metrics = evaluate_predictions(y_true, predictions)
        llm_results[mode_name] = {
            **metrics,
            "invalid_rate": round(predictions.count("INVALID") / N_QUERIES, 4),
            "error_rate": round(predictions.count("ERROR") / N_QUERIES, 4),
            "latency_avg_ms": round(float(np.mean(latencies)), 1) if latencies else None,
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
            "avg_prompt_tokens": round(float(np.mean(prompt_tokens)), 1) if prompt_tokens else None,
            "avg_completion_tokens": round(float(np.mean(completion_tokens)), 1) if completion_tokens else None,
            "est_cost_per_1000_calls_usd": round(float(np.mean(costs)) * 1000, 4) if costs else None,
        }
        print(f"  {mode_name}: {llm_results[mode_name]}")

    # Trained models on the SAME sample, for a fair comparison.
    tfidf_model = joblib.load(MODELS_DIR / "tfidf_lr.joblib")
    tfidf_metrics = evaluate_predictions(y_true, tfidf_model.predict(sample_df["text"]))

    minilm_lr = joblib.load(MODELS_DIR / "minilm_lr.joblib")
    test_embeddings = np.load(MODELS_DIR / "embeddings" / "test.npy")
    # sample_df.index = the row numbers of the sampled rows in test_df,
    # which are also the row numbers in the cached test embeddings.
    minilm_metrics = evaluate_predictions(y_true, minilm_lr.predict(test_embeddings[sample_df.index]))

    print_header("COMPARISON (same test sample)")
    print(f"{'model':<16}{'accuracy':>10}{'macro_f1':>10}{'latency_ms':>12}{'tokens/call':>13}{'$ per 1k calls':>16}")
    for model_name, m, metrics_file in [("tfidf_lr", tfidf_metrics, "tfidf_lr"),
                                        ("minilm_lr", minilm_metrics, "minilm_lr")]:
        latency = load_json(METRICS_DIR / f"{metrics_file}.json")["latency_avg_ms"]
        print(f"{model_name:<16}{m['accuracy']:>10.3f}{m['macro_f1']:>10.3f}{latency:>12.2f}{'0':>13}{'~0':>16}")
    for mode_name, m in llm_results.items():
        tokens = (m["avg_prompt_tokens"] or 0) + (m["avg_completion_tokens"] or 0)
        cost = m["est_cost_per_1000_calls_usd"]
        print(f"{mode_name:<16}{m['accuracy']:>10.3f}{m['macro_f1']:>10.3f}"
              f"{(m['latency_avg_ms'] or 0):>12.0f}{tokens:>13.0f}{cost:>16.4f}")
    print("(Cost uses LLM_PRICE_INPUT_PER_1M / LLM_PRICE_OUTPUT_PER_1M from .env; 0 = local model.)")

    save_json(
        {"llm_model": llm_model, "n_queries": N_QUERIES, "llm": llm_results,
         "tfidf_lr_same_sample": tfidf_metrics, "minilm_lr_same_sample": minilm_metrics},
        REPORTS_DIR / "step8_llm_vs_classic.json",
    )
    for mode_name, m in llm_results.items():
        log_mlflow_run(
            run_name=mode_name,
            params={"model": mode_name, "llm_model": llm_model, "n_queries": N_QUERIES},
            metrics={key: value for key, value in m.items() if value is not None},
        )
    print("Saved: reports/step8_llm_vs_classic.json (+ MLflow runs)")

# ---------------------------------------------------------------------------
# (b) RAG FAQ answers
# ---------------------------------------------------------------------------
print_header("(b) RAG: FAQ ANSWERS WITH SOURCES")
print(f"Primary LLM configured : {is_primary_configured()}")
print(f"Fallback LLM configured: {is_fallback_configured()}")
print("Chain: primary LLM -> fallback LLM -> extractive answer (best FAQ text, no LLM)")

demo_questions = [
    "How long will my new card take to arrive?",
    "What is the daily limit for taking cash out of an ATM?",
    "Is there a fee if I exchange money on a Saturday?",
    "Can I use Google Pay with my card?",
    "What interest rate do you offer on mortgages?",     # NOT in the docs -> should refuse
    "What's the weather like in London?",                # off-topic -> should refuse
]
for question in demo_questions:
    result = answer_question(question)
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"   sources={result['sources']}  refused={result['refused']}  "
          f"answered_by={result['llm_provider']}  top_score={result['top_score']:.3f}")
