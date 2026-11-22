"""
streamlit_app.py: a simple web UI for the Banking IVR Intent Router.

What it is:
    A "front end". It does NOT load any model itself. It sends the customer's message to our
    FastAPI "back end" (the /route, /answer, /stats and /health endpoints) over HTTP, and shows
    the JSON reply in a friendly way. This is how a real IVR or chat screen would use the API.

        Browser  ──►  Streamlit (this file)  ──HTTP/JSON──►  FastAPI on Render (app/main.py)

Run (from the ivr-reference folder, with the venv switched on):
    streamlit run ui/streamlit_app.py
    then open http://localhost:8501

Definitions:
    - Streamlit: a Python library that turns a normal script into a web page. Every st.something()
      call draws one element (a title, a text box, a button, a chart).
    - Rerun: when you click a button or change a widget, Streamlit runs this WHOLE file again from
      the top. That is why we only call the API inside "if st.button(...)" blocks: otherwise every
      click anywhere would send new requests.
    - requests: a library for sending HTTP requests (GET = read, POST = send data).
"""

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# The deployed API on Render. You can change it in the sidebar, e.g. to http://127.0.0.1:8000
# when you run the API on your own laptop (uvicorn app.main:app --reload).
DEFAULT_API_URL = "https://ivr-intent-router.onrender.com"

# Render's free instance sleeps after 15 minutes without traffic and needs about a minute to
# wake up, so we wait up to 90 seconds before giving up.
API_TIMEOUT_SECONDS = 90

# The result files written by Steps 8 and 9 (shown in the "Model results" tab).
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

EXAMPLE_MESSAGES = [
    "How long does card delivery take?",
    "My card still hasn't arrived after two weeks",
    "I lost my card yesterday, I think it was stolen",
    "Why was I charged twice for the same payment?",
    "What is the daily limit for taking cash out of an ATM?",
    "I want to close my account",
    "What's the weather like in Bangalore today?",
]

EXAMPLE_QUESTIONS = [
    "Is there a fee if I exchange money on a Saturday?",
    "How old do I have to be to open an account?",
    "Which documents can I use to verify my identity?",
    "Can I get a mortgage from Demo Bank?",
]

# Who wrote the answer (the "llm_provider" field from the API) -> a readable label.
PROVIDER_LABELS = {
    "primary": "Groq (primary LLM)",
    "fallback": "Ollama (local fallback LLM)",
    "none": "no LLM (FAQ text read out directly, or no answer needed)",
}


def call_api(api_url: str, method: str, path: str, payload: dict | None = None) -> tuple[dict | None, str]:
    """Call our FastAPI back end.

    Returns (data, error_message):
        success -> (the JSON reply as a dict, "")
        failure -> (None, a message explaining what went wrong)
    We never let an error crash the page: we show it to the user instead.
    """
    url = api_url + path
    try:
        if method == "GET":
            response = requests.get(url, timeout=API_TIMEOUT_SECONDS)
        else:
            response = requests.post(url, json=payload, timeout=API_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as error:
        return None, (
            f"Could not reach the API at {api_url} ({type(error).__name__}). "
            "If you use the Render URL, it may be waking up: wait a minute and try again. "
            "If you use a local URL, start the API first: uvicorn app.main:app --reload"
        )

    if response.status_code != 200:
        # 422 = the request was rejected by validation (e.g. empty text); 500 = server error.
        return None, f"The API answered with HTTP {response.status_code}: {response.text[:300]}"
    return response.json(), ""


def show_route_result(data: dict) -> None:
    """Draw the reply of POST /route."""
    confidence = data["confidence"]
    threshold = data["threshold"]

    # Three numbers side by side.
    column_1, column_2, column_3 = st.columns(3)
    column_1.metric("Intent", data["intent"])
    column_2.metric("Confidence", f"{confidence:.0%}")
    column_3.metric("Latency", f"{data['latency_ms']:.0f} ms")

    # A bar from 0 to 100% for the confidence, with the threshold written next to it.
    st.progress(min(confidence, 1.0), text=f"Confidence {confidence:.2f}  (threshold {threshold})")

    # The routing decision, in a colour that matches what happens to the call.
    route = data["route"]
    if route == "self_service_faq":
        st.success("Route: **self_service_faq**. The FAQ bot answered the caller.")
    elif route == "route_to_intent_queue":
        st.info(f"Route: **route_to_intent_queue**. The call goes to the **{data['intent']}** team/flow.")
    else:
        st.warning(
            f"Route: **transfer_to_human**. Confidence {confidence:.2f} is below the threshold "
            f"{threshold} (or the FAQ docs had no answer), so a human agent takes the call."
        )

    if data["answer"]:
        st.markdown("**Answer read to the caller:**")
        st.write(data["answer"])
        st.caption("Sources: " + ", ".join(data["sources"]))
    st.caption("Answer written by: " + PROVIDER_LABELS.get(data["llm_provider"], data["llm_provider"]))

    with st.expander("Raw JSON reply from the API"):
        st.json(data)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Banking IVR Intent Router", page_icon="🏦", layout="wide")
st.title("🏦 Banking IVR Intent Router")
st.caption(
    "Routes bank customer messages to one of 77 intents, answers FAQs from the bank's own "
    "documents (RAG), and transfers to a human when it is not confident."
)

# ---------------------------------------------------------------------------
# Sidebar: which API to talk to + health check
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL).strip().rstrip("/")
st.sidebar.caption(
    "Render's free tier sleeps after 15 minutes without traffic. The first request after that "
    "can take about a minute."
)

if st.sidebar.button("Check API health", key="health_button"):
    with st.sidebar:
        with st.spinner("Calling /health ..."):
            health, error = call_api(api_url, "GET", "/health")
    if error:
        st.sidebar.error(error)
    else:
        st.sidebar.success("API is up")
        st.sidebar.write(f"Model: `{health['model_version']}`")
        st.sidebar.write(f"Confidence threshold: `{health['confidence_threshold']}`")
        st.sidebar.write(f"Primary LLM (Groq): {'yes' if health['primary_llm_configured'] else 'no'}")
        st.sidebar.write(f"Fallback LLM (Ollama): {'yes' if health['fallback_llm_configured'] else 'no'}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
route_tab, faq_tab, stats_tab, results_tab, about_tab = st.tabs(
    ["📞 Route a message", "💬 Ask the FAQ bot", "📊 Monitoring", "🏆 Model results", "ℹ️ How it works"]
)

# ---- Tab 1: POST /route ----------------------------------------------------
with route_tab:
    st.subheader("What does the customer want, and where should the call go?")
    example = st.selectbox("Pick an example (or type your own message below)", EXAMPLE_MESSAGES)
    # value=example: choosing another example puts it into the text box.
    message = st.text_area("Customer message", value=example, max_chars=500)

    if st.button("Route this message", type="primary", key="route_button"):
        if message.strip() == "":
            st.warning("Please type a message first.")
        else:
            with st.spinner("Asking the router ..."):
                route_data, error = call_api(api_url, "POST", "/route", {"text": message})
            if error:
                st.error(error)
            else:
                show_route_result(route_data)

# ---- Tab 2: POST /answer ---------------------------------------------------
with faq_tab:
    st.subheader("Ask the FAQ bot directly (RAG)")
    st.caption(
        "The bot searches the bank's FAQ documents, gives the best sections to the LLM, and only "
        "answers from them. If the documents do not contain the answer, it refuses."
    )
    st.caption(
        "Note: this tab is a direct test of the FAQ bot and SKIPS the intent router. In the real "
        "call flow ('Route a message' tab), only messages the router confidently classifies as an "
        "FAQ topic reach the FAQ bot; vague messages go to a human first."
    )
    example_question = st.selectbox("Pick an example question", EXAMPLE_QUESTIONS)
    question = st.text_input("Question", value=example_question, max_chars=500)

    if st.button("Ask", type="primary", key="answer_button"):
        if question.strip() == "":
            st.warning("Please type a question first.")
        else:
            with st.spinner("Searching the FAQ documents ..."):
                answer_data, error = call_api(api_url, "POST", "/answer", {"question": question})
            if error:
                st.error(error)
            elif answer_data["refused"]:
                st.warning(answer_data["answer"])
                st.caption("Refused: the FAQ documents do not contain this answer, so a human should take it.")
            else:
                st.success(answer_data["answer"])
                st.caption("Sources: " + ", ".join(answer_data["sources"]))
                st.caption("Answer written by: " + PROVIDER_LABELS.get(answer_data["llm_provider"], ""))
                st.caption(f"Latency: {answer_data['latency_ms']:.0f} ms")

# ---- Tab 3: GET /stats -----------------------------------------------------
with stats_tab:
    st.subheader("Monitoring: what is happening in production?")
    st.caption(
        "Every request is logged to a database. A rising handoff rate or falling average "
        "confidence can mean customers are saying new things the model does not know (drift)."
    )
    if st.button("Load latest stats", key="stats_button"):
        with st.spinner("Calling /stats ..."):
            stats, error = call_api(api_url, "GET", "/stats")
        if error:
            st.error(error)
        elif stats.get("total_route_requests", 0) == 0:
            st.info("No /route requests logged yet. Route a few messages first.")
        else:
            column_1, column_2, column_3, column_4 = st.columns(4)
            column_1.metric("Routed messages", stats["total_route_requests"])
            column_2.metric("Handoff rate", f"{stats['handoff_rate']:.0%}")
            column_3.metric("Avg confidence", f"{stats['avg_confidence']:.2f}")
            column_4.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")

            st.markdown("**Top intents**")
            intents_df = pd.DataFrame(stats["top_intents"]).set_index("intent")
            st.bar_chart(intents_df)

            st.markdown("**Who wrote the answers**")
            providers_df = pd.DataFrame(
                list(stats["answers_by_llm_provider"].items()), columns=["provider", "requests"]
            )
            st.dataframe(providers_df, hide_index=True)

# ---- Tab 4: offline results from Steps 8 and 9 ------------------------------
with results_tab:
    st.subheader("How good are the models? (measured offline, Steps 8 and 9)")
    final_results_path = REPORTS_DIR / "step9_final_results.json"
    llm_results_path = REPORTS_DIR / "step8_llm_vs_classic.json"

    if not final_results_path.exists():
        st.info("Run src/09_evaluate.py to create reports/step9_final_results.json.")
    else:
        final_results = json.loads(final_results_path.read_text(encoding="utf-8"))

        st.markdown("**Intent models on the Banking77 test set (3,080 messages)**")
        models_df = pd.DataFrame(final_results["intent_models"]).rename(columns={
            "model": "Model",
            "test_accuracy": "Test accuracy",
            "test_macro_f1": "Test macro-F1",
            "val_macro_f1": "Validation macro-F1",
            "latency_avg_ms": "Latency (ms)",
            "model_size_mb": "Size (MB)",
        })
        st.dataframe(models_df, hide_index=True)
        st.caption("Deployed: tfidf_lr (fits in Render's 512 MB; a MiniLM-based API needs ~665 MB).")

        deployed = final_results["deployed_with_threshold"]
        column_1, column_2, column_3 = st.columns(3)
        column_1.metric("Calls handled by the bot", f"{deployed['coverage']:.1%}")
        column_2.metric("Accuracy on those calls", f"{deployed['auto_accuracy']:.1%}")
        column_3.metric("Sent to a human", f"{deployed['handoff_rate']:.1%}")

        rag = final_results["rag"]
        st.markdown("**FAQ bot (RAG) on a 30-question golden set**")
        column_1, column_2, column_3 = st.columns(3)
        column_1.metric("Retrieval hit@3", f"{rag['tfidf_hit_at_3']:.0%}")
        column_2.metric("Correct answers", f"{rag['correct_answer_rate']:.0%}")
        column_3.metric("Correct refusals", f"{rag['correct_refusal_rate']:.0%}")

    if llm_results_path.exists():
        llm_results = json.loads(llm_results_path.read_text(encoding="utf-8"))
        st.markdown(f"**LLM vs trained models on the same {llm_results['n_queries']} test messages**")
        rows = [
            {"Model": "tfidf_lr", "Accuracy": llm_results["tfidf_lr_same_sample"]["accuracy"],
             "Latency (ms)": 3.6, "Tokens per call": 0},
            {"Model": "minilm_lr", "Accuracy": llm_results["minilm_lr_same_sample"]["accuracy"],
             "Latency (ms)": 27.7, "Tokens per call": 0},
        ]
        for mode_name, result in llm_results["llm"].items():
            rows.append({
                "Model": f"{mode_name} ({llm_results['llm_model']})",
                "Accuracy": result["accuracy"],
                "Latency (ms)": result["latency_avg_ms"],
                "Tokens per call": round(result["avg_prompt_tokens"] + result["avg_completion_tokens"]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("A trained classifier is more accurate, ~150x faster and has no per-call token cost.")

# ---- Tab 5: explanation ----------------------------------------------------
with about_tab:
    st.subheader("How a message is handled")
    st.markdown(
        """
1. **Classify**: stemming → TF-IDF → Logistic Regression gives an **intent** (one of 77) and a
   **confidence** (the highest predicted probability).
2. **Route**:
   - confidence **below 0.45** → `transfer_to_human` (not sure: a person takes the call)
   - a simple **FAQ intent** → `self_service_faq` (the RAG bot answers from the FAQ documents)
   - anything else → `route_to_intent_queue` (e.g. a lost card goes to the card-blocking flow)
3. **Answer FAQs (RAG)**: find the best FAQ sections (TF-IDF + FAISS) → the LLM answers **only**
   from them, with sources. LLM chain: **Groq** → local **Ollama** fallback → read out the FAQ
   text. Nothing relevant found, or the LLM says "I don't know" → transfer to a human.
4. **Log** every request (intent, confidence, route, LLM provider, latency) → `/stats`.

**Why 0.45?** It gave the lowest business cost on validation data, assuming a wrong automatic
route costs 4× a human handoff: the bot handles ~82% of calls at ~96% accuracy.
"""
    )
    st.caption("Data: Banking77 (PolyAI). FAQ documents describe a fictional 'Demo Bank'.")
