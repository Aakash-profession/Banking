# Banking IVR Intent Router

> A learning project for AI-engineering interview preparation, built step by step with
> Claude Code (an AI pair programmer) and explained line by line. Every number below was
> measured on a laptop CPU (Intel i5-10310U, no GPU).

A bank's phone line (IVR) or chat receives messages like *"my card hasn't arrived"* or
*"why was I charged twice?"*. This project:

1. **Routes** each message to one of **77 intents** (Banking77 dataset).
2. **Answers simple FAQ questions** from the bank's own documents (RAG), with sources.
3. **Transfers to a human** when the model is not confident or the docs have no answer.
4. **Discovers new intents** by clustering the messages the model was unsure about.

## Architecture

```
Customer text
     │
     ▼
POST /route ──► stemming → TF-IDF → Logistic Regression ──► intent + confidence
     │
     ├── confidence < 0.45 ─────────────────────────────► transfer_to_human
     ├── FAQ intent ──► RAG: TF-IDF retrieval (FAISS) → LLM answer + sources
     │                   LLM chain: Groq (primary) → Ollama (local fallback) → extractive answer
     │                   nothing relevant / LLM says "I don't know" → transfer_to_human
     └── other intent ──────────────────────────────────► route_to_intent_queue
     │
     ▼
every request logged (intent, confidence, route, LLM provider, latency) → GET /stats
low-confidence messages → clustering (Step 7) → "new intent" suggestions
```

## Results (real numbers, laptop CPU: Intel i5-10310U, no GPU)

### Intent classification (Banking77 test set: 3,080 messages, 40 per intent, used once)

| Model | Test accuracy | Test macro-F1 | Latency / message | Size | Fits Render free (512 MB)? |
|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** (stemmed), **deployed** | 0.896 | 0.897 | 3.6 ms | 12 MB | ✅ API uses ~257 MB |
| MiniLM embeddings + Logistic Regression | **0.928** | **0.927** | 28 ms | 87 MB | ❌ API would need ~665 MB |
| MiniLM embeddings + XGBoost | 0.901 | 0.900 | 49 ms | 101 MB | ❌ |
| DistilBERT fine-tuned (3 epochs, CPU, 117 min) | 0.902 | 0.902 | 85 ms | 256 MB | ❌ |

LLM vs trained models on the **same 200 test messages** (Groq `openai/gpt-oss-20b`):

| Model | Accuracy | macro-F1 | Latency | Tokens / call |
|---|---|---|---|---|
| TF-IDF + LR | 0.870 | 0.833 | 3.6 ms | 0 |
| MiniLM + LR | 0.895 | 0.884 | 28 ms | 0 |
| LLM few-shot (1 example per intent) | 0.820 | 0.764 | 569 ms | 1,923 |
| LLM zero-shot | 0.780 | 0.743 | 452 ms | 576 |

### Routing with human handoff (deployed model, threshold 0.45)

| | Validation | Test |
|---|---|---|
| Calls handled by the bot (coverage) | 83.5% | 82.4% |
| Accuracy on those calls | 96.1% | 96.2% |
| Sent to a human | 16.5% | 17.6% |
| Off-topic questions sent to a human | 30 / 30 | |

### FAQ answers (RAG, 30-question golden set: 25 answerable, 5 not in the docs)

| Metric | Value |
|---|---|
| Retrieval hit@1 / hit@3, TF-IDF (deployed) | 0.96 / 1.00 |
| Retrieval hit@1 / hit@3, MiniLM embeddings | 1.00 / 1.00 |
| Correct answers (answerable, Groq) | 92% |
| Correct refusals (not in the docs) | 100% |
| Wrong refusals (answerable) | 4% (1 question: TF-IDF missed "abroad" ≈ "international") |

### Intent discovery (5 intents hidden from training)

82.5% of their messages were flagged as low-confidence; K-means (K = 6 by silhouette) put
99–100% of each hidden intent into one cluster; the LLM named them e.g. `delete_account`,
`lost_phone_support`, `atm_card_issue`.

## Setup (Windows PowerShell, inside this folder)

```powershell
py -3.11 -m venv .venv                     # create the virtual environment
.venv\Scripts\Activate.ps1                 # switch it on (the prompt shows "(.venv)")
python -m pip install --upgrade pip
pip install -r requirements-dev.txt        # all laptop libraries (includes requirements.txt)
python -m spacy download en_core_web_sm    # spaCy English model (Step 2 only)
Copy-Item .env.example .env                # then put your Groq key in .env (optional)
```

The LLM is optional. Without one, the FAQ bot reads out the best-matching FAQ answer word
for word and Step 8 part (a) is skipped. For the local fallback: install Ollama and run
`ollama pull llama3.2:3b`.

## Run order

| Step | Command | What it does |
|---|---|---|
| 0 | `python src/00_check_setup.py` | venv + library check, test MLflow run |
| 1 | `python src/01_load_and_eda.py` | download Banking77, EDA, stratified train/val split |
| 2 | `python src/02_preprocess.py` | spaCy/NLTK preprocessing comparison |
| 3 | `python src/03_baseline_tfidf.py` | TF-IDF + Logistic Regression baseline |
| 4 | `python src/04_embeddings_models.py` | MiniLM embeddings + LR / XGBoost |
| 5 | `python src/05_finetune_distilbert.py` | fine-tune DistilBERT (slow on CPU) |
| 6 | `python src/06_confidence_handoff.py` | confidence threshold + out-of-scope test |
| 7 | `python src/07_intent_discovery.py` | hide 5 intents, rediscover them by clustering |
| 8 | `python src/08_llm_classify_and_rag.py` | LLM zero/few-shot vs classifier, RAG demo |
| 9 | `python src/09_evaluate.py` | final test-set scores + RAG golden-set evaluation |
| 10 | `python src/10_export_model.py` | promote the chosen model into `app/artifacts/` |
| tests | `pytest` | 18 automatic tests (routing, RAG, API, quality regression) |
| API | `uvicorn app.main:app --reload` | then open http://127.0.0.1:8000/docs |
| UI | `streamlit run ui/streamlit_app.py` | then open http://localhost:8501 |
| MLflow | `mlflow ui` | then open http://127.0.0.1:5000 |

Always run from this folder (MLflow saves to `mlflow.db` here).

## Live demo

- **API docs (try it in the browser):** https://ivr-intent-router.onrender.com/docs
  (free tier: the first request after 15 idle minutes takes about a minute)

## Web UI (Streamlit)

`ui/streamlit_app.py` is a small front end that calls the API over HTTP (it loads no model):

| Tab | What it shows |
|---|---|
| 📞 Route a message | intent, confidence bar vs threshold, route (FAQ bot / team queue / human), FAQ answer + sources |
| 💬 Ask the FAQ bot | grounded answer with sources, or a refusal when the docs have no answer |
| 📊 Monitoring | `/stats`: routed messages, handoff rate, average confidence, latency, top intents, LLM provider |
| 🏆 Model results | offline results from Steps 8 and 9 |
| ℹ️ How it works | the routing rules in plain words |

```powershell
streamlit run ui/streamlit_app.py     # then open http://localhost:8501
```

The sidebar switches between the Render API (default) and a local one (`http://127.0.0.1:8000`).
To host the UI for free: Streamlit Community Cloud → Create app → this repo, branch `main`,
main file `ui/streamlit_app.py` (it installs `ui/requirements.txt`, which sits next to the app).

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, model version, threshold, which LLMs are configured |
| `POST /route` `{"text": "..."}` | intent, confidence, route, FAQ answer + sources if answered |
| `POST /answer` `{"question": "..."}` | FAQ answer with sources, or a refusal |
| `GET /stats` | handoff rate, average confidence, latency, top intents, answers per LLM provider |

## Deploy on Render (free tier)

1. Push the repo to GitHub (`app/artifacts/` must be committed: it holds the model).
2. Render → **New → Blueprint** → pick the repo. `render.yaml` creates the web service
   and a free PostgreSQL database for the request logs.
3. When asked, paste the Groq key into `LLM_API_KEY` (never into Git).
4. Open `https://<service>.onrender.com/docs`.

Free-tier facts (Render docs, checked 2026-11-22), and how this project handles them:

| Limit | Effect | How the project handles it |
|---|---|---|
| 512 MB RAM, 0.1 CPU | big models do not fit | deploy TF-IDF + LR (API uses ~257 MB; a MiniLM API needs ~665 MB) |
| Sleeps after 15 min idle | first request takes ~1 minute (cold start) | expected on the free tier; a paid instance stays awake |
| Disk wiped on restart | a SQLite log file would be lost | logs go to Render PostgreSQL via `DATABASE_URL` |
| Free PostgreSQL expires after 30 days | logs disappear after that | logging never crashes the API; recreate the DB or upgrade |
| No local LLM possible | no Ollama fallback on Render | chain on Render = Groq → extractive answer |
| Default Python is newer | pickle/library mismatch risk | `.python-version` pins 3.11.9, `requirements.txt` pins versions |

## Limitations and next steps

- TF-IDF retrieval matches words, not meaning; MiniLM retrieval scores better (Step 9)
  but needs more RAM. With a bigger instance: MiniLM for retrieval and intent classification.
- The refusal threshold was tuned on the same 30 golden questions it is evaluated on;
  a larger golden set split into tune/report halves would be cleaner.
- The FAQ answer check is a key-phrase match; an LLM-as-judge (checked by hand) would grade
  paraphrased answers better.
- The Ollama fallback on a laptop CPU takes many seconds per answer: too slow for a live call.
  In production the fallback would be a second hosted provider or a self-hosted GPU model.
- Stretch (Step 11): LangGraph flow for action intents (e.g. block a lost card with
  identity checks and human approval), Whisper speech-to-text.
