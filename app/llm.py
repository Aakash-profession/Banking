"""
llm.py: ONE small place that talks to Large Language Models (LLMs), with a FALLBACK.

The fallback chain (graceful degradation):
    1. PRIMARY LLM  = Groq (cloud, fast, free tier)
    2. FALLBACK LLM = Ollama (runs on my own machine) -> used only if the primary fails
                      (Groq down, timeout, rate limit, wrong key, no internet to Groq...)
    3. If BOTH fail, ask_llm raises an error, and the caller uses its own non-LLM fallback
       (rag.py then reads out the best FAQ answer word for word; Step 7 uses keywords).
So the caller ALWAYS gets an answer or a handoff. The app never crashes because of an LLM.

WHY one file: the cluster namer (Step 7), the LLM classifier (Step 8) and the FAQ answers
(RAG) all need an LLM. Keeping the calls here means providers, timeouts, retries and the
fallback rule are written once.

Which providers work? Any provider with an "OpenAI-compatible" API, through the official
`openai` Python library. You only change settings in your .env file:
    Groq          -> base URL https://api.groq.com/openai/v1      (my primary)
    Ollama (local)-> base URL http://localhost:11434/v1           (my fallback; any text as key)
    OpenAI        -> base URL empty
    Gemini        -> base URL https://generativelanguage.googleapis.com/v1beta/openai/

Note for deployment: on Render there is no Ollama (512 MB RAM is far too small for a local
LLM), so leave the FALLBACK_ settings empty there. The chain is then Groq -> extractive answer.
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

# load_dotenv() reads the .env file (if there is one) and puts its lines into
# environment variables. Existing environment variables are NOT overwritten, so
# settings made on Render's dashboard always win over a local file.
load_dotenv()

# ---- Primary LLM (Groq) ----
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# ---- Fallback LLM (Ollama) ----
FALLBACK_LLM_API_KEY = os.getenv("FALLBACK_LLM_API_KEY", "")
FALLBACK_LLM_BASE_URL = os.getenv("FALLBACK_LLM_BASE_URL", "")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "")

# Price per 1 million tokens of the PRIMARY, used only to ESTIMATE cost in Step 8.
# A token is a piece of a word; roughly 1 token = 3/4 of an English word.
LLM_PRICE_INPUT_PER_1M = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0"))
LLM_PRICE_OUTPUT_PER_1M = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "0"))

# Timeouts and retries.
# A live caller is waiting on the phone, so the PRIMARY gets a short timeout and few
# retries: if Groq is in trouble we switch to the fallback quickly instead of waiting.
PRIMARY_TIMEOUT_SECONDS = 20
PRIMARY_MAX_RETRIES = 2
# Batch experiments (Step 8) can wait. Free tiers limit requests/tokens PER MINUTE and
# answer "429 Too Many Requests" with a "retry-after" header; the openai library waits
# that long and retries automatically, up to this many times.
BATCH_MAX_RETRIES = 8
# The fallback runs on a laptop CPU, which is slow, so it gets more time but only 1 retry
# (if Ollama is not running, we want to find out fast).
FALLBACK_TIMEOUT_SECONDS = 90
FALLBACK_MAX_RETRIES = 1


def is_primary_configured() -> bool:
    return LLM_API_KEY != "" and LLM_MODEL != ""


def is_fallback_configured() -> bool:
    return FALLBACK_LLM_API_KEY != "" and FALLBACK_LLM_MODEL != ""


def is_llm_configured() -> bool:
    """True when at least one LLM (primary or fallback) is set up."""
    return is_primary_configured() or is_fallback_configured()


# Create the clients ONCE (when this file is first imported), not on every call.
_primary_client = None
if is_primary_configured():
    _primary_client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL or None,   # None = the default OpenAI address
        timeout=PRIMARY_TIMEOUT_SECONDS,
        max_retries=PRIMARY_MAX_RETRIES,
    )

_fallback_client = None
if is_fallback_configured():
    _fallback_client = OpenAI(
        api_key=FALLBACK_LLM_API_KEY,
        base_url=FALLBACK_LLM_BASE_URL or None,
        timeout=FALLBACK_TIMEOUT_SECONDS,
        max_retries=FALLBACK_MAX_RETRIES,
    )


def _call_model(client: OpenAI, model: str, provider: str,
                system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
    """Send ONE request to ONE provider. Raises an exception if it fails.

    temperature=0 = always pick the most likely next word -> stable, repeatable answers
                    (we want consistency for classification and FAQ answers, not creativity)
    """
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    text = response.choices[0].message.content or ""
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0

    # Only the primary has a price; the local fallback costs nothing per token.
    cost_usd = 0.0
    if provider == "primary":
        cost_usd = (prompt_tokens / 1_000_000 * LLM_PRICE_INPUT_PER_1M
                    + completion_tokens / 1_000_000 * LLM_PRICE_OUTPUT_PER_1M)

    return {
        "text": text.strip(),
        "provider": provider,     # "primary" or "fallback": which LLM actually answered
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
    }


def ask_llm(system_prompt: str, user_prompt: str, max_tokens: int = 300,
            batch_mode: bool = False) -> dict:
    """Ask the primary LLM; if it fails, ask the fallback LLM.

    system_prompt = the rules/role for the model ("You are a bank FAQ assistant...")
    user_prompt   = the actual task or question
    batch_mode    = True for offline experiments (Step 8): use ONE model only (the primary
                    if it is set, otherwise the fallback), retry patiently, and never switch
                    models halfway, so the measurement is about one model.

    Returns a dict: text, provider, model, prompt_tokens, completion_tokens, latency_ms, cost_usd
    Raises an exception if no LLM could answer; the caller then uses its non-LLM fallback.
    """
    # ---- Batch mode: one model, patient retries, no switching ----
    if batch_mode:
        if _primary_client is not None:
            # with_options = the same client, but with different settings for this call.
            patient_client = _primary_client.with_options(max_retries=BATCH_MAX_RETRIES)
            return _call_model(patient_client, LLM_MODEL, "primary", system_prompt, user_prompt, max_tokens)
        if _fallback_client is not None:
            return _call_model(_fallback_client, FALLBACK_LLM_MODEL, "fallback",
                               system_prompt, user_prompt, max_tokens)
        raise RuntimeError("No LLM configured. Set LLM_API_KEY and LLM_MODEL in .env")

    # ---- Live mode: primary first, then the fallback ----
    # 1. Try the primary (Groq).
    if _primary_client is not None:
        try:
            return _call_model(_primary_client, LLM_MODEL, "primary", system_prompt, user_prompt, max_tokens)
        except Exception as error:
            if _fallback_client is None:
                raise   # nothing else to try: let the caller handle it
            print(f"[llm] primary failed ({type(error).__name__}); trying the fallback LLM")

    # 2. Try the fallback (Ollama).
    if _fallback_client is not None:
        return _call_model(_fallback_client, FALLBACK_LLM_MODEL, "fallback",
                           system_prompt, user_prompt, max_tokens)

    raise RuntimeError("No LLM configured. Set LLM_API_KEY and LLM_MODEL in .env")


def batch_model_name() -> str:
    """The model that batch_mode calls will use (for printing in experiment reports)."""
    if is_primary_configured():
        return LLM_MODEL
    if is_fallback_configured():
        return FALLBACK_LLM_MODEL
    return ""
