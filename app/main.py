"""
main.py: the FastAPI web API for the Banking IVR Intent Router.

Endpoints:
    GET  /health  -> is the service alive? which model version is loaded?
    POST /route   -> customer text in -> intent, confidence, route (+ FAQ answer if possible)
    POST /answer  -> FAQ question in -> grounded answer with sources (or a refusal)
    GET  /stats   -> monitoring report: handoff rate, average confidence, top intents
    GET  /docs    -> interactive documentation, made automatically by FastAPI

Run locally (from the ivr-reference folder):
    uvicorn app.main:app --reload
    then open http://127.0.0.1:8000/docs

Definitions:
    - API (Application Programming Interface): a way for programs to talk to each other.
      Here the IVR system sends JSON to our URL and gets JSON back.
    - Endpoint: one URL + method, e.g. POST /route.
    - GET vs POST: GET = read something; POST = send data to be processed.
    - @app.get(...) / @app.post(...): FastAPI "decorators". The line above a function
      tells FastAPI: "call this function when a request arrives at this URL".
    - We use plain "def" (not "async def") because our model code is normal blocking
      Python code; FastAPI runs plain functions in a thread pool so other requests
      are not blocked.
"""

import os
import time

from fastapi import FastAPI

from app.db import get_stats, log_request
from app.llm import is_fallback_configured, is_primary_configured
from app.rag import answer_question
from app.router import (
    CONFIDENCE_THRESHOLD,
    MODEL_INFO,
    ROUTE_FAQ,
    ROUTE_HUMAN,
    decide_route,
    predict_intent,
)
from app.schemas import AnswerRequest, AnswerResponse, RouteRequest, RouteResponse

app = FastAPI(
    title="Banking IVR Intent Router",
    description="Routes bank customer messages to the right intent, answers FAQs, "
                "and transfers to a human when not confident.",
    version="1.0.0",
)


@app.get("/")
def home() -> dict:
    return {"message": "Banking IVR Intent Router. See /docs for the interactive API."}


@app.get("/health")
def health() -> dict:
    """Used by Render (and by people) to check the service is up."""
    return {
        "status": "ok",
        # Which code is running? Render sets RENDER_GIT_COMMIT to the deployed commit, so you
        # can compare it with `git log` and know whether your latest push is live.
        "code_version": os.getenv("RENDER_GIT_COMMIT", "local")[:7],
        "model_version": MODEL_INFO["model_version"],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "primary_llm_configured": is_primary_configured(),     # Groq
        "fallback_llm_configured": is_fallback_configured(),   # local Ollama
    }


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> RouteResponse:
    """Classify the message and decide where the call goes."""
    start = time.perf_counter()

    intent, confidence = predict_intent(request.text)
    route_name = decide_route(intent, confidence)

    answer = None
    sources: list[str] = []
    llm_provider = "none"
    if route_name == ROUTE_FAQ:
        rag_result = answer_question(request.text)
        llm_provider = rag_result["llm_provider"]
        if rag_result["refused"]:
            # The FAQ docs do not cover this question -> a human should take it.
            route_name = ROUTE_HUMAN
        else:
            answer = rag_result["answer"]
            sources = rag_result["sources"]

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    log_request("/route", request.text, intent, confidence, route_name, llm_provider, latency_ms)

    return RouteResponse(
        intent=intent,
        confidence=round(confidence, 4),
        route=route_name,
        threshold=CONFIDENCE_THRESHOLD,
        answer=answer,
        sources=sources,
        llm_provider=llm_provider,
        latency_ms=latency_ms,
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    """Answer an FAQ question directly from the docs (RAG)."""
    start = time.perf_counter()
    result = answer_question(request.question)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    # For /answer we store the retrieval score in the "confidence" column.
    log_request("/answer", request.question, "", result["top_score"],
                "refused" if result["refused"] else "answered", result["llm_provider"], latency_ms)

    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        refused=result["refused"],
        used_llm=result["used_llm"],
        llm_provider=result["llm_provider"],
        latency_ms=latency_ms,
    )


@app.get("/stats")
def stats() -> dict:
    """Monitoring: handoff rate, average confidence, latency and top intents."""
    return get_stats()
