"""
schemas.py: the shape of the data going IN and OUT of the API (Pydantic models).

WHY: FastAPI uses these classes to
    - check every request automatically (empty text? too long? -> error 422 for the caller)
    - convert the response to JSON
    - draw the interactive documentation page at /docs

Definitions:
    - Pydantic: a library that checks data against a description (types + rules).
    - BaseModel: the parent class for these descriptions. Each attribute = one JSON field.
    - Field(...): extra rules for a field, e.g. min_length / max_length.
"""

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    """What the IVR sends us: the customer's message (already converted from speech to text)."""
    text: str = Field(
        min_length=1,
        max_length=500,   # protects us from huge inputs (cost, latency, abuse)
        description="The customer's message",
        examples=["My card still hasn't arrived"],
    )


class RouteResponse(BaseModel):
    """What we send back: which intent, how sure we are, and where the call goes."""
    intent: str                    # e.g. "card_arrival"
    confidence: float              # the model's probability for that intent (0..1)
    route: str                     # "self_service_faq" | "route_to_intent_queue" | "transfer_to_human"
    threshold: float               # the confidence threshold in use (from Step 6)
    answer: str | None = None      # filled only when the FAQ bot answered the question
    sources: list[str] = []        # which FAQ sections the answer came from
    llm_provider: str = "none"     # who wrote the answer: "primary" (Groq), "fallback" (Ollama), "none"
    latency_ms: float              # how long the whole request took


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, examples=["How long does card delivery take?"])


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    refused: bool                  # True = "I don't know", the caller should go to a human
    used_llm: bool                 # False = extractive fallback (no LLM configured or all LLMs failed)
    llm_provider: str              # "primary" (Groq), "fallback" (Ollama) or "none"
    latency_ms: float
