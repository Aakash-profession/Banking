"""
db.py: log every API request to a database with SQLAlchemy, and compute simple stats.

WHY log: this is the start of "LLMOps / MLOps monitoring". From the logs we can see:
    - handoff rate going up       -> the model may be getting worse (customers say new things)
    - average confidence dropping -> "data drift": today's messages differ from training data
    - latency going up            -> a performance problem
    - low-confidence messages     -> input for intent discovery (Step 7)
    - fallback LLM used often     -> the primary LLM (Groq) has a problem

Where do logs go?
    - Locally:   a SQLite file, request_log.db (no setup needed)
    - On Render: PostgreSQL, through the DATABASE_URL environment variable.
      (Render's disk is wiped on every restart, so a SQLite file there would be lost.)

Definitions:
    - SQLAlchemy: a Python library to work with SQL databases. The same code works with
      SQLite and PostgreSQL; only the connection URL changes.
    - ORM (Object-Relational Mapping): a Python class represents a table, and an object
      of that class represents one row.
    - Engine: SQLAlchemy's connection to the database.
    - Session: a short "conversation" with the database (add rows, then commit).
"""

import os
import re
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///request_log.db")

# Render gives URLs starting with "postgresql://" (or the old "postgres://").
# SQLAlchemy must be told to use the psycopg (version 3) driver: "postgresql+psycopg://".
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# pool_pre_ping=True: test a connection before using it, so a connection the database
# closed while the app was idle does not cause an error.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


class Base(DeclarativeBase):
    """Parent class for all our tables (SQLAlchemy needs one)."""


class RequestLog(Base):
    """One row per API request."""
    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    endpoint: Mapped[str] = mapped_column(String(20))            # "/route" or "/answer"
    text: Mapped[str] = mapped_column(String(300))               # redacted + shortened message
    intent: Mapped[str] = mapped_column(String(80), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    route: Mapped[str] = mapped_column(String(40), default="")
    llm_provider: Mapped[str] = mapped_column(String(20), default="none")   # primary / fallback / none
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)


# Create the table if it does not exist yet (safe to run on every startup).
Base.metadata.create_all(engine)


def redact(text: str) -> str:
    """Hide long numbers (card numbers, account numbers, phone numbers) before saving.

    Customer messages can contain personal data. Logging it in plain text is a privacy
    risk, so we replace any run of 4+ digits with [NUMBER] and keep at most 300 characters.
    (A real bank would use a proper PII-detection tool; this is the simple version.)
    """
    return re.sub(r"\d{4,}", "[NUMBER]", text)[:300]


def log_request(endpoint: str, text: str, intent: str, confidence: float,
                route: str, llm_provider: str, latency_ms: float) -> None:
    """Save one request. NEVER raises: if the database is down, the call must still work."""
    try:
        with Session(engine) as session:
            session.add(RequestLog(
                endpoint=endpoint,
                text=redact(text),
                intent=intent,
                confidence=confidence,
                route=route,
                llm_provider=llm_provider,
                latency_ms=latency_ms,
            ))
            session.commit()
    except Exception as error:
        print(f"[db] could not log request: {type(error).__name__}: {error}")


def get_stats() -> dict:
    """Simple monitoring report: /route numbers + which LLM wrote the answers."""
    with Session(engine) as session:
        route_filter = RequestLog.endpoint == "/route"
        total = session.scalar(select(func.count(RequestLog.id)).where(route_filter)) or 0
        if total == 0:
            return {"total_route_requests": 0}

        handoffs = session.scalar(
            select(func.count(RequestLog.id)).where(route_filter, RequestLog.route == "transfer_to_human")
        ) or 0
        avg_confidence = session.scalar(select(func.avg(RequestLog.confidence)).where(route_filter))
        avg_latency = session.scalar(select(func.avg(RequestLog.latency_ms)).where(route_filter))

        # Top 5 intents: GROUP BY intent, count, biggest first.
        top_rows = session.execute(
            select(RequestLog.intent, func.count(RequestLog.id).label("count"))
            .where(route_filter)
            .group_by(RequestLog.intent)
            .order_by(func.count(RequestLog.id).desc())
            .limit(5)
        ).all()

        # Which LLM wrote the answers (all endpoints)? A rising "fallback" count means
        # the primary LLM (Groq) is failing.
        provider_rows = session.execute(
            select(RequestLog.llm_provider, func.count(RequestLog.id).label("count"))
            .group_by(RequestLog.llm_provider)
        ).all()

    return {
        "total_route_requests": total,
        "handoff_rate": round(handoffs / total, 4),
        "avg_confidence": round(float(avg_confidence), 4),
        "avg_latency_ms": round(float(avg_latency), 2),
        "top_intents": [{"intent": row.intent, "count": row.count} for row in top_rows],
        "answers_by_llm_provider": {row.llm_provider: row.count for row in provider_rows},
    }
