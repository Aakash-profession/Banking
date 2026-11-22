"""
Tests for FAQ retrieval and refusals (no LLM: conftest.py switches it off).
"""

from app.rag import CHUNKS, REFUSAL_MESSAGE, answer_question, retrieve


def test_faq_documents_are_loaded():
    assert len(CHUNKS) >= 30
    assert all(chunk["answer"] for chunk in CHUNKS)   # no empty sections


def test_retrieval_finds_the_right_document():
    results = retrieve("How long does it take to get my new card delivered?")
    assert results[0]["doc"] == "card_delivery.md"


def test_answer_contains_the_fact_and_a_source():
    result = answer_question("What is the daily ATM withdrawal limit?")
    assert result["refused"] is False
    assert "£500" in result["answer"]
    assert result["sources"][0].startswith("atm_withdrawals.md")


def test_question_not_in_docs_is_refused():
    result = answer_question("What is the weather like in London today?")
    assert result["refused"] is True
    assert result["answer"] == REFUSAL_MESSAGE
    assert result["sources"] == []
