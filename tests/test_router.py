"""
Unit tests for the routing rules and the deployed intent model.

A "unit test" checks one small piece of code on its own. Each test is a function whose
name starts with test_. "assert X" means: the test FAILS if X is not true.
"""

from app.router import (
    CONFIDENCE_THRESHOLD,
    ROUTE_FAQ,
    ROUTE_HUMAN,
    ROUTE_QUEUE,
    decide_route,
    predict_intent,
)


# --- The routing rules, with made-up numbers (no model needed) ---------------------------

def test_low_confidence_goes_to_human():
    assert decide_route("card_arrival", confidence=0.10, threshold=0.5) == ROUTE_HUMAN


def test_confident_faq_intent_goes_to_faq_bot():
    assert decide_route("card_arrival", confidence=0.90, threshold=0.5) == ROUTE_FAQ


def test_confident_action_intent_goes_to_queue():
    # Blocking a lost card is an ACTION (needs identity checks), not an FAQ.
    assert decide_route("lost_or_stolen_card", confidence=0.90, threshold=0.5) == ROUTE_QUEUE


def test_confidence_exactly_at_threshold_is_automated():
    # The rule is "confidence < threshold -> human", so equal means automated.
    assert decide_route("card_arrival", confidence=0.5, threshold=0.5) == ROUTE_FAQ


# --- The real deployed model -------------------------------------------------------------

def test_clear_message_gets_the_right_intent():
    intent, confidence = predict_intent("I lost my card, someone may have stolen it")
    assert intent == "lost_or_stolen_card"
    assert 0.0 <= confidence <= 1.0


def test_off_topic_message_goes_to_human():
    intent, confidence = predict_intent("what's the weather like in Bangalore today")
    assert decide_route(intent, confidence) == ROUTE_HUMAN


def test_threshold_is_a_sensible_number():
    assert 0.0 < CONFIDENCE_THRESHOLD < 1.0
