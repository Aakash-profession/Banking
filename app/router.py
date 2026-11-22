"""
router.py: loads the intent model ONCE and decides where each call should go.

The routing rules (the heart of the IVR):
    1. confidence <  threshold          -> "transfer_to_human"      (not sure: let a person handle it)
    2. intent is a simple FAQ question  -> "self_service_faq"       (the bot answers from docs/faq)
    3. anything else                    -> "route_to_intent_queue"  (send to the right team/flow,
                                                                     e.g. the card-blocking flow)

The model and its threshold come from app/artifacts/, written by src/10_export_model.py.
"""

import json
from pathlib import Path

import joblib

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

ROUTE_HUMAN = "transfer_to_human"
ROUTE_FAQ = "self_service_faq"
ROUTE_QUEUE = "route_to_intent_queue"

# Intents whose answer is a FACT from our FAQ docs (no account access or action needed).
# Action intents like lost_or_stolen_card or terminate_account are NOT here: they need
# identity checks and an action, so they go to their own queue/flow.
FAQ_INTENTS = {
    "card_delivery_estimate",
    "card_arrival",
    "activate_my_card",
    "change_pin",
    "top_up_limits",
    "topping_up_by_card",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "exchange_rate",
    "exchange_charge",
    "atm_support",
    "cash_withdrawal_charge",
    "transfer_timing",
    "verify_my_identity",
    "why_verify_identity",
    "country_support",
    "fiat_currency_support",
    "supported_cards_and_currencies",
    "visa_or_mastercard",
    "age_limit",
    "apple_pay_or_google_pay",
}

# Load the model and its settings ONCE, when this module is first imported (at API startup).
# WHY: loading takes time; doing it on every request would make every call slow.
MODEL = joblib.load(ARTIFACTS_DIR / "intent_model.joblib")
with open(ARTIFACTS_DIR / "model_info.json", "r", encoding="utf-8") as file:
    MODEL_INFO = json.load(file)
CONFIDENCE_THRESHOLD: float = MODEL_INFO["confidence_threshold"]


def predict_intent(text: str) -> tuple[str, float]:
    """Return (intent, confidence) for one customer message."""
    probabilities = MODEL.predict_proba([text])[0]   # 77 probabilities for this one message
    best_index = probabilities.argmax()
    return str(MODEL.classes_[best_index]), float(probabilities[best_index])


def decide_route(intent: str, confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> str:
    """Apply the 3 routing rules. Kept separate from the model so it is easy to unit-test."""
    if confidence < threshold:
        return ROUTE_HUMAN
    if intent in FAQ_INTENTS:
        return ROUTE_FAQ
    return ROUTE_QUEUE
