"""
Tests for the FastAPI endpoints.

TestClient sends fake HTTP requests to our app WITHOUT starting a real server,
so the tests are fast and need no network.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200          # 200 = OK
    assert response.json()["status"] == "ok"


def test_route_returns_all_fields():
    response = client.post("/route", json={"text": "My card still hasn't arrived"})
    assert response.status_code == 200
    body = response.json()
    for field in ["intent", "confidence", "route", "threshold", "latency_ms"]:
        assert field in body


def test_route_faq_question_gets_an_answer():
    response = client.post("/route", json={"text": "How long does card delivery take?"})
    body = response.json()
    assert body["route"] == "self_service_faq"
    assert body["answer"]                        # not empty
    assert body["sources"]


def test_route_rejects_empty_text():
    # Pydantic's min_length=1 rule -> FastAPI answers 422 (Unprocessable Content).
    response = client.post("/route", json={"text": ""})
    assert response.status_code == 422


def test_answer_endpoint_refuses_unknown_question():
    response = client.post("/answer", json={"question": "Can I buy Bitcoin in the app?"})
    assert response.status_code == 200
    assert response.json()["refused"] is True


def test_stats_counts_requests():
    client.post("/route", json={"text": "I want to close my account"})
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json()["total_route_requests"] >= 1
