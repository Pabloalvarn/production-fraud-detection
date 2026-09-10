from fastapi.testclient import TestClient

from fraud_detection.api.main import app
from tests.conftest import FakeProbabilityModel


def test_health_without_model():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert "model_loaded" in response.json()


def test_prediction_with_injected_bundle():
    with TestClient(app) as client:
        app.state.bundle = {
            "model": FakeProbabilityModel(),
            "threshold": 0.50,
            "high_risk_threshold": 0.90,
            "model_version": "test",
            "raw_features": ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"],
        }
        response = client.post(
            "/predict",
            json={
                "step": 1,
                "type": "TRANSFER",
                "amount": 8000,
                "oldbalanceOrg": 10000,
                "oldbalanceDest": 0,
            },
        )
        assert response.status_code == 200
        assert response.json()["decision"] == "manual_review"


def test_invalid_negative_amount_is_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "step": 1,
                "type": "PAYMENT",
                "amount": -1,
                "oldbalanceOrg": 10,
                "oldbalanceDest": 10,
            },
        )
        assert response.status_code == 422
