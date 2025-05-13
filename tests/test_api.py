from fastapi.testclient import TestClient

from src.app import app
from .utils import listing_1
from pytest import approx

client = TestClient(app)


def test_prediction_random_model():
    """Multiple requests to the random model should return different predictions."""
    request = {
        "user_id": "617d32dc-5c87-4ac6-a89d-b991624c529b",  # After hashing, this will always select the random model
        "data": listing_1
    }

    responses = [
        client.post("/predict/", json=request)
        for _ in range(10)
    ]

    for response in responses:
        assert response.status_code == 200
        assert "predicted_avg_rating" in response.json()
        prediction = response.json()["predicted_avg_rating"]
        assert 0 <= prediction <= 5.0

    for i in range(1, len(responses)):
        pred_1 = responses[i].json()["predicted_avg_rating"]
        pred_2 = responses[i - 1].json()["predicted_avg_rating"]
        assert not pred_1 == approx(pred_2), f"Predictions should be different: {pred_1} == {pred_2}"

def test_predict_deterministic_model():
    """Linear regression model should return the same prediction for the same input."""
    request = {
        "user_id": "617d32dc-5c87-4ac6-a89d-b991624c529e",  # after hashing, this will always select the linear  model
        "data": listing_1
    }

    responses = [
        client.post("/predict/", json=request)
        for _ in range(10)
    ]

    for response in responses:
        assert response.status_code == 200
        assert "predicted_avg_rating" in response.json()
        prediction = response.json()["predicted_avg_rating"]
        assert 0 <= prediction <= 5.0

    for i in range(1, len(responses)):
        pred_1 = responses[i].json()["predicted_avg_rating"]
        pred_2 = responses[i - 1].json()["predicted_avg_rating"]
        assert pred_1 == approx(pred_2), f"Predictions should be the same: {pred_1} != {pred_2}"