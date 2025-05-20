from __future__ import annotations

import logging

from fastapi import FastAPI

from src.listings_preprocessing import ListingTransformer
from src.model.linear_regression import LinearRegressionModel
from src.model.neural_net.model import NeuralNetPredictionModel
from src.model.random import RandomPredictionModel
from src.prediction_service import PredictionService
from src.schema import PredictRequest

import os

SCALER_FILE = (
    os.environ["SCALER_FILE"] if "SCALER_FILE" in os.environ else "models/scaler.pkl"
)
LR_MODEL_FILE = (
    os.environ["LR_MODEL_FILE"]
    if "LR_MODEL_FILE" in os.environ
    else "models/linear_regression.pkl"
)
IMPUTATION_FILE = (
    os.environ["IMPUTATION_FILE"]
    if "IMPUTATION_FILE" in os.environ
    else "models/imputer_pipeline.pkl"
)

NEURAL_NET_FILE = (
    os.environ["NEURAL_NET_FILE"]
    if "NEURAL_NET_FILE" in os.environ
    else "models/nn.pkl"
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("uvicorn")

listing_transformer = ListingTransformer(SCALER_FILE, IMPUTATION_FILE)
random_model = RandomPredictionModel(mean=4.77, std=0.27, lower=0.0, upper=5.0)
linear_regression_model = LinearRegressionModel(LR_MODEL_FILE, listing_transformer)
neural_net_model = NeuralNetPredictionModel(NEURAL_NET_FILE, listing_transformer)
prediction_service = PredictionService(neural_net_model, random_model)
app = FastAPI()


@app.post("/predict/")
def predict(request: PredictRequest):
    user_id = request.user_id
    listing = request.data
    prediction = prediction_service.predict(listing, user_id)
    return {"predicted_avg_rating": prediction}
