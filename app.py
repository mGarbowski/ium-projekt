from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from hashlib import md5
from typing import override

import pandas as pd
from fastapi import FastAPI
from scipy.stats import truncnorm
from sklearn.linear_model import LinearRegression

from listings_preprocessing import transform_item
from schema import PredictRequest, Listing

SCALER_FILE = 'models/scaler.pkl'
LR_MODEL_FILE = 'models/linear_regression.pkl'

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("uvicorn")


class ListingTransformer:
    def __init__(self, scaler_file: str):
        self.scaler_file = scaler_file

    def transform(self, listing: Listing) -> pd.DataFrame:
        df = self.convert_to_dataframe(listing)
        return transform_item(df, self.scaler_file)

    def convert_to_dataframe(self, listing: Listing) -> pd.DataFrame:
        """Convert a Listing object to a DataFrame."""
        return pd.DataFrame([listing.model_dump()])


class AvgRatingPredictionModel(ABC):

    @abstractmethod
    def predict(self, listing: Listing) -> float:
        """Predict the average rating for a listing."""
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class LinearRegressionModel(AvgRatingPredictionModel):
    def __init__(self, model_file: str, listing_transformer: ListingTransformer):
        self.listing_transformer = listing_transformer
        with open(model_file, 'rb') as f:
            self.model: LinearRegression = pickle.load(f)

    @override
    def predict(self, listing: Listing) -> float:
        listing_df = self.listing_transformer.transform(listing)
        return self.model.predict(listing_df)[0]

    @override
    def name(self) -> str:
        return "LinearRegressionModel"


class RandomPredictionModel(AvgRatingPredictionModel):
    """Random model based on truncated normal distribution."""

    def __init__(self, mean: float, std: float, lower: float, upper: float):
        self.mean = mean
        self.std = std
        self.lower = lower
        self.upper = upper

    @override
    def predict(self, listing: Listing) -> float:
        a = (self.lower - self.mean) / self.std
        b = (self.upper - self.mean) / self.std
        return truncnorm.rvs(a, b, loc=self.mean, scale=self.std, size=1)[0]

    @override
    def name(self) -> str:
        return "RandomBaseModel"


class PredictionService:
    def __init__(self, model_a: AvgRatingPredictionModel, model_b: AvgRatingPredictionModel):
        self.model_a = model_a
        self.model_b = model_b
        self.logger = logging.getLogger(PredictionService.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = True

    def select_model(self, user_id: str) -> AvgRatingPredictionModel:
        """Select the model based on user_id using consistent hashing."""
        hash_value = md5(user_id.encode()).hexdigest()
        hash_value = int(hash_value, 16)
        model_idx = hash_value % 2
        return self.model_a if model_idx == 0 else self.model_b

    def predict(self, listing: Listing, user_id: str) -> float:
        model = self.select_model(user_id)
        prediction = model.predict(listing)
        self.logger.info(
            f"Prediction for user {user_id}, given listing {listing.id}, using model {model.name()}: {prediction}"
        )
        return prediction


listing_transformer = ListingTransformer(SCALER_FILE)
random_model = RandomPredictionModel(mean=4.77, std=0.27, lower=0.0, upper=5.0)
linear_regression_model = LinearRegressionModel(LR_MODEL_FILE, listing_transformer)
prediction_service = PredictionService(linear_regression_model, random_model)
app = FastAPI()


@app.post("/predict/")
def predict(request: PredictRequest):
    user_id = request.user_id
    listing = request.data
    prediction = prediction_service.predict(listing, user_id)
    return {"predicted_avg_rating": prediction}
