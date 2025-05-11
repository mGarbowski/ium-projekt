from __future__ import annotations

import logging
from hashlib import md5

from src.model.api import AvgRatingPredictionModel
from src.schema import Listing


class PredictionService:
    def __init__(self, model_a: AvgRatingPredictionModel, model_b: AvgRatingPredictionModel):
        self.model_a = model_a
        self.model_b = model_b
        self.logger = self.get_logger()

    @staticmethod
    def get_logger():
        """Get the logger for the PredictionService."""
        logger = logging.getLogger(PredictionService.__name__)
        logger.setLevel(logging.INFO)
        return logger

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
