from __future__ import annotations

from abc import ABC, abstractmethod

from src.schema import Listing


class AvgRatingPredictionModel(ABC):

    @abstractmethod
    def predict(self, listing: Listing) -> float:
        """Predict the average rating for a listing."""
        pass

    @abstractmethod
    def name(self) -> str:
        pass
