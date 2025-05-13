from __future__ import annotations

from abc import ABC, abstractmethod

from src.schema import Listing


class AvgRatingPredictionModel(ABC):

    def predict(self, listing: Listing) -> float:
        """Predict the average rating for a listing."""
        prediction = self._do_predict(listing)
        return self.clamp(prediction)

    @abstractmethod
    def _do_predict(self, listing: Listing) -> float:
        """Predict the average rating for a listing."""
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def clamp(self, value: float, low: float = 0.0, high: float = 5.0) -> float:
        return max(low, min(value, high))
