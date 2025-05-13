from __future__ import annotations

from typing import override

from scipy.stats import truncnorm

from src.model.api import AvgRatingPredictionModel
from src.schema import Listing


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
