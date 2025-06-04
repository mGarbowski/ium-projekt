from __future__ import annotations

from typing import override

from scipy.stats import truncnorm
from numpy.random import default_rng

from src.model.api import AvgRatingPredictionModel
from src.schema import Listing


class RandomPredictionModel(AvgRatingPredictionModel):
    """Random model based on truncated normal distribution."""

    def __init__(self, mean: float, std: float, lower: float, upper: float, seed: int = 42):
        self.mean = mean
        self.std = std
        self.lower = lower
        self.upper = upper
        self.random_state = default_rng(seed)

    @override
    def _do_predict(self, listing: Listing) -> float:
        a = (self.lower - self.mean) / self.std
        b = (self.upper - self.mean) / self.std
        return truncnorm.rvs(a, b, loc=self.mean, scale=self.std, size=1, random_state=self.random_state)[0]

    @override
    def name(self) -> str:
        return "RandomBaseModel"
