from __future__ import annotations

import pickle
from typing import override

from sklearn.linear_model import LinearRegression

from src.listings_preprocessing import ListingTransformer
from src.model.api import AvgRatingPredictionModel
from src.schema import Listing


class LinearRegressionModel(AvgRatingPredictionModel):
    def __init__(self, model_file: str, listing_transformer: ListingTransformer):
        self.listing_transformer = listing_transformer
        with open(model_file, 'rb') as f:
            self.model: LinearRegression = pickle.load(f)

    @override
    def _do_predict(self, listing: Listing) -> float:
        listing_df = self.listing_transformer.transform(listing)
        return self.model.predict(listing_df)[0]

    @override
    def name(self) -> str:
        return "LinearRegressionModel"
