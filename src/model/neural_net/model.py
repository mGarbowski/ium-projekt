import pickle
import torch
from typing import override
from src.listings_preprocessing import ListingTransformer
from src.model.api import AvgRatingPredictionModel
from src.model.neural_net.network import NeuralNetworkAvgRatingRegressor
from src.schema import Listing


class NeuralNetPredictionModel(AvgRatingPredictionModel):

    def __init__(self, model_file: str, listing_transformer: ListingTransformer):
        self.listing_transformer = listing_transformer
        with open(model_file, "rb") as f:
            self.model: NeuralNetworkAvgRatingRegressor = torch.load(
                f, weights_only=False
            )

    @override
    def _do_predict(self, listing: Listing) -> float:
        transformed = self.listing_transformer.transform(listing)
        prediction = self.model.predict(torch.tensor(transformed))
        return self.clamp(prediction.item())

    @override
    def name(self) -> str:
        return "NeuralNetPredictionModel"
