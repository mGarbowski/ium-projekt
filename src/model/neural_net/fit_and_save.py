"""Script to fit and save the neural network model."""

from torch import nn
import torch
from src.model.neural_net.network import NeuralNetworkAvgRatingRegressor
from src.model.neural_net.sweep import get_data_loaders

NEURAL_NET_MODEL_FILE = "models/nn.pkl"
NEURAL_NET_WEIGHTS_FILE = "models/nn_weights.pth"


if __name__ == "__main__":
    # Configuration based on a wandb sweep
    regressor = NeuralNetworkAvgRatingRegressor(
        hidden_layers=(256, 128, 64),
        dropout_rate=0.3,
        learning_rate=0.01,
        activation=nn.Tanh(),
        early_stopping_patience=5,
        early_stopping_delta=0.001,
        use_batch_norm=False,
        weight_decay=0.001,
    )

    epochs = 20
    train_set, val_set = get_data_loaders()
    regressor.fit(train_set, val_set, epochs=epochs)
    torch.save(regressor.model.state_dict(), NEURAL_NET_WEIGHTS_FILE)
    torch.save(regressor, NEURAL_NET_MODEL_FILE)
