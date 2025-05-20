import torch
import pandas as pd
import torch.nn as nn
import torch.utils.data as data
from src.model.neural_net.network import NeuralNetworkAvgRatingRegressor
from src.scripts.split_dataset import TRAIN_SET_FILE
import wandb
import numpy as np
from typing import Any
import os


class WandbGridSearch:

    def __init__(
        self,
        project_name: str,
        sweep_config: dict[str, Any],
        data_loader_fn,
        epochs: int = 100,
        device: str = "cpu",
    ):
        """
        Args:
            project_name: Name of the W&B project
            sweep_config: Configuration for the W&B sweep
            data_loader_fn: Function that returns train and validation data loaders
            epochs: Maximum number of epochs to train each model
            device: Device to use for training ('cpu' or 'cuda')
        """
        self.project_name = project_name
        self.sweep_config = sweep_config
        self.data_loader_fn = data_loader_fn
        self.epochs = epochs
        self.device = device
        self.sweep_id = None

    def train(self, config=None):
        with wandb.init(config=config) as run:
            config = wandb.config

            train_loader, val_loader = self.data_loader_fn()

            if config.activation == "relu":
                activation = nn.ReLU()
            elif config.activation == "leaky_relu":
                activation = nn.LeakyReLU(negative_slope=config.negative_slope)
            elif config.activation == "tanh":
                activation = nn.Tanh()
            else:
                activation = nn.ReLU()

            if config.loss_function == "mse":
                loss_function = nn.MSELoss()
            elif config.loss_function == "mae":
                loss_function = nn.L1Loss()
            elif config.loss_function == "huber":
                loss_function = nn.SmoothL1Loss()
            else:
                loss_function = nn.MSELoss()

            hidden_layers = tuple(map(int, config.hidden_layers.split(",")))

            model = NeuralNetworkAvgRatingRegressor(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                activation=activation,
                loss_function=loss_function,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                dropout_rate=config.dropout_rate,
                use_batch_norm=config.use_batch_norm,
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_delta=config.early_stopping_delta,
                device=self.device,
            )

            for epoch in range(self.epochs):
                train_losses = []
                for batch_x, batch_y in train_loader:
                    loss = model.train_step(batch_x, batch_y)
                    train_losses.append(loss)

                avg_train_loss = np.mean(train_losses)

                val_losses = []
                for batch_x, batch_y in val_loader:
                    val_loss = model.evaluate(batch_x, batch_y)
                    val_losses.append(val_loss)

                avg_val_loss = np.mean(val_losses)

                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "learning_rate": model.optimizer.param_groups[0]["lr"],
                    }
                )

                if (
                    hasattr(model, "early_stopping")
                    and model.early_stopping is not None
                ):
                    if model.early_stopping.early_stop:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            model_path = f"model_{run.id}.pt"
            model.save_model(model_path)

            artifact = wandb.Artifact(f"model-{run.id}", type="model")
            artifact.add_file(model_path)
            run.log_artifact(artifact)

            if os.path.exists(model_path):
                os.remove(model_path)

    def start_sweep(self):
        """
        Start the W&B sweep
        """
        # Initialize the sweep
        self.sweep_id = wandb.sweep(self.sweep_config, project=self.project_name)

        # Start the sweep agent
        wandb.agent(self.sweep_id, function=self.train)

        return self.sweep_id


def create_sample_sweep_config():
    """
    Create a sample sweep configuration for the neural network
    """
    sweep_config = {
        "method": "random",
        "name": "nn-sweep",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "hidden_layers": {
                "values": ["64,32", "128,64", "256,128,64", "512,256,128"]
            },
            "activation": {"values": ["relu", "leaky_relu", "tanh"]},
            "negative_slope": {"values": [0.01, 0.1]},
            "learning_rate": {"values": [0.001, 0.01, 0.0001]},
            "weight_decay": {"values": [0, 0.0001, 0.001]},
            "dropout_rate": {"values": [0, 0.2, 0.5]},
            "use_batch_norm": {"values": [True, False]},
            "loss_function": {"values": ["mse", "mae", "huber"]},
            "early_stopping_patience": {"values": [5, 10, 15]},
            "early_stopping_delta": {"values": [0, 0.001]},
        },
    }

    return sweep_config


def get_dataset_from_csv(file_path: str):
    df = pd.read_csv(file_path)
    X = df.drop(columns=["avg_rating"])
    Y = df["avg_rating"]
    X = torch.tensor(X.values, dtype=torch.float32)
    Y = torch.tensor(Y.values, dtype=torch.float32)
    return data.TensorDataset(X, Y)


def get_data_loaders():
    dataset = get_dataset_from_csv(TRAIN_SET_FILE)
    train_dataset, val_dataset = data.random_split(dataset, [0.8, 0.2])
    return (
        data.DataLoader(train_dataset, batch_size=32, shuffle=True),
        data.DataLoader(val_dataset, batch_size=32, shuffle=False),
    )


if __name__ == "__main__":

    wandb.login()

    sweep_config = create_sample_sweep_config()

    grid_search = WandbGridSearch(
        project_name="ium-projekt",
        sweep_config=sweep_config,
        data_loader_fn=get_data_loaders,
        epochs=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    sweep_id = grid_search.start_sweep()
    print(f"Sweep started with ID: {sweep_id}")
