import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.model.neural_net.early_stopping import EarlyStopping


class CustomNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 15,
        hidden_layers: tuple[int, ...] = (64, 32),
        activation: nn.Module = nn.LeakyReLU(negative_slope=0.01),
        dropout_rate: float = 0.3,
        use_batch_norm: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super(CustomNN, self).__init__()

        self.device = device
        self.activation = activation

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(activation)

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)
        self.to(device)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NeuralNetworkAvgRatingRegressor:
    def __init__(
        self,
        input_dim: int = 15,
        hidden_layers: tuple[int, ...] = (64, 32),
        activation: nn.Module = nn.LeakyReLU(negative_slope=0.01),
        loss_function: nn.Module = nn.MSELoss(),
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        early_stopping_patience: int = 0,
        early_stopping_delta: float = 0.0,
        device: str = "cpu",
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

        self.model = CustomNN(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            device=self.device,
        )

        self.loss_function = loss_function

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self.early_stopping = None
        if early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_delta,
                verbose=True,
            )

        self.history = {"train_loss": [], "val_loss": [], "lr": []}

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.to(self.device)
        y = y.to(self.device)

        self.model.train()

        self.optimizer.zero_grad()

        outputs = self.model(x)

        loss = self.loss_function(outputs, y.view(-1, 1))

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.to(self.device)
        y = y.to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(x)
            loss = self.loss_function(outputs, y.view(-1, 1))

        return loss.item()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(x)

        return outputs.cpu()

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None = None,
        epochs: int = 100,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        for epoch in range(epochs):
            train_losses = []
            for batch_x, batch_y in train_loader:
                loss = self.train_step(batch_x, batch_y)
                train_losses.append(loss)

            avg_train_loss = np.mean(train_losses)
            self.history["train_loss"].append(avg_train_loss)

            if val_loader is not None:
                val_losses = []
                for batch_x, batch_y in val_loader:
                    val_loss = self.evaluate(batch_x, batch_y)
                    val_losses.append(val_loss)

                avg_val_loss = np.mean(val_losses)
                self.history["val_loss"].append(avg_val_loss)

                self.scheduler.step(avg_val_loss)

                if self.early_stopping is not None:
                    if self.early_stopping(
                        avg_val_loss, self.model  # pyright: ignore[reportArgumentType]
                    ):
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        self.model.load_state_dict(self.early_stopping.get_best_model())
                        break

            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if verbose and (epoch + 1) % 10 == 0:
                if val_loader is not None:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                    )
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")

        return self.history

    def save_model(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint["history"]
