from dataclasses import dataclass, field
import torch.nn as nn
import numpy as np
import torch


@dataclass
class EarlyStopping:
    patience: int = field(default=5)
    min_delta: float = field(default=0)
    verbose: bool = field(default=False)

    counter: int = field(default=0, init=False)
    best_score: float | None = field(default=None, init=False)
    early_stop: bool = field(default=False, init=False)
    val_loss_min: float = field(default=np.inf, init=False)

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        self.best_model = model.state_dict().copy()
        self.val_loss_min = val_loss

    def get_best_model(self) -> dict[str, torch.Tensor]:
        return self.best_model
