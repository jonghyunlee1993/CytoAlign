"""Reusable PyTorch multi-output regressor."""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np


class MLPRegressor:
    def __init__(
        self,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        epochs: int = 30,
        batch_size: int = 4096,
        learning_rate: float = 1.0e-3,
        patience: int = 5,
        random_state: int = 42,
    ):
        self.hidden_dims = tuple(map(int, hidden_dims))
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.patience = int(patience)
        self.random_state = int(random_state)

    def _network(self, input_dim: int, output_dim: int):
        from torch import nn

        layers = []
        width = input_dim
        for hidden in self.hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.GELU()))
            width = hidden
        layers.append(nn.Linear(width, output_dim))
        return nn.Sequential(*layers)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        groups: Sequence,
        device: str,
    ) -> "MLPRegressor":
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        group_values = np.asarray(groups).astype(str)
        self.feature_mean_ = x.mean(axis=0)
        self.feature_scale_ = x.std(axis=0)
        self.feature_scale_[self.feature_scale_ < 1.0e-6] = 1.0
        self.target_mean_ = y.mean(axis=0)
        self.target_scale_ = y.std(axis=0)
        self.target_scale_[self.target_scale_ < 1.0e-6] = 1.0
        scaled_x = (x - self.feature_mean_) / self.feature_scale_
        scaled_y = (y - self.target_mean_) / self.target_scale_

        rng = np.random.RandomState(self.random_state)
        unique_groups = np.unique(group_values)
        validation_count = max(1, int(round(0.1 * unique_groups.size)))
        validation_groups = set(
            rng.permutation(unique_groups)[:validation_count].tolist()
        )
        validation = np.asarray(
            [group in validation_groups for group in group_values], dtype=bool
        )
        training = ~validation

        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(scaled_x[training]),
                torch.from_numpy(scaled_y[training]),
            ),
            batch_size=min(self.batch_size, int(training.sum())),
            shuffle=True,
            generator=generator,
        )
        self.model_ = self._network(x.shape[1], y.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=1.0e-5,
        )
        loss_fn = nn.SmoothL1Loss()
        validation_x = torch.from_numpy(scaled_x[validation]).to(device)
        validation_y = torch.from_numpy(scaled_y[validation]).to(device)
        best_loss = float("inf")
        best_state = None
        stale = 0
        self.history_ = []
        for epoch in range(self.epochs):
            self.model_.train()
            total = 0.0
            rows = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * batch_x.shape[0]
                rows += batch_x.shape[0]
            self.model_.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_fn(self.model_(validation_x), validation_y)
                )
            self.history_.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total / rows,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1.0e-5:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_validation_loss_ = best_loss
        return self

    def predict(
        self,
        features: np.ndarray,
        *,
        device: str,
        batch_size: int = 8192,
    ) -> np.ndarray:
        import torch

        x = np.asarray(features, dtype=np.float32)
        scaled = (x - self.feature_mean_) / self.feature_scale_
        self.model_.to(device).eval()
        pieces = []
        with torch.no_grad():
            for start in range(0, len(scaled), batch_size):
                batch = torch.from_numpy(scaled[start : start + batch_size]).to(device)
                pieces.append(self.model_(batch).cpu().numpy())
        prediction = np.concatenate(pieces)
        return (
            prediction * self.target_scale_[None, :] + self.target_mean_[None, :]
        ).astype(np.float32)

    def cpu(self) -> "MLPRegressor":
        self.model_.cpu()
        return self
