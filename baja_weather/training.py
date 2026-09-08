from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 7,
    device: torch.device | None = None,
) -> list[float]:
    seed_everything(seed)
    device = device or choose_device()
    model.to(device)

    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    losses: list[float] = []

    model.train()
    for epoch in range(epochs):
        running = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        epoch_loss = running / max(seen, 1)
        losses.append(epoch_loss)
        print(f"epoch {epoch + 1:02d}/{epochs}  train_mse={epoch_loss:.6f}")

    return losses


def predict_numpy(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or choose_device()
    model.to(device)
    model.eval()
    outputs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            outputs.append(model(xb).cpu().numpy())

    return np.concatenate(outputs, axis=0).astype(np.float32)
