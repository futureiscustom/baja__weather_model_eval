from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(pred - target)))


def per_channel_rmse(
    pred: np.ndarray,
    target: np.ndarray,
    channel_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    """RMSE for arrays shaped [N,C,H,W]."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("pred and target must share shape [N,C,H,W]")

    c = pred.shape[1]
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(c)]
    if len(channel_names) != c:
        raise ValueError("channel_names length must match channel dimension")

    return {
        str(name): float(np.sqrt(np.mean((pred[:, i] - target[:, i]) ** 2)))
        for i, name in enumerate(channel_names)
    }


def improvement_percent(model_score: float, baseline_score: float) -> float:
    """Positive means model error is lower than baseline error."""
    if baseline_score <= 0:
        raise ValueError("baseline_score must be > 0")
    return float(100.0 * (baseline_score - model_score) / baseline_score)
