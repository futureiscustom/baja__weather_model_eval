import numpy as np

from baja_weather.metrics import improvement_percent, mae, per_channel_rmse, rmse


def test_rmse_and_mae():
    target = np.array([0.0, 2.0], dtype=np.float32)
    pred = np.array([0.0, 0.0], dtype=np.float32)
    assert np.isclose(rmse(pred, target), np.sqrt(2.0))
    assert np.isclose(mae(pred, target), 1.0)


def test_per_channel_rmse():
    target = np.zeros((2, 2, 3, 3), dtype=np.float32)
    pred = np.zeros_like(target)
    pred[:, 0] = 2.0
    pred[:, 1] = 3.0
    scores = per_channel_rmse(pred, target, ["a", "b"])
    assert np.isclose(scores["a"], 2.0)
    assert np.isclose(scores["b"], 3.0)


def test_improvement_percent():
    assert np.isclose(improvement_percent(0.8, 1.0), 20.0)
