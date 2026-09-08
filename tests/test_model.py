import torch

from baja_weather.model import TinyWeatherNet, count_parameters


def test_model_shape_and_parameters():
    model = TinyWeatherNet(channels=4, input_steps=4, hidden=16)
    x = torch.zeros(2, 16, 8, 8)
    y = model(x)
    assert y.shape == (2, 4, 8, 8)
    assert count_parameters(model) > 0
