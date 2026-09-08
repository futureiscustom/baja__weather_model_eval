import numpy as np

from baja_weather.data import ChannelNormalizer, flatten_time_channels, generate_synthetic_weather, make_windows


def test_window_shapes_and_target_index():
    frames = np.arange(12 * 2 * 3 * 4, dtype=np.float32).reshape(12, 2, 3, 4)
    x, y = make_windows(frames, input_steps=3, lead_steps=2)
    assert x.shape == (8, 3, 2, 3, 4)
    assert y.shape == (8, 2, 3, 4)
    np.testing.assert_array_equal(x[0, -1], frames[2])
    np.testing.assert_array_equal(y[0], frames[4])


def test_flatten_time_channels():
    x = np.zeros((5, 4, 3, 8, 9), dtype=np.float32)
    flat = flatten_time_channels(x)
    assert flat.shape == (5, 12, 8, 9)


def test_normalizer_roundtrip():
    frames = generate_synthetic_weather(timesteps=50, channels=4, height=8, width=8)
    normalizer = ChannelNormalizer.fit(frames)
    restored = normalizer.inverse_frames(normalizer.transform_frames(frames))
    np.testing.assert_allclose(restored, frames, rtol=1e-5, atol=1e-5)
