from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, frames: np.ndarray) -> "ChannelNormalizer":
        """Fit per-channel stats from [time, channel, height, width]."""
        if frames.ndim != 4:
            raise ValueError("frames must have shape [time, channel, height, width]")
        mean = frames.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        std = frames.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        std = np.maximum(std, 1e-6)
        return cls(mean=mean, std=std)

    def transform_frames(self, frames: np.ndarray) -> np.ndarray:
        return ((frames - self.mean) / self.std).astype(np.float32)

    def inverse_frames(self, frames: np.ndarray) -> np.ndarray:
        return (frames * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.reshape(-1).tolist(),
            "std": self.std.reshape(-1).tolist(),
        }


def _smooth(field: np.ndarray, passes: int = 3) -> np.ndarray:
    out = field.astype(np.float32, copy=True)
    for _ in range(passes):
        out = (
            4.0 * out
            + np.roll(out, 1, axis=-1)
            + np.roll(out, -1, axis=-1)
            + np.roll(out, 1, axis=-2)
            + np.roll(out, -1, axis=-2)
        ) / 8.0
    return out


def generate_synthetic_weather(
    timesteps: int = 1400,
    channels: int = 4,
    height: int = 16,
    width: int = 16,
    seed: int = 7,
) -> np.ndarray:
    """Create a toy evolving spatial system for learning the forecast/eval loop.

    This is not meteorology and must not be presented as a scientific result.
    It is intentionally predictable enough that a tiny model can learn dynamics
    beyond a pure persistence forecast.
    """
    if timesteps < 20:
        raise ValueError("timesteps must be >= 20")

    rng = np.random.default_rng(seed)
    frames = np.empty((timesteps, channels, height, width), dtype=np.float32)
    state = _smooth(rng.normal(size=(channels, height, width)).astype(np.float32), 5)

    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    spatial = np.sin(np.pi * xx) * np.cos(np.pi * yy)

    for t in range(timesteps):
        phase = np.float32(np.sin(2 * np.pi * t / 24.0))
        advected = (
            0.90 * state
            + 0.045 * np.roll(state, 1, axis=-1)
            + 0.035 * np.roll(state, -1, axis=-2)
        )
        forcing = np.empty_like(state)
        for c in range(channels):
            forcing[c] = (0.012 + 0.004 * c) * phase * spatial
        noise = _smooth(rng.normal(scale=0.012, size=state.shape).astype(np.float32), 1)
        state = advected + forcing + noise
        frames[t] = state

    return frames


def make_windows(
    frames: np.ndarray,
    input_steps: int = 4,
    lead_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build X/y windows from [time, channel, height, width].

    The target is `lead_steps` time increments after the most recent input.
    X shape: [samples, input_steps, channel, height, width]
    y shape: [samples, channel, height, width]
    """
    if frames.ndim != 4:
        raise ValueError("frames must have shape [time, channel, height, width]")
    if input_steps < 1 or lead_steps < 1:
        raise ValueError("input_steps and lead_steps must be >= 1")

    n = len(frames) - input_steps - lead_steps + 1
    if n <= 0:
        raise ValueError("Not enough frames for requested window")

    xs = np.empty((n, input_steps, *frames.shape[1:]), dtype=np.float32)
    ys = np.empty((n, *frames.shape[1:]), dtype=np.float32)
    for i in range(n):
        xs[i] = frames[i : i + input_steps]
        target_index = i + input_steps - 1 + lead_steps
        ys[i] = frames[target_index]
    return xs, ys


def flatten_time_channels(x: np.ndarray) -> np.ndarray:
    """Convert [N,T,C,H,W] -> [N,T*C,H,W]."""
    if x.ndim != 5:
        raise ValueError("x must have shape [N,T,C,H,W]")
    n, t, c, h, w = x.shape
    return x.reshape(n, t * c, h, w).astype(np.float32)


def chronological_split(frames: np.ndarray, train_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    cut = int(len(frames) * train_fraction)
    return frames[:cut], frames[cut:]
