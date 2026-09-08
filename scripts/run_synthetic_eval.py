from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baja_weather.data import (
    ChannelNormalizer,
    chronological_split,
    flatten_time_channels,
    generate_synthetic_weather,
    make_windows,
)
from baja_weather.metrics import improvement_percent, rmse
from baja_weather.model import TinyWeatherNet, count_parameters
from baja_weather.training import choose_device, predict_numpy, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn the model/eval loop on toy weather dynamics.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    input_steps = 4
    lead_steps = 4
    channels = 4

    frames = generate_synthetic_weather(timesteps=args.timesteps, channels=channels, seed=args.seed)
    train_frames, test_frames = chronological_split(frames, train_fraction=0.8)
    normalizer = ChannelNormalizer.fit(train_frames)
    train_norm = normalizer.transform_frames(train_frames)
    test_norm = normalizer.transform_frames(test_frames)

    x_train, y_train = make_windows(train_norm, input_steps=input_steps, lead_steps=lead_steps)
    x_test, y_test = make_windows(test_norm, input_steps=input_steps, lead_steps=lead_steps)
    x_train = flatten_time_channels(x_train)
    x_test = flatten_time_channels(x_test)

    persistence = x_test[:, -channels:, :, :]
    persistence_rmse = rmse(persistence, y_test)

    device = choose_device()
    model = TinyWeatherNet(channels=channels, input_steps=input_steps, hidden=48)
    print(f"device: {device}")
    print(f"train samples: {len(x_train)} | test samples: {len(x_test)}")
    print(f"trainable parameters: {count_parameters(model):,}")

    losses = train_model(
        model,
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=32,
        learning_rate=1e-3,
        seed=args.seed,
        device=device,
    )
    prediction = predict_numpy(model, x_test, device=device)
    model_rmse = rmse(prediction, y_test)
    improvement = improvement_percent(model_rmse, persistence_rmse)

    print("\n=== SYNTHETIC EVAL SCOREBOARD ===")
    print(f"Persistence normalized RMSE: {persistence_rmse:.6f}")
    print(f"TinyWeatherNet normalized RMSE: {model_rmse:.6f}")
    print(f"Improvement vs persistence: {improvement:.2f}%")
    print("NOTE: synthetic scores are for learning/software verification, not grant evidence.")

    ROOT.joinpath("results").mkdir(exist_ok=True)
    ROOT.joinpath("artifacts").mkdir(exist_ok=True)
    result = {
        "dataset": "synthetic",
        "seed": args.seed,
        "input_steps": input_steps,
        "lead_steps": lead_steps,
        "persistence_normalized_rmse": persistence_rmse,
        "model_normalized_rmse": model_rmse,
        "improvement_percent": improvement,
        "parameters": count_parameters(model),
        "epochs": args.epochs,
        "final_train_mse": losses[-1],
        "device": str(device),
    }
    ROOT.joinpath("results/synthetic_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(
        {
            "model_state": model.state_dict(),
            "channels": channels,
            "input_steps": input_steps,
            "hidden": 48,
            "normalizer": normalizer.to_dict(),
            "seed": args.seed,
        },
        ROOT / "artifacts/synthetic_tinyweathernet.pt",
    )


if __name__ == "__main__":
    main()
