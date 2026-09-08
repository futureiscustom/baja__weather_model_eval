from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baja_weather.data import ChannelNormalizer, chronological_split, flatten_time_channels, make_windows
from baja_weather.metrics import improvement_percent, per_channel_rmse, rmse
from baja_weather.model import TinyWeatherNet, count_parameters
from baja_weather.training import choose_device, predict_numpy, train_model

UNITS = {
    "2m_temperature": "K",
    "10m_u_component_of_wind": "m s^-1",
    "10m_v_component_of_wind": "m s^-1",
    "mean_sea_level_pressure": "Pa",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyWeatherNet on the cropped Baja ERA5 benchmark.")
    parser.add_argument("--data", default=str(ROOT / "data/era5_baja.npz"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--input-steps", type=int, default=4, help="Number of 6-hour states used as context")
    parser.add_argument("--lead-steps", type=int, default=4, help="6-hour increments after latest input; 4 = 24h")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit("ERA5 data not found. Run: python scripts/prepare_era5_baja.py")

    loaded = np.load(data_path, allow_pickle=False)
    frames = loaded["frames"].astype(np.float32)
    variables = [str(v) for v in loaded["variables"].tolist()]
    if frames.ndim != 4:
        raise RuntimeError(f"Expected [time,channel,lat,lon], got {frames.shape}")

    train_frames, test_frames = chronological_split(frames, train_fraction=args.train_fraction)
    normalizer = ChannelNormalizer.fit(train_frames)
    train_norm = normalizer.transform_frames(train_frames)
    test_norm = normalizer.transform_frames(test_frames)

    x_train, y_train = make_windows(train_norm, args.input_steps, args.lead_steps)
    x_test, y_test = make_windows(test_norm, args.input_steps, args.lead_steps)
    x_train = flatten_time_channels(x_train)
    x_test = flatten_time_channels(x_test)

    channels = frames.shape[1]
    persistence_norm = x_test[:, -channels:, :, :]
    persistence_norm_rmse = rmse(persistence_norm, y_test)

    device = choose_device()
    model = TinyWeatherNet(channels=channels, input_steps=args.input_steps, hidden=args.hidden)
    print(f"device: {device}")
    print(f"data: {data_path}")
    print(f"frames: {frames.shape} | variables: {variables}")
    print(f"train samples: {len(x_train)} | test samples: {len(x_test)}")
    print(f"lead: {args.lead_steps * 6} hours | parameters: {count_parameters(model):,}")

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

    pred_norm = predict_numpy(model, x_test, device=device)
    model_norm_rmse = rmse(pred_norm, y_test)
    improvement = improvement_percent(model_norm_rmse, persistence_norm_rmse)

    pred_physical = normalizer.inverse_frames(pred_norm)
    target_physical = normalizer.inverse_frames(y_test)
    persistence_physical = normalizer.inverse_frames(persistence_norm)
    model_per_var = per_channel_rmse(pred_physical, target_physical, variables)
    persistence_per_var = per_channel_rmse(persistence_physical, target_physical, variables)

    print("\n=== ERA5 HELD-OUT EVAL SCOREBOARD ===")
    print(f"Persistence normalized RMSE: {persistence_norm_rmse:.6f}")
    print(f"TinyWeatherNet normalized RMSE: {model_norm_rmse:.6f}")
    print(f"Improvement vs persistence: {improvement:.2f}%")
    print("\nPer-variable RMSE (native units):")
    for name in variables:
        print(
            f"  {name}: model={model_per_var[name]:.4f} {UNITS.get(name, '')} | "
            f"persistence={persistence_per_var[name]:.4f} {UNITS.get(name, '')}"
        )

    data_hash = sha256_file(data_path)
    ROOT.joinpath("artifacts").mkdir(exist_ok=True)
    ROOT.joinpath("results").mkdir(exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "channels": channels,
        "variables": variables,
        "input_steps": args.input_steps,
        "lead_steps": args.lead_steps,
        "hidden": args.hidden,
        "train_fraction": args.train_fraction,
        "normalizer": normalizer.to_dict(),
        "seed": args.seed,
        "data_sha256": data_hash,
    }
    checkpoint_path = ROOT / "artifacts/era5_tinyweathernet.pt"
    torch.save(checkpoint, checkpoint_path)

    result = {
        "dataset": "WeatherBench 2 ERA5 Baja crop",
        "data_sha256": data_hash,
        "variables": variables,
        "units": {name: UNITS.get(name, "") for name in variables},
        "input_steps": args.input_steps,
        "lead_steps": args.lead_steps,
        "lead_hours": args.lead_steps * 6,
        "train_fraction": args.train_fraction,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "seed": args.seed,
        "epochs": args.epochs,
        "parameters": count_parameters(model),
        "final_train_mse": losses[-1],
        "persistence_normalized_rmse": persistence_norm_rmse,
        "model_normalized_rmse": model_norm_rmse,
        "improvement_percent": improvement,
        "model_per_variable_rmse": model_per_var,
        "persistence_per_variable_rmse": persistence_per_var,
    }
    metrics_path = ROOT / "results/era5_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nSaved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print("Re-run fixed eval with: python scripts/eval_era5.py --checkpoint artifacts/era5_tinyweathernet.pt")


if __name__ == "__main__":
    main()
