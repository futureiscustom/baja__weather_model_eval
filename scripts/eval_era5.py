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
from baja_weather.training import choose_device, predict_numpy

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
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the fixed chronological ERA5 test split.")
    parser.add_argument("--data", default=str(ROOT / "data/era5_baja.npz"))
    parser.add_argument("--checkpoint", default=str(ROOT / "artifacts/era5_tinyweathernet.pt"))
    parser.add_argument("--output", default=str(ROOT / "results/era5_eval_only.json"))
    parser.add_argument("--allow-data-mismatch", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    if not data_path.exists():
        raise SystemExit(f"Missing data: {data_path}")
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    actual_hash = sha256_file(data_path)
    expected_hash = ckpt.get("data_sha256")
    if expected_hash and actual_hash != expected_hash and not args.allow_data_mismatch:
        raise SystemExit(
            "Data SHA-256 does not match the checkpoint's benchmark dataset. "
            "Use the original data, or pass --allow-data-mismatch only for exploratory work."
        )

    loaded = np.load(data_path, allow_pickle=False)
    frames = loaded["frames"].astype(np.float32)
    variables = [str(v) for v in loaded["variables"].tolist()]

    ckpt_variables = [str(v) for v in ckpt["variables"]]
    if variables != ckpt_variables:
        raise SystemExit(f"Variable mismatch. data={variables}, checkpoint={ckpt_variables}")

    train_frames, test_frames = chronological_split(frames, float(ckpt["train_fraction"]))
    channels = int(ckpt["channels"])
    mean = np.asarray(ckpt["normalizer"]["mean"], dtype=np.float32).reshape(1, channels, 1, 1)
    std = np.asarray(ckpt["normalizer"]["std"], dtype=np.float32).reshape(1, channels, 1, 1)
    normalizer = ChannelNormalizer(mean=mean, std=std)
    test_norm = normalizer.transform_frames(test_frames)

    x_test, y_test = make_windows(
        test_norm,
        input_steps=int(ckpt["input_steps"]),
        lead_steps=int(ckpt["lead_steps"]),
    )
    x_test = flatten_time_channels(x_test)
    persistence_norm = x_test[:, -channels:, :, :]

    model = TinyWeatherNet(
        channels=channels,
        input_steps=int(ckpt["input_steps"]),
        hidden=int(ckpt["hidden"]),
    )
    model.load_state_dict(ckpt["model_state"])
    device = choose_device()
    pred_norm = predict_numpy(model, x_test, device=device)

    persistence_norm_rmse = rmse(persistence_norm, y_test)
    model_norm_rmse = rmse(pred_norm, y_test)
    improvement = improvement_percent(model_norm_rmse, persistence_norm_rmse)

    pred_physical = normalizer.inverse_frames(pred_norm)
    target_physical = normalizer.inverse_frames(y_test)
    persistence_physical = normalizer.inverse_frames(persistence_norm)
    model_per_var = per_channel_rmse(pred_physical, target_physical, variables)
    persistence_per_var = per_channel_rmse(persistence_physical, target_physical, variables)

    result = {
        "benchmark": "Baja ERA5 held-out chronological test split",
        "data_sha256": actual_hash,
        "checkpoint": checkpoint_path.name,
        "variables": variables,
        "units": {name: UNITS.get(name, "") for name in variables},
        "input_steps": int(ckpt["input_steps"]),
        "lead_steps": int(ckpt["lead_steps"]),
        "lead_hours": int(ckpt["lead_steps"]) * 6,
        "test_samples": len(x_test),
        "parameters": count_parameters(model),
        "persistence_normalized_rmse": persistence_norm_rmse,
        "model_normalized_rmse": model_norm_rmse,
        "improvement_percent": improvement,
        "model_per_variable_rmse": model_per_var,
        "persistence_per_variable_rmse": persistence_per_var,
    }

    print("=== FIXED ERA5 EVAL ===")
    print(f"data sha256: {actual_hash}")
    print(f"test samples: {len(x_test)}")
    print(f"lead: {result['lead_hours']} hours")
    print(f"parameters: {result['parameters']:,}")
    print(f"persistence normalized RMSE: {persistence_norm_rmse:.6f}")
    print(f"model normalized RMSE: {model_norm_rmse:.6f}")
    print(f"improvement vs persistence: {improvement:.2f}%")
    for name in variables:
        print(
            f"{name}: model={model_per_var[name]:.4f} {UNITS.get(name, '')}; "
            f"persistence={persistence_per_var[name]:.4f} {UNITS.get(name, '')}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
