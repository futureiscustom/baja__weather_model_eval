from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gcsfs
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = "weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream and crop public WeatherBench 2 ERA5 data to Baja.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2020-12-31T18:00:00")
    parser.add_argument("--lat-min", type=float, default=20.0)
    parser.add_argument("--lat-max", type=float, default=32.0)
    parser.add_argument("--lon-min", type=float, default=242.0)
    parser.add_argument("--lon-max", type=float, default=254.0)
    parser.add_argument("--output", default=str(ROOT / "data/era5_baja.npz"))
    args = parser.parse_args()

    print("Opening public WeatherBench 2 ERA5 dataset (anonymous GCS)...")
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(DATASET)
    ds = xr.open_zarr(mapper, consolidated=True, chunks=None)

    missing = [v for v in VARIABLES if v not in ds.data_vars]
    if missing:
        raise RuntimeError(f"Expected variables not present in dataset: {missing}")

    lat_values = np.asarray(ds.latitude.values)
    lon_values = np.asarray(ds.longitude.values)
    lat_idx = np.where((lat_values >= args.lat_min) & (lat_values <= args.lat_max))[0]
    lon_idx = np.where((lon_values >= args.lon_min) & (lon_values <= args.lon_max))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise RuntimeError("Requested crop contains no grid points")

    subset = ds[VARIABLES].sel(time=slice(args.start, args.end)).isel(
        latitude=lat_idx,
        longitude=lon_idx,
    )

    if subset.sizes.get("time", 0) < 100:
        raise RuntimeError("Too few timesteps selected; choose a longer date range")

    print(
        f"Loading {subset.sizes['time']} timesteps x {len(VARIABLES)} variables x "
        f"{subset.sizes['latitude']}x{subset.sizes['longitude']} Baja grid..."
    )

    arrays = []
    for name in VARIABLES:
        values = subset[name].transpose("time", "latitude", "longitude").values.astype(np.float32)
        arrays.append(values)
    frames = np.stack(arrays, axis=1)

    if not np.isfinite(frames).all():
        bad = int((~np.isfinite(frames)).sum())
        raise RuntimeError(f"ERA5 crop contains {bad} NaN/Inf values; adjust dates/variables before training")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=frames,
        time=np.asarray(subset.time.values).astype("datetime64[ns]"),
        latitude=np.asarray(subset.latitude.values, dtype=np.float32),
        longitude=np.asarray(subset.longitude.values, dtype=np.float32),
        variables=np.asarray(VARIABLES),
        source=np.asarray(["WeatherBench 2 ERA5 240x121 (1.5 degree)"]),
    )

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Saved {output} ({size_mb:.1f} MB)")
    print(f"frames shape: {frames.shape} = [time, channel, latitude, longitude]")
    print("Next: python scripts/train_era5.py")


if __name__ == "__main__":
    main()
