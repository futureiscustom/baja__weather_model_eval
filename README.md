# Baja Weather Model Eval

A small, reproducible world-model/evaluation project for forecasting weather over Baja California Sur.

## Current benchmark result

**24-hour forecast, real WeatherBench 2 ERA5 data, 2018–2020**

| Metric | Persistence | TinyWeatherNet |
| --- | ---: | ---: |
| Normalized aggregate RMSE | 0.614698 | **0.478526** |
| 2m temperature RMSE | 1.3587 K | **1.1978 K** |
| 10m U wind RMSE | 1.9430 m/s | **1.5291 m/s** |
| 10m V wind RMSE | 2.1673 m/s | **1.5727 m/s** |
| Mean sea-level pressure RMSE | 201.3330 Pa | **163.6974 Pa** |

**Error reduction vs persistence: 22.15%**

- Held-out test samples: 870
- Trainable parameters: 29,476
- Input context: 4 × 6-hourly states
- Forecast horizon: 24 hours
- Benchmark data SHA-256: `18e0243d584a91b2162aace3abe4c745335dfe6960d81b3791118bb1a3c8bd6e`
- Seed: 7
- Training epochs: 20

This is a compact regional baseline, not an operational forecast system and not a claim of state-of-the-art WeatherBench performance. It uses a custom Baja regional held-out eval intended to give automated research systems a fixed, inexpensive optimization target.

Machine-readable result: [`results/era5_eval_only.json`](results/era5_eval_only.json)

Grant application draft: [`results/ENSUE_APPLICATION_READY.md`](results/ENSUE_APPLICATION_READY.md)

## What an eval is

An eval is a repeatable test:

`past weather -> model -> prediction -> compare with actual future weather -> score`

The main score in this repo is **RMSE (root mean squared error)**. Lower is better.

The first baseline is **persistence**: “the weather 24 hours from now will look exactly like the most recent observation.” TinyWeatherNet is useful only if it can beat that baseline on data it did not train on.

## Learn the eval loop first

Python 3.11+ recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_synthetic_eval.py
```

Synthetic mode is only for learning and software verification. Do not use its score as scientific or grant evidence.

For a plain-English walkthrough, read [`docs/EVALS_101.md`](docs/EVALS_101.md).

## Reproduce the real ERA5 benchmark

```bash
python scripts/prepare_era5_baja.py \
  --start 2018-01-01 \
  --end 2020-12-31T18:00:00

python scripts/train_era5.py \
  --epochs 20 \
  --hidden 48 \
  --seed 7

python scripts/eval_era5.py \
  --checkpoint artifacts/era5_tinyweathernet.pt
```

The standalone evaluator checks the prepared dataset SHA-256 stored inside the checkpoint. That prevents a candidate model from being silently scored on different data.

## Data

The project streams the public WeatherBench 2 ERA5 dataset:

`weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`

### Variables

- `2m_temperature`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `mean_sea_level_pressure`

### Region

Default crop:

- latitude: 20°N to 32°N
- longitude: 242°E to 254°E (118°W to 106°W)

At the WeatherBench 1.5° resolution this produces an 8×8 regional grid.

## Model

`TinyWeatherNet` is a deliberately small residual convolutional network.

It receives recent weather maps stacked as channels and predicts a correction to the latest state:

`future = latest_state + learned_change`

The baseline model has **29,476 trainable parameters**.

The goal is not to out-scale GraphCast, Pangu-Weather, NeuralGCM, or operational numerical weather prediction. The point is to create a cheap, inspectable model/eval loop where an automated research system can rapidly test architecture and training changes.

## Reproducibility rules

- Chronological train/test split, not random.
- Normalization statistics computed from training data only.
- Held-out test data are never used for fitting.
- Persistence and learned model are scored on identical targets.
- Fixed random seed where practical.
- Prepared benchmark data are SHA-256 hashed.
- Evaluation emits machine-readable JSON.
- CI compiles the code, runs unit tests, and runs both synthetic and real-data pipeline smoke tests.

## Repository structure

```text
baja_weather/
  data.py
  metrics.py
  model.py
  training.py
scripts/
  run_synthetic_eval.py
  prepare_era5_baja.py
  train_era5.py
  eval_era5.py
  render_grant_application.py
tests/
docs/
  EVALS_101.md
  PROJECT_PAGE.md
  GRANT_APPLICATION.md
  index.html
results/
  era5_eval_only.json
  era5_metrics.json
  ENSUE_APPLICATION_READY.md
artifacts/
  era5_tinyweathernet.pt
```

## Next research directions

1. Run multiple seeds and report mean ± standard deviation.
2. Add climatology and latitude-weighted baselines.
3. Try residual U-Nets and ConvLSTMs.
4. Test 6h, 12h, 24h and 48h lead times.
5. Add geopotential and upper-air fields.
6. Track forecast skill per parameter and inference cost.
7. Export forecasts in a WeatherBench-X-compatible format.

## Data and tooling references

- WeatherBench: https://sites.research.google/gr/weatherbench/
- WeatherBench 2 data guide: https://weatherbench2.readthedocs.io/en/latest/data-guide.html
- WeatherBench-X: https://github.com/google-research/weatherbenchX
- Google ARCO ERA5: https://github.com/google-research/arco-era5

## Status

The first real ERA5 grant baseline is complete and reproducible. The current held-out result is a **22.15% normalized-RMSE reduction versus persistence** for the 24-hour regional task.
