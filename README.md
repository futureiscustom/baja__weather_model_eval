# Baja Weather Model Eval

A small, reproducible world-model/evaluation project for forecasting weather over Baja California Sur.

The project has two modes:

1. **Learn the eval loop with synthetic weather** — no data download or API key required.
2. **Run the same model/eval on real ERA5 data** from the public WeatherBench 2 Google Cloud bucket.

The goal is deliberately simple: give a model recent atmospheric states, predict a future state, and measure whether it beats a naive persistence forecast.

## What an eval is

An eval is just a repeatable test:

`past weather -> model -> prediction -> compare with actual future weather -> score`

For this repo, the main score is **RMSE (root mean squared error)**. Lower is better.

The first benchmark is **persistence**: "the weather 24 hours from now will look exactly like the most recent observation." A useful learned model should beat that baseline on held-out data.

## Quick start: learn without downloading weather data

Python 3.11+ recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_synthetic_eval.py
```

You should see a scoreboard similar to:

```text
Persistence normalized RMSE: ...
TinyWeatherNet normalized RMSE: ...
Improvement vs persistence: ... %
```

Exact values depend on platform/library versions, so the repo never hard-codes a claimed result. The generated result is saved to `results/synthetic_metrics.json`.

## Real ERA5 / WeatherBench 2 experiment

WeatherBench 2 publishes cloud-hosted ERA5 datasets at several resolutions. This repo uses the 1.5-degree `240x121` dataset for a manageable first experiment and crops it to Baja California Sur plus surrounding ocean.

Install dependencies, then:

```bash
python scripts/prepare_era5_baja.py \
  --start 2018-01-01 \
  --end 2020-12-31

python scripts/train_era5.py

python scripts/eval_era5.py \
  --checkpoint artifacts/era5_tinyweathernet.pt
```

The default task uses four recent 6-hourly states as input and predicts the state **24 hours after the latest input**.

### Variables

- `2m_temperature`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `mean_sea_level_pressure`

### Region

Default crop:

- latitude: 20°N to 32°N
- longitude: 242°E to 254°E (equivalent to 118°W to 106°W)

This includes Baja California Sur and enough surrounding Pacific/Gulf context to make the regional task more meaningful.

## Grant benchmark

The grant-ready benchmark is:

> 24-hour regional forecast error on a chronological held-out ERA5 test split for 2m temperature, 10m U/V wind and mean sea-level pressure. Report per-variable RMSE in native physical units, normalized aggregate RMSE, model parameter count, and percentage improvement versus persistence.

This is intentionally small. It gives an optimization system a concrete model, a fixed evaluation and an unambiguous objective.

## Repository structure

```text
baja_weather/
  data.py          synthetic data, windows, normalization
  metrics.py       RMSE/MAE and per-variable scores
  model.py         TinyWeatherNet convolutional model
scripts/
  run_synthetic_eval.py
  prepare_era5_baja.py
  train_era5.py
  eval_era5.py
tests/
  test_data.py
  test_metrics.py
docs/
  EVALS_101.md
artifacts/         generated checkpoints (gitignored)
results/           generated metrics (JSON files gitignored)
```

## The model

`TinyWeatherNet` is deliberately tiny. It stacks recent weather maps as channels, predicts a correction to the latest state, and uses a few 2D convolutions. It is not intended to compete with GraphCast or operational weather systems. It is a clean baseline that is cheap enough to train, inspect and optimize.

That is useful for an automated-research grant: there is a measurable starting point and plenty of room to change architecture, loss, input history, lead time, normalization and training strategy.

## Reproducibility rules

- Train/test splits are chronological, not random.
- Normalization statistics come only from the training split.
- Test data are never used for fitting.
- Persistence is evaluated on exactly the same targets as the learned model.
- Random seeds are fixed where practical.
- Evaluation produces machine-readable JSON.

## Next research directions

Once the baseline is running:

1. Try residual U-Nets or ConvLSTMs.
2. Add geopotential/upper-air fields.
3. Compare 6h, 12h, 24h and 48h lead times.
4. Add latitude-weighted RMSE and climatology baselines.
5. Run multiple seeds and report mean/std.
6. Export predictions in a WeatherBench-compatible format and evaluate with WeatherBench-X.

## Data sources

- WeatherBench: https://sites.research.google/gr/weatherbench/
- WeatherBench 2 data guide: https://weatherbench2.readthedocs.io/en/latest/data-guide.html
- WeatherBench-X: https://github.com/google-research/weatherbenchX
- Google ARCO ERA5: https://github.com/google-research/arco-era5

## Project status

Early research baseline. Synthetic mode is for learning and software verification; grant claims should use the held-out **real ERA5** results produced by `train_era5.py` + `eval_era5.py`.