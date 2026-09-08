# Ensue World Model Grant Application

> Baseline verified by a successful GitHub Actions run on September 8, 2026. Replace only `[YOUR NAME]` and confirm the public project URL after enabling GitHub Pages.

**Name:** [YOUR NAME]

**Affiliation:** Future Is Custom

**GitHub:** https://github.com/futureiscustom/baja__weather_model_eval

**Public project link:** https://futureiscustom.github.io/baja__weather_model_eval/

## Project: Baja Weather Model Eval

Baja Weather Model Eval is an open, reproducible regional world-model experiment for Baja California Sur. It uses public ERA5 ground truth distributed through WeatherBench 2. TinyWeatherNet, the initial baseline, uses 4 recent six-hourly atmospheric states and predicts the state 24 hours after the latest input. The initial variables are 2m temperature, 10m U/V wind, and mean sea-level pressure.

## Benchmark / eval

The benchmark uses a chronological held-out ERA5 test split over a fixed Baja regional crop. The primary score is normalized aggregate RMSE; we also report per-variable RMSE in physical units, parameter count, and improvement versus a persistence forecast evaluated on the identical targets. The prepared data are SHA-256 hashed and the standalone evaluation command checks the hash before scoring.

- Data SHA-256: `18e0243d584a91b2162aace3abe4c745335dfe6960d81b3791118bb1a3c8bd6e`
- Forecast horizon: 24 hours
- Held-out samples: 870
- Trainable parameters: 29,476
- Persistence normalized RMSE: 0.614698
- TinyWeatherNet normalized RMSE: 0.478526
- Improvement vs persistence: 22.15%

### Per-variable RMSE

- 2m_temperature: model 1.1978 K; persistence 1.3587 K
- 10m_u_component_of_wind: model 1.5291 m s^-1; persistence 1.9430 m s^-1
- 10m_v_component_of_wind: model 1.5727 m s^-1; persistence 2.1673 m s^-1
- mean_sea_level_pressure: model 163.6974 Pa; persistence 201.3330 Pa

## What we want to optimize with Ensue

We want to improve forecast skill per unit of model complexity. The fixed eval makes it possible to explore architecture depth/width, receptive field, U-Net or ConvLSTM alternatives, input-history length, regional context, loss design, additional ERA5 variables, training strategy, and parameter/inference-cost constraints while preserving an apples-to-apples held-out benchmark. Successful changes can be measured immediately and reproduced from the repo.

## Why this project

Most high-profile weather world models are large global systems. This project asks how much useful state-transition modeling can be extracted from a compact regional model that an independent builder can train and iterate on. Baja California Sur is a useful test region because the Pacific, Gulf of California, desert terrain and tropical systems interact across a compact geography. The grant output would be reproducible experiment history, benchmark results, and model improvements committed back to the open repository.
