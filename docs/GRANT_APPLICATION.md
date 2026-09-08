# Ensue World Model Grant — application draft

**Do not submit until the real ERA5 benchmark has been run.** Replace every bracketed field with the generated result from `results/era5_eval_only.json`.

## Applicant

Name: [YOUR NAME]

Affiliation: Future Is Custom

GitHub repository: https://github.com/futureiscustom/baja__weather_model_eval

Public project page: [ADD PUBLIC FUTURE IS CUSTOM PROJECT-PAGE URL]

## Project

**Baja Weather Model Eval — a compact regional world-model benchmark for Baja California Sur**

Baja Weather Model Eval is an open, reproducible experiment in forecasting the future atmospheric state over Baja California Sur and its surrounding Pacific/Gulf context using a deliberately small neural model.

The project uses public ERA5 ground truth from the WeatherBench 2 dataset. The initial model, TinyWeatherNet, receives four recent six-hourly atmospheric states and predicts the state 24 hours after the latest input.

The initial variables are 2m temperature, 10m U wind, 10m V wind, and mean sea-level pressure.

The objective is not to compete with large global operational models. The objective is to establish a small, inexpensive, inspectable baseline with a fixed evaluation that can support rapid automated experimentation across architecture, loss functions, input history, model size, training strategy and regional context.

## Benchmark / eval

The benchmark is a chronological held-out ERA5 test split over a fixed Baja regional crop.

Primary metric:

- normalized aggregate RMSE across the four predicted variables

Additional metrics:

- per-variable RMSE in physical units
- persistence baseline on the identical test targets
- percentage error reduction versus persistence
- trainable parameter count

The prepared benchmark dataset is SHA-256 hashed. The standalone evaluation script checks the hash before scoring a checkpoint so experimental model changes can be compared against the same data and test definition.

### Current baseline result

Data SHA-256: `[PASTE data_sha256]`

Forecast horizon: `[PASTE lead_hours]` hours

Held-out samples: `[PASTE test_samples]`

Trainable parameters: `[PASTE parameters]`

Persistence normalized RMSE: `[PASTE persistence_normalized_rmse]`

TinyWeatherNet normalized RMSE: `[PASTE model_normalized_rmse]`

Improvement versus persistence: `[PASTE improvement_percent]%`

Per-variable RMSE:

- 2m temperature: `[PASTE] K`
- 10m U wind: `[PASTE] m/s`
- 10m V wind: `[PASTE] m/s`
- mean sea-level pressure: `[PASTE] Pa`

## What we want to optimize with Ensue

We want to maximize forecast skill per unit of model complexity rather than simply scaling parameter count.

Promising experimental dimensions include:

1. residual CNN depth/width and receptive field
2. U-Net style multiscale architectures
3. temporal encoders such as ConvLSTM
4. variable-specific or uncertainty-aware losses
5. alternative normalization and residual targets
6. input-history length
7. regional crop/context size
8. additional ERA5 surface and upper-air variables
9. 6h / 12h / 24h / 48h forecast horizons
10. parameter-count and inference-cost constraints

The evaluation command is intentionally independent from the training command so candidate architectures can be trained or modified while the held-out scoreboard stays fixed.

## Why this project

Most high-profile weather world models are large global systems. This project asks a narrower question: how much useful state-transition modeling can be extracted from a compact regional model that is cheap enough for an independent builder to train and iterate on?

Baja California Sur is a useful regional test case because a narrow peninsula, the Pacific Ocean, the Gulf of California, desert terrain and tropical weather systems interact over a compact geographic area.

The output of the grant would be open experiment history, reproducible benchmark results and model improvements committed back to the repository.