# Baja Weather Model Eval

## A compact world-model experiment for Baja California Sur

Baja Weather Model Eval is an open experiment in teaching a small neural network to predict how regional weather conditions evolve over time.

The project focuses on Baja California Sur and its surrounding Pacific and Gulf of California context. Rather than starting with a massive global weather system, the experiment deliberately asks a smaller question:

**How much useful forecasting ability can we extract from a compact model that is inexpensive to train, easy to inspect, and straightforward to benchmark?**

## The experiment

The model receives several recent atmospheric states and predicts a future atmospheric state.

The first real-data benchmark uses public ERA5 data distributed through WeatherBench 2 and focuses on four surface variables:

- 2m temperature
- 10m east-west wind
- 10m north-south wind
- mean sea-level pressure

The initial task uses four recent six-hourly states as context and predicts conditions 24 hours after the most recent observation.

## A scoreboard instead of intuition

The project is built around a fixed evaluation.

Every model is tested against atmospheric conditions it did not see during training. Forecast error is measured using root mean squared error (RMSE), and the learned model is compared with a simple persistence baseline: assume the future will look exactly like the latest observation.

That gives every experiment a clear question:

**Did this change actually make the forecast better?**

The evaluation also reports per-variable errors in physical units, model parameter count, and the percentage improvement or regression versus persistence.

## Why keep the first model small?

The initial model, TinyWeatherNet, is intentionally modest. It uses a few convolutional layers and predicts a correction to the latest observed state.

This is not an attempt to replace operational weather forecasting or compete directly with systems such as GraphCast, Pangu-Weather or NeuralGCM.

A small baseline has a different advantage: it creates a fast experimental surface. Architecture, temporal context, variables, losses and training strategies can be changed without requiring enormous compute budgets.

## Why Baja?

Baja California Sur is a compact but interesting regional system. The peninsula sits between the Pacific Ocean and Gulf of California, with desert terrain, mountain ranges, coastal microclimates and periodic tropical systems interacting over relatively short distances.

The first benchmark intentionally includes surrounding ocean rather than cropping tightly to political boundaries, because the atmosphere does not care where a state line ends.

## Reproducible by design

The project uses chronological train/test splits so later observations remain genuinely held out.

Normalization statistics are calculated from training data only. The persistence baseline and learned model are scored on identical targets. Prepared benchmark data are hashed so an evaluation cannot silently switch to a different dataset.

Results are written to machine-readable JSON in addition to being printed for humans.

## What comes next

The initial baseline is a starting point rather than a conclusion.

Future experiments can test:

- deeper residual CNNs
- U-Net architectures
- ConvLSTM temporal models
- longer and shorter input histories
- additional ERA5 variables and upper-air fields
- 6-, 12-, 24- and 48-hour forecast horizons
- latitude-weighted evaluation
- climatology baselines
- WeatherBench-X compatible forecast exports
- forecast-skill versus model-size and inference-cost tradeoffs

The most useful result may not be the single model with the lowest error. It may be discovering which changes reliably improve a compact regional world model and which merely add complexity.

## Open experiment

The code, benchmark definition and evaluation scripts are available in the project repository. Real benchmark numbers will be published only after running the held-out ERA5 evaluation; synthetic development data are used solely to verify and teach the experimental pipeline.

Built by Future Is Custom as an experiment in applying modern AI research workflows at independent-builder scale.