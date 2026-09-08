# Evals 101: what you are actually doing

You do **not** need to begin by inventing a giant AI model.

For this project, an eval is a controlled experiment with five pieces:

1. **Input** — past weather states.
2. **Model** — a function that predicts a future weather state.
3. **Ground truth** — what ERA5 says actually happened.
4. **Metric** — a number measuring prediction error.
5. **Baseline** — a simple method the model should beat.

## The simplest mental model

Suppose today's temperature is 28°C.

A persistence forecast says:

> Tomorrow will also be 28°C.

Your model says:

> Tomorrow will be 26.5°C.

ERA5 says tomorrow was actually 27°C.

The model's error is 0.5°C. Persistence's error is 1°C. The learned model wins that example.

Real evals repeat that process over many dates, grid cells and variables, then aggregate the errors.

## Why RMSE?

This repo starts with root mean squared error (RMSE):

`RMSE = sqrt(mean((prediction - truth)^2))`

Lower is better. Larger mistakes are penalized more heavily because errors are squared.

RMSE is common in deterministic weather evaluation and easy to understand.

## Why normalize before training?

Our variables have completely different scales:

- temperature: around hundreds of kelvin
- wind: often single-digit or tens of m/s
- pressure: around 100,000 Pa

If we trained directly on raw numbers, pressure could dominate the loss simply because its numeric scale is larger.

So the model trains on standardized variables:

`normalized = (value - training_mean) / training_std`

The normalization mean/std are calculated from **training data only**. We then convert predictions back to physical units when reporting per-variable scores.

## Why chronological train/test splits?

Weather is a time series. Randomly mixing future and past observations into train/test sets can leak information and make a model look better than it is.

This project uses the earlier 80% of observations for training and the later 20% for testing.

The test split is not touched during model fitting.

## What is the model here?

`TinyWeatherNet` is a small convolutional neural network (CNN).

It gets several weather maps from recent times. The maps are stacked as channels. The CNN predicts a correction to the latest observed state.

Conceptually:

`future = latest_weather + learned_change`

This is useful because the naive persistence forecast is already `future = latest_weather`. The network only has to learn how conditions tend to evolve away from persistence.

## Where do models normally come from?

There are four common routes:

### 1. Build a small model yourself

That is what this repository does. This is often best for learning and for research grants where the model needs to be modified repeatedly.

### 2. Use an open-source architecture

You can implement or adapt architectures from papers/GitHub: U-Net, ConvLSTM, ResNet, Vision Transformer, graph neural networks, etc.

### 3. Use pretrained model weights

Repositories such as Hugging Face and research GitHub repos distribute trained weights. In weather, examples of major research systems include GraphCast, Pangu-Weather and NeuralGCM.

A pretrained model is useful when you want inference immediately, but it can be harder or more expensive to retrain and modify.

### 4. Use an API model

For language/vision tasks, an eval can call an API model and score its answers. That is common for LLM evals. It is less appropriate for this grant's world-model optimization goal because Ensue wants a model/training/evaluation loop they can experiment with.

## How do I test a different model?

Keep the eval fixed and swap the model.

For example:

- Model A: persistence (no learning)
- Model B: TinyWeatherNet
- Model C: larger CNN
- Model D: U-Net
- Model E: ConvLSTM

Every model receives the same test inputs and is compared with the same ERA5 targets.

Do **not** keep changing the test set until your favorite model wins. That destroys the value of the eval.

## What makes an eval credible?

A credible eval is:

- reproducible
- measured on held-out data
- compared to a baseline
- defined before you optimize against it
- machine-readable
- honest about what it does and does not prove

This repo adds a SHA-256 hash of the prepared ERA5 dataset to each model checkpoint. `eval_era5.py` refuses to silently evaluate against different data unless explicitly overridden.

## Your learning sequence

Run these in order:

```bash
python scripts/run_synthetic_eval.py
```

Learn what `persistence_normalized_rmse`, `model_normalized_rmse`, and `improvement_percent` mean.

Then prepare real data:

```bash
python scripts/prepare_era5_baja.py --start 2018-01-01 --end 2020-12-31
```

Train:

```bash
python scripts/train_era5.py
```

Evaluate the frozen checkpoint without training:

```bash
python scripts/eval_era5.py --checkpoint artifacts/era5_tinyweathernet.pt
```

Then make **one change at a time** to the model or training configuration and compare the resulting score.

That is experimental ML research in its basic form.