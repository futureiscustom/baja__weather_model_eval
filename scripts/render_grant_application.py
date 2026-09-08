from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a ready-to-paste Ensue grant application from real ERA5 metrics.")
    parser.add_argument("--metrics", default=str(ROOT / "results/era5_eval_only.json"))
    parser.add_argument("--name", default="[YOUR NAME]")
    parser.add_argument("--affiliation", default="Future Is Custom")
    parser.add_argument("--project-url", default="[ADD PUBLIC PROJECT PAGE URL]")
    parser.add_argument("--output", default=str(ROOT / "results/ENSUE_APPLICATION_READY.md"))
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(
            f"Missing {metrics_path}. First run the real benchmark:\n"
            "  python scripts/prepare_era5_baja.py --start 2018-01-01 --end 2020-12-31\n"
            "  python scripts/train_era5.py\n"
            "  python scripts/eval_era5.py"
        )

    m = json.loads(metrics_path.read_text())
    per_var = m["model_per_variable_rmse"]
    persistence = m["persistence_per_variable_rmse"]

    text = "# Ensue World Model Grant Application\n\n"
    text += f"**Name:** {args.name}\n\n"
    text += f"**Affiliation:** {args.affiliation}\n\n"
    text += "**GitHub:** https://github.com/futureiscustom/baja__weather_model_eval\n\n"
    text += f"**Public project link:** {args.project_url}\n\n"
    text += "## Project: Baja Weather Model Eval\n\n"
    text += (
        "Baja Weather Model Eval is an open, reproducible regional world-model experiment for Baja California Sur. "
        "It uses public ERA5 ground truth distributed through WeatherBench 2. TinyWeatherNet, the initial baseline, "
        f"uses {m['input_steps']} recent six-hourly atmospheric states and predicts the state {m['lead_hours']} hours "
        "after the latest input. The initial variables are 2m temperature, 10m U/V wind, and mean sea-level pressure.\n\n"
    )
    text += "## Benchmark / eval\n\n"
    text += (
        "The benchmark uses a chronological held-out ERA5 test split over a fixed Baja regional crop. "
        "The primary score is normalized aggregate RMSE; we also report per-variable RMSE in physical units, "
        "parameter count, and improvement versus a persistence forecast evaluated on the identical targets. "
        "The prepared data are SHA-256 hashed and the standalone evaluation command checks the hash before scoring.\n\n"
    )
    text += f"- Data SHA-256: `{m['data_sha256']}`\n"
    text += f"- Forecast horizon: {m['lead_hours']} hours\n"
    text += f"- Held-out samples: {m['test_samples']}\n"
    text += f"- Trainable parameters: {m['parameters']:,}\n"
    text += f"- Persistence normalized RMSE: {m['persistence_normalized_rmse']:.6f}\n"
    text += f"- TinyWeatherNet normalized RMSE: {m['model_normalized_rmse']:.6f}\n"
    text += f"- Improvement vs persistence: {m['improvement_percent']:.2f}%\n\n"
    text += "### Per-variable RMSE\n\n"
    for variable in m["variables"]:
        unit = m.get("units", {}).get(variable, "")
        text += (
            f"- {variable}: model {per_var[variable]:.4f} {unit}; "
            f"persistence {persistence[variable]:.4f} {unit}\n"
        )
    text += "\n## What we want to optimize with Ensue\n\n"
    text += (
        "We want to improve forecast skill per unit of model complexity. The fixed eval makes it possible to explore "
        "architecture depth/width, receptive field, U-Net or ConvLSTM alternatives, input-history length, regional context, "
        "loss design, additional ERA5 variables, training strategy, and parameter/inference-cost constraints while preserving "
        "an apples-to-apples held-out benchmark. Successful changes can be measured immediately and reproduced from the repo.\n\n"
    )
    text += "## Why this project\n\n"
    text += (
        "Most high-profile weather world models are large global systems. This project asks how much useful state-transition "
        "modeling can be extracted from a compact regional model that an independent builder can train and iterate on. "
        "Baja California Sur is a useful test region because the Pacific, Gulf of California, desert terrain and tropical "
        "systems interact across a compact geography. The grant output would be reproducible experiment history, benchmark "
        "results, and model improvements committed back to the open repository.\n"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(f"Rendered application: {output}")


if __name__ == "__main__":
    main()
