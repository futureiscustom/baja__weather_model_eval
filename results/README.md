# Results

Real ERA5 benchmark JSON belongs in this directory and should be committed so reviewers can inspect the exact score used in the grant application.

Generated files:

- `era5_metrics.json` — produced after training
- `era5_eval_only.json` — produced by the standalone held-out evaluator; this is the preferred grant scoreboard
- `ENSUE_APPLICATION_READY.md` — rendered from `era5_eval_only.json`

Synthetic metrics are intentionally gitignored because they are only a software/learning check and are not evidence for the grant.

Before publishing a claimed result, run:

```bash
python scripts/eval_era5.py
python scripts/render_grant_application.py
```

Then commit the real `era5_eval_only.json` and application draft.