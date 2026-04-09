# Leakage-Safe Soil Moisture Baseline

Use `leakage_safe_baseline.py` instead of the current random-split notebooks when you want a baseline you can trust.

## What It Fixes

- No random row split across the full dataset.
- No overlapping train/validation windows.
- No validation reuse as the final test report.
- Explicit time-blocked splits per area, with an embargo gap.
- A simple, auditable tree baseline built from TIFF patch statistics.

## What It Trains

- Image features:
  - Per-band mean, std, min, max, p10, p50, p90, center pixel, zero fraction.
  - Sentinel-style indices when the stack looks like a 12-band Sentinel-2 cube: NDVI, NDMI, NBR, EVI.
- Metadata features:
  - Seasonal sine/cosine terms.
  - Year offset.
  - Days since the previous observation in the same area.
- Models:
  - `ExtraTreesRegressor`
  - `RandomForestRegressor`
  - `HistGradientBoostingRegressor`

By default it compares those models on validation, picks the best one, then refits on `train + val` and reports on the held-out test split.

## Install

```bash
pip install numpy scikit-learn rasterio
```

## Recommended Runs

Temporal blocked split inside every area:

```bash
python leakage_safe_baseline.py
```

Harder leave-one-area-out test:

```bash
python leakage_safe_baseline.py --split-scheme area_holdout --holdout-area Area6
```

Same-site forecasting where area ID is known at inference time:

```bash
python leakage_safe_baseline.py --include-area-feature
```

## Outputs

The script writes a `baseline_outputs` folder with:

- `metrics.json`: dataset summary, split summary, model selection results, and final metrics.
- `predictions.csv`: row-level predictions for train, val, and test.
- `feature_importance.csv`: only when the chosen model exposes feature importances.
- `model_bundle.pkl`: fitted model plus feature vectorizer.
- `feature_cache.pkl`: cached TIFF-derived features to speed up reruns.

## Notes

- The default split is still a same-area, future-time forecast. It is leakage-safe, but it does not test transfer to unseen sites.
- For site transfer, prefer `--split-scheme area_holdout`.
- The old notebook generators are still in the folder for comparison, but they should not be your reference baseline anymore.
