# Leakage-Safe CNN Temporal Baseline

Use `leakage_safe_cnn_temporal.py` when you want a sequence model without repeating the leakage problems from the old CNN-LSTM notebook.

## What This Script Does

- Uses the same leakage-safe blocked split policy as `leakage_safe_baseline.py`.
- Builds sequences *after* splitting, so train/val/test windows do not overlap across boundaries.
- Adds explicit time features per step:
  - `delta_days`
  - `log1p(delta_days)`
  - `cumulative_days`
  - day-of-year sine/cosine
- Encodes each TIFF patch with a CNN, then applies either:
  - `GRU`
  - `Transformer`
- Reports metrics on a true held-out test split.

## Install

```bash
pip install numpy scikit-learn torch rasterio
```

## Recommended Runs

GRU baseline:

```bash
python leakage_safe_cnn_temporal.py --temporal-head gru
```

Transformer baseline:

```bash
python leakage_safe_cnn_temporal.py --temporal-head transformer
```

Harder leave-one-area-out evaluation:

```bash
python leakage_safe_cnn_temporal.py --temporal-head gru --split-scheme area_holdout --holdout-area Area6
```

Less correlated windows:

```bash
python leakage_safe_cnn_temporal.py --stride 5
```

## Outputs

The script writes a `deep_baseline_outputs` folder with:

- `metrics.json`
- `sequence_predictions.csv`
- `model_checkpoint.pt`

## Notes

- `GRU` is the safer default for this dataset size.
- `Transformer` is available, but with only ~1k labeled examples it may not beat the GRU.
- This script still depends on `rasterio` to read the TIFFs.
