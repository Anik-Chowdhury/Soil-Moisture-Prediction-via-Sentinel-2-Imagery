#!/usr/bin/env python3
"""Leakage-safe soil moisture baseline for labeled Sentinel patch data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REQUIRED_COLUMNS = {
    "Area",
    "date_time",
    "Image_name",
    "soil_moisture_depth_0.050000",
}

DEFAULT_SENTINEL2_BAND_NAMES = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]


@dataclass(frozen=True)
class Record:
    row_id: int
    area: str
    date_time: datetime
    image_name: str
    target: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing the CSV and image directory.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Path to the labeled CSV. Defaults to the combined CSV in --base-dir.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Path to the TIFF directory. Defaults to All_3_areas_cropped in --base-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder where metrics, predictions, and the fitted model will be written.",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Optional pickle cache for extracted image features.",
    )
    parser.add_argument(
        "--split-scheme",
        choices=("temporal", "area_holdout"),
        default="temporal",
        help="Leakage-safe split strategy.",
    )
    parser.add_argument(
        "--holdout-area",
        default=None,
        help="Area reserved as the final test set when --split-scheme=area_holdout.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction reserved for validation inside each training area.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction reserved for test inside each area for temporal splits.",
    )
    parser.add_argument(
        "--gap-samples",
        type=int,
        default=1,
        help="Embargo gap between train/val and val/test boundaries inside each area.",
    )
    parser.add_argument(
        "--model-family",
        choices=("auto", "extra_trees", "random_forest", "hist_gbr"),
        default="auto",
        help="Baseline regressor family. auto compares a few safe defaults on validation.",
    )
    parser.add_argument(
        "--include-area-feature",
        action="store_true",
        help="Include area ID as a categorical feature. Useful for same-site forecasting.",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Ignore any feature cache and re-read every TIFF.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for model reproducibility.",
    )
    return parser.parse_args()


def ensure_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    base_dir = args.base_dir.resolve()
    csv_path = (args.csv_path or (base_dir / "Combined_Area1_Area2_Area3_Area5_Area6_with_tif.csv")).resolve()
    image_dir = (args.image_dir or (base_dir / "All_3_areas_cropped")).resolve()
    output_dir = (args.output_dir or (base_dir / "baseline_outputs")).resolve()
    feature_cache = (args.feature_cache or (output_dir / "feature_cache.pkl")).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    return csv_path, image_dir, output_dir, feature_cache


def load_records(csv_path: Path, image_dir: Path) -> tuple[list[Record], dict[str, Any]]:
    records: list[Record] = []
    dropped_missing_image = 0
    dropped_invalid_target = 0
    invalid_target_examples: list[str] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

        for row_idx, row in enumerate(reader):
            image_name = row["Image_name"].strip()
            image_path = image_dir / image_name
            if not image_path.exists():
                dropped_missing_image += 1
                continue

            target_raw = row["soil_moisture_depth_0.050000"].strip()
            try:
                target = float(target_raw)
            except ValueError:
                dropped_invalid_target += 1
                if len(invalid_target_examples) < 5:
                    invalid_target_examples.append(f"row {row_idx}: {target_raw!r}")
                continue

            if not np.isfinite(target):
                dropped_invalid_target += 1
                if len(invalid_target_examples) < 5:
                    invalid_target_examples.append(f"row {row_idx}: {target_raw!r}")
                continue

            records.append(
                Record(
                    row_id=row_idx,
                    area=row["Area"].strip(),
                    date_time=datetime.fromisoformat(row["date_time"].strip()),
                    image_name=image_name,
                    target=target,
                )
            )

    if not records:
        raise ValueError("No valid labeled records remained after filtering.")

    summary = {
        "rows_kept": len(records),
        "rows_dropped_missing_image": dropped_missing_image,
        "rows_dropped_invalid_target": dropped_invalid_target,
        "invalid_target_examples": invalid_target_examples,
        "areas": dict(sorted(Counter(record.area for record in records).items())),
        "target_min": float(min(record.target for record in records)),
        "target_max": float(max(record.target for record in records)),
        "target_mean": float(sum(record.target for record in records) / len(records)),
    }
    return records, summary


def group_records_by_area(records: list[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.area].append(record)
    for area_records in grouped.values():
        area_records.sort(key=lambda item: (item.date_time, item.image_name))
    return dict(sorted(grouped.items()))


def split_train_val_test_blocked(
    area_records: list[Record],
    val_fraction: float,
    test_fraction: float,
    gap_samples: int,
) -> tuple[list[Record], list[Record], list[Record], list[Record]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.")
    if gap_samples < 0:
        raise ValueError("gap_samples must be >= 0.")

    count = len(area_records)
    val_count = max(1, math.ceil(count * val_fraction))
    test_count = max(1, math.ceil(count * test_fraction))
    train_count = count - val_count - test_count - (2 * gap_samples)
    if train_count < 1:
        raise ValueError(
            f"Area {area_records[0].area!r} has only {count} rows; not enough for "
            f"train/val/test with val_fraction={val_fraction}, "
            f"test_fraction={test_fraction}, gap_samples={gap_samples}."
        )

    train_end = train_count
    val_start = train_end + gap_samples
    val_end = val_start + val_count
    test_start = val_end + gap_samples

    train_records = area_records[:train_end]
    val_records = area_records[val_start:val_end]
    test_records = area_records[test_start:]
    gap_records = area_records[train_end:val_start] + area_records[val_end:test_start]
    return train_records, val_records, test_records, gap_records


def split_train_val_blocked(
    area_records: list[Record],
    val_fraction: float,
    gap_samples: int,
) -> tuple[list[Record], list[Record], list[Record]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if gap_samples < 0:
        raise ValueError("gap_samples must be >= 0.")

    count = len(area_records)
    val_count = max(1, math.ceil(count * val_fraction))
    train_count = count - val_count - gap_samples
    if train_count < 1:
        raise ValueError(
            f"Area {area_records[0].area!r} has only {count} rows; not enough for "
            f"train/val with val_fraction={val_fraction}, gap_samples={gap_samples}."
        )

    train_end = train_count
    val_start = train_end + gap_samples
    train_records = area_records[:train_end]
    val_records = area_records[val_start:]
    gap_records = area_records[train_end:val_start]
    return train_records, val_records, gap_records


def build_splits(
    records: list[Record],
    split_scheme: str,
    val_fraction: float,
    test_fraction: float,
    gap_samples: int,
    holdout_area: str | None,
) -> dict[str, list[Record]]:
    grouped = group_records_by_area(records)
    train_records: list[Record] = []
    val_records: list[Record] = []
    test_records: list[Record] = []
    gap_records: list[Record] = []

    if split_scheme == "temporal":
        for area_records in grouped.values():
            area_train, area_val, area_test, area_gap = split_train_val_test_blocked(
                area_records=area_records,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                gap_samples=gap_samples,
            )
            train_records.extend(area_train)
            val_records.extend(area_val)
            test_records.extend(area_test)
            gap_records.extend(area_gap)
    elif split_scheme == "area_holdout":
        if not holdout_area:
            raise ValueError("--holdout-area is required when --split-scheme=area_holdout.")
        if holdout_area not in grouped:
            raise ValueError(
                f"Holdout area {holdout_area!r} not found. Available areas: {sorted(grouped)}"
            )
        test_records.extend(grouped[holdout_area])
        for area_name, area_records in grouped.items():
            if area_name == holdout_area:
                continue
            area_train, area_val, area_gap = split_train_val_blocked(
                area_records=area_records,
                val_fraction=val_fraction,
                gap_samples=gap_samples,
            )
            train_records.extend(area_train)
            val_records.extend(area_val)
            gap_records.extend(area_gap)
    else:
        raise ValueError(f"Unsupported split scheme: {split_scheme}")

    if not train_records or not val_records or not test_records:
        raise ValueError("A split is empty. Adjust fractions or choose another split scheme.")

    return {
        "train": sorted(train_records, key=lambda item: (item.area, item.date_time, item.image_name)),
        "val": sorted(val_records, key=lambda item: (item.area, item.date_time, item.image_name)),
        "test": sorted(test_records, key=lambda item: (item.area, item.date_time, item.image_name)),
        "gap": sorted(gap_records, key=lambda item: (item.area, item.date_time, item.image_name)),
    }


def summarize_split(records: list[Record]) -> dict[str, Any]:
    by_area = group_records_by_area(records)
    return {
        "count": len(records),
        "areas": {
            area: {
                "count": len(area_records),
                "start": area_records[0].date_time.date().isoformat(),
                "end": area_records[-1].date_time.date().isoformat(),
            }
            for area, area_records in by_area.items()
        },
    }


def import_rasterio():
    try:
        import rasterio  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "rasterio is required to read the TIFF patches. Install it with "
            "`pip install rasterio` before running this baseline."
        ) from exc
    return rasterio


def infer_band_names(band_count: int) -> list[str]:
    if band_count == len(DEFAULT_SENTINEL2_BAND_NAMES):
        return DEFAULT_SENTINEL2_BAND_NAMES[:]
    return [f"band_{index:02d}" for index in range(1, band_count + 1)]


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / (denominator + 1e-6)


def summarize_array(features: dict[str, float], prefix: str, values: np.ndarray) -> None:
    flat = values.reshape(-1).astype(np.float32)
    features[f"{prefix}_mean"] = float(np.mean(flat))
    features[f"{prefix}_std"] = float(np.std(flat))
    features[f"{prefix}_p10"] = float(np.quantile(flat, 0.10))
    features[f"{prefix}_p50"] = float(np.quantile(flat, 0.50))
    features[f"{prefix}_p90"] = float(np.quantile(flat, 0.90))


def extract_image_features(image_path: Path) -> dict[str, float]:
    rasterio = import_rasterio()
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)
        nodata_value = src.nodata

    if image.ndim != 3:
        raise ValueError(f"Expected 3D raster (bands, height, width) but got {image.shape} for {image_path}")

    if nodata_value is not None:
        image = np.where(image == nodata_value, np.nan, image)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.clip(image, a_min=0.0, a_max=None)

    band_count, height, width = image.shape
    band_names = infer_band_names(band_count)
    features: dict[str, float] = {
        "patch_height": float(height),
        "patch_width": float(width),
    }

    for band_index, band_name in enumerate(band_names):
        band = image[band_index]
        flat = band.reshape(-1)
        features[f"{band_name}_mean"] = float(np.mean(flat))
        features[f"{band_name}_std"] = float(np.std(flat))
        features[f"{band_name}_min"] = float(np.min(flat))
        features[f"{band_name}_max"] = float(np.max(flat))
        features[f"{band_name}_p10"] = float(np.quantile(flat, 0.10))
        features[f"{band_name}_p50"] = float(np.quantile(flat, 0.50))
        features[f"{band_name}_p90"] = float(np.quantile(flat, 0.90))
        features[f"{band_name}_center"] = float(band[height // 2, width // 2])
        features[f"{band_name}_zero_fraction"] = float(np.mean(flat == 0.0))

    band_lookup = {band_name: image[idx] for idx, band_name in enumerate(band_names)}
    if {"B02", "B04", "B08", "B11", "B12"}.issubset(band_lookup):
        summarize_array(
            features,
            "ndvi",
            safe_ratio(band_lookup["B08"] - band_lookup["B04"], band_lookup["B08"] + band_lookup["B04"]),
        )
        summarize_array(
            features,
            "ndmi",
            safe_ratio(band_lookup["B08"] - band_lookup["B11"], band_lookup["B08"] + band_lookup["B11"]),
        )
        summarize_array(
            features,
            "nbr",
            safe_ratio(band_lookup["B08"] - band_lookup["B12"], band_lookup["B08"] + band_lookup["B12"]),
        )
        evi = 2.5 * safe_ratio(
            band_lookup["B08"] - band_lookup["B04"],
            band_lookup["B08"] + 6.0 * band_lookup["B04"] - 7.5 * band_lookup["B02"] + 1.0,
        )
        summarize_array(features, "evi", evi)

    return features


def load_feature_cache(feature_cache_path: Path, rebuild_features: bool) -> dict[str, dict[str, float]]:
    if rebuild_features or not feature_cache_path.exists():
        return {}
    with feature_cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected cache format in {feature_cache_path}")
    return payload


def save_feature_cache(feature_cache_path: Path, feature_cache: dict[str, dict[str, float]]) -> None:
    with feature_cache_path.open("wb") as handle:
        pickle.dump(feature_cache, handle)


def build_feature_lookup(
    records: list[Record],
    image_dir: Path,
    feature_cache_path: Path,
    rebuild_features: bool,
) -> dict[str, dict[str, float]]:
    feature_cache = load_feature_cache(feature_cache_path, rebuild_features=rebuild_features)
    needed_images = sorted({record.image_name for record in records})
    missing_images = [image_name for image_name in needed_images if image_name not in feature_cache]

    if missing_images:
        print(f"Extracting features for {len(missing_images)} TIFFs...")
        for index, image_name in enumerate(missing_images, start=1):
            image_path = image_dir / image_name
            feature_cache[image_name] = extract_image_features(image_path)
            if index % 50 == 0 or index == len(missing_images):
                print(f"  processed {index}/{len(missing_images)}")
        save_feature_cache(feature_cache_path, feature_cache)
    else:
        print("Feature cache is already warm; no TIFFs needed to be re-read.")

    return {image_name: feature_cache[image_name] for image_name in needed_images}


def build_days_since_previous_lookup(records: list[Record]) -> dict[int, float]:
    lookup: dict[int, float] = {}
    grouped = group_records_by_area(records)
    for area_records in grouped.values():
        previous_date: datetime | None = None
        for record in area_records:
            if previous_date is None:
                lookup[record.row_id] = -1.0
            else:
                lookup[record.row_id] = float((record.date_time - previous_date).days)
            previous_date = record.date_time
    return lookup


def add_metadata_features(
    record: Record,
    base_features: dict[str, float],
    include_area_feature: bool,
    days_since_previous_lookup: dict[int, float],
) -> dict[str, Any]:
    features: dict[str, Any] = dict(base_features)
    day_of_year = record.date_time.timetuple().tm_yday
    month = record.date_time.month
    year = record.date_time.year

    features["day_of_year_sin"] = math.sin((2.0 * math.pi * day_of_year) / 365.25)
    features["day_of_year_cos"] = math.cos((2.0 * math.pi * day_of_year) / 365.25)
    features["month_sin"] = math.sin((2.0 * math.pi * month) / 12.0)
    features["month_cos"] = math.cos((2.0 * math.pi * month) / 12.0)
    features["year_offset"] = float(year - 2017)
    features["days_since_previous_obs"] = days_since_previous_lookup.get(record.row_id, -1.0)
    if include_area_feature:
        features["area_id"] = record.area
    return features


def records_to_matrix(
    records: list[Record],
    feature_lookup: dict[str, dict[str, float]],
    include_area_feature: bool,
    days_since_previous_lookup: dict[int, float],
    vectorizer: DictVectorizer | None = None,
) -> tuple[np.ndarray, np.ndarray, DictVectorizer]:
    row_features = [
        add_metadata_features(
            record=record,
            base_features=feature_lookup[record.image_name],
            include_area_feature=include_area_feature,
            days_since_previous_lookup=days_since_previous_lookup,
        )
        for record in records
    ]
    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=False)
        matrix = vectorizer.fit_transform(row_features)
    else:
        matrix = vectorizer.transform(row_features)
    targets = np.array([record.target for record in records], dtype=np.float32)
    return matrix, targets, vectorizer


def build_candidate_models(random_seed: int) -> dict[str, Any]:
    return {
        "extra_trees": ExtraTreesRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_seed,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_seed,
        ),
        "hist_gbr": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=500,
            min_samples_leaf=10,
            random_state=random_seed,
        ),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        pearson_r = 0.0
    else:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "bias": bias,
        "pearson_r": pearson_r,
    }


def evaluate_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_family: str,
    random_seed: int,
) -> tuple[str, Any, dict[str, dict[str, float]], np.ndarray]:
    candidates = build_candidate_models(random_seed=random_seed)
    if model_family != "auto":
        candidates = {model_family: candidates[model_family]}

    validation_scores: dict[str, dict[str, float]] = {}
    best_name = ""
    best_model: Any = None
    best_predictions = np.empty_like(y_val)
    best_rmse = float("inf")

    for model_name, model in candidates.items():
        fitted_model = clone(model)
        fitted_model.fit(x_train, y_train)
        predictions = fitted_model.predict(x_val)
        metrics = compute_metrics(y_true=y_val, y_pred=predictions)
        validation_scores[model_name] = metrics
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_name = model_name
            best_model = fitted_model
            best_predictions = predictions

    if best_model is None:
        raise RuntimeError("No model candidate was trained.")
    return best_name, best_model, validation_scores, best_predictions


def fit_final_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_seed: int,
) -> Any:
    candidates = build_candidate_models(random_seed=random_seed)
    final_model = clone(candidates[model_name])
    final_model.fit(np.vstack([x_train, x_val]), np.concatenate([y_train, y_val]))
    return final_model


def per_area_metrics(records: list[Record], predictions: np.ndarray) -> dict[str, dict[str, float]]:
    grouped_truth: dict[str, list[float]] = defaultdict(list)
    grouped_pred: dict[str, list[float]] = defaultdict(list)
    for record, prediction in zip(records, predictions):
        grouped_truth[record.area].append(record.target)
        grouped_pred[record.area].append(float(prediction))

    return {
        area: compute_metrics(
            y_true=np.array(grouped_truth[area], dtype=np.float32),
            y_pred=np.array(grouped_pred[area], dtype=np.float32),
        )
        for area in sorted(grouped_truth)
    }


def save_predictions(
    records_by_split: dict[str, list[Record]],
    predictions_by_split: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "split",
            "row_id",
            "area",
            "date_time",
            "image_name",
            "target",
            "prediction",
            "abs_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            records = records_by_split[split_name]
            predictions = predictions_by_split[split_name]
            for record, prediction in zip(records, predictions):
                writer.writerow(
                    {
                        "split": split_name,
                        "row_id": record.row_id,
                        "area": record.area,
                        "date_time": record.date_time.date().isoformat(),
                        "image_name": record.image_name,
                        "target": f"{record.target:.10f}",
                        "prediction": f"{float(prediction):.10f}",
                        "abs_error": f"{abs(float(prediction) - record.target):.10f}",
                    }
                )


def save_feature_importances(model: Any, vectorizer: DictVectorizer, output_path: Path) -> None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    feature_names = vectorizer.get_feature_names_out()
    ranking = sorted(
        zip(feature_names, importances, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_name", "importance"])
        for feature_name, importance in ranking:
            writer.writerow([feature_name, f"{float(importance):.10f}"])


def save_model_bundle(
    model: Any,
    vectorizer: DictVectorizer,
    split_config: dict[str, Any],
    output_path: Path,
) -> None:
    payload = {
        "model": model,
        "vectorizer": vectorizer,
        "split_config": split_config,
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)


def main() -> None:
    args = parse_args()
    csv_path, image_dir, output_dir, feature_cache_path = ensure_paths(args)

    records, dataset_summary = load_records(csv_path=csv_path, image_dir=image_dir)
    splits = build_splits(
        records=records,
        split_scheme=args.split_scheme,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        gap_samples=args.gap_samples,
        holdout_area=args.holdout_area,
    )

    days_since_previous_lookup = build_days_since_previous_lookup(records)
    feature_lookup = build_feature_lookup(
        records=records,
        image_dir=image_dir,
        feature_cache_path=feature_cache_path,
        rebuild_features=args.rebuild_features,
    )

    x_train, y_train, vectorizer = records_to_matrix(
        records=splits["train"],
        feature_lookup=feature_lookup,
        include_area_feature=args.include_area_feature,
        days_since_previous_lookup=days_since_previous_lookup,
        vectorizer=None,
    )
    x_val, y_val, vectorizer = records_to_matrix(
        records=splits["val"],
        feature_lookup=feature_lookup,
        include_area_feature=args.include_area_feature,
        days_since_previous_lookup=days_since_previous_lookup,
        vectorizer=vectorizer,
    )
    x_test, y_test, vectorizer = records_to_matrix(
        records=splits["test"],
        feature_lookup=feature_lookup,
        include_area_feature=args.include_area_feature,
        days_since_previous_lookup=days_since_previous_lookup,
        vectorizer=vectorizer,
    )

    best_model_name, _, validation_scores, val_predictions = evaluate_candidates(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        model_family=args.model_family,
        random_seed=args.random_seed,
    )
    final_model = fit_final_model(
        model_name=best_model_name,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        random_seed=args.random_seed,
    )

    train_predictions = final_model.predict(x_train)
    test_predictions = final_model.predict(x_test)

    train_metrics = compute_metrics(y_true=y_train, y_pred=train_predictions)
    val_metrics = compute_metrics(y_true=y_val, y_pred=val_predictions)
    test_metrics = compute_metrics(y_true=y_test, y_pred=test_predictions)

    split_summary = {
        "train": summarize_split(splits["train"]),
        "val": summarize_split(splits["val"]),
        "test": summarize_split(splits["test"]),
        "gap": summarize_split(splits["gap"]),
    }

    report = {
        "dataset_summary": dataset_summary,
        "split_config": {
            "split_scheme": args.split_scheme,
            "holdout_area": args.holdout_area,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "gap_samples": args.gap_samples,
            "include_area_feature": args.include_area_feature,
            "random_seed": args.random_seed,
        },
        "split_summary": split_summary,
        "feature_count": int(x_train.shape[1]),
        "selected_model": best_model_name,
        "validation_model_scores": validation_scores,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "test_per_area": per_area_metrics(splits["test"], test_predictions),
        },
    }

    predictions_by_split = {
        "train": train_predictions,
        "val": val_predictions,
        "test": test_predictions,
    }
    save_predictions(
        records_by_split=splits,
        predictions_by_split=predictions_by_split,
        output_path=output_dir / "predictions.csv",
    )
    save_feature_importances(
        model=final_model,
        vectorizer=vectorizer,
        output_path=output_dir / "feature_importance.csv",
    )
    save_model_bundle(
        model=final_model,
        vectorizer=vectorizer,
        split_config=report["split_config"],
        output_path=output_dir / "model_bundle.pkl",
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Leakage-safe baseline completed.")
    print(f"Selected model: {best_model_name}")
    print(
        "Test metrics: "
        f"RMSE={test_metrics['rmse']:.5f}, "
        f"MAE={test_metrics['mae']:.5f}, "
        f"R2={test_metrics['r2']:.5f}, "
        f"Pearson={test_metrics['pearson_r']:.5f}"
    )
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
