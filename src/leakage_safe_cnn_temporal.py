#!/usr/bin/env python3
"""Leakage-safe CNN + GRU/Transformer soil moisture baseline."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from leakage_safe_baseline import (
    Record,
    build_days_since_previous_lookup,
    build_splits,
    compute_metrics,
    group_records_by_area,
    load_records,
    summarize_split,
)


@dataclass(frozen=True)
class SequenceSample:
    split_name: str
    area: str
    records: tuple[Record, ...]
    image_names: tuple[str, ...]
    delta_days: tuple[float, ...]
    cumulative_days: tuple[float, ...]
    day_of_year_sin: tuple[float, ...]
    day_of_year_cos: tuple[float, ...]
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
        help="Folder where metrics, checkpoints, and predictions will be written.",
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
        "--seq-len",
        type=int,
        default=5,
        help="Number of observations per sequence.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding-window stride used to build sequences inside each split.",
    )
    parser.add_argument(
        "--temporal-head",
        choices=("gru", "transformer"),
        default="gru",
        help="Temporal aggregation block after the CNN encoder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--cnn-embed-dim",
        type=int,
        default=128,
        help="Embedding size produced by the CNN encoder.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden size used by the GRU/Transformer head.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of GRU or Transformer layers.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads for the Transformer.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout used throughout the network.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience in epochs.",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Max gradient norm. Set <= 0 to disable clipping.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    base_dir = args.base_dir.resolve()
    csv_path = (args.csv_path or (base_dir / "Combined_Area1_Area2_Area3_Area5_Area6_with_tif.csv")).resolve()
    image_dir = (args.image_dir or (base_dir / "All_3_areas_cropped")).resolve()
    output_dir = (args.output_dir or (base_dir / "deep_baseline_outputs")).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return csv_path, image_dir, output_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_rasterio():
    try:
        import rasterio  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "rasterio is required to read the TIFF patches. Install it with "
            "`pip install rasterio` before running this deep baseline."
        ) from exc
    return rasterio


def load_image_lookup(records: list[Record], image_dir: Path) -> dict[str, np.ndarray]:
    rasterio = import_rasterio()
    needed_images = sorted({record.image_name for record in records})
    image_lookup: dict[str, np.ndarray] = {}

    print(f"Loading {len(needed_images)} labeled TIFFs into memory...")
    for index, image_name in enumerate(needed_images, start=1):
        image_path = image_dir / image_name
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
            nodata_value = src.nodata

        if nodata_value is not None:
            image = np.where(image == nodata_value, np.nan, image)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.clip(image, a_min=0.0, a_max=None)
        image_lookup[image_name] = image

        if index % 100 == 0 or index == len(needed_images):
            print(f"  loaded {index}/{len(needed_images)}")

    return image_lookup


def compute_image_stats(train_records: list[Record], image_lookup: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    first_image = image_lookup[train_records[0].image_name]
    band_count = first_image.shape[0]
    sums = np.zeros(band_count, dtype=np.float64)
    squared_sums = np.zeros(band_count, dtype=np.float64)
    pixel_count = 0

    for record in train_records:
        image = image_lookup[record.image_name]
        flat = image.reshape(image.shape[0], -1).astype(np.float64)
        sums += flat.sum(axis=1)
        squared_sums += np.square(flat).sum(axis=1)
        pixel_count += flat.shape[1]

    means = sums / pixel_count
    variances = (squared_sums / pixel_count) - np.square(means)
    stds = np.sqrt(np.clip(variances, a_min=1e-6, a_max=None))
    return means.astype(np.float32), stds.astype(np.float32)


def build_sequences(
    records: list[Record],
    split_name: str,
    seq_len: int,
    stride: int,
    days_since_previous_lookup: dict[int, float],
) -> list[SequenceSample]:
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")

    sequences: list[SequenceSample] = []
    grouped = group_records_by_area(records)

    for area, area_records in grouped.items():
        if len(area_records) < seq_len:
            continue

        for start_index in range(0, len(area_records) - seq_len + 1, stride):
            window = area_records[start_index : start_index + seq_len]
            cumulative_days: list[float] = []
            running_days = 0.0
            delta_days: list[float] = []
            day_of_year_sin: list[float] = []
            day_of_year_cos: list[float] = []

            for time_index, record in enumerate(window):
                if time_index == 0:
                    delta_value = max(0.0, days_since_previous_lookup.get(record.row_id, 0.0))
                    running_days = 0.0
                else:
                    delta_value = float((record.date_time - window[time_index - 1].date_time).days)
                    running_days += delta_value

                doy = record.date_time.timetuple().tm_yday
                delta_days.append(float(max(0.0, delta_value)))
                cumulative_days.append(float(running_days))
                day_of_year_sin.append(math.sin((2.0 * math.pi * doy) / 365.25))
                day_of_year_cos.append(math.cos((2.0 * math.pi * doy) / 365.25))

            sequences.append(
                SequenceSample(
                    split_name=split_name,
                    area=area,
                    records=tuple(window),
                    image_names=tuple(record.image_name for record in window),
                    delta_days=tuple(delta_days),
                    cumulative_days=tuple(cumulative_days),
                    day_of_year_sin=tuple(day_of_year_sin),
                    day_of_year_cos=tuple(day_of_year_cos),
                    target=float(window[-1].target),
                )
            )

    return sequences


def summarize_sequences(sequences: list[SequenceSample]) -> dict[str, Any]:
    grouped: dict[str, list[SequenceSample]] = {}
    for sequence in sequences:
        grouped.setdefault(sequence.area, []).append(sequence)

    summary = {"count": len(sequences), "areas": {}}
    for area, area_sequences in sorted(grouped.items()):
        target_dates = [sequence.records[-1].date_time.date().isoformat() for sequence in area_sequences]
        summary["areas"][area] = {
            "count": len(area_sequences),
            "first_target_date": target_dates[0],
            "last_target_date": target_dates[-1],
        }
    return summary


def compute_time_stats(train_sequences: list[SequenceSample]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    for sequence in train_sequences:
        for index in range(len(sequence.records)):
            delta_days = float(sequence.delta_days[index])
            rows.append(
                [
                    delta_days,
                    math.log1p(delta_days),
                    float(sequence.cumulative_days[index]),
                    float(sequence.day_of_year_sin[index]),
                    float(sequence.day_of_year_cos[index]),
                ]
            )
    matrix = np.array(rows, dtype=np.float32)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return means.astype(np.float32), stds.astype(np.float32)


def compute_target_stats(train_sequences: list[SequenceSample]) -> tuple[float, float]:
    targets = np.array([sequence.target for sequence in train_sequences], dtype=np.float32)
    mean = float(targets.mean())
    std = float(targets.std())
    return mean, (std if std >= 1e-6 else 1.0)


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[SequenceSample],
        image_lookup: dict[str, np.ndarray],
        band_means: np.ndarray,
        band_stds: np.ndarray,
        time_means: np.ndarray,
        time_stds: np.ndarray,
        target_mean: float,
        target_std: float,
    ) -> None:
        self.sequences = sequences
        self.image_lookup = image_lookup
        self.band_means = torch.tensor(band_means, dtype=torch.float32).view(-1, 1, 1)
        self.band_stds = torch.tensor(band_stds, dtype=torch.float32).view(-1, 1, 1)
        self.time_means = torch.tensor(time_means, dtype=torch.float32)
        self.time_stds = torch.tensor(time_stds, dtype=torch.float32)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequences[index]

        images = []
        for image_name in sequence.image_names:
            image = torch.from_numpy(self.image_lookup[image_name]).float()
            image = (image - self.band_means) / self.band_stds
            images.append(image)
        image_tensor = torch.stack(images, dim=0)

        time_rows = []
        for step in range(len(sequence.records)):
            delta_days = float(sequence.delta_days[step])
            time_rows.append(
                [
                    delta_days,
                    math.log1p(delta_days),
                    float(sequence.cumulative_days[step]),
                    float(sequence.day_of_year_sin[step]),
                    float(sequence.day_of_year_cos[step]),
                ]
            )
        time_tensor = torch.tensor(time_rows, dtype=torch.float32)
        time_tensor = (time_tensor - self.time_means) / self.time_stds

        target_norm = (sequence.target - self.target_mean) / self.target_std
        target_tensor = torch.tensor(target_norm, dtype=torch.float32)
        return image_tensor, time_tensor, target_tensor


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=padding)
        self.conv_can = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        
    def forward(self, input_tensor: torch.Tensor, h_cur: torch.Tensor | None = None) -> torch.Tensor:
        if h_cur is None:
            b, _, h, w = input_tensor.shape
            h_cur = torch.zeros(b, self.hidden_dim, h, w, device=input_tensor.device)
            
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.split(gates, self.hidden_dim, dim=1)
        
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        
        combined_reset = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        can = torch.tanh(self.conv_can(combined_reset))
        
        h_next = (1 - update_gate) * h_cur + update_gate * can
        return h_next


class CNNTemporalRegressor(nn.Module):
    """Spatio-Temporal ConvGRU Regressor."""
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        time_feature_dim: int,
        cnn_embed_dim: int,  # Ignored (kept for arg compatibility)
        hidden_dim: int,
        num_layers: int,     # Ignored (kept for arg compatibility)
        num_heads: int,      # Ignored (kept for arg compatibility)
        dropout: float,
        temporal_head: str,  # Ignored (kept for arg compatibility)
    ) -> None:
        super().__init__()
        
        self.conv_gru = ConvGRUCell(
            input_dim=in_channels + time_feature_dim,
            hidden_dim=hidden_dim,
            kernel_size=3
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, images: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        b, s, c, h, w = images.shape
        
        time_expanded = time_features.view(b, s, -1, 1, 1).expand(b, s, -1, h, w)
        fused = torch.cat([images, time_expanded], dim=2)
        
        h_cur = None
        for t in range(s):
            h_cur = self.conv_gru(fused[:, t, :, :, :], h_cur)
            
        return self.head(h_cur).squeeze(-1)


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for images, time_features, targets in loader:
        images = images.to(device)
        time_features = time_features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images, time_features)
        loss = loss_fn(predictions, targets)
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for images, time_features, _ in loader:
            images = images.to(device)
            time_features = time_features.to(device)
            batch_predictions = model(images, time_features).cpu().numpy()
            predictions.append(batch_predictions)

    if not predictions:
        return np.array([], dtype=np.float32)

    pred_norm = np.concatenate(predictions).astype(np.float32)
    return (pred_norm * target_std) + target_mean


def save_sequence_predictions(
    output_path: Path,
    split_sequences: dict[str, list[SequenceSample]],
    predictions_by_split: dict[str, np.ndarray],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "split",
            "area",
            "sequence_start_date",
            "target_date",
            "target_image_name",
            "seq_len",
            "image_names",
            "delta_days",
            "target",
            "prediction",
            "abs_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for split_name in ("train", "val", "test"):
            sequences = split_sequences[split_name]
            predictions = predictions_by_split[split_name]
            for sequence, prediction in zip(sequences, predictions):
                writer.writerow(
                    {
                        "split": split_name,
                        "area": sequence.area,
                        "sequence_start_date": sequence.records[0].date_time.date().isoformat(),
                        "target_date": sequence.records[-1].date_time.date().isoformat(),
                        "target_image_name": sequence.image_names[-1],
                        "seq_len": len(sequence.image_names),
                        "image_names": "|".join(sequence.image_names),
                        "delta_days": "|".join(f"{value:.2f}" for value in sequence.delta_days),
                        "target": f"{sequence.target:.10f}",
                        "prediction": f"{float(prediction):.10f}",
                        "abs_error": f"{abs(float(prediction) - sequence.target):.10f}",
                    }
                )


def main() -> None:
    args = parse_args()
    set_seed(args.random_seed)
    csv_path, image_dir, output_dir = resolve_paths(args)

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
    split_sequences = {
        split_name: build_sequences(
            records=split_records,
            split_name=split_name,
            seq_len=args.seq_len,
            stride=args.stride,
            days_since_previous_lookup=days_since_previous_lookup,
        )
        for split_name, split_records in splits.items()
        if split_name in {"train", "val", "test"}
    }

    if not split_sequences["train"] or not split_sequences["val"] or not split_sequences["test"]:
        raise ValueError(
            "One of train/val/test has zero sequences. Reduce seq_len, reduce the split gap, "
            "or use a less aggressive split setup."
        )

    image_lookup = load_image_lookup(records=records, image_dir=image_dir)
    band_means, band_stds = compute_image_stats(train_records=splits["train"], image_lookup=image_lookup)
    time_means, time_stds = compute_time_stats(train_sequences=split_sequences["train"])
    target_mean, target_std = compute_target_stats(train_sequences=split_sequences["train"])

    train_dataset = SequenceDataset(
        sequences=split_sequences["train"],
        image_lookup=image_lookup,
        band_means=band_means,
        band_stds=band_stds,
        time_means=time_means,
        time_stds=time_stds,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_dataset = SequenceDataset(
        sequences=split_sequences["val"],
        image_lookup=image_lookup,
        band_means=band_means,
        band_stds=band_stds,
        time_means=time_means,
        time_stds=time_stds,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_dataset = SequenceDataset(
        sequences=split_sequences["test"],
        image_lookup=image_lookup,
        band_means=band_means,
        band_stds=band_stds,
        time_means=time_means,
        time_stds=time_stds,
        target_mean=target_mean,
        target_std=target_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    eval_loader_kwargs = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.num_workers}
    train_eval_loader = DataLoader(train_dataset, **eval_loader_kwargs)
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)

    sample_image = image_lookup[splits["train"][0].image_name]
    in_channels = int(sample_image.shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNTemporalRegressor(
        in_channels=in_channels,
        seq_len=args.seq_len,
        time_feature_dim=len(time_means),
        cnn_embed_dim=args.cnn_embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        temporal_head=args.temporal_head,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, args.patience // 3),
    )
    loss_fn = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    print(f"Training on device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_training_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            gradient_clip=args.gradient_clip,
        )

        val_predictions = predict_dataset(
            model=model,
            loader=val_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        val_targets = np.array([sequence.target for sequence in split_sequences["val"]], dtype=np.float32)
        val_metrics = compute_metrics(y_true=val_targets, y_pred=val_predictions)
        scheduler.step(val_metrics["rmse"])

        history_entry = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_rmse": float(val_metrics["rmse"]),
            "val_mae": float(val_metrics["mae"]),
            "val_r2": float(val_metrics["r2"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(history_entry)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_rmse={val_metrics['rmse']:.5f} | "
            f"val_mae={val_metrics['mae']:.5f} | "
            f"val_r2={val_metrics['r2']:.5f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    model.load_state_dict(best_state)

    train_predictions = predict_dataset(
        model=model,
        loader=train_eval_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_predictions = predict_dataset(
        model=model,
        loader=val_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_predictions = predict_dataset(
        model=model,
        loader=test_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    train_targets = np.array([sequence.target for sequence in split_sequences["train"]], dtype=np.float32)
    val_targets = np.array([sequence.target for sequence in split_sequences["val"]], dtype=np.float32)
    test_targets = np.array([sequence.target for sequence in split_sequences["test"]], dtype=np.float32)

    metrics = {
        "train": compute_metrics(y_true=train_targets, y_pred=train_predictions),
        "val": compute_metrics(y_true=val_targets, y_pred=val_predictions),
        "test": compute_metrics(y_true=test_targets, y_pred=test_predictions),
    }

    report = {
        "dataset_summary": dataset_summary,
        "split_config": {
            "split_scheme": args.split_scheme,
            "holdout_area": args.holdout_area,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "gap_samples": args.gap_samples,
            "seq_len": args.seq_len,
            "stride": args.stride,
            "temporal_head": args.temporal_head,
            "random_seed": args.random_seed,
        },
        "model_config": {
            "cnn_embed_dim": args.cnn_embed_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "gradient_clip": args.gradient_clip,
        },
        "split_summary": {
            "records": {name: summarize_split(splits[name]) for name in ("train", "val", "test", "gap")},
            "sequences": {name: summarize_sequences(split_sequences[name]) for name in ("train", "val", "test")},
        },
        "normalization": {
            "band_means": band_means.tolist(),
            "band_stds": band_stds.tolist(),
            "time_means": time_means.tolist(),
            "time_stds": time_stds.tolist(),
            "target_mean": target_mean,
            "target_std": target_std,
        },
        "best_epoch": best_epoch,
        "history": history,
        "metrics": metrics,
    }

    predictions_by_split = {
        "train": train_predictions,
        "val": val_predictions,
        "test": test_predictions,
    }
    save_sequence_predictions(
        output_path=output_dir / "sequence_predictions.csv",
        split_sequences=split_sequences,
        predictions_by_split=predictions_by_split,
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": report["model_config"],
        "split_config": report["split_config"],
        "normalization": report["normalization"],
        "in_channels": in_channels,
    }
    torch.save(checkpoint, output_dir / "model_checkpoint.pt")

    print("Leakage-safe deep baseline completed.")
    print(f"Best epoch: {best_epoch}")
    print(
        "Test metrics: "
        f"RMSE={metrics['test']['rmse']:.5f}, "
        f"MAE={metrics['test']['mae']:.5f}, "
        f"R2={metrics['test']['r2']:.5f}, "
        f"Pearson={metrics['test']['pearson_r']:.5f}"
    )
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
