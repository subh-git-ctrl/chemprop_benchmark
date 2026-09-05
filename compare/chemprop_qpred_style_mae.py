#!/usr/bin/env python3
"""
Compute Chemprop MAE in the same normalization convention used by qpred-app.

Key idea:
- qpred-app normalizes targets using training-set mean and MAD
- It reports MAE on the normalized scale and also converts that to physical units
- This script reproduces that convention WITHOUT importing QPred code

Usage examples:
    python chemprop_qpred_style_mae.py \
        --train-path ../compare/train.csv \
        --test-path ../compare/test.csv \
        --predictions-path ../compare/chemprop_predictions.csv \
        --target-column Polarizability \
        --id-column Zinc_id \
        --prediction-column prediction

    python chemprop_qpred_style_mae.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_mean_mad(train_df: pd.DataFrame, target_col: str) -> tuple[float, float]:
    """Compute mean and median absolute deviation (MAD) from the training split."""
    mean = float(train_df[target_col].mean())
    mad = float(np.abs(train_df[target_col] - mean).mean())

    if np.isclose(mad, 0.0):
        raise ValueError(f"MAD is zero for target column '{target_col}'. Cannot normalize.")

    return mean, mad


def load_and_merge(
    train_path: str,
    test_path: str,
    predictions_path: str,
    target_col: str,
    id_col: str,
    prediction_col: str,
) -> tuple[pd.DataFrame, float, float, str]:
    """Load train/test/prediction files and align them by molecule ID."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    pred_df = pd.read_csv(predictions_path)

    if id_col not in test_df.columns:
        raise ValueError(f"Column '{id_col}' not found in test file: {test_path}")
    if id_col not in pred_df.columns:
        raise ValueError(f"Column '{id_col}' not found in predictions file: {predictions_path}")
    if target_col not in test_df.columns:
        raise ValueError(f"Column '{target_col}' not found in test file: {test_path}")

    if prediction_col is None or str(prediction_col).lower() in {"auto", "none", "default"}:
        for candidate in ["prediction", "pred_0", "predicted", "predictions"]:
            if candidate in pred_df.columns:
                prediction_col = candidate
                break
        if prediction_col is None or str(prediction_col).lower() in {"auto", "none", "default"}:
            available = ", ".join(pred_df.columns.tolist())
            raise ValueError(
                "Could not auto-detect the prediction column in predictions file. "
                f"Available columns: {available}"
            )

    if prediction_col not in pred_df.columns:
        raise ValueError(
            f"Column '{prediction_col}' not found in predictions file: {predictions_path}"
        )

    mean, mad = compute_mean_mad(train_df, target_col)

    truth = test_df[[id_col, target_col]].copy()
    preds = pred_df[[id_col, prediction_col]].copy()

    merged = truth.merge(preds, on=id_col, how='inner')

    if len(merged) != len(test_df):
        print(
            f"Warning: {len(merged)} matched rows out of {len(test_df)} test rows. "
            "This usually means prediction IDs do not match the test set exactly."
        )

    return merged, mean, mad, prediction_col


def compute_metrics(
    df: pd.DataFrame,
    target_col: str,
    prediction_col: str,
    mean: float,
    mad: float,
) -> dict[str, float]:
    """Compute MAE and RMSE on both normalized and physical scales."""
    y_true = df[target_col].to_numpy(dtype=float)
    y_pred = df[prediction_col].to_numpy(dtype=float)

    # Match qpred-app normalization logic:
    # normalized_target = (raw_target - mean) / mad
    # normalized_pred   = (raw_pred   - mean) / mad
    y_true_norm = (y_true - mean) / mad
    y_pred_norm = (y_pred - mean) / mad

    mae_normalized = float(np.mean(np.abs(y_true_norm - y_pred_norm)))
    mae_physical = float(mae_normalized * mad)

    raw_mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse_physical = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rmse_normalized = float(rmse_physical / mad)

    return {
        "mean": mean,
        "mad": mad,
        "mae_normalized": mae_normalized,
        "mae_physical": mae_physical,
        "mae_raw": raw_mae,
        "rmse_normalized": rmse_normalized,
        "rmse_physical": rmse_physical,
    }


def print_summary(metrics: dict[str, float], target_col: str) -> None:
    """Pretty-print the benchmark metrics."""
    print("\n" + "=" * 80)
    print(f"Chemprop benchmark metrics for {target_col}")
    print("=" * 80)
    print(f"Training-set mean:       {metrics['mean']:.6f}")
    print(f"Training-set MAD:        {metrics['mad']:.6f}")
    print()
    print("QPred-style normalization convention:")
    print(f"  MAE (normalized)        = {metrics['mae_normalized']:.6f}")
    print(f"  MAE (physical units)    = {metrics['mae_physical']:.6f}")
    print(f"  MAE (raw difference)    = {metrics['mae_raw']:.6f}")
    print()
    print("RMSE:")
    print(f"  RMSE (normalized)       = {metrics['rmse_normalized']:.6f}")
    print(f"  RMSE (physical units)   = {metrics['rmse_physical']:.6f}")
    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Chemprop MAE using the same normalization convention as qpred-app "
            "(train-set mean and MAD), while reporting the equivalent physical-unit MAE."
        )
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="../compare/train.csv",
        help="Path to the training CSV used to compute mean/MAD. Default: ../compare/train.csv",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="../compare/test.csv",
        help="Path to the test CSV with ground-truth values. Default: ../compare/test.csv",
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        default="../compare/chemprop_predictions.csv",
        help="Path to Chemprop predictions CSV. Default: ../compare/chemprop_predictions.csv",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Polarizability",
        help="Name of the target column in the CSV files. Default: Polarizability",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="Zinc_id",
        help="Molecule ID column used to join the test truth table with predictions. Default: Zinc_id",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="auto",
        help="Prediction column in the Chemprop predictions CSV. Auto-detects: prediction, pred_0, predicted, predictions.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save a merged predictions file with raw and normalized metrics per row.",
    )

    args = parser.parse_args()

    train_path = Path(args.train_path).resolve()
    test_path = Path(args.test_path).resolve()
    pred_path = Path(args.predictions_path).resolve()

    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {pred_path}")

    merged, mean, mad, prediction_column = load_and_merge(
        str(train_path),
        str(test_path),
        str(pred_path),
        args.target_column,
        args.id_column,
        args.prediction_column,
    )

    metrics = compute_metrics(merged, args.target_column, prediction_column, mean, mad)
    print_summary(metrics, args.target_column)

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        merged = merged.copy()
        merged["y_true_norm"] = (merged[args.target_column].to_numpy(dtype=float) - mean) / mad
        merged["y_pred_norm"] = (merged[prediction_column].to_numpy(dtype=float) - mean) / mad
        merged["abs_error_norm"] = np.abs(merged["y_true_norm"] - merged["y_pred_norm"])
        merged["abs_error_physical"] = merged["abs_error_norm"] * mad

        merged.to_csv(out_path, index=False)
        print(f"Saved per-row metrics to: {out_path}")


def entry_point() -> None:
    main()


if __name__ == "__main__":
    entry_point()
