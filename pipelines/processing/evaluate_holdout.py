"""Evaluate V1 direct multi-horizon forecasts on the untouched temporal holdout."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["store_nbr", "family", "target_date"]


def digest(path: Path) -> str:
    """Hash the exact input artifact used in this evaluation."""
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(frame: pd.DataFrame) -> dict:
    """Use portfolio-weighted absolute error, not mean individual percentage errors."""
    actual = frame["target_sales"].to_numpy(dtype=float)
    pred = frame["prediction"].to_numpy(dtype=float)
    error = pred - actual
    volume = float(np.abs(actual).sum())
    return {
        "count": len(frame),
        "actual_abs_volume": volume,
        "wape": float(np.abs(error).sum() / volume) if volume > 0 else None,
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "normalized_bias": float(error.sum() / volume) if volume > 0 else None,
    }


def evaluate(forecasts: pd.DataFrame, actuals: pd.DataFrame,
             origin: str, expected_rows: int) -> tuple[pd.DataFrame, dict]:
    """Strict one-to-one holdout join with no silent label imputation."""
    forecasts = forecasts.copy()
    actuals = actuals.copy()
    for frame in (forecasts, actuals):
        frame["target_date"] = pd.to_datetime(frame["target_date"])
        frame["family"] = frame["family"].astype(str)
    forecasts["forecast_origin"] = pd.to_datetime(forecasts["forecast_origin"])

    if len(forecasts) != expected_rows or len(actuals) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in each input")
    if forecasts.duplicated(KEYS).any() or actuals.duplicated(KEYS).any():
        raise ValueError("Duplicate store-family-target date keys")
    if not forecasts["forecast_origin"].eq(pd.Timestamp(origin)).all():
        raise ValueError("Unexpected forecast origin")
    if not forecasts["horizon"].between(1, 14).all():
        raise ValueError("Invalid 14-day forecasting horizon")
    expected_date = forecasts["forecast_origin"] + pd.to_timedelta(
        forecasts["horizon"], unit="D"
    )
    if not forecasts["target_date"].eq(expected_date).all():
        raise ValueError("Forecast dates and horizons do not agree")

    aligned = forecasts.merge(
        actuals[KEYS + ["target_sales"]], on=KEYS,
        how="left", validate="one_to_one", indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Missing observed sales for forecast keys")
    aligned = aligned.drop(columns="_merge")
    if not np.isfinite(aligned[["prediction", "target_sales"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite actual sales or predictions")
    if aligned.groupby("horizon").size().nunique() != 1 or aligned["horizon"].nunique() != 14:
        raise ValueError("Unequal or missing horizon coverage")

    report = {
        "status": "PASSED",
        "validation_type": "frozen_v1_final_temporal_holdout",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_origin": origin,
        "holdout_start": str(aligned["target_date"].min().date()),
        "holdout_end": str(aligned["target_date"].max().date()),
        "series_count": int(aligned[["store_nbr", "family"]].drop_duplicates().shape[0]),
        "portfolio": metrics(aligned),
    }
    return aligned, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecasts", type=Path, default=Path(
        "/opt/ml/processing/input/forecasts/final_holdout_forecast.parquet"))
    parser.add_argument("--actuals", type=Path, default=Path(
        "/opt/ml/processing/input/actuals/holdout_actuals.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("/opt/ml/processing/output"))
    parser.add_argument("--origin", default="2017-08-01")
    parser.add_argument("--expected-rows", type=int, default=24948)
    parser.add_argument("--reference-wape", type=float, required=True)
    parser.add_argument("--wape-tolerance", type=float, default=0.00001)
    args = parser.parse_args()

    aligned, report = evaluate(
        pd.read_parquet(args.forecasts), pd.read_parquet(args.actuals),
        args.origin, args.expected_rows,
    )
    report["input_sha256"] = {
        "forecasts": digest(args.forecasts),
        "actuals": digest(args.actuals),
    }
    reference_diff = abs(report["portfolio"]["wape"] - args.reference_wape)
    report["reference_wape"] = args.reference_wape
    report["reference_wape_absolute_difference"] = reference_diff
    if reference_diff > args.wape_tolerance:
        raise ValueError(
            f"WAPE does not match notebook reference: "
            f"actual={report['portfolio']['wape']:.9f}, "
            f"reference={args.reference_wape:.9f}, difference={reference_diff:.9f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_horizon = pd.DataFrame([
        {"horizon": int(name), **metrics(group)}
        for name, group in aligned.groupby("horizon", sort=True)
    ])
    by_family = pd.DataFrame([
        {"family": name, **metrics(group)}
        for name, group in aligned.groupby("family", sort=True)
    ])
    by_horizon.to_csv(args.output_dir / "metrics_by_horizon.csv", index=False)
    by_family.to_csv(args.output_dir / "metrics_by_family.csv", index=False)
    aligned.sort_values(KEYS).to_parquet(
        args.output_dir / "predictions_with_actuals.parquet", index=False
    )
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("SageMaker V1 holdout evaluation: PASSED")


if __name__ == "__main__":
    main()
