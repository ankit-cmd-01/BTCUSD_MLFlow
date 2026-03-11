from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mlflow.tracking import MlflowClient

from clean_data import load_data_from_fetch_data


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT_DIR / "btc_usd_2y_1h_data.csv"
DEFAULT_MODEL = ROOT_DIR / "models" / "linear_regression_model.joblib"
FRONTEND_DIR = ROOT_DIR / "frontend"

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "volume",
    "rsi",
    "ma_20",
    "ma_50",
    "volatility",
]


app = FastAPI(title="BTC MLflow Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if np.isfinite(val):
            return val
        return None
    except (TypeError, ValueError):
        return None


def _load_dataset() -> pd.DataFrame:
    if not DEFAULT_DATASET.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Run fetch_data.py first.")

    df = load_data_from_fetch_data(DEFAULT_DATASET)
    df = df.sort_values("time_stamp").reset_index(drop=True)
    return df


def _load_model_artifact() -> dict[str, Any]:
    if not DEFAULT_MODEL.exists():
        raise HTTPException(status_code=404, detail="Model artifact not found. Run fit_models.py first.")

    artifact = joblib.load(DEFAULT_MODEL)

    required_keys = {"model", "scaler", "feature_columns", "metrics", "target_column"}
    missing_keys = required_keys.difference(artifact.keys())
    if missing_keys:
        raise HTTPException(
            status_code=500,
            detail=f"Model artifact is missing required keys: {sorted(missing_keys)}",
        )

    return artifact


def _get_mlflow_client() -> MlflowClient:
    tracking_uri = mlflow.get_tracking_uri()

    if tracking_uri == "file:///mlruns":
        mlflow.set_tracking_uri(f"file:///{(ROOT_DIR / 'mlruns').as_posix()}")

    return MlflowClient()


def _latest_runs(limit: int = 10) -> list[dict[str, Any]]:
    client = _get_mlflow_client()
    experiments = client.search_experiments()

    all_runs: list[dict[str, Any]] = []

    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=limit,
            order_by=["attributes.start_time DESC"],
        )

        for run in runs:
            all_runs.append(
                {
                    "experiment_name": exp.name,
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": {k: _safe_float(v) for k, v in run.data.metrics.items()},
                    "params": dict(run.data.params),
                }
            )

    all_runs.sort(key=lambda item: item["start_time"] or 0, reverse=True)
    return all_runs[:limit]


def _get_model_metrics_summary() -> dict[str, Any]:
    runs = _latest_runs(limit=50)

    linear_metrics: dict[str, float | None] | None = None
    arima_metrics: dict[str, float | None] | None = None

    for run in runs:
        model_name = str(run.get("params", {}).get("model", "")).strip().lower()
        metrics = run.get("metrics", {})

        if model_name == "linearregression" and linear_metrics is None:
            linear_metrics = {
                "rmse": _safe_float(metrics.get("rmse")),
                "mae": _safe_float(metrics.get("mae")),
                "r2": _safe_float(metrics.get("r2")),
            }

        if model_name == "arima" and arima_metrics is None:
            arima_metrics = {
                "rmse": _safe_float(metrics.get("rmse")),
                "mae": _safe_float(metrics.get("mae")),
                "r2": _safe_float(metrics.get("r2")),
            }

        if linear_metrics and arima_metrics:
            break

    # Fallback to parent run metrics if child runs were not found.
    if linear_metrics is None or arima_metrics is None:
        for run in runs:
            metrics = run.get("metrics", {})

            if linear_metrics is None and metrics.get("linear_rmse") is not None:
                linear_metrics = {
                    "rmse": _safe_float(metrics.get("linear_rmse")),
                    "mae": _safe_float(metrics.get("linear_mae")),
                    "r2": _safe_float(metrics.get("linear_r2")),
                }

            if arima_metrics is None and metrics.get("arima_rmse") is not None:
                arima_metrics = {
                    "rmse": _safe_float(metrics.get("arima_rmse")),
                    "mae": _safe_float(metrics.get("arima_mae")),
                    "r2": None,
                }

            if linear_metrics and arima_metrics:
                break

    linear_metrics = linear_metrics or {"rmse": None, "mae": None, "r2": None}
    arima_metrics = arima_metrics or {"rmse": None, "mae": None, "r2": None}

    def _delta(arima_value: float | None, linear_value: float | None) -> float | None:
        if arima_value is None or linear_value is None:
            return None
        return round(arima_value - linear_value, 6)

    return {
        "linear_regression": linear_metrics,
        "arima": arima_metrics,
        "delta_arima_minus_linear": {
            "rmse": _delta(arima_metrics["rmse"], linear_metrics["rmse"]),
            "mae": _delta(arima_metrics["mae"], linear_metrics["mae"]),
            "r2": _delta(arima_metrics["r2"], linear_metrics["r2"]),
        },
    }


def _calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.replace([np.inf, -np.inf], np.nan).dropna()
    current = current.replace([np.inf, -np.inf], np.nan).dropna()

    if len(reference) < bins or len(current) < bins:
        return float("nan")

    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return float("nan")

    ref_hist, _ = np.histogram(reference, bins=breakpoints)
    cur_hist, _ = np.histogram(current, bins=breakpoints)

    ref_pct = np.where(ref_hist == 0, 1e-6, ref_hist / max(ref_hist.sum(), 1))
    cur_pct = np.where(cur_hist == 0, 1e-6, cur_hist / max(cur_hist.sum(), 1))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/series")
def price_series(points: int = Query(default=300, ge=50, le=2000)) -> dict[str, Any]:
    df = _load_dataset().tail(points)
    return {
        "points": len(df),
        "time_stamp": df["time_stamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
        "close": df["close"].astype(float).round(4).tolist(),
    }


@app.get("/api/mlflow/runs")
def mlflow_runs(limit: int = Query(default=12, ge=1, le=50)) -> dict[str, Any]:
    runs = _latest_runs(limit=limit)
    return {"count": len(runs), "runs": runs}


@app.get("/api/predict-next")
def predict_next_hour() -> dict[str, Any]:
    df = _load_dataset()
    artifact = _load_model_artifact()

    feature_columns = artifact["feature_columns"]
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing columns in dataset: {missing}")

    latest_row = df.iloc[[-1]].copy()
    x_latest = artifact["scaler"].transform(latest_row[feature_columns])
    prediction = float(artifact["model"].predict(x_latest)[0])

    current_close = float(latest_row["close"].iloc[0])
    delta = prediction - current_close
    pct_change = (delta / current_close * 100.0) if current_close else None

    return {
        "last_timestamp": latest_row["time_stamp"].iloc[0].isoformat(),
        "last_close": round(current_close, 4),
        "predicted_next_hour_close": round(prediction, 4),
        "delta": round(delta, 4),
        "delta_pct": round(pct_change, 4) if pct_change is not None else None,
        "note": "Prediction uses latest engineered feature row as a one-hour proxy.",
    }


@app.get("/api/performance")
def performance(window: int = Query(default=168, ge=24, le=2000)) -> dict[str, Any]:
    df = _load_dataset()
    artifact = _load_model_artifact()

    feature_columns = artifact["feature_columns"]
    target_column = artifact["target_column"]

    model = artifact["model"]
    scaler = artifact["scaler"]

    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:].copy()

    x_test = scaler.transform(test_df[feature_columns])
    y_test = test_df[target_column].astype(float).values
    y_pred = model.predict(x_test)

    if window > len(test_df):
        window = len(test_df)

    y_recent = y_test[-window:]
    y_recent_pred = y_pred[-window:]

    rmse = float(np.sqrt(np.mean((y_recent - y_recent_pred) ** 2)))
    mae = float(np.mean(np.abs(y_recent - y_recent_pred)))

    denom = np.sum((y_recent - y_recent.mean()) ** 2)
    r2 = float(1 - np.sum((y_recent - y_recent_pred) ** 2) / denom) if denom else 0.0

    return {
        "window": int(window),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "r2": round(r2, 6),
        "mlflow_logged_metrics": artifact.get("metrics", {}),
    }


@app.get("/api/model-drift")
def model_drift(
    reference_window: int = Query(default=720, ge=120, le=5000),
    current_window: int = Query(default=168, ge=24, le=1000),
    rmse_alert_threshold: float = Query(default=0.15, ge=0.01, le=1.0),
) -> dict[str, Any]:
    df = _load_dataset()
    artifact = _load_model_artifact()

    feature_columns = artifact["feature_columns"]
    target_column = artifact["target_column"]
    model = artifact["model"]
    scaler = artifact["scaler"]

    min_required = reference_window + current_window
    if len(df) < min_required:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough rows for requested windows. Need >= {min_required}, got {len(df)}",
        )

    reference_df = df.iloc[-(reference_window + current_window) : -current_window].copy()
    current_df = df.iloc[-current_window:].copy()

    x_ref = scaler.transform(reference_df[feature_columns])
    y_ref = reference_df[target_column].astype(float).values
    pred_ref = model.predict(x_ref)

    x_cur = scaler.transform(current_df[feature_columns])
    y_cur = current_df[target_column].astype(float).values
    pred_cur = model.predict(x_cur)

    rmse_ref = float(np.sqrt(np.mean((y_ref - pred_ref) ** 2)))
    rmse_cur = float(np.sqrt(np.mean((y_cur - pred_cur) ** 2)))

    rmse_drift_ratio = (rmse_cur - rmse_ref) / rmse_ref if rmse_ref else 0.0
    performance_drift = rmse_drift_ratio > rmse_alert_threshold

    psi_by_feature: dict[str, float | None] = {}
    for col in feature_columns:
        psi = _calculate_psi(reference_df[col], current_df[col])
        psi_by_feature[col] = None if not np.isfinite(psi) else round(float(psi), 6)

    flagged_features = [
        col for col, psi in psi_by_feature.items() if psi is not None and psi >= 0.2
    ]

    return {
        "reference_window": reference_window,
        "current_window": current_window,
        "rmse_reference": round(rmse_ref, 6),
        "rmse_current": round(rmse_cur, 6),
        "rmse_drift_ratio": round(float(rmse_drift_ratio), 6),
        "performance_drift_alert": performance_drift,
        "rmse_alert_threshold": rmse_alert_threshold,
        "feature_psi": psi_by_feature,
        "feature_drift_alert_features": flagged_features,
        "psi_alert_rule": "PSI >= 0.2 indicates moderate to high drift.",
    }


@app.get("/api/dashboard/overview")
def dashboard_overview() -> dict[str, Any]:
    prediction = predict_next_hour()
    perf = performance(window=168)
    drift = model_drift(reference_window=720, current_window=168, rmse_alert_threshold=0.15)
    model_metrics = _get_model_metrics_summary()
    runs = mlflow_runs(limit=8)

    return {
        "prediction": prediction,
        "performance": perf,
        "drift": drift,
        "model_metrics": model_metrics,
        "runs": runs,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dashboard_api:app", host="0.0.0.0", port=8000, reload=True)


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
