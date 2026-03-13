from __future__ import annotations

import os
from pathlib import Path

import mlflow


ROOT_DIR = Path(__file__).resolve().parent


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(env_name: str, default_relative_path: str) -> Path:
    raw_value = os.getenv(env_name)
    if not raw_value:
        return ROOT_DIR / default_relative_path

    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return ROOT_DIR / candidate


DATASET_PATH = _resolve_path("DATASET_PATH", "btc_usd_2y_1h_data.csv")
MODEL_PATH = _resolve_path("MODEL_PATH", "models/linear_regression_model.joblib")
FRONTEND_DIR = _resolve_path("FRONTEND_DIR", "frontend")
MLFLOW_DB_PATH = _resolve_path("MLFLOW_DB_PATH", "mlflow.db")

APP_HOST = os.getenv("HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", "8000"))
WEB_CONCURRENCY = max(int(os.getenv("WEB_CONCURRENCY", "1")), 1)
UVICORN_RELOAD = _to_bool(os.getenv("UVICORN_RELOAD"), default=False)

CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "*").strip()
if CORS_ORIGINS_RAW == "*":
    CORS_ORIGINS = ["*"]
else:
    CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_RAW.split(",") if origin.strip()]

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "btc_forecasting_pipeline")


def configure_mlflow_tracking() -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        return tracking_uri

    default_tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    mlflow.set_tracking_uri(default_tracking_uri)
    return default_tracking_uri
