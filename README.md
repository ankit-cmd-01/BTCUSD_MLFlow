# BTCUSD MLflow Forecasting Dashboard

End-to-end BTC-USD forecasting workflow with MLflow experiment tracking, a FastAPI backend, and a custom dashboard frontend.

## Project Overview
This project contains:
- Data fetch and feature engineering for BTC-USD hourly candles
- Data cleaning and exploratory checks
- Model training for:
  - Linear Regression (with `GridSearchCV` + `TimeSeriesSplit`)
  - ARIMA (order selected by AIC)
- MLflow logging for runs, params, and metrics
- FastAPI endpoints for prediction, performance, and drift checks
- Browser dashboard served directly by FastAPI

## Project Structure
- `fetch_data.py`: Download BTC-USD data and create features (`rsi`, `ma_20`, `ma_50`, `volatility`)
- `clean_data.py`: Load dataset and run EDA-style checks
- `fit_models.py`: Train models and log outputs to MLflow
- `dashboard_api.py`: API layer + static frontend hosting
- `frontend/index.html`: Dashboard UI
- `frontend/app.js`: Dashboard data loading + interactions
- `frontend/styles.css`: Dashboard styling
- `requirements.txt`: Python dependencies
- `models/`: Saved model artifacts (joblib/pkl)
- `mlruns/`: MLflow tracking artifacts

## Requirements
- Python 3.10+
- Virtual environment (recommended)

Install dependencies:

```powershell
D:/ML_Flow/.venv/Scripts/python.exe -m pip install -r D:/ML_Flow/requirements.txt
```

## Run Pipeline
Run each step from `D:/ML_Flow`.

1. Fetch and engineer dataset

```powershell
D:/ML_Flow/.venv/Scripts/python.exe D:/ML_Flow/fetch_data.py
```

2. Optional: run cleaning + EDA checks

```powershell
D:/ML_Flow/.venv/Scripts/python.exe D:/ML_Flow/clean_data.py
```

3. Train and log models

```powershell
D:/ML_Flow/.venv/Scripts/python.exe D:/ML_Flow/fit_models.py
```

4. Start API + dashboard

```powershell
D:/ML_Flow/.venv/Scripts/python.exe -m uvicorn dashboard_api:app --host 127.0.0.1 --port 8000
```

Open in browser:
- `http://127.0.0.1:8000`

## API Endpoints
- `GET /api/health`
- `GET /api/dashboard/overview`
- `GET /api/predict-next`
- `GET /api/performance?window=168`
- `GET /api/model-drift?reference_window=720&current_window=168&rmse_alert_threshold=0.15`
- `GET /api/mlflow/runs?limit=12`
- `GET /api/series?points=300`

## Model Drift Logic
- Performance drift alert:
  - `(rmse_current - rmse_reference) / rmse_reference > rmse_alert_threshold`
- Feature drift:
  - Population Stability Index (PSI)
  - PSI `>= 0.2` is flagged

## Notes
- `fit_models.py` is the main training script used by the API model artifact path.
- If dataset or model files are missing, API endpoints return clear HTTP errors with guidance.
- MLflow artifacts are stored locally in this project (`mlruns/`).
