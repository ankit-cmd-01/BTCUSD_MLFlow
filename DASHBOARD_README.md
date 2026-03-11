# BTC Custom MLflow Dashboard

## What this adds
- FastAPI backend for MLflow + model analytics.
- Frontend dashboard for prediction, performance, and model drift checks.

## Files
- `dashboard_api.py`: backend API and static file host.
- `frontend/index.html`: dashboard layout.
- `frontend/styles.css`: visual style.
- `frontend/app.js`: frontend logic and API calls.
- `requirements.txt`: required packages.

## Run
1. Activate your virtual environment.
2. Install packages:
   - `D:/ML_Flow/.venv/Scripts/python.exe -m pip install -r requirements.txt`
3. Start API + UI server:
   - `D:/ML_Flow/.venv/Scripts/python.exe -m uvicorn dashboard_api:app --host 127.0.0.1 --port 8000`
4. Open browser:
   - `http://127.0.0.1:8000`

## Main API endpoints
- `GET /api/dashboard/overview`
- `GET /api/predict-next`
- `GET /api/performance?window=168`
- `GET /api/model-drift?reference_window=720&current_window=168&rmse_alert_threshold=0.15`
- `GET /api/mlflow/runs?limit=12`
- `GET /api/series?points=300`

## Drift options
You can tune these in the frontend drift panel:
- Reference Window: baseline period in hours.
- Current Window: recent period in hours.
- RMSE Alert Threshold: relative increase threshold for performance drift.

### Drift logic used
- Performance drift alert triggers when:
  - `(rmse_current - rmse_reference) / rmse_reference > rmse_alert_threshold`
- Feature drift is measured by PSI for each feature.
  - PSI >= 0.2 is flagged as drift.
