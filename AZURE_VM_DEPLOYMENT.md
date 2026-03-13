# Azure VM Deployment

This project is ready to run on a Linux Azure VM using `uvicorn`, `systemd`, and optional `nginx`.

## 1. Copy project to the VM

Upload the repo to your VM, for example into:

```bash
/home/azureuser/BTCUSD_MLFlow
```

## 2. Install dependencies

From the project root:

```bash
chmod +x deploy/setup_vm.sh deploy/run_vm.sh
./deploy/setup_vm.sh
```

This will:
- install Python and `nginx`
- create `.venv`
- install `requirements.txt`
- create `.env` from `.env.example` if needed

## 3. Configure environment

Edit `.env` and set the values you want:

```bash
HOST=0.0.0.0
PORT=8000
WEB_CONCURRENCY=1
UVICORN_RELOAD=false
CORS_ORIGINS=*
```

You can also change:
- `DATASET_PATH`
- `MODEL_PATH`
- `MLFLOW_DB_PATH`
- `MLFLOW_TRACKING_URI`

## 4. Start the app manually

```bash
./deploy/run_vm.sh
```

Open:

```text
http://<your-vm-public-ip>/
```

If Azure NSG is enabled, allow inbound traffic on port `80` or `8000`.

## 5. Run as a background service

Update the paths inside:

```text
deploy/systemd/btcusd-mlflow.service
```

Then install it:

```bash
sudo cp deploy/systemd/btcusd-mlflow.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable btcusd-mlflow
sudo systemctl start btcusd-mlflow
sudo systemctl status btcusd-mlflow
```

## 6. Put nginx in front

```bash
sudo cp deploy/nginx/btcusd-mlflow.conf /etc/nginx/sites-available/btcusd-mlflow
sudo ln -s /etc/nginx/sites-available/btcusd-mlflow /etc/nginx/sites-enabled/btcusd-mlflow
sudo nginx -t
sudo systemctl restart nginx
```

After that, browse to:

```text
http://<your-vm-public-ip>/
```

## Notes

- The FastAPI app serves the frontend directly from `frontend/`.
- MLflow defaults to a local SQLite database at `mlflow.db` unless `MLFLOW_TRACKING_URI` is set.
- If the dataset or trained model is missing, run:

```bash
.venv/bin/python fetch_data.py
.venv/bin/python fit_models.py
```
