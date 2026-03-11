from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import mlflow.statsmodels

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMA

from clean_data import load_data_from_fetch_data


# MLflow experiment
mlflow.set_experiment("btc_forecasting_pipeline")


def get_cleaned_data(csv_path: str | Path = "btc_usd_2y_1h_data.csv") -> pd.DataFrame:
	"""Load cleaned BTC-USD data from clean_data.py and ensure time ordering."""
	df = load_data_from_fetch_data(csv_path=csv_path).copy()
	df = df.sort_values("time_stamp").reset_index(drop=True)
	return df


def normalize_features(
	train_df: pd.DataFrame,
	test_df: pd.DataFrame,
	feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
	"""Fit scaler on train data and transform both train/test sets."""
	scaler = StandardScaler()

	x_train = scaler.fit_transform(train_df[feature_columns])
	x_test = scaler.transform(test_df[feature_columns])

	return x_train, x_test, scaler


def train_and_save_linear_regression(
	df: pd.DataFrame,
	model_path: str | Path = "models/linear_regression_model.joblib",
	test_size: float = 0.2,
) -> dict[str, Any]:

	feature_columns = [
		"open",
		"high",
		"low",
		"volume",
		"rsi",
		"ma_20",
		"ma_50",
		"volatility",
	]

	target_column = "close"

	split_index = int(len(df) * (1 - test_size))

	train_df = df.iloc[:split_index].copy()
	test_df = df.iloc[split_index:].copy()

	x_train, x_test, scaler = normalize_features(train_df, test_df, feature_columns)

	y_train = train_df[target_column].values
	y_test = test_df[target_column].values

	param_grid = {
		"fit_intercept": [True, False],
		"positive": [False, True],
	}

	tscv = TimeSeriesSplit(n_splits=5)

	with mlflow.start_run(run_name="LinearRegression", nested=True):

		mlflow.log_param("model", "LinearRegression")
		mlflow.log_param("test_size", test_size)
		mlflow.log_param("features", feature_columns)

		search = GridSearchCV(
			estimator=LinearRegression(),
			param_grid=param_grid,
			scoring="neg_mean_squared_error",
			cv=tscv,
			n_jobs=-1,
		)

		search.fit(x_train, y_train)

		best_model: LinearRegression = search.best_estimator_

		pred_test = best_model.predict(x_test)

		metrics = {
			"rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
			"mae": float(mean_absolute_error(y_test, pred_test)),
			"r2": float(r2_score(y_test, pred_test)),
		}

		mlflow.log_params(search.best_params_)
		mlflow.log_metrics(metrics)

		mlflow.sklearn.log_model(
			sk_model=best_model,
			artifact_path="linear_regression_model"
		)

		artifact = {
			"model": best_model,
			"scaler": scaler,
			"feature_columns": feature_columns,
			"target_column": target_column,
			"metrics": metrics,
			"best_params": search.best_params_,
		}

		output_path = Path(model_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)

		joblib.dump(artifact, output_path)

		mlflow.log_artifact(str(output_path))

		return {
			"model_path": str(output_path),
			"metrics": metrics,
			"best_params": search.best_params_,
		}


def _select_best_arima_order(
	series: pd.Series,
	p_values: list[int] | None = None,
	d_values: list[int] | None = None,
	q_values: list[int] | None = None,
) -> tuple[int, int, int]:

	p_values = p_values or [0, 1, 2, 3]
	d_values = d_values or [0, 1]
	q_values = q_values or [0, 1, 2, 3]

	best_order = (1, 1, 1)
	best_aic = float("inf")

	for p in p_values:
		for d in d_values:
			for q in q_values:
				try:
					fit = ARIMA(series, order=(p, d, q)).fit()

					if fit.aic < best_aic:
						best_aic = fit.aic
						best_order = (p, d, q)

				except Exception:
					continue

	return best_order


def train_and_save_arima(
	df: pd.DataFrame,
	model_path: str | Path = "models/arima_model.pkl",
	test_size: float = 0.2,
) -> dict[str, Any]:

	close_series = df["close"].astype(float)

	split_index = int(len(close_series) * (1 - test_size))

	train_series = close_series.iloc[:split_index]
	test_series = close_series.iloc[split_index:]

	with mlflow.start_run(run_name="ARIMA", nested=True):

		best_order = _select_best_arima_order(train_series)

		model_fit = ARIMA(train_series, order=best_order).fit()

		forecast = model_fit.forecast(steps=len(test_series))

		metrics = {
			"rmse": float(np.sqrt(mean_squared_error(test_series.values, forecast.values))),
			"mae": float(mean_absolute_error(test_series.values, forecast.values)),
		}

		mlflow.log_param("model", "ARIMA")
		mlflow.log_param("order", best_order)
		mlflow.log_param("test_size", test_size)

		mlflow.log_metrics(metrics)

		mlflow.statsmodels.log_model(
			statsmodels_model=model_fit,
			artifact_path="arima_model"
		)

		output_path = Path(model_path)

		output_path.parent.mkdir(parents=True, exist_ok=True)

		model_fit.save(str(output_path))

		mlflow.log_artifact(str(output_path))

		return {
			"model_path": str(output_path),
			"best_order": best_order,
			"metrics": metrics,
		}


def train_and_save_models(
	csv_path: str | Path = "btc_usd_2y_1h_data.csv",
) -> dict[str, dict[str, Any]]:

	df = get_cleaned_data(csv_path=csv_path)

	with mlflow.start_run(run_name="BTC_Forecasting_Pipeline"):

		linear_result = train_and_save_linear_regression(df)

		arima_result = train_and_save_arima(df)

		# Log pipeline-level summary metrics so they are visible on the parent run.
		parent_metrics = {
			"linear_rmse": float(linear_result["metrics"]["rmse"]),
			"linear_mae": float(linear_result["metrics"]["mae"]),
			"linear_r2": float(linear_result["metrics"]["r2"]),
			"arima_rmse": float(arima_result["metrics"]["rmse"]),
			"arima_mae": float(arima_result["metrics"]["mae"]),
		}

		mlflow.log_metrics(parent_metrics)

		mlflow.log_params(
			{
				"linear_best_fit_intercept": bool(linear_result["best_params"]["fit_intercept"]),
				"linear_best_positive": bool(linear_result["best_params"]["positive"]),
				"arima_best_order": str(arima_result["best_order"]),
			}
		)

	return {
		"linear_regression": linear_result,
		"arima": arima_result,
	}


if __name__ == "__main__":

	results = train_and_save_models()

	print("\nModel training complete.")

	print(f"Linear Regression saved: {results['linear_regression']['model_path']}")
	print(f"Linear Regression metrics: {results['linear_regression']['metrics']}")
	print(f"Linear Regression best params: {results['linear_regression']['best_params']}")

	print(f"ARIMA saved: {results['arima']['model_path']}")
	print(f"ARIMA best order: {results['arima']['best_order']}")
	print(f"ARIMA metrics: {results['arima']['metrics']}")