from __future__ import annotations

from pathlib import Path

import pandas as pd

from fetch_data import get_btc_usd_hourly_data


def load_data_from_fetch_data(csv_path: str | Path = "btc_usd_2y_1h_data.csv") -> pd.DataFrame:
	"""Load dataset from fetch_data output; generate it first if missing."""
	path = Path(csv_path)

	if not path.exists():
		get_btc_usd_hourly_data(csv_path=path)

	df = pd.read_csv(path)
	df["time_stamp"] = pd.to_datetime(df["time_stamp"], utc=True, errors="coerce")
	return df


def perform_eda(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series | int | str]:
	"""Run core EDA checks: shape, dtypes, nulls, duplicates, and descriptive stats."""
	numeric_cols = df.select_dtypes(include="number").columns.tolist()

	duplicate_rows = int(df.duplicated().sum())
	timestamp_duplicates = int(df.duplicated(subset=["time_stamp"]).sum())

	null_counts = df.isna().sum()
	null_percent = (null_counts / len(df) * 100).round(4)
	null_report = pd.DataFrame({"null_count": null_counts, "null_percent": null_percent})

	stats_numeric = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()

	eda_report: dict[str, pd.DataFrame | pd.Series | int | str] = {
		"rows": len(df),
		"columns": df.shape[1],
		"column_names": ", ".join(df.columns),
		"dtypes": df.dtypes,
		"duplicate_rows": duplicate_rows,
		"duplicate_time_stamp": timestamp_duplicates,
		"null_report": null_report,
		"numeric_summary": stats_numeric,
	}

	if "time_stamp" in df.columns:
		eda_report["time_min"] = str(df["time_stamp"].min())
		eda_report["time_max"] = str(df["time_stamp"].max())

	return eda_report


def print_eda_report(eda_report: dict[str, pd.DataFrame | pd.Series | int | str]) -> None:
	"""Pretty-print EDA results to console."""
	print("\n=== Basic Shape ===")
	print(f"Rows: {eda_report['rows']}")
	print(f"Columns: {eda_report['columns']}")
	print(f"Time Range: {eda_report.get('time_min')} -> {eda_report.get('time_max')}")

	print("\n=== Data Types ===")
	print(eda_report["dtypes"])

	print("\n=== Duplicates ===")
	print(f"Duplicate full rows: {eda_report['duplicate_rows']}")
	print(f"Duplicate timestamps: {eda_report['duplicate_time_stamp']}")

	print("\n=== Null Values ===")
	print(eda_report["null_report"])

	print("\n=== Numeric Summary ===")
	print(eda_report["numeric_summary"])


if __name__ == "__main__":
	data = load_data_from_fetch_data()
	print("Data preview:")
	print(data.head())

	report = perform_eda(data)
	print_eda_report(report)
