from __future__ import annotations

from pathlib import Path
import pandas as pd
import yfinance as yf


def get_btc_usd_hourly_data(
	csv_path: str | Path = "btc_usd_2y_1h_data.csv",
	period: str = "2y",
	interval: str = "1h",
	drop_incomplete_rows: bool = True,
) -> pd.DataFrame:
	"""
	Fetch BTC-USD data and engineer required features.

	Fields created:
	- stock_name
	- time_stamp
	- ohlcv: open, high, low, close, volume
	- rsi
	- moving averages: ma_20, ma_50
	- volatility (rolling std of 1-hour returns)

	The resulting DataFrame is saved to CSV and also returned.
	"""
	ticker_symbol = "BTC-USD"

	df = yf.download(
		tickers=ticker_symbol,
		period=period,
		interval=interval,
		auto_adjust=False,
		progress=False,
	)

	if df.empty:
		raise ValueError("No data returned from yfinance for BTC-USD.")

	# Normalize yfinance output (single-level or MultiIndex columns).
	if isinstance(df.columns, pd.MultiIndex):
		df.columns = [str(col[0]).lower().replace(" ", "_") for col in df.columns]
	else:
		df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]

	# Calculate indicators without mutating the original frame repeatedly.
	close = df["close"]
	delta = close.diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)
	avg_gain = gain.rolling(window=14, min_periods=14).mean()
	avg_loss = loss.rolling(window=14, min_periods=14).mean()
	rs = avg_gain / avg_loss
	rsi = 100 - (100 / (1 + rs))

	ma_20 = close.rolling(window=20, min_periods=20).mean()
	ma_50 = close.rolling(window=50, min_periods=50).mean()
	returns_1h = close.pct_change()
	volatility = returns_1h.rolling(window=24, min_periods=24).std()

	# Build final output directly, avoiding reset_index() copy issues in some pandas setups.
	result_df = pd.DataFrame(
		{
			"stock_name": ticker_symbol,
			"time_stamp": df.index,
			"open": df["open"],
			"high": df["high"],
			"low": df["low"],
			"close": close,
			"volume": df["volume"],
			"rsi": rsi,
			"ma_20": ma_20,
			"ma_50": ma_50,
			"volatility": volatility,
		}
	)
	result_df.index = pd.RangeIndex(start=0, stop=len(result_df))

	if drop_incomplete_rows:
		# Remove initial warm-up rows where rolling indicators are not yet available.
		result_df = result_df.dropna(subset=["rsi", "ma_20", "ma_50", "volatility"]).reset_index(drop=True)

	output_path = Path(csv_path)
	result_df.to_csv(output_path, index=False)

	return result_df


if __name__ == "__main__":
	data = get_btc_usd_hourly_data()
	print(data.head())
	print(f"Saved {len(data)} rows to btc_usd_2y_1h_data.csv")
