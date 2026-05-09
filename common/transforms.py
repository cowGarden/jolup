from __future__ import annotations

import numpy as np
import pandas as pd


def _parse_timestamp_series(values: pd.Series) -> pd.Series:
    """Parse mixed timestamp formats into UTC-aware datetimes.

    Args:
        values: Timestamp-like series in either string format or epoch numbers.

    Returns:
        UTC-aware datetime series.

    Raises:
        ValueError: If any timestamp cannot be parsed.
    """
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().all():
            raise ValueError("No valid numeric timestamps found")

        unit = "ms" if numeric.dropna().abs().median() > 1e11 else "s"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        # `format='mixed'` handles rows with and without fractional seconds.
        parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
        if parsed.isna().any():
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.notna().any():
                unit = "ms" if numeric.dropna().abs().median() > 1e11 else "s"
                parsed_numeric = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
                parsed = parsed.fillna(parsed_numeric)

    if parsed.isna().any():
        bad_examples = values.loc[parsed.isna()].head(3).tolist()
        raise ValueError(f"Failed to parse timestamp values. examples={bad_examples}")

    return parsed


def funding_to_daily_annualized(df: pd.DataFrame, time_col: str = "fundingTime", rate_col: str = "fundingRate") -> pd.DataFrame:
    """Convert intraday funding rates to daily and annualized series.

    Args:
        df: Funding dataframe containing timestamp and rate columns.
        time_col: Timestamp column name.
        rate_col: Funding rate column name.

    Returns:
        Dataframe with columns: symbol, date, funding_daily, funding_ann.
        `date` is normalized `datetime64[ns]` (UTC timezone removed).

    Raises:
        KeyError: If required columns are missing.
        ValueError: If timestamp parsing fails.
    """
    if time_col not in df.columns:
        raise KeyError(f"Missing required time column: {time_col}")
    if rate_col not in df.columns:
        raise KeyError(f"Missing required rate column: {rate_col}")

    out = df.copy()
    out["date"] = _parse_timestamp_series(out[time_col]).dt.tz_convert(None).dt.normalize()
    daily = out.groupby(["symbol", "date"], as_index=False)[rate_col].sum()
    daily["funding_daily"] = daily[rate_col]
    daily["funding_ann"] = 365.0 * daily["funding_daily"]
    return daily.drop(columns=[rate_col])


def log_return(series: pd.Series) -> pd.Series:
    """Compute first-difference log return from a price series."""
    return np.log(series).diff()


def build_spread(eth: pd.DataFrame, btc: pd.DataFrame, value_col: str, out_col: str = "spread") -> pd.DataFrame:
    """Build ETH-BTC spread series from aligned daily values."""
    merged = eth[["date", value_col]].merge(
        btc[["date", value_col]], on="date", suffixes=("_eth", "_btc"), how="inner"
    )
    merged[out_col] = merged[f"{value_col}_eth"] - merged[f"{value_col}_btc"]
    return merged
