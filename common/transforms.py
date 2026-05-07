from __future__ import annotations

import numpy as np
import pandas as pd


def funding_to_daily_annualized(df: pd.DataFrame, time_col: str = "fundingTime", rate_col: str = "fundingRate") -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out[time_col], utc=True).dt.date
    daily = out.groupby(["symbol", "date"], as_index=False)[rate_col].sum()
    daily["funding_daily"] = daily[rate_col]
    daily["funding_ann"] = 365.0 * daily["funding_daily"]
    return daily.drop(columns=[rate_col])


def log_return(series: pd.Series) -> pd.Series:
    return np.log(series).diff()


def build_spread(eth: pd.DataFrame, btc: pd.DataFrame, value_col: str, out_col: str = "spread") -> pd.DataFrame:
    merged = eth[["date", value_col]].merge(
        btc[["date", value_col]], on="date", suffixes=("_eth", "_btc"), how="inner"
    )
    merged[out_col] = merged[f"{value_col}_eth"] - merged[f"{value_col}_btc"]
    return merged
