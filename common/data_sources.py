"""Data collection helpers for exchange and market data."""

from __future__ import annotations

import requests
import pandas as pd

BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BYBIT_FUNDING_URL = "https://api.bybit.com/v5/market/funding/history"


def fetch_binance_funding(symbol: str, start_time_ms: int | None = None, limit: int = 1000) -> pd.DataFrame:
    params = {"symbol": symbol, "limit": limit}
    if start_time_ms is not None:
        params["startTime"] = start_time_ms
    r = requests.get(BINANCE_FUNDING_URL, params=params, timeout=30)
    if r.status_code == 451:
        raise RuntimeError(
            "Binance API returned HTTP 451 (regional restriction). "
            "Use fetch_bybit_funding() or run collect script with --funding-source bybit."
        )
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df[["symbol", "fundingTime", "fundingRate"]]


def fetch_bybit_funding(symbol: str, start_time_ms: int | None = None, limit: int = 200) -> pd.DataFrame:
    """
    Bybit USDT perpetual funding history.
    symbol example: BTCUSDT, ETHUSDT
    """
    params = {"category": "linear", "symbol": symbol, "limit": limit}
    if start_time_ms is not None:
        params["startTime"] = start_time_ms
    r = requests.get(BYBIT_FUNDING_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("result", {}).get("list", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "fundingTime", "fundingRate"])
    df["fundingTime"] = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["symbol"] = symbol
    return df[["symbol", "fundingTime", "fundingRate"]].sort_values("fundingTime")


def fetch_binance_daily_klines(symbol: str, start_time_ms: int | None = None, limit: int = 1000) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": "1d", "limit": limit}
    if start_time_ms is not None:
        params["startTime"] = start_time_ms
    r = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
    r.raise_for_status()
    raw = r.json()
    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.date
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["date", "open", "high", "low", "close", "volume"]]
