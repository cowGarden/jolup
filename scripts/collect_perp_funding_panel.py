#!/usr/bin/env python3
"""Build ETH-BTC perpetual funding spread panels with staking-yield controls.

This script is intentionally runnable as plain Python, in notebooks via `%run`,
or in Colab after installing the repository requirements. It collects paginated
Binance/Bybit funding histories, open interest, price/volume controls, optional
basis and LST-risk proxies, then merges them with a Lido staking-yield panel and
estimates the requested OLS-HAC model ladder.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

# Non-retryable HTTP failures (e.g., malformed/unsupported request ranges).
class HTTPNonRetryableError(RuntimeError):
    """Raised when retrying the same request is not useful."""

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
START_DATE = "2023-01-01"
END_DATE = datetime.now(timezone.utc).date().isoformat()
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
OUTPUT_DIR_RAW = ROOT / "data" / "raw"
OUTPUT_DIR_PROCESSED = ROOT / "data" / "processed"
ETH_YIELD_PRIORITY = [
    "wsteth_implied_yield_7d",
    "wsteth_implied_yield_30d",
    "eth_native_yield",
    "stake_yield",
    "wsteth_defillama_apy",
]
LIDO_YIELD_CANDIDATES = [
    OUTPUT_DIR_PROCESSED / "funding_yield_panel.csv",
    OUTPUT_DIR_PROCESSED / "eth_yield_panel.csv",
    OUTPUT_DIR_PROCESSED / "lido_staking_yield_daily.csv",
    OUTPUT_DIR_PROCESSED / "lido_eth_yield_panel.csv",
    OUTPUT_DIR_PROCESSED / "lido_yield_panel.csv",
    OUTPUT_DIR_PROCESSED / "lido_wsteth_share_rate.csv",
]

BINANCE_FAPI_BASE = "https://fapi.binance.com"
BINANCE_SPOT_BASE = "https://api.binance.com"
BYBIT_BASE = "https://api.bybit.com"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

BINANCE_FUNDING_URL = f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate"
BINANCE_OI_HIST_URL = f"{BINANCE_FAPI_BASE}/futures/data/openInterestHist"
BINANCE_KLINES_URL = f"{BINANCE_FAPI_BASE}/fapi/v1/klines"
BINANCE_MARK_PRICE_KLINES_URL = f"{BINANCE_FAPI_BASE}/fapi/v1/markPriceKlines"
BINANCE_INDEX_PRICE_KLINES_URL = f"{BINANCE_FAPI_BASE}/fapi/v1/indexPriceKlines"
BINANCE_TAKER_LONG_SHORT_URL = f"{BINANCE_FAPI_BASE}/futures/data/takerlongshortRatio"
BINANCE_GLOBAL_LONG_SHORT_URL = f"{BINANCE_FAPI_BASE}/futures/data/globalLongShortAccountRatio"
BYBIT_FUNDING_URL = f"{BYBIT_BASE}/v5/market/funding/history"
BYBIT_OI_URL = f"{BYBIT_BASE}/v5/market/open-interest"
BYBIT_KLINES_URL = f"{BYBIT_BASE}/v5/market/kline"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_utc(dt: str | date | datetime) -> datetime:
    if isinstance(dt, datetime):
        out = dt
    elif isinstance(dt, date):
        out = datetime(dt.year, dt.month, dt.day)
    else:
        out = datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def to_ms(dt: str | date | datetime) -> int:
    return int(_parse_utc(dt).timestamp() * 1000)


def from_ms(ms: int | float | str) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


def utc_date_from_ms(ms: int | float | str) -> date:
    return from_ms(ms).date()


def end_date_inclusive_ms(dt: str | date | datetime) -> int:
    d = _parse_utc(dt).date()
    return to_ms(datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(days=1)) - 1


def safe_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    max_retries: int = 5,
    sleep_sec: float = 0.2,
    bybit: bool = False,
) -> Any:
    """GET JSON with retry/backoff for 429/403/5xx and Bybit retCode errors."""
    params = params or {}
    last_error: Exception | None = None
    for attempt in range(max_retries):
        if attempt:
            time.sleep(sleep_sec * (2 ** attempt))
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code in {403, 429} or resp.status_code >= 500:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            if 400 <= resp.status_code < 500:
                raise HTTPNonRetryableError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()
            payload = resp.json()
            if bybit and int(payload.get("retCode", 0)) != 0:
                raise RuntimeError(f"Bybit retCode={payload.get('retCode')}: {payload.get('retMsg')}")
            time.sleep(sleep_sec)
            return payload
        except HTTPNonRetryableError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface final exchange/API error cleanly
            last_error = exc
    raise RuntimeError(f"GET failed after {max_retries} attempts: {url} params={params} error={last_error}")


def validate_no_duplicates(df: pd.DataFrame, keys: list[str], name: str = "dataframe") -> None:
    if df.empty or not set(keys).issubset(df.columns):
        return
    n_dup = int(df.duplicated(keys).sum())
    if n_dup:
        raise ValueError(f"{name} has {n_dup} duplicate rows on keys={keys}")


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path.relative_to(ROOT)} rows={len(df):,}")


def _empty(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _warn_rate_scale(df: pd.DataFrame, rate_col: str, label: str) -> None:
    if df.empty or rate_col not in df:
        return
    max_abs = pd.to_numeric(df[rate_col], errors="coerce").abs().max()
    if pd.notna(max_abs) and max_abs > 0.05:
        print(f"[WARN] {label} {rate_col} max abs={max_abs:.6g}; rate may be percent-scale. No automatic correction applied.")


# ---------------------------------------------------------------------------
# Funding collection
# ---------------------------------------------------------------------------
def fetch_binance_funding(symbol: str, start_date: str, end_date: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch full Binance USD-M funding history using forward startTime pagination.

    Binance returns at most 1000 observations. We advance startTime to the last
    returned fundingTime + 1 ms, which prevents duplicate timestamps and walks
    the whole requested sample instead of stopping after roughly 333 days.
    """
    start_ms, end_ms = to_ms(start_date), end_date_inclusive_ms(end_date)
    rows: list[dict[str, Any]] = []
    current = start_ms
    while current <= end_ms:
        params = {"symbol": symbol, "startTime": current, "endTime": end_ms, "limit": min(limit, 1000)}
        batch = safe_get_json(BINANCE_FUNDING_URL, params=params, sleep_sec=0.2)
        if not batch:
            break
        rows.extend(batch)
        last_time = max(int(x["fundingTime"]) for x in batch)
        next_start = last_time + 1
        if next_start <= current:
            break
        current = next_start
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty(["exchange", "symbol", "fundingTime", "fundingRate", "date", "source"])
    df["fundingTime"] = pd.to_numeric(df["fundingTime"], errors="coerce").astype("int64")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df[(df["fundingTime"] >= start_ms) & (df["fundingTime"] <= end_ms)]
    df = df.drop_duplicates(["symbol", "fundingTime"]).sort_values("fundingTime")
    df["date"] = df["fundingTime"].map(utc_date_from_ms).astype(str)
    df["exchange"] = "binance"
    df["source"] = BINANCE_FUNDING_URL
    out = df[["exchange", "symbol", "fundingTime", "fundingRate", "date", "source"]]
    _warn_rate_scale(out, "fundingRate", f"Binance {symbol} funding")
    validate_no_duplicates(out, ["exchange", "symbol", "fundingTime"], f"binance funding {symbol}")
    save_csv(out, OUTPUT_DIR_RAW / f"binance_funding_{symbol}.csv")
    return out


def fetch_binance_funding_all(symbols: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    return {s: fetch_binance_funding(s, start_date, end_date) for s in symbols}


def fetch_bybit_funding_backward(symbol: str, start_date: str, end_date: str, limit: int = 200) -> pd.DataFrame:
    """Fetch Bybit funding with safe backward endTime pagination.

    Bybit's funding endpoint is safest when moving backward from endTime. Each
    page returns up to 200 observations before endTime; the next request uses the
    oldest fundingRateTimestamp - 1 ms, then the final frame is trimmed to
    start_date and sorted ascending.
    """
    start_ms, end_ms = to_ms(start_date), end_date_inclusive_ms(end_date)
    rows: list[dict[str, Any]] = []
    current_end = end_ms
    while current_end >= start_ms:
        params = {"category": "linear", "symbol": symbol, "endTime": current_end, "limit": min(limit, 200)}
        payload = safe_get_json(BYBIT_FUNDING_URL, params=params, sleep_sec=0.25, bybit=True)
        batch = payload.get("result", {}).get("list", [])
        if not batch:
            break
        rows.extend(batch)
        timestamps = [int(x["fundingRateTimestamp"]) for x in batch]
        oldest = min(timestamps)
        if oldest < start_ms:
            break
        next_end = oldest - 1
        if next_end >= current_end:
            break
        current_end = next_end
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty(["exchange", "symbol", "fundingRateTimestamp", "fundingTime", "fundingRate", "date", "source"])
    df["fundingRateTimestamp"] = pd.to_numeric(df["fundingRateTimestamp"], errors="coerce").astype("int64")
    df["fundingTime"] = df["fundingRateTimestamp"]
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df[(df["fundingTime"] >= start_ms) & (df["fundingTime"] <= end_ms)]
    df = df.drop_duplicates(["symbol", "fundingTime"]).sort_values("fundingTime")
    df["date"] = df["fundingTime"].map(utc_date_from_ms).astype(str)
    df["exchange"] = "bybit"
    df["source"] = BYBIT_FUNDING_URL
    out = df[["exchange", "symbol", "fundingRateTimestamp", "fundingTime", "fundingRate", "date", "source"]]
    _warn_rate_scale(out, "fundingRate", f"Bybit {symbol} funding")
    validate_no_duplicates(out, ["exchange", "symbol", "fundingTime"], f"bybit funding {symbol}")
    if len(out) <= limit and (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days > 100:
        print(f"[WARN] Bybit {symbol} funding returned only {len(out)} rows; check API coverage/pagination.")
    save_csv(out, OUTPUT_DIR_RAW / f"bybit_funding_{symbol}.csv")
    return out


def fetch_bybit_funding_all(symbols: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    return {s: fetch_bybit_funding_backward(s, start_date, end_date) for s in symbols}


def funding_to_daily_annualized(
    df: pd.DataFrame,
    time_col: str = "fundingTime",
    rate_col: str = "fundingRate",
    exchange: str | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return _empty(["date", "exchange", "symbol", "funding_daily", "funding_ann", "funding_ann_pct", "n_funding_obs"])
    out = df.copy()
    out["date"] = pd.to_datetime(out[time_col].astype("int64"), unit="ms", utc=True).dt.date.astype(str)
    out[rate_col] = pd.to_numeric(out[rate_col], errors="coerce")
    daily = (
        out.groupby("date", as_index=False)
        .agg(funding_daily=(rate_col, "sum"), n_funding_obs=(rate_col, "count"))
        .sort_values("date")
    )
    daily["funding_ann"] = daily["funding_daily"] * 365.0
    daily["funding_ann_pct"] = daily["funding_ann"] * 100.0
    daily["exchange"] = exchange or str(out["exchange"].iloc[0])
    daily["symbol"] = symbol or str(out["symbol"].iloc[0])
    cols = ["date", "exchange", "symbol", "funding_daily", "funding_ann", "funding_ann_pct", "n_funding_obs"]
    daily = daily[cols]
    save_csv(daily, OUTPUT_DIR_PROCESSED / f"{daily['exchange'].iloc[0]}_{daily['symbol'].iloc[0]}_funding_daily.csv")
    return daily


def build_eth_btc_spread(eth_daily: pd.DataFrame, btc_daily: pd.DataFrame, exchange: str) -> pd.DataFrame:
    merged = eth_daily.merge(btc_daily, on="date", suffixes=("_eth", "_btc"), how="inner")
    merged["eth_funding_ann"] = merged["funding_ann_eth"]
    merged["btc_funding_ann"] = merged["funding_ann_btc"]
    merged["spread"] = merged["eth_funding_ann"] - merged["btc_funding_ann"]
    merged["eth_n_funding_obs"] = merged["n_funding_obs_eth"]
    merged["btc_n_funding_obs"] = merged["n_funding_obs_btc"]
    merged["exchange"] = exchange
    out = merged[["date", "eth_funding_ann", "btc_funding_ann", "spread", "eth_n_funding_obs", "btc_n_funding_obs", "exchange"]]
    save_csv(out, OUTPUT_DIR_PROCESSED / f"{exchange}_eth_btc_funding_spread_daily.csv")
    return out


# ---------------------------------------------------------------------------
# OI, price, volume, basis, leverage-demand proxies
# ---------------------------------------------------------------------------
def fetch_bybit_open_interest(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    start_ms, end_ms = to_ms(start_date), end_date_inclusive_ms(end_date)
    window_days = 190 if interval == "1d" else 7
    rows: list[dict[str, Any]] = []
    window_start = _parse_utc(start_date)
    final_dt = _parse_utc(end_date) + timedelta(days=1) - timedelta(milliseconds=1)
    while window_start <= final_dt:
        window_end = min(window_start + timedelta(days=window_days) - timedelta(milliseconds=1), final_dt)
        cursor = None
        while True:
            params = {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": interval,
                "startTime": to_ms(window_start),
                "endTime": to_ms(window_end),
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            payload = safe_get_json(BYBIT_OI_URL, params=params, sleep_sec=0.25, bybit=True)
            result = payload.get("result", {})
            rows.extend(result.get("list", []))
            cursor = result.get("nextPageCursor")
            if not cursor:
                break
        window_start = window_end + timedelta(milliseconds=1)
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty(["exchange", "symbol", "timestamp", "date", "open_interest_raw", "source"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    df["open_interest_raw"] = pd.to_numeric(df["openInterest"], errors="coerce")
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]
    df = df.drop_duplicates(["timestamp"]).sort_values("timestamp")
    df["date"] = df["timestamp"].map(utc_date_from_ms).astype(str)
    df["exchange"] = "bybit"
    df["symbol"] = symbol
    df["source"] = BYBIT_OI_URL
    out = df[["exchange", "symbol", "timestamp", "date", "open_interest_raw", "source"]]
    save_csv(out, OUTPUT_DIR_RAW / f"bybit_oi_{symbol}.csv")
    return out


def _binance_futures_data_start_floor(max_days: int = 30) -> datetime:
    """Return the oldest timestamp accepted by Binance futures/data metrics endpoints.

    Binance documents openInterestHist and long/short metric endpoints as recent
    metrics feeds. In practice, requests with startTime older than the latest
    30 days can fail with "startTime is invalid", so callers clamp historical
    requests before sending them to Binance and use another OI source when a
    longer research sample is required.
    """
    now_utc = datetime.now(timezone.utc)
    return now_utc - timedelta(days=max_days) + timedelta(milliseconds=1)


def fetch_binance_open_interest_hist(symbol: str, start_date: str, end_date: str, period: str = "1d") -> pd.DataFrame:
    # Binance futures/data openInterestHist is a recent metrics endpoint: the
    # official connector notes that only the latest 30 days are available. Clamp
    # before requesting to avoid deterministic "startTime is invalid" 400s.
    requested_start = _parse_utc(start_date)
    end_dt = _parse_utc(end_date)
    start_dt = max(requested_start, _binance_futures_data_start_floor(max_days=30))
    if start_dt > end_dt:
        print(f"[WARN] Binance openInterestHist requested range is outside the latest 30 days for {symbol}.")
        return _empty(["exchange", "symbol", "timestamp", "date", "open_interest_raw", "open_interest_usd", "source"])
    if start_dt > requested_start:
        print(
            f"[WARN] Binance openInterestHist only exposes recent history; "
            f"clamped {symbol} OI start from {requested_start.date()} to {start_dt.date()}."
        )

    params = {
        "symbol": symbol,
        "period": period,
        "startTime": to_ms(start_dt),
        "endTime": end_date_inclusive_ms(end_dt),
        "limit": 500,
    }
    try:
        batch = safe_get_json(BINANCE_OI_HIST_URL, params=params, sleep_sec=0.2)
    except HTTPNonRetryableError as exc:
        print(f"[WARN] Binance openInterestHist rejected bounded request for {symbol}: {exc}")
        print(f"[WARN] Retrying Binance openInterestHist for {symbol} without startTime/endTime (recent-only).")
        try:
            batch = safe_get_json(BINANCE_OI_HIST_URL, params={"symbol": symbol, "period": period, "limit": 500}, sleep_sec=0.2)
        except Exception as fallback_exc:  # noqa: BLE001
            print(f"[WARN] Binance openInterestHist recent-only fallback failed for {symbol}: {fallback_exc}")
            return _empty(["exchange", "symbol", "timestamp", "date", "open_interest_raw", "open_interest_usd", "source"])
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Binance openInterestHist failed for {symbol}: {exc}")
        return _empty(["exchange", "symbol", "timestamp", "date", "open_interest_raw", "open_interest_usd", "source"])

    df = pd.DataFrame(batch if isinstance(batch, list) else [])
    if df.empty:
        return _empty(["exchange", "symbol", "timestamp", "date", "open_interest_raw", "open_interest_usd", "source"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    df["open_interest_raw"] = pd.to_numeric(df.get("sumOpenInterest"), errors="coerce")
    df["open_interest_usd"] = pd.to_numeric(df.get("sumOpenInterestValue"), errors="coerce")
    df = df[(df["timestamp"] >= to_ms(start_dt)) & (df["timestamp"] <= end_date_inclusive_ms(end_dt))]
    df = df.drop_duplicates(["timestamp"]).sort_values("timestamp")
    df["date"] = df["timestamp"].map(utc_date_from_ms).astype(str)
    df["exchange"] = "binance"
    df["symbol"] = symbol
    df["source"] = BINANCE_OI_HIST_URL
    out = df[["exchange", "symbol", "timestamp", "date", "open_interest_raw", "open_interest_usd", "source"]]
    save_csv(out, OUTPUT_DIR_RAW / f"binance_oi_{symbol}.csv")
    return out


def fetch_binance_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    url: str = BINANCE_KLINES_URL,
    symbol_param: str = "symbol",
) -> pd.DataFrame:
    start_ms, end_ms = to_ms(start_date), end_date_inclusive_ms(end_date)
    current = start_ms
    rows: list[list[Any]] = []
    while current <= end_ms:
        params = {symbol_param: symbol, "interval": interval, "startTime": current, "endTime": end_ms, "limit": 1500}
        batch = safe_get_json(url, params=params, sleep_sec=0.2)
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_start = last_open + 1
        if next_start <= current:
            break
        current = next_start
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return _empty(["exchange", "symbol", "interval", "open_time", "date", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote", "source"])
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["open_time"] >= start_ms) & (df["open_time"] <= end_ms)]
    df = df.drop_duplicates(["open_time"]).sort_values("open_time")
    df["date"] = df["open_time"].map(utc_date_from_ms).astype(str)
    df["exchange"] = "binance"
    df["symbol"] = symbol
    df["interval"] = interval
    df["source"] = url
    out = df[["exchange", "symbol", "interval", "open_time", "date", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote", "source"]]
    if url == BINANCE_MARK_PRICE_KLINES_URL:
        filename = f"binance_mark_klines_{symbol}_{interval}.csv"
    elif url == BINANCE_INDEX_PRICE_KLINES_URL:
        filename = f"binance_index_klines_{symbol}_{interval}.csv"
    else:
        filename = f"binance_klines_{symbol}_{interval}.csv"
    save_csv(out, OUTPUT_DIR_RAW / filename)
    return out


def fetch_bybit_klines(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    bybit_interval = {"1d": "D", "1h": "60"}.get(interval, interval)
    start_ms, end_ms = to_ms(start_date), end_date_inclusive_ms(end_date)
    current_end = end_ms
    rows: list[list[Any]] = []
    while current_end >= start_ms:
        params = {"category": "linear", "symbol": symbol, "interval": bybit_interval, "end": current_end, "limit": 1000}
        payload = safe_get_json(BYBIT_KLINES_URL, params=params, sleep_sec=0.25, bybit=True)
        batch = payload.get("result", {}).get("list", [])
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(x[0]) for x in batch)
        if oldest < start_ms:
            break
        current_end = oldest - 1
    cols = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return _empty(["exchange", "symbol", "interval", "open_time", "date", "open", "high", "low", "close", "volume", "quote_volume", "source"])
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["open_time"] >= start_ms) & (df["open_time"] <= end_ms)]
    df = df.drop_duplicates(["open_time"]).sort_values("open_time")
    df["date"] = df["open_time"].map(utc_date_from_ms).astype(str)
    df["exchange"] = "bybit"
    df["symbol"] = symbol
    df["interval"] = interval
    df["source"] = BYBIT_KLINES_URL
    out = df[["exchange", "symbol", "interval", "open_time", "date", "open", "high", "low", "close", "volume", "quote_volume", "source"]]
    save_csv(out, OUTPUT_DIR_RAW / f"bybit_klines_{symbol}_{interval}.csv")
    return out


def build_price_features(daily: dict[str, pd.DataFrame], hourly: dict[str, pd.DataFrame] | None, exchange: str) -> pd.DataFrame:
    btc = daily["BTCUSDT"].rename(columns={"close": "btc_close", "quote_volume": "btc_usd_volume"})
    eth = daily["ETHUSDT"].rename(columns={"close": "eth_close", "quote_volume": "eth_usd_volume"})
    out = eth[["date", "eth_close", "eth_usd_volume"]].merge(btc[["date", "btc_close", "btc_usd_volume"]], on="date", how="inner")
    out = out.sort_values("date")
    out["ret_eth"] = np.log(out["eth_close"]).diff()
    out["ret_btc"] = np.log(out["btc_close"]).diff()
    out["ret_eth_btc"] = out["ret_eth"] - out["ret_btc"]
    out["momentum_eth_btc_7d"] = out["ret_eth_btc"].rolling(7).sum()
    out["momentum_eth_btc_30d"] = out["ret_eth_btc"].rolling(30).sum()
    out["volume_ratio"] = np.log(out["eth_usd_volume"] / out["btc_usd_volume"])
    out["dlog_volume_eth"] = np.log(out["eth_usd_volume"]).diff()
    out["dlog_volume_btc"] = np.log(out["btc_usd_volume"]).diff()
    out["dlog_volume_eth_btc"] = out["dlog_volume_eth"] - out["dlog_volume_btc"]
    if hourly:
        rv_parts = []
        for symbol, prefix in [("ETHUSDT", "eth"), ("BTCUSDT", "btc")]:
            h = hourly[symbol].copy().sort_values("open_time")
            h["hourly_ret"] = np.log(h["close"]).diff()
            rv = h.groupby("date", as_index=False)["hourly_ret"].apply(lambda s: float(np.nansum(np.square(s))))
            rv = rv.rename(columns={"hourly_ret": f"rv_{prefix}"})
            rv_parts.append(rv)
        out = out.merge(rv_parts[0], on="date", how="left").merge(rv_parts[1], on="date", how="left")
        out["rv_diff"] = out["rv_eth"] - out["rv_btc"]
        out["rv_log_ratio"] = np.log(out["rv_eth"].replace(0, np.nan)) - np.log(out["rv_btc"].replace(0, np.nan))
    else:
        out["rv_eth"] = np.nan
        out["rv_btc"] = np.nan
        out["rv_diff"] = np.nan
        out["rv_log_ratio"] = np.nan
    out["exchange"] = exchange
    save_csv(out, OUTPUT_DIR_PROCESSED / f"{exchange}_price_features_daily.csv")
    return out


def _first_non_null(values: pd.Series) -> str | None:
    non_null = values.dropna()
    return str(non_null.iloc[0]) if not non_null.empty else None


def build_oi_features(eth_oi: pd.DataFrame, btc_oi: pd.DataFrame, price_features: pd.DataFrame, exchange: str) -> pd.DataFrame:
    eth = eth_oi.rename(columns={"open_interest_raw": "oi_eth_raw", "open_interest_usd": "oi_eth_usd"})
    btc = btc_oi.rename(columns={"open_interest_raw": "oi_btc_raw", "open_interest_usd": "oi_btc_usd"})
    keep_eth = [c for c in ["date", "oi_eth_raw", "oi_eth_usd"] if c in eth.columns]
    keep_btc = [c for c in ["date", "oi_btc_raw", "oi_btc_usd"] if c in btc.columns]
    out = eth[keep_eth].merge(btc[keep_btc], on="date", how="inner").sort_values("date")
    if out.empty:
        out = _empty(["date", "oi_eth_raw", "oi_btc_raw", "oi_eth_usd", "oi_btc_usd", "dlog_oi_eth", "dlog_oi_btc", "oi_eth_btc", "oi_ratio", "exchange", "oi_source_exchange"])
        save_csv(out, OUTPUT_DIR_PROCESSED / f"{exchange}_oi_features_daily.csv")
        return out
    prices = price_features[["date", "eth_close", "btc_close"]]
    out = out.merge(prices, on="date", how="left")
    if "oi_eth_usd" not in out or out["oi_eth_usd"].isna().all():
        out["oi_eth_usd"] = out["oi_eth_raw"] * out["eth_close"]
    if "oi_btc_usd" not in out or out["oi_btc_usd"].isna().all():
        out["oi_btc_usd"] = out["oi_btc_raw"] * out["btc_close"]
    out["dlog_oi_eth"] = np.log(out["oi_eth_usd"]).diff()
    out["dlog_oi_btc"] = np.log(out["oi_btc_usd"]).diff()
    out["oi_eth_btc"] = out["dlog_oi_eth"] - out["dlog_oi_btc"]
    out["oi_ratio"] = np.log(out["oi_eth_usd"] / out["oi_btc_usd"])
    out["exchange"] = exchange
    source_exchange = _first_non_null(eth_oi.get("exchange", pd.Series(dtype=object))) or _first_non_null(btc_oi.get("exchange", pd.Series(dtype=object)))
    out["oi_source_exchange"] = source_exchange or exchange
    save_csv(out, OUTPUT_DIR_PROCESSED / f"{exchange}_oi_features_daily.csv")
    return out


def fetch_binance_ratio_series(symbol: str, start_date: str, end_date: str, url: str, period: str = "1d") -> pd.DataFrame:
    def _recent_only() -> pd.DataFrame:
        """Fallback for endpoints that reject historical start/end parameters."""
        params = {"symbol": symbol, "period": period, "limit": 500}
        try:
            batch = safe_get_json(url, params=params, sleep_sec=0.2)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Binance ratio endpoint recent-only fallback failed for {symbol}: {exc}")
            return _empty(["date", "symbol"])
        df_recent = pd.DataFrame(batch)
        if df_recent.empty:
            return _empty(["date", "symbol"])
        df_recent["timestamp"] = pd.to_numeric(df_recent["timestamp"], errors="coerce").astype("int64")
        df_recent["date"] = df_recent["timestamp"].map(utc_date_from_ms).astype(str)
        df_recent["symbol"] = symbol
        print(f"[WARN] Binance ratio endpoint uses recent-only fallback for {symbol}; historical windowing unavailable.")
        return df_recent.drop_duplicates(["date"]).sort_values("date")

    rows: list[dict[str, Any]] = []
    requested_start = _parse_utc(start_date)
    end_dt = _parse_utc(end_date)
    start_dt = max(requested_start, _binance_futures_data_start_floor(max_days=30))
    if start_dt > requested_start:
        print(
            f"[WARN] Binance ratio endpoint only exposes recent history; "
            f"clamped {symbol} start from {requested_start.date()} to {start_dt.date()}."
        )
    window_days = 30 if period == "1d" else 7
    current_dt = start_dt
    while current_dt <= end_dt:
        window_end = min(current_dt + timedelta(days=window_days) - timedelta(milliseconds=1), end_dt)
        current = to_ms(current_dt)
        end_ms = to_ms(window_end)
        while current <= end_ms:
            params = {"symbol": symbol, "period": period, "startTime": current, "endTime": end_ms, "limit": 500}
            try:
                batch = safe_get_json(url, params=params, sleep_sec=0.2)
            except HTTPNonRetryableError as exc:
                print(f"[WARN] Binance ratio endpoint non-retryable for {symbol} window {current_dt.date()}~{window_end.date()}: {exc}")
                msg = str(exc).lower()
                if "starttime" in msg or "endtime" in msg:
                    return _recent_only()
                # These endpoints often hard-fail for unsupported historical ranges.
                # Stop early to avoid noisy repeated warnings.
                current_dt = end_dt + timedelta(days=1)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Binance ratio endpoint failed for {symbol} window {current_dt.date()}~{window_end.date()}: {exc}")
                break
            if not batch:
                break
            rows.extend(batch)
            last = max(int(x["timestamp"]) for x in batch)
            next_start = last + 1
            if next_start <= current:
                break
            current = next_start
        current_dt = window_end + timedelta(milliseconds=1)
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty(["date", "symbol"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    df["date"] = df["timestamp"].map(utc_date_from_ms).astype(str)
    df["symbol"] = symbol
    return df.drop_duplicates(["date"]).sort_values("date")


def build_optional_leverage_controls(start_date: str, end_date: str) -> pd.DataFrame:
    """Optional Binance taker imbalance and long-short ratio controls."""
    try:
        taker = {s: fetch_binance_ratio_series(s, start_date, end_date, BINANCE_TAKER_LONG_SHORT_URL) for s in SYMBOLS}
        ls = {s: fetch_binance_ratio_series(s, start_date, end_date, BINANCE_GLOBAL_LONG_SHORT_URL) for s in SYMBOLS}
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Optional taker/long-short controls unavailable: {exc}")
        return _empty(["date", "eth_btc_taker_imbalance", "ls_ratio_eth_btc"])
    parts = []
    for symbol, prefix in [("ETHUSDT", "eth"), ("BTCUSDT", "btc")]:
        t = taker[symbol].copy()
        if not t.empty and {"buyVol", "sellVol"}.issubset(t.columns):
            t[f"taker_imbalance_{prefix}"] = (pd.to_numeric(t["buyVol"], errors="coerce") - pd.to_numeric(t["sellVol"], errors="coerce")) / (
                pd.to_numeric(t["buyVol"], errors="coerce") + pd.to_numeric(t["sellVol"], errors="coerce")
            )
            parts.append(t[["date", f"taker_imbalance_{prefix}"]])
        l = ls[symbol].copy()
        if not l.empty and "longShortRatio" in l:
            l[f"long_short_ratio_{prefix}"] = pd.to_numeric(l["longShortRatio"], errors="coerce")
            parts.append(l[["date", f"long_short_ratio_{prefix}"]])
    if not parts:
        return _empty(["date", "eth_btc_taker_imbalance", "ls_ratio_eth_btc"])
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="date", how="outer")
    if {"taker_imbalance_eth", "taker_imbalance_btc"}.issubset(out.columns):
        out["eth_btc_taker_imbalance"] = out["taker_imbalance_eth"] - out["taker_imbalance_btc"]
    if {"long_short_ratio_eth", "long_short_ratio_btc"}.issubset(out.columns):
        out["ls_ratio_eth_btc"] = np.log(out["long_short_ratio_eth"]) - np.log(out["long_short_ratio_btc"])
    save_csv(out, OUTPUT_DIR_PROCESSED / "binance_optional_leverage_controls_daily.csv")
    return out


def build_binance_basis_features(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        mark = {s: fetch_binance_klines(s, "1d", start_date, end_date, BINANCE_MARK_PRICE_KLINES_URL) for s in SYMBOLS}
        index_symbol = {"BTCUSDT": "BTCUSDT", "ETHUSDT": "ETHUSDT"}
        index = {
            s: fetch_binance_klines(
                index_symbol[s],
                "1d",
                start_date,
                end_date,
                BINANCE_INDEX_PRICE_KLINES_URL,
                symbol_param="pair",
            )
            for s in SYMBOLS
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Basis controls unavailable: {exc}")
        return _empty(["date", "basis_eth", "basis_btc", "basis_spread"])
    features = None
    for symbol, prefix in [("ETHUSDT", "eth"), ("BTCUSDT", "btc")]:
        m = mark[symbol][["date", "close"]].rename(columns={"close": f"mark_{prefix}"})
        i = index[symbol][["date", "close"]].rename(columns={"close": f"index_{prefix}"})
        one = m.merge(i, on="date", how="inner")
        one[f"basis_{prefix}"] = (one[f"mark_{prefix}"] - one[f"index_{prefix}"]) / one[f"index_{prefix}"]
        features = one if features is None else features.merge(one, on="date", how="inner")
    features["basis_spread"] = features["basis_eth"] - features["basis_btc"]
    save_csv(features, OUTPUT_DIR_PROCESSED / "binance_basis_features_daily.csv")
    return features


def fetch_steth_discount(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch stETH/ETH market ratio from CoinGecko as an LST depeg-risk proxy."""
    from_ts = int(to_ms(start_date) / 1000)
    to_ts = int(end_date_inclusive_ms(end_date) / 1000)
    try:
        params = {"vs_currency": "eth", "from": from_ts, "to": to_ts}
        payload = safe_get_json(f"{COINGECKO_BASE}/coins/staked-ether/market_chart/range", params=params, sleep_sec=1.5)
        prices = payload.get("prices", [])
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        # Public CoinGecko API allows only recent history windows.
        if "10012" in msg or "past 365 days" in msg.lower():
            fallback_from = max(from_ts, to_ts - 365 * 24 * 60 * 60)
            print("[WARN] stETH discount full history unavailable on public API; falling back to last 365 days.")
            try:
                params = {"vs_currency": "eth", "from": fallback_from, "to": to_ts}
                payload = safe_get_json(f"{COINGECKO_BASE}/coins/staked-ether/market_chart/range", params=params, sleep_sec=1.5)
                prices = payload.get("prices", [])
            except Exception as fallback_exc:  # noqa: BLE001
                print(f"[WARN] stETH discount fallback failed: {fallback_exc}")
                return _empty(["date", "steth_price_eth", "steth_discount"])
        else:
            print(f"[WARN] stETH discount unavailable: {exc}")
            return _empty(["date", "steth_price_eth", "steth_discount"])
    df = pd.DataFrame(prices, columns=["timestamp", "steth_price_eth"])
    if df.empty:
        return _empty(["date", "steth_price_eth", "steth_discount"])
    df["date"] = df["timestamp"].map(utc_date_from_ms).astype(str)
    out = df.groupby("date", as_index=False)["steth_price_eth"].last()
    out["steth_discount"] = out["steth_price_eth"] - 1.0
    save_csv(out, OUTPUT_DIR_PROCESSED / "steth_discount_daily.csv")
    return out


def add_event_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out["date"])
    btc_etf = pd.Timestamp("2024-01-11")
    eth_etf = pd.Timestamp("2024-07-23")
    out["post_btc_etf"] = (d >= btc_etf).astype(int)
    out["post_eth_etf"] = (d >= eth_etf).astype(int)
    out["btc_etf_event_window_7d"] = ((d >= btc_etf - pd.Timedelta(days=7)) & (d <= btc_etf + pd.Timedelta(days=7))).astype(int)
    out["eth_etf_event_window_7d"] = ((d >= eth_etf - pd.Timedelta(days=7)) & (d <= eth_etf + pd.Timedelta(days=7))).astype(int)
    return out


# ---------------------------------------------------------------------------
# Master panel, validation, and OLS-HAC model ladder
# ---------------------------------------------------------------------------
def load_lido_yield(path: str | Path | None = None) -> pd.DataFrame:
    candidates = [Path(path)] if path else LIDO_YIELD_CANDIDATES
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if "stake_yield" not in df.columns and "annualized_apr_decimal" in df.columns:
                df["stake_yield"] = df["annualized_apr_decimal"]
            selected = next((col for col in ETH_YIELD_PRIORITY if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()), None)
            if selected is None or "date" not in df.columns:
                raise ValueError(f"{p} must contain date plus one ETH yield candidate: {ETH_YIELD_PRIORITY}")
            if selected == "wsteth_defillama_apy":
                print("[WARN] Using wsteth_defillama_apy only as robustness fallback; it is not contract implied yield.")
            out = df[["date", selected]].copy().rename(columns={selected: "stake_yield"})
            out["stake_yield_source"] = selected
            out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
            out["stake_yield"] = pd.to_numeric(out["stake_yield"], errors="coerce")
            _warn_rate_scale(out, "stake_yield", f"ETH yield ({selected})")
            return out[["date", "stake_yield", "stake_yield_source"]].drop_duplicates("date").sort_values("date")
    print("[WARN] No Lido staking yield CSV found; master datasets will omit stake_yield until one is created.")
    return _empty(["date", "stake_yield"])


def build_master_dataset(
    exchange: str,
    spread: pd.DataFrame,
    price_features: pd.DataFrame,
    oi_features: pd.DataFrame,
    lido_yield: pd.DataFrame,
    optional_controls: pd.DataFrame | None = None,
    basis: pd.DataFrame | None = None,
    steth_discount: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = spread.merge(price_features, on="date", how="left", suffixes=("", "_price"))
    out = out.merge(oi_features, on="date", how="left", suffixes=("", "_oi"))
    if not lido_yield.empty:
        out = out.merge(lido_yield, on="date", how="left")
    if optional_controls is not None and not optional_controls.empty:
        out = out.merge(optional_controls, on="date", how="left")
    if basis is not None and not basis.empty:
        out = out.merge(basis[["date", "basis_eth", "basis_btc", "basis_spread"]], on="date", how="left")
    if steth_discount is not None and not steth_discount.empty:
        out = out.merge(steth_discount[["date", "steth_price_eth", "steth_discount"]], on="date", how="left")
    out = add_event_dummies(out).sort_values("date")
    out["spread_lag1"] = out["spread"].shift(1)
    out["spread_lag7"] = out["spread"].shift(7)
    out["funding_eth_lag1"] = out["eth_funding_ann"].shift(1)
    out["funding_btc_lag1"] = out["btc_funding_ann"].shift(1)
    out["exchange"] = exchange
    save_csv(out, OUTPUT_DIR_PROCESSED / f"master_{exchange}_eth_btc_funding_staking_daily.csv")
    return out


def summarize_coverage(name: str, df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "date" not in df.columns:
        return {"dataset": name, "min_date": None, "max_date": None, "n_rows": len(df), "missing_dates": None}
    dates = pd.to_datetime(df["date"])
    full = pd.date_range(dates.min(), dates.max(), freq="D")
    missing = int(len(full.difference(pd.DatetimeIndex(dates.drop_duplicates()))))
    row = {"dataset": name, "min_date": dates.min().date().isoformat(), "max_date": dates.max().date().isoformat(), "n_rows": len(df), "missing_dates": missing}
    print(f"[COVERAGE] {name}: {row}")
    return row


def validation_summary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [summarize_coverage(name, df) for name, df in datasets.items()]
    report = pd.DataFrame(rows)
    save_csv(report, OUTPUT_DIR_PROCESSED / "data_coverage_report.csv")
    for name, df in datasets.items():
        if df.empty:
            continue
        if {"eth_n_funding_obs", "btc_n_funding_obs"}.issubset(df.columns):
            print(f"\n[VALIDATION] Funding observations per day for {name}")
            print(df[["eth_n_funding_obs", "btc_n_funding_obs"]].describe())
            abnormal = df[(df["eth_n_funding_obs"] != df["eth_n_funding_obs"].mode().iloc[0]) | (df["btc_n_funding_obs"] != df["btc_n_funding_obs"].mode().iloc[0])]
            print(f"[VALIDATION] abnormal funding-observation days={len(abnormal)}")
        desc_cols = [c for c in ["funding_ann", "spread", "stake_yield", "oi_eth_btc", "oi_ratio", "rv_log_ratio"] if c in df]
        if desc_cols:
            print(f"\n[DESCRIBE] {name}")
            print(df[desc_cols].describe())
        if {"spread", "eth_funding_ann", "btc_funding_ann"}.issubset(df.columns):
            err = (df["spread"] - (df["eth_funding_ann"] - df["btc_funding_ann"])).abs().max()
            print(f"[VALIDATION] {name} max spread identity error={err}")
    return report


def ols_hac(y: pd.Series, X: pd.DataFrame, maxlags: int = 7):
    data = pd.concat([y.rename("y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) <= X.shape[1] + 5:
        raise ValueError(f"Insufficient observations for OLS-HAC: n={len(data)}, k={X.shape[1]}")
    X_ = sm.add_constant(data[X.columns], has_constant="add")
    return sm.OLS(data["y"], X_).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def _missing_counts_for_model(master: pd.DataFrame, required_cols: list[str]) -> str:
    counts = []
    for col in required_cols:
        if col in master.columns:
            n_missing = int(master[col].replace([np.inf, -np.inf], np.nan).isna().sum())
            counts.append(f"{col}:{n_missing}")
        else:
            counts.append(f"{col}:missing_column")
    return ";".join(counts)


def run_model_ladder(master: pd.DataFrame, exchange: str, maxlags: int = 7) -> tuple[pd.DataFrame, dict[str, Any]]:
    models = {
        "M1": ["stake_yield"],
        "M2": ["stake_yield", "ret_eth_btc", "rv_log_ratio"],
        "M3": ["stake_yield", "ret_eth_btc", "rv_log_ratio", "oi_eth_btc", "oi_ratio"],
        "M4": ["stake_yield", "ret_eth_btc", "rv_log_ratio", "oi_eth_btc", "oi_ratio", "volume_ratio"],
        "M5": ["stake_yield", "ret_eth_btc", "rv_log_ratio", "oi_eth_btc", "oi_ratio", "volume_ratio", "spread_lag1"],
        "M6": ["stake_yield", "ret_eth_btc", "rv_log_ratio", "oi_eth_btc", "oi_ratio", "volume_ratio", "spread_lag1", "basis_spread"],
        "M7": ["stake_yield", "ret_eth_btc", "rv_log_ratio", "oi_eth_btc", "oi_ratio", "volume_ratio", "spread_lag1", "steth_discount"],
        "M8": ["stake_yield", "ret_eth_btc", "rv_log_ratio", "oi_eth_btc", "oi_ratio", "volume_ratio", "spread_lag1", "post_btc_etf", "post_eth_etf", "btc_etf_event_window_7d", "eth_etf_event_window_7d"],
    }
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    for name, cols in models.items():
        required = ["spread", *cols]
        missing = [c for c in required if c not in master.columns]
        min_n = len(cols) + 6
        complete_rows = 0
        if not missing:
            complete_rows = len(master[required].replace([np.inf, -np.inf], np.nan).dropna())
        availability = {
            "exchange": exchange,
            "model": name,
            "status": "pending",
            "complete_rows": complete_rows,
            "min_required_rows": min_n,
            "missing_columns": ",".join(missing),
            "missing_value_counts": _missing_counts_for_model(master, required),
            "controls": ",".join(cols),
            "reason": "",
        }
        if missing:
            availability["status"] = "skipped"
            availability["reason"] = f"missing columns: {missing}"
            availability_rows.append(availability)
            print(f"[WARN] {exchange} {name} skipped; missing columns={missing}")
            continue
        if complete_rows < min_n:
            availability["status"] = "skipped"
            availability["reason"] = f"insufficient complete rows n={complete_rows} (<{min_n})"
            availability_rows.append(availability)
            print(f"[WARN] {exchange} {name} skipped; insufficient complete rows n={complete_rows} (<{min_n})")
            continue
        try:
            res = ols_hac(master["spread"], master[cols], maxlags=maxlags)
        except Exception as exc:  # noqa: BLE001
            availability["status"] = "failed"
            availability["reason"] = str(exc)
            availability_rows.append(availability)
            print(f"[WARN] {exchange} {name} failed: {exc}")
            continue
        results[name] = res
        coef = float(res.params.get("stake_yield", np.nan))
        pval = float(res.pvalues.get("stake_yield", np.nan))
        rows.append({
            "exchange": exchange,
            "model": name,
            "nobs": int(res.nobs),
            "r2": float(res.rsquared),
            "stake_yield_coef": coef,
            "stake_yield_pvalue": pval,
            "stake_yield_sign": "positive" if coef > 0 else "negative" if coef < 0 else "zero",
            "stake_yield_sig_10pct": pval < 0.10,
            "stake_yield_sig_5pct": pval < 0.05,
            "controls": ",".join(cols),
        })
        availability["status"] = "estimated"
        availability["complete_rows"] = int(res.nobs)
        availability_rows.append(availability)
        print(f"[MODEL] {exchange} {name}: stake_yield coef={coef:.6g}, p={pval:.4g}, n={int(res.nobs)}")
    availability_report = pd.DataFrame(availability_rows)
    save_csv(availability_report, OUTPUT_DIR_PROCESSED / f"model_availability_report_{exchange}.csv")
    summary = pd.DataFrame(rows)
    save_csv(summary, OUTPUT_DIR_PROCESSED / f"ols_hac_model_summary_{exchange}.csv")
    return summary, results


def _empty_oi_raw() -> dict[str, pd.DataFrame]:
    cols = ["exchange", "symbol", "timestamp", "date", "open_interest_raw", "open_interest_usd", "source"]
    return {symbol: _empty(cols) for symbol in SYMBOLS}


def _oi_complete_days(oi_raw: dict[str, pd.DataFrame]) -> int:
    if any(oi_raw.get(symbol, pd.DataFrame()).empty for symbol in SYMBOLS):
        return 0
    eth_dates = set(oi_raw["ETHUSDT"].get("date", pd.Series(dtype=object)).dropna().astype(str))
    btc_dates = set(oi_raw["BTCUSDT"].get("date", pd.Series(dtype=object)).dropna().astype(str))
    return len(eth_dates & btc_dates)


def _requested_days(start_date: str, end_date: str) -> int:
    return max((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1, 1)


def _fetch_oi_raw_for_exchange(exchange: str, start_date: str, end_date: str, oi_source: str) -> dict[str, pd.DataFrame]:
    if oi_source == "none":
        print(f"[WARN] OI source disabled for {exchange}; OI-dependent models will be unavailable.")
        return _empty_oi_raw()

    if exchange == "bybit":
        if oi_source == "binance":
            print("[WARN] --oi-source=binance is ignored for Bybit panel; using Bybit OI.")
        return {s: fetch_bybit_open_interest(s, start_date, end_date) for s in SYMBOLS}

    def _fetch_binance() -> dict[str, pd.DataFrame]:
        return {s: fetch_binance_open_interest_hist(s, start_date, end_date) for s in SYMBOLS}

    def _fetch_bybit_fallback() -> dict[str, pd.DataFrame]:
        print("[WARN] Using Bybit OI as fallback/source for Binance master panel.")
        return {s: fetch_bybit_open_interest(s, start_date, end_date) for s in SYMBOLS}

    if oi_source == "bybit":
        return _fetch_bybit_fallback()
    if oi_source == "binance":
        return _fetch_binance()

    oi_raw = _fetch_binance()
    complete_days = _oi_complete_days(oi_raw)
    requested_days = _requested_days(start_date, end_date)
    if complete_days < min(60, requested_days * 0.5):
        print(
            f"[WARN] Binance OI coverage has {complete_days} complete ETH/BTC days "
            f"for a {requested_days}-day requested sample; falling back to Bybit OI."
        )
        oi_raw = _fetch_bybit_fallback()
    return oi_raw

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def collect_exchange_panel(exchange: str, start_date: str, end_date: str, hourly: bool = True, oi_source: str = "auto") -> dict[str, pd.DataFrame]:
    if exchange == "binance":
        funding_raw = fetch_binance_funding_all(SYMBOLS, start_date, end_date)
        daily_klines = {s: fetch_binance_klines(s, "1d", start_date, end_date) for s in SYMBOLS}
        hourly_klines = {s: fetch_binance_klines(s, "1h", start_date, end_date) for s in SYMBOLS} if hourly else None
        oi_raw = _fetch_oi_raw_for_exchange(exchange, start_date, end_date, oi_source)
    elif exchange == "bybit":
        funding_raw = fetch_bybit_funding_all(SYMBOLS, start_date, end_date)
        daily_klines = {s: fetch_bybit_klines(s, "1d", start_date, end_date) for s in SYMBOLS}
        hourly_klines = {s: fetch_bybit_klines(s, "1h", start_date, end_date) for s in SYMBOLS} if hourly else None
        oi_raw = _fetch_oi_raw_for_exchange(exchange, start_date, end_date, oi_source)
    else:
        raise ValueError(f"Unsupported exchange={exchange}")
    funding_daily = {s: funding_to_daily_annualized(funding_raw[s], exchange=exchange, symbol=s) for s in SYMBOLS}
    spread = build_eth_btc_spread(funding_daily["ETHUSDT"], funding_daily["BTCUSDT"], exchange)
    price_features = build_price_features(daily_klines, hourly_klines, exchange)
    oi_features = build_oi_features(oi_raw["ETHUSDT"], oi_raw["BTCUSDT"], price_features, exchange)
    return {
        "spread": spread,
        "price_features": price_features,
        "oi_features": oi_features,
        "funding_BTCUSDT": funding_raw["BTCUSDT"],
        "funding_ETHUSDT": funding_raw["ETHUSDT"],
        "oi_BTCUSDT": oi_raw["BTCUSDT"],
        "oi_ETHUSDT": oi_raw["ETHUSDT"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect funding/OI/price controls and estimate ETH-BTC funding spread models.")
    p.add_argument("--start-date", default=START_DATE, help="UTC start date YYYY-MM-DD")
    p.add_argument("--end-date", default=END_DATE, help="UTC end date YYYY-MM-DD; default=today UTC")
    p.add_argument("--exchange", choices=["binance", "bybit", "both"], default="both")
    p.add_argument("--lido-yield-csv", default=None, help="CSV with date,stake_yield; defaults to processed Lido candidates")
    p.add_argument("--skip-hourly", action="store_true", help="Skip hourly klines/RV to reduce API calls")
    p.add_argument("--skip-optional", action="store_true", help="Skip optional basis, taker, long-short, and stETH controls")
    p.add_argument(
        "--oi-source",
        choices=["auto", "binance", "bybit", "none"],
        default="auto",
        help="OI source for Binance panels: auto tries Binance recent metrics then falls back to Bybit; bybit forces fallback; none disables OI.",
    )
    p.add_argument("--hac-lags", type=int, default=7, help="Newey-West/HAC max lags")
    return p.parse_args()


def main() -> None:
    ensure_dir(OUTPUT_DIR_RAW)
    ensure_dir(OUTPUT_DIR_PROCESSED)
    args = parse_args()
    exchanges = ["binance", "bybit"] if args.exchange == "both" else [args.exchange]
    lido_yield = load_lido_yield(args.lido_yield_csv)
    optional_controls = basis = steth = None
    if not args.skip_optional:
        optional_controls = build_optional_leverage_controls(args.start_date, args.end_date)
        basis = build_binance_basis_features(args.start_date, args.end_date)
        steth = fetch_steth_discount(args.start_date, args.end_date)
    all_reports: dict[str, pd.DataFrame] = {}
    summaries = []
    for exchange in exchanges:
        try:
            panel = collect_exchange_panel(exchange, args.start_date, args.end_date, hourly=not args.skip_hourly, oi_source=args.oi_source)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Exchange panel failed for {exchange}: {exc}")
            continue
        master = build_master_dataset(
            exchange,
            panel["spread"],
            panel["price_features"],
            panel["oi_features"],
            lido_yield,
            optional_controls=optional_controls,
            basis=basis,
            steth_discount=steth,
        )
        all_reports.update({f"{exchange}_{k}": v for k, v in panel.items()})
        all_reports[f"master_{exchange}"] = master
        if "stake_yield" in master.columns:
            summary, _ = run_model_ladder(master, exchange, maxlags=args.hac_lags)
            if not summary.empty:
                summaries.append(summary)
    validation_summary(all_reports)
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        save_csv(combined, OUTPUT_DIR_PROCESSED / "ols_hac_model_summary_all_exchanges.csv")
        stable = combined.groupby("exchange").agg(
            models=("model", "count"),
            positive_share=("stake_yield_coef", lambda s: float((s > 0).mean())),
            sig_5pct_share=("stake_yield_sig_5pct", "mean"),
            sig_10pct_share=("stake_yield_sig_10pct", "mean"),
        )
        print("\n[SUMMARY] Stake-yield sign/significance stability across model ladder")
        print(stable)


if __name__ == "__main__":
    main()
