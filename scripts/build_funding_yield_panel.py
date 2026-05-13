#!/usr/bin/env python3
"""Build a daily funding/yield panel with wstETH contract-rate LST yield.

Primary ETH LST yield is the wstETH contract exchange rate change from
getStETHByWstETH(1e18), not DeFiLlama APY and not a market price ratio.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys
import time

import requests

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from common.transforms import funding_to_daily_annualized
from common.yield_pipeline import (
    fetch_wsteth_contract_rate_history,
    load_wsteth_rate_csv,
    print_wsteth_rate_summary,
)

YIELD_PRIORITY = [
    "wsteth_implied_yield_7d",
    "wsteth_implied_yield_30d",
    "stake_yield",
    "eth_native_yield",
    "wsteth_defillama_apy",  # robustness fallback only
]

ASSET_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "XRP": "XRPUSDT",
    "DOGE": "DOGEUSDT",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect BTC/ETH/XRP/DOGE funding and build a panel merged with ETH yield proxies."
    )
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--exchange", default="binance", choices=["binance", "bybit"])
    p.add_argument(
        "--assets",
        nargs="+",
        default=["BTC", "ETH", "XRP", "DOGE"],
        help="Assets to include; BTC and ETH are required for spread.",
    )
    p.add_argument("--input-dir", type=Path, default=PROCESSED_DATA_DIR)
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument(
        "--out-csv", type=Path, default=PROCESSED_DATA_DIR / "funding_yield_panel.csv"
    )
    p.add_argument(
        "--eth-yield-panel-csv",
        type=Path,
        default=PROCESSED_DATA_DIR / "eth_yield_panel.csv",
        help="Existing ETH yield panel to merge as-is.",
    )
    p.add_argument(
        "--skip-fetch-missing",
        action="store_true",
        help="Do not collect missing funding/price data from exchange APIs.",
    )
    p.add_argument(
        "--request-sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between paginated exchange API requests.",
    )
    p.add_argument(
        "--fetch-basis",
        action="store_true",
        help="Collect/merge daily perp-spot basis columns for requested assets.",
    )
    p.add_argument(
        "--basis-source",
        default="auto",
        choices=["auto", "mark_price", "perp_close_fallback"],
        help="Perp price source for basis; auto tries mark price then futures close fallback.",
    )
    p.add_argument("--fetch-onchain-lst-rates", action="store_true")
    p.add_argument("--ethereum-rpc-url", default=os.getenv("ETHEREUM_RPC_URL"))
    p.add_argument("--eth-blocks-csv", type=Path, default=None)
    p.add_argument(
        "--lst-rate-cache-dir", type=Path, default=ROOT / "data" / "cache" / "lst_rates"
    )
    p.add_argument(
        "--wsteth-rate-csv",
        type=Path,
        default=None,
        help="CSV containing contract wstETH exchange rate, not market ratio.",
    )
    p.add_argument("--sample-time-utc", default="12:00:00")
    p.add_argument("--rpc-sleep-seconds", type=float, default=0.0)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--fallback-yield-csv",
        type=Path,
        default=None,
        help="Optional robustness/fallback yield CSV.",
    )
    return p.parse_args()


def _parse_date_utc(value: str) -> datetime:
    return pd.Timestamp(value).to_pydatetime().replace(tzinfo=timezone.utc)


def _to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _end_date_inclusive_ms(value: str) -> int:
    end = _parse_date_utc(value) + timedelta(days=1) - timedelta(milliseconds=1)
    return _to_ms(end)


def _request_json(
    url: str, params: dict[str, object], *, source_name: str, sleep_seconds: float
) -> object:
    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 451:
        raise RuntimeError(f"{source_name} returned HTTP 451 (regional restriction).")
    response.raise_for_status()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return response.json()


def fetch_binance_funding_history(
    symbol: str, start_date: str, end_date: str, sleep_seconds: float = 0.2
) -> pd.DataFrame:
    """Collect paginated Binance USD-M perpetual funding history for one symbol."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    start_ms = _to_ms(_parse_date_utc(start_date))
    end_ms = _end_date_inclusive_ms(end_date)
    rows: list[dict[str, object]] = []
    current = start_ms
    while current <= end_ms:
        params = {
            "symbol": symbol,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }
        batch = _request_json(
            url, params, source_name="Binance funding", sleep_seconds=sleep_seconds
        )
        if not batch:
            break
        if not isinstance(batch, list):
            raise RuntimeError(
                f"Unexpected Binance funding payload for {symbol}: {batch}"
            )
        rows.extend(batch)
        last_ms = max(int(item["fundingTime"]) for item in batch)
        next_ms = last_ms + 1
        if next_ms <= current:
            break
        current = next_ms
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "fundingTime", "fundingRate"])
    df["fundingTime"] = pd.to_numeric(df["fundingTime"], errors="coerce").astype(
        "int64"
    )
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["symbol"] = symbol
    df = df[(df["fundingTime"] >= start_ms) & (df["fundingTime"] <= end_ms)]
    return (
        df[["symbol", "fundingTime", "fundingRate"]]
        .drop_duplicates(["symbol", "fundingTime"])
        .sort_values("fundingTime")
    )


def fetch_bybit_funding_history(
    symbol: str, start_date: str, end_date: str, sleep_seconds: float = 0.2
) -> pd.DataFrame:
    """Collect paginated Bybit linear perpetual funding history for one symbol."""
    url = "https://api.bybit.com/v5/market/funding/history"
    start_ms = _to_ms(_parse_date_utc(start_date))
    end_ms = _end_date_inclusive_ms(end_date)
    rows: list[dict[str, object]] = []
    current_end = end_ms
    while current_end >= start_ms:
        params = {
            "category": "linear",
            "symbol": symbol,
            "endTime": current_end,
            "limit": 200,
        }
        payload = _request_json(
            url, params, source_name="Bybit funding", sleep_seconds=sleep_seconds
        )
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Unexpected Bybit funding payload for {symbol}: {payload}"
            )
        batch = payload.get("result", {}).get("list", [])
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(item["fundingRateTimestamp"]) for item in batch)
        if oldest < start_ms:
            break
        next_end = oldest - 1
        if next_end >= current_end:
            break
        current_end = next_end
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "fundingTime", "fundingRate"])
    df["fundingTime"] = pd.to_numeric(
        df["fundingRateTimestamp"], errors="coerce"
    ).astype("int64")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["symbol"] = symbol
    df = df[(df["fundingTime"] >= start_ms) & (df["fundingTime"] <= end_ms)]
    return (
        df[["symbol", "fundingTime", "fundingRate"]]
        .drop_duplicates(["symbol", "fundingTime"])
        .sort_values("fundingTime")
    )


def fetch_binance_klines_history(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    url: str,
    source_name: str,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """Collect paginated Binance 1d kline-style data and normalize OHLCV columns."""
    start_ms = _to_ms(_parse_date_utc(start_date))
    end_ms = _end_date_inclusive_ms(end_date)
    rows: list[list[object]] = []
    current = start_ms
    while current <= end_ms:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }
        batch = _request_json(
            url, params, source_name=source_name, sleep_seconds=sleep_seconds
        )
        if not batch:
            break
        if not isinstance(batch, list):
            raise RuntimeError(
                f"Unexpected {source_name} payload for {symbol}: {batch}"
            )
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_ms = last_open + 24 * 60 * 60 * 1000
        if next_ms <= current:
            break
        current = next_ms
        if len(batch) < 1000:
            break
    base_cols = ["open_time", "open", "high", "low", "close", "volume"]
    if not rows:
        return pd.DataFrame(columns=["date", *base_cols[1:]])
    normalized_rows = [list(row[:6]) for row in rows]
    df = pd.DataFrame(normalized_rows, columns=base_cols)
    df["date"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return (
        df[["date", "open", "high", "low", "close", "volume"]]
        .drop_duplicates("date")
        .sort_values("date")
    )


def fetch_binance_daily_klines_history(
    symbol: str, start_date: str, end_date: str, sleep_seconds: float = 0.2
) -> pd.DataFrame:
    """Collect paginated Binance daily spot OHLCV history."""
    return fetch_binance_klines_history(
        symbol,
        start_date,
        end_date,
        url="https://api.binance.com/api/v3/klines",
        source_name="Binance spot klines",
        sleep_seconds=sleep_seconds,
    )


def fetch_binance_futures_klines_history(
    symbol: str, start_date: str, end_date: str, sleep_seconds: float = 0.2
) -> pd.DataFrame:
    """Collect paginated Binance USD-M futures daily OHLCV history."""
    return fetch_binance_klines_history(
        symbol,
        start_date,
        end_date,
        url="https://fapi.binance.com/fapi/v1/klines",
        source_name="Binance futures klines",
        sleep_seconds=sleep_seconds,
    )


def fetch_binance_mark_price_klines_history(
    symbol: str, start_date: str, end_date: str, sleep_seconds: float = 0.2
) -> pd.DataFrame:
    """Collect paginated Binance USD-M mark-price daily OHLC history."""
    return fetch_binance_klines_history(
        symbol,
        start_date,
        end_date,
        url="https://fapi.binance.com/fapi/v1/markPriceKlines",
        source_name="Binance mark-price klines",
        sleep_seconds=sleep_seconds,
    )


def _date_filter(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    out = df.copy()
    out["date"] = (
        pd.to_datetime(out["date"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return out[(out["date"] >= start) & (out["date"] <= end)].sort_values("date")


def asset_to_symbol(asset: str) -> str:
    """Map an asset ticker to the USDT perpetual symbol used by local funding files."""
    asset_upper = asset.upper()
    return ASSET_SYMBOLS.get(asset_upper, f"{asset_upper}USDT")


def collect_funding_history(asset: str, args: argparse.Namespace) -> pd.DataFrame:
    symbol = asset_to_symbol(asset)
    print(f"[INFO] Collecting {args.exchange} funding history for {symbol}...")
    if args.exchange == "binance":
        raw = fetch_binance_funding_history(
            symbol, args.start_date, args.end_date, args.request_sleep_seconds
        )
    elif args.exchange == "bybit":
        raw = fetch_bybit_funding_history(
            symbol, args.start_date, args.end_date, args.request_sleep_seconds
        )
    else:
        raise ValueError(f"Unsupported exchange: {args.exchange}")
    if raw.empty:
        raise RuntimeError(f"No funding rows collected for {args.exchange} {symbol}")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_dir / f"{args.exchange}_funding_{symbol}.csv"
    raw.to_csv(raw_path, index=False)
    print(f"[INFO] Saved raw funding: {raw_path} ({len(raw):,} rows)")
    return raw


def load_daily_funding(asset: str, args: argparse.Namespace) -> pd.DataFrame:
    symbol = asset_to_symbol(asset)
    processed_path = args.input_dir / f"{args.exchange}_{symbol}_funding_daily.csv"
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        if "funding_ann" not in df.columns:
            raise ValueError(f"{processed_path} must contain funding_ann")
        out = _date_filter(df, args.start_date, args.end_date)
        return out[["date", "funding_ann"]].rename(
            columns={"funding_ann": f"f_{asset.lower()}"}
        )

    raw_candidates = [
        args.raw_dir / f"{args.exchange}_funding_{symbol}.csv",
        args.raw_dir / f"{args.exchange}_{symbol.lower()}_funding.csv",
    ]
    raw_path = next((p for p in raw_candidates if p.exists()), None)
    if raw_path is None:
        if getattr(args, "skip_fetch_missing", False):
            raise FileNotFoundError(
                f"No daily or raw funding file found for {args.exchange} {symbol}"
            )
        raw = collect_funding_history(asset, args)
    else:
        raw = pd.read_csv(raw_path)
    daily = funding_to_daily_annualized(
        raw, time_col="fundingTime", rate_col="fundingRate"
    )
    args.input_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(processed_path, index=False)
    print(f"[INFO] Saved daily funding: {processed_path} ({len(daily):,} rows)")
    out = _date_filter(daily, args.start_date, args.end_date)
    return out[["date", "funding_ann"]].rename(
        columns={"funding_ann": f"f_{asset.lower()}"}
    )


def build_funding_panel(args: argparse.Namespace) -> pd.DataFrame:
    assets = [a.upper() for a in args.assets]
    if not {"BTC", "ETH"}.issubset(set(assets)):
        raise ValueError(
            "--assets must include BTC and ETH to build the BTC-minus-ETH spread."
        )
    panel = None
    for asset in assets:
        frame = load_daily_funding(asset, args)
        panel = frame if panel is None else panel.merge(frame, on="date", how="outer")
    panel = panel.sort_values("date")
    panel["exchange"] = args.exchange
    panel["spread_btc_minus_eth"] = panel["f_btc"] - panel["f_eth"]
    for asset in assets:
        src = f"f_{asset.lower()}"
        if src in panel.columns:
            panel[f"{asset.lower()}_funding"] = panel[src]
    return panel


def load_price_controls(args: argparse.Namespace) -> pd.DataFrame:
    """Load or collect ETH/BTC daily close files and build relative return/RV controls."""
    candidates = {
        "eth": [
            args.raw_dir / "binance_ethusdt_1d.csv",
            args.raw_dir / "binance_ETHUSDT_1d.csv",
            args.input_dir / "binance_ETHUSDT_1d.csv",
        ],
        "btc": [
            args.raw_dir / "binance_btcusdt_1d.csv",
            args.raw_dir / "binance_BTCUSDT_1d.csv",
            args.input_dir / "binance_BTCUSDT_1d.csv",
        ],
    }
    symbols = {"eth": "ETHUSDT", "btc": "BTCUSDT"}
    frames = {}
    for asset, paths in candidates.items():
        path = next((p for p in paths if p.exists()), None)
        if path is None and not getattr(args, "skip_fetch_missing", False):
            symbol = symbols[asset]
            print(
                f"[INFO] Collecting Binance daily klines for {symbol} price controls..."
            )
            df = fetch_binance_daily_klines_history(
                symbol, args.start_date, args.end_date, args.request_sleep_seconds
            )
            if not df.empty:
                args.raw_dir.mkdir(parents=True, exist_ok=True)
                path = args.raw_dir / f"binance_{symbol.lower()}_1d.csv"
                df.to_csv(path, index=False)
                print(f"[INFO] Saved daily klines: {path} ({len(df):,} rows)")
        if path is None:
            continue
        df = pd.read_csv(path)
        if not {"date", "close"}.issubset(df.columns):
            continue
        tmp = _date_filter(df[["date", "close"]], args.start_date, args.end_date)
        tmp[f"close_{asset}"] = pd.to_numeric(tmp["close"], errors="coerce")
        frames[asset] = tmp[["date", f"close_{asset}"]]
    if not {"eth", "btc"}.issubset(frames):
        print(
            "WARNING: ETH/BTC daily close files not found; ret_eth_btc and rv_eth_btc were not generated."
        )
        return pd.DataFrame(columns=["date"])
    out = frames["eth"].merge(frames["btc"], on="date", how="inner").sort_values("date")
    ratio = out["close_eth"] / out["close_btc"]
    clean_ratio = ratio.where(ratio > 0)
    out["ret_eth_btc"] = np.log(clean_ratio).diff()
    out["rv_eth_btc"] = out["ret_eth_btc"].rolling(7, min_periods=2).std() * (
        365.0**0.5
    )
    return out[["date", "ret_eth_btc", "rv_eth_btc"]]


def _kline_candidates(args: argparse.Namespace, symbol: str, kind: str) -> list[Path]:
    lower = symbol.lower()
    upper = symbol.upper()
    if kind == "spot":
        return [
            args.raw_dir / f"binance_{lower}_1d.csv",
            args.raw_dir / f"binance_{upper}_1d.csv",
            args.raw_dir / f"binance_{lower}_spot_1d.csv",
            args.input_dir / f"binance_{upper}_1d.csv",
        ]
    if kind == "mark":
        return [
            args.raw_dir / f"binance_{lower}_mark_1d.csv",
            args.raw_dir / f"binance_{upper}_mark_1d.csv",
            args.input_dir / f"binance_{upper}_mark_1d.csv",
        ]
    if kind == "perp":
        return [
            args.raw_dir / f"binance_{lower}_perp_1d.csv",
            args.raw_dir / f"binance_{upper}_perp_1d.csv",
            args.raw_dir / f"binance_{lower}_futures_1d.csv",
            args.input_dir / f"binance_{upper}_perp_1d.csv",
        ]
    raise ValueError(f"Unknown kline kind: {kind}")


def _load_or_collect_binance_klines(
    symbol: str, kind: str, args: argparse.Namespace
) -> pd.DataFrame:
    path = next((p for p in _kline_candidates(args, symbol, kind) if p.exists()), None)
    if path is not None:
        df = pd.read_csv(path)
    else:
        if getattr(args, "skip_fetch_missing", False):
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        if kind == "spot":
            df = fetch_binance_daily_klines_history(
                symbol, args.start_date, args.end_date, args.request_sleep_seconds
            )
            path = args.raw_dir / f"binance_{symbol.lower()}_spot_1d.csv"
        elif kind == "mark":
            df = fetch_binance_mark_price_klines_history(
                symbol, args.start_date, args.end_date, args.request_sleep_seconds
            )
            path = args.raw_dir / f"binance_{symbol.lower()}_mark_1d.csv"
        elif kind == "perp":
            df = fetch_binance_futures_klines_history(
                symbol, args.start_date, args.end_date, args.request_sleep_seconds
            )
            path = args.raw_dir / f"binance_{symbol.lower()}_perp_1d.csv"
        else:
            raise ValueError(f"Unknown kline kind: {kind}")
        if not df.empty:
            df.to_csv(path, index=False)
            print(f"[INFO] Saved {kind} klines: {path} ({len(df):,} rows)")
    if not {"date", "close"}.issubset(df.columns):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    keep = [
        col
        for col in ["date", "open", "high", "low", "close", "volume"]
        if col in df.columns
    ]
    out = _date_filter(df[keep], args.start_date, args.end_date)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_basis_panel(args: argparse.Namespace) -> pd.DataFrame:
    """Build daily perp-spot basis columns for requested Binance assets."""
    if not getattr(args, "fetch_basis", False):
        return pd.DataFrame(columns=["date"])
    if args.exchange != "binance":
        print(
            "WARNING: --fetch-basis currently supports Binance symbols only; skipping basis."
        )
        return pd.DataFrame(columns=["date"])

    panel: pd.DataFrame | None = None
    for asset in [a.upper() for a in args.assets]:
        symbol = asset_to_symbol(asset)
        prefix = asset.lower()
        spot = _load_or_collect_binance_klines(symbol, "spot", args)
        if spot.empty:
            print(
                f"WARNING: missing spot klines for {symbol}; skipping {prefix}_basis."
            )
            continue

        mark = pd.DataFrame()
        mark_error: Exception | None = None
        if args.basis_source in {"auto", "mark_price"}:
            try:
                mark = _load_or_collect_binance_klines(symbol, "mark", args)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - basis can fall back to futures close.
                mark_error = exc
                mark = pd.DataFrame()
                print(
                    f"WARNING: mark-price basis fetch/load failed for {symbol}: {exc}"
                )
        if args.basis_source == "mark_price" and mark.empty:
            detail = f" ({mark_error})" if mark_error else ""
            print(
                f"WARNING: mark-price basis unavailable for {symbol}{detail}; skipping asset."
            )
            continue

        perp = pd.DataFrame()
        basis_source = "mark_price"
        if mark.empty:
            perp = _load_or_collect_binance_klines(symbol, "perp", args)
            basis_source = "perp_close_fallback"
            if perp.empty:
                print(
                    f"WARNING: missing futures close fallback for {symbol}; skipping {prefix}_basis."
                )
                continue
        source_df = mark if basis_source == "mark_price" else perp
        merged = spot[["date", "close"]].rename(
            columns={"close": f"{prefix}_spot_close"}
        )
        price_cols = ["date", "close"]
        for optional_col in ["open", "high", "low"]:
            if optional_col in source_df.columns:
                price_cols.append(optional_col)
        rhs = source_df[price_cols].copy()
        if basis_source == "mark_price":
            rhs = rhs.rename(columns={"close": f"{prefix}_perp_mark"})
        else:
            rhs = rhs.rename(columns={"close": f"{prefix}_perp_close"})
        merged = merged.merge(rhs, on="date", how="inner")

        perp_price_col = (
            f"{prefix}_perp_mark"
            if basis_source == "mark_price"
            else f"{prefix}_perp_close"
        )
        merged[f"{prefix}_basis_close"] = (
            merged[perp_price_col] / merged[f"{prefix}_spot_close"] - 1.0
        )
        merged[f"{prefix}_basis"] = merged[f"{prefix}_basis_close"]
        if {"open", "high", "low"}.issubset(merged.columns):
            avg_perp = merged[["open", "high", "low", perp_price_col]].mean(axis=1)
            merged[f"{prefix}_basis_avg"] = (
                avg_perp / merged[f"{prefix}_spot_close"] - 1.0
            )
        else:
            merged[f"{prefix}_basis_avg"] = merged[f"{prefix}_basis_close"]
        merged[f"{prefix}_basis_source"] = basis_source

        keep_cols = [
            "date",
            f"{prefix}_spot_close",
            perp_price_col,
            f"{prefix}_basis",
            f"{prefix}_basis_close",
            f"{prefix}_basis_avg",
            f"{prefix}_basis_source",
        ]
        if (
            basis_source == "mark_price"
            and f"{prefix}_perp_close" not in merged.columns
        ):
            merged[f"{prefix}_perp_close"] = np.nan
            keep_cols.insert(3, f"{prefix}_perp_close")
        if (
            basis_source == "perp_close_fallback"
            and f"{prefix}_perp_mark" not in merged.columns
        ):
            merged[f"{prefix}_perp_mark"] = np.nan
            keep_cols.insert(2, f"{prefix}_perp_mark")
        print(
            f"[INFO] Built {prefix}_basis with source={basis_source} ({len(merged):,} rows)"
        )
        panel = (
            merged[keep_cols]
            if panel is None
            else panel.merge(merged[keep_cols], on="date", how="outer")
        )
    if panel is None:
        return pd.DataFrame(columns=["date"])
    return panel.sort_values("date")


def load_fallback_yield(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Fallback yield CSV must contain date: {path}")
    out = df.copy()
    out["date"] = (
        pd.to_datetime(out["date"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )
    if "apy" in out.columns and "wsteth_defillama_apy" not in out.columns:
        out["wsteth_defillama_apy"] = pd.to_numeric(out["apy"], errors="coerce")
    return out


def load_eth_yield_panel(path: Path | None, args: argparse.Namespace) -> pd.DataFrame:
    """Load the prebuilt ETH yield panel without regenerating yield data."""
    if path is None:
        return pd.DataFrame(columns=["date"])
    if not path.exists():
        print(
            f"WARNING: ETH yield panel not found: {path}. Build scripts/build_eth_yield_panel.py first."
        )
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"ETH yield panel must contain date: {path}")
    out = _date_filter(df, args.start_date, args.end_date)
    print(
        f"[INFO] Loaded ETH yield panel as-is: {path} ({len(out):,} rows in requested range)"
    )
    return out


def merge_without_overwriting(
    left: pd.DataFrame, right: pd.DataFrame, *, on: str = "date"
) -> pd.DataFrame:
    """Merge right-hand columns that are not already present in left."""
    if right.empty:
        return left
    keep_cols = [
        on,
        *[col for col in right.columns if col != on and col not in left.columns],
    ]
    if keep_cols == [on]:
        return left
    return left.merge(right[keep_cols], on=on, how="left")


def choose_primary_yield(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    selected = None
    for col in YIELD_PRIORITY:
        if (
            col in out.columns
            and pd.to_numeric(out[col], errors="coerce").notna().any()
        ):
            selected = col
            break
    if selected:
        if selected == "wsteth_defillama_apy":
            print(
                "WARNING: using wsteth_defillama_apy only as robustness fallback; it is not wstETH implied yield."
            )
        out["eth_yield_primary"] = pd.to_numeric(out[selected], errors="coerce")
        out["eth_yield_primary_source"] = selected
        # Keep existing analysis scripts compatible while preserving explicit source metadata.
        out["stake_yield"] = out["eth_yield_primary"]
    else:
        print(
            "WARNING: no ETH yield candidate found. Expected wsteth_implied_yield_7d as primary."
        )
    return out


def load_wsteth_rates(args: argparse.Namespace) -> pd.DataFrame:
    if args.wsteth_rate_csv:
        print(f"[INFO] Using user wstETH contract-rate CSV: {args.wsteth_rate_csv}")
        rates = load_wsteth_rate_csv(args.wsteth_rate_csv)
        print_wsteth_rate_summary(rates)
        return _date_filter(rates, args.start_date, args.end_date)

    if not args.fetch_onchain_lst_rates:
        return pd.DataFrame(columns=["date"])

    if not args.ethereum_rpc_url:
        print(
            "WARNING: --fetch-onchain-lst-rates requested but no --ethereum-rpc-url or ETHEREUM_RPC_URL was provided."
        )
        print_wsteth_rate_summary(pd.DataFrame())
        return pd.DataFrame(columns=["date"])

    args.lst_rate_cache_dir.mkdir(parents=True, exist_ok=True)
    block_cache = args.lst_rate_cache_dir / "ethereum_daily_blocks.csv"
    rate_cache = args.lst_rate_cache_dir / "wsteth_rate_history.csv"
    try:
        rates = fetch_wsteth_contract_rate_history(
            rpc_url=args.ethereum_rpc_url,
            start_date=args.start_date,
            end_date=args.end_date,
            sample_time_utc=args.sample_time_utc,
            eth_blocks_csv=args.eth_blocks_csv,
            block_cache_csv=block_cache,
            rate_cache_csv=rate_cache,
            sleep_seconds=args.rpc_sleep_seconds,
            show_progress=not args.no_progress,
        )
        print_wsteth_rate_summary(rates)
        return rates
    except (
        Exception
    ) as exc:  # noqa: BLE001 - CLI should warn and continue with no primary yield.
        print(f"WARNING: wstETH contract exchange-rate fetch failed: {exc}")
        print(
            "WARNING: wstETH contract exchange-rate fetch failed. No primary ETH LST-implied yield was generated. DeFiLlama APY or market price ratios are not substitutes for wstETH_implied_yield."
        )
        return pd.DataFrame(columns=["date"])


def main() -> None:
    args = parse_args()
    panel = build_funding_panel(args)
    price_controls = load_price_controls(args)
    panel = merge_without_overwriting(panel, price_controls)
    basis_panel = build_basis_panel(args)
    panel = merge_without_overwriting(panel, basis_panel)
    eth_yield_panel = load_eth_yield_panel(args.eth_yield_panel_csv, args)
    panel = merge_without_overwriting(panel, eth_yield_panel)
    wsteth_rates = load_wsteth_rates(args)
    panel = merge_without_overwriting(panel, wsteth_rates)
    fallback = load_fallback_yield(args.fallback_yield_csv)
    if not fallback.empty:
        forbidden = [
            c for c in fallback.columns if c.startswith("wsteth_implied_yield_")
        ]
        if forbidden:
            raise ValueError(
                f"Fallback yield CSV must not provide wsteth_implied_yield_* columns: {forbidden}"
            )
        panel = panel.merge(fallback, on="date", how="left", suffixes=("", "_fallback"))
    panel = choose_primary_yield(panel)
    for asset in [a.upper() for a in args.assets]:
        expected = f"{asset.lower()}_funding"
        if expected not in panel.columns:
            print(f"WARNING: expected funding output column missing: {expected}")
    for expected in ["ret_eth_btc", "rv_eth_btc"]:
        if expected not in panel.columns:
            print(f"WARNING: optional control column missing: {expected}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} ({len(panel):,} rows)")


if __name__ == "__main__":
    main()
