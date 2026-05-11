#!/usr/bin/env python3
"""Build a daily funding/yield panel with wstETH contract-rate LST yield.

Primary ETH LST yield is the wstETH contract exchange rate change from
getStETHByWstETH(1e18), not DeFiLlama APY and not a market price ratio.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

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
    "eth_native_yield",
    "stake_yield",
    "wsteth_defillama_apy",  # robustness fallback only
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BTC/ETH funding panel merged with ETH yield proxies.")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--exchange", default="binance", choices=["binance", "bybit"])
    p.add_argument("--assets", nargs="+", default=["BTC", "ETH"], help="Assets to include; BTC and ETH are required for spread.")
    p.add_argument("--input-dir", type=Path, default=PROCESSED_DATA_DIR)
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--out-csv", type=Path, default=PROCESSED_DATA_DIR / "funding_yield_panel.csv")
    p.add_argument("--fetch-onchain-lst-rates", action="store_true")
    p.add_argument("--ethereum-rpc-url", default=os.getenv("ETHEREUM_RPC_URL"))
    p.add_argument("--eth-blocks-csv", type=Path, default=None)
    p.add_argument("--lst-rate-cache-dir", type=Path, default=ROOT / "data" / "cache" / "lst_rates")
    p.add_argument("--wsteth-rate-csv", type=Path, default=None, help="CSV containing contract wstETH exchange rate, not market ratio.")
    p.add_argument("--sample-time-utc", default="12:00:00")
    p.add_argument("--rpc-sleep-seconds", type=float, default=0.0)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--fallback-yield-csv", type=Path, default=None, help="Optional robustness/fallback yield CSV.")
    return p.parse_args()


def _date_filter(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return out[(out["date"] >= start) & (out["date"] <= end)].sort_values("date")


def load_daily_funding(asset: str, args: argparse.Namespace) -> pd.DataFrame:
    symbol = f"{asset.upper()}USDT"
    processed_path = args.input_dir / f"{args.exchange}_{symbol}_funding_daily.csv"
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        if "funding_ann" not in df.columns:
            raise ValueError(f"{processed_path} must contain funding_ann")
        out = _date_filter(df, args.start_date, args.end_date)
        return out[["date", "funding_ann"]].rename(columns={"funding_ann": f"f_{asset.lower()}"})

    raw_candidates = [
        args.raw_dir / f"{args.exchange}_funding_{symbol}.csv",
        args.raw_dir / f"{args.exchange}_{symbol.lower()}_funding.csv",
    ]
    raw_path = next((p for p in raw_candidates if p.exists()), None)
    if raw_path is None:
        raise FileNotFoundError(f"No daily or raw funding file found for {args.exchange} {symbol}")
    raw = pd.read_csv(raw_path)
    daily = funding_to_daily_annualized(raw, time_col="fundingTime", rate_col="fundingRate")
    out = _date_filter(daily, args.start_date, args.end_date)
    return out[["date", "funding_ann"]].rename(columns={"funding_ann": f"f_{asset.lower()}"})


def build_funding_panel(args: argparse.Namespace) -> pd.DataFrame:
    assets = [a.upper() for a in args.assets]
    if not {"BTC", "ETH"}.issubset(set(assets)):
        raise ValueError("--assets must include BTC and ETH to build the BTC-minus-ETH spread.")
    panel = None
    for asset in assets:
        frame = load_daily_funding(asset, args)
        panel = frame if panel is None else panel.merge(frame, on="date", how="outer")
    panel = panel.sort_values("date")
    panel["exchange"] = args.exchange
    panel["spread_btc_minus_eth"] = panel["f_btc"] - panel["f_eth"]
    return panel


def load_fallback_yield(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Fallback yield CSV must contain date: {path}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    if "apy" in out.columns and "wsteth_defillama_apy" not in out.columns:
        out["wsteth_defillama_apy"] = pd.to_numeric(out["apy"], errors="coerce")
    return out


def choose_primary_yield(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    selected = None
    for col in YIELD_PRIORITY:
        if col in out.columns and pd.to_numeric(out[col], errors="coerce").notna().any():
            selected = col
            break
    if selected:
        if selected == "wsteth_defillama_apy":
            print("WARNING: using wsteth_defillama_apy only as robustness fallback; it is not wstETH implied yield.")
        out["eth_yield_primary"] = pd.to_numeric(out[selected], errors="coerce")
        out["eth_yield_primary_source"] = selected
        # Keep existing analysis scripts compatible while preserving explicit source metadata.
        out["stake_yield"] = out["eth_yield_primary"]
    else:
        print("WARNING: no ETH yield candidate found. Expected wsteth_implied_yield_7d as primary.")
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
        print("WARNING: --fetch-onchain-lst-rates requested but no --ethereum-rpc-url or ETHEREUM_RPC_URL was provided.")
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
    except Exception as exc:  # noqa: BLE001 - CLI should warn and continue with no primary yield.
        print(f"WARNING: wstETH contract exchange-rate fetch failed: {exc}")
        print("WARNING: wstETH contract exchange-rate fetch failed. No primary ETH LST-implied yield was generated. DeFiLlama APY or market price ratios are not substitutes for wstETH_implied_yield.")
        return pd.DataFrame(columns=["date"])


def main() -> None:
    args = parse_args()
    panel = build_funding_panel(args)
    wsteth_rates = load_wsteth_rates(args)
    if not wsteth_rates.empty:
        panel = panel.merge(wsteth_rates, on="date", how="left")
    fallback = load_fallback_yield(args.fallback_yield_csv)
    if not fallback.empty:
        forbidden = [c for c in fallback.columns if c.startswith("wsteth_implied_yield_")]
        if forbidden:
            raise ValueError(f"Fallback yield CSV must not provide wsteth_implied_yield_* columns: {forbidden}")
        panel = panel.merge(fallback, on="date", how="left", suffixes=("", "_fallback"))
    panel = choose_primary_yield(panel)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} ({len(panel):,} rows)")


if __name__ == "__main__":
    main()
