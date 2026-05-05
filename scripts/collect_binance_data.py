#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import RAW_DATA_DIR
from common.data_sources import fetch_binance_daily_klines, fetch_binance_funding, fetch_bybit_funding


def dt_to_ms(dt_str: str) -> int:
    return int(datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2023-11-27", help="YYYY-MM-DD")
    p.add_argument("--funding-source", default="auto", choices=["auto", "binance", "bybit"])
    return p.parse_args()


def main():
    args = parse_args()
    start_ms = dt_to_ms(args.start_date)

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        fund = None
        source_used = None
        if args.funding_source in ("auto", "binance"):
            try:
                fund = fetch_binance_funding(symbol=symbol, start_time_ms=start_ms, limit=1000)
                source_used = "binance"
            except Exception as e:
                if args.funding_source == "binance":
                    raise
                print(f"[WARN] Binance funding failed for {symbol}: {e}")
        if fund is None:
            fund = fetch_bybit_funding(symbol=symbol, start_time_ms=start_ms, limit=200)
            source_used = "bybit"
        fund.to_csv(RAW_DATA_DIR / f"{source_used}_{symbol.lower()}_funding.csv", index=False)
        print(f"Saved {source_used} funding for {symbol}: {len(fund)} rows")

        kline = fetch_binance_daily_klines(symbol=symbol, start_time_ms=start_ms, limit=1000)
        kline.to_csv(RAW_DATA_DIR / f"binance_{symbol.lower()}_1d.csv", index=False)

    print("Saved funding + Binance daily OHLCV under data/raw")


if __name__ == "__main__":
    main()
