#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import RAW_DATA_DIR
from common.data_sources import fetch_binance_daily_klines, fetch_binance_funding


def dt_to_ms(dt_str: str) -> int:
    return int(datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def main():
    start_ms = dt_to_ms("2023-11-27")

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        fund = fetch_binance_funding(symbol=symbol, start_time_ms=start_ms, limit=1000)
        fund.to_csv(RAW_DATA_DIR / f"binance_{symbol.lower()}_funding.csv", index=False)

        kline = fetch_binance_daily_klines(symbol=symbol, start_time_ms=start_ms, limit=1000)
        kline.to_csv(RAW_DATA_DIR / f"binance_{symbol.lower()}_1d.csv", index=False)

    print("Saved Binance funding and daily OHLCV under data/raw")


if __name__ == "__main__":
    main()
