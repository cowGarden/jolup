#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from common.yield_pipeline import build_eth_yield_panel


def parse_args():
    p = argparse.ArgumentParser(description="Build ETH yield panel from CF ETH_SRR and optional staking ratio inputs")
    p.add_argument("--cf-srr-csv", default=str(RAW_DATA_DIR / "cf_eth_srr.csv"))
    p.add_argument("--staked-csv", default=str(RAW_DATA_DIR / "staked_eth_daily.csv"))
    p.add_argument("--supply-csv", default=str(RAW_DATA_DIR / "eth_total_supply_daily.csv"))
    p.add_argument("--out-csv", default=str(PROCESSED_DATA_DIR / "eth_yield_panel.csv"))
    p.add_argument("--with-ratio", action="store_true", help="Include staking ratio from staked/supply CSVs")
    return p.parse_args()


def main():
    args = parse_args()
    staked = args.staked_csv if args.with_ratio else None
    supply = args.supply_csv if args.with_ratio else None

    panel = build_eth_yield_panel(
        cf_srr_csv=args.cf_srr_csv,
        out_csv=args.out_csv,
        staked_csv=staked,
        supply_csv=supply,
    )
    print(f"Saved: {args.out_csv}")
    print(panel.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
