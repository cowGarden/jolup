#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from common.yield_pipeline import (
    build_eth_yield_panel,
    build_assumed_eth_yield_history,
    fetch_lido_current_apr,
    fetch_stakingrewards_eth_reward_rate_history,
    save_yield_csv,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build ETH yield panel from CF ETH_SRR or supported proxy sources")
    p.add_argument("--yield-source", default="manual", choices=["manual", "assumed", "stakingrewards", "lido-current"],
                   help="manual expects --cf-srr-csv to exist; assumed/stakingrewards/lido-current create it first")
    p.add_argument("--cf-srr-csv", default=str(RAW_DATA_DIR / "cf_eth_srr.csv"),
                   help="Input/output normalized yield CSV. For manual CF data, provide date,eth_srr columns.")
    p.add_argument("--start-date", default="2023-11-27", help="Start date for API history or assumed series")
    p.add_argument("--end-date", default=None, help="End date for assumed series; defaults to today")
    p.add_argument("--assumed-apr", type=float, default=3.0,
                   help="Constant APR for --yield-source assumed. Accepts 3.0 or 0.03 for 3%.")
    p.add_argument("--stakingrewards-api-key", default=os.getenv("STAKING_REWARDS_API_KEY"),
                   help="Staking Rewards API key; defaults to STAKING_REWARDS_API_KEY env var")
    p.add_argument("--history-limit", type=int, default=500,
                   help="Maximum Staking Rewards daily observations to request")
    p.add_argument("--staked-csv", default=str(RAW_DATA_DIR / "staked_eth_daily.csv"))
    p.add_argument("--supply-csv", default=str(RAW_DATA_DIR / "eth_total_supply_daily.csv"))
    p.add_argument("--out-csv", default=str(PROCESSED_DATA_DIR / "eth_yield_panel.csv"))
    p.add_argument("--with-ratio", action="store_true", help="Include staking ratio from staked/supply CSVs")
    return p.parse_args()


def ensure_yield_input(args) -> Path:
    cf_srr_csv = Path(args.cf_srr_csv)

    if args.yield_source == "manual":
        return cf_srr_csv

    if args.yield_source == "assumed":
        df = build_assumed_eth_yield_history(
            start_date=args.start_date,
            end_date=args.end_date,
            assumed_apr=args.assumed_apr,
        )
        save_yield_csv(df, cf_srr_csv)
        print(
            f"Saved assumed constant ETH staking APR series: {cf_srr_csv} "
            f"({len(df)} rows, APR={df['stake_yield'].iloc[0]:.4%})"
        )
        return cf_srr_csv

    if args.yield_source == "stakingrewards":
        if not args.stakingrewards_api_key:
            raise ValueError(
                "--yield-source stakingrewards requires --stakingrewards-api-key "
                "or STAKING_REWARDS_API_KEY environment variable."
            )
        df = fetch_stakingrewards_eth_reward_rate_history(
            api_key=args.stakingrewards_api_key,
            start_date=args.start_date,
            limit=args.history_limit,
        )
        if df.empty:
            raise RuntimeError("Staking Rewards returned no ETH reward_rate rows.")
        save_yield_csv(df, cf_srr_csv)
        print(f"Saved normalized Staking Rewards yield history: {cf_srr_csv} ({len(df)} rows)")
        return cf_srr_csv

    df = fetch_lido_current_apr(use_sma=True)
    save_yield_csv(df, cf_srr_csv)
    print(
        f"Saved one-row Lido APR proxy: {cf_srr_csv}. "
        "This is useful for pipeline smoke tests, not full regressions."
    )
    return cf_srr_csv


def main():
    args = parse_args()
    cf_srr_csv = ensure_yield_input(args)
    staked = args.staked_csv if args.with_ratio else None
    supply = args.supply_csv if args.with_ratio else None

    panel = build_eth_yield_panel(
        cf_srr_csv=cf_srr_csv,
        out_csv=args.out_csv,
        staked_csv=staked,
        supply_csv=supply,
    )
    print(f"Saved: {args.out_csv}")
    print(panel.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
