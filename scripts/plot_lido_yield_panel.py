#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.config import FIGURES_DIR, PROCESSED_DATA_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Plot Lido wstETH share-rate yield panel")
    p.add_argument("--panel-csv", default=str(PROCESSED_DATA_DIR / "eth_yield_panel.csv"))
    p.add_argument("--out-dir", default=str(FIGURES_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.panel_csv)
    df["date"] = pd.to_datetime(df["date"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "share_rate" in df.columns:
        ax = df.plot(x="date", y="share_rate", figsize=(11, 4), legend=False, title="Lido wstETH/stETH protocol exchange rate")
        ax.set_ylabel("stETH per 1 wstETH")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / "lido_share_rate.png", dpi=150)
        plt.close()

    if "annualized_apr_pct" in df.columns:
        ax = df.plot(x="date", y="annualized_apr_pct", figsize=(11, 4), legend=False, title="Lido retail-accessible staking yield proxy")
        ax.set_ylabel("Annualized APR (%)")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / "lido_annualized_apr_pct.png", dpi=150)
        plt.close()

    print(f"Saved plots under {out_dir}")


if __name__ == "__main__":
    main()
