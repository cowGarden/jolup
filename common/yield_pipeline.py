from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_cf_eth_srr(csv_path: str | Path) -> pd.DataFrame:
    """
    Normalize CF ETH_SRR-like CSV to columns: date, stake_yield
    Accepts loose input names such as date/Date and ETH_SRR/stake_yield/value.
    """
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    date_col = cols.get("date") or cols.get("asofdate") or cols.get("timestamp")
    y_col = cols.get("eth_srr") or cols.get("stake_yield") or cols.get("value") or cols.get("apr")
    if not date_col or not y_col:
        raise ValueError("CF ETH_SRR CSV must contain date + yield column (e.g., date, eth_srr)")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "stake_yield": pd.to_numeric(df[y_col], errors="coerce"),
    }).dropna()

    # If values are percentages (e.g., 3.2), convert to decimal (0.032)
    if out["stake_yield"].median() > 1:
        out["stake_yield"] = out["stake_yield"] / 100.0

    return out.sort_values("date").drop_duplicates("date")


def build_staking_ratio(staked_csv: str | Path, supply_csv: str | Path) -> pd.DataFrame:
    """Build staking ratio from two daily series CSVs."""
    s = pd.read_csv(staked_csv)
    t = pd.read_csv(supply_csv)

    def normalize(df: pd.DataFrame, value_candidates: list[str], out_name: str) -> pd.DataFrame:
        cols = {c.lower().strip(): c for c in df.columns}
        date_col = cols.get("date") or cols.get("timestamp")
        value_col = next((cols.get(k) for k in value_candidates if cols.get(k)), None)
        if not date_col or not value_col:
            raise ValueError(f"Missing required columns for {out_name}")
        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            out_name: pd.to_numeric(df[value_col], errors="coerce"),
        }).dropna()
        return out

    staked = normalize(s, ["staked_eth", "staked", "value"], "staked_eth")
    supply = normalize(t, ["total_supply", "supply", "value"], "total_supply")

    m = staked.merge(supply, on="date", how="inner")
    m["staking_ratio"] = m["staked_eth"] / m["total_supply"]
    return m[["date", "staked_eth", "total_supply", "staking_ratio"]].sort_values("date")


def build_eth_yield_panel(cf_srr_csv: str | Path, out_csv: str | Path,
                          staked_csv: str | Path | None = None,
                          supply_csv: str | Path | None = None) -> pd.DataFrame:
    y = load_cf_eth_srr(cf_srr_csv)

    if staked_csv and supply_csv:
        ratio = build_staking_ratio(staked_csv, supply_csv)
        panel = y.merge(ratio, on="date", how="left")
    else:
        panel = y.copy()
        panel["staked_eth"] = pd.NA
        panel["total_supply"] = pd.NA
        panel["staking_ratio"] = pd.NA

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_csv, index=False)
    return panel
