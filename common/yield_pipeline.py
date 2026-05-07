from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

STAKING_REWARDS_GRAPHQL_URL = "https://api.stakingrewards.com/public/query"
LIDO_LAST_APR_URL = "https://eth-api.lido.fi/v1/protocol/steth/apr/last"
LIDO_SMA_APR_URL = "https://eth-api.lido.fi/v1/protocol/steth/apr/sma"


def _normalize_yield_value(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return values
    # If values are percentages (e.g., 3.2), convert to decimal (0.032).
    if values.dropna().median() > 1:
        values = values / 100.0
    return values


def load_cf_eth_srr(csv_path: str | Path) -> pd.DataFrame:
    """
    Normalize CF ETH_SRR-like CSV to columns: date, stake_yield.

    Accepted yield column names include eth_srr, stake_yield, reward_rate, value,
    apr, and apy. Values can be decimals (0.03) or percentages (3.0).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ETH staking yield input not found: {csv_path}\n"
            "Create it by either:\n"
            "  1) downloading/licensing CF ETH_SRR history and saving date,eth_srr columns, or\n"
            "  2) running scripts/build_eth_yield_panel.py --yield-source assumed --assumed-apr 3.0, or\n"
            "  3) running scripts/build_eth_yield_panel.py --yield-source stakingrewards with an API key, or\n"
            "  4) running scripts/build_eth_yield_panel.py --yield-source lido-current for a one-row smoke-test proxy."
        )

    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    date_col = cols.get("date") or cols.get("asofdate") or cols.get("timestamp") or cols.get("createdat")
    y_col = (
        cols.get("eth_srr")
        or cols.get("stake_yield")
        or cols.get("reward_rate")
        or cols.get("value")
        or cols.get("apr")
        or cols.get("apy")
    )
    if not date_col or not y_col:
        raise ValueError(
            "ETH yield CSV must contain a date column and a yield column "
            "(e.g. date, eth_srr or date, stake_yield)."
        )

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None),
        "stake_yield": _normalize_yield_value(df[y_col]),
    }).dropna()

    if "source" in cols:
        out["source"] = df.loc[out.index, cols["source"]].astype(str).values
    else:
        out["source"] = "manual_cf_eth_srr"

    return out.sort_values("date").drop_duplicates("date")


def build_assumed_eth_yield_history(start_date: str, end_date: str | None = None, assumed_apr: float = 0.03) -> pd.DataFrame:
    """Build a daily constant-yield panel for assumption/sensitivity analysis.

    `assumed_apr` accepts either decimal (0.03) or percent-style (3.0) input.
    This is not observed ETH_SRR data; it should be labeled as an assumption in
    paper tables and used for scenario/sensitivity analysis.
    """
    start = pd.to_datetime(start_date, errors="raise")
    end = pd.to_datetime(end_date, errors="raise") if end_date else pd.Timestamp.utcnow().normalize().tz_localize(None)
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    apr = float(assumed_apr)
    if apr > 1:
        apr = apr / 100.0

    dates = pd.date_range(start=start, end=end, freq="D")
    return pd.DataFrame({
        "date": dates,
        "stake_yield": apr,
        "source": f"assumed_constant_apr_{apr:.4f}",
    })


def fetch_stakingrewards_eth_reward_rate_history(api_key: str, start_date: str, limit: int = 500) -> pd.DataFrame:
    """Fetch daily ETH reward_rate history from Staking Rewards GraphQL API."""
    query = """
    query ethRewardRateHistory($timeStart: Date, $limit: Int) {
      rewardOptions(where: {inputAsset: {symbols: ["ETH"]}, typeKeys: ["solo-staking", "pos"]}, limit: 1) {
        metrics(where: {metricKeys: ["reward_rate"], createdAt_gt: $timeStart}, interval: day, limit: $limit) {
          metricKey
          defaultValue
          createdAt
        }
      }
    }
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"query": query, "variables": {"timeStart": start_date, "limit": limit}}
    r = requests.post(STAKING_REWARDS_GRAPHQL_URL, headers=headers, json=payload, timeout=45)
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(f"Staking Rewards API returned errors: {body['errors']}")

    options = body.get("data", {}).get("rewardOptions", [])
    metrics = options[0].get("metrics", []) if options else []
    out = pd.DataFrame(metrics)
    if out.empty:
        return pd.DataFrame(columns=["date", "stake_yield", "source"])

    out = pd.DataFrame({
        "date": pd.to_datetime(out["createdAt"], errors="coerce", utc=True).dt.tz_convert(None),
        "stake_yield": _normalize_yield_value(out["defaultValue"]),
        "source": "stakingrewards_reward_rate",
    }).dropna()
    return out.sort_values("date").drop_duplicates("date")


def fetch_lido_current_apr(use_sma: bool = True) -> pd.DataFrame:
    """Fetch current Lido stETH APR as a one-row proxy/smoke-test dataset."""
    url = LIDO_SMA_APR_URL if use_sma else LIDO_LAST_APR_URL
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()

    value = payload.get("data", {}).get("smaApr") if use_sma else payload.get("data", {}).get("apr")
    if value is None:
        value = payload.get("smaApr") if use_sma else payload.get("apr")
    if value is None:
        raise RuntimeError(f"Could not find APR value in Lido response: {payload}")

    return pd.DataFrame({
        "date": [datetime.now(timezone.utc).date().isoformat()],
        "stake_yield": _normalize_yield_value(pd.Series([value])),
        "source": ["lido_sma_apr" if use_sma else "lido_last_apr"],
    })


def save_yield_csv(df: pd.DataFrame, out_csv: str | Path) -> pd.DataFrame:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


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
            "date": pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None),
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
