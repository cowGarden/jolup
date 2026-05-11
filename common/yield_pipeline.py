from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
import warnings

from Crypto.Hash import keccak
import numpy as np
import pandas as pd
import requests

STAKING_REWARDS_GRAPHQL_URL = "https://api.stakingrewards.com/public/query"
LIDO_LAST_APR_URL = "https://eth-api.lido.fi/v1/protocol/steth/apr/last"
LIDO_SMA_APR_URL = "https://eth-api.lido.fi/v1/protocol/steth/apr/sma"
WSTETH_MAINNET_ADDRESS = "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0"
WSTETH_RATE_SOURCE = "ethereum_rpc_wsteth_contract"
WSTETH_ABI = [
    {
        "name": "getStETHByWstETH",
        "outputs": [{"type": "uint256"}],
        "inputs": [{"name": "_wstETHAmount", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    }
]
SECONDS_PER_DAY = 24 * 60 * 60
ONE_WSTETH_WEI = 10**18


def _normalize_yield_value(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return values
    # If values are percentages (e.g., 3.2), convert to decimal (0.032).
    if values.dropna().median() > 1:
        values = values / 100.0
    return values



def _normalize_daily_date(series: pd.Series) -> pd.Series:
    """Parse date-like values to naive normalized daily timestamps."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        unit = "ms" if numeric.dropna().abs().median() > 1e11 else "s"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
        if parsed.isna().any():
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                unit = "ms" if numeric.dropna().abs().median() > 1e11 else "s"
                parsed = parsed.fillna(pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce"))
    return parsed.dt.tz_convert(None).dt.normalize()


def add_implied_yields_from_rate(
    df: pd.DataFrame,
    rate_col: str = "wsteth_rate",
    windows: tuple[int, ...] = (1, 7, 30),
    prefix: str = "wsteth_implied_yield",
) -> pd.DataFrame:
    """Add log-change annualized implied yields from a protocol exchange-rate column.

    Yields are decimal annualized units. For example, 0.035 means 3.5% annualized.
    """
    if rate_col not in df.columns:
        raise KeyError(f"Missing rate column: {rate_col}")
    out = df.copy().sort_values("date") if "date" in df.columns else df.copy()
    rate = pd.to_numeric(out[rate_col], errors="coerce")
    log_rate = pd.Series(np.nan, index=out.index, dtype="float64")
    valid = rate > 0
    log_rate.loc[valid] = np.log(rate.loc[valid])
    for window in windows:
        out[f"{prefix}_{window}d"] = (365.0 / float(window)) * (log_rate - log_rate.shift(window))
    return out


def validate_wsteth_rate_shape(df: pd.DataFrame, rate_col: str = "wsteth_rate") -> dict:
    """Return diagnostics and warn if a rate CSV looks like a market ratio, not contract rate."""
    if rate_col not in df.columns:
        raise KeyError(f"Missing rate column: {rate_col}")
    rates = pd.to_numeric(df[rate_col], errors="coerce").dropna()
    diagnostics = {
        "obs": int(len(rates)),
        "start": float(rates.iloc[0]) if len(rates) else float("nan"),
        "end": float(rates.iloc[-1]) if len(rates) else float("nan"),
        "min": float(rates.min()) if len(rates) else float("nan"),
        "max": float(rates.max()) if len(rates) else float("nan"),
        "negative_steps": int((rates.diff().dropna() < 0).sum()),
        "total_log_drift": float(np.log(rates.iloc[-1] / rates.iloc[0])) if len(rates) >= 2 and rates.iloc[0] > 0 and rates.iloc[-1] > 0 else float("nan"),
    }
    if len(rates) >= 5:
        near_one = 0.95 <= diagnostics["min"] <= diagnostics["max"] <= 1.08
        low_or_negative_drift = diagnostics["total_log_drift"] < 0.01
        many_down_moves = diagnostics["negative_steps"] > max(2, int(0.05 * (len(rates) - 1)))
        if near_one and (low_or_negative_drift or many_down_moves):
            warnings.warn(
                "wstETH rate CSV looks like a market price ratio rather than a contract exchange rate: "
                "values hover near 1, lack clear upward drift, or frequently move down. "
                "Do not use market ratios as wsteth_implied_yield_*.",
                UserWarning,
                stacklevel=2,
            )
    return diagnostics


def load_wsteth_rate_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a user-supplied wstETH contract exchange-rate CSV and add implied yields."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("timestamp") or cols.get("day")
    rate_col = cols.get("wsteth_rate") or cols.get("share_rate") or cols.get("exchange_rate") or cols.get("steth_per_wsteth")
    block_col = cols.get("wsteth_rate_block") or cols.get("block_number") or cols.get("block")
    if not date_col or not rate_col:
        raise ValueError("wstETH rate CSV must contain date plus wsteth_rate/share_rate/exchange_rate.")
    out = pd.DataFrame({
        "date": _normalize_daily_date(df[date_col]),
        "wsteth_rate": pd.to_numeric(df[rate_col], errors="coerce"),
    }).dropna(subset=["date", "wsteth_rate"])
    out["wsteth_rate_block"] = pd.to_numeric(df[block_col], errors="coerce") if block_col else pd.NA
    out["wsteth_rate_source"] = "user_csv_wsteth_contract_rate"
    out = out.sort_values("date").drop_duplicates("date")
    validate_wsteth_rate_shape(out)
    return add_implied_yields_from_rate(out)

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
    # Primary regression yield priority: contract-rate wstETH implied yields first.
    # DeFiLlama/APY-style fields are accepted only as fallback/robustness inputs
    # and are never renamed to wsteth_implied_yield_* here.
    y_col = (
        cols.get("wsteth_implied_yield_7d")
        or cols.get("wsteth_implied_yield_30d")
        or cols.get("eth_native_yield")
        or cols.get("stake_yield")
        or cols.get("eth_srr")
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

    passthrough_cols = [
        "wsteth_rate",
        "wsteth_rate_block",
        "wsteth_rate_source",
        "wsteth_implied_yield_1d",
        "wsteth_implied_yield_7d",
        "wsteth_implied_yield_30d",
        "eth_native_yield",
        "wsteth_defillama_apy",
        "wsteth_eth_basis",
    ]
    for name in passthrough_cols:
        source_col = cols.get(name)
        if source_col and source_col in df.columns:
            out[name] = df.loc[out.index, source_col].values

    if "source" in cols:
        out["source"] = df.loc[out.index, cols["source"]].astype(str).values
    elif "wsteth_rate_source" in out.columns:
        out["source"] = out["wsteth_rate_source"].astype(str)
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


def _function_selector(signature: str) -> str:
    hasher = keccak.new(digest_bits=256)
    hasher.update(signature.encode("ascii"))
    return "0x" + hasher.hexdigest()[:8]


def _rpc_post(rpc_url: str, method: str, params: list, request_id: int = 1) -> dict:
    response = requests.post(
        rpc_url,
        json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"Ethereum RPC error for {method}: {payload['error']}")
    return payload["result"]


def _get_block(rpc_url: str, block_number: int) -> dict:
    return _rpc_post(rpc_url, "eth_getBlockByNumber", [hex(block_number), False])


def _latest_block_number(rpc_url: str) -> int:
    return int(_rpc_post(rpc_url, "eth_blockNumber", []), 16)


def _find_block_at_or_before_timestamp(rpc_url: str, target_ts: int, latest_block: int | None = None) -> int:
    latest = latest_block if latest_block is not None else _latest_block_number(rpc_url)
    latest_ts = int(_get_block(rpc_url, latest)["timestamp"], 16)
    if target_ts > latest_ts:
        raise ValueError("target timestamp is after the latest available Ethereum block")

    low, high = 0, latest
    while low < high:
        mid = (low + high + 1) // 2
        mid_ts = int(_get_block(rpc_url, mid)["timestamp"], 16)
        if mid_ts <= target_ts:
            low = mid
        else:
            high = mid - 1
    return low



def _find_closest_block_to_timestamp(rpc_url: str, target_ts: int, latest_block: int | None = None) -> int:
    """Find the Ethereum block closest to a UTC timestamp."""
    before = _find_block_at_or_before_timestamp(rpc_url, target_ts, latest_block=latest_block)
    latest = latest_block if latest_block is not None else _latest_block_number(rpc_url)
    before_ts = int(_get_block(rpc_url, before)["timestamp"], 16)
    if before >= latest:
        return before
    after = before + 1
    after_ts = int(_get_block(rpc_url, after)["timestamp"], 16)
    return after if abs(after_ts - target_ts) < abs(target_ts - before_ts) else before


def _load_date_block_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["date", "block_number"])
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("day") or cols.get("timestamp")
    block_col = cols.get("block_number") or cols.get("block") or cols.get("wsteth_rate_block")
    if not date_col or not block_col:
        raise ValueError(f"Block CSV must contain date and block_number columns: {path}")
    out = pd.DataFrame({
        "date": _normalize_daily_date(df[date_col]),
        "block_number": pd.to_numeric(df[block_col], errors="coerce"),
    }).dropna()
    out["block_number"] = out["block_number"].astype("int64")
    return out.sort_values("date").drop_duplicates("date")


def get_daily_blocks(
    rpc_url: str,
    dates: pd.DatetimeIndex,
    sample_time_utc: str = "12:00:00",
    eth_blocks_csv: str | Path | None = None,
    block_cache_csv: str | Path | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Return date/block_number using user CSV, cache, and RPC binary search for misses."""
    needed = pd.DataFrame({"date": pd.to_datetime(dates).normalize()}).drop_duplicates("date")
    provided = _load_date_block_csv(eth_blocks_csv)
    cached = _load_date_block_csv(block_cache_csv)
    known = pd.concat([provided, cached], ignore_index=True).dropna().drop_duplicates("date", keep="first")
    out = needed.merge(known, on="date", how="left")

    missing_idx = out["block_number"].isna()
    missing_dates = out.loc[missing_idx, "date"].tolist()
    if missing_dates:
        latest_block = _latest_block_number(rpc_url)
        rows = []
        for idx, date in enumerate(missing_dates, start=1):
            if show_progress:
                _print_progress("Ethereum block lookup", idx, len(missing_dates))
            target_ts = int(pd.Timestamp(f"{date.date()} {sample_time_utc}", tz="UTC").timestamp())
            rows.append({
                "date": date,
                "block_number": _find_closest_block_to_timestamp(rpc_url, target_ts, latest_block=latest_block),
            })
        if show_progress:
            print()
        fetched = pd.DataFrame(rows)
        known = pd.concat([known, fetched], ignore_index=True).drop_duplicates("date", keep="first")
        out = needed.merge(known, on="date", how="left")

    out["block_number"] = pd.to_numeric(out["block_number"], errors="coerce").astype("Int64")
    if block_cache_csv is not None:
        cache_path = Path(block_cache_csv)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        known.sort_values("date").drop_duplicates("date").to_csv(cache_path, index=False)
    return out.sort_values("date")


def _get_steth_by_wsteth_call_data(amount_wei: int = ONE_WSTETH_WEI) -> str:
    selector = _function_selector("getStETHByWstETH(uint256)")
    encoded_amount = hex(int(amount_wei))[2:].rjust(64, "0")
    return selector + encoded_amount

def _eth_call_uint256(rpc_url: str, to_address: str, data: str, block_number: int) -> int:
    result = _rpc_post(
        rpc_url,
        "eth_call",
        [{"to": to_address, "data": data}, hex(block_number)],
    )
    return int(result, 16)


def _print_progress(prefix: str, current: int, total: int, width: int = 28) -> None:
    """Render a single-line progress bar in the terminal."""
    if total <= 0:
        return
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = ratio * 100.0
    print(f"\r{prefix} [{bar}] {current}/{total} ({percent:5.1f}%)", end="", flush=True)



def fetch_wsteth_contract_rate_history(
    rpc_url: str,
    start_date: str,
    end_date: str | None = None,
    sample_time_utc: str = "12:00:00",
    eth_blocks_csv: str | Path | None = None,
    block_cache_csv: str | Path | None = None,
    rate_cache_csv: str | Path | None = None,
    sleep_seconds: float = 0.0,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Fetch wstETH contract exchange-rate history via historical eth_call.

    Uses wstETH.getStETHByWstETH(1e18) at each date's historical block and
    stores the stETH claim per 1 wstETH as `wsteth_rate`. This is a protocol
    exchange rate, not wstETH/ETH market price ratio or DeFiLlama APY.
    """
    start = pd.to_datetime(start_date, errors="raise").normalize()
    end = (pd.to_datetime(end_date, errors="raise") if end_date else pd.Timestamp.utcnow()).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    dates = pd.date_range(start=start, end=end, freq="D")
    needed = pd.DataFrame({"date": dates})

    cache = pd.DataFrame(columns=["date", "wsteth_rate", "wsteth_rate_block", "wsteth_rate_source"])
    if rate_cache_csv is not None and Path(rate_cache_csv).exists():
        cache_raw = pd.read_csv(rate_cache_csv)
        cols = {c.lower().strip(): c for c in cache_raw.columns}
        if "date" in cols and "wsteth_rate" in cols:
            block_col = cols.get("wsteth_rate_block") or cols.get("block_number") or cols.get("block")
            cache = pd.DataFrame({
                "date": _normalize_daily_date(cache_raw[cols["date"]]),
                "wsteth_rate": pd.to_numeric(cache_raw[cols["wsteth_rate"]], errors="coerce"),
                "wsteth_rate_block": pd.to_numeric(cache_raw[block_col], errors="coerce") if block_col else pd.NA,
            }).dropna(subset=["date", "wsteth_rate"])
            cache["wsteth_rate_source"] = WSTETH_RATE_SOURCE
            cache = cache.sort_values("date").drop_duplicates("date")

    merged = needed.merge(cache, on="date", how="left")
    missing_dates = merged.loc[merged["wsteth_rate"].isna(), "date"]
    rows = []
    if not missing_dates.empty:
        blocks = get_daily_blocks(
            rpc_url=rpc_url,
            dates=pd.DatetimeIndex(missing_dates),
            sample_time_utc=sample_time_utc,
            eth_blocks_csv=eth_blocks_csv,
            block_cache_csv=block_cache_csv,
            show_progress=show_progress,
        )
        call_data = _get_steth_by_wsteth_call_data(ONE_WSTETH_WEI)
        for idx, row in enumerate(blocks.itertuples(index=False), start=1):
            if show_progress:
                _print_progress("wstETH contract eth_call", idx, len(blocks))
            block_number = int(row.block_number)
            raw_rate = _eth_call_uint256(rpc_url, WSTETH_MAINNET_ADDRESS, call_data, block_number)
            rows.append({
                "date": row.date,
                "wsteth_rate": raw_rate / 1e18,
                "wsteth_rate_block": block_number,
                "wsteth_rate_source": WSTETH_RATE_SOURCE,
            })
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        if show_progress:
            print()

    fetched = pd.DataFrame(rows)
    combined = pd.concat([cache, fetched], ignore_index=True)
    if combined.empty:
        out = pd.DataFrame(columns=[
            "date", "wsteth_rate", "wsteth_rate_block", "wsteth_rate_source",
            "wsteth_implied_yield_1d", "wsteth_implied_yield_7d", "wsteth_implied_yield_30d",
        ])
    else:
        combined = combined.sort_values("date").drop_duplicates("date", keep="last")
        out = needed.merge(combined, on="date", how="left").dropna(subset=["wsteth_rate"])
        out["wsteth_rate_source"] = WSTETH_RATE_SOURCE
        out = add_implied_yields_from_rate(out)

    if rate_cache_csv is not None and not combined.empty:
        cache_path = Path(rate_cache_csv)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.sort_values("date").to_csv(cache_path, index=False)
    return out


def print_wsteth_rate_summary(df: pd.DataFrame) -> None:
    """Print source and sanity-check diagnostics for wstETH contract-rate yield."""
    print("ETH LST yield source summary:")
    if df.empty or "wsteth_rate" not in df:
        print("WARNING: wstETH contract exchange-rate fetch failed. No primary ETH LST-implied yield was generated. DeFiLlama APY or market price ratios are not substitutes for wstETH_implied_yield.")
        return
    rates = pd.to_numeric(df["wsteth_rate"], errors="coerce").dropna()
    y7 = pd.to_numeric(df.get("wsteth_implied_yield_7d", pd.Series(dtype=float)), errors="coerce").dropna()
    source = str(df["wsteth_rate_source"].dropna().iloc[0]) if "wsteth_rate_source" in df.columns and df["wsteth_rate_source"].dropna().any() else WSTETH_RATE_SOURCE
    method = "getStETHByWstETH(1e18) historical eth_call" if source == WSTETH_RATE_SOURCE else "user-supplied wstETH contract exchange-rate CSV"
    print("- primary ETH yield source: wstETH contract exchange rate")
    print(f"- method: {method}")
    print(f"- source: {source}")
    print(f"- wsteth_rate obs: {len(rates)}")
    if not rates.empty:
        print(f"- wsteth_rate start/end/min/max: {rates.iloc[0]:.8f} / {rates.iloc[-1]:.8f} / {rates.min():.8f} / {rates.max():.8f}")
        negative_steps = int((rates.diff().dropna() < 0).sum())
        if negative_steps > max(2, int(0.01 * max(len(rates) - 1, 1))):
            print(f"WARNING: wsteth_rate has {negative_steps} negative daily steps; verify this is not a market ratio or bad data.")
    if not y7.empty:
        print(f"- wsteth_implied_yield_7d mean: {y7.mean():.6f}")
        print(f"- wsteth_implied_yield_7d std/min/max: {y7.std():.6f} / {y7.min():.6f} / {y7.max():.6f}")
        bad = int(((y7 < 0) | (y7.abs() > 0.50)).sum())
        if bad > max(3, int(0.05 * len(y7))):
            print(f"WARNING: {bad} wsteth_implied_yield_7d observations are negative or above 50% annualized; inspect rate/block data.")

def fetch_lido_wsteth_share_rate_history(
    rpc_url: str,
    start_date: str,
    end_date: str | None = None,
    sample_time_utc: str = "00:00:00",
    sleep_seconds: float = 0.0,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Fetch historical wstETH protocol exchange rate from Ethereum RPC.

    The rate is `wstETH.stEthPerToken() / 1e18`, i.e. stETH per 1 wstETH.
    This is a protocol exchange rate, not the secondary-market wstETH/ETH price,
    so it avoids mixing staking accrual with market premium/discount.
    """
    start = pd.to_datetime(start_date, errors="raise").date()
    end = (pd.to_datetime(end_date, errors="raise") if end_date else pd.Timestamp.utcnow()).date()
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    selector = _function_selector("stEthPerToken()")
    latest_block = _latest_block_number(rpc_url)
    latest_ts = int(_get_block(rpc_url, latest_block)["timestamp"], 16)
    rows = []
    all_dates = pd.date_range(start=start, end=end, freq="D")
    total_days = len(all_dates)

    for idx, date in enumerate(all_dates, start=1):
        if show_progress:
            _print_progress("Lido RPC", idx, total_days)
        ts = int(pd.Timestamp(f"{date.date()} {sample_time_utc}", tz="UTC").timestamp())
        if ts > latest_ts:
            continue
        block_number = _find_block_at_or_before_timestamp(rpc_url, ts, latest_block=latest_block)
        block = _get_block(rpc_url, block_number)
        raw_rate = _eth_call_uint256(rpc_url, WSTETH_MAINNET_ADDRESS, selector, block_number)
        rows.append({
            "date": date.normalize(),
            "block_number": block_number,
            "block_timestamp_utc": pd.to_datetime(int(block["timestamp"], 16), unit="s", utc=True).tz_convert(None),
            "share_rate": raw_rate / 1e18,
            "exchange_rate": raw_rate / 1e18,
            "source": "lido_wsteth_stEthPerToken_rpc",
        })
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if show_progress and total_days > 0:
        print()

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "date", "block_number", "block_timestamp_utc", "share_rate", "exchange_rate",
            "daily_yield_decimal", "annualized_apr_decimal", "annualized_apr_pct", "stake_yield", "source"
        ])

    out = out.sort_values("date").drop_duplicates("date")
    out["daily_yield_decimal"] = out["share_rate"].pct_change()
    out["annualized_apr_decimal"] = out["daily_yield_decimal"] * 365.0
    out["annualized_apr_pct"] = out["annualized_apr_decimal"] * 100.0
    out["stake_yield"] = out["annualized_apr_decimal"]
    return out


def validate_lido_yield_panel(df: pd.DataFrame, extreme_apr_abs_threshold: float = 0.50) -> dict:
    """Return validation diagnostics for a Lido/wstETH daily yield panel."""
    diagnostics = {
        "rows": len(df),
        "duplicate_dates": int(df["date"].duplicated().sum()) if "date" in df else None,
        "missing_required_columns": [],
        "missing_daily_dates": [],
        "negative_daily_yield_rows": 0,
        "extreme_apr_rows": 0,
    }
    required = ["date", "share_rate", "daily_yield_decimal", "annualized_apr_decimal", "annualized_apr_pct"]
    diagnostics["missing_required_columns"] = [c for c in required if c not in df.columns]
    if diagnostics["missing_required_columns"]:
        return diagnostics

    dates = pd.to_datetime(df["date"]).sort_values()
    if not dates.empty:
        full_dates = pd.date_range(dates.min(), dates.max(), freq="D")
        missing = full_dates.difference(pd.DatetimeIndex(dates))
        diagnostics["missing_daily_dates"] = [d.date().isoformat() for d in missing]

    diagnostics["negative_daily_yield_rows"] = int((df["daily_yield_decimal"].dropna() < 0).sum())
    diagnostics["extreme_apr_rows"] = int((df["annualized_apr_decimal"].dropna().abs() > extreme_apr_abs_threshold).sum())
    return diagnostics


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
