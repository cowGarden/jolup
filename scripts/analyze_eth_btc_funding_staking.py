#!/usr/bin/env python3
"""Robust ETH-BTC funding spread vs ETH staking yield analysis.

Input files are expected at:
    data/processed/master_{exchange}_eth_btc_funding_staking_daily.csv

The script validates each master dataframe, runs OLS/HAC Newey-West regressions,
robustness checks, group tests, carry-gap diagnostics, pooled exchange tests, and
saves CSV tables plus matplotlib figures for thesis-ready replication.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg

DEFAULT_HAC_LAGS = [1, 3, 5, 7, 14]
STAKE_LAG_HORIZONS = [1, 3, 7, 14, 30]
FORWARD_HORIZONS = [1, 3, 7, 14]
MA_WINDOWS = [7, 14, 30]
BASE_REQUIRED = [
    "date",
    "spread",
    "eth_funding_ann",
    "btc_funding_ann",
    "stake_yield",
    "ret_eth_btc",
]
OPTIONAL_CONTROLS = [
    "rv_eth_btc",
    "rv_log_ratio",
    "rv_diff",
    "oi_eth_btc",
    "oi_ratio",
    "volume_ratio",
    "spread_lag1",
    "basis_spread",
    "premium_spread",
    "steth_discount",
]
ROLLING_WINDOWS = [180, 365]
REGRESSION_MIN_NOBS = 30
STAKE_YIELD_CANDIDATES = [
    "eth_yield_panel.csv",
    "lido_wsteth_share_rate.csv",
    "lido_staking_yield_daily.csv",
]



@dataclass
class ExchangeOutputs:
    exchange: str
    frames: dict[str, pd.DataFrame]
    warnings: list[str]


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def normalize_utc_daily(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.normalize()


def save_table(df: pd.DataFrame, path: Path, title: str | None = None, aliases: list[Path] | None = None) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    for alias in aliases or []:
        alias.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(alias, index=False)
    if title:
        print(f"\n=== {title} ===")
    print(f"Saved: {path}")
    for alias in aliases or []:
        print(f"Saved alias: {alias}")
    if not df.empty:
        print(df.to_string(index=False, max_rows=20))
    else:
        print("(empty table; required columns unavailable)")
    return df


def load_master(exchange: str, input_dir: Path) -> pd.DataFrame | None:
    path = input_dir / f"master_{exchange}_eth_btc_funding_staking_daily.csv"
    if not path.exists():
        print(f"[WARN] Missing {path}; skipping {exchange}.")
        return None
    df = pd.read_csv(path)
    print(f"Loaded {exchange}: {path} ({len(df):,} rows)")
    return df


def load_stake_yield_series(input_dir: Path) -> pd.DataFrame | None:
    """Load a canonical stake_yield time series from processed candidates."""
    for filename in STAKE_YIELD_CANDIDATES:
        path = input_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "stake_yield" not in df.columns and "annualized_apr_decimal" in df.columns:
            df["stake_yield"] = df["annualized_apr_decimal"]
        if not {"date", "stake_yield"}.issubset(df.columns):
            print(f"[WARN] {path} has no (date, stake_yield); skipping.")
            continue
        out = df[["date", "stake_yield"]].copy()
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.date.astype(str)
        out["stake_yield"] = pd.to_numeric(out["stake_yield"], errors="coerce")
        out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        print(f"[INFO] Loaded stake_yield fallback: {path} ({len(out):,} rows)")
        return out
    print("[WARN] No fallback stake_yield CSV found in input-dir.")
    return None


def attach_stake_yield_if_missing(df: pd.DataFrame, stake_yield: pd.DataFrame | None, exchange: str) -> pd.DataFrame:
    """Attach stake_yield by date when master is missing or sparse."""
    if stake_yield is None or "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.date.astype(str)
    if "stake_yield" not in out.columns:
        out = out.merge(stake_yield, on="date", how="left")
        filled = int(out["stake_yield"].notna().sum())
        print(f"[INFO] {exchange}: attached missing stake_yield by date (non-null rows={filled:,}).")
        return out
    before = int(pd.to_numeric(out["stake_yield"], errors="coerce").notna().sum())
    fallback = stake_yield.rename(columns={"stake_yield": "stake_yield_fallback"})
    out = out.merge(fallback, on="date", how="left")
    out["stake_yield"] = pd.to_numeric(out["stake_yield"], errors="coerce").fillna(out["stake_yield_fallback"])
    out = out.drop(columns=["stake_yield_fallback"])
    after = int(out["stake_yield"].notna().sum())
    if after > before:
        print(f"[INFO] {exchange}: filled missing stake_yield from fallback ({before:,} -> {after:,}).")
    return out


def validate_master_df(df: pd.DataFrame, exchange: str) -> tuple[pd.DataFrame, list[str]]:
    warnings_out: list[str] = []
    missing = [col for col in BASE_REQUIRED if col not in df.columns]
    if missing:
        raise ValueError(f"{exchange}: missing required columns: {missing}")
    if not any(col in df.columns for col in ["rv_eth_btc", "rv_log_ratio"]):
        raise ValueError(f"{exchange}: need at least one RV column: rv_eth_btc or rv_log_ratio")

    out = df.copy()
    out["date"] = normalize_utc_daily(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    duplicates = int(out["date"].duplicated().sum())
    if duplicates:
        msg = f"{exchange}: {duplicates} duplicate date rows detected"
        warnings_out.append(msg)
        print(f"[WARN] {msg}")
        print(out.loc[out["date"].duplicated(keep=False), ["date"]].head(10).to_string(index=False))

    is_sorted = out["date"].is_monotonic_increasing
    full_range = pd.date_range(out["date"].min(), out["date"].max(), freq="D", tz="UTC")
    missing_dates = full_range.difference(pd.DatetimeIndex(out["date"]))
    print(f"\n[{exchange}] date coverage: {out['date'].min().date()} to {out['date'].max().date()}, rows={len(out):,}")
    print(f"[{exchange}] sorted={is_sorted}, duplicate_dates={duplicates}, missing_daily_dates={len(missing_dates)}")
    if len(missing_dates):
        msg = f"{exchange}: missing daily dates, first examples: {[d.date().isoformat() for d in missing_dates[:10]]}"
        warnings_out.append(msg)
        print(f"[WARN] {msg}")

    out["spread_manual"] = out["eth_funding_ann"] - out["btc_funding_ann"]
    out["spread_diff"] = out["spread"] - out["spread_manual"]
    max_abs_diff = float(out["spread_diff"].abs().max(skipna=True))
    print(f"[{exchange}] max abs spread diff = {max_abs_diff:.6g}")
    if max_abs_diff > 1e-12:
        sample = out.loc[out["spread_diff"].abs() > 1e-12, [
            "date", "spread", "eth_funding_ann", "btc_funding_ann", "spread_manual", "spread_diff"
        ]].head(10)
        print(sample.to_string(index=False))
        raise ValueError(f"{exchange}: spread != eth_funding_ann - btc_funding_ann")

    for cols, label in [(["eth_funding_ann", "btc_funding_ann", "spread"], "annualized funding/spread"), (["stake_yield"], "staking yield")]:
        print(f"\n[{exchange}] {label} describe")
        print(out[cols].describe().to_string())

    scale_mask = (out["eth_funding_ann"].abs() > 5) | (out["btc_funding_ann"].abs() > 5) | (out["spread"].abs() > 5)
    if scale_mask.any():
        msg = f"{exchange}: {int(scale_mask.sum())} rows have abs annualized funding/spread > 5; check decimal annualization"
        warnings_out.append(msg)
        print(f"[WARN] {msg}")
        print(out.loc[scale_mask, ["date", "eth_funding_ann", "btc_funding_ann", "spread"]].head(10).to_string(index=False))
    if out["stake_yield"].median(skipna=True) > 1:
        msg = f"{exchange}: stake_yield median > 1; likely percent units, expected decimal APR"
        warnings_out.append(msg)
        print(f"[WARN] {msg}")
    stake_mask = (out["stake_yield"] < 0) | (out["stake_yield"] > 0.5)
    if stake_mask.any():
        msg = f"{exchange}: {int(stake_mask.sum())} rows have stake_yield outside [0, 0.5]"
        warnings_out.append(msg)
        print(f"[WARN] {msg}")
        print(out.loc[stake_mask, ["date", "stake_yield"]].head(10).to_string(index=False))

    available = [c for c in BASE_REQUIRED + OPTIONAL_CONTROLS if c in out.columns]
    print(f"\n[{exchange}] missing counts")
    print(out[available].isna().sum().to_string())
    oi_cols = [c for c in ["oi_eth_btc", "oi_ratio"] if c in out.columns]
    basis_cols = [c for c in ["basis_spread", "premium_spread"] if c in out.columns]
    print(f"[{exchange}] available OI columns: {oi_cols or 'none'}")
    print(f"[{exchange}] available basis/premium columns: {basis_cols or 'none'}")
    return out.drop(columns=["spread_manual", "spread_diff"]), warnings_out


def drop_regression_na(df: pd.DataFrame, cols: list[str], context: str) -> pd.DataFrame:
    before = len(df)
    out = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    print(f"[{context}] dropna rows: {before:,} -> {len(out):,} (dropped {before - len(out):,})")
    return out


def run_ols_hac(y: pd.Series, X: pd.DataFrame, maxlags: int = 5, add_const: bool = True):
    X_ = sm.add_constant(X, has_constant="add") if add_const else X
    return sm.OLS(y, X_, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags)})


def extract_reg_results(res, model_name: str, key_var: str = "stake_yield", hac_lags: int | None = None, controls: list[str] | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for term in res.params.index:
        rows.append({
            "model_name": model_name,
            "nobs": int(res.nobs),
            "r2": getattr(res, "rsquared", np.nan),
            "adj_r2": getattr(res, "rsquared_adj", np.nan),
            "hac_lags": hac_lags,
            "term": term,
            "coefficient": res.params.get(term, np.nan),
            "std_error": res.bse.get(term, np.nan),
            "t_stat": res.tvalues.get(term, np.nan),
            "pvalue": res.pvalues.get(term, np.nan),
            "key_var": key_var,
            "key_coef": res.params.get(key_var, np.nan),
            "key_se": res.bse.get(key_var, np.nan),
            "key_t": res.tvalues.get(key_var, np.nan),
            "key_pvalue": res.pvalues.get(key_var, np.nan),
            "const_coef": res.params.get("const", np.nan),
            "controls": ",".join(controls or []),
        })
    return rows


def term_value(table: pd.DataFrame, model_name: str, term: str, field: str, hac_lags: int = 5) -> float:
    if table.empty or "term" not in table.columns:
        return np.nan
    hit = table.loc[(table["model_name"] == model_name) & (table["term"] == term) & (table["hac_lags"] == hac_lags), field]
    if hit.empty:
        hit = table.loc[(table["model_name"] == model_name) & (table["term"] == term), field]
    return float(hit.iloc[0]) if not hit.empty else np.nan


def run_model_specs(df: pd.DataFrame, specs: list[tuple[str, list[str]]], y_col: str = "spread", hac_lags: Iterable[int] = DEFAULT_HAC_LAGS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, x_cols in specs:
        needed = [y_col] + x_cols
        reg = drop_regression_na(df, needed, model_name)
        if len(reg) <= len(x_cols) + 2:
            print(f"[WARN] {model_name}: too few observations; skipped")
            continue
        for lag in hac_lags:
            res = run_ols_hac(reg[y_col], reg[x_cols], maxlags=int(lag))
            rows.extend(extract_reg_results(res, model_name, key_var=x_cols[0], hac_lags=int(lag), controls=x_cols[1:]))
    return pd.DataFrame(rows)


def build_main_specs(df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    specs = [("M1_stake_only", ["stake_yield"]), ("M2_return", ["stake_yield", "ret_eth_btc"])]
    if "rv_eth_btc" in df.columns:
        specs.append(("M3_rv_eth_btc", ["stake_yield", "ret_eth_btc", "rv_eth_btc"]))
    if "rv_log_ratio" in df.columns:
        specs.append(("M4_rv_log_ratio", ["stake_yield", "ret_eth_btc", "rv_log_ratio"]))
        if "oi_eth_btc" in df.columns:
            specs.append(("M5_oi_eth_btc", ["stake_yield", "ret_eth_btc", "rv_log_ratio", "oi_eth_btc"]))
        if "oi_ratio" in df.columns:
            specs.append(("M6_oi_ratio", ["stake_yield", "ret_eth_btc", "rv_log_ratio", "oi_ratio"]))
        if "oi_eth_btc" in df.columns and "basis_spread" in df.columns:
            specs.append(("M7_basis_spread", ["stake_yield", "ret_eth_btc", "rv_log_ratio", "oi_eth_btc", "basis_spread"]))
    return specs


def primary_controls(df: pd.DataFrame) -> list[str]:
    controls = ["ret_eth_btc"]
    if "rv_log_ratio" in df.columns:
        controls.append("rv_log_ratio")
    elif "rv_eth_btc" in df.columns:
        controls.append("rv_eth_btc")
    return controls


def run_main_regressions(df: pd.DataFrame, exchange: str, results_dir: Path, hac_lags: list[int]) -> pd.DataFrame:
    out = run_model_specs(df, build_main_specs(df), y_col="spread", hac_lags=hac_lags)
    return save_table(out, results_dir / f"{exchange}_main_regression_hac_lag_sensitivity.csv", f"{exchange} main HAC regressions")


def run_lag_regressions(df: pd.DataFrame, exchange: str, results_dir: Path) -> pd.DataFrame:
    work = df.copy()
    controls = primary_controls(work)
    rows = []
    for h in STAKE_LAG_HORIZONS:
        work[f"stake_yield_l{h}"] = work["stake_yield"].shift(h)
        lagged_controls = []
        for col in controls:
            lag_col = f"{col}_l{h}"
            work[lag_col] = work[col].shift(h)
            lagged_controls.append(lag_col)
        specs = [
            (f"lag{h}_A_yield_lag_current_controls", [f"stake_yield_l{h}"] + controls),
            (f"lag{h}_B_yield_lag_lagged_controls", [f"stake_yield_l{h}"] + lagged_controls),
            (f"lag{h}_C_yield_lag_only", [f"stake_yield_l{h}"]),
        ]
        for name, x_cols in specs:
            res_df = run_model_specs(work, [(name, x_cols)], "spread", [max(5, h)])
            if not res_df.empty:
                res_df["horizon"] = h
                res_df["regression_type"] = name.split("_")[1]
                rows.append(res_df)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return save_table(out, results_dir / f"{exchange}_lagged_stake_yield_models.csv", f"{exchange} lag regressions", alias_for(exchange, ["lagged_stake_yield_models.csv"], results_dir))


def run_forward_spread_regressions(df: pd.DataFrame, exchange: str, results_dir: Path) -> pd.DataFrame:
    work = df.copy()
    controls = primary_controls(work)
    rows = []
    for h in FORWARD_HORIZONS:
        y_col = f"spread_fwd{h}"
        work[y_col] = work["spread"].shift(-h)
        model = f"forward_h{h}"
        res_df = run_model_specs(work, [(model, ["stake_yield"] + controls)], y_col, [max(5, h)])
        if not res_df.empty:
            res_df["horizon"] = h
            res_df["note"] = "descriptive overlapping-horizon regression; not a trading forecast"
            rows.append(res_df)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return save_table(out, results_dir / f"{exchange}_forward_spread_models.csv", f"{exchange} forward spread regressions", alias_for(exchange, ["forward_spread_models.csv"], results_dir))


def run_moving_average_regressions(df: pd.DataFrame, exchange: str, results_dir: Path) -> pd.DataFrame:
    work = df.copy()
    base_controls = primary_controls(work)
    if "oi_eth_btc" in work.columns:
        base_controls.append("oi_eth_btc")
    rows = []
    for window in MA_WINDOWS:
        y_col = f"spread_ma{window}"
        stake_col = f"stake_yield_ma{window}"
        work[y_col] = work["spread"].rolling(window, min_periods=window).mean()
        work[stake_col] = work["stake_yield"].rolling(window, min_periods=window).mean()
        ma_controls = []
        for col in base_controls:
            ma_col = f"{col}_ma{window}"
            work[ma_col] = work[col].rolling(window, min_periods=window).mean()
            ma_controls.append(ma_col)
        specs = [
            (f"MA{window}_stake_ret_rv", [stake_col] + ma_controls),
            (f"MA{window}_ret_rv", ma_controls),
        ]
        res_df = run_model_specs(work, specs, y_col, [min(window, 14)])
        if not res_df.empty:
            res_df["window"] = window
            rows.append(res_df)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return save_table(out, results_dir / f"{exchange}_moving_average_models.csv", f"{exchange} moving-average regressions", alias_for(exchange, ["moving_average_models.csv"], results_dir))


def winsorize_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)


def run_winsorized_regressions(df: pd.DataFrame, exchange: str, results_dir: Path, hac_lags: list[int]) -> pd.DataFrame:
    low, high = df["spread"].quantile([0.01, 0.99])
    extreme = df.loc[(df["spread"] <= low) | (df["spread"] >= high), ["date", "spread", "eth_funding_ann", "btc_funding_ann", "stake_yield"]].copy()
    save_table(extreme, results_dir / f"{exchange}_extreme_spread_dates.csv", f"{exchange} extreme spread dates", alias_for(exchange, ["extreme_spread_dates.csv"], results_dir))
    rows = []
    target_cols = [c for c in ["spread", "stake_yield", "ret_eth_btc", "rv_eth_btc", "rv_log_ratio", "oi_eth_btc", "oi_ratio"] if c in df.columns]
    robustness_specs = [("stake_only", ["stake_yield"]), ("stake_ret_rv", ["stake_yield"] + primary_controls(df)), ("ret_rv", primary_controls(df))]
    for lower, upper, label in [(0.01, 0.99, "winsor_1_99"), (0.05, 0.95, "winsor_5_95")]:
        work = df.copy()
        for col in target_cols:
            work[col] = winsorize_series(work[col], lower, upper)
        res_df = run_model_specs(work, robustness_specs, "spread", hac_lags)
        if not res_df.empty:
            res_df["outlier_method"] = label
            rows.append(res_df)
    trimmed = df.loc[~((df["spread"] <= low) | (df["spread"] >= high))].copy()
    res_df = run_model_specs(trimmed, robustness_specs, "spread", hac_lags)
    if not res_df.empty:
        res_df["outlier_method"] = "drop_abs_spread_top_bottom_1pct"
        rows.append(res_df)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return save_table(out, results_dir / f"{exchange}_outlier_robustness.csv", f"{exchange} winsorized/outlier robustness", alias_for(exchange, ["outlier_robustness.csv"], results_dir))


def run_robust_quantile_regressions(df: pd.DataFrame, exchange: str, results_dir: Path) -> pd.DataFrame:
    x_cols = ["stake_yield"] + primary_controls(df)
    reg = drop_regression_na(df, ["spread"] + x_cols, f"{exchange}_robust_quantile")
    rows = []
    if len(reg) > len(x_cols) + 2:
        X = sm.add_constant(reg[x_cols], has_constant="add")
        try:
            rlm = sm.RLM(reg["spread"], X, M=sm.robust.norms.HuberT()).fit()
            for term in rlm.params.index:
                rows.append({"model_name": "RLM_HuberT", "quantile": np.nan, "nobs": int(rlm.nobs), "term": term, "coefficient": rlm.params.get(term, np.nan), "std_error": rlm.bse.get(term, np.nan), "t_stat": rlm.tvalues.get(term, np.nan), "pvalue": rlm.pvalues.get(term, np.nan), "key_coef": rlm.params.get("stake_yield", np.nan), "key_se": rlm.bse.get("stake_yield", np.nan), "key_t": rlm.tvalues.get("stake_yield", np.nan), "key_pvalue": rlm.pvalues.get("stake_yield", np.nan), "controls": ",".join(x_cols[1:])})
        except Exception as exc:  # numerical convergence can fail on small/noisy samples
            warnings.warn(f"{exchange}: RLM failed: {exc}")
        for q in [0.5]:
            try:
                qr = QuantReg(reg["spread"], X).fit(q=q)
                for term in qr.params.index:
                    rows.append({"model_name": "QuantReg", "quantile": q, "nobs": int(qr.nobs), "term": term, "coefficient": qr.params.get(term, np.nan), "std_error": qr.bse.get(term, np.nan), "t_stat": qr.tvalues.get(term, np.nan), "pvalue": qr.pvalues.get(term, np.nan), "key_coef": qr.params.get("stake_yield", np.nan), "key_se": qr.bse.get("stake_yield", np.nan), "key_t": qr.tvalues.get("stake_yield", np.nan), "key_pvalue": qr.pvalues.get("stake_yield", np.nan), "controls": ",".join(x_cols[1:])})
            except Exception as exc:
                warnings.warn(f"{exchange}: QuantReg q={q} failed: {exc}")
    out = pd.DataFrame(rows)
    return save_table(out, results_dir / f"{exchange}_robust_regression_results.csv", f"{exchange} robust and quantile regressions", alias_for(exchange, ["robust_regression_results.csv"], results_dir))


def run_high_low_tests(df: pd.DataFrame, exchange: str, results_dir: Path) -> pd.DataFrame:
    reg = drop_regression_na(df, ["stake_yield", "spread", "ret_eth_btc"], f"{exchange}_high_low")
    rows = []
    if not reg.empty:
        median = reg["stake_yield"].median()
        high_df = reg.loc[reg["stake_yield"] > median]
        low_df = reg.loc[reg["stake_yield"] <= median]
        high = high_df["spread"]
        low = low_df["spread"]
        t_stat, t_p = stats.ttest_ind(high, low, equal_var=False, nan_policy="omit")
        try:
            u_stat, u_p = stats.mannwhitneyu(high, low, alternative="two-sided")
        except ValueError:
            u_stat, u_p = np.nan, np.nan
        rows.extend([
            {"test": "median_split_low", "group": "low", "n": len(low), "mean_spread": low.mean(), "median_spread": low.median(), "mean_ret_eth_btc": low_df["ret_eth_btc"].mean(), "difference_high_minus_low": np.nan, "stat": np.nan, "pvalue": np.nan},
            {"test": "median_split_high", "group": "high", "n": len(high), "mean_spread": high.mean(), "median_spread": high.median(), "mean_ret_eth_btc": high_df["ret_eth_btc"].mean(), "difference_high_minus_low": high.mean() - low.mean(), "stat": np.nan, "pvalue": np.nan},
            {"test": "welch_t_high_minus_low", "group": "high-low", "n": len(high) + len(low), "mean_spread": np.nan, "median_spread": np.nan, "mean_ret_eth_btc": np.nan, "difference_high_minus_low": high.mean() - low.mean(), "stat": t_stat, "pvalue": t_p},
            {"test": "mann_whitney_high_vs_low", "group": "high-low", "n": len(high) + len(low), "mean_spread": np.nan, "median_spread": np.nan, "mean_ret_eth_btc": np.nan, "difference_high_minus_low": high.mean() - low.mean(), "stat": u_stat, "pvalue": u_p},
        ])
        tercile = pd.qcut(reg["stake_yield"], 3, labels=["low", "mid", "high"], duplicates="drop")
        for group, sub in reg.groupby(tercile, observed=True):
            rows.append({"test": "tercile_split", "group": str(group), "n": len(sub), "mean_spread": sub["spread"].mean(), "median_spread": sub["spread"].median(), "difference_high_minus_low": np.nan, "stat": np.nan, "pvalue": np.nan})
    out = pd.DataFrame(rows)
    return save_table(out, results_dir / f"{exchange}_high_low_staking_yield_tests.csv", f"{exchange} high/low staking tests", alias_for(exchange, ["high_low_staking_yield_tests.csv"], results_dir))


def run_quintile_analysis(df: pd.DataFrame, exchange: str, results_dir: Path, figures_dir: Path) -> pd.DataFrame:
    rv = "rv_log_ratio" if "rv_log_ratio" in df.columns else "rv_eth_btc"
    reg = drop_regression_na(df, ["stake_yield", "spread", "ret_eth_btc", rv, "eth_funding_ann", "btc_funding_ann"], f"{exchange}_quintile")
    rows = []
    if reg["stake_yield"].nunique() >= 5:
        reg = reg.copy()
        reg["quintile"] = pd.qcut(reg["stake_yield"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        grouped = reg.groupby("quintile", observed=True)
        summary = grouped.agg(
            n=("spread", "size"),
            mean_stake_yield=("stake_yield", "mean"),
            mean_spread=("spread", "mean"),
            median_spread=("spread", "median"),
            mean_eth_funding_ann=("eth_funding_ann", "mean"),
            mean_btc_funding_ann=("btc_funding_ann", "mean"),
            mean_ret_eth_btc=("ret_eth_btc", "mean"),
            mean_rv_control=(rv, "mean"),
        ).reset_index()
        rows = summary.to_dict("records")
        q1 = reg.loc[reg["quintile"].astype(str) == "1", "spread"]
        q5 = reg.loc[reg["quintile"].astype(str) == "5", "spread"]
        t_stat, t_p = stats.ttest_ind(q5, q1, equal_var=False, nan_policy="omit")
        spear = stats.spearmanr(summary["quintile"].astype(float), summary["mean_spread"], nan_policy="omit")
        rows.append({"quintile": "Q5-Q1_test", "n": len(q5) + len(q1), "mean_stake_yield": np.nan, "mean_spread": q5.mean() - q1.mean(), "median_spread": np.nan, "mean_eth_funding_ann": np.nan, "mean_btc_funding_ann": np.nan, "mean_ret_eth_btc": np.nan, "mean_rv_control": np.nan, "welch_t": t_stat, "welch_pvalue": t_p, "spearman_rho_quintile_mean_spread": spear.statistic, "spearman_pvalue": spear.pvalue})
        fig, ax = plt.subplots(figsize=(8, 5))
        summary.plot.bar(x="quintile", y="mean_spread", ax=ax, legend=False, color="#4C78A8")
        ax.set_title(f"{exchange}: Mean ETH-BTC Funding Spread by Staking-Yield Quintile")
        ax.set_xlabel("Staking-yield quintile")
        ax.set_ylabel("Mean annualized spread (decimal)")
        fig.tight_layout()
        save_figure_with_aliases(fig, figures_dir / f"{exchange}_quintile_mean_spread.png", figure_alias_for(exchange, ["quintile_mean_spread.png"], figures_dir))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        summary.plot.bar(x="quintile", y="mean_ret_eth_btc", ax=ax, legend=False, color="#F58518")
        ax.set_title(f"{exchange}: Mean ETH/BTC Return by Staking-Yield Quintile")
        ax.set_xlabel("Staking-yield quintile")
        ax.set_ylabel("Mean ETH/BTC return")
        fig.tight_layout()
        save_figure_with_aliases(fig, figures_dir / f"{exchange}_quintile_mean_ret_eth_btc.png", figure_alias_for(exchange, ["quintile_mean_ret_eth_btc.png"], figures_dir))
        plt.close(fig)
    else:
        print(f"[WARN] {exchange}: not enough unique staking-yield values for quintiles")
    out = pd.DataFrame(rows)
    return save_table(out, results_dir / f"{exchange}_quintile_analysis_stake_yield.csv", f"{exchange} quintile analysis", alias_for(exchange, ["quintile_analysis_stake_yield.csv"], results_dir))


def run_carry_gap_analysis(df: pd.DataFrame, exchange: str, results_dir: Path, figures_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["carry_gap"] = work["spread"] + work["stake_yield"]
    work["carry_gap_l1"] = work["carry_gap"].shift(1)
    work["delta_carry_gap_fwd1"] = work["carry_gap"].shift(-1) - work["carry_gap"]
    work["carry_gap_ma7"] = work["carry_gap"].rolling(7, min_periods=7).mean()
    work["carry_gap_ma30"] = work["carry_gap"].rolling(30, min_periods=30).mean()
    cg = work["carry_gap"].dropna()
    t_stat, two_sided_p = stats.ttest_1samp(cg, 0, nan_policy="omit") if len(cg) else (np.nan, np.nan)
    one_sided_p = two_sided_p / 2 if t_stat > 0 else 1 - two_sided_p / 2
    desc = cg.describe()
    cg_ret = work[["carry_gap", "ret_eth_btc"]].dropna()
    pos_ret = cg_ret.loc[cg_ret["carry_gap"] > 0, "ret_eth_btc"]
    neg_ret = cg_ret.loc[cg_ret["carry_gap"] <= 0, "ret_eth_btc"]
    ret_t, ret_p = stats.ttest_ind(pos_ret, neg_ret, equal_var=False, nan_policy="omit") if len(pos_ret) and len(neg_ret) else (np.nan, np.nan)
    summary = pd.DataFrame([{
        "n": int(desc.get("count", 0)), "mean": desc.get("mean", np.nan), "std": desc.get("std", np.nan),
        "min": desc.get("min", np.nan), "p25": desc.get("25%", np.nan), "median": desc.get("50%", np.nan),
        "p75": desc.get("75%", np.nan), "max": desc.get("max", np.nan), "positive_ratio": float((cg > 0).mean()) if len(cg) else np.nan,
        "one_sample_t_mean_gt_0": t_stat, "one_sided_pvalue_mean_gt_0": one_sided_p,
        "corr_carry_gap_ret_eth_btc": cg_ret["carry_gap"].corr(cg_ret["ret_eth_btc"]) if len(cg_ret) else np.nan,
        "mean_ret_when_carry_gap_positive": pos_ret.mean(), "mean_ret_when_carry_gap_nonpositive": neg_ret.mean(),
        "ret_diff_positive_minus_nonpositive": pos_ret.mean() - neg_ret.mean(), "ret_diff_welch_t": ret_t, "ret_diff_welch_pvalue": ret_p,
        "interpretation_note": "carry_gap is not a realized strategy return; fees, slippage, LST depeg, custody/exchange, margin and liquidation risks are excluded",
    }])
    rows = []
    ar = run_model_specs(work.rename(columns={"carry_gap": "carry_gap_t"}), [("carry_gap_AR1", ["carry_gap_l1"])], "carry_gap_t", [5])
    if not ar.empty:
        rows.append(ar.rename(columns={"key_coef": "coef", "key_pvalue": "pvalue"}))
    mr = run_model_specs(work, [("carry_gap_mean_reversion", ["carry_gap"])], "delta_carry_gap_fwd1", [5])
    if not mr.empty:
        rows.append(mr.rename(columns={"key_coef": "coef", "key_pvalue": "pvalue"}))
    meanrev = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(work["date"], work["carry_gap"], label="carry_gap", alpha=0.5)
    ax.plot(work["date"], work["carry_gap_ma7"], label="MA7")
    ax.plot(work["date"], work["carry_gap_ma30"], label="MA30")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{exchange}: Carry Gap = Funding Spread + Staking Yield")
    ax.set_ylabel("Annualized decimal")
    ax.legend()
    fig.tight_layout()
    save_figure_with_aliases(fig, figures_dir / f"{exchange}_carry_gap_timeseries.png", figure_alias_for(exchange, ["carry_gap_timeseries.png"], figures_dir))
    plt.close(fig)
    return (
        save_table(summary, results_dir / f"{exchange}_carry_gap_summary.csv", f"{exchange} carry gap summary", alias_for(exchange, ["carry_gap_summary.csv"], results_dir)),
        save_table(meanrev, results_dir / f"{exchange}_carry_gap_mean_reversion.csv", f"{exchange} carry gap mean reversion", alias_for(exchange, ["carry_gap_mean_reversion.csv"], results_dir)),
    )


def run_rv_proxy_comparison(df: pd.DataFrame, exchange: str, results_dir: Path) -> pd.DataFrame:
    work = df.copy()
    if "rv_diff" not in work.columns and {"rv_eth_btc", "rv_log_ratio"}.issubset(work.columns):
        work["rv_diff"] = work["rv_log_ratio"] - work["rv_eth_btc"]
    specs = []
    if "rv_eth_btc" in work.columns:
        specs.append(("RV_A_abs_return_diff", ["stake_yield", "ret_eth_btc", "rv_eth_btc"]))
    if "rv_log_ratio" in work.columns:
        specs.append(("RV_B_log_ratio", ["stake_yield", "ret_eth_btc", "rv_log_ratio"]))
    if "rv_diff" in work.columns:
        specs.append(("RV_C_rv_diff", ["stake_yield", "ret_eth_btc", "rv_diff"]))
    out = run_model_specs(work, specs, "spread", [5]) if specs else pd.DataFrame()
    return save_table(out, results_dir / f"{exchange}_rv_proxy_comparison.csv", f"{exchange} RV proxy comparison")


def run_oi_control_comparison(df: pd.DataFrame, exchange: str, results_dir: Path) -> pd.DataFrame:
    rv = "rv_log_ratio" if "rv_log_ratio" in df.columns else "rv_eth_btc"
    specs = [("OI_no_oi", ["stake_yield", "ret_eth_btc", rv])]
    if "oi_eth_btc" in df.columns:
        specs.append(("OI_with_oi_eth_btc", ["stake_yield", "ret_eth_btc", rv, "oi_eth_btc"]))
    if "oi_ratio" in df.columns:
        specs.append(("OI_with_oi_ratio", ["stake_yield", "ret_eth_btc", rv, "oi_ratio"]))
    out = run_model_specs(df, specs, "spread", [5])
    return save_table(out, results_dir / f"{exchange}_oi_control_comparison.csv", f"{exchange} OI control comparison")


def run_basis_control_comparison(df: pd.DataFrame, exchange: str, results_dir: Path) -> pd.DataFrame:
    rv = "rv_log_ratio" if "rv_log_ratio" in df.columns else "rv_eth_btc"
    specs = []
    for col in ["basis_spread", "premium_spread"]:
        if col in df.columns:
            controls = ["stake_yield", "ret_eth_btc", rv]
            if "oi_eth_btc" in df.columns:
                controls.append("oi_eth_btc")
            controls.append(col)
            specs.append((f"strict_{col}_control", controls))
    out = run_model_specs(df, specs, "spread", [5]) if specs else pd.DataFrame()
    return save_table(out, results_dir / f"{exchange}_basis_control_comparison.csv", f"{exchange} basis/premium control comparison")


def make_core_figures(df: pd.DataFrame, exchange: str, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.sort_values("date").copy()
    reg = drop_regression_na(plot_df, ["stake_yield", "spread"], f"{exchange}_figures_scatter")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(reg["stake_yield"], reg["spread"], alpha=0.35, s=18)
    if len(reg) > 1:
        coef = np.polyfit(reg["stake_yield"], reg["spread"], 1)
        xs = np.linspace(reg["stake_yield"].min(), reg["stake_yield"].max(), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="red", linewidth=1.5)
    ax.set_title(f"{exchange}: Staking Yield vs ETH-BTC Funding Spread")
    ax.set_xlabel("Staking yield (decimal APR)")
    ax.set_ylabel("ETH-BTC annualized funding spread")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{exchange}_stake_yield_vs_spread_scatter.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(plot_df["date"], plot_df["spread"], label="spread", color="#4C78A8")
    ax1.set_ylabel("Funding spread", color="#4C78A8")
    ax2 = ax1.twinx()
    ax2.plot(plot_df["date"], plot_df["stake_yield"], label="stake_yield", color="#F58518", alpha=0.8)
    ax2.set_ylabel("Staking yield", color="#F58518")
    ax1.set_title(f"{exchange}: Spread and Staking Yield")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{exchange}_spread_and_stake_yield_timeseries.png", dpi=160)
    plt.close(fig)

    roll = plot_df[["date", "stake_yield", "spread"]].copy()
    roll["rolling_corr_30"] = roll["stake_yield"].rolling(30, min_periods=20).corr(roll["spread"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(roll["date"], roll["rolling_corr_30"], color="#54A24B")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{exchange}: 30-day Rolling Correlation, Staking Yield vs Spread")
    ax.set_ylabel("Correlation")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{exchange}_rolling_corr_stake_spread.png", dpi=160)
    plt.close(fig)



def alias_for(exchange: str, aliases: list[str], results_dir: Path) -> list[Path]:
    """Write thesis-requested unprefixed filenames for the primary exchange only."""
    return [results_dir / name for name in aliases] if exchange == "binance" else []


def figure_alias_for(exchange: str, aliases: list[str], figures_dir: Path) -> list[Path]:
    return [figures_dir / name for name in aliases] if exchange == "binance" else []


def save_figure_with_aliases(fig, path: Path, aliases: list[Path] | None = None, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    for alias in aliases or []:
        alias.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(alias, dpi=dpi)


def run_data_audit(df: pd.DataFrame, exchange: str, results_dir: Path, figures_dir: Path) -> pd.DataFrame:
    work = df.sort_values("date").copy()
    year_counts = work.groupby(work["date"].dt.year).size().rename("nobs").reset_index().rename(columns={"date": "year"})
    year_counts["exchange"] = exchange
    sy = work["stake_yield"].dropna()
    same_run = work["stake_yield"].eq(work["stake_yield"].shift()).astype(int)
    run_id = same_run.eq(0).cumsum()
    runs = work.groupby(run_id).agg(start_date=("date", "min"), end_date=("date", "max"), value=("stake_yield", "first"), length=("stake_yield", "size"))
    long_runs = runs.loc[runs["length"] >= 7].copy()
    audit = pd.DataFrame([{
        "exchange": exchange,
        "start_date": work["date"].min(),
        "end_date": work["date"].max(),
        "nobs": len(work),
        "duplicate_dates": int(work["date"].duplicated().sum()),
        "missing_dates": int(len(pd.date_range(work["date"].min(), work["date"].max(), freq="D", tz="UTC").difference(pd.DatetimeIndex(work["date"])))),
        "stake_yield_count": int(sy.count()),
        "stake_yield_mean": sy.mean(),
        "stake_yield_std": sy.std(),
        "stake_yield_min": sy.min(),
        "stake_yield_median": sy.median(),
        "stake_yield_max": sy.max(),
        "stake_yield_unique": int(sy.nunique()),
        "stake_yield_ge7day_same_value_runs": int(len(long_runs)),
        "stake_yield_max_same_value_run": int(runs["length"].max()) if not runs.empty else 0,
    }])
    save_table(year_counts, results_dir / f"{exchange}_yearly_observation_counts.csv")
    save_table(long_runs.reset_index(drop=True).head(200), results_dir / f"{exchange}_stake_yield_same_value_runs.csv")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(work["date"], work["stake_yield"], color="#F58518")
    ax.set_title(f"{exchange}: Staking Yield Time Series")
    ax.set_ylabel("Decimal APR")
    fig.tight_layout()
    save_figure_with_aliases(fig, figures_dir / f"{exchange}_stake_yield_timeseries.png", figure_alias_for(exchange, ["stake_yield_timeseries.png"], figures_dir))
    plt.close(fig)
    return save_table(audit, results_dir / f"{exchange}_data_audit_summary.csv")


def baseline_specs() -> list[tuple[str, list[str]]]:
    return [
        ("Model_A_stake_yield", ["stake_yield"]),
        ("Model_B_stake_ret_rv", ["stake_yield", "ret_eth_btc", "rv_eth_btc"]),
        ("Model_C_ret_only", ["ret_eth_btc"]),
        ("Model_D_ret_rv", ["ret_eth_btc", "rv_eth_btc"]),
        ("Model_E_stake_ret", ["stake_yield", "ret_eth_btc"]),
    ]


def run_requested_baseline_models(df: pd.DataFrame, exchange: str, results_dir: Path, hac_lags: list[int]) -> pd.DataFrame:
    specs = [(name, cols) for name, cols in baseline_specs() if all(c in df.columns for c in cols)]
    out = run_model_specs(df, specs, "spread", hac_lags)
    return save_table(out, results_dir / f"{exchange}_full_sample_baseline_models.csv", f"{exchange} requested full-sample baseline models", alias_for(exchange, ["full_sample_baseline_models.csv"], results_dir))


def run_period_regressions(df: pd.DataFrame, exchange: str, results_dir: Path, period: str) -> pd.DataFrame:
    work = df.copy()
    if period == "year":
        work["period"] = work["date"].dt.year.astype(str)
        filename = f"{exchange}_subsample_yearly_regressions.csv"
        aliases = ["subsample_yearly_regressions.csv"]
    else:
        half = np.where(work["date"].dt.month <= 6, "H1", "H2")
        work["period"] = work["date"].dt.year.astype(str) + half
        filename = f"{exchange}_subsample_halfyear_regressions.csv"
        aliases = ["subsample_halfyear_regressions.csv"]
    specs = [("stake_only", ["stake_yield"]), ("stake_ret_rv", ["stake_yield"] + primary_controls(work)), ("ret_rv", primary_controls(work))]
    rows = []
    for label, sub in work.groupby("period", sort=True):
        for model_name, x_cols in specs:
            if not all(c in sub.columns for c in x_cols):
                continue
            reg = drop_regression_na(sub, ["spread"] + x_cols, f"{exchange}_{period}_{label}_{model_name}")
            if len(reg) < max(REGRESSION_MIN_NOBS, len(x_cols) + 3):
                continue
            res = run_ols_hac(reg["spread"], reg[x_cols], maxlags=5)
            for row in extract_reg_results(res, model_name, x_cols[0], 5, x_cols[1:]):
                row[period] = label
                row["period"] = label
                rows.append(row)
    out = pd.DataFrame(rows)
    return save_table(out, results_dir / filename, f"{exchange} {period} subsample regressions", alias_for(exchange, aliases, results_dir))


def rolling_regression(df: pd.DataFrame, exchange: str, results_dir: Path, figures_dir: Path, window: int) -> pd.DataFrame:
    work = df.sort_values("date").reset_index(drop=True).copy()
    x_cols = ["stake_yield"] + primary_controls(work)
    rows = []
    for end in range(window - 1, len(work)):
        sub = work.iloc[end - window + 1:end + 1]
        reg = sub[["date", "spread"] + x_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(reg) < max(REGRESSION_MIN_NOBS, len(x_cols) + 3):
            continue
        res = run_ols_hac(reg["spread"], reg[x_cols], maxlags=5)
        rows.append({
            "start_date": reg["date"].min(), "end_date": reg["date"].max(), "nobs": int(res.nobs), "r2": res.rsquared,
            "stake_yield_coef": res.params.get("stake_yield", np.nan), "stake_yield_se": res.bse.get("stake_yield", np.nan), "stake_yield_t": res.tvalues.get("stake_yield", np.nan), "stake_yield_pvalue": res.pvalues.get("stake_yield", np.nan),
            "ret_eth_btc_coef": res.params.get("ret_eth_btc", np.nan), "ret_eth_btc_se": res.bse.get("ret_eth_btc", np.nan), "ret_eth_btc_t": res.tvalues.get("ret_eth_btc", np.nan), "ret_eth_btc_pvalue": res.pvalues.get("ret_eth_btc", np.nan),
        })
    out = pd.DataFrame(rows)
    save_table(out, results_dir / f"{exchange}_rolling_{window}d_coefficients.csv", f"{exchange} rolling {window}d coefficients", alias_for(exchange, [f"rolling_{window}d_coefficients.csv"], results_dir))
    return out


def plot_rolling_coefficients(rolls: dict[int, pd.DataFrame], exchange: str, figures_dir: Path, term: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for window, table in rolls.items():
        if table.empty:
            continue
        coef_col = f"{term}_coef"
        se_col = f"{term}_se"
        dates = pd.to_datetime(table["end_date"])
        ax.plot(dates, table[coef_col], label=f"{window}d")
        if se_col in table:
            ax.fill_between(dates, table[coef_col] - 1.96 * table[se_col], table[coef_col] + 1.96 * table[se_col], alpha=0.12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{exchange}: Rolling {term} coefficient")
    ax.legend()
    fig.tight_layout()
    alias_name = "rolling_stake_yield_coef.png" if term == "stake_yield" else "rolling_ret_eth_btc_coef.png"
    save_figure_with_aliases(fig, figures_dir / f"{exchange}_{alias_name}", figure_alias_for(exchange, [alias_name], figures_dir))
    plt.close(fig)


def add_regime_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    work = df.sort_values("date").copy()
    rv = "rv_log_ratio" if "rv_log_ratio" in work.columns else "rv_eth_btc"
    work["momentum_7d"] = work["ret_eth_btc"].rolling(7, min_periods=5).sum()
    work["high_momentum"] = (work["momentum_7d"] > work["momentum_7d"].median()).astype(int)
    work["high_abs_spread"] = (work["spread"].abs() > work["spread"].abs().median()).astype(int)
    work["high_volatility"] = (work[rv] > work[rv].median()).astype(int)
    work["momentum_30d"] = work["ret_eth_btc"].rolling(30, min_periods=20).sum()
    work["bull"] = (work["momentum_30d"] > work["momentum_30d"].median()).astype(int)
    return work, rv


def run_regime_analysis(df: pd.DataFrame, exchange: str, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    work, rv = add_regime_columns(df)
    regimes = ["high_momentum", "high_abs_spread", "high_volatility", "bull"]
    specs = [("stake_only", ["stake_yield"]), ("stake_ret_rv", ["stake_yield", "ret_eth_btc", rv]), ("ret_rv", ["ret_eth_btc", rv])]
    rows = []
    for regime in regimes:
        for value, sub in work.dropna(subset=[regime]).groupby(regime):
            for model_name, x_cols in specs:
                reg = drop_regression_na(sub, ["spread"] + x_cols, f"{exchange}_{regime}_{value}_{model_name}")
                if len(reg) < max(REGRESSION_MIN_NOBS, len(x_cols) + 3):
                    continue
                res = run_ols_hac(reg["spread"], reg[x_cols], 5)
                for row in extract_reg_results(res, model_name, x_cols[0], 5, x_cols[1:]):
                    row.update({"regime": regime, "regime_value": int(value), "rv_control": rv})
                    rows.append(row)
    reg_out = pd.DataFrame(rows)
    save_table(reg_out, results_dir / f"{exchange}_regime_regressions.csv", f"{exchange} regime regressions", alias_for(exchange, ["regime_regressions.csv"], results_dir))

    int_rows = []
    for regime in regimes:
        temp = work.copy()
        xint = f"stake_yield_x_{regime}"
        temp[xint] = temp["stake_yield"] * temp[regime]
        x_cols = ["stake_yield", regime, xint, "ret_eth_btc", rv]
        reg = drop_regression_na(temp, ["spread"] + x_cols, f"{exchange}_{regime}_interaction")
        if len(reg) < max(REGRESSION_MIN_NOBS, len(x_cols) + 3):
            continue
        res = run_ols_hac(reg["spread"], reg[x_cols], 5)
        for row in extract_reg_results(res, f"interaction_{regime}", "stake_yield", 5, x_cols[1:]):
            row.update({"regime": regime, "interaction_term": xint, "rv_control": rv})
            int_rows.append(row)
    int_out = pd.DataFrame(int_rows)
    save_table(int_out, results_dir / f"{exchange}_regime_interaction_models.csv", f"{exchange} regime interaction models", alias_for(exchange, ["regime_interaction_models.csv"], results_dir))
    return reg_out, int_out

def run_pooled_exchange_comparison(exchange_dfs: dict[str, pd.DataFrame], results_dir: Path) -> pd.DataFrame:
    if not {"binance", "bybit"}.issubset(exchange_dfs):
        print("[WARN] Need both Binance and Bybit for pooled comparison; skipping.")
        return pd.DataFrame()
    frames = []
    for exchange, df in exchange_dfs.items():
        temp = df.copy()
        temp["exchange"] = exchange
        frames.append(temp)
    pooled = pd.concat(frames, ignore_index=True)
    rv = "rv_log_ratio" if "rv_log_ratio" in pooled.columns else "rv_eth_btc"
    pooled["exchange_bybit"] = (pooled["exchange"] == "bybit").astype(int)
    pooled["stake_yield_x_bybit"] = pooled["stake_yield"] * pooled["exchange_bybit"]
    specs = [
        ("pooled_exchange_fe", ["stake_yield", "ret_eth_btc", rv, "exchange_bybit"]),
        ("pooled_exchange_interaction", ["stake_yield", "ret_eth_btc", rv, "exchange_bybit", "stake_yield_x_bybit"]),
    ]
    out = run_model_specs(pooled, specs, "spread", [5])
    return save_table(out, results_dir / "pooled_exchange_comparison.csv", "pooled exchange comparison")


def print_console_summary(outputs: dict[str, ExchangeOutputs], pooled: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("CONSOLE SUMMARY: ETH staking yield vs ETH-BTC funding spread")
    print("=" * 72)
    by_exchange_signs = {}
    for exchange, output in outputs.items():
        frames = output.frames
        main = frames.get("main", pd.DataFrame())
        print(f"\n[{exchange}]")
        if not main.empty:
            priority = ["M4_rv_log_ratio", "M3_rv_eth_btc", "M2_return", "M1_stake_only"]
            chosen_name = next((m for m in priority if ((main["model_name"] == m) & (main["hac_lags"] == 5)).any()), main.iloc[0]["model_name"])
            chosen = main.loc[(main["model_name"] == chosen_name) & (main["hac_lags"] == 5)].head(1)
            if chosen.empty:
                chosen = main.loc[main["model_name"] == chosen_name].head(1)
            row = chosen.iloc[0]
            by_exchange_signs[exchange] = np.sign(row["key_coef"])
            print(f"1. Main result ({chosen_name}, HAC=5): coef={row['key_coef']:.6g}, p={row['key_pvalue']:.4g}, R2={row['r2']:.4g}, n={int(row['nobs'])}")
            same_model = main.loc[main["model_name"] == chosen_name]
            print(f"2. Negative across HAC lags: {bool((same_model['key_coef'] < 0).all())}")
            if row["r2"] < 0.05:
                print("   [WARN] Low explanatory power: R-squared < 0.05.")
            if row["nobs"] < 250:
                print("   [WARN] Short sample: fewer than 250 observations.")
        lag = frames.get("lag", pd.DataFrame())
        ma = frames.get("moving_average", pd.DataFrame())
        highlow = frames.get("high_low", pd.DataFrame())
        carry = frames.get("carry_gap_summary", pd.DataFrame())
        print(f"3. Negative in lag regressions: {False if lag.empty else bool((lag['key_coef'] < 0).mean() >= 0.5)}")
        print(f"4. Negative in MA regressions: {False if ma.empty else bool((ma['key_coef'] < 0).all())}")
        if not highlow.empty and "welch_t_high_minus_low" in set(highlow["test"]):
            diff = highlow.loc[highlow["test"] == "welch_t_high_minus_low", "difference_high_minus_low"].iloc[0]
            print(f"5. High staking yield group lower spread: {bool(diff < 0)} (high-low={diff:.6g})")
        if not carry.empty:
            print(f"6. Carry gap positive on average: {bool(carry.iloc[0]['mean'] > 0)} (mean={carry.iloc[0]['mean']:.6g}, one-sided p={carry.iloc[0]['one_sided_pvalue_mean_gt_0']:.4g})")
        baseline = frames.get("baseline", pd.DataFrame())
        yearly = frames.get("yearly", pd.DataFrame())
        halfyear = frames.get("halfyear", pd.DataFrame())
        rolling_180 = frames.get("rolling_180", pd.DataFrame())
        rolling_365 = frames.get("rolling_365", pd.DataFrame())
        stake_p = term_value(baseline, "Model_B_stake_ret_rv", "stake_yield", "pvalue")
        stake_coef = term_value(baseline, "Model_B_stake_ret_rv", "stake_yield", "coefficient")
        ret_p = term_value(baseline, "Model_B_stake_ret_rv", "ret_eth_btc", "pvalue")
        ret_coef = term_value(baseline, "Model_B_stake_ret_rv", "ret_eth_btc", "coefficient")
        sig_periods = pd.concat([yearly, halfyear], ignore_index=True) if not yearly.empty or not halfyear.empty else pd.DataFrame()
        sig_stake_periods = sig_periods.loc[(sig_periods.get("term", pd.Series(dtype=object)) == "stake_yield") & (sig_periods.get("pvalue", pd.Series(dtype=float)) < 0.05), "period"].astype(str).unique().tolist() if not sig_periods.empty else []
        roll_tables = [t for t in [rolling_180, rolling_365] if not t.empty and "stake_yield_pvalue" in t]
        roll_sig_share = np.nan if not roll_tables else float(np.mean([t["stake_yield_pvalue"].lt(0.05).mean() for t in roll_tables]))
        ma_sig = False if ma.empty else bool((((ma.get("term") == "stake_yield") | (ma.get("term", pd.Series(dtype=object)).astype(str).str.startswith("stake_yield_ma"))) & (ma.get("pvalue", pd.Series(dtype=float)) < 0.05)).any())
        highlow_p = highlow.loc[highlow["test"] == "welch_t_high_minus_low", "pvalue"].iloc[0] if not highlow.empty and (highlow["test"] == "welch_t_high_minus_low").any() else np.nan
        conclusion = "D: 어떤 변수도 안정적으로 설명하지 못한다."
        if pd.notna(ret_p) and ret_p < 0.05 and (pd.isna(stake_p) or stake_p >= 0.05):
            conclusion = "C: 전체적으로 staking yield보다는 ETH/BTC 상대수익률, 즉 leverage demand/momentum이 funding spread를 더 잘 설명한다."
        elif pd.notna(stake_p) and stake_p < 0.05 and sig_stake_periods:
            conclusion = "B: staking yield 효과는 특정 regime/부분표본에서만 나타난다."
        elif pd.notna(stake_p) and stake_p < 0.05:
            conclusion = "A: ETH staking yield가 funding spread에 안정적으로 반영된다."
        print("\n논문 방향 자동 요약")
        print(f"1. 전체 샘플 stake_yield 유의성(Model B): coef={stake_coef:.6g}, p={stake_p:.4g}, significant={bool(pd.notna(stake_p) and stake_p < 0.05)}")
        print(f"2. 전체 샘플 ret_eth_btc 유의성(Model B): coef={ret_coef:.6g}, p={ret_p:.4g}, significant={bool(pd.notna(ret_p) and ret_p < 0.05)}")
        print(f"3. 연도/반기별 stake_yield 유의 구간(p<0.05): {sig_stake_periods or 'none'}")
        print(f"4. Rolling stake_yield 유의 window 평균 비중: {roll_sig_share if pd.notna(roll_sig_share) else 'n/a'}")
        print(f"5. Moving-average에서 stake_yield 유의성 회복: {ma_sig}")
        print(f"6. High/low staking yield spread 차이 Welch p-value: {highlow_p if pd.notna(highlow_p) else 'n/a'}")
        if not carry.empty:
            print(f"7. Carry gap 평균 양수 여부: {bool(carry.iloc[0]['mean'] > 0)} (mean={carry.iloc[0]['mean']:.6g})")
        print(f"8. 더 타당한 결론: {conclusion}")
        print(f"9. Column availability: OI={any(c in frames.get('source_columns', []) for c in ['oi_eth_btc','oi_ratio'])}, basis/premium={any(c in frames.get('source_columns', []) for c in ['basis_spread','premium_spread'])}")
        for msg in output.warnings:
            print(f"   [DATA WARN] {msg}")
    if {"binance", "bybit"}.issubset(by_exchange_signs):
        print(f"\n[pooled] Bybit confirms Binance direction: {bool(by_exchange_signs['binance'] == by_exchange_signs['bybit'])}")
    if not pooled.empty:
        print("[pooled] Results saved in pooled_exchange_comparison.csv")
    print("\nInterpretation warnings:")
    print("- Results are conditional correlations, not causal effects.")
    print("- Low R-squared means staking yield is only one of many drivers of funding spreads.")
    print("- Funding spreads are strongly affected by leverage demand, sentiment, volatility, microstructure, and exchange rules.")
    print("- carry_gap is not a realized strategy return and excludes transaction costs, slippage, LST depeg, exchange/custody risk, margin and liquidation risk.")


def run_exchange(exchange: str, raw_df: pd.DataFrame, results_dir: Path, figures_dir: Path, hac_lags: list[int]) -> tuple[pd.DataFrame, ExchangeOutputs]:
    df, warn = validate_master_df(raw_df, exchange)
    make_core_figures(df, exchange, figures_dir)
    frames: dict[str, pd.DataFrame] = {"source_columns": list(df.columns)}  # type: ignore[dict-item]
    frames["data_audit"] = run_data_audit(df, exchange, results_dir, figures_dir)
    frames["baseline"] = run_requested_baseline_models(df, exchange, results_dir, hac_lags)
    frames["main"] = run_main_regressions(df, exchange, results_dir, hac_lags)
    frames["yearly"] = run_period_regressions(df, exchange, results_dir, "year")
    frames["halfyear"] = run_period_regressions(df, exchange, results_dir, "halfyear")
    rolls = {window: rolling_regression(df, exchange, results_dir, figures_dir, window) for window in ROLLING_WINDOWS}
    frames["rolling_180"] = rolls.get(180, pd.DataFrame())
    frames["rolling_365"] = rolls.get(365, pd.DataFrame())
    plot_rolling_coefficients(rolls, exchange, figures_dir, "stake_yield")
    plot_rolling_coefficients(rolls, exchange, figures_dir, "ret_eth_btc")
    regime, regime_interaction = run_regime_analysis(df, exchange, results_dir)
    frames["regime"] = regime
    frames["regime_interaction"] = regime_interaction
    frames["lag"] = run_lag_regressions(df, exchange, results_dir)
    frames["forward"] = run_forward_spread_regressions(df, exchange, results_dir)
    frames["moving_average"] = run_moving_average_regressions(df, exchange, results_dir)
    frames["high_low"] = run_high_low_tests(df, exchange, results_dir)
    frames["quintile"] = run_quintile_analysis(df, exchange, results_dir, figures_dir)
    frames["winsorized"] = run_winsorized_regressions(df, exchange, results_dir, hac_lags)
    frames["robust_quantile"] = run_robust_quantile_regressions(df, exchange, results_dir)
    carry_summary, carry_mr = run_carry_gap_analysis(df, exchange, results_dir, figures_dir)
    frames["carry_gap_summary"] = carry_summary
    frames["carry_gap_mean_reversion"] = carry_mr
    frames["rv_proxy"] = run_rv_proxy_comparison(df, exchange, results_dir)
    frames["oi_control"] = run_oi_control_comparison(df, exchange, results_dir)
    frames["basis_control"] = run_basis_control_comparison(df, exchange, results_dir)
    return df, ExchangeOutputs(exchange=exchange, frames=frames, warnings=warn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETH-BTC funding spread vs staking yield robustness analysis.")
    parser.add_argument("--exchanges", nargs="+", default=["binance", "bybit"], help="Exchanges to analyze.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"), help="Directory containing master CSV files.")
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"), help="Directory for CSV outputs.")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"), help="Directory for PNG figures.")
    parser.add_argument("--hac-lags", nargs="+", type=int, default=DEFAULT_HAC_LAGS, help="HAC Newey-West maxlags for main robustness tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(args.results_dir, args.figures_dir)
    exchange_dfs: dict[str, pd.DataFrame] = {}
    outputs: dict[str, ExchangeOutputs] = {}
    stake_yield_series = load_stake_yield_series(args.input_dir)
    for exchange in args.exchanges:
        raw = load_master(exchange, args.input_dir)
        if raw is None:
            continue
        raw = attach_stake_yield_if_missing(raw, stake_yield_series, exchange)
        try:
            df, out = run_exchange(exchange, raw, args.results_dir, args.figures_dir, args.hac_lags)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {exchange}: analysis failed and will be skipped: {exc}")
            continue
        exchange_dfs[exchange] = df
        outputs[exchange] = out
    pooled = run_pooled_exchange_comparison(exchange_dfs, args.results_dir) if exchange_dfs else pd.DataFrame()
    print_console_summary(outputs, pooled)


if __name__ == "__main__":
    main()
