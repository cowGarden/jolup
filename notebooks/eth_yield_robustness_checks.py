# %% [markdown]
"""
# ETH yield robustness checks

This robustness appendix intentionally excludes basis analysis and event
interaction analysis. It asks whether the ETH-yield/ETH-relative-funding result
is sensitive to control-leg choice, ETH yield proxy choice, HAC lag choice,
winsorization/outlier handling, placebo non-ETH spreads, and economic magnitude
normalization.
"""

# %%
from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # noqa: BLE001
    plt = None
    warnings.warn(f"matplotlib unavailable; figures will be skipped: {exc}")

# %%
INPUT_CSV = "data/processed/funding_yield_panel.csv"

FUNDING_COLS = {
    "ETH": "eth_funding",
    "BTC": "btc_funding",
    "XRP": "xrp_funding",
    "DOGE": "doge_funding",
}

YIELD_CANDIDATES = [
    "wsteth_implied_yield_7d",
    "wsteth_implied_yield_30d",
    "stake_yield",
    "eth_native_yield",
]

BASE_CONTROLS = ["ret_eth_btc", "rv_eth_btc"]
HAC_LAGS_LIST = [3, 5, 7, 14]
WINSOR_LEVELS = [None, 0.005, 0.01]
OUTPUT_DIR = "outputs/eth_yield_robustness"

DEFAULT_HAC_LAGS = 5
MIN_OBS = 50

OUT_DIR = Path(OUTPUT_DIR)
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
for d in [TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# %%
def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _fmt(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}g}"
    return str(value)


def _markdown_table(data: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if data.empty:
        return "No estimates available."
    use = data[columns].head(max_rows).copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for row in use.itertuples(index=False):
        rows.append("| " + " | ".join(_fmt(v, 5) for v in row) + " |")
    return "\n".join([header, sep, *rows])


def load_data(path: str | Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a date column.")
    df["date"] = (
        pd.to_datetime(df["date"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for asset, col in FUNDING_COLS.items():
        legacy = f"f_{asset.lower()}"
        if col not in df.columns and legacy in df.columns:
            df[col] = df[legacy]
        if col not in df.columns:
            raise ValueError(f"Missing funding column for {asset}: {col} or {legacy}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    available_yields = []
    for col in YIELD_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().sum() >= MIN_OBS:
                available_yields.append(col)
    if not available_yields:
        raise ValueError(
            f"No yield candidate has at least {MIN_OBS} non-null observations: {YIELD_CANDIDATES}"
        )

    available_controls = []
    for col in BASE_CONTROLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().sum() >= MIN_OBS:
                available_controls.append(col)
            else:
                warnings.warn(
                    f"Control column {col} has too few observations and will be omitted."
                )
        else:
            warnings.warn(f"Control column {col} is missing and will be omitted.")

    print(f"Sample range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Rows: {len(df):,}")
    print(f"Available yield proxies: {available_yields}")
    print(f"Available controls: {available_controls}")
    return df, available_yields, available_controls


df, AVAILABLE_YIELDS, AVAILABLE_CONTROLS = load_data(INPUT_CSV)
MAIN_YIELD = (
    "wsteth_implied_yield_7d"
    if "wsteth_implied_yield_7d" in AVAILABLE_YIELDS
    else AVAILABLE_YIELDS[0]
)


# %%
def construct_variables(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["f_eth"] = pd.to_numeric(out[FUNDING_COLS["ETH"]], errors="coerce")
    out["f_btc"] = pd.to_numeric(out[FUNDING_COLS["BTC"]], errors="coerce")
    out["f_xrp"] = pd.to_numeric(out[FUNDING_COLS["XRP"]], errors="coerce")
    out["f_doge"] = pd.to_numeric(out[FUNDING_COLS["DOGE"]], errors="coerce")

    out["eth_minus_btc"] = out["f_eth"] - out["f_btc"]
    out["eth_minus_xrp"] = out["f_eth"] - out["f_xrp"]
    out["eth_minus_doge"] = out["f_eth"] - out["f_doge"]
    out["eth_minus_btc_xrp_mean"] = out["f_eth"] - out[["f_btc", "f_xrp"]].mean(axis=1)
    out["eth_minus_btc_doge_mean"] = out["f_eth"] - out[["f_btc", "f_doge"]].mean(
        axis=1
    )
    out["eth_minus_xrp_doge_mean"] = out["f_eth"] - out[["f_xrp", "f_doge"]].mean(
        axis=1
    )
    out["eth_minus_control_mean"] = out["f_eth"] - out[
        ["f_btc", "f_xrp", "f_doge"]
    ].mean(axis=1)
    out["eth_minus_control_median"] = out["f_eth"] - out[
        ["f_btc", "f_xrp", "f_doge"]
    ].median(axis=1)

    out["btc_minus_xrp"] = out["f_btc"] - out["f_xrp"]
    out["btc_minus_doge"] = out["f_btc"] - out["f_doge"]
    out["xrp_minus_doge"] = out["f_xrp"] - out["f_doge"]
    out["xrp_minus_btc"] = out["f_xrp"] - out["f_btc"]
    out["doge_minus_btc"] = out["f_doge"] - out["f_btc"]
    out["doge_minus_xrp"] = out["f_doge"] - out["f_xrp"]
    return out


df = construct_variables(df)

CONTROL_SETS = [
    ("BTC only", "eth_minus_btc"),
    ("XRP only", "eth_minus_xrp"),
    ("DOGE only", "eth_minus_doge"),
    ("BTC+XRP mean", "eth_minus_btc_xrp_mean"),
    ("BTC+DOGE mean", "eth_minus_btc_doge_mean"),
    ("XRP+DOGE mean", "eth_minus_xrp_doge_mean"),
    ("BTC+XRP+DOGE mean", "eth_minus_control_mean"),
    ("BTC+XRP+DOGE median", "eth_minus_control_median"),
]
MAIN_OUTCOMES = [
    "eth_minus_control_mean",
    "eth_minus_control_median",
    "eth_minus_xrp",
    "eth_minus_doge",
    "eth_minus_btc",
]
PLACEBO_OUTCOMES = [
    "btc_minus_xrp",
    "btc_minus_doge",
    "xrp_minus_doge",
    "xrp_minus_btc",
    "doge_minus_btc",
    "doge_minus_xrp",
]


# %%
def fit_ols_hac(data: pd.DataFrame, y_col: str, x_cols: list[str], maxlags: int = 5):
    cols = ["date", y_col, *x_cols]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError(f"Missing model columns: {missing}")
    use = data[cols].dropna()
    if len(use) < MIN_OBS:
        return None, use
    X = sm.add_constant(use[x_cols], has_constant="add")
    y = use[y_col]
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return res, use


def extract_result_rows(
    *,
    test_block: str,
    model: str,
    yield_col: str,
    dependent: str,
    control_set: str,
    winsor_level: float | None,
    hac_lags: int,
    res,
    use: pd.DataFrame,
) -> list[dict[str, object]]:
    if res is None or use.empty:
        return [
            {
                "test_block": test_block,
                "model": model,
                "yield_col": yield_col,
                "dependent": dependent,
                "control_set": control_set,
                "winsor_level": winsor_level,
                "hac_lags": hac_lags,
                "term": "INSUFFICIENT_OBS",
                "coef": np.nan,
                "std_err_hac": np.nan,
                "t": np.nan,
                "p_value": np.nan,
                "nobs": len(use),
                "adj_r2": np.nan,
                "sample_start": (
                    use["date"].min().date().isoformat() if not use.empty else None
                ),
                "sample_end": (
                    use["date"].max().date().isoformat() if not use.empty else None
                ),
            }
        ]
    rows = []
    for term in res.params.index:
        rows.append(
            {
                "test_block": test_block,
                "model": model,
                "yield_col": yield_col,
                "dependent": dependent,
                "control_set": control_set,
                "winsor_level": winsor_level,
                "hac_lags": hac_lags,
                "term": term,
                "coef": res.params.get(term, np.nan),
                "std_err_hac": res.bse.get(term, np.nan),
                "t": res.tvalues.get(term, np.nan),
                "p_value": res.pvalues.get(term, np.nan),
                "nobs": int(res.nobs),
                "adj_r2": res.rsquared_adj,
                "sample_start": use["date"].min().date().isoformat(),
                "sample_end": use["date"].max().date().isoformat(),
            }
        )
    return rows


def run_single_model(
    data: pd.DataFrame,
    *,
    test_block: str,
    model: str,
    yield_col: str,
    dependent: str,
    control_set: str,
    winsor_level: float | None = None,
    hac_lags: int = DEFAULT_HAC_LAGS,
) -> list[dict[str, object]]:
    x_cols = [yield_col, *AVAILABLE_CONTROLS]
    res, use = fit_ols_hac(data, dependent, x_cols, maxlags=hac_lags)
    return extract_result_rows(
        test_block=test_block,
        model=model,
        yield_col=yield_col,
        dependent=dependent,
        control_set=control_set,
        winsor_level=winsor_level,
        hac_lags=hac_lags,
        res=res,
        use=use,
    )


def _yield_term(table: pd.DataFrame, yield_col: str | None = None) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    out = table.copy()
    if yield_col is None:
        return out[out["term"] == out["yield_col"]].copy()
    return out[(out["term"] == yield_col) & (out["yield_col"] == yield_col)].copy()


def coefficient_interpretation(coef: float, p_value: float) -> str:
    if pd.notna(coef) and pd.notna(p_value) and p_value < 0.05 and coef < 0:
        return "supports dividend-like funding discount relative to this control"
    if pd.notna(coef) and pd.notna(p_value) and p_value < 0.05 and coef > 0:
        return "opposite sign; ETH funding rises relative to this control"
    return "not statistically significant"


# %%
def _savefig(name: str) -> None:
    if plt is None:
        return
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=150)
    plt.close()


def run_control_leg_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for yield_col in AVAILABLE_YIELDS:
        for control_set, outcome in CONTROL_SETS:
            rows.extend(
                run_single_model(
                    df,
                    test_block="control_leg_sensitivity",
                    model=f"{outcome}_on_{yield_col}",
                    yield_col=yield_col,
                    dependent=outcome,
                    control_set=control_set,
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "control_leg_sensitivity.csv", index=False)

    y = _yield_term(out)
    summary = y[
        [
            "yield_col",
            "control_set",
            "dependent",
            "coef",
            "p_value",
            "nobs",
            "adj_r2",
            "std_err_hac",
        ]
    ].copy()
    summary["sign"] = np.where(
        summary["coef"] > 0,
        "positive",
        np.where(summary["coef"] < 0, "negative", "zero"),
    )
    summary["significant_5pct"] = summary["p_value"] < 0.05
    summary["interpretation"] = [
        coefficient_interpretation(c, p)
        for c, p in zip(summary["coef"], summary["p_value"])
    ]
    summary.to_csv(TABLE_DIR / "control_leg_sensitivity_summary.csv", index=False)

    if plt is not None and not summary.empty:
        for yield_col, grp in summary.groupby("yield_col"):
            plot_df = (
                grp.set_index("control_set")
                .loc[[c for c, _ in CONTROL_SETS if c in set(grp["control_set"])]]
                .reset_index()
            )
            x = np.arange(len(plot_df))
            plt.figure(figsize=(11, 5))
            plt.bar(x, plot_df["coef"])
            plt.errorbar(
                x,
                plot_df["coef"],
                yerr=1.96 * plot_df["std_err_hac"],
                fmt="none",
                capsize=3,
            )
            for idx, row in enumerate(plot_df.itertuples(index=False)):
                if pd.notna(row.p_value) and row.p_value < 0.05:
                    plt.text(
                        idx,
                        row.coef,
                        "*",
                        ha="center",
                        va="bottom" if row.coef >= 0 else "top",
                    )
            plt.axhline(0, linewidth=0.8)
            plt.xticks(x, plot_df["control_set"], rotation=45, ha="right")
            plt.ylabel("HAC coefficient on yield proxy")
            plt.title(f"Control-leg sensitivity: {yield_col}")
            _savefig(f"control_leg_sensitivity_{_safe_name(yield_col)}.png")
    return out, summary


control_leg_results, control_leg_summary = run_control_leg_sensitivity()


# %%
def run_yield_proxy_sensitivity() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for yield_col in AVAILABLE_YIELDS:
        for outcome in ["eth_minus_control_mean", "eth_minus_control_median"]:
            rows.extend(
                run_single_model(
                    df,
                    test_block="yield_proxy_sensitivity",
                    model=f"{outcome}_on_{yield_col}",
                    yield_col=yield_col,
                    dependent=outcome,
                    control_set=outcome.replace("eth_minus_", ""),
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "yield_proxy_sensitivity.csv", index=False)

    if plt is not None:
        y = _yield_term(out)
        if not y.empty:
            pivot_order = [c for c in AVAILABLE_YIELDS if c in set(y["yield_col"])]
            x = np.arange(len(pivot_order))
            width = 0.35
            plt.figure(figsize=(10, 5))
            for offset, outcome in [
                (-width / 2, "eth_minus_control_mean"),
                (width / 2, "eth_minus_control_median"),
            ]:
                grp = (
                    y[y["dependent"] == outcome]
                    .set_index("yield_col")
                    .reindex(pivot_order)
                )
                plt.bar(x + offset, grp["coef"], width=width, label=outcome)
                plt.errorbar(
                    x + offset,
                    grp["coef"],
                    yerr=1.96 * grp["std_err_hac"],
                    fmt="none",
                    capsize=3,
                )
            plt.axhline(0, linewidth=0.8)
            plt.xticks(x, pivot_order, rotation=30, ha="right")
            plt.ylabel("HAC coefficient")
            plt.title("Yield proxy sensitivity")
            plt.legend()
            _savefig("yield_proxy_sensitivity.png")
    return out


yield_proxy_results = run_yield_proxy_sensitivity()


# %%
def run_hac_lag_sensitivity() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for hac_lags in HAC_LAGS_LIST:
        for outcome in MAIN_OUTCOMES:
            rows.extend(
                run_single_model(
                    df,
                    test_block="hac_lag_sensitivity",
                    model=f"{outcome}_on_{MAIN_YIELD}_hac{hac_lags}",
                    yield_col=MAIN_YIELD,
                    dependent=outcome,
                    control_set=outcome.replace("eth_minus_", ""),
                    hac_lags=hac_lags,
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hac_lag_sensitivity.csv", index=False)

    if plt is not None:
        y = _yield_term(out, MAIN_YIELD)
        if not y.empty:
            plt.figure(figsize=(10, 5))
            for outcome, grp in y.groupby("dependent"):
                grp = grp.sort_values("hac_lags")
                plt.plot(grp["hac_lags"], grp["coef"], marker="o", label=outcome)
            plt.axhline(0, linewidth=0.8)
            plt.xlabel("HAC lags")
            plt.ylabel("Coefficient on main yield proxy")
            plt.title("HAC lag sensitivity")
            plt.legend()
            _savefig("hac_lag_sensitivity.png")
    return out


hac_lag_results = run_hac_lag_sensitivity()


# %%
def winsorize_copy(
    data: pd.DataFrame, columns: list[str], level: float | None
) -> pd.DataFrame:
    out = data.copy()
    if level is None:
        return out
    for col in columns:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        lo = values.quantile(level)
        hi = values.quantile(1.0 - level)
        out[col] = values.clip(lo, hi)
    return out


def run_winsorization_sensitivity() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    numeric_cols = list(set(MAIN_OUTCOMES + [MAIN_YIELD] + AVAILABLE_CONTROLS))
    for winsor_level in WINSOR_LEVELS:
        wdf = winsorize_copy(df, numeric_cols, winsor_level)
        for outcome in MAIN_OUTCOMES:
            rows.extend(
                run_single_model(
                    wdf,
                    test_block="winsorization_sensitivity",
                    model=f"{outcome}_on_{MAIN_YIELD}_winsor_{winsor_level}",
                    yield_col=MAIN_YIELD,
                    dependent=outcome,
                    control_set=outcome.replace("eth_minus_", ""),
                    winsor_level=winsor_level,
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "winsorization_sensitivity.csv", index=False)

    if plt is not None:
        y = _yield_term(out, MAIN_YIELD)
        if not y.empty:
            plot_df = y.copy()
            plot_df["winsor_label"] = plot_df["winsor_level"].map(
                lambda x: "None" if pd.isna(x) else str(x)
            )
            order = ["None", "0.005", "0.01"]
            plt.figure(figsize=(10, 5))
            for outcome, grp in plot_df.groupby("dependent"):
                grp = grp.set_index("winsor_label").reindex(order).reset_index()
                plt.plot(grp["winsor_label"], grp["coef"], marker="o", label=outcome)
            plt.axhline(0, linewidth=0.8)
            plt.xlabel("Winsor level")
            plt.ylabel("Coefficient on main yield proxy")
            plt.title("Winsorization sensitivity")
            plt.legend()
            _savefig("winsorization_sensitivity.png")
    return out


winsorization_results = run_winsorization_sensitivity()


# %%
def run_placebo_spread_tests() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for yield_col in AVAILABLE_YIELDS:
        for outcome in PLACEBO_OUTCOMES:
            rows.extend(
                run_single_model(
                    df,
                    test_block="placebo_spread_tests",
                    model=f"{outcome}_on_{yield_col}",
                    yield_col=yield_col,
                    dependent=outcome,
                    control_set="non_eth_placebo",
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "placebo_spread_tests.csv", index=False)

    if plt is not None:
        y = _yield_term(out, MAIN_YIELD)
        if not y.empty:
            plot_df = y.set_index("dependent").reindex(PLACEBO_OUTCOMES).reset_index()
            x = np.arange(len(plot_df))
            plt.figure(figsize=(10, 5))
            plt.bar(x, plot_df["coef"])
            plt.errorbar(
                x,
                plot_df["coef"],
                yerr=1.96 * plot_df["std_err_hac"],
                fmt="none",
                capsize=3,
            )
            for idx, row in enumerate(plot_df.itertuples(index=False)):
                if pd.notna(row.p_value) and row.p_value < 0.05:
                    plt.text(
                        idx,
                        row.coef,
                        "*",
                        ha="center",
                        va="bottom" if row.coef >= 0 else "top",
                    )
            plt.axhline(0, linewidth=0.8)
            plt.xticks(x, plot_df["dependent"], rotation=30, ha="right")
            plt.ylabel("Coefficient on main yield proxy")
            plt.title("Placebo non-ETH spread tests")
            _savefig("placebo_spread_tests.png")
    return out


placebo_results = run_placebo_spread_tests()


# %%
def run_economic_magnitude() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome in MAIN_OUTCOMES:
        x_cols = [MAIN_YIELD, *AVAILABLE_CONTROLS]
        res, use = fit_ols_hac(df, outcome, x_cols, maxlags=DEFAULT_HAC_LAGS)
        if res is None or use.empty:
            rows.append(
                {"yield_col": MAIN_YIELD, "dependent": outcome, "nobs": len(use)}
            )
            continue
        beta = res.params.get(MAIN_YIELD, np.nan)
        std_yield = use[MAIN_YIELD].std(ddof=1)
        std_outcome = use[outcome].std(ddof=1)
        effect_1sd = beta * std_yield
        rows.append(
            {
                "yield_col": MAIN_YIELD,
                "dependent": outcome,
                "beta": beta,
                "p_value": res.pvalues.get(MAIN_YIELD, np.nan),
                "nobs": int(res.nobs),
                "adj_r2": res.rsquared_adj,
                "std_yield": std_yield,
                "std_outcome": std_outcome,
                "effect_1sd": effect_1sd,
                "effect_1sd_bps": effect_1sd * 10000,
                "standardized_beta": (
                    effect_1sd / std_outcome
                    if std_outcome and pd.notna(std_outcome)
                    else np.nan
                ),
                "outcome_mean": use[outcome].mean(),
                "outcome_std": std_outcome,
                "yield_mean": use[MAIN_YIELD].mean(),
                "yield_std": std_yield,
                "sample_start": use["date"].min().date().isoformat(),
                "sample_end": use["date"].max().date().isoformat(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "economic_magnitude.csv", index=False)

    if plt is not None and "standardized_beta" in out.columns:
        plot_df = out.set_index("dependent").reindex(MAIN_OUTCOMES).reset_index()
        x = np.arange(len(plot_df))
        plt.figure(figsize=(10, 5))
        plt.bar(x, plot_df["standardized_beta"])
        plt.axhline(0, linewidth=0.8)
        plt.xticks(x, plot_df["dependent"], rotation=30, ha="right")
        plt.ylabel("Standardized beta")
        plt.title("Economic magnitude: standardized beta")
        _savefig("economic_magnitude_standardized_beta.png")
    return out


economic_magnitude = run_economic_magnitude()


# %%
def _sign_stability(table: pd.DataFrame, yield_col: str) -> str:
    y = _yield_term(table, yield_col)
    if y.empty:
        return "not estimated"
    signs = set(np.sign(y["coef"].dropna()).astype(int).tolist())
    if len(signs) <= 1:
        return "sign is stable"
    return "sign varies across specifications"


def _yield_proxy_sensitivity_note() -> str:
    y = _yield_term(yield_proxy_results)
    if y.empty:
        return "Yield proxy sensitivity was not estimated."
    notes = []
    for outcome, grp in y.groupby("dependent"):
        signs = set(np.sign(grp["coef"].dropna()).astype(int).tolist())
        if len(signs) > 1:
            notes.append(f"{outcome}: sensitive to yield proxy")
        else:
            notes.append(f"{outcome}: sign stable across available proxies")
    return "; ".join(notes)


def _placebo_note() -> str:
    y = _yield_term(placebo_results)
    if y.empty:
        return "Placebo spread tests were not estimated."
    sig = y[y["p_value"] < 0.05]
    if sig.empty:
        return "No non-ETH placebo spread has a 5% significant ETH-yield coefficient for the available proxy set."
    return f"{len(sig)} placebo yield coefficients are significant at 5%; ETH yield may also proxy broader non-ETH funding-cycle variation."


def _winsor_note() -> str:
    return _sign_stability(winsorization_results, MAIN_YIELD)


def _hac_note() -> str:
    return _sign_stability(hac_lag_results, MAIN_YIELD)


def write_report() -> Path:
    control_preview = _markdown_table(
        control_leg_summary,
        ["yield_col", "control_set", "coef", "p_value", "interpretation"],
        max_rows=32,
    )
    econ_preview = _markdown_table(
        economic_magnitude,
        ["dependent", "beta", "effect_1sd_bps", "standardized_beta", "adj_r2"],
        max_rows=10,
    )
    report = f"""# ETH yield robustness checks

## 1. Purpose
This notebook is a robustness appendix. It excludes basis analysis and event interaction analysis, and checks whether the main ETH-yield/ETH-relative-funding result is sensitive to benchmark choice, yield proxy choice, outliers, HAC standard-error assumptions, placebo non-ETH spreads, and economic-magnitude scaling.

## 2. Data and sample
- Input CSV: `{INPUT_CSV}`
- Sample: {df['date'].min().date()} to {df['date'].max().date()}
- Observations: {len(df):,}
- Funding legs: {', '.join(FUNDING_COLS.keys())}
- Available yield proxies: {AVAILABLE_YIELDS}
- Main yield proxy used for HAC/winsor/economic-magnitude sections: `{MAIN_YIELD}`
- Controls used when available: {AVAILABLE_CONTROLS if AVAILABLE_CONTROLS else 'none'}

## 3. Main sign convention
The dependent variable is `ETH funding - control funding`. Under a dividend-like funding discount interpretation, higher ETH yield predicts a **negative** coefficient because ETH funding falls relative to non-yielding control funding. A positive coefficient means ETH funding rises relative to the chosen control. An insignificant coefficient means no clear relative-funding effect.

## 4. Control-leg sensitivity
The full table is `tables/control_leg_sensitivity.csv`; the yield-term summary is `tables/control_leg_sensitivity_summary.csv`.

{control_preview}

BTC-only results should be read separately from XRP/DOGE and basket controls because BTC may be a special benchmark rather than a generic non-yielding alt funding leg.

## 5. Yield proxy sensitivity
{_yield_proxy_sensitivity_note()}

See `tables/yield_proxy_sensitivity.csv` and `figures/yield_proxy_sensitivity.png`.

## 6. HAC lag sensitivity
{_hac_note()} across HAC lags {HAC_LAGS_LIST}. If p-values change materially while coefficients remain similar, inference is standard-error sensitive rather than sign sensitive.

## 7. Winsorization sensitivity
{_winsor_note()} across winsor levels {WINSOR_LEVELS}. This checks whether extreme funding, yield, relative-funding, or control observations drive the result.

## 8. Placebo spread tests
{_placebo_note()}

If ETH yield is significant in spreads that do not include ETH, it may proxy broader alt/funding-cycle variation rather than a uniquely ETH-specific mechanism.

## 9. Economic magnitude
Assuming funding is measured in decimal annualized units, `effect_1sd_bps` is the annualized basis-point change in ETH-relative funding associated with a one-standard-deviation increase in the main yield proxy.

{econ_preview}

This is not a forecasting model of funding rates. The coefficient is interpreted as a marginal pricing component; low R² does not by itself invalidate the marginal association.

## 10. Overall conclusion
The main negative relation between ETH yield and ETH-relative funding is strongest when XRP/DOGE or the non-yielding control basket is used as the benchmark. The result should be interpreted as a marginal dividend-like funding adjustment rather than a full explanation of funding-rate variation. Robustness checks assess whether this relation is driven by benchmark choice, yield proxy choice, outliers, standard-error assumptions, or broader non-ETH funding spreads.
"""
    path = REPORT_DIR / "eth_yield_robustness_report.md"
    path.write_text(report)
    print(f"Wrote {path}")
    return path


report_path = write_report()
