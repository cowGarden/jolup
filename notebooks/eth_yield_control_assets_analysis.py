# %% [markdown]
"""
# ETH/wstETH yield vs non-yield control funding legs

This reproducible analysis separates ETH-specific yield pricing from a broader
crypto-wide funding cycle. ETH/wstETH yield is not a purely exogenous coupon: it
can rise with Ethereum on-chain activity, priority fees, MEV, and general crypto
risk appetite. Those same conditions can raise perpetual funding across
non-yielding or staking-yield-non-core assets such as BTC, XRP, and DOGE. A
positive relation between ETH yield and ETH funding therefore does not by itself
prove ETH-specific yield pricing. The key test is whether ETH yield explains ETH
funding relative to non-yielding control assets.

Expected signs:
- `control_funding_mean`: positive yield coefficient means yield comoves with the
  crypto-wide funding cycle.
- `eth_minus_control_mean`: positive yield coefficient is consistent with
  dividend-like ETH carry pricing; negative may indicate an LST-long/perp-short
  hedged carry channel; insignificant implies no ETH-specific relative effect
  after removing common funding conditions.
"""

# %%
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # noqa: BLE001
    plt = None
    warnings.warn(f"matplotlib unavailable; figures will be skipped: {exc}")

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # noqa: BLE001
    PCA = None
    StandardScaler = None
    warnings.warn(f"sklearn unavailable; pc1_control_funding will be skipped: {exc}")

# %%
INPUT_CSV = "data/processed/funding_yield_panel.csv"
YIELD_COL = None
YIELD_CANDIDATES = [
    "wsteth_implied_yield_7d",
    "wsteth_implied_yield_30d",
    "stake_yield",
    "eth_native_yield",
]
CONTROL_ASSETS = ["btc", "xrp", "doge"]
EVENT_DATES = ["2024-03-13", "2024-07-24", "2025-05-07", "2025-09-11"]
EVENT_LABELS = {
    "2024-03-13": "Dencun / EIP-4844 candidate",
    "2024-07-24": "US spot ETH ETF start candidate",
    "2025-05-07": "Pectra candidate",
    "2025-09-11": "validator exit queue / staking liquidity shock candidate",
}
HAC_LAGS = 5
OUTPUT_DIR = Path("outputs/eth_yield_control_assets")
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
for d in [TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# %%
def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            return col
    return None


def load_and_validate(path: str | Path) -> tuple[pd.DataFrame, str, list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input CSV: {path}. Build it with scripts/build_funding_yield_panel.py first."
        )
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a date column.")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Support both explicit *_funding names and older f_* names.
    for asset in ["eth", *CONTROL_ASSETS]:
        explicit = f"{asset}_funding"
        legacy = f"f_{asset}"
        if explicit not in df.columns and legacy in df.columns:
            df[explicit] = df[legacy]
        if legacy not in df.columns and explicit in df.columns:
            df[legacy] = df[explicit]

    if "eth_funding" not in df.columns:
        raise ValueError("Input CSV must contain eth_funding or f_eth.")
    available_controls = [a for a in CONTROL_ASSETS if f"{a}_funding" in df.columns]
    if len(available_controls) < 2:
        raise ValueError("Need at least two of btc_funding, xrp_funding, doge_funding.")

    selected_yield = YIELD_COL or _first_existing(df, YIELD_CANDIDATES)
    if selected_yield is None:
        raise ValueError(f"No usable yield column found. Tried: {YIELD_CANDIDATES}")
    df["eth_yield"] = pd.to_numeric(df[selected_yield], errors="coerce")

    for col in ["ret_eth_btc", "rv_eth_btc"]:
        if col not in df.columns:
            warnings.warn(f"Missing control column {col}; creating NaN so affected models will skip rows.")
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Sample range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Rows: {len(df):,}")
    print(f"Selected yield column: {selected_yield}")
    print(f"Available controls: {available_controls}")
    return df, selected_yield, available_controls


df, SELECTED_YIELD_COL, AVAILABLE_CONTROLS = load_and_validate(INPUT_CSV)

# %%
def construct_variables(data: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    out = data.copy()
    for asset in ["eth", *controls]:
        out[f"f_{asset}"] = pd.to_numeric(out[f"{asset}_funding"], errors="coerce")

    control_cols = [f"f_{a}" for a in controls]
    out["control_funding_mean"] = out[control_cols].mean(axis=1)
    out["control_funding_median"] = out[control_cols].median(axis=1)

    if PCA is not None and StandardScaler is not None and len(control_cols) >= 2:
        use = out[control_cols].dropna()
        if len(use) >= 3:
            z = StandardScaler().fit_transform(use)
            pc1 = PCA(n_components=1).fit_transform(z).ravel()
            out.loc[use.index, "pc1_control_funding"] = pc1
        else:
            warnings.warn("Too few complete rows to compute pc1_control_funding.")
    else:
        warnings.warn("Skipping pc1_control_funding because sklearn is unavailable or controls are insufficient.")

    out["eth_minus_control_mean"] = out["f_eth"] - out["control_funding_mean"]
    out["control_minus_eth_mean"] = out["control_funding_mean"] - out["f_eth"]
    for asset in controls:
        out[f"eth_minus_{asset}"] = out["f_eth"] - out[f"f_{asset}"]
        out[f"{asset}_minus_eth"] = out[f"f_{asset}"] - out["f_eth"]
    return out


df = construct_variables(df, AVAILABLE_CONTROLS)

# %%
def save_correlation_table(data: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "eth_yield",
        "f_eth",
        "f_btc",
        "f_xrp",
        "f_doge",
        "control_funding_mean",
        "eth_minus_control_mean",
        "eth_minus_btc",
        "ret_eth_btc",
        "rv_eth_btc",
    ]
    cols = [c for c in cols if c in data.columns]
    corr = data[cols].corr()
    corr.to_csv(TABLE_DIR / "correlation_matrix.csv")
    print(corr.round(3))
    return corr


corr = save_correlation_table(df)

# %%
def fit_ols_hac(data: pd.DataFrame, y_col: str, x_cols: list[str], maxlags: int = 5):
    use = data[[y_col, *x_cols, "date"]].dropna()
    if len(use) <= len(x_cols) + 2:
        raise ValueError(f"Insufficient observations for {y_col} on {x_cols}: n={len(use)}")
    X = sm.add_constant(use[x_cols], has_constant="add")
    y = use[y_col]
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return res, use


def extract_terms(model_name: str, dependent: str, res, use: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for term in res.params.index:
        rows.append(
            {
                "model": model_name,
                "dependent": dependent,
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


def run_model_table(model_specs: list[tuple[str, str, list[str]]], out_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, y_col, x_cols in model_specs:
        try:
            res, use = fit_ols_hac(df, y_col, x_cols, HAC_LAGS)
            rows.extend(extract_terms(model_name, y_col, res, use))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Skipping {model_name}: {exc}")
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / out_name, index=False)
    return out


BASE_X = ["eth_yield", "ret_eth_btc", "rv_eth_btc"]

# %%
individual_specs = [(f"{asset}_funding_leg", f"f_{asset}", BASE_X) for asset in ["eth", *AVAILABLE_CONTROLS]]
individual_models = run_model_table(individual_specs, "individual_funding_leg_models.csv")
individual_models.query("term == 'eth_yield'") if not individual_models.empty else individual_models

# %%
common_specs = [
    ("control_funding_mean", "control_funding_mean", BASE_X),
    ("control_funding_median", "control_funding_median", BASE_X),
]
if "pc1_control_funding" in df.columns:
    common_specs.append(("pc1_control_funding", "pc1_control_funding", BASE_X))
common_models = run_model_table(common_specs, "common_funding_models.csv")
common_models.query("term == 'eth_yield'") if not common_models.empty else common_models

# %%
relative_outcomes = ["eth_minus_control_mean", *[f"eth_minus_{a}" for a in AVAILABLE_CONTROLS]]
relative_specs = [(outcome, outcome, BASE_X) for outcome in relative_outcomes if outcome in df.columns]
relative_models = run_model_table(relative_specs, "relative_funding_models.csv")
relative_models.query("term == 'eth_yield'") if not relative_models.empty else relative_models

# %%
control_x = ["eth_yield", "control_funding_mean", "ret_eth_btc", "rv_eth_btc"]
leg_x = ["eth_yield", *[f"f_{a}" for a in AVAILABLE_CONTROLS], "ret_eth_btc", "rv_eth_btc"]
f_eth_specs = [
    ("f_eth_with_control_funding_mean", "f_eth", control_x),
    ("f_eth_with_control_legs", "f_eth", leg_x),
]
f_eth_control_models = run_model_table(f_eth_specs, "f_eth_with_control_funding_models.csv")
f_eth_control_models.query("term == 'eth_yield'") if not f_eth_control_models.empty else f_eth_control_models

# %%
decomp = pd.concat(
    [
        individual_models.query("dependent == 'f_eth' and term == 'eth_yield'"),
        common_models.query("dependent == 'control_funding_mean' and term == 'eth_yield'"),
        relative_models.query("dependent == 'eth_minus_control_mean' and term == 'eth_yield'"),
    ],
    ignore_index=True,
)
print(decomp[["dependent", "coef", "t", "p_value", "nobs", "adj_r2"]] if not decomp.empty else decomp)

# %%
def wald_pvalue_for_sum(res, a: str, b: str) -> float:
    names = list(res.params.index)
    R = np.zeros((1, len(names)))
    R[0, names.index(a)] = 1.0
    R[0, names.index(b)] = 1.0
    try:
        return float(res.wald_test(R, scalar=True).pvalue)
    except Exception:  # noqa: BLE001
        return np.nan


def run_event_interactions() -> pd.DataFrame:
    rows = []
    outcomes = ["eth_minus_control_mean", "f_eth", "control_funding_mean"]
    for event_date in EVENT_DATES:
        event_ts = pd.Timestamp(event_date)
        work = df.copy()
        work["post_event"] = (work["date"] >= event_ts).astype(int)
        work["eth_yield_x_post"] = work["eth_yield"] * work["post_event"]
        x_cols = ["eth_yield", "post_event", "eth_yield_x_post", "ret_eth_btc", "rv_eth_btc"]
        for outcome in outcomes:
            try:
                res, use = fit_ols_hac(work, outcome, x_cols, HAC_LAGS)
                pre = res.params.get("eth_yield", np.nan)
                inter = res.params.get("eth_yield_x_post", np.nan)
                rows.append(
                    {
                        "event_date": event_date,
                        "event_label": EVENT_LABELS.get(event_date, ""),
                        "outcome": outcome,
                        "pre_slope": pre,
                        "interaction": inter,
                        "post_slope": pre + inter,
                        "interaction_p_value": res.pvalues.get("eth_yield_x_post", np.nan),
                        "post_slope_wald_p_value": wald_pvalue_for_sum(res, "eth_yield", "eth_yield_x_post"),
                        "n_pre": int((use["date"] < event_ts).sum()),
                        "n_post": int((use["date"] >= event_ts).sum()),
                        "nobs": int(res.nobs),
                        "adj_r2": res.rsquared_adj,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Skipping event {event_date} / {outcome}: {exc}")
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "event_interaction_results.csv", index=False)
    return out


event_results = run_event_interactions()
event_results.head()

# %%
def run_placebo_grid() -> pd.DataFrame:
    rows = []
    candidates = pd.date_range("2024-01-01", "2026-01-01", freq="7D")
    for d in candidates:
        work = df.copy()
        work["post_d"] = (work["date"] >= d).astype(int)
        use_check = work[["eth_minus_control_mean", "eth_yield", "ret_eth_btc", "rv_eth_btc", "date"]].dropna()
        n_pre = int((use_check["date"] < d).sum())
        n_post = int((use_check["date"] >= d).sum())
        if n_pre < 120 or n_post < 120:
            continue
        work["eth_yield_x_post_d"] = work["eth_yield"] * work["post_d"]
        x_cols = ["eth_yield", "post_d", "eth_yield_x_post_d", "ret_eth_btc", "rv_eth_btc"]
        try:
            res, _ = fit_ols_hac(work, "eth_minus_control_mean", x_cols, HAC_LAGS)
            pre = res.params.get("eth_yield", np.nan)
            inter = res.params.get("eth_yield_x_post_d", np.nan)
            rows.append(
                {
                    "date": d.date().isoformat(),
                    "interaction_coef": inter,
                    "interaction_t": res.tvalues.get("eth_yield_x_post_d", np.nan),
                    "interaction_p": res.pvalues.get("eth_yield_x_post_d", np.nan),
                    "pre_slope": pre,
                    "post_slope": pre + inter,
                    "n_pre": n_pre,
                    "n_post": n_post,
                }
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Skipping placebo date {d.date()}: {exc}")
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "placebo_event_grid.csv", index=False)
    return out


placebo_grid = run_placebo_grid()
placebo_grid.head()

# %%
def run_rolling() -> pd.DataFrame:
    rows = []
    windows = [90, 120, 180]
    outcomes = ["control_funding_mean", "eth_minus_control_mean"]
    x_cols = ["eth_yield", "ret_eth_btc", "rv_eth_btc"]
    base = df[["date", *outcomes, *x_cols]].dropna().sort_values("date")
    for window in windows:
        for end_pos in range(window, len(base) + 1):
            chunk = base.iloc[end_pos - window:end_pos]
            for outcome in outcomes:
                try:
                    res, use = fit_ols_hac(chunk, outcome, x_cols, HAC_LAGS)
                    coef = res.params.get("eth_yield", np.nan)
                    sx = use["eth_yield"].std(ddof=0)
                    sy = use[outcome].std(ddof=0)
                    rows.append(
                        {
                            "window_end_date": chunk["date"].iloc[-1].date().isoformat(),
                            "window_size": window,
                            "outcome": outcome,
                            "coef": coef,
                            "p_value": res.pvalues.get("eth_yield", np.nan),
                            "standardized_beta": coef * sx / sy if sy and pd.notna(sy) else np.nan,
                            "nobs": int(res.nobs),
                            "r2": res.rsquared,
                        }
                    )
                except Exception:
                    continue
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "rolling_results.csv", index=False)
    return out


rolling_results = run_rolling()
rolling_results.head()

# %%
def _plot_or_skip():
    if plt is None:
        return
    plt.style.use("seaborn-v0_8-whitegrid")

    funding_cols = ["f_eth", *[f"f_{a}" for a in AVAILABLE_CONTROLS]]
    ax = df.set_index("date")[funding_cols].plot(figsize=(12, 5), title="Funding legs")
    ax.set_ylabel("Annualized funding")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "funding_legs.png", dpi=150)
    plt.close()

    ax = df.set_index("date")[["control_funding_mean", "eth_minus_control_mean"]].plot(
        figsize=(12, 5), title="Common vs ETH-specific relative funding"
    )
    ax.set_ylabel("Annualized funding")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "common_vs_relative_funding.png", dpi=150)
    plt.close()

    ax = df.plot.scatter("eth_yield", "control_funding_mean", alpha=0.5, figsize=(6, 5), title="Yield vs common funding")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "yield_vs_common_funding_scatter.png", dpi=150)
    plt.close()

    ax = df.plot.scatter("eth_yield", "eth_minus_control_mean", alpha=0.5, figsize=(6, 5), title="Yield vs ETH-minus-control funding")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "yield_vs_relative_funding_scatter.png", dpi=150)
    plt.close()

    if not placebo_grid.empty:
        plot_df = placebo_grid.copy()
        plot_df["date"] = pd.to_datetime(plot_df["date"])
        ax = plot_df.plot("date", "interaction_t", figsize=(12, 5), legend=False, title="Placebo event interaction t-stat")
        for event_date in EVENT_DATES:
            ax.axvline(pd.Timestamp(event_date), color="red", linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("t-stat")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "event_grid_interaction_tstat.png", dpi=150)
        plt.close()

    for outcome, fname in [
        ("control_funding_mean", "rolling_common_coef.png"),
        ("eth_minus_control_mean", "rolling_relative_coef.png"),
    ]:
        sub = rolling_results[rolling_results["outcome"] == outcome].copy()
        if sub.empty:
            continue
        sub["window_end_date"] = pd.to_datetime(sub["window_end_date"])
        fig, ax = plt.subplots(figsize=(12, 5))
        for window, g in sub.groupby("window_size"):
            ax.plot(g["window_end_date"], g["coef"], label=f"{window}d")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"Rolling eth_yield coefficient: {outcome}")
        ax.set_ylabel("Coefficient")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / fname, dpi=150)
        plt.close()


_plot_or_skip()

# %%
def _yield_summary(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    return table[table["term"] == "eth_yield"].copy()


def _significance_text(row: pd.Series | None) -> str:
    if row is None or row.empty:
        return "not estimated"
    coef = row.get("coef", np.nan)
    p = row.get("p_value", np.nan)
    direction = "positive" if coef > 0 else "negative" if coef < 0 else "zero"
    sig = "statistically significant" if pd.notna(p) and p < 0.05 else "not statistically significant"
    return f"{direction} and {sig} (coef={coef:.6g}, p={p:.4g})"


def write_report() -> Path:
    ind_y = _yield_summary(individual_models)
    common_y = _yield_summary(common_models)
    rel_y = _yield_summary(relative_models)
    feth_y = _yield_summary(f_eth_control_models)

    all_legs_positive = False
    if not ind_y.empty:
        expected_deps = {"f_eth", *[f"f_{a}" for a in AVAILABLE_CONTROLS]}
        legs = ind_y[ind_y["dependent"].isin(expected_deps)]
        all_legs_positive = len(legs) >= len(expected_deps) and bool((legs["coef"] > 0).all())

    common_row = common_y[common_y["dependent"] == "control_funding_mean"].iloc[0] if not common_y[common_y["dependent"] == "control_funding_mean"].empty else None
    rel_row = rel_y[rel_y["dependent"] == "eth_minus_control_mean"].iloc[0] if not rel_y[rel_y["dependent"] == "eth_minus_control_mean"].empty else None
    feth_control_row = feth_y[feth_y["model"] == "f_eth_with_control_funding_mean"].iloc[0] if not feth_y[feth_y["model"] == "f_eth_with_control_funding_mean"].empty else None

    if rel_row is not None and pd.notna(rel_row.get("p_value")) and rel_row.get("p_value") < 0.05 and rel_row.get("coef") > 0:
        interpretation = "There is evidence consistent with ETH yield being priced as ETH-specific carry in perpetual markets."
    elif rel_row is not None and pd.notna(rel_row.get("p_value")) and rel_row.get("p_value") < 0.05 and rel_row.get("coef") < 0:
        interpretation = "The relative-funding relation is negative, consistent with an LST-long/perp-short hedged carry channel potentially dominating."
    else:
        interpretation = "wstETH yield level appears to comove with crypto-wide funding conditions rather than uniquely pricing ETH-specific carry."

    event_note = "No event interaction models estimated."
    if not event_results.empty:
        event_note = f"Estimated {len(event_results)} event/outcome interaction models. See event_interaction_results.csv; 2025-09-11 is treated as a staking liquidity/exit queue shock candidate, not causal proof."

    placebo_note = "No placebo grid estimates passed sample-size filters."
    if not placebo_grid.empty:
        top = placebo_grid.reindex(placebo_grid["interaction_t"].abs().sort_values(ascending=False).index).head(3)
        placebo_note = "Top absolute placebo interaction t-stat dates: " + ", ".join(
            f"{r.date} (t={r.interaction_t:.2f})" for r in top.itertuples()
        )

    report = f"""# ETH yield control-assets funding analysis

## Data
- Input CSV: `{INPUT_CSV}`
- Data period: {df['date'].min().date()} to {df['date'].max().date()}
- Observations: {len(df):,}
- Selected yield column: `{SELECTED_YIELD_COL}`
- Available funding legs: ETH plus {', '.join(a.upper() for a in AVAILABLE_CONTROLS)}

## Theoretical framing
ETH/wstETH yield is not a purely exogenous coupon. It can increase when Ethereum
on-chain activity, priority fees, MEV, and broader crypto risk appetite increase.
These same conditions can also raise perp funding rates across non-yielding
assets such as BTC, XRP, and DOGE. Therefore, a positive relation between ETH
yield and ETH funding does not by itself prove ETH-specific yield pricing. The
appropriate test is whether ETH yield explains ETH funding relative to
non-yielding control assets.

## Key model interpretation
| Dependent variable | Yield coefficient interpretation |
|---|---|
| `f_eth` | Simple relation with ETH funding |
| `control_funding_mean` | Relation with crypto-wide funding cycle |
| `eth_minus_control_mean` | ETH-specific relative funding effect |
| `f_eth ~ yield + control_funding` | ETH yield effect after removing common funding |

## Findings to inspect
- ETH yield positive across all estimated funding legs: {all_legs_positive}
- `eth_yield -> control_funding_mean`: {_significance_text(common_row)}
- `eth_yield -> eth_minus_control_mean`: {_significance_text(rel_row)}
- `eth_yield -> f_eth` controlling for `control_funding_mean`: {_significance_text(feth_control_row)}
- Event interactions: {event_note}
- Placebo grid: {placebo_note}

## Interpretation
{interpretation}

If yield explains common funding but not relative funding, the result supports an
activity/common-cycle explanation. If yield explains relative funding positively,
it is evidence consistent with ETH-specific dividend-like carry pricing. If the
relationship changes around candidate dates, the pricing role of ETH yield
appears regime-dependent.

## Cautions
- This is not definitive causal identification.
- XRP and DOGE are controls for common crypto funding conditions, but they may
  still have idiosyncratic meme/liquidity cycles.
- Event dates are mechanism-consistent candidates, not strict exogenous shocks.

## Reproduction
Build the panel:
```bash
python scripts/build_funding_yield_panel.py \\
  --start-date 2023-01-01 \\
  --end-date 2026-05-09 \\
  --exchange binance \\
  --assets BTC ETH XRP DOGE \\
  --fetch-onchain-lst-rates \\
  --ethereum-rpc-url "$ETHEREUM_RPC_URL" \\
  --out-csv data/processed/funding_yield_panel.csv
```

Run this analysis:
```bash
python notebooks/eth_yield_control_assets_analysis.py
```
"""
    path = REPORT_DIR / "eth_yield_control_assets_report.md"
    path.write_text(report)
    print(f"Wrote {path}")
    return path


report_path = write_report()
