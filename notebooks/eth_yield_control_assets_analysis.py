# %% [markdown]
"""
# ETH/wstETH yield, common funding cycles, and perp-spot basis

This notebook tests whether ETH/wstETH yield behaves like ETH-specific carry in
perpetual funding or mainly co-moves with a crypto-wide leverage/funding cycle.
It separates four empirical objects:

1. individual funding legs (`f_eth`, `f_btc`, `f_xrp`, `f_doge`),
2. a non-yield control funding factor,
3. ETH-relative funding after removing control legs, and
4. perp-spot basis as a possible mechanism or mediator.

The sign convention is `eth_minus_control_mean = f_eth - mean(f_btc, f_xrp,
f_doge)`. A positive yield coefficient in this relative outcome is consistent
with dividend-like ETH carry pricing. A negative coefficient is consistent with
hedged carry / wstETH-long plus ETH-perp-short pressure.
"""

# %%
from __future__ import annotations

from itertools import combinations
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
ALL_ASSETS = ["eth", *CONTROL_ASSETS]

HAC_LAGS = 5
RV_WINDOW = 30
ROLLING_WINDOWS = [90, 120, 180]

RUN_EVENT_ANALYSIS = False
RUN_ROLLING_ANALYSIS = True
RUN_BASIS_ANALYSIS = True
RUN_CONTROL_LEG_SENSITIVITY = True

MAIN_RELATIVE_OUTCOME = "eth_minus_control_mean"
MAIN_RELATIVE_BASIS = "eth_minus_control_basis"

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


def _fmt(x: object, digits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{digits}g}"
    return str(x)


def load_and_validate(path: str | Path) -> tuple[pd.DataFrame, str, list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input CSV: {path}. Build it with scripts/build_funding_yield_panel.py first."
        )
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a date column.")
    df["date"] = (
        pd.to_datetime(df["date"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for asset in ALL_ASSETS:
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
            warnings.warn(f"Missing control column {col}; models will omit it.")
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Sample range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Rows: {len(df):,}")
    print(f"Selected yield column: {selected_yield}")
    print(f"Yield non-null count: {df['eth_yield'].notna().sum():,}")
    print(f"Available controls: {available_controls}")
    return df, selected_yield, available_controls


df, SELECTED_YIELD_COL, AVAILABLE_CONTROLS = load_and_validate(INPUT_CSV)
MODEL_CONTROLS = [c for c in ["ret_eth_btc", "rv_eth_btc"] if df[c].notna().any()]
if len(MODEL_CONTROLS) < 2:
    warnings.warn(f"Using available return/volatility controls only: {MODEL_CONTROLS}")


# %%
def construct_funding_variables(
    data: pd.DataFrame, controls: list[str]
) -> pd.DataFrame:
    out = data.copy()
    for asset in ["eth", *controls]:
        out[f"f_{asset}"] = pd.to_numeric(out[f"{asset}_funding"], errors="coerce")

    control_cols = [f"f_{a}" for a in controls]
    out["control_funding_mean"] = out[control_cols].mean(axis=1)
    out["control_funding_median"] = out[control_cols].median(axis=1)

    for asset in controls:
        out[f"eth_minus_{asset}"] = out["f_eth"] - out[f"f_{asset}"]
        out[f"{asset}_minus_eth"] = out[f"f_{asset}"] - out["f_eth"]

    for a, b in combinations(controls, 2):
        out[f"eth_minus_{a}_{b}_mean"] = out["f_eth"] - out[[f"f_{a}", f"f_{b}"]].mean(
            axis=1
        )

    out["eth_minus_control_mean"] = out["f_eth"] - out[control_cols].mean(axis=1)
    out["eth_minus_control_median"] = out["f_eth"] - out[control_cols].median(axis=1)
    out["control_minus_eth_mean"] = out["control_funding_mean"] - out["f_eth"]
    return out


df = construct_funding_variables(df, AVAILABLE_CONTROLS)


# %%
def pick_basis_col(data: pd.DataFrame, asset: str) -> str | None:
    candidates = [f"{asset}_basis_close", f"{asset}_basis", f"{asset}_basis_avg"]
    return _first_existing(data, candidates)


def construct_basis_variables(
    data: pd.DataFrame, controls: list[str]
) -> tuple[pd.DataFrame, dict[str, str], bool]:
    out = data.copy()
    selected: dict[str, str] = {}
    for asset in ["eth", *controls]:
        col = pick_basis_col(out, asset)
        if col is not None:
            out[f"basis_{asset}"] = pd.to_numeric(out[col], errors="coerce")
            selected[asset] = col

    basis_available = (
        "eth" in selected and len([a for a in controls if a in selected]) >= 2
    )
    if not basis_available:
        warnings.warn(
            "Basis data unavailable or incomplete; basis sections will be skipped."
        )
        return out, selected, False

    control_basis_cols = [f"basis_{a}" for a in controls if a in selected]
    out["control_basis_mean"] = out[control_basis_cols].mean(axis=1)
    out["control_basis_median"] = out[control_basis_cols].median(axis=1)

    for asset in controls:
        if asset in selected:
            out[f"eth_minus_{asset}_basis"] = out["basis_eth"] - out[f"basis_{asset}"]

    for a, b in combinations([a for a in controls if a in selected], 2):
        out[f"eth_minus_{a}_{b}_basis"] = out["basis_eth"] - out[
            [f"basis_{a}", f"basis_{b}"]
        ].mean(axis=1)

    out["eth_minus_control_basis"] = out["basis_eth"] - out[control_basis_cols].mean(
        axis=1
    )
    out["eth_minus_control_basis_median"] = out["basis_eth"] - out[
        control_basis_cols
    ].median(axis=1)

    desc_cols = [
        f"basis_{a}" for a in ["eth", *controls] if f"basis_{a}" in out.columns
    ]
    basis_desc = out[desc_cols].describe().T
    basis_desc.to_csv(TABLE_DIR / "basis_descriptive_stats.csv")
    print("Basis columns selected:", selected)
    print(basis_desc[["mean", "std", "min", "max"]].round(6))
    if out[desc_cols].abs().max().max() > 0.25:
        warnings.warn(
            "At least one absolute basis observation exceeds 25%; inspect basis_descriptive_stats.csv."
        )
    return out, selected, True


df, SELECTED_BASIS_COLS, BASIS_AVAILABLE = construct_basis_variables(
    df, AVAILABLE_CONTROLS
)


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
        "basis_eth",
        "control_basis_mean",
        "eth_minus_control_basis",
    ]
    cols = [c for c in cols if c in data.columns]
    corr = data[cols].corr()
    corr.to_csv(TABLE_DIR / "correlation_matrix.csv")
    print(corr.round(3))
    return corr


corr = save_correlation_table(df)


# %%
def fit_ols_hac(data: pd.DataFrame, y_col: str, x_cols: list[str], maxlags: int = 5):
    cols = ["date", y_col, *x_cols]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError(f"Missing model columns: {missing}")
    use = data[cols].dropna()
    if len(use) <= len(x_cols) + 2:
        raise ValueError(
            f"Insufficient observations for {y_col} on {x_cols}: n={len(use)}"
        )
    X = sm.add_constant(use[x_cols], has_constant="add")
    y = use[y_col]
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return res, use


def extract_terms(
    model_name: str, dependent: str, res, use: pd.DataFrame
) -> list[dict[str, object]]:
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


def run_model_table(
    model_specs: list[tuple[str, str, list[str]]], out_name: str
) -> pd.DataFrame:
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


def term_row(
    table: pd.DataFrame,
    model: str | None = None,
    dependent: str | None = None,
    term: str = "eth_yield",
) -> pd.Series | None:
    if table.empty:
        return None
    use = table[table["term"] == term]
    if model is not None:
        use = use[use["model"] == model]
    if dependent is not None:
        use = use[use["dependent"] == dependent]
    if use.empty:
        return None
    return use.iloc[0]


BASE_X = ["eth_yield", *MODEL_CONTROLS]

# %%
individual_specs = [
    (f"{asset}_funding_on_yield", f"f_{asset}", BASE_X)
    for asset in ["eth", *AVAILABLE_CONTROLS]
]
individual_models = run_model_table(
    individual_specs, "individual_funding_leg_models.csv"
)

common_specs = [
    ("control_funding_mean_on_yield", "control_funding_mean", BASE_X),
    ("control_funding_median_on_yield", "control_funding_median", BASE_X),
]
common_models = run_model_table(common_specs, "common_funding_models.csv")

RELATIVE_OUTCOMES = [
    ("BTC only", "eth_minus_btc", ["btc"]),
    ("XRP only", "eth_minus_xrp", ["xrp"]),
    ("DOGE only", "eth_minus_doge", ["doge"]),
    ("BTC+XRP mean", "eth_minus_btc_xrp_mean", ["btc", "xrp"]),
    ("BTC+DOGE mean", "eth_minus_btc_doge_mean", ["btc", "doge"]),
    ("XRP+DOGE mean", "eth_minus_xrp_doge_mean", ["xrp", "doge"]),
    ("BTC+XRP+DOGE mean", "eth_minus_control_mean", AVAILABLE_CONTROLS),
    ("BTC+XRP+DOGE median", "eth_minus_control_median", AVAILABLE_CONTROLS),
]
RELATIVE_OUTCOMES = [spec for spec in RELATIVE_OUTCOMES if spec[1] in df.columns]
relative_specs = [
    (f"{outcome}_on_yield", outcome, BASE_X) for _, outcome, _ in RELATIVE_OUTCOMES
]
relative_models = run_model_table(
    relative_specs, "relative_funding_sensitivity_models.csv"
)

f_eth_control_specs = [
    (
        "f_eth_with_control_funding_mean",
        "f_eth",
        ["eth_yield", "control_funding_mean", *MODEL_CONTROLS],
    ),
    (
        "f_eth_with_control_funding_median",
        "f_eth",
        ["eth_yield", "control_funding_median", *MODEL_CONTROLS],
    ),
]
if all(f"f_{a}" in df.columns for a in AVAILABLE_CONTROLS):
    f_eth_control_specs.append(
        (
            "f_eth_with_all_control_legs",
            "f_eth",
            ["eth_yield", *[f"f_{a}" for a in AVAILABLE_CONTROLS], *MODEL_CONTROLS],
        )
    )
f_eth_control_models = run_model_table(
    f_eth_control_specs, "f_eth_with_control_funding_models.csv"
)


# %%
def sign_interpretation(coef: float, p_value: float) -> str:
    if pd.notna(coef) and pd.notna(p_value) and p_value < 0.05 and coef > 0:
        return "consistent with dividend-like ETH carry relative to this control set"
    if pd.notna(coef) and pd.notna(p_value) and p_value < 0.05 and coef < 0:
        return "consistent with hedged carry / ETH perp short pressure relative to this control set"
    return "no clear relative funding effect"


def build_control_leg_sensitivity() -> pd.DataFrame:
    rows = []
    for control_set, outcome, assets in RELATIVE_OUTCOMES:
        row = term_row(relative_models, model=f"{outcome}_on_yield", term="eth_yield")
        if row is None:
            continue
        rows.append(
            {
                "control_set": control_set,
                "outcome": outcome,
                "control_assets_used": "+".join(a.upper() for a in assets),
                "eth_yield_coef": row["coef"],
                "eth_yield_std_err_hac": row["std_err_hac"],
                "eth_yield_pvalue": row["p_value"],
                "nobs": row["nobs"],
                "adj_r2": row["adj_r2"],
                "sign_interpretation": sign_interpretation(row["coef"], row["p_value"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "control_leg_sensitivity_summary.csv", index=False)
    return out


control_leg_sensitivity = (
    build_control_leg_sensitivity() if RUN_CONTROL_LEG_SENSITIVITY else pd.DataFrame()
)


# %%
def _savefig(name: str) -> None:
    if plt is not None:
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / name, dpi=150)
        plt.close()


def plot_core_figures() -> None:
    if plt is None:
        return
    funding_cols = [
        f"f_{a}" for a in ["eth", *AVAILABLE_CONTROLS] if f"f_{a}" in df.columns
    ]
    if funding_cols:
        df.plot(
            x="date",
            y=funding_cols,
            figsize=(11, 5),
            title="Daily annualized funding legs",
        )
        plt.axhline(0, linewidth=0.8)
        _savefig("funding_legs.png")

    if {"control_funding_mean", MAIN_RELATIVE_OUTCOME}.issubset(df.columns):
        df.plot(
            x="date",
            y=["control_funding_mean", MAIN_RELATIVE_OUTCOME],
            figsize=(11, 5),
            title="Common vs ETH-relative funding",
        )
        plt.axhline(0, linewidth=0.8)
        _savefig("common_vs_relative_funding.png")

    if {"eth_yield", "control_funding_mean"}.issubset(df.columns):
        df.plot.scatter(
            x="eth_yield",
            y="control_funding_mean",
            figsize=(6, 5),
            title="Yield vs common funding",
        )
        plt.axhline(0, linewidth=0.8)
        _savefig("yield_vs_common_funding_scatter.png")

    if {"eth_yield", MAIN_RELATIVE_OUTCOME}.issubset(df.columns):
        df.plot.scatter(
            x="eth_yield",
            y=MAIN_RELATIVE_OUTCOME,
            figsize=(6, 5),
            title="Yield vs ETH-relative funding",
        )
        plt.axhline(0, linewidth=0.8)
        _savefig("yield_vs_relative_funding_scatter.png")

    if not control_leg_sensitivity.empty:
        plot_df = control_leg_sensitivity.copy()
        x = np.arange(len(plot_df))
        plt.figure(figsize=(11, 5))
        plt.bar(x, plot_df["eth_yield_coef"])
        plt.errorbar(
            x,
            plot_df["eth_yield_coef"],
            yerr=1.96 * plot_df["eth_yield_std_err_hac"],
            fmt="none",
            capsize=3,
        )
        for i, pval in enumerate(plot_df["eth_yield_pvalue"]):
            if pd.notna(pval) and pval < 0.05:
                plt.text(
                    i, plot_df["eth_yield_coef"].iloc[i], "*", ha="center", va="bottom"
                )
        plt.axhline(0, linewidth=0.8)
        plt.xticks(x, plot_df["control_set"], rotation=45, ha="right")
        plt.title("ETH yield coefficient by control-leg construction")
        plt.ylabel("HAC coefficient on eth_yield")
        _savefig("control_leg_sensitivity_coefficients.png")


plot_core_figures()


# %%
def run_basis_sanity_checks() -> pd.DataFrame:
    if not (RUN_BASIS_ANALYSIS and BASIS_AVAILABLE):
        out = pd.DataFrame()
        out.to_csv(TABLE_DIR / "basis_sanity_checks.csv", index=False)
        return out
    specs: list[tuple[str, str, list[str]]] = []
    for asset in ["eth", *AVAILABLE_CONTROLS]:
        if f"basis_{asset}" in df.columns and f"f_{asset}" in df.columns:
            specs.append(
                (
                    f"f_{asset}_on_basis",
                    f"f_{asset}",
                    [f"basis_{asset}", *MODEL_CONTROLS],
                )
            )
    for outcome, basis_col in [
        ("eth_minus_btc", "eth_minus_btc_basis"),
        ("eth_minus_xrp", "eth_minus_xrp_basis"),
        ("eth_minus_doge", "eth_minus_doge_basis"),
        (MAIN_RELATIVE_OUTCOME, MAIN_RELATIVE_BASIS),
    ]:
        if outcome in df.columns and basis_col in df.columns:
            specs.append(
                (f"{outcome}_on_{basis_col}", outcome, [basis_col, *MODEL_CONTROLS])
            )
    return run_model_table(specs, "basis_sanity_checks.csv")


basis_sanity_checks = run_basis_sanity_checks()

# %%
BASIS_OUTCOME_MAP = {
    "eth_minus_btc": "eth_minus_btc_basis",
    "eth_minus_xrp": "eth_minus_xrp_basis",
    "eth_minus_doge": "eth_minus_doge_basis",
    "eth_minus_btc_xrp_mean": "eth_minus_btc_xrp_basis",
    "eth_minus_btc_doge_mean": "eth_minus_btc_doge_basis",
    "eth_minus_xrp_doge_mean": "eth_minus_xrp_doge_basis",
    "eth_minus_control_mean": "eth_minus_control_basis",
    "eth_minus_control_median": "eth_minus_control_basis_median",
}


def basis_adjustment_interpretation(
    no_basis_coef: float, with_basis_coef: float, yield_p: float, basis_p: float
) -> str:
    if pd.notna(basis_p) and basis_p < 0.05 and (pd.isna(yield_p) or yield_p >= 0.05):
        return "relative funding primarily reflects perp-spot basis rather than yield directly"
    if pd.notna(yield_p) and yield_p < 0.05:
        return "yield has direct effect beyond basis"
    if pd.notna(no_basis_coef) and pd.notna(with_basis_coef):
        if abs(with_basis_coef) < abs(no_basis_coef):
            return "basis may mediate or absorb yield effect"
        if abs(with_basis_coef) > abs(no_basis_coef):
            return "basis dislocation may have masked yield effect"
    return "no clear basis-adjustment pattern"


def run_basis_adjusted_models() -> pd.DataFrame:
    if not (RUN_BASIS_ANALYSIS and BASIS_AVAILABLE):
        out = pd.DataFrame()
        out.to_csv(
            TABLE_DIR / "basis_adjusted_relative_funding_models.csv", index=False
        )
        return out
    rows = []
    for control_set, outcome, _assets in RELATIVE_OUTCOMES:
        basis_col = BASIS_OUTCOME_MAP.get(outcome)
        if basis_col not in df.columns:
            continue
        try:
            res_no, use_no = fit_ols_hac(
                df, outcome, ["eth_yield", *MODEL_CONTROLS], HAC_LAGS
            )
            res_with, use_with = fit_ols_hac(
                df, outcome, ["eth_yield", basis_col, *MODEL_CONTROLS], HAC_LAGS
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Skipping basis-adjusted model for {outcome}: {exc}")
            continue
        no_coef = res_no.params.get("eth_yield", np.nan)
        with_coef = res_with.params.get("eth_yield", np.nan)
        basis_coef = res_with.params.get(basis_col, np.nan)
        yield_p = res_with.pvalues.get("eth_yield", np.nan)
        basis_p = res_with.pvalues.get(basis_col, np.nan)
        rows.append(
            {
                "control_set": control_set,
                "outcome": outcome,
                "basis_col": basis_col,
                "eth_yield_coef_no_basis": no_coef,
                "eth_yield_p_no_basis": res_no.pvalues.get("eth_yield", np.nan),
                "eth_yield_coef_with_basis": with_coef,
                "eth_yield_p_with_basis": yield_p,
                "basis_coef": basis_coef,
                "basis_pvalue": basis_p,
                "nobs": int(res_with.nobs),
                "adj_r2_no_basis": res_no.rsquared_adj,
                "adj_r2_with_basis": res_with.rsquared_adj,
                "change_in_yield_coef": with_coef - no_coef,
                "interpretation": basis_adjustment_interpretation(
                    no_coef, with_coef, yield_p, basis_p
                ),
                "sample_start": use_with["date"].min().date().isoformat(),
                "sample_end": use_with["date"].max().date().isoformat(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "basis_adjusted_relative_funding_models.csv", index=False)
    return out


basis_adjusted_models = run_basis_adjusted_models()


# %%
def run_basis_mediation_tests() -> pd.DataFrame:
    if not (RUN_BASIS_ANALYSIS and BASIS_AVAILABLE):
        out = pd.DataFrame()
        out.to_csv(TABLE_DIR / "basis_mediation_results.csv", index=False)
        return out
    rows = []
    for control_set, outcome, _assets in RELATIVE_OUTCOMES:
        basis_col = BASIS_OUTCOME_MAP.get(outcome)
        if basis_col not in df.columns:
            continue
        try:
            baseline, _use_base = fit_ols_hac(
                df, outcome, ["eth_yield", *MODEL_CONTROLS], HAC_LAGS
            )
            first, _use_first = fit_ols_hac(
                df, basis_col, ["eth_yield", *MODEL_CONTROLS], HAC_LAGS
            )
            second, _use_second = fit_ols_hac(
                df, outcome, ["eth_yield", basis_col, *MODEL_CONTROLS], HAC_LAGS
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Skipping mediation test for {outcome}: {exc}")
            continue
        base_coef = baseline.params.get("eth_yield", np.nan)
        direct_coef = second.params.get("eth_yield", np.nan)
        material_change = (
            pd.notna(base_coef)
            and pd.notna(direct_coef)
            and abs(direct_coef - base_coef) >= 0.25 * max(abs(base_coef), 1e-12)
        )
        flag = (
            first.pvalues.get("eth_yield", np.nan) < 0.10
            and second.pvalues.get(basis_col, np.nan) < 0.10
            and material_change
        )
        rows.append(
            {
                "control_set": control_set,
                "relative_basis_outcome": basis_col,
                "funding_outcome": outcome,
                "yield_to_basis_coef": first.params.get("eth_yield", np.nan),
                "yield_to_basis_p": first.pvalues.get("eth_yield", np.nan),
                "basis_to_funding_coef": second.params.get(basis_col, np.nan),
                "basis_to_funding_p": second.pvalues.get(basis_col, np.nan),
                "yield_direct_coef": direct_coef,
                "yield_direct_p": second.pvalues.get("eth_yield", np.nan),
                "baseline_yield_coef": base_coef,
                "possible_mediation_flag": bool(flag),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "basis_mediation_results.csv", index=False)
    return out


basis_mediation_results = run_basis_mediation_tests()


# %%
def run_basis_dislocation_regime() -> pd.DataFrame:
    if not (
        RUN_BASIS_ANALYSIS and BASIS_AVAILABLE and MAIN_RELATIVE_BASIS in df.columns
    ):
        out = pd.DataFrame()
        out.to_csv(TABLE_DIR / "basis_dislocation_regime_results.csv", index=False)
        return out
    out_df = df.copy()
    out_df["abs_eth_relative_basis"] = out_df[MAIN_RELATIVE_BASIS].abs()
    q75 = out_df["abs_eth_relative_basis"].quantile(0.75)
    out_df["high_basis_dislocation"] = (out_df["abs_eth_relative_basis"] >= q75).astype(
        float
    )
    out_df["eth_yield_x_high_basis_dislocation"] = (
        out_df["eth_yield"] * out_df["high_basis_dislocation"]
    )
    try:
        out_df["basis_dislocation_tercile"] = pd.qcut(
            out_df["abs_eth_relative_basis"], 3, labels=["low", "mid", "high"]
        )
    except ValueError:
        out_df["basis_dislocation_tercile"] = np.nan

    rows: list[dict[str, object]] = []
    try:
        res, use = fit_ols_hac(
            out_df,
            MAIN_RELATIVE_OUTCOME,
            [
                "eth_yield",
                "high_basis_dislocation",
                "eth_yield_x_high_basis_dislocation",
                MAIN_RELATIVE_BASIS,
                *MODEL_CONTROLS,
            ],
            HAC_LAGS,
        )
        rows.extend(
            extract_terms(
                "high_dislocation_interaction", MAIN_RELATIVE_OUTCOME, res, use
            )
        )
        base = res.params.get("eth_yield", np.nan)
        inter = res.params.get("eth_yield_x_high_basis_dislocation", np.nan)
        rows.append(
            {
                "model": "high_dislocation_combined_effect",
                "dependent": MAIN_RELATIVE_OUTCOME,
                "term": "eth_yield_plus_interaction",
                "coef": base + inter,
                "std_err_hac": np.nan,
                "t": np.nan,
                "p_value": np.nan,
                "nobs": int(res.nobs),
                "adj_r2": res.rsquared_adj,
                "sample_start": use["date"].min().date().isoformat(),
                "sample_end": use["date"].max().date().isoformat(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Skipping basis dislocation interaction: {exc}")

    for label, mask in [
        ("low_basis_dislocation", out_df["abs_eth_relative_basis"] < q75),
        ("high_basis_dislocation", out_df["abs_eth_relative_basis"] >= q75),
    ]:
        try:
            res, use = fit_ols_hac(
                out_df.loc[mask],
                MAIN_RELATIVE_OUTCOME,
                ["eth_yield", MAIN_RELATIVE_BASIS, *MODEL_CONTROLS],
                HAC_LAGS,
            )
            rows.extend(extract_terms(label, MAIN_RELATIVE_OUTCOME, res, use))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Skipping {label}: {exc}")

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "basis_dislocation_regime_results.csv", index=False)
    return out


basis_dislocation_regime_results = run_basis_dislocation_regime()


# %%
def plot_basis_figures() -> None:
    if plt is None or not (RUN_BASIS_ANALYSIS and BASIS_AVAILABLE):
        return
    basis_cols = [
        f"basis_{a}" for a in ["eth", *AVAILABLE_CONTROLS] if f"basis_{a}" in df.columns
    ]
    if basis_cols:
        df.plot(
            x="date", y=basis_cols, figsize=(11, 5), title="Perp-spot basis by asset"
        )
        plt.axhline(0, linewidth=0.8)
        _savefig("basis_series.png")
    if MAIN_RELATIVE_BASIS in df.columns:
        df.plot(
            x="date",
            y=MAIN_RELATIVE_BASIS,
            figsize=(11, 4),
            title="ETH minus control basis",
        )
        plt.axhline(0, linewidth=0.8)
        _savefig("eth_minus_control_basis.png")
    if {MAIN_RELATIVE_OUTCOME, MAIN_RELATIVE_BASIS}.issubset(df.columns):
        df.plot.scatter(
            x=MAIN_RELATIVE_BASIS,
            y=MAIN_RELATIVE_OUTCOME,
            figsize=(6, 5),
            title="Relative funding vs relative basis",
        )
        plt.axhline(0, linewidth=0.8)
        plt.axvline(0, linewidth=0.8)
        _savefig("relative_funding_vs_relative_basis.png")
    if {"eth_yield", MAIN_RELATIVE_BASIS}.issubset(df.columns):
        df.plot.scatter(
            x="eth_yield",
            y=MAIN_RELATIVE_BASIS,
            figsize=(6, 5),
            title="Yield vs ETH-relative basis",
        )
        plt.axhline(0, linewidth=0.8)
        _savefig("yield_vs_relative_basis.png")
    if not basis_dislocation_regime_results.empty:
        yrows = basis_dislocation_regime_results[
            (basis_dislocation_regime_results["term"] == "eth_yield")
            & (
                basis_dislocation_regime_results["model"].isin(
                    ["low_basis_dislocation", "high_basis_dislocation"]
                )
            )
        ]
        if not yrows.empty:
            plt.figure(figsize=(7, 4))
            x = np.arange(len(yrows))
            plt.bar(x, yrows["coef"])
            plt.errorbar(
                x,
                yrows["coef"],
                yerr=1.96 * yrows["std_err_hac"],
                fmt="none",
                capsize=3,
            )
            plt.axhline(0, linewidth=0.8)
            plt.xticks(x, yrows["model"], rotation=15, ha="right")
            plt.title("Yield effect by basis-dislocation regime")
            _savefig("yield_effect_by_basis_regime.png")


plot_basis_figures()


# %%
def run_rolling_analysis() -> pd.DataFrame:
    if not RUN_ROLLING_ANALYSIS:
        out = pd.DataFrame()
        out.to_csv(TABLE_DIR / "rolling_results.csv", index=False)
        return out
    rows: list[dict[str, object]] = []
    outcomes = ["control_funding_mean", MAIN_RELATIVE_OUTCOME]
    for window in ROLLING_WINDOWS:
        for end_idx in range(window, len(df) + 1):
            sub = df.iloc[end_idx - window : end_idx]
            end_date = sub["date"].iloc[-1]
            for outcome in outcomes:
                try:
                    res, use = fit_ols_hac(
                        sub, outcome, ["eth_yield", *MODEL_CONTROLS], HAC_LAGS
                    )
                    rows.append(
                        {
                            "window": window,
                            "date": end_date.date().isoformat(),
                            "outcome": outcome,
                            "eth_yield_coef": res.params.get("eth_yield", np.nan),
                            "eth_yield_pvalue": res.pvalues.get("eth_yield", np.nan),
                            "nobs": int(res.nobs),
                            "adj_r2": res.rsquared_adj,
                            "sample_start": use["date"].min().date().isoformat(),
                            "sample_end": use["date"].max().date().isoformat(),
                        }
                    )
                except Exception:
                    continue
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "rolling_results.csv", index=False)
    return out


rolling_results = run_rolling_analysis()


# %%
def plot_rolling() -> None:
    if plt is None or rolling_results.empty:
        return
    plot_df = rolling_results.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    for outcome, fname, title in [
        (
            "control_funding_mean",
            "rolling_common_coef.png",
            "Rolling yield coefficient: common funding",
        ),
        (
            MAIN_RELATIVE_OUTCOME,
            "rolling_relative_coef.png",
            "Rolling yield coefficient: ETH-relative funding",
        ),
    ]:
        use = plot_df[plot_df["outcome"] == outcome]
        if use.empty:
            continue
        plt.figure(figsize=(11, 5))
        for window, grp in use.groupby("window"):
            plt.plot(grp["date"], grp["eth_yield_coef"], label=f"{window}d")
        plt.axhline(0, linewidth=0.8)
        plt.title(title)
        plt.legend()
        _savefig(fname)


plot_rolling()


# %%
def _significance_text(
    row: pd.Series | None, coef_col: str = "coef", p_col: str = "p_value"
) -> str:
    if row is None or row.empty:
        return "not estimated"
    coef = row.get(coef_col, np.nan)
    p = row.get(p_col, np.nan)
    direction = "positive" if coef > 0 else "negative" if coef < 0 else "zero"
    sig = (
        "statistically significant"
        if pd.notna(p) and p < 0.05
        else "not statistically significant"
    )
    return f"{direction} and {sig} (coef={_fmt(coef, 6)}, p={_fmt(p, 4)})"


def _answer_bool(table: pd.DataFrame, deps: list[str], term: str = "eth_yield") -> str:
    if table.empty:
        return "not estimated"
    rows = table[(table["term"] == term) & (table["dependent"].isin(deps))]
    if len(rows) < len(deps):
        return "partially estimated"
    return "yes" if bool((rows["coef"] > 0).all()) else "mixed/no"


def _basis_source_summary() -> str:
    source_cols = [c for c in df.columns if c.endswith("_basis_source")]
    if not source_cols:
        return "No basis source columns available."
    parts = []
    for col in source_cols:
        vals = sorted(v for v in df[col].dropna().unique())
        parts.append(f"`{col}`={','.join(vals) if vals else 'n/a'}")
    return "; ".join(parts)


def write_report() -> Path:
    individual_deps = [
        f"f_{a}" for a in ["eth", *AVAILABLE_CONTROLS] if f"f_{a}" in df.columns
    ]
    common_row = term_row(common_models, dependent="control_funding_mean")
    relative_row = term_row(relative_models, dependent=MAIN_RELATIVE_OUTCOME)
    feth_control_row = term_row(
        f_eth_control_models, model="f_eth_with_control_funding_mean"
    )
    basis_sanity_main = term_row(
        basis_sanity_checks, dependent=MAIN_RELATIVE_OUTCOME, term=MAIN_RELATIVE_BASIS
    )
    basis_adjust_main = None
    if not basis_adjusted_models.empty:
        use = basis_adjusted_models[
            basis_adjusted_models["outcome"] == MAIN_RELATIVE_OUTCOME
        ]
        if not use.empty:
            basis_adjust_main = use.iloc[0]
    mediation_main = None
    if not basis_mediation_results.empty:
        use = basis_mediation_results[
            basis_mediation_results["funding_outcome"] == MAIN_RELATIVE_OUTCOME
        ]
        if not use.empty:
            mediation_main = use.iloc[0]
    high_regime = term_row(
        basis_dislocation_regime_results,
        model="high_dislocation_interaction",
        term="eth_yield_x_high_basis_dislocation",
    )

    top_sensitivity = "No control-leg sensitivity estimates were available."
    if not control_leg_sensitivity.empty:
        subset = control_leg_sensitivity[
            ["control_set", "eth_yield_coef", "eth_yield_pvalue", "sign_interpretation"]
        ].copy()
        header = (
            "| control_set | eth_yield_coef | eth_yield_pvalue | sign_interpretation |"
        )
        sep = "|---|---:|---:|---|"
        body = [
            f"| {r.control_set} | {_fmt(r.eth_yield_coef, 6)} | {_fmt(r.eth_yield_pvalue, 4)} | {r.sign_interpretation} |"
            for r in subset.itertuples(index=False)
        ]
        top_sensitivity = "\n".join([header, sep, *body])

    report = f"""# ETH yield, common funding cycles, relative funding, and basis mechanism

## 1. Research question
Does wstETH/ETH yield behave like ETH-specific carry in perpetual funding, or does it mainly co-move with a crypto-wide funding cycle? The analysis separates three questions: whether `eth_yield` loads on broad funding legs, whether it explains ETH-relative funding after non-yielding control legs, and whether perp-spot basis mediates or washes out that relation.

## 2. Data
- Input CSV: `{INPUT_CSV}`
- Sample: {df['date'].min().date()} to {df['date'].max().date()}
- Observations: {len(df):,}
- Yield column: `{SELECTED_YIELD_COL}` with {df['eth_yield'].notna().sum():,} non-null observations
- Funding legs: {', '.join(dep.replace('f_', '').upper() for dep in individual_deps)}
- Return/volatility controls used: {MODEL_CONTROLS if MODEL_CONTROLS else 'none available'}
- Basis columns selected: {SELECTED_BASIS_COLS if SELECTED_BASIS_COLS else 'none'}
- Basis source metadata: {_basis_source_summary()}

Basis is not a yield. It is a perp-spot premium/dislocation variable computed as `perp_price / spot_close - 1`, with mark price preferred and futures close used only as fallback when mark price is unavailable.

## 3. Empirical framework
The core models estimate individual funding legs, non-yield control funding factors, ETH-relative funding, control-leg sensitivity, basis-funding sanity checks, basis-adjusted relative funding, and basis-dislocation regimes. OLS-HAC standard errors are used throughout. These are associations, not causal estimates.

## 4. Main result: individual funding legs
- Does `eth_yield` load positively on all estimated funding legs? **{_answer_bool(individual_models, individual_deps)}**.
- See `individual_funding_leg_models.csv` for leg-level coefficients.

## 5. Common funding result
- Does `eth_yield` load on non-yield control funding? `{_significance_text(common_row)}` for `control_funding_mean`.
- ETH funding controlling for common funding: `{_significance_text(feth_control_row)}`.

## 6. Relative funding and control-leg sensitivity
- Main relative outcome `{MAIN_RELATIVE_OUTCOME}`: `{_significance_text(relative_row)}`.
- Sign convention: positive supports dividend-like ETH carry relative to controls; negative supports hedged carry / ETH-perp-short pressure relative to controls.

Control-leg sensitivity summary:

{top_sensitivity}

The BTC-only comparison can differ from XRP/DOGE and basket controls. Therefore, the relative ETH funding effect should be interpreted as control-set sensitive rather than a single unconditional ETH-specific carry estimate.

## 7. Basis sanity and mechanism
- Does basis explain relative funding? `{_significance_text(basis_sanity_main)}`
"""
    if basis_adjust_main is not None:
        report += f"- Basis-adjusted main model: yield coefficient changes from {_fmt(basis_adjust_main['eth_yield_coef_no_basis'], 6)} to {_fmt(basis_adjust_main['eth_yield_coef_with_basis'], 6)}; basis coefficient is {_fmt(basis_adjust_main['basis_coef'], 6)} with p={_fmt(basis_adjust_main['basis_pvalue'], 4)}. Interpretation: {basis_adjust_main['interpretation']}.\n"
    else:
        report += "- Basis-adjusted main model: not estimated because basis data were unavailable or insufficient.\n"
    if mediation_main is not None:
        report += f"- Mediation diagnostic for main outcome: yield-to-basis p={_fmt(mediation_main['yield_to_basis_p'], 4)}, basis-to-funding p={_fmt(mediation_main['basis_to_funding_p'], 4)}, possible mediation flag={mediation_main['possible_mediation_flag']}.\n"
    else:
        report += "- Mediation diagnostic: not estimated.\n"
    report += f"- Does high basis dislocation wash out the yield effect? Interaction diagnostic: {_significance_text(high_regime)}.\n"

    report += f"""
Important caveat: basis may be a mediator, not merely a control. No-basis regressions estimate total association; basis-adjusted regressions estimate direct association conditional on current perp-spot dislocation.

## 8. Rolling diagnostics
Rolling windows {ROLLING_WINDOWS} are diagnostic only, not identification. They focus on `control_funding_mean` and `{MAIN_RELATIVE_OUTCOME}`. See `rolling_results.csv` and rolling coefficient figures.

## 9. Interpretation
The evidence should not be read as a simple unconditional dividend-like pricing story in which wstETH yield mechanically raises ETH funding relative to non-yielding crypto assets. Instead, wstETH yield can load on a crypto-wide funding cycle. After removing non-yielding control funding, the relative ETH funding response is sensitive to the chosen control legs and may turn negative, consistent with hedged carry or LST-long/perp-short pressure. Basis-adjusted tests clarify whether this relative effect is mediated by perp-spot dislocation or whether basis dislocation masks/reveals yield pricing.

## 10. Limitations
- BTC, XRP, and DOGE are imperfect controls with their own liquidity and narrative shocks.
- Basis data may use mark price or futures close fallback; source metadata should be inspected.
- OLS-HAC estimates are associative, not causal.
- wstETH yield can include MEV, fee, and activity components rather than a clean exogenous dividend.
- Funding and basis can be jointly determined.

## 11. Conclusion
The revised thesis is that wstETH yield is not cleanly priced as a simple dividend-like carry in ETH perps. Its relationship with funding is dominated by common crypto funding conditions and basis/hedged-carry mechanisms.

## Reproduction
Build the panel with basis:
```bash
python scripts/build_funding_yield_panel.py \\
  --start-date 2023-01-01 \\
  --end-date 2026-05-09 \\
  --exchange binance \\
  --assets BTC ETH XRP DOGE \\
  --fetch-onchain-lst-rates \\
  --ethereum-rpc-url "$ETHEREUM_RPC_URL" \\
  --fetch-basis \\
  --out-csv data/processed/funding_yield_panel.csv
```

Run this analysis:
```bash
python notebooks/eth_yield_control_assets_analysis.py
```

Event interaction analysis is disabled by default (`RUN_EVENT_ANALYSIS = False`) and is not part of the main report.
"""
    path = REPORT_DIR / "eth_yield_control_assets_report.md"
    path.write_text(report)
    print(f"Wrote {path}")
    return path


report_path = write_report()
