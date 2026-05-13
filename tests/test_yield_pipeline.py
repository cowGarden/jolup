from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from common.yield_pipeline import (
    add_implied_yields_from_rate,
    validate_wsteth_rate_shape,
)


class YieldPipelineTests(unittest.TestCase):
    def test_add_implied_yields_from_rate_uses_log_annualized_units(self):
        daily_log_growth = 0.0001
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "wsteth_rate": np.exp(daily_log_growth * np.arange(len(dates))),
            }
        )

        out = add_implied_yields_from_rate(df, windows=(7,))

        expected = 365.0 * daily_log_growth
        self.assertAlmostEqual(
            out.loc[7, "wsteth_implied_yield_7d"], expected, places=12
        )
        self.assertTrue(pd.isna(out.loc[6, "wsteth_implied_yield_7d"]))

    def test_defillama_fallback_does_not_create_wsteth_implied_yields(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "build_funding_yield_panel.py"
        )
        spec = importlib.util.spec_from_file_location(
            "build_funding_yield_panel", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        tmp = Path("/tmp/defillama_fallback_test.csv")
        tmp.write_text("date,apy\n2024-01-01,0.035\n")
        try:
            out = module.load_fallback_yield(tmp)
        finally:
            tmp.unlink(missing_ok=True)

        self.assertIn("wsteth_defillama_apy", out.columns)
        self.assertNotIn("wsteth_implied_yield_1d", out.columns)
        self.assertNotIn("wsteth_implied_yield_7d", out.columns)
        self.assertNotIn("wsteth_implied_yield_30d", out.columns)

    def test_build_funding_panel_supports_xrp_doge_alias_columns(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "build_funding_yield_panel.py"
        )
        spec = importlib.util.spec_from_file_location(
            "build_funding_yield_panel", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class Args:
            start_date = "2024-01-01"
            end_date = "2024-01-02"
            exchange = "binance"
            assets = ["BTC", "ETH", "XRP", "DOGE"]
            input_dir = Path("/tmp/jolup_test_processed")
            raw_dir = Path("/tmp/jolup_test_raw")

        Args.input_dir.mkdir(exist_ok=True)
        try:
            for asset in Args.assets:
                symbol = module.asset_to_symbol(asset)
                (Args.input_dir / f"binance_{symbol}_funding_daily.csv").write_text(
                    "date,funding_ann\n2024-01-01,0.01\n2024-01-02,0.02\n"
                )
            out = module.build_funding_panel(Args)
        finally:
            for path in Args.input_dir.glob("*.csv"):
                path.unlink()
            Args.input_dir.rmdir()

        for col in ["btc_funding", "eth_funding", "xrp_funding", "doge_funding"]:
            self.assertIn(col, out.columns)
        self.assertEqual(module.asset_to_symbol("XRP"), "XRPUSDT")
        self.assertEqual(module.asset_to_symbol("DOGE"), "DOGEUSDT")

    def test_build_basis_panel_uses_mark_price_when_available(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "build_funding_yield_panel.py"
        )
        spec = importlib.util.spec_from_file_location(
            "build_funding_yield_panel", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            input_dir = root / "processed"
            raw_dir.mkdir()
            input_dir.mkdir()

            class Args:
                start_date = "2024-01-01"
                end_date = "2024-01-02"
                exchange = "binance"
                assets = ["BTC", "ETH"]
                skip_fetch_missing = True
                fetch_basis = True
                basis_source = "auto"
                request_sleep_seconds = 0.0

            Args.raw_dir = raw_dir
            Args.input_dir = input_dir

            for symbol in ["btcusdt", "ethusdt"]:
                (raw_dir / f"binance_{symbol}_1d.csv").write_text(
                    "date,open,high,low,close,volume\n"
                    "2024-01-01,100,101,99,100,1\n"
                    "2024-01-02,100,102,99,101,1\n"
                )
                (raw_dir / f"binance_{symbol}_mark_1d.csv").write_text(
                    "date,open,high,low,close,volume\n"
                    "2024-01-01,100.1,101.1,99.1,100.2,1\n"
                    "2024-01-02,100.2,102.2,99.2,101.3,1\n"
                )

            out = module.build_basis_panel(Args)

        self.assertIn("btc_basis_close", out.columns)
        self.assertIn("eth_perp_mark", out.columns)
        self.assertEqual(
            out["btc_basis_source"].dropna().unique().tolist(), ["mark_price"]
        )
        self.assertAlmostEqual(out.loc[out.index[0], "btc_basis_close"], 0.002)

    def test_market_like_csv_validation_warning(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=8, freq="D"),
                "wsteth_rate": [1.001, 0.999, 1.002, 0.998, 1.001, 1.000, 0.999, 1.000],
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diagnostics = validate_wsteth_rate_shape(df)

        self.assertGreater(diagnostics["negative_steps"], 0)
        self.assertTrue(any("market price ratio" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
