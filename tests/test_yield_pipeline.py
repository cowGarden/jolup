from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
import warnings

import numpy as np
import pandas as pd

from common.yield_pipeline import add_implied_yields_from_rate, validate_wsteth_rate_shape


class YieldPipelineTests(unittest.TestCase):
    def test_add_implied_yields_from_rate_uses_log_annualized_units(self):
        daily_log_growth = 0.0001
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "wsteth_rate": np.exp(daily_log_growth * np.arange(len(dates))),
        })

        out = add_implied_yields_from_rate(df, windows=(7,))

        expected = 365.0 * daily_log_growth
        self.assertAlmostEqual(out.loc[7, "wsteth_implied_yield_7d"], expected, places=12)
        self.assertTrue(pd.isna(out.loc[6, "wsteth_implied_yield_7d"]))

    def test_defillama_fallback_does_not_create_wsteth_implied_yields(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_funding_yield_panel.py"
        spec = importlib.util.spec_from_file_location("build_funding_yield_panel", script_path)
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

    def test_market_like_csv_validation_warning(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "wsteth_rate": [1.001, 0.999, 1.002, 0.998, 1.001, 1.000, 0.999, 1.000],
        })

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diagnostics = validate_wsteth_rate_shape(df)

        self.assertGreater(diagnostics["negative_steps"], 0)
        self.assertTrue(any("market price ratio" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
