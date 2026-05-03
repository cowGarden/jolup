from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def ols_hac(y: pd.Series, X: pd.DataFrame, maxlags: int = 5):
    X_ = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_, missing="drop")
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
