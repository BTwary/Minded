import numpy as np
import pandas as pd
from packages.analytics_core.src.statistics.inference import execute_regression


def test_hc3_for_heteroskedastic_data():
    rng = np.random.default_rng(42)
    x = np.linspace(1, 100, 250)
    y = 2.5 * x + rng.normal(0, x * 0.6)
    res = execute_regression(pd.DataFrame({"x": x, "y": y}), ["x"], "y")
    assert res["problem"] == "regression_inference"
    assert res["covariance"]["method"] in {"HC3", "classical"}
    assert "heteroskedasticity" in res["diagnostics"]
    assert "formula" in res


def test_multicollinearity_is_flagged():
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=200)
    x2 = x1 + rng.normal(0, 0.0001, 200)
    y = 3 * x1 - 2 * x2 + rng.normal(0, 0.5, 200)
    res = execute_regression(pd.DataFrame({"x1": x1, "x2": x2, "y": y}), ["x1", "x2"], "y")
    assert max(res["diagnostics"]["multicollinearity"]["vif"].values()) > 10
    assert res["assumption_status"]["multicollinearity"] == "HIGH_RISK"


def test_clustered_covariance_is_used_when_group_supplied():
    rng = np.random.default_rng(11)
    groups = np.repeat(np.arange(25), 12)
    x = rng.normal(size=len(groups))
    group_effect = rng.normal(0, 1.5, size=25)[groups]
    y = 1.8 * x + group_effect + rng.normal(0, 0.3, len(groups))
    df = pd.DataFrame({"x": x, "y": y, "cluster": groups})
    res = execute_regression(df, ["x"], "y", cluster_column="cluster")
    assert res["covariance"]["method"] == "clustered"
    assert "cluster" in res["covariance"]["reason"]


def test_time_dependence_enables_hac():
    rng = np.random.default_rng(21)
    n = 180
    eps = np.zeros(n)
    noise = rng.normal(0, 0.3, n)
    for i in range(1, n):
        eps[i] = 0.8 * eps[i-1] + noise[i]
    x = np.linspace(0, 10, n)
    y = 1.2 * x + eps
    df = pd.DataFrame({"x": x, "y": y, "time": pd.date_range("2025-01-01", periods=n, freq="D")})
    res = execute_regression(df, ["x"], "y", time_column="time")
    assert "dependence_screen" in res["diagnostics"]
    assert res["covariance"]["method"] in {"HAC/Newey-West", "classical", "HC3"}


def test_traceable_assumption_fields_exist():
    x = np.arange(1, 30, dtype=float)
    y = 4 * x + 2
    res = execute_regression(pd.DataFrame({"x": x, "y": y}), ["x"], "y")
    for key in ("formula", "covariance", "diagnostics", "assumption_status", "warnings", "limitations"):
        assert key in res
