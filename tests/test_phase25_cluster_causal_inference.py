import numpy as np
import pandas as pd
import pytest

from packages.analytics_core.src.causal.effect_estimation import estimate_binary_treatment_effect


def make_clustered(seed=101, n_clusters=40, repeats=3):
    rng = np.random.default_rng(seed)
    rows = []
    for cluster in range(n_clusters):
        treated = int(cluster % 2 == 0)
        age = 35 + rng.normal(0, 4)
        cluster_effect = rng.normal(0, 0.7)
        for t_idx in range(repeats):
            outcome = 8.0 + 0.04 * age + 2.0 * treated + cluster_effect + rng.normal(0, 0.35)
            rows.append({
                "customer_id": f"C{cluster:03d}",
                "period": t_idx,
                "age": age + rng.normal(0, 0.2),
                "treated": treated,
                "outcome": outcome,
            })
    return pd.DataFrame(rows)


def test_cluster_aware_aipw_uses_cluster_crossfit_and_bootstrap():
    df = make_clustered()
    res = estimate_binary_treatment_effect(
        df,
        treatment_col="treated",
        outcome_col="outcome",
        adjustment_set=["age"],
        estimator="AIPW",
        bootstrap_resamples=12,
        cluster_col="customer_id",
        time_col="period",
    )
    assert 1.4 < res.effect < 2.6
    assert res.diagnostics["dependence"]["cluster_aware"] is True
    assert res.diagnostics["dependence"]["repeated_observations"] is True
    assert res.diagnostics["dependence"]["n_clusters"] == 40
    assert res.reproducibility["bootstrap_unit"] == "cluster"
    assert res.diagnostics["nuisance"]["crossfit_mode"] == "stratified_clusters"


def test_time_varying_treatment_fails_closed_under_point_treatment_estimator():
    df = make_clustered(n_clusters=20, repeats=2)
    df.loc[df["customer_id"].isin(["C000", "C001"]), "treated"] = [0, 1, 0, 1]
    with pytest.raises(ValueError, match="Time-varying treatment detected"):
        estimate_binary_treatment_effect(
            df,
            treatment_col="treated",
            outcome_col="outcome",
            adjustment_set=["age"],
            estimator="AIPW",
            bootstrap_resamples=0,
            cluster_col="customer_id",
            time_col="period",
        )


def test_cluster_col_missing_values_fail_closed():
    df = make_clustered(n_clusters=20, repeats=2)
    df.loc[0, "customer_id"] = np.nan
    with pytest.raises(ValueError, match="cluster_col contains missing values"):
        estimate_binary_treatment_effect(
            df,
            treatment_col="treated",
            outcome_col="outcome",
            adjustment_set=["age"],
            estimator="AIPW",
            bootstrap_resamples=0,
            cluster_col="customer_id",
        )
