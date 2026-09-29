import numpy as np

from packages.analytics_core.src.causal.longitudinal import estimate_longitudinal_msm
from tests.test_phase26_longitudinal_causal import make_longitudinal


def make_binary(seed=101, n=60, periods=2):
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n):
        age = 40 + rng.normal(0, 4)
        prev = 0
        score = rng.normal(0, 0.2)
        for t in range(periods):
            conf = 0.05 * age + 0.4 * prev + rng.normal(0, 0.2)
            p = 1 / (1 + np.exp(-(-0.4 + 0.02 * age + 0.3 * prev + 0.1 * conf)))
            a = int(rng.random() < p)
            py = 1 / (1 + np.exp(-(-2.0 + 0.04 * age + 0.8 * a + 0.2 * prev + score)))
            y = int(rng.random() < py)
            rows.append({"unit": unit, "time": t, "age": age, "conf": conf, "treated": a, "outcome": y})
            prev = a
    return __import__('pandas').DataFrame(rows)


def test_binary_longitudinal_msm_exposes_quantitative_sensitivity():
    df = make_binary()
    res = estimate_longitudinal_msm(
        df, unit_col='unit', time_col='time', treatment_col='treated', outcome_col='outcome',
        baseline_covariates=['age'], time_varying_covariates=['conf'], outcome_type='binary',
        bootstrap_resamples=0,
    )
    sens = res.model_diagnostics['sensitivity']
    assert sens['available'] is True
    assert sens['quantitative_confounding_grid']['available'] is True
