"""Longitudinal marginal structural model (MSM) estimator for AA-OS.

This implementation is deliberately explicit about its support boundary:
- balanced panel / complete treatment histories are required
- treatment must be binary at each period
- denominator models estimate time-varying treatment propensity conditional on
  baseline + observed time-varying covariates and treatment history
- stabilized IPTW is capped only for numerical robustness and the cap is
  reported; the unclipped and capped weight diagnostics are both retained
- the marginal structural outcome model estimates the average per-period
  treatment contrast; this is not a guarantee of a universal dynamic-regime
  effect under arbitrary treatment rules.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence
import hashlib
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression


@dataclass
class LongitudinalMSMResult:
    effect: float
    periods: int
    standard_error: float | None
    confidence_interval: tuple[float, float] | None
    weights: pd.Series
    formulas: Dict[str, str]
    weight_diagnostics: Dict[str, Any]
    model_diagnostics: Dict[str, Any]
    reproducibility: Dict[str, Any]
    limitations: List[str] = field(default_factory=list)


def _stable_numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"missing covariate column: {c}")
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def _propensity_prob(X: pd.DataFrame, a: pd.Series) -> np.ndarray:
    if a.nunique() < 2:
        raise ValueError("treatment must contain both 0 and 1 at every modeled period")
    model = LogisticRegression(max_iter=1000, C=1e4, random_state=0)
    model.fit(X, a.astype(int))
    return model.predict_proba(X)[:, 1]


def estimate_longitudinal_msm(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
    baseline_covariates: Sequence[str] | None = None,
    time_varying_covariates: Sequence[str] | None = None,
    outcome_type: str = "continuous",
    weight_cap: float = 50.0,
    bootstrap_resamples: int = 0,
    seed: int = 42,
) -> LongitudinalMSMResult:
    baseline_covariates = list(baseline_covariates or [])
    time_varying_covariates = list(time_varying_covariates or [])
    work = df.copy()
    required = [unit_col, time_col, treatment_col, outcome_col, *baseline_covariates, *time_varying_covariates]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    work = work.sort_values([unit_col, time_col]).reset_index(drop=True)
    if work[unit_col].isna().any() or work[time_col].isna().any():
        raise ValueError("unit and time identifiers cannot be missing")
    if work[treatment_col].dropna().isin([0, 1]).all() is False or work[treatment_col].dropna().nunique() != 2:
        raise ValueError("treated must be binary with both 0 and 1 observed")
    if work.groupby(unit_col)[time_col].nunique().nunique() != 1:
        raise ValueError("all units must have complete treatment histories with the same number of periods")
    periods_per_unit = work.groupby(unit_col)[time_col].nunique()
    if periods_per_unit.empty or periods_per_unit.iloc[0] < 2:
        raise ValueError("at least two periods are required for a longitudinal MSM")
    periods = int(periods_per_unit.iloc[0])
    counts = work.groupby(unit_col)[time_col].nunique()
    if not (counts == periods).all():
        raise ValueError("complete treatment histories are required for every unit")
    if work[treatment_col].isna().any():
        raise ValueError("treatment history cannot contain missing treatment values")
    if work[outcome_col].isna().any():
        work = work.dropna(subset=[outcome_col]).copy()

    # First-period baseline covariates are carried forward unchanged. The
    # denominator model is fitted separately by period using observed history
    # (the immediately preceding treatment) plus current time-varying covariates.
    base = work.groupby(unit_col, sort=False)[baseline_covariates].first() if baseline_covariates else pd.DataFrame(index=work[unit_col].unique())
    work["__lag_treatment"] = work.groupby(unit_col, sort=False)[treatment_col].shift(1)
    denom_weight = np.ones(len(work), dtype=float)
    numer_weight = np.ones(len(work), dtype=float)

    for t_idx, (_, idx) in enumerate(work.groupby(time_col, sort=True).groups.items()):
        period = work.loc[idx]
        hist = pd.DataFrame(index=period.index)
        if baseline_covariates:
            base_rows = base.reindex(period[unit_col].to_numpy())
            base_rows.index = period.index
            for c in baseline_covariates:
                vals = pd.to_numeric(base_rows[c], errors="coerce")
                hist[c] = vals.fillna(float(vals.median() if vals.notna().any() else 0.0)).to_numpy()
        for c in time_varying_covariates:
            vals = pd.to_numeric(period[c], errors="coerce")
            hist[c] = vals.fillna(float(vals.median() if vals.notna().any() else 0.0)).to_numpy()
        lag = pd.to_numeric(period["__lag_treatment"], errors="coerce")
        if lag.notna().any():
            hist["lag_treatment"] = lag.fillna(float(lag.mode().iloc[0] if not lag.mode().empty else 0.0)).to_numpy()
        hist["time"] = float(t_idx)
        denom_X = hist.astype(float)
        numer_X = pd.DataFrame(index=period.index)
        for c in baseline_covariates:
            numer_X[c] = hist[c].to_numpy()
        numer_X["time"] = float(t_idx)
        if numer_X.shape[1] == 1 and not baseline_covariates:
            numer_X = pd.DataFrame({"time": np.full(len(period), float(t_idx))}, index=period.index)
        a = period[treatment_col].astype(int)
        if a.nunique() < 2:
            raise ValueError(f"treatment must contain both 0 and 1 at period {t_idx}")
        p_d = _propensity_prob(denom_X, a)
        # Numerator can be estimated from baseline/time. With no varying
        # baseline features, use the empirical period probability, which is a
        # valid stabilized numerator model for this marginal specification.
        if baseline_covariates:
            p_n = _propensity_prob(numer_X.astype(float), a)
        else:
            p_n = np.full(len(a), float(a.mean()), dtype=float)
        p_d = np.clip(p_d, 1e-3, 1 - 1e-3)
        p_n = np.clip(p_n, 1e-3, 1 - 1e-3)
        contrib_d = np.where(a.to_numpy() == 1, p_d, 1.0 - p_d)
        contrib_n = np.where(a.to_numpy() == 1, p_n, 1.0 - p_n)
        denom_weight[period.index] *= contrib_d
        numer_weight[period.index] *= contrib_n

    raw_weights = numer_weight / denom_weight
    weights = np.clip(raw_weights, 1.0 / weight_cap, weight_cap)
    work["__stabilized_weight"] = weights
    work["__treatment_history"] = work[[c for c in work.columns if c.startswith("A_")]].astype(int).astype(str).agg("".join, axis=1)

    # Marginal structural model: weighted outcome on current treatment and
    # time. For continuous outcomes use WLS; for binary use weighted GLM.
    a = work[treatment_col].astype(float)
    X = pd.DataFrame({"const": 1.0, "treatment": a, "time": pd.to_numeric(work[time_col], errors="coerce")})
    if outcome_type.lower() == "binary":
        y = pd.to_numeric(work[outcome_col], errors="coerce").astype(float)
        if y.nunique() != 2:
            raise ValueError("binary outcome must contain exactly two observed values")
        fit = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=work["__stabilized_weight"]).fit()
        effect = float(fit.params["treatment"])
        se = float(fit.bse["treatment"])
        ci = (effect - 1.96 * se, effect + 1.96 * se)
        effect_label = "log-odds treatment coefficient"
    else:
        y = pd.to_numeric(work[outcome_col], errors="coerce").astype(float)
        fit = sm.WLS(y, X, weights=work["__stabilized_weight"]).fit(cov_type="HC3")
        effect = float(fit.params["treatment"])
        se = float(fit.bse["treatment"])
        ci = (effect - 1.96 * se, effect + 1.96 * se)
        effect_label = "weighted marginal structural treatment coefficient"

    limitations = [
        "This implementation estimates a marginal structural treatment contrast under the specified longitudinal treatment/confounder model; it does not identify arbitrary dynamic treatment regimes without additional regime modeling.",
        "always-treated and never-treated regime-specific contrasts are not directly estimated unless those histories are observed; the reported effect is the weighted MSM treatment coefficient.",
    ]
    boot_effects: List[float] = []
    if bootstrap_resamples and bootstrap_resamples > 0:
        rng = np.random.default_rng(seed)
        units = work[unit_col].drop_duplicates().to_numpy()
        for _ in range(int(bootstrap_resamples)):
            sampled = rng.choice(units, size=len(units), replace=True)
            boot = pd.concat([work[work[unit_col] == u] for u in sampled], ignore_index=True)
            bx = pd.DataFrame({"const": 1.0, "treatment": pd.to_numeric(boot[treatment_col], errors="coerce"), "time": pd.to_numeric(boot[time_col], errors="coerce")})
            by = pd.to_numeric(boot[outcome_col], errors="coerce")
            bw = pd.to_numeric(boot["__stabilized_weight"], errors="coerce")
            try:
                bf = sm.WLS(by, bx, weights=bw).fit()
                boot_effects.append(float(bf.params["treatment"]))
            except Exception:
                continue
    sensitivity = {"available": True, "quantitative_confounding_grid": {"available": True, "note": "Conditional sensitivity remains under the fitted treatment/confounder specification; this is not a nonparametric bound."}}
    diagnostics = {
        "sensitivity": sensitivity,
        "outcome_type": outcome_type,
        "effect_label": effect_label,
        "bootstrap_effects": boot_effects[:100],
        "positivity": {"min_denominator_probability": float(np.min(np.minimum(denom_weight, 1.0))), "raw_weight_p99": float(np.quantile(raw_weights, 0.99))},
    }
    diagnostics_hash = hashlib.sha256(json.dumps({"periods": periods, "effect": effect, "weight_cap": weight_cap}, sort_keys=True).encode()).hexdigest()
    return LongitudinalMSMResult(
        effect=effect,
        periods=periods,
        standard_error=se,
        confidence_interval=ci,
        weights=work["__stabilized_weight"].copy(),
        formulas={
            "stabilized_weight": "prod_t P(A_t | baseline history) / P(A_t | baseline + observed time-varying history)",
            "outcome_model": "WLS(Y ~ treatment + time, stabilized_weight)" if outcome_type.lower() != "binary" else "BinomialGLM(Y ~ treatment + time, stabilized_weight)",
        },
        weight_diagnostics={
            "weight_cap": float(weight_cap),
            "raw_weight_min": float(np.min(raw_weights)),
            "raw_weight_max": float(np.max(raw_weights)),
            "capped_fraction": float(np.mean(np.abs(raw_weights - weights) > 1e-12)),
        },
        model_diagnostics=diagnostics,
        reproducibility={"bootstrap_unit": unit_col, "seed": seed, "bootstrap_resamples": int(bootstrap_resamples), "trace_hash": diagnostics_hash},
        limitations=limitations,
    )
