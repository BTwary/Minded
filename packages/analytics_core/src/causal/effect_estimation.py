"""Observed-data causal effect estimation after identification.

Implements conservative ATE/ATT estimation for binary treatment under an
already-validated adjustment set. The module never performs identification;
that responsibility remains with CausalIdentifiabilityGate.

Estimators:
- outcome-regression g-computation (G-formula)
- inverse-probability weighting (IPW)
- augmented IPW / doubly robust (AIPW)

The result includes positivity/overlap diagnostics, balance diagnostics,
bootstrap uncertainty, and an audit-ready formula/assumption payload.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from .sensitivity import e_value_report

try:
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


@dataclass(frozen=True)
class CausalEffectEstimate:
    estimand: str
    estimator: str
    effect: float
    confidence_interval: List[float]
    standard_error: Optional[float]
    n: int
    treated_n: int
    control_n: int
    positivity: Dict[str, Any]
    balance: Dict[str, Any]
    formulas: Dict[str, str]
    assumptions: List[str]
    limitations: List[str]
    diagnostics: Dict[str, Any]
    reproducibility: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimand": self.estimand,
            "estimator": self.estimator,
            "effect": self.effect,
            "confidence_interval": self.confidence_interval,
            "standard_error": self.standard_error,
            "n": self.n,
            "treated_n": self.treated_n,
            "control_n": self.control_n,
            "positivity": self.positivity,
            "balance": self.balance,
            "formulas": self.formulas,
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "diagnostics": self.diagnostics,
            "reproducibility": self.reproducibility,
        }


def _design_matrix(df: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    if not covariates:
        return pd.DataFrame({"_intercept": np.ones(len(df), dtype=float)}, index=df.index)
    X = df[covariates].copy()
    for col in list(X.columns):
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X[col] = X[col].astype("string")
    X = pd.get_dummies(X, drop_first=True, dummy_na=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(float)


def _validate_binary_treatment(t: pd.Series) -> None:
    vals = set(pd.Series(t).dropna().unique().tolist())
    if vals - {0, 1}:
        raise ValueError("Causal effect estimation currently requires a binary treatment coded 0/1.")
    if len(vals) != 2:
        raise ValueError("Both treatment groups must be observed.")


def _propensity_scores(X: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for propensity-score estimation in the current implementation.")
    model = LogisticRegression(max_iter=2000, random_state=20260908)
    model.fit(X, t)
    return model.predict_proba(X)[:, 1]


def _standardized_mean_differences(X: pd.DataFrame, t: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if weights is None:
        weights = np.ones(len(X), dtype=float)
    for col in X.columns:
        x = X[col].to_numpy(dtype=float)
        treated = t == 1
        control = t == 0
        def wmean(mask: np.ndarray) -> float:
            return float(np.average(x[mask], weights=weights[mask])) if np.sum(mask) else np.nan
        mt, mc = wmean(treated), wmean(control)
        vt = np.average((x[treated] - mt) ** 2, weights=weights[treated]) if np.sum(treated) else np.nan
        vc = np.average((x[control] - mc) ** 2, weights=weights[control]) if np.sum(control) else np.nan
        denom = np.sqrt(max((vt + vc) / 2.0, 1e-16))
        out[col] = float((mt - mc) / denom) if np.isfinite(denom) else 0.0
    return out


def _positivity(ps: np.ndarray, t: np.ndarray) -> Dict[str, Any]:
    clipped = np.clip(ps, 1e-6, 1 - 1e-6)
    treated = clipped[t == 1]
    control = clipped[t == 0]
    overlap_low = max(float(np.min(treated)) if len(treated) else 0.0, float(np.min(control)) if len(control) else 0.0)
    overlap_high = min(float(np.max(treated)) if len(treated) else 1.0, float(np.max(control)) if len(control) else 1.0)
    extreme = int(np.sum((clipped < 0.05) | (clipped > 0.95)))
    common = max(0.0, overlap_high - overlap_low)
    status = "ADEQUATE" if common > 0.05 and extreme / len(clipped) < 0.20 else "CAUTION"
    return {
        "status": status,
        "propensity_min": float(np.min(clipped)),
        "propensity_max": float(np.max(clipped)),
        "common_support_width": float(common),
        "extreme_score_count": extreme,
        "extreme_score_fraction": float(extreme / len(clipped)) if len(clipped) else 1.0,
        "warning": "Strong positivity violations can make causal estimates unstable; no automatic trimming is applied by default.",
    }


def _outcome_predictions(X: pd.DataFrame, t: np.ndarray, y: np.ndarray, outcome_type: str) -> tuple[np.ndarray, np.ndarray]:
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for outcome regression.")
    y1 = np.empty(len(y), dtype=float)
    y0 = np.empty(len(y), dtype=float)
    for arm, target in ((1, y1), (0, y0)):
        mask = t == arm
        if np.sum(mask) < max(10, X.shape[1] + 2):
            raise ValueError("Insufficient observations for outcome regression within a treatment arm.")
        if outcome_type == "binary":
            model = LogisticRegression(max_iter=2000, random_state=20260908).fit(X.loc[mask], y[mask].astype(int))
            target[:] = model.predict_proba(X)[:, 1]
        else:
            model = LinearRegression().fit(X.loc[mask], y[mask])
            target[:] = model.predict(X)
    return y1, y0


def _cross_fitted_nuisance(
    X: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    folds: int = 5,
    outcome_type: str = "continuous",
    groups: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for cross-fitted AIPW.")
    min_class = int(min(np.sum(t == 0), np.sum(t == 1)))
    if groups is None:
        n_splits = min(folds, min_class)
        if n_splits < 2:
            raise ValueError("At least 2 observations per treatment arm are required for cross-fitted nuisance models.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20260908)
        split_iter = splitter.split(X, t)
        crossfit_mode = "stratified_rows"
    else:
        unique_groups = np.unique(groups)
        n_splits = min(folds, len(unique_groups))
        if n_splits < 2:
            raise ValueError("At least 2 clusters are required for cluster-aware cross-fitting.")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=20260908)
        split_iter = splitter.split(X, t, groups=groups)
        crossfit_mode = "stratified_clusters"
    ehat = np.empty(len(y), dtype=float)
    mu1 = np.empty(len(y), dtype=float)
    mu0 = np.empty(len(y), dtype=float)
    for train_idx, test_idx in split_iter:
        if len(np.unique(t[train_idx])) < 2 or len(np.unique(t[test_idx])) < 2:
            raise ValueError("Cluster-aware cross-fitting produced a fold without both treatment arms; increase the number of clusters or provide better treatment allocation.")
        ps_model = LogisticRegression(max_iter=2000, random_state=20260908).fit(X.iloc[train_idx], t[train_idx])
        ehat[test_idx] = ps_model.predict_proba(X.iloc[test_idx])[:, 1]
        for arm, target in ((1, mu1), (0, mu0)):
            arm_idx = train_idx[t[train_idx] == arm]
            if len(arm_idx) < max(10, X.shape[1] + 2):
                raise ValueError("Insufficient training observations within one treatment arm for cross-fitted outcome models.")
            if outcome_type == "binary":
                om = LogisticRegression(max_iter=2000, random_state=20260908).fit(X.iloc[arm_idx], y[arm_idx].astype(int))
                target[test_idx] = om.predict_proba(X.iloc[test_idx])[:, 1]
            else:
                om = LinearRegression().fit(X.iloc[arm_idx], y[arm_idx])
                target[test_idx] = om.predict(X.iloc[test_idx])
    return np.clip(ehat, 1e-3, 1 - 1e-3), mu1, mu0


def estimate_binary_treatment_effect(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    adjustment_set: List[str],
    estimand: str = "ATE",
    estimator: str = "AIPW",
    confidence: float = 0.95,
    bootstrap_resamples: int = 1000,
    seed: int = 20260908,
    outcome_type: str = "auto",
    cluster_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> CausalEffectEstimate:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    optional = [c for c in (cluster_col, time_col) if c]
    required = [treatment_col, outcome_col] + optional + list(dict.fromkeys(adjustment_set))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing causal estimation columns: {missing}")
    work = df[required].copy()
    work[treatment_col] = pd.to_numeric(work[treatment_col], errors="coerce")
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[treatment_col, outcome_col]).reset_index(drop=True)
    dependence: Dict[str, Any] = {"cluster_col": cluster_col, "time_col": time_col, "cluster_aware": bool(cluster_col)}
    if cluster_col:
        cluster_missing = work[cluster_col].isna()
        if bool(cluster_missing.any()):
            raise ValueError("cluster_col contains missing values; cluster-aware inference requires a complete cluster identifier.")
        group_nunique_t = work.groupby(cluster_col, dropna=False)[treatment_col].nunique()
        switching = group_nunique_t[group_nunique_t > 1]
        if not switching.empty:
            raise ValueError("Time-varying treatment detected within clusters. The current estimator is a point-treatment causal estimator and does not support longitudinal treatment regimes; use a longitudinal marginal structural model or another time-varying-treatment estimator.")
        cluster_sizes = work.groupby(cluster_col, dropna=False).size().to_numpy(dtype=int)
        dependence.update({
            "n_clusters": int(len(cluster_sizes)),
            "repeated_observations": bool(len(cluster_sizes) < len(work)),
            "min_cluster_size": int(cluster_sizes.min()),
            "median_cluster_size": float(np.median(cluster_sizes)),
            "max_cluster_size": int(cluster_sizes.max()),
            "treatment_switching_clusters": 0,
            "treatment_constant_within_cluster": True,
            "warning": "Repeated observations are treated as dependent units; uncertainty is cluster-bootstrap based. If the estimand is inherently longitudinal or time-varying, this point-treatment estimator is not sufficient.",
        })
    else:
        dependence.update({"n_clusters": len(work), "repeated_observations": False})
    _validate_binary_treatment(work[treatment_col])
    if len(work) < 40:
        raise ValueError("At least 40 complete observations are required for stable causal estimation.")
    t = work[treatment_col].to_numpy(dtype=int)
    y = work[outcome_col].to_numpy(dtype=float)
    unique_y = np.unique(y)
    if outcome_type not in {"auto", "continuous", "binary"}:
        raise ValueError("outcome_type must be auto, continuous, or binary")
    if outcome_type == "auto":
        outcome_type = "binary" if len(unique_y) == 2 and set(unique_y.tolist()).issubset({0.0, 1.0}) else "continuous"
    if outcome_type == "binary" and not set(unique_y.tolist()).issubset({0.0, 1.0}):
        raise ValueError("Binary outcome_type requires the outcome to be coded 0/1.")
    X = _design_matrix(work, adjustment_set)
    groups = work[cluster_col].to_numpy() if cluster_col else None
    ps = _propensity_scores(X, t)
    pos = _positivity(ps, t)
    ipw = np.where(t == 1, 1.0 / ps, 1.0 / (1.0 - ps))

    if estimator.upper() == "G-COMPUTATION":
        mu1, mu0 = _outcome_predictions(X, t, y, outcome_type)
        pseudo = mu1 - mu0
        effect = float(np.mean(pseudo) if estimand.upper() == "ATE" else np.mean(mu1[t == 1] - mu0[t == 1]))
        formula = "ATE = mean(m(1,L) - m(0,L)); ATT = mean(m(1,L)-m(0,L) | A=1)"
        nuisance = {"mu1_mean": float(np.mean(mu1)), "mu0_mean": float(np.mean(mu0))}
    elif estimator.upper() == "IPW":
        if estimand.upper() == "ATE":
            effect = float(np.sum((t * y / ps) - ((1 - t) * y / (1 - ps))) / len(y))
        elif estimand.upper() == "ATT":
            treated_mean = float(np.mean(y[t == 1]))
            weighted_control = np.sum((1 - t) * y * ps / (1 - ps)) / max(np.sum((1 - t) * ps / (1 - ps)), 1e-12)
            effect = treated_mean - float(weighted_control)
        else:
            raise ValueError("estimand must be ATE or ATT")
        formula = "ATE = n^-1 * Σ[AY/e(X) - (1-A)Y/(1-e(X))]; ATT uses ATT-specific reweighting"
        nuisance = {}
    elif estimator.upper() == "AIPW":
        ehat, mu1, mu0 = _cross_fitted_nuisance(X, t, y, outcome_type=outcome_type, groups=groups)
        if estimand.upper() == "ATE":
            pseudo = mu1 - mu0 + t * (y - mu1) / ehat - (1 - t) * (y - mu0) / (1 - ehat)
            effect = float(np.mean(pseudo))
        elif estimand.upper() == "ATT":
            treated = t == 1
            score = t * (y - mu0) - (1 - t) * (ehat / (1 - ehat)) * (y - mu0)
            effect = float(np.sum(score) / max(np.sum(treated), 1))
        else:
            raise ValueError("estimand must be ATE or ATT")
        pseudo = mu1 - mu0 + t * (y - mu1) / ehat - (1 - t) * (y - mu0) / (1 - ehat)
        formula = "AIPW = mean[m1(L)-m0(L) + A/e(L)(Y-m1(L)) - (1-A)/(1-e(L))(Y-m0(L))]"
        nuisance = {"cross_fitted_folds": min(5, int(min(np.sum(t == 0), np.sum(t == 1)))), "mu1_mean": float(np.mean(mu1)), "mu0_mean": float(np.mean(mu0)), "crossfit_mode": "stratified_clusters" if cluster_col else "stratified_rows"}
    else:
        raise ValueError("estimator must be G-COMPUTATION, IPW, or AIPW")

    # Deterministic bootstrap of the full estimation procedure. We preserve the
    # same seed and method so the uncertainty is reproducible and traceable.
    rng = np.random.default_rng(seed)
    boot: List[float] = []
    if bootstrap_resamples > 0:
        if cluster_col:
            unique_clusters = work[cluster_col].drop_duplicates().tolist()
            for _ in range(bootstrap_resamples):
                sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
                parts = [work.loc[work[cluster_col] == cluster_id].copy() for cluster_id in sampled_clusters]
                sample = pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0].copy()
                try:
                    b = estimate_binary_treatment_effect(
                        sample,
                        treatment_col=treatment_col,
                        outcome_col=outcome_col,
                        adjustment_set=adjustment_set,
                        estimand=estimand,
                        estimator=estimator,
                        confidence=confidence,
                        bootstrap_resamples=0,
                        seed=seed,
                        outcome_type=outcome_type,
                        cluster_col=cluster_col,
                        time_col=time_col,
                    ).effect
                    boot.append(float(b))
                except Exception:
                    continue
        else:
            for _ in range(bootstrap_resamples):
                idx = rng.integers(0, len(work), len(work))
                try:
                    b = estimate_binary_treatment_effect(
                        work.iloc[idx].reset_index(drop=True),
                        treatment_col=treatment_col,
                        outcome_col=outcome_col,
                        adjustment_set=adjustment_set,
                        estimand=estimand,
                        estimator=estimator,
                        confidence=confidence,
                        bootstrap_resamples=0,
                        seed=seed,
                        outcome_type=outcome_type,
                    ).effect
                    boot.append(float(b))
                except Exception:
                    continue
    required_valid = int(np.ceil(bootstrap_resamples * 0.8)) if bootstrap_resamples > 0 else 0
    if bootstrap_resamples > 0 and len(boot) < required_valid:
        raise ValueError(f"Causal bootstrap unstable: {len(boot)} of {bootstrap_resamples} resamples valid; required at least {required_valid}.")
    if boot:
        alpha = 1 - confidence
        ci = np.quantile(np.asarray(boot), [alpha / 2, 1 - alpha / 2]).astype(float).tolist()
        se = float(np.std(np.asarray(boot), ddof=1))
    else:
        ci = [float("nan"), float("nan")]
        se = None

    base_balance = _standardized_mean_differences(X, t)
    weighted_balance = _standardized_mean_differences(X, t, weights=ipw)
    max_smd_before = max((abs(v) for v in base_balance.values()), default=0.0)
    max_smd_after = max((abs(v) for v in weighted_balance.values()), default=0.0)
    balance = {
        "standardized_mean_difference_before": base_balance,
        "standardized_mean_difference_after_ipw": weighted_balance,
        "max_abs_smd_before": float(max_smd_before),
        "max_abs_smd_after_ipw": float(max_smd_after),
        "threshold": 0.10,
        "status": "BALANCED" if max_smd_after <= 0.10 else "IMBALANCED",
    }
    limitations = [
        "Causal interpretation is conditional on the supplied identification assumptions and adjustment set.",
        "Positivity is diagnosed but not repaired automatically; extreme propensity scores can make estimates unstable.",
        "Bootstrap uncertainty uses independent observed rows when no cluster is supplied; for repeated observations, supply cluster_col so uncertainty is resampled at the cluster level.",
    ]
    if pos["status"] != "ADEQUATE":
        limitations.append("Positivity/overlap is weak; the effect estimate should be treated as unstable or potentially not practically estimable.")
    if balance["status"] == "IMBALANCED":
        limitations.append("IPW weighted covariate balance remains above the 0.10 standardized-mean-difference diagnostic threshold.")
    if outcome_type == "binary":
        limitations.append("The E-value is a sensitivity summary for unmeasured confounding on the risk-ratio scale; it is not a replacement for a full bias-function or tipping-point analysis.")
    if bootstrap_resamples and bootstrap_resamples < 100:
        limitations.append("Bootstrap resample count is below 100; the reported interval is suitable for development/testing but is lower precision than the default 1,000-resample analysis.")

    treated_risk = float(np.mean(y[t == 1])) if outcome_type == "binary" else None
    control_risk = float(np.mean(y[t == 0])) if outcome_type == "binary" else None
    sensitivity = (
        e_value_report(treated_risk=treated_risk, control_risk=control_risk)
        if outcome_type == "binary" and treated_risk is not None and control_risk is not None
        else {"available": False, "reason": "E-value is currently available for binary-outcome risk ratios."}
    )
    diagnostics = {"nuisance": nuisance, "outcome_mean": float(np.mean(y)), "treatment_rate": float(np.mean(t)), "bootstrap_valid_resamples": len(boot), "outcome_type": outcome_type, "treated_risk": treated_risk, "control_risk": control_risk, "sensitivity": sensitivity, "dependence": dependence}
    return CausalEffectEstimate(
        estimand=estimand.upper(), estimator=estimator.upper(), effect=float(effect),
        confidence_interval=[float(ci[0]), float(ci[1])], standard_error=se,
        n=len(work), treated_n=int(np.sum(t)), control_n=int(np.sum(t == 0)),
        positivity=pos, balance=balance,
        formulas={"estimand": formula, "outcome_model": "logistic regression for binary outcome; linear regression for continuous outcome", "binary_outcome_effect": "Risk difference = P(Y=1|do(A=1)) - P(Y=1|do(A=0))" if outcome_type == "binary" else "N/A"},
        assumptions=[
            "Binary treatment is correctly defined and consistently measured.",
            "Consistency: observed outcome under the observed treatment equals the corresponding potential outcome.",
            "Conditional exchangeability/no unmeasured confounding given the validated adjustment set.",
            "Positivity/overlap holds over the target population.",
            "Outcome and treatment measurements are correctly coded.",
            "For binary outcomes, the outcome is coded 0/1 and nuisance outcome models estimate probabilities.",
        ],
        limitations=limitations,
        diagnostics=diagnostics,
        reproducibility={
            "bootstrap_seed": seed,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_unit": "cluster" if cluster_col else "row",
            "cluster_col": cluster_col,
            "time_col": time_col,
            "feature_columns": list(X.columns),
        },
    )
