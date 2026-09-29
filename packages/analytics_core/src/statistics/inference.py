"""Execution of selected inferential statistical methods with diagnostics and trace payloads."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .method_selection import select_two_group, select_multi_group, select_categorical, select_correlation
from .multiple_comparisons import pairwise_posthoc
from .preflight import _to_clean_float_array, _to_clean_float_series
from .uncertainty import bootstrap_ci, hedges_g, rank_biserial, eta_squared, cramers_v


def _ci_mean_difference(a: np.ndarray, b: np.ndarray, equal_var: bool) -> tuple[float, float]:
    diff = float(np.mean(a) - np.mean(b))
    n1, n2 = len(a), len(b)
    if equal_var:
        s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
        sp2 = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
        se = np.sqrt(sp2 * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        v1, v2 = np.var(a, ddof=1)/n1, np.var(b, ddof=1)/n2
        se = np.sqrt(v1 + v2)
        df = (v1 + v2)**2 / ((v1**2)/(n1-1) + (v2**2)/(n2-1)) if (n1 > 1 and n2 > 1 and v1 + v2 > 0) else 1
    margin = float(stats.t.ppf(0.975, df) * se) if se > 0 else 0.0
    return diff - margin, diff + margin


def execute_two_group(a: Sequence[float], b: Sequence[float], *, paired: bool = False, alpha: float = 0.05) -> Dict[str, Any]:
    decision = select_two_group(a, b, paired=paired, alpha=alpha)
    if paired:
        sa, sb = _to_clean_float_series(a), _to_clean_float_series(b)
        mask = np.isfinite(sa) & np.isfinite(sb)
        aa, bb = sa[mask].to_numpy(dtype=float), sb[mask].to_numpy(dtype=float)
        if decision.method == "Paired t-test":
            result = stats.ttest_rel(aa, bb)
            d = aa - bb
            ci = stats.t.interval(1-alpha, len(d)-1, loc=float(np.mean(d)), scale=stats.sem(d)) if len(d) > 1 else (np.nan, np.nan)
            effect = float(np.mean(d) / np.std(d, ddof=1)) if len(d) > 1 and np.std(d, ddof=1) > 0 else 0.0
            effect_ci = bootstrap_ci(lambda x: float(np.mean(x) / np.std(x, ddof=1)) if np.std(x, ddof=1) > 0 else 0.0, d)
        elif decision.method == "Wilcoxon signed-rank":
            result = stats.wilcoxon(aa, bb, alternative="two-sided", method="auto")
            effect = rank_biserial(aa, bb)
            effect_ci = bootstrap_ci(rank_biserial, aa, bb)
            ci = (np.nan, np.nan)
        else:
            return {"decision": decision.to_dict(), "error": "No defensible paired method available."}
        return {"decision": decision.to_dict(), "method": decision.method, "statistic": float(result.statistic), "p_value": float(result.pvalue), "effect_size": effect, "confidence_interval": [float(ci[0]), float(ci[1])], "effect_size_confidence_interval": [effect_ci["lower"], effect_ci["upper"]], "uncertainty": effect_ci, "alpha": alpha}

    aa, bb = _to_clean_float_array(a), _to_clean_float_array(b)
    if decision.method == "Welch t-test":
        result = stats.ttest_ind(aa, bb, equal_var=False)
        ci = _ci_mean_difference(aa, bb, equal_var=False)
        effect = float(decision.diagnostics["hedges_g"])
        effect_ci = bootstrap_ci(hedges_g, aa, bb)
    elif decision.method == "Student t-test":
        result = stats.ttest_ind(aa, bb, equal_var=True)
        ci = _ci_mean_difference(aa, bb, equal_var=True)
        effect = float(decision.diagnostics["hedges_g"])
        effect_ci = bootstrap_ci(hedges_g, aa, bb)
    elif decision.method == "Mann-Whitney U":
        result = stats.mannwhitneyu(aa, bb, alternative="two-sided", method="auto")
        effect = rank_biserial(aa, bb)
        effect_ci = bootstrap_ci(rank_biserial, aa, bb)
        ci = (np.nan, np.nan)
    else:
        return {"decision": decision.to_dict(), "error": "No defensible inferential method available."}
    return {"decision": decision.to_dict(), "method": decision.method, "statistic": float(result.statistic), "p_value": float(result.pvalue), "effect_size": effect, "confidence_interval_mean_difference": [float(ci[0]), float(ci[1])], "effect_size_confidence_interval": [effect_ci["lower"], effect_ci["upper"]], "uncertainty": effect_ci, "alpha": alpha}


def _welch_anova(groups: Dict[str, Sequence[float]], alpha: float = 0.05) -> Dict[str, float]:
    clean = {k: _to_clean_float_array(v) for k, v in groups.items()}
    clean = {k: v for k, v in clean.items() if len(v) >= 2}
    k = len(clean)
    n = np.array([len(v) for v in clean.values()], dtype=float)
    means = np.array([np.mean(v) for v in clean.values()], dtype=float)
    vars_ = np.array([np.var(v, ddof=1) for v in clean.values()], dtype=float)
    w = n / vars_
    w_sum = np.sum(w)
    mean_w = np.sum(w * means) / w_sum
    numerator = np.sum(w * (means - mean_w) ** 2) / (k - 1)
    correction = (2 * (k - 2) / (k**2 - 1)) * np.sum(((1 - w / w_sum) ** 2) / (n - 1))
    f = numerator / (1 + correction)
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * np.sum(((1 - w / w_sum) ** 2) / (n - 1)))
    p = 1 - stats.f.cdf(f, df1, df2)
    return {"f_statistic": float(f), "p_value": float(p), "df1": float(df1), "df2": float(df2), "alpha": alpha}


def execute_multi_group(groups: Dict[str, Sequence[float]], *, alpha: float = 0.05) -> Dict[str, Any]:
    decision = select_multi_group(groups, alpha=alpha)
    clean = {k: _to_clean_float_array(v) for k, v in groups.items()}
    clean = {k: v for k, v in clean.items() if len(v) >= 2}
    if decision.method == "One-way ANOVA":
        res = stats.f_oneway(*clean.values())
        eta_ci = bootstrap_ci(lambda *xs: eta_squared(dict(zip(clean.keys(), xs))), *clean.values())
        posthoc = pairwise_posthoc(clean, omnibus_method=decision.method, omnibus_p_value=float(res.pvalue), alpha=alpha)
        return {"decision": decision.to_dict(), "method": decision.method, "statistic": float(res.statistic), "p_value": float(res.pvalue), "eta_squared": eta_squared(clean), "eta_squared_confidence_interval": [eta_ci["lower"], eta_ci["upper"]], "uncertainty": eta_ci, "alpha": alpha, "posthoc": posthoc}
    if decision.method == "Welch ANOVA":
        omnibus = _welch_anova(clean, alpha=alpha)
        posthoc = pairwise_posthoc(clean, omnibus_method=decision.method, omnibus_p_value=float(omnibus["p_value"]), alpha=alpha)
        return {"decision": decision.to_dict(), "method": decision.method, **omnibus, "posthoc": posthoc}
    if decision.method == "Kruskal-Wallis":
        res = stats.kruskal(*clean.values())
        k, n = len(clean), sum(len(v) for v in clean.values())
        epsilon2 = max(0.0, float((res.statistic - k + 1.0) / max(n - k, 1)))
        posthoc = pairwise_posthoc(clean, omnibus_method=decision.method, omnibus_p_value=float(res.pvalue), alpha=alpha)
        return {"decision": decision.to_dict(), "method": decision.method, "statistic": float(res.statistic), "p_value": float(res.pvalue), "epsilon_squared": epsilon2, "alpha": alpha, "posthoc": posthoc}
    return {"decision": decision.to_dict(), "error": "No defensible omnibus method available."}


def execute_categorical(df: pd.DataFrame, col_a: str, col_b: str, *, alpha: float = 0.05) -> Dict[str, Any]:
    contingency = pd.crosstab(df[col_a], df[col_b])
    decision = select_categorical(contingency, alpha=alpha)
    if decision.method == "Fisher exact test":
        odds, p = stats.fisher_exact(contingency.to_numpy())
        from statsmodels.stats.contingency_tables import Table2x2
        table2 = Table2x2(contingency.to_numpy(), shift_zeros=True)
        or_ci = table2.oddsratio_confint(alpha=alpha, method="normal")
        unc = {"estimate": float(odds), "lower": float(or_ci[0]), "upper": float(or_ci[1]), "confidence": 1-alpha, "method": "log-odds normal approximation", "zero_cell_handling": "Haldane-Anscombe shift via statsmodels when needed"}
        return {"decision": decision.to_dict(), "method": decision.method, "odds_ratio": float(odds), "odds_ratio_confidence_interval": [float(or_ci[0]), float(or_ci[1])], "uncertainty": unc, "p_value": float(p), "alpha": alpha}
    if decision.method == "Chi-square test of independence":
        chi2, p, dof, expected = stats.chi2_contingency(contingency, correction=False)
        n = float(contingency.to_numpy().sum())
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        v = float(np.sqrt(chi2 / (n * min_dim))) if n and min_dim else 0.0
        # Bootstrap rows of the observed contingency table; preserve the table shape.
        row_labels = df[col_a].dropna().unique(); col_labels = df[col_b].dropna().unique()
        codes_a = pd.Categorical(df[col_a], categories=row_labels).codes
        codes_b = pd.Categorical(df[col_b], categories=col_labels).codes
        valid = (codes_a >= 0) & (codes_b >= 0)
        pairs = np.column_stack([codes_a[valid], codes_b[valid]])
        rng = np.random.default_rng(20260908)
        vals = []
        for _ in range(2000):
            idx = rng.integers(0, len(pairs), len(pairs))
            boot = np.zeros((len(row_labels), len(col_labels)), dtype=float)
            for ra, cb in pairs[idx]: boot[ra, cb] += 1
            vals.append(cramers_v(boot))
        ci = np.quantile(np.asarray(vals), [alpha/2, 1-alpha/2])
        cv_unc = {"estimate": v, "lower": float(ci[0]), "upper": float(ci[1]), "confidence": 1-alpha, "resamples": 2000, "seed": 20260908, "method": "percentile_bootstrap"}
        return {"decision": decision.to_dict(), "method": decision.method, "chi2_statistic": float(chi2), "p_value": float(p), "degrees_of_freedom": int(dof), "cramers_v": v, "cramers_v_confidence_interval": [float(ci[0]), float(ci[1])], "uncertainty": cv_unc, "expected_counts": expected.tolist(), "alpha": alpha}
    return {"decision": decision.to_dict(), "error": "No defensible categorical method available."}


def execute_correlation(x: Sequence[float], y: Sequence[float], *, alpha: float = 0.05) -> Dict[str, Any]:
    decision = select_correlation(x, y, alpha=alpha)
    sx, sy = _to_clean_float_series(x), _to_clean_float_series(y)
    mask = np.isfinite(sx) & np.isfinite(sy)
    xx, yy = sx[mask].to_numpy(dtype=float), sy[mask].to_numpy(dtype=float)
    if decision.method == "Pearson correlation":
        res = stats.pearsonr(xx, yy)
        ci = res.confidence_interval(1-alpha)
        return {"decision": decision.to_dict(), "method": decision.method, "coefficient": float(res.statistic), "p_value": float(res.pvalue), "confidence_interval": [float(ci.low), float(ci.high)], "alpha": alpha}
    if decision.method == "Spearman correlation":
        res = stats.spearmanr(xx, yy)
        rank = lambda x, y: float(stats.spearmanr(x, y).statistic)
        ci = bootstrap_ci(rank, xx, yy)
        return {"decision": decision.to_dict(), "method": decision.method, "coefficient": float(res.statistic), "p_value": float(res.pvalue), "confidence_interval": [ci["lower"], ci["upper"]], "uncertainty": ci, "alpha": alpha}
    return {"decision": decision.to_dict(), "error": "No defensible association method available."}


def _vif_diagnostics(X: pd.DataFrame) -> Dict[str, Any]:
    """Compute variance inflation factors for numeric design columns."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if X.shape[1] <= 1:
        return {"vif": {}, "high_vif_features": [], "moderate_vif_features": [], "note": "No non-intercept predictors to assess."}
    values = X.to_numpy(dtype=float)
    out: Dict[str, float] = {}
    names = list(X.columns)
    for i, name in enumerate(names):
        try:
            v = float(variance_inflation_factor(values, i))
        except Exception:
            v = float("inf")
        out[name] = v
    high = [name for name, v in out.items() if v >= 10.0]
    moderate = [name for name, v in out.items() if 5.0 <= v < 10.0]
    return {"vif": out, "high_vif_features": high, "moderate_vif_features": moderate}


def _regression_diagnostics(model, X: pd.DataFrame, y: pd.Series, *, alpha: float) -> Dict[str, Any]:
    from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset, acorr_breusch_godfrey
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from statsmodels.stats.outliers_influence import OLSInfluence

    resid = np.asarray(model.resid, dtype=float)
    fitted = np.asarray(model.fittedvalues, dtype=float)
    bp = het_breuschpagan(resid, model.model.exog)
    reset_p = None
    try:
        reset = linear_reset(model, power=2, use_f=True)
        reset_p = float(reset.pvalue)
    except Exception:
        pass
    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
    infl = OLSInfluence(model)
    cooks = np.asarray(infl.cooks_distance[0], dtype=float)
    leverage = np.asarray(infl.hat_matrix_diag, dtype=float)
    threshold = 4.0 / max(len(y) - model.df_model - 1.0, 1.0)
    influential = np.where(cooks > threshold)[0].tolist()
    dw = float(durbin_watson(resid))
    bg_p = None
    if len(y) >= 20:
        try:
            bg_lag = min(4, max(1, len(y) // 10))
            bg_p = float(acorr_breusch_godfrey(model, nlags=bg_lag)[1])
        except Exception:
            pass
    return {
        "heteroskedasticity": {
            "breusch_pagan_lm_p": float(bp[1]),
            "breusch_pagan_f_p": float(bp[3]),
            "detected": bool(bp[1] < alpha),
        },
        # Backward-compatible scalar aliases retained for existing consumers.
        "breusch_pagan_p": float(bp[1]),
        "breusch_pagan_f_p": float(bp[3]),
        "functional_form": {
            "ramsey_reset_p": reset_p,
            "potential_misspecification": bool(reset_p is not None and reset_p < alpha),
        },
        "residual_distribution": {
            "jarque_bera_p": float(jb_p),
            "skewness": float(skew),
            "excess_kurtosis": float(kurt - 3.0),
            "non_normal_residuals": bool(jb_p < alpha),
        },
        "dependence_screen": {
            "durbin_watson": dw,
            "breusch_godfrey_p": bg_p,
            "serial_correlation_detected": bool(bg_p is not None and bg_p < alpha),
        },
        "influence": {
            "max_cooks_distance": float(np.max(cooks)) if len(cooks) else 0.0,
            "influential_observation_count": len(influential),
            "influential_indices": influential[:100],
            "cook_threshold": float(threshold),
            "high_leverage_count": int(np.sum(leverage > (2.0 * (model.df_model + 1.0) / max(len(y), 1)))),
        },
        "residual_sample_size": int(len(resid)),
        "residual_mean": float(np.mean(resid)),
        "residual_std": float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0,
        "fitted_range": [float(np.min(fitted)), float(np.max(fitted))] if len(fitted) else [None, None],
    }


def execute_regression(
    df: pd.DataFrame,
    x_columns: Sequence[str],
    y_column: str,
    *,
    alpha: float = 0.05,
    cluster_column: str | None = None,
    time_column: str | None = None,
) -> Dict[str, Any]:
    """Execute regression inference with assumption diagnostics and defensible covariance selection.

    The coefficients remain OLS estimates. Inference covariance is selected as follows:
    clustered covariance when an explicit grouping variable is supplied; otherwise HC3
    when heteroskedasticity is detected; otherwise classical covariance. For temporal data,
    a HAC/Newey-West covariance is used when a time column is supplied and serial dependence
    is detected. Diagnostics are never treated as proof of causal identification.
    """
    cols = list(dict.fromkeys(list(x_columns) + [y_column] + ([cluster_column] if cluster_column else []) + ([time_column] if time_column else [])))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return {"error": f"Regression columns not found: {missing}"}
    data = df[cols].copy()
    for c in list(x_columns) + [y_column]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    if cluster_column:
        data[cluster_column] = data[cluster_column].astype("object")
    if time_column:
        parsed = pd.to_datetime(data[time_column], errors="coerce")
        data[time_column] = parsed
    required = list(x_columns) + [y_column] + ([cluster_column] if cluster_column else []) + ([time_column] if time_column else [])
    data = data.dropna(subset=required)
    if len(data) <= len(x_columns) + 2:
        return {"error": "Insufficient degrees of freedom for regression inference."}

    X = sm.add_constant(data[list(x_columns)], has_constant="add")
    y = data[y_column]
    ols = sm.OLS(y, X).fit()
    diagnostics = _regression_diagnostics(ols, X, y, alpha=alpha)
    vif = _vif_diagnostics(data[list(x_columns)])
    diagnostics["multicollinearity"] = vif
    diagnostics["condition_number"] = float(ols.condition_number)
    diagnostics["rank"] = int(np.linalg.matrix_rank(X.to_numpy(dtype=float)))
    diagnostics["full_column_rank"] = bool(diagnostics["rank"] == X.shape[1])

    covariance_method = "classical"
    fitted = ols
    covariance_reason = "No explicit cluster/time structure supplied and no heteroskedasticity detected."
    if cluster_column:
        groups = data[cluster_column]
        n_groups = int(groups.nunique(dropna=True))
        if n_groups >= 2:
            fitted = ols.get_robustcov_results(cov_type="cluster", groups=groups)
            covariance_method = "clustered"
            covariance_reason = f"Explicit cluster column supplied with {n_groups} groups."
        else:
            covariance_reason = "Cluster column supplied but fewer than 2 usable groups; classical covariance retained."
    elif time_column and diagnostics["dependence_screen"]["serial_correlation_detected"]:
        # HAC bandwidth chosen from the observed sample size; the choice is persisted in the result.
        nlags = max(1, int(np.floor(4.0 * (len(data) / 100.0) ** (2.0 / 9.0))))
        fitted = ols.get_robustcov_results(cov_type="HAC", maxlags=nlags)
        covariance_method = "HAC/Newey-West"
        covariance_reason = f"Serial dependence detected; HAC covariance with maxlags={nlags}."
    elif diagnostics["heteroskedasticity"]["detected"]:
        fitted = ols.get_robustcov_results(cov_type="HC3")
        covariance_method = "HC3"
        covariance_reason = "Breusch-Pagan test detected heteroskedasticity; HC3 used for coefficient inference."

    names = ["const"] + list(x_columns)
    params = dict(zip(names, fitted.params))
    ses = dict(zip(names, fitted.bse))
    pvals = dict(zip(names, fitted.pvalues))
    ci_arr = fitted.conf_int(alpha=alpha)
    if hasattr(ci_arr, "iloc"):
        cis = {name: [float(ci_arr.iloc[i, 0]), float(ci_arr.iloc[i, 1])] for i, name in enumerate(names)}
    else:
        cis = {name: [float(ci_arr[i][0]), float(ci_arr[i][1])] for i, name in enumerate(names)}

    warnings = []
    if not diagnostics["full_column_rank"]:
        warnings.append("Design matrix is rank deficient; coefficient estimates are not uniquely identified for all predictors.")
    if vif["high_vif_features"]:
        warnings.append(f"Severe multicollinearity detected in: {vif['high_vif_features']}.")
    if vif["moderate_vif_features"]:
        warnings.append(f"Moderate multicollinearity detected in: {vif['moderate_vif_features']}.")
    if diagnostics["functional_form"]["potential_misspecification"]:
        warnings.append("Ramsey RESET indicates possible functional-form misspecification.")
    if diagnostics["dependence_screen"]["serial_correlation_detected"] and not (time_column or cluster_column):
        warnings.append("Serial dependence detected, but no clustering/time structure was supplied; inferential covariance may be inadequate.")
    if diagnostics["influence"]["influential_observation_count"]:
        warnings.append("Influential observations detected; coefficient stability should be assessed with sensitivity analysis.")
    if diagnostics["residual_distribution"]["non_normal_residuals"]:
        warnings.append("Residual normality is rejected; with adequate sample size coefficient inference may rely on asymptotic/robust methods, but small-sample normal-theory intervals should be treated cautiously.")

    return {
        "problem": "regression_inference",
        "method": {"classical": "OLS classical SE", "HC3": "OLS with HC3 robust SE", "clustered": "OLS with clustered SE", "HAC/Newey-West": "OLS with HAC/Newey-West SE"}[covariance_method],
        "estimand": "Conditional mean difference in the outcome associated with each predictor, holding included predictors constant.",
        "sample_size": int(len(data)),
        "formula": "y = beta_0 + beta_1*x_1 + ... + beta_p*x_p + error",
        "target_variable": y_column,
        "feature_variables": list(x_columns),
        "covariance": {"method": covariance_method, "reason": covariance_reason, "alpha": alpha},
        "r_squared": float(ols.rsquared),
        "adjusted_r_squared": float(ols.rsquared_adj),
        "aic": float(ols.aic),
        "bic": float(ols.bic),
        "coefficients": {
            name: {
                "coefficient": float(params[name]),
                "std_error": float(ses[name]),
                "p_value": float(pvals[name]),
                "confidence_interval": cis[name],
            }
            for name in names
        },
        "diagnostics": diagnostics,
        "warnings": warnings,
        "assumption_status": {
            "linearity": "SUPPORTED" if not diagnostics["functional_form"]["potential_misspecification"] else "QUESTIONABLE",
            "multicollinearity": "OK" if not vif["high_vif_features"] else "HIGH_RISK",
            "heteroskedasticity": "DETECTED" if diagnostics["heteroskedasticity"]["detected"] else "NOT_DETECTED",
            "serial_dependence": "DETECTED" if diagnostics["dependence_screen"]["serial_correlation_detected"] else "NOT_DETECTED_OR_NOT_TESTED",
            "influential_points": "DETECTED" if diagnostics["influence"]["influential_observation_count"] else "NOT_DETECTED",
            "residual_normality": "QUESTIONABLE" if diagnostics["residual_distribution"]["non_normal_residuals"] else "NOT_REJECTED",
        },
        "limitations": [
            "OLS coefficients are associational unless a valid causal design and identification strategy are supplied.",
            "Robust or clustered covariance changes standard errors/inference, not the underlying coefficient estimate.",
            "Diagnostics are evidence of potential violations, not proofs that assumptions hold or fail.",
            "Omitted variables, measurement error, selection bias, and reverse causality are not resolved by OLS diagnostics.",
        ],
    }

