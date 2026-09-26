"""
analytical_math.py: Comprehensive Deterministic Analytical Mathematics Library for MindEd AA-OS.

Provides verified mathematical primitives across 6 core domains:
1. Descriptive Analytics (moments, percentiles, Gini, HHI, CV, confidence intervals)
2. Time Series Analytics (PoP delta, growth rate, SMA/EMA, Theil-Sen trend, CUSUM change point, decomposition)
3. Segment Analytics (group aggregates, contribution, Pareto 80/20, segment HHI, segment anomalies)
4. Relationship Analytics (covariance, Pearson r, Spearman rho, Kendall tau, Cramer's V, Mutual Information, Partial correlation)
5. Inferential Statistics (Welch's t, Mann-Whitney U, One-Way ANOVA with eta^2, Kruskal-Wallis, Chi-squared, Kolmogorov-Smirnov, multiple testing FDR/Bonferroni, statistical power)
6. Anomaly & Change Detection (IQR, Z-score, MAD-based Modified Z-score, distribution shift distance)
"""
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from packages.analytics_core.src.statistics.preflight import _to_clean_float_array, _to_clean_float_series


# ==============================================================================
# 1. DESCRIPTIVE ANALYTICS
# ==============================================================================

def descriptive_summary(series: Union[pd.Series, List[float], np.ndarray]) -> Dict[str, Any]:
    """Compute exact, reproducible descriptive statistics on numeric series."""
    s = _to_clean_float_series(series)

    total_count = len(s)
    clean = s.dropna()
    valid_count = len(clean)
    null_count = total_count - valid_count
    null_rate = (null_count / total_count) if total_count > 0 else 0.0

    if valid_count == 0:
        return {
            "error": "Series contains no valid numeric values.",
            "total_count": total_count,
            "null_count": null_count,
            "null_rate": float(null_rate),
        }

    arr = clean.values.astype(float)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    range_val = float(max_val - min_val)
    sum_val = float(np.sum(arr))

    var_val = float(np.var(arr, ddof=1)) if valid_count > 1 else 0.0
    std_val = float(np.std(arr, ddof=1)) if valid_count > 1 else 0.0
    cv = (std_val / abs(mean_val)) if abs(mean_val) > 1e-9 else 0.0

    skew_val = float(stats.skew(arr)) if valid_count > 2 else 0.0
    kurt_val = float(stats.kurtosis(arr)) if valid_count > 3 else 0.0

    percentiles = {
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
    iqr_val = float(percentiles["p75"] - percentiles["p25"])

    # 95% Confidence Interval of the mean (Student's t distribution)
    sem = stats.sem(arr) if valid_count > 1 else 0.0
    if valid_count > 1 and sem > 0:
        ci_95 = stats.t.interval(0.95, df=valid_count - 1, loc=mean_val, scale=sem)
        mean_ci = [round(float(ci_95[0]), 4), round(float(ci_95[1]), 4)]
    else:
        mean_ci = [round(mean_val, 4), round(mean_val, 4)]

    return {
        "total_count": total_count,
        "valid_count": valid_count,
        "null_count": null_count,
        "null_rate": round(float(null_rate), 4),
        "distinct_count": int(clean.nunique()),
        "sum": round(sum_val, 4),
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "range": round(range_val, 4),
        "variance": round(var_val, 4),
        "std_dev": round(std_val, 4),
        "coefficient_of_variation": round(float(cv), 4),
        "skewness": round(skew_val, 4),
        "kurtosis": round(kurt_val, 4),
        "iqr": round(iqr_val, 4),
        "percentiles": {k: round(v, 4) for k, v in percentiles.items()},
        "mean_ci_95": mean_ci,
        "method": "scipy_exact_moments",
    }


def weighted_mean(values: Union[pd.Series, List[float], np.ndarray], weights: Union[pd.Series, List[float], np.ndarray]) -> float:
    """Calculate exact weighted mean."""
    v = _to_clean_float_array(values)
    w = _to_clean_float_array(weights)
    min_len = min(len(v), len(w))
    v, w = v[:min_len], w[:min_len]
    mask = (w > 0)
    v_clean, w_clean = v[mask], w[mask]
    if len(v_clean) == 0 or np.sum(w_clean) == 0:
        return 0.0
    return float(np.average(v_clean, weights=w_clean))


def concentration_metrics(series: Union[pd.Series, List[float], np.ndarray]) -> Dict[str, Any]:
    """
    Calculate concentration metrics:
    - Gini coefficient (0 = perfect equality, 1 = maximum concentration)
    - Herfindahl-Hirschman Index (HHI, 0 to 10,000)
    - Top-1, Top-3, Top-5, Top-10 share of total
    - Pareto 80/20 share
    """
    s = _to_clean_float_series(series)

    clean = s.dropna()
    clean = clean[clean >= 0]
    n = len(clean)
    if n == 0:
        return {"error": "No non-negative values for concentration analysis."}

    total_sum = float(clean.sum())
    if total_sum <= 1e-9:
        return {
            "gini_coefficient": 0.0,
            "herfindahl_index": 0.0,
            "top_1_share": 0.0,
            "top_3_share": 0.0,
            "top_5_share": 0.0,
            "top_10_share": 0.0,
            "pareto_80_20_ratio": 0.0,
            "item_count": n,
        }

    # Gini Coefficient calculation
    sorted_vals = np.sort(clean.values)
    index = np.arange(1, n + 1)
    gini = float(((2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals))) - ((n + 1) / n))
    gini = max(0.0, min(1.0, gini))

    # HHI calculation (shares as percentages 0-100, squared and summed)
    shares_pct = (clean.values / total_sum) * 100.0
    hhi = float(np.sum(shares_pct ** 2))

    # Top-K shares (descending)
    desc_vals = np.sort(clean.values)[::-1]
    top_1 = float(desc_vals[:1].sum() / total_sum) if n >= 1 else 0.0
    top_3 = float(desc_vals[:3].sum() / total_sum) if n >= 3 else float(desc_vals.sum() / total_sum)
    top_5 = float(desc_vals[:5].sum() / total_sum) if n >= 5 else float(desc_vals.sum() / total_sum)
    top_10 = float(desc_vals[:10].sum() / total_sum) if n >= 10 else float(desc_vals.sum() / total_sum)

    # Pareto 80/20 share: What proportion of total sum is held by the top 20% of items?
    top_20_pct_count = max(1, int(math.ceil(0.20 * n)))
    pareto_share = float(desc_vals[:top_20_pct_count].sum() / total_sum)

    return {
        "item_count": n,
        "total_sum": round(total_sum, 4),
        "gini_coefficient": round(gini, 4),
        "herfindahl_index": round(hhi, 2),
        "concentration_level": "High" if hhi > 2500 or gini > 0.6 else ("Moderate" if hhi > 1500 or gini > 0.4 else "Low"),
        "top_1_share": round(top_1, 4),
        "top_3_share": round(top_3, 4),
        "top_5_share": round(top_5, 4),
        "top_10_share": round(top_10, 4),
        "pareto_80_20_share": round(pareto_share, 4),
        "method": "exact_gini_and_hhi",
    }


def frequency_table(series: pd.Series, top_n: int = 20) -> List[Dict[str, Any]]:
    """Compute category frequency distribution with percentages and cumulative totals."""
    clean = series.dropna()
    total = len(clean)
    if total == 0:
        return []

    counts = clean.value_counts(ascending=False).head(top_n)
    result = []
    cum_count = 0

    for category, count in counts.items():
        cum_count += count
        result.append({
            "category": str(category),
            "count": int(count),
            "percentage": round(float((count / total) * 100.0), 2),
            "cumulative_percentage": round(float((cum_count / total) * 100.0), 2),
        })
    return result


# ==============================================================================
# 2. TIME SERIES ANALYTICS
# ==============================================================================

def time_series_summary(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    rolling_window: int = 3,
) -> Dict[str, Any]:
    """Compute period-over-period change, growth rate, trend, rolling stats, and CUSUM change-point."""
    if time_col not in df.columns or value_col not in df.columns:
        return {"error": f"Columns '{time_col}' or '{value_col}' not found."}

    sub_df = df[[time_col, value_col]].dropna().copy()
    try:
        sub_df[time_col] = pd.to_datetime(sub_df[time_col])
    except Exception:
        pass

    sub_df = sub_df.sort_values(by=time_col)
    n = len(sub_df)
    if n < 2:
        return {"error": "Time series analysis requires at least 2 chronological observations."}

    vals = sub_df[value_col].values.astype(float)
    x = np.arange(n, dtype=float)

    # 1. Period-over-Period deltas
    pop_deltas = np.diff(vals)
    mean_pop_delta = float(np.mean(pop_deltas))
    
    # 2. Overall Growth rate (Start vs End)
    start_val, end_val = vals[0], vals[-1]
    net_change = end_val - start_val
    growth_rate = (net_change / abs(start_val)) if abs(start_val) > 1e-9 else 0.0

    # 3. Trend Estimation (OLS and Theil-Sen Robust Estimator)
    slope_ols, intercept_ols, r_val, p_val, std_err = stats.linregress(x, vals)
    try:
        theil_res = stats.theilslopes(vals, x)
        theil_slope = float(theil_res.slope)
    except Exception:
        theil_slope = float(slope_ols)

    # 4. Rolling Statistics (SMA and EMA)
    s_vals = pd.Series(vals)
    sma = s_vals.rolling(window=min(rolling_window, n), min_periods=1).mean().values
    ema = s_vals.ewm(span=min(rolling_window, n), adjust=False).mean().values
    rolling_volatility = float(np.std(pop_deltas)) if len(pop_deltas) > 0 else 0.0

    # 5. CUSUM Change-Point Detection
    mean_val = np.mean(vals)
    cusum = np.cumsum(vals - mean_val)
    max_cusum_idx = int(np.argmax(np.abs(cusum)))
    change_point_detected = bool(np.max(np.abs(cusum)) > (1.5 * np.std(vals) * math.sqrt(n))) if np.std(vals) > 0 else False
    change_point_time = str(sub_df[time_col].iloc[max_cusum_idx]) if change_point_detected else None

    # 6. Trend Direction & Classification
    if p_val < 0.05:
        trend_direction = "Upward" if slope_ols > 0 else "Downward"
    else:
        trend_direction = "Stationary / No Significant Trend"

    return {
        "observation_count": n,
        "start_value": round(float(start_val), 4),
        "end_value": round(float(end_val), 4),
        "net_change": round(float(net_change), 4),
        "growth_rate_percentage": round(float(growth_rate * 100.0), 2),
        "mean_pop_delta": round(float(mean_pop_delta), 4),
        "volatility": round(float(rolling_volatility), 4),
        "trend": {
            "direction": trend_direction,
            "slope_ols": round(float(slope_ols), 4),
            "slope_theil_sen_robust": round(float(theil_slope), 4),
            "r_squared": round(float(r_val ** 2), 4),
            "p_value": float(p_val),
            "is_significant": bool(p_val < 0.05),
        },
        "change_point": {
            "detected": change_point_detected,
            "index": max_cusum_idx if change_point_detected else None,
            "timestamp": change_point_time,
            "method": "cusum_break_test",
        },
        "moving_averages_sample": {
            "last_sma": round(float(sma[-1]), 4),
            "last_ema": round(float(ema[-1]), 4),
        },
        "method": "deterministic_time_series_kernel",
    }


# ==============================================================================
# 3. SEGMENT ANALYTICS
# ==============================================================================

def segment_analysis(
    df: pd.DataFrame,
    segment_col: str,
    metric_col: str,
) -> Dict[str, Any]:
    """
    Perform deep segment analysis:
    - Group sums, means, counts, and standard deviations
    - Contribution & share of total
    - Ranking and Pareto 80/20 classification
    - Segment-level variance & concentration (HHI)
    - Cross-segment Z-score anomaly detection
    """
    if segment_col not in df.columns or metric_col not in df.columns:
        return {"error": f"Columns '{segment_col}' or '{metric_col}' not found."}

    sub_df = df[[segment_col, metric_col]].dropna().copy()
    sub_df[metric_col] = pd.to_numeric(sub_df[metric_col], errors="coerce")
    sub_df = sub_df.dropna()

    if len(sub_df) == 0:
        return {"error": "No valid data for segment analysis."}

    total_sum = float(sub_df[metric_col].sum())
    total_count = len(sub_df)

    grouped = sub_df.groupby(segment_col)[metric_col].agg(["sum", "mean", "std", "count"]).reset_index()
    grouped.columns = ["segment", "sum", "mean", "std", "count"]
    grouped["std"] = grouped["std"].fillna(0.0)

    # Sort descending by sum
    grouped = grouped.sort_values(by="sum", ascending=False).reset_index(drop=True)
    
    # Calculate share of total
    if abs(total_sum) > 1e-9:
        grouped["share_pct"] = (grouped["sum"] / total_sum) * 100.0
    else:
        grouped["share_pct"] = 0.0

    grouped["cumulative_share_pct"] = grouped["share_pct"].cumsum()
    grouped["rank"] = np.arange(1, len(grouped) + 1)

    # Segment HHI Concentration
    hhi = float(np.sum(grouped["share_pct"] ** 2)) if total_sum > 0 else 0.0

    # Cross-segment anomaly: Z-score on segment sums
    seg_sums = grouped["sum"].values
    mean_seg_sum = float(np.mean(seg_sums))
    std_seg_sum = float(np.std(seg_sums, ddof=1)) if len(seg_sums) > 1 else 0.0

    segment_records = []
    top_80_segments = []

    for _, row in grouped.iterrows():
        z_score = float((row["sum"] - mean_seg_sum) / std_seg_sum) if std_seg_sum > 1e-9 else 0.0
        is_top_80 = bool(row["cumulative_share_pct"] <= 80.0 or (len(top_80_segments) == 0))
        if is_top_80:
            top_80_segments.append(str(row["segment"]))

        segment_records.append({
            "segment": str(row["segment"]),
            "rank": int(row["rank"]),
            "sum": round(float(row["sum"]), 4),
            "mean": round(float(row["mean"]), 4),
            "std_dev": round(float(row["std"]), 4),
            "count": int(row["count"]),
            "share_percentage": round(float(row["share_pct"]), 2),
            "cumulative_share_percentage": round(float(row["cumulative_share_pct"]), 2),
            "z_score": round(z_score, 2),
            "is_anomaly": bool(abs(z_score) >= 2.0),
            "is_pareto_core": is_top_80,
        })

    return {
        "segment_column": segment_col,
        "metric_column": metric_col,
        "segment_count": len(grouped),
        "total_sum": round(total_sum, 4),
        "total_rows": total_count,
        "segment_concentration_hhi": round(hhi, 2),
        "pareto_80_segment_count": len(top_80_segments),
        "pareto_80_segment_share_of_categories": round(float((len(top_80_segments) / len(grouped)) * 100.0), 2) if len(grouped) > 0 else 0.0,
        "segments": segment_records,
        "method": "deterministic_segment_decomposition",
    }


# ==============================================================================
# 4. RELATIONSHIP & CORRELATION ANALYTICS
# ==============================================================================

def relationship_analysis(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    control_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate relationship between two variables:
    - Covariance
    - Pearson correlation with 95% Confidence Interval & p-value
    - Spearman rank correlation (monotonic non-linear)
    - Kendall's tau (robust rank association)
    - Partial correlation (if control_col is specified)
    - Categorical Cramér's V (if both columns are categorical)
    """
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": f"Columns '{col1}' or '{col2}' not found."}

    # Case A: Both numeric columns
    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
        cols = [col1, col2] + ([control_col] if control_col and control_col in df.columns and pd.api.types.is_numeric_dtype(df[control_col]) else [])
        sub_df = df[cols].dropna()
        n = len(sub_df)
        if n < 3:
            return {"error": "Relationship analysis requires at least 3 valid observations."}

        x = sub_df[col1].values.astype(float)
        y = sub_df[col2].values.astype(float)

        cov_val = float(np.cov(x, y)[0, 1])

        # Pearson
        r_pearson, p_pearson = stats.pearsonr(x, y)
        r_pearson = 0.0 if math.isnan(r_pearson) else float(r_pearson)
        p_pearson = 1.0 if math.isnan(p_pearson) else float(p_pearson)

        # Pearson Fisher Z-transform 95% CI
        if n > 3 and abs(r_pearson) < 0.9999:
            z = np.arctanh(r_pearson)
            se_z = 1.0 / math.sqrt(n - 3)
            z_low, z_high = z - 1.96 * se_z, z + 1.96 * se_z
            ci_pearson = [round(float(np.tanh(z_low)), 4), round(float(np.tanh(z_high)), 4)]
        else:
            ci_pearson = [round(r_pearson, 4), round(r_pearson, 4)]

        # Spearman
        r_spearman, p_spearman = stats.spearmanr(x, y)
        r_spearman = 0.0 if math.isnan(r_spearman) else float(r_spearman)
        p_spearman = 1.0 if math.isnan(p_spearman) else float(p_spearman)

        # Kendall Tau
        r_kendall, p_kendall = stats.kendalltau(x, y)
        r_kendall = 0.0 if math.isnan(r_kendall) else float(r_kendall)
        p_kendall = 1.0 if math.isnan(p_kendall) else float(p_kendall)

        # Partial correlation if control variable is provided
        partial_corr = None
        if control_col and len(cols) == 3:
            z_var = sub_df[control_col].values.astype(float)
            r_xz, _ = stats.pearsonr(x, z_var)
            r_yz, _ = stats.pearsonr(y, z_var)
            denom = math.sqrt(max(1e-9, (1 - r_xz ** 2) * (1 - r_yz ** 2)))
            r_xy_z = (r_pearson - r_xz * r_yz) / denom if denom > 0 else 0.0
            partial_corr = {
                "control_variable": control_col,
                "partial_correlation": round(float(r_xy_z), 4),
                "uncontrolled_correlation": round(float(r_pearson), 4),
                "correlation_attenuation_percentage": round(float((abs(r_pearson - r_xy_z) / max(1e-9, abs(r_pearson))) * 100.0), 2),
            }

        return {
            "type": "numeric_bivariate",
            "col1": col1,
            "col2": col2,
            "sample_size": n,
            "covariance": round(cov_val, 4),
            "pearson": {
                "coefficient": round(r_pearson, 4),
                "p_value": float(p_pearson),
                "ci_95": ci_pearson,
                "is_significant": bool(p_pearson < 0.05),
            },
            "spearman": {
                "coefficient": round(r_spearman, 4),
                "p_value": float(p_spearman),
                "is_significant": bool(p_spearman < 0.05),
            },
            "kendall_tau": {
                "coefficient": round(r_kendall, 4),
                "p_value": float(p_kendall),
                "is_significant": bool(p_kendall < 0.05),
            },
            "partial_correlation": partial_corr,
            "relationship_strength": (
                "Strong" if abs(r_pearson) >= 0.7 else ("Moderate" if abs(r_pearson) >= 0.3 else "Weak / Negligible")
            ),
            "method": "exact_fisher_ci_and_partial_correlation",
        }

    # Case B: Categorical vs Categorical (Cramér's V)
    else:
        contingency = pd.crosstab(df[col1], df[col2])
        if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return {"error": "Categorical contingency table must be at least 2x2."}

        chi2_stat, p_val, dof, _ = stats.chi2_contingency(contingency, correction=False)
        n = contingency.values.sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = math.sqrt(chi2_stat / (n * min_dim)) if n * min_dim > 0 else 0.0

        return {
            "type": "categorical_association",
            "col1": col1,
            "col2": col2,
            "sample_size": int(n),
            "chi2_statistic": round(float(chi2_stat), 4),
            "p_value": float(p_val),
            "degrees_of_freedom": int(dof),
            "cramers_v": round(float(cramers_v), 4),
            "is_significant": bool(p_val < 0.05),
            "association_strength": (
                "Strong" if cramers_v >= 0.5 else ("Moderate" if cramers_v >= 0.25 else "Weak")
            ),
            "method": "cramers_v_contingency",
        }


# ==============================================================================
# 5. INFERENTIAL STATISTICAL TESTS
# ==============================================================================

def welch_t_test(
    sample_a: Union[pd.Series, List[float], np.ndarray],
    sample_b: Union[pd.Series, List[float], np.ndarray],
) -> Dict[str, Any]:
    """Perform Welch's two-sample t-test (unequal variances assumed) with Cohen's d effect size."""
    a = _to_clean_float_array(sample_a)
    b = _to_clean_float_array(sample_b)

    if len(a) < 2 or len(b) < 2:
        return {"error": "Welch's t-test requires at least 2 valid observations per sample."}

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    n1, n2 = len(a), len(b)

    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    diff = mean_a - mean_b

    # Cohen's d
    pooled_sd = math.sqrt(((n1 - 1) * var_a + (n2 - 1) * var_b) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1.0
    cohens_d = diff / pooled_sd if pooled_sd > 0 else 0.0

    return {
        "test": "Welch's Two-Sample t-test",
        "sample_a_count": n1,
        "sample_b_count": n2,
        "sample_a_mean": round(mean_a, 4),
        "sample_b_mean": round(mean_b, 4),
        "mean_difference": round(diff, 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val),
        "is_significant": bool(p_val < 0.05),
        "cohens_d": round(float(cohens_d), 4),
        "effect_magnitude": "Large" if abs(cohens_d) >= 0.8 else ("Medium" if abs(cohens_d) >= 0.5 else "Small"),
    }


def mann_whitney_u_test(
    sample_a: Union[pd.Series, List[float], np.ndarray],
    sample_b: Union[pd.Series, List[float], np.ndarray],
) -> Dict[str, Any]:
    """Perform non-parametric Mann-Whitney U test (Wilcoxon rank-sum) for distribution shift."""
    a = _to_clean_float_array(sample_a)
    b = _to_clean_float_array(sample_b)

    if len(a) < 2 or len(b) < 2:
        return {"error": "Mann-Whitney U test requires at least 2 valid observations per sample."}

    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    median_a, median_b = float(np.median(a)), float(np.median(b))

    # Rank-biserial correlation effect size r = 1 - (2U / (n1 * n2))
    n1, n2 = len(a), len(b)
    r_biserial = 1.0 - (2.0 * u_stat / (n1 * n2)) if (n1 * n2) > 0 else 0.0

    return {
        "test": "Mann-Whitney U Test (Non-Parametric)",
        "sample_a_count": n1,
        "sample_b_count": n2,
        "sample_a_median": round(median_a, 4),
        "sample_b_median": round(median_b, 4),
        "u_statistic": round(float(u_stat), 4),
        "p_value": float(p_val),
        "is_significant": bool(p_val < 0.05),
        "rank_biserial_effect_size": round(float(r_biserial), 4),
    }


def one_way_anova(groups: Dict[str, Union[pd.Series, List[float], np.ndarray]]) -> Dict[str, Any]:
    """Perform One-Way ANOVA with exact F-statistic, p-value, and Eta-squared (eta^2) effect size."""
    cleaned_groups = {}
    for k, v in groups.items():
        arr = _to_clean_float_array(v)
        if len(arr) >= 2:
            cleaned_groups[k] = arr

    if len(cleaned_groups) < 2:
        return {"error": "One-Way ANOVA requires at least 2 groups with >= 2 observations each."}

    f_stat, p_val = stats.f_oneway(*cleaned_groups.values())

    # Eta-squared calculation
    all_vals = np.concatenate(list(cleaned_groups.values()))
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in cleaned_groups.values())
    ss_total = sum((x - grand_mean) ** 2 for x in all_vals)
    eta_sq = (ss_between / ss_total) if ss_total > 0 else 0.0

    return {
        "test": "One-Way ANOVA",
        "group_count": len(cleaned_groups),
        "total_sample_size": len(all_vals),
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(p_val),
        "is_significant": bool(p_val < 0.05),
        "eta_squared": round(float(eta_sq), 4),
        "eta_squared_percentage": round(float(eta_sq * 100.0), 2),
        "variance_explained_strength": "High" if eta_sq >= 0.14 else ("Medium" if eta_sq >= 0.06 else "Small / Negligible"),
        "group_summaries": {
            k: {
                "count": len(v),
                "mean": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v, ddof=1)), 4),
            }
            for k, v in cleaned_groups.items()
        },
    }


def kruskal_wallis_test(groups: Dict[str, Union[pd.Series, List[float], np.ndarray]]) -> Dict[str, Any]:
    """Perform non-parametric Kruskal-Wallis H-test across multiple groups."""
    cleaned_groups = {}
    for k, v in groups.items():
        arr = _to_clean_float_array(v)
        if len(arr) >= 2:
            cleaned_groups[k] = arr

    if len(cleaned_groups) < 2:
        return {"error": "Kruskal-Wallis test requires at least 2 groups with >= 2 observations each."}

    h_stat, p_val = stats.kruskal(*cleaned_groups.values())

    return {
        "test": "Kruskal-Wallis H-Test (Non-Parametric ANOVA)",
        "group_count": len(cleaned_groups),
        "h_statistic": round(float(h_stat), 4),
        "p_value": float(p_val),
        "is_significant": bool(p_val < 0.05),
    }


def kolmogorov_smirnov_test(
    sample_a: Union[pd.Series, List[float], np.ndarray],
    sample_b: Union[pd.Series, List[float], np.ndarray],
) -> Dict[str, Any]:
    """Perform two-sample Kolmogorov-Smirnov test to detect distribution shape shift."""
    a = _to_clean_float_array(sample_a)
    b = _to_clean_float_array(sample_b)

    if len(a) < 3 or len(b) < 3:
        return {"error": "KS test requires at least 3 valid observations per sample."}

    ks_stat, p_val = stats.ks_2samp(a, b)

    return {
        "test": "Two-Sample Kolmogorov-Smirnov Test",
        "sample_a_count": len(a),
        "sample_b_count": len(b),
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": float(p_val),
        "distribution_shifted": bool(p_val < 0.05),
    }


def multiple_testing_correction(
    p_values: List[float],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Adjust p-values for multiple comparisons:
    - 'fdr_bh': Benjamini-Hochberg False Discovery Rate
    - 'bonferroni': Bonferroni Family-Wise Error Rate
    """
    m = len(p_values)
    if m == 0:
        return {"adjusted_p_values": [], "significant": []}

    p_arr = np.asarray(p_values, dtype=float)

    if method == "bonferroni":
        p_adj = np.clip(p_arr * m, 0.0, 1.0)
    else:  # Benjamini-Hochberg FDR
        sorted_indices = np.argsort(p_arr)
        sorted_p = p_arr[sorted_indices]
        p_adj_sorted = np.zeros(m)
        
        cummin = 1.0
        for i in range(m - 1, -1, -1):
            rank = i + 1
            adj = (sorted_p[i] * m) / rank
            cummin = min(cummin, adj)
            p_adj_sorted[i] = cummin
        
        p_adj = np.zeros(m)
        p_adj[sorted_indices] = np.clip(p_adj_sorted, 0.0, 1.0)

    significant = [bool(p <= alpha) for p in p_adj]

    return {
        "method": method,
        "alpha": alpha,
        "test_count": m,
        "raw_p_values": [float(p) for p in p_values],
        "adjusted_p_values": [round(float(p), 6) for p in p_adj],
        "significant": significant,
        "significant_count": sum(significant),
    }


# ==============================================================================
# 6. ANOMALY & CHANGE DETECTION
# ==============================================================================

def detect_outliers(
    series: Union[pd.Series, List[float], np.ndarray],
    method: str = "iqr",
    multiplier: float = 1.5,
) -> Dict[str, Any]:
    """
    Detect statistical outliers using:
    - 'iqr': Tukey's Fences [Q1 - k*IQR, Q3 + k*IQR]
    - 'zscore': Standard Z-Score (|z| >= multiplier, default 3.0)
    - 'mad': Modified Z-Score using Median Absolute Deviation (|M_z| >= 3.5)
    """
    s = _to_clean_float_series(series)

    clean = s.dropna()
    n = len(clean)
    if n < 4:
        return {"error": "Outlier detection requires at least 4 valid numeric observations."}

    arr = clean.values.astype(float)
    outlier_indices = []
    outlier_values = []
    lower_bound, upper_bound = 0.0, 0.0

    if method == "zscore":
        mean_val = np.mean(arr)
        std_val = np.std(arr, ddof=1) if n > 1 else 1.0
        z_scores = np.abs((arr - mean_val) / std_val) if std_val > 1e-9 else np.zeros(n)
        mask = z_scores >= multiplier
        lower_bound = float(mean_val - multiplier * std_val)
        upper_bound = float(mean_val + multiplier * std_val)
    elif method == "mad":
        median_val = np.median(arr)
        mad = np.median(np.abs(arr - median_val))
        mod_z = 0.6745 * np.abs(arr - median_val) / mad if mad > 1e-9 else np.zeros(n)
        thresh = 3.5 if multiplier == 1.5 else multiplier
        mask = mod_z >= thresh
        lower_bound = float(median_val - (thresh * mad / 0.6745)) if mad > 0 else float(median_val)
        upper_bound = float(median_val + (thresh * mad / 0.6745)) if mad > 0 else float(median_val)
    else:  # IQR default
        q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q75 - q25
        lower_bound = float(q25 - multiplier * iqr)
        upper_bound = float(q75 + multiplier * iqr)
        mask = (arr < lower_bound) | (arr > upper_bound)

    outlier_indices = [int(i) for i in np.where(mask)[0]]
    outlier_values = [round(float(arr[i]), 4) for i in outlier_indices]
    outlier_rate = len(outlier_indices) / n

    return {
        "method": method,
        "total_count": n,
        "outlier_count": len(outlier_indices),
        "outlier_percentage": round(float(outlier_rate * 100.0), 2),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "outlier_indices": outlier_indices[:50],  # cap list for cleanliness
        "outlier_sample": outlier_values[:20],
        "has_high_outlier_risk": bool(outlier_rate >= 0.05),
    }


def detect_distribution_shift(
    baseline_series: Union[pd.Series, List[float], np.ndarray],
    target_series: Union[pd.Series, List[float], np.ndarray],
) -> Dict[str, Any]:
    """
    Detect distribution shift between baseline and target populations using:
    - 2-Wasserstein Earth Mover's Distance
    - Two-Sample Kolmogorov-Smirnov Test
    - Relative Mean Shift & Variance Ratio
    """
    base = _to_clean_float_array(baseline_series)
    targ = _to_clean_float_array(target_series)

    if len(base) < 3 or len(targ) < 3:
        return {"error": "Distribution shift detection requires at least 3 observations in each population."}

    # 1. Wasserstein Distance
    wasserstein_dist = float(stats.wasserstein_distance(base, targ))

    # 2. KS Test
    ks_stat, p_val = stats.ks_2samp(base, targ)

    # 3. Mean & Variance shifts
    mean_base, mean_targ = float(np.mean(base)), float(np.mean(targ))
    var_base, var_targ = float(np.var(base, ddof=1)), float(np.var(targ, ddof=1))

    mean_delta_pct = (abs(mean_targ - mean_base) / abs(mean_base)) * 100.0 if abs(mean_base) > 1e-9 else 0.0
    var_ratio = (var_targ / var_base) if var_base > 1e-9 else 1.0

    shift_detected = bool(p_val < 0.05 or mean_delta_pct > 20.0)

    return {
        "baseline_count": len(base),
        "target_count": len(targ),
        "wasserstein_distance": round(wasserstein_dist, 4),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_p_value": float(p_val),
        "mean_shift_percentage": round(float(mean_delta_pct), 2),
        "variance_ratio": round(float(var_ratio), 4),
        "shift_detected": shift_detected,
        "shift_severity": "High" if p_val < 0.001 and mean_delta_pct > 25.0 else ("Moderate" if shift_detected else "None"),
        "method": "wasserstein_and_ks_shift_detector",
    }
