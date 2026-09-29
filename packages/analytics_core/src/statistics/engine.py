"""Deterministic Statistical Computation Engine."""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from packages.analytics_core.src.statistics.method_selection import (
    StatisticalMethodDecision,
    select_two_group,
    select_multi_group,
    select_categorical,
    select_correlation,
)
from packages.analytics_core.src.statistics.multiple_comparisons import pairwise_posthoc, _adjust
from packages.analytics_core.src.statistics.inference import (
    execute_two_group,
    execute_multi_group,
    execute_categorical,
    execute_correlation,
    execute_regression,
)

from packages.analytics_core.src.statistics.analytical_math import (
    concentration_metrics,
    descriptive_summary,
    detect_distribution_shift,
    detect_outliers,
    frequency_table,
    kolmogorov_smirnov_test,
    kruskal_wallis_test,
    mann_whitney_u_test,
    multiple_testing_correction,
    one_way_anova,
    relationship_analysis,
    segment_analysis,
    time_series_summary,
    weighted_mean,
    welch_t_test,
)


class StatisticalEngine:
    """Production-grade deterministic statistical computation engine."""

    # Phase 18: data-aware inferential method selection and execution.
    def select_two_group_method(self, sample_a, sample_b, *, paired: bool = False, alpha: float = 0.05) -> Dict[str, Any]:
        return select_two_group(sample_a, sample_b, paired=paired, alpha=alpha).to_dict()

    def select_multi_group_method(self, groups: Dict[str, Union[pd.Series, List[float]]], *, alpha: float = 0.05) -> Dict[str, Any]:
        return select_multi_group(groups, alpha=alpha).to_dict()

    def select_categorical_method(self, df: pd.DataFrame, col1: str, col2: str, *, alpha: float = 0.05) -> Dict[str, Any]:
        return select_categorical(pd.crosstab(df[col1], df[col2]), alpha=alpha).to_dict()

    def select_correlation_method(self, sample_a, sample_b, *, alpha: float = 0.05) -> Dict[str, Any]:
        return select_correlation(sample_a, sample_b, alpha=alpha).to_dict()

    def infer_two_group(self, sample_a, sample_b, *, paired: bool = False, alpha: float = 0.05) -> Dict[str, Any]:
        return execute_two_group(sample_a, sample_b, paired=paired, alpha=alpha)

    def infer_multi_group(self, groups: Dict[str, Union[pd.Series, List[float]]], *, alpha: float = 0.05) -> Dict[str, Any]:
        return execute_multi_group(groups, alpha=alpha)

    def posthoc_multi_group(
        self,
        groups: Dict[str, Union[pd.Series, List[float]]],
        *,
        omnibus_method: str,
        omnibus_p_value: Optional[float] = None,
        alpha: float = 0.05,
        correction: str = "holm",
        require_significant_omnibus: bool = True,
    ) -> Dict[str, Any]:
        """Run an auditable, multiplicity-controlled pairwise family."""
        return pairwise_posthoc(
            groups, omnibus_method=omnibus_method, omnibus_p_value=omnibus_p_value,
            alpha=alpha, correction=correction, require_significant_omnibus=require_significant_omnibus
        )

    def infer_categorical(self, df: pd.DataFrame, col1: str, col2: str, *, alpha: float = 0.05) -> Dict[str, Any]:
        return execute_categorical(df, col1, col2, alpha=alpha)

    def infer_correlation(self, sample_a, sample_b, *, alpha: float = 0.05) -> Dict[str, Any]:
        return execute_correlation(sample_a, sample_b, alpha=alpha)

    def infer_regression(self, df: pd.DataFrame, x_columns: List[str], y_column: str, *, alpha: float = 0.05) -> Dict[str, Any]:
        return execute_regression(df, x_columns, y_column, alpha=alpha)


    def descriptive_summary(self, series: pd.Series) -> Dict[str, Any]:
        """Compute full descriptive statistics deterministically."""
        return descriptive_summary(series)

    def concentration_metrics(self, series: Union[pd.Series, List[float], np.ndarray]) -> Dict[str, Any]:
        """Compute Gini coefficient, HHI, Top-K shares, and Pareto 80/20 ratio."""
        return concentration_metrics(series)

    def frequency_table(self, series: pd.Series, top_n: int = 20) -> List[Dict[str, Any]]:
        """Compute category frequency distribution with percentages and cumulative totals."""
        return frequency_table(series, top_n=top_n)

    def time_series_summary(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_col: str,
        rolling_window: int = 3,
    ) -> Dict[str, Any]:
        """Compute period-over-period change, growth rate, trend, rolling stats, and CUSUM change-point."""
        return time_series_summary(df, time_col=time_col, value_col=value_col, rolling_window=rolling_window)

    def segment_analysis(
        self,
        df: pd.DataFrame,
        segment_col: str,
        metric_col: str,
    ) -> Dict[str, Any]:
        """Compute deep segment analysis, Pareto 80/20, ranking, and concentration."""
        return segment_analysis(df, segment_col=segment_col, metric_col=metric_col)

    def relationship_analysis(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
        control_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute correlation (Pearson, Spearman, Kendall, Partial) or categorical association."""
        return relationship_analysis(df, col1=col1, col2=col2, control_col=control_col)

    def two_sample_t_test(
        self,
        sample_a: Union[pd.Series, List[float], np.ndarray],
        sample_b: Union[pd.Series, List[float], np.ndarray],
        equal_var: bool = False,
    ) -> Dict[str, Any]:
        """Perform two-sample Welch's or Student's t-test with effect size and CI."""
        if not equal_var:
            res = welch_t_test(sample_a, sample_b)
            if "error" in res:
                return res
            # Maintain backward compatible fields
            return {
                "test_type": "Welch's Two-Sample t-test",
                "t_statistic": res["t_statistic"],
                "p_value": res["p_value"],
                "is_significant": res["is_significant"],
                "sample_a_count": res["sample_a_count"],
                "sample_a_mean": res["sample_a_mean"],
                "sample_b_count": res["sample_b_count"],
                "sample_b_mean": res["sample_b_mean"],
                "mean_difference": res["mean_difference"],
                "cohens_d": res["cohens_d"],
                "conclusion": (
                    f"Statistically significant difference (p = {res['p_value']:.4e} < 0.05, Cohen's d = {res['cohens_d']:.2f})."
                    if res["is_significant"]
                    else f"No statistically significant difference (p = {res['p_value']:.4f} >= 0.05)."
                ),
            }
        else:
            a = np.asarray(sample_a, dtype=float)
            b = np.asarray(sample_b, dtype=float)
            a = a[~np.isnan(a)]
            b = b[~np.isnan(b)]
            if len(a) < 2 or len(b) < 2:
                return {"error": "Both samples must contain at least 2 observations."}
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=True)
            mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
            diff = mean_a - mean_b
            n1, n2 = len(a), len(b)
            s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
            pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1.0
            cohens_d = diff / pooled_sd if pooled_sd > 0 else 0.0
            is_sig = bool(p_val < 0.05)
            return {
                "test_type": "Student's t-test",
                "t_statistic": round(float(t_stat), 4),
                "p_value": float(p_val),
                "is_significant": is_sig,
                "sample_a_count": len(a),
                "sample_a_mean": round(mean_a, 4),
                "sample_b_count": len(b),
                "sample_b_mean": round(mean_b, 4),
                "mean_difference": round(diff, 4),
                "cohens_d": round(float(cohens_d), 4),
                "conclusion": f"Difference: {diff:.4f} (p = {p_val:.4f}).",
            }

    def mann_whitney_u_test(
        self,
        sample_a: Union[pd.Series, List[float], np.ndarray],
        sample_b: Union[pd.Series, List[float], np.ndarray],
    ) -> Dict[str, Any]:
        """Perform non-parametric Mann-Whitney U test."""
        return mann_whitney_u_test(sample_a, sample_b)

    def one_way_anova(self, groups: Dict[str, Union[pd.Series, List[float]]]) -> Dict[str, Any]:
        """Perform one-way ANOVA across multiple groups."""
        res = one_way_anova(groups)
        if "error" in res:
            return res
        return {
            "test_type": "One-Way ANOVA",
            "f_statistic": res["f_statistic"],
            "p_value": res["p_value"],
            "is_significant": res["is_significant"],
            "eta_squared": res["eta_squared"],
            "group_count": res["group_count"],
            "group_summaries": res["group_summaries"],
            "conclusion": (
                f"Statistically significant variance across groups (F = {res['f_statistic']:.2f}, p = {res['p_value']:.4e}, eta2 = {res['eta_squared']:.3f})."
                if res["is_significant"]
                else f"No significant variance across groups (F = {res['f_statistic']:.2f}, p = {res['p_value']:.4f})."
            ),
        }

    def kruskal_wallis_test(self, groups: Dict[str, Union[pd.Series, List[float], np.ndarray]]) -> Dict[str, Any]:
        """Perform non-parametric Kruskal-Wallis test."""
        return kruskal_wallis_test(groups)

    def chi_squared_test(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """Perform Chi-Squared Test of Independence between two categorical columns."""
        contingency = pd.crosstab(df[col1], df[col2])
        if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return {"error": "Contingency table must be at least 2x2."}

        chi2_stat, p_val, dof, _ = stats.chi2_contingency(contingency)
        n = contingency.values.sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if n * min_dim > 0 else 0.0
        is_significant = bool(p_val < 0.05)

        return {
            "test_type": "Chi-Squared Test of Independence",
            "chi2_statistic": round(float(chi2_stat), 4),
            "p_value": float(p_val),
            "degrees_of_freedom": int(dof),
            "cramers_v": round(float(cramers_v), 4),
            "is_significant": is_significant,
            "table_dimensions": [int(contingency.shape[0]), int(contingency.shape[1])],
            "conclusion": (
                f"Significant association detected between '{col1}' and '{col2}' (Chi2 = {chi2_stat:.2f}, p = {p_val:.4e}, Cramér's V = {cramers_v:.3f})."
                if is_significant
                else f"No significant association detected between '{col1}' and '{col2}' (p = {p_val:.4f})."
            ),
        }

    def kolmogorov_smirnov_test(
        self,
        sample_a: Union[pd.Series, List[float], np.ndarray],
        sample_b: Union[pd.Series, List[float], np.ndarray],
    ) -> Dict[str, Any]:
        """Perform two-sample Kolmogorov-Smirnov test."""
        return kolmogorov_smirnov_test(sample_a, sample_b)

    def multiple_testing_correction(
        self,
        p_values: List[float],
        method: str = "fdr_bh",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Perform multiple comparison adjustment."""
        return multiple_testing_correction(p_values, method=method, alpha=alpha)

    def detect_outliers(
        self,
        series: Union[pd.Series, List[float], np.ndarray],
        method: str = "iqr",
        multiplier: float = 1.5,
    ) -> Dict[str, Any]:
        """Detect outliers using IQR, Z-Score, or Modified Z-Score (MAD)."""
        return detect_outliers(series, method=method, multiplier=multiplier)

    def detect_distribution_shift(
        self,
        baseline_series: Union[pd.Series, List[float], np.ndarray],
        target_series: Union[pd.Series, List[float], np.ndarray],
    ) -> Dict[str, Any]:
        """Detect distribution shift via Wasserstein distance and KS test."""
        return detect_distribution_shift(baseline_series, target_series)

    def correlation_analysis(
        self, df: pd.DataFrame, columns: List[str], method: str = "pearson", correction: str = "fdr_bh"
    ) -> Dict[str, Any]:
        """Pairwise correlations with family-level multiplicity control."""
        valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if len(valid_cols) < 2:
            return {"error": "Correlation requires at least 2 numeric columns."}
        sub_df = df[valid_cols].dropna()
        n = len(sub_df)
        if n < 3:
            return {"error": "Insufficient data points for correlation (minimum 3 required)."}

        pairs=[]; raw=[]
        for i, c1 in enumerate(valid_cols):
            for j in range(i+1, len(valid_cols)):
                c2=valid_cols[j]
                if method == "spearman":
                    r,p=stats.spearmanr(sub_df[c1], sub_df[c2])
                else:
                    r,p=stats.pearsonr(sub_df[c1], sub_df[c2])
                r=float(r) if np.isfinite(r) else 0.0
                p=float(p) if np.isfinite(p) else 1.0
                pairs.append((c1,c2,r,p)); raw.append(p)
        adjusted, corr_key = _adjust(raw, correction, 0.05)

        corr_matrix={c:{d:(1.0 if c==d else 0.0) for d in valid_cols} for c in valid_cols}
        pval_matrix={c:{d:(0.0 if c==d else 1.0) for d in valid_cols} for c in valid_cols}
        adj_matrix={c:{d:(0.0 if c==d else 1.0) for d in valid_cols} for c in valid_cols}
        strong=[]
        for (c1,c2,r,p), adj in zip(pairs, adjusted):
            corr_matrix[c1][c2]=corr_matrix[c2][c1]=round(r,4)
            pval_matrix[c1][c2]=pval_matrix[c2][c1]=p
            adj=float(adj); adj_matrix[c1][c2]=adj_matrix[c2][c1]=adj
            if abs(r)>=0.5 and adj < 0.05:
                strong.append({"col1":c1,"col2":c2,"correlation":round(r,4),"p_value":p,"adjusted_p_value":adj,"relationship":"Strong positive" if r>0 else "Strong negative"})
        return {"method":method,"columns":valid_cols,"sample_size":n,"comparison_count":len(pairs),"multiple_testing_correction":corr_key,"correlation_matrix":corr_matrix,"p_value_matrix":pval_matrix,"adjusted_p_value_matrix":adj_matrix,"strong_correlations":strong}

    def linear_regression_analysis(
        self, df: pd.DataFrame, x_columns: List[str], y_column: str,
        *, cluster_column: Optional[str] = None, time_column: Optional[str] = None, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Backward-compatible facade for the hardened regression inference engine."""
        return self.infer_regression(
            df, x_columns, y_column, alpha=alpha,
            cluster_column=cluster_column, time_column=time_column
        )
