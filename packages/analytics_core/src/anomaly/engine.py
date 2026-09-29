"""Deterministic Anomaly Detection Engine."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class AnomalyDetectionEngine:
    """Production-grade deterministic anomaly detection engine."""

    def detect_numerical_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> Dict[str, Any]:
        """Detect statistical outliers using IQR or Z-Score."""
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return {"error": f"Column '{column}' is not a valid numeric column."}

        series = df[column].dropna().astype(float)
        n = len(series)
        if n < 4:
            return {"error": "Insufficient data points for outlier detection."}

        outliers = []
        lower_bound, upper_bound = 0.0, 0.0

        if method == "iqr":
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - threshold * iqr
            upper_bound = q75 + threshold * iqr
            mask = (series < lower_bound) | (series > upper_bound)
            outlier_series = series[mask]
        else:  # z-score
            mean = float(series.mean())
            std = float(series.std(ddof=1)) or 1.0
            z_scores = np.abs((series - mean) / std)
            mask = z_scores > threshold
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            outlier_series = series[mask]

        outlier_count = int(mask.sum())
        outlier_percentage = round((outlier_count / n) * 100, 2)

        sample_outliers = [
            {"index": int(idx), "value": float(val), "severity": "severe" if (val < lower_bound * 1.5 or val > upper_bound * 1.5) else "moderate"}
            for idx, val in outlier_series.head(20).items()
        ]

        return {
            "column": column,
            "method": method,
            "threshold": threshold,
            "total_points": n,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "sample_outliers": sample_outliers,
            "has_anomalies": outlier_count > 0,
        }

    def detect_time_series_anomalies(
        self,
        df: pd.DataFrame,
        date_column: str,
        metric_column: str,
        window: int = 5,
        sigma: float = 2.0,
    ) -> Dict[str, Any]:
        """Detect anomalies in time-series data using dynamic rolling baselines."""
        sub_df = df[[date_column, metric_column]].dropna().copy()
        sub_df[date_column] = pd.to_datetime(sub_df[date_column], errors="coerce")
        sub_df = sub_df.dropna().sort_values(by=date_column)

        # Aggregate by date
        agg = sub_df.groupby(date_column)[metric_column].sum().reset_index()
        n = len(agg)
        if n < 4:
            return {"error": "Insufficient time-series observations for dynamic anomaly detection."}

        series = agg[metric_column].astype(float)
        history = series.shift(1)
        min_history = max(3, min(window, 3))
        rolling_mean = history.rolling(window=window, min_periods=min_history).mean()
        rolling_std = history.rolling(window=window, min_periods=min_history).std(ddof=1)

        # Robust fallback for flat/degenerate history.
        global_median = float(history.dropna().median()) if history.notna().any() else 0.0
        global_mad = float(np.median(np.abs(history.dropna().to_numpy(dtype=float) - global_median))) if history.notna().any() else 0.0
        robust_scale = max(1.4826 * global_mad, np.finfo(float).eps)
        rolling_mean = rolling_mean.fillna(history.expanding(min_periods=1).mean()).fillna(global_median)
        rolling_std = rolling_std.replace([np.inf, -np.inf], np.nan).fillna(robust_scale)

        upper_band = rolling_mean + sigma * rolling_std
        lower_band = rolling_mean - sigma * rolling_std
        lower_band = lower_band.clip(lower=0.0)

        anomalies = []
        for i in range(n):
            val = float(series.iloc[i])
            dt = agg[date_column].iloc[i].strftime("%Y-%m-%d")
            exp_val = float(rolling_mean.iloc[i])
            ub = float(upper_band.iloc[i])
            lb = float(lower_band.iloc[i])

            # Do not make a scientific anomaly call without historical context.
            historical_n = int(history.iloc[:i].notna().sum())
            if historical_n < min_history:
                continue

            if val < lb or val > ub:
                deviation_pct = round(((val - exp_val) / exp_val) * 100, 2) if exp_val != 0 else 0.0
                anomalies.append({
                    "date": dt,
                    "actual_value": round(val, 2),
                    "expected_baseline": round(exp_val, 2),
                    "lower_band": round(lb, 2),
                    "upper_band": round(ub, 2),
                    "deviation_pct": deviation_pct,
                    "anomaly_direction": "Significant Drop" if val < lb else "Significant Spike",
                })

        return {
            "date_column": date_column,
            "metric_column": metric_column,
            "total_periods": n,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "has_anomalies": len(anomalies) > 0,
        }

    def detect_category_drift(
        self,
        df_current: pd.DataFrame,
        df_baseline: pd.DataFrame,
        categorical_column: str,
        threshold_pct_shift: float = 10.0,
    ) -> Dict[str, Any]:
        """Detect shifts in category proportions between baseline and current data."""
        if categorical_column not in df_current.columns or categorical_column not in df_baseline.columns:
            return {"error": f"Column '{categorical_column}' not found in both datasets."}

        p_curr = df_current[categorical_column].value_counts(normalize=True) * 100.0
        p_base = df_baseline[categorical_column].value_counts(normalize=True) * 100.0

        all_cats = set(p_curr.index).union(set(p_base.index))
        drifting_categories = []

        for cat in all_cats:
            c_pct = float(p_curr.get(cat, 0.0))
            b_pct = float(p_base.get(cat, 0.0))
            diff = c_pct - b_pct
            if abs(diff) >= threshold_pct_shift:
                drifting_categories.append({
                    "category": str(cat),
                    "baseline_share_pct": round(b_pct, 2),
                    "current_share_pct": round(c_pct, 2),
                    "shift_pct_points": round(diff, 2),
                    "direction": "Surge" if diff > 0 else "Decline",
                })

        drifting_categories.sort(key=lambda x: abs(x["shift_pct_points"]), reverse=True)

        return {
            "categorical_column": categorical_column,
            "drift_detected": len(drifting_categories) > 0,
            "drifting_categories": drifting_categories,
        }
