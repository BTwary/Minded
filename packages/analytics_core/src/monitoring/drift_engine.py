"""DriftMonitorEngine: Continuous non-parametric drift detection (Kolmogorov-Smirnov, PSI, and CUSUM)."""
from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
from scipy import stats


@dataclass
class DriftAlert:
    """Alert triggered by statistical distribution drift."""
    table_name: str
    column_name: str
    drift_type: str  # 'CONTINUOUS_KS', 'CATEGORICAL_PSI', 'TIMESERIES_CUSUM'
    severity_score: float
    p_value: Optional[float]
    auto_investigation_question: str


class DriftMonitorEngine:
    """Non-parametric drift monitoring engine for autonomous anomaly detection."""

    KS_P_VALUE_THRESHOLD = 0.01
    PSI_SEVERE_THRESHOLD = 0.20
    CUSUM_THRESHOLD_MULTIPLIER = 4.0

    @staticmethod
    def evaluate_continuous_drift(
        baseline: np.ndarray,
        recent: np.ndarray,
        col_name: str,
        table_name: str,
    ) -> Optional[DriftAlert]:
        """Uses Kolmogorov-Smirnov test to detect distribution shifts in continuous metrics."""
        if len(baseline) < 30 or len(recent) < 30:
            return None

        stat, p_value = stats.ks_2samp(baseline, recent)

        if p_value < DriftMonitorEngine.KS_P_VALUE_THRESHOLD:
            question = (
                f"Why has the distribution of '{col_name}' in '{table_name}' shifted "
                f"significantly in the last 7 days compared to the historical baseline?"
            )
            return DriftAlert(
                table_name=table_name,
                column_name=col_name,
                drift_type="CONTINUOUS_KS",
                severity_score=float(1.0 - p_value),
                p_value=float(p_value),
                auto_investigation_question=question,
            )
        return None

    @staticmethod
    def evaluate_categorical_drift(
        baseline_counts: Dict[str, int],
        recent_counts: Dict[str, int],
        col_name: str,
        table_name: str,
    ) -> Optional[DriftAlert]:
        """Uses Population Stability Index (PSI) to detect categorical mix shifts."""
        total_base = sum(baseline_counts.values())
        total_recent = sum(recent_counts.values())

        if total_base == 0 or total_recent == 0:
            return None

        psi = 0.0
        all_keys = set(baseline_counts.keys()).union(set(recent_counts.keys()))

        for k in all_keys:
            pct_base = (baseline_counts.get(k, 0) + 0.0001) / total_base
            pct_recent = (recent_counts.get(k, 0) + 0.0001) / total_recent
            psi += (pct_recent - pct_base) * np.log(pct_recent / pct_base)

        if psi > DriftMonitorEngine.PSI_SEVERE_THRESHOLD:
            question = (
                f"What is driving the sudden mix shift in the categorical distribution "
                f"of '{col_name}' in '{table_name}'?"
            )
            return DriftAlert(
                table_name=table_name,
                column_name=col_name,
                drift_type="CATEGORICAL_PSI",
                severity_score=float(min(psi / 0.5, 1.0)),
                p_value=None,
                auto_investigation_question=question,
            )
        return None

    @staticmethod
    def evaluate_timeseries_cusum(
        timeseries: np.ndarray,
        col_name: str,
        table_name: str,
        warmup_period: int = 20,
    ) -> Optional[DriftAlert]:
        """Uses CUSUM control chart to detect sustained mean shifts in sequential time-series."""
        if len(timeseries) <= warmup_period + 5:
            return None

        baseline_mu = float(np.mean(timeseries[:warmup_period]))
        baseline_sigma = float(np.std(timeseries[:warmup_period]))
        if baseline_sigma < 1e-9:
            return None

        k = 0.5 * baseline_sigma
        h = DriftMonitorEngine.CUSUM_THRESHOLD_MULTIPLIER * baseline_sigma

        s_pos = 0.0
        s_neg = 0.0

        for val in timeseries[warmup_period:]:
            s_pos = max(0.0, s_pos + (val - baseline_mu) - k)
            s_neg = max(0.0, s_neg - (val - baseline_mu) - k)
            if s_pos > h or s_neg > h:
                direction = "increase" if s_pos > h else "decrease"
                question = (
                    f"Why did '{col_name}' in '{table_name}' experience a sustained "
                    f"statistical {direction} in its time-series trend?"
                )
                return DriftAlert(
                    table_name=table_name,
                    column_name=col_name,
                    drift_type="TIMESERIES_CUSUM",
                    severity_score=float(min(max(s_pos, s_neg) / (h * 2.0), 1.0)),
                    p_value=None,
                    auto_investigation_question=question,
                )
        return None
