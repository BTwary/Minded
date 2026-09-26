"""EvidencePatternDetector: extensible boundary for evidence-driven emergent hypothesis discovery.

OBSERVATION != HYPOTHESIS.
A detector discovers evidence patterns in raw observation data;
a hypothesis synthesizer interprets that evidence to generate candidate explanations.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class CandidateExplanation:
    """A pattern a detector found in a result dataframe that warrants hypothesis formation."""
    pattern_name: str
    dimension_col: Optional[str] = None
    value_col: Optional[str] = None
    dominant_value: Optional[Any] = None
    dominant_share: Optional[float] = None
    description: str = ""
    triggering_evidence: Optional[Dict[str, Any]] = None




def _resolve_explicit_column(
    metadata: Dict[str, Any],
    keys: tuple[str, ...],
    candidates: List[str],
) -> Optional[str]:
    """Resolve a column only from explicit analytical metadata or a unique candidate.

    Column order is never treated as analytical evidence.  If the caller has not
    bound the estimand to a column and more than one candidate remains, the
    detector fails closed rather than silently analysing an arbitrary column.
    """
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value in candidates:
            return value
    return candidates[0] if len(candidates) == 1 else None

class EvidencePatternDetector:
    """Base class for evidence-pattern detectors."""
    name: str = "base"

    def detect(self, result_df: pd.DataFrame, already_modeled_dims: Dict[str, Any], is_additive: bool = True) -> Optional[CandidateExplanation]:
        raise NotImplementedError


class ConcentrationPatternDetector(EvidencePatternDetector):
    """Detects a single categorical value dominating observed variance (> 40% share).

    Phase 11 Part 12: "share of total" is only a valid statement for an
    ADDITIVE metric (SUM). For a non-additive metric (rate/ratio/proportion/
    mean/weighted-mean), summing the column and reporting one row's fraction
    of that sum is meaningless (e.g. summing four segments' conversion rates
    has no business interpretation). When `is_additive=False`, this detector
    instead looks for a materially divergent value against the rest.
    """
    name = "concentration"
    CONCENTRATION_THRESHOLD = 0.40
    # For non-additive metrics: a value at least this many times the median
    # of the rest is considered a material divergence worth surfacing.
    DIVERGENCE_RATIO_THRESHOLD = 1.5

    def detect(self, result_df: pd.DataFrame, already_modeled_dims: Dict[str, Any], is_additive: bool = True) -> Optional[CandidateExplanation]:
        if result_df is None or result_df.empty:
            return None

        numeric_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
        cat_cols = [c for c in result_df.columns if c not in numeric_cols]
        if not numeric_cols or not cat_cols:
            return None

        value_col = _resolve_explicit_column(
            already_modeled_dims,
            ("value_column", "target_metric", "metric", "primary_result_column"),
            numeric_cols,
        )
        dim_col = _resolve_explicit_column(
            already_modeled_dims,
            ("dimension_column", "target_dimension", "dimension"),
            cat_cols,
        )
        if not value_col or not dim_col:
            return None

        if not is_additive:
            # Non-additive metric: compare the top value against the median of
            # the rest instead of computing a share-of-total.
            if len(result_df) < 2:
                return None
            sorted_df = result_df.reindex(result_df[value_col].sort_values(ascending=False).index)
            top_row = sorted_df.iloc[0]
            rest_median = float(sorted_df.iloc[1:][value_col].median())
            top_val = float(top_row[value_col])
            if rest_median == 0 or abs(top_val / rest_median) < self.DIVERGENCE_RATIO_THRESHOLD:
                return None
            top_value = top_row[dim_col]
            already_modeled = str(already_modeled_dims.get(dim_col)) == str(top_value)
            if already_modeled:
                return None
            return CandidateExplanation(
                pattern_name=self.name,
                dimension_col=dim_col,
                value_col=value_col,
                dominant_value=top_value,
                dominant_share=None,
                description=(
                    f"'{top_value}' ({dim_col}) has {value_col}={top_val:.4g}, materially diverging from "
                    f"the median of the remaining categories ({rest_median:.4g}). Reported as a divergent "
                    f"value, not a share of total, because {value_col} is a non-additive metric."
                ),
                triggering_evidence={
                    "dimension": dim_col,
                    "top_value": str(top_value),
                    "top_metric_value": round(top_val, 6),
                    "rest_median": round(rest_median, 6),
                    "is_additive": False,
                },
            )

        total = float(result_df[value_col].abs().sum())
        if total <= 0:
            return None

        sorted_df = result_df.reindex(result_df[value_col].abs().sort_values(ascending=False).index)
        top_row = sorted_df.iloc[0]
        top_share = float(abs(top_row[value_col]) / total)
        top_value = top_row[dim_col]

        already_modeled = str(already_modeled_dims.get(dim_col)) == str(top_value)
        if top_share < self.CONCENTRATION_THRESHOLD or already_modeled:
            return None

        return CandidateExplanation(
            pattern_name=self.name,
            dimension_col=dim_col,
            value_col=value_col,
            dominant_value=top_value,
            dominant_share=top_share,
            description=(
                f"'{top_value}' ({dim_col}) accounts for {round(top_share * 100, 1)}% of total {value_col}, "
                f"exceeding the {int(self.CONCENTRATION_THRESHOLD * 100)}% concentration threshold."
            ),
            triggering_evidence={
                "dimension": dim_col,
                "top_value": str(top_value),
                "top_share_pct": round(top_share * 100, 2),
            },
        )


class SegmentDifferenceDetector(EvidencePatternDetector):
    """Detect subgroup differences only when inferential evidence exists.

    A high coefficient of variation across one-row-per-group aggregates is
    descriptive, not a statistical test. The old implementation promoted such
    CV into a "significant" hypothesis and could create false positives.
    This detector therefore requires at least two observations per group and a
    valid one-way ANOVA or Welch-style fallback with a material effect.
    """
    name = "segment_difference"

    def detect(self, result_df: pd.DataFrame, already_modeled_dims: Dict[str, Any], is_additive: bool = True) -> Optional[CandidateExplanation]:
        if result_df is None or len(result_df) < 6:
            return None
        numeric_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
        cat_cols = [c for c in result_df.columns if c not in numeric_cols]
        if not numeric_cols or not cat_cols:
            return None
        val_col = _resolve_explicit_column(
            already_modeled_dims,
            ("value_column", "target_metric", "metric", "primary_result_column"),
            numeric_cols,
        )
        dim_col = _resolve_explicit_column(
            already_modeled_dims,
            ("dimension_column", "target_dimension", "dimension"),
            cat_cols,
        )
        if not val_col or not dim_col:
            return None
        work = result_df[[dim_col, val_col]].dropna().copy()
        if work[dim_col].nunique() < 2:
            return None
        sizes = work.groupby(dim_col)[val_col].size()
        if sizes.min() < 3:
            # Aggregated group summaries have no within-group variance and
            # therefore cannot support an inferential segment claim.
            return None
        groups = [g[val_col].to_numpy(float) for _, g in work.groupby(dim_col)]
        try:
            from scipy import stats
            statistic, p_value = stats.f_oneway(*groups)
            grand_mean = float(work[val_col].mean())
            ss_total = float(((work[val_col] - grand_mean) ** 2).sum())
            ss_between = float(sum(len(g) * (float(g.mean()) - grand_mean) ** 2 for g in groups))
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        except Exception:
            return None
        # Require both inferential evidence and a material effect. No positive
        # signal is emitted for a noisy small-sample fluctuation.
        if not np.isfinite(p_value) or p_value >= 0.05 or eta_sq < 0.05:
            return None
        top_val = work.groupby(dim_col)[val_col].mean().abs().idxmax()
        return CandidateExplanation(
            pattern_name=self.name,
            dimension_col=dim_col,
            value_col=val_col,
            dominant_value=top_val,
            dominant_share=float(eta_sq),
            description=f"Inferential segment divergence across {dim_col}: ANOVA p={p_value:.4g}, eta-squared={eta_sq:.3f}.",
            triggering_evidence={
                "dimension": dim_col,
                "p_value": float(p_value),
                "eta_squared": float(eta_sq),
                "minimum_group_n": int(sizes.min()),
            },
        )


class TemporalChangeDetector(EvidencePatternDetector):
    """Detects temporal trend change or period acceleration in time-series data."""
    name = "temporal_change"

    def detect(self, result_df: pd.DataFrame, already_modeled_dims: Dict[str, Any], is_additive: bool = True) -> Optional[CandidateExplanation]:
        if result_df is None or len(result_df) < 3:
            return None

        time_cols = [c for c in result_df.columns if "time" in c.lower() or "date" in c.lower() or "month" in c.lower() or "year" in c.lower()]
        numeric_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c]) and c not in time_cols]
        if not time_cols or not numeric_cols:
            return None

        t_col = _resolve_explicit_column(
            already_modeled_dims,
            ("time_column", "temporal_column", "dimension_column"),
            time_cols,
        )
        v_col = _resolve_explicit_column(
            already_modeled_dims,
            ("value_column", "target_metric", "metric", "primary_result_column"),
            numeric_cols,
        )
        if not t_col:
            return None
        if not v_col:
            return None
        vals = result_df[v_col].values
        deltas = np.diff(vals)
        if len(deltas) > 0 and np.max(np.abs(deltas)) > 1.5 * (np.mean(np.abs(deltas)) + 1e-9):
            idx = int(np.argmax(np.abs(deltas)))
            t_val = str(result_df.iloc[idx + 1][t_col])
            return CandidateExplanation(
                pattern_name=self.name,
                dimension_col=t_col,
                value_col=v_col,
                dominant_value=t_val,
                description=f"Sharp temporal transition detected at {t_val} ({t_col}).",
                triggering_evidence={"time_point": t_val, "metric": v_col},
            )
        return None


class AnomalyPatternDetector(EvidencePatternDetector):
    """Detects statistical outlier spikes (> 3 std dev)."""
    name = "anomaly"

    def detect(self, result_df: pd.DataFrame, already_modeled_dims: Dict[str, Any], is_additive: bool = True) -> Optional[CandidateExplanation]:
        if result_df is None or len(result_df) < 4:
            return None

        numeric_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
        if not numeric_cols:
            return None

        v_col = _resolve_explicit_column(
            already_modeled_dims,
            ("value_column", "target_metric", "metric", "primary_result_column"),
            numeric_cols,
        )
        if not v_col:
            return None
        vals = result_df[v_col].values
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std <= 1e-9:
            return None

        z_scores = np.abs((vals - mean) / std)
        max_idx = int(np.argmax(z_scores))
        z_threshold = 2.5 if len(vals) >= 10 else min(2.0, 0.85 * ((len(vals) - 1) / np.sqrt(len(vals))))
        if z_scores[max_idx] >= z_threshold:
            cat_cols = [c for c in result_df.columns if c not in numeric_cols]
            dim_col = cat_cols[0] if cat_cols else None
            dim_val = str(result_df.iloc[max_idx][dim_col]) if dim_col else f"row_{max_idx}"
            return CandidateExplanation(
                pattern_name=self.name,
                dimension_col=dim_col,
                value_col=v_col,
                dominant_value=dim_val,
                dominant_share=float(z_scores[max_idx]),
                description=f"Outlier anomaly detected for '{dim_val}' (z-score = {z_scores[max_idx]:.2f}).",
                triggering_evidence={"target": dim_val, "z_score": round(float(z_scores[max_idx]), 2)},
            )
        return None


class ResidualPatternDetector(EvidencePatternDetector):
    """Detects unexplained variance residuals in subgroup models."""
    name = "residual"

    def detect(self, result_df: pd.DataFrame, already_modeled_dims: Dict[str, Any], is_additive: bool = True) -> Optional[CandidateExplanation]:
        return None  # Extension point


ACTIVE_DETECTORS: List[EvidencePatternDetector] = [
    ConcentrationPatternDetector(),
    SegmentDifferenceDetector(),
    TemporalChangeDetector(),
    AnomalyPatternDetector(),
]
