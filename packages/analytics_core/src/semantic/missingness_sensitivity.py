"""Phase 12: Missingness & Selection-Bias Safety.

Prevents AA-OS from producing a confident analytical conclusion when the
observed data may be systematically selected or missing in a way that could
materially change that conclusion.

This module NEVER claims to discover the true missingness mechanism (MCAR /
MAR / MNAR) from observed data alone -- that is generally not identifiable
without additional assumptions (Rubin 1976; Little & Rubin). Instead it
performs sensitivity analysis under an EXPLICIT, recorded bounding
assumption and reports:

  - whether the resulting interval crosses whatever decision boundary the
    caller's conclusion actually depends on (a fixed threshold, or a
    ranking between two groups), and
  - a categorical epistemic classification (ROBUST / SENSITIVE /
    UNIDENTIFIABLE / INSUFFICIENT_EVIDENCE) derived from that interval --
    never a hand-tuned numeric confidence penalty (see
    AUDIT_PHASE12_MISSINGNESS_SELECTION_BIAS.md Section 6/9 for why: any
    such penalty would be exactly the "arbitrary confidence deduction"
    this phase exists to avoid).

Consumes ``MetricDefinition`` (packages/analytics_core/src/semantic/
metric_semantics.py, v1.1 -- see docs/METRIC_SEMANTICS.md) rather than
assuming SUM universally. SUM, MEAN, COUNT, and RATE/RATIO/PROPORTION each
get their own, mathematically distinct bounding calculation (Section 8 of
the Phase 12 spec) -- they are never conflated.

Bounding assumption used (documented, fixed, and identical for every
investigation -- never invented per-case):

  SUM / event-volume:  each missing value is assumed to plausibly lie
      within the observed [min, max] range of the column's own non-null
      values. Lower bound = all missing values at the observed minimum;
      upper bound = all missing values at the observed maximum.

  MEAN: same per-value range assumption as SUM, but propagated through
      BOTH the numerator (sum) and the denominator (count), since a
      missing observation changes the sample size, not just the total.

  COUNT ("how many X occurred"): the worst-case bound for an occurrence
      count under selective non-response -- lower bound is the observed
      non-null count (current behavior); upper bound assumes every missing
      row was in fact a real, unrecorded occurrence.

  RATE / RATIO / PROPORTION with a recovered numerator/denominator pair
      (Section 7 of docs/METRIC_SEMANTICS.md): the numerator and
      denominator sums are each bounded independently using the SUM rule
      above, and the resulting rate is bounded by evaluating all four
      corner combinations (interval arithmetic) rather than assuming the
      direction of the effect.

  RATE / RATIO / PROPORTION with no recoverable numerator/denominator
      (already degraded to WEIGHTED_MEAN/MEAN by the resolver -- see
      docs/METRIC_SEMANTICS.md Section 7): degrades honestly to the MEAN
      rule on the raw column, exactly mirroring the resolver's own honesty
      policy rather than fabricating a numerator/denominator split it
      cannot observe.

If the required inputs for a defensible bound are not available (fewer than
two non-null observations, an unrecognized/unresolved metric with no safe
fallback, or a non-finite computed bound), the result is
INSUFFICIENT_EVIDENCE, never a fabricated interval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition
from packages.schemas.src.analysis import AggregationType

# --- Epistemic classifications (Phase 12 spec Section 5) -------------------
ROBUST = "ROBUST"
SENSITIVE = "SENSITIVE"
UNIDENTIFIABLE = "UNIDENTIFIABLE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

# --- Fixed, documented thresholds (Section 6: NOT per-case tuning) ---------
# Below this many non-null observations, no defensible [min, max] bound can
# be established at all.
_MIN_NONNULL_FOR_BOUND = 2
# When no explicit decision boundary or competing group is supplied, a
# relative sensitivity width above this fraction of the observed estimate's
# magnitude is treated as material ("crosses an implicit boundary"). This is
# a single fixed materiality default applied identically everywhere, not a
# per-investigation confidence dial.
_DEFAULT_MATERIALITY_RELATIVE_WIDTH = 0.15
# When the *relevant* missingness rate (the rate on the specific group/
# column driving the affected conclusion) is at or above this level, plus
# the conclusion is sensitive, the interval is judged too data-starved to
# defend even a wide range -- UNIDENTIFIABLE rather than SENSITIVE.
_UNIDENTIFIABLE_MISSING_RATE = 0.50

_RATE_LIKE = (AggregationType.RATE, AggregationType.RATIO, AggregationType.PROPORTION)


def _is_finite(x: Optional[float]) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


@dataclass
class SensitivityResult:
    """Structured, reproducible record of one missingness sensitivity analysis.

    Every field required by Phase 12 spec Section 20 (provenance) is present
    here as a distinct, typed value -- never only a narrative sentence -- so
    the calculation can be reproduced by another process from this record
    alone.
    """
    target_column: str
    grouping_column: Optional[str]
    metric_name: str
    aggregation_type: str
    semantic_resolution_status: str
    metric_definition_version: str = "v1.1"

    claim_relevant: bool = True
    missingness_rate: float = 0.0
    missingness_by_group: Dict[str, float] = field(default_factory=dict)
    affected_observations: int = 0
    total_observations: int = 0

    observed_estimate: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    sensitivity_width: Optional[float] = None
    relative_sensitivity: Optional[float] = None

    assumption_type: str = ""
    assumption_parameters: Dict[str, Any] = field(default_factory=dict)
    reference_range: Optional[Tuple[float, float]] = None

    decision_boundary_status: str = "NOT_EVALUATED"  # CROSSED / NOT_CROSSED / NOT_EVALUATED
    classification: str = INSUFFICIENT_EVIDENCE
    rationale: str = ""

    # Present only for ranking/competing-hypothesis analyses (Section 24).
    ranking_stable: Optional[bool] = None
    leading_group_observed: Optional[str] = None
    leading_group_worst_case: Optional[str] = None

    def to_provenance_dict(self) -> Dict[str, Any]:
        return {
            "target_column": self.target_column,
            "grouping_column": self.grouping_column,
            "metric_name": self.metric_name,
            "aggregation_type": self.aggregation_type,
            "semantic_resolution_status": self.semantic_resolution_status,
            "metric_definition_version": self.metric_definition_version,
            "claim_relevant": self.claim_relevant,
            "missingness_rate": self.missingness_rate,
            "missingness_by_group": self.missingness_by_group,
            "affected_observations": self.affected_observations,
            "total_observations": self.total_observations,
            "observed_estimate": self.observed_estimate,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "sensitivity_width": self.sensitivity_width,
            "relative_sensitivity": self.relative_sensitivity,
            "assumption_type": self.assumption_type,
            "assumption_parameters": self.assumption_parameters,
            "reference_range": list(self.reference_range) if self.reference_range else None,
            "decision_boundary_status": self.decision_boundary_status,
            "epistemic_classification": self.classification,
            "rationale": self.rationale,
            "ranking_stable": self.ranking_stable,
            "leading_group_observed": self.leading_group_observed,
            "leading_group_worst_case": self.leading_group_worst_case,
        }

    @property
    def is_high_confidence_eligible(self) -> bool:
        """True only when the sensitivity gate permits a high-confidence
        (DIAGNOSED-tier) verdict for the affected conclusion (Section 21)."""
        return self.classification == ROBUST


def _insufficient(
    target_column: str,
    grouping_column: Optional[str],
    metric_def: Optional[MetricDefinition],
    reason: str,
    missingness_rate: float = 0.0,
    total_observations: int = 0,
    affected_observations: int = 0,
) -> SensitivityResult:
    return SensitivityResult(
        target_column=target_column,
        grouping_column=grouping_column,
        metric_name=metric_def.name if metric_def else target_column,
        aggregation_type=metric_def.aggregation_type.value if metric_def else "unknown",
        semantic_resolution_status=metric_def.semantic_resolution_status if metric_def else "LEGACY_FALLBACK",
        missingness_rate=missingness_rate,
        total_observations=total_observations,
        affected_observations=affected_observations,
        assumption_type="none",
        classification=INSUFFICIENT_EVIDENCE,
        decision_boundary_status="NOT_EVALUATED",
        rationale=reason,
    )


class MissingnessSensitivityEngine:
    """Deterministic, MetricDefinition-aware missingness sensitivity analysis.

    No AI, no network calls, no fabricated numbers. Every bound is derived
    from observed data plus one explicitly recorded assumption (see module
    docstring). Mirrors the design discipline of
    ``MetricSemanticsResolver`` (Phase 11): schema-agnostic, no hardcoded
    column or group names anywhere in this class.
    """

    # -- Section 9: missingness characterization -----------------------
    @staticmethod
    def characterize_missingness(
        df: pd.DataFrame,
        target_col: str,
        group_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Overall + per-group missingness rate for target_col. Makes no
        claim about MCAR/MAR/MNAR -- purely descriptive (Section 9)."""
        if target_col not in df.columns or len(df) == 0:
            return {"overall_rate": None, "by_group": {}, "total": 0, "missing": 0}
        total = int(len(df))
        missing = int(df[target_col].isna().sum())
        overall_rate = missing / total if total else None
        by_group: Dict[str, float] = {}
        if group_col and group_col in df.columns:
            for g, sub in df.groupby(group_col, dropna=False):
                g_total = len(sub)
                if g_total == 0:
                    continue
                by_group[str(g)] = float(sub[target_col].isna().sum()) / g_total
        return {"overall_rate": overall_rate, "by_group": by_group, "total": total, "missing": missing}

    # -- Internal: SUM-style [observed_min, observed_max] reallocation bound --
    @staticmethod
    def _sum_bounds(series: pd.Series) -> Optional[Tuple[float, float, float, float, float]]:
        """Returns (observed_sum, lower, upper, obs_min, obs_max) or None if
        there are too few non-null values to establish a defensible range."""
        nonnull = series.dropna()
        n_nonnull = len(nonnull)
        n_missing = int(series.isna().sum())
        if n_nonnull < _MIN_NONNULL_FOR_BOUND:
            return None
        obs_min, obs_max = float(nonnull.min()), float(nonnull.max())
        observed_sum = float(nonnull.sum())
        lower = observed_sum + n_missing * obs_min
        upper = observed_sum + n_missing * obs_max
        if not (_is_finite(lower) and _is_finite(upper)):
            return None
        return observed_sum, lower, upper, obs_min, obs_max

    @staticmethod
    def _mean_bounds(series: pd.Series) -> Optional[Tuple[float, float, float, float, float]]:
        nonnull = series.dropna()
        n_nonnull = len(nonnull)
        n_missing = int(series.isna().sum())
        if n_nonnull < _MIN_NONNULL_FOR_BOUND:
            return None
        obs_min, obs_max = float(nonnull.min()), float(nonnull.max())
        observed_mean = float(nonnull.mean())
        total_n = n_nonnull + n_missing
        observed_sum = float(nonnull.sum())
        lower = (observed_sum + n_missing * obs_min) / total_n
        upper = (observed_sum + n_missing * obs_max) / total_n
        if not (_is_finite(lower) and _is_finite(upper)):
            return None
        return observed_mean, lower, upper, obs_min, obs_max

    @staticmethod
    def _count_bounds(series: pd.Series) -> Tuple[float, float, float]:
        """Worst-case occurrence-count bound: lower = observed non-null
        count (current COUNT semantics); upper = every missing row assumed
        to have been a real, unrecorded occurrence."""
        n_nonnull = int(series.notna().sum())
        n_missing = int(series.isna().sum())
        return float(n_nonnull), float(n_nonnull), float(n_nonnull + n_missing)

    @staticmethod
    def _distinct_count_bounds(series: pd.Series) -> Tuple[float, float, float]:
        """Worst-case distinct-entity-count bound for a COUNT_DISTINCT metric
        (e.g. counting unique customer_id/product_id). BUGFIX (DEFECT-005):
        a COUNT_DISTINCT target column is very often a non-numeric identifier
        (e.g. 'CUST-0001'), so it must never be routed into the SUM/MEAN
        numeric-bounds path (_sum_bounds/_mean_bounds), which calls float()
        on the raw values and crashes on any string identifier. lower =
        observed distinct count (every missing row could in principle
        duplicate an already-observed entity); upper = observed distinct
        count + every missing row assumed to be a brand-new distinct entity.
        """
        n_distinct = int(series.dropna().nunique())
        n_missing = int(series.isna().sum())
        return float(n_distinct), float(n_distinct), float(n_distinct + n_missing)

    @classmethod
    def _rate_bounds(
        cls,
        df: pd.DataFrame,
        numerator_col: str,
        denominator_col: str,
    ) -> Optional[Dict[str, Any]]:
        """Interval-arithmetic bound for numerator/denominator ratios: bound
        numerator and denominator sums independently (SUM rule), then take
        the min/max of the rate over all four corner combinations."""
        num_bounds = cls._sum_bounds(df[numerator_col]) if numerator_col in df.columns else None
        den_bounds = cls._sum_bounds(df[denominator_col]) if denominator_col in df.columns else None
        if num_bounds is None or den_bounds is None:
            return None
        num_obs, num_lo, num_hi, _, _ = num_bounds
        den_obs, den_lo, den_hi, _, _ = den_bounds
        if den_obs == 0 or den_lo == 0 or den_hi == 0:
            # Any corner divides by zero -- cannot defend a bound.
            corners = []
            for n in (num_lo, num_hi):
                for d in (den_lo, den_hi):
                    if d != 0:
                        corners.append(n / d)
            if not corners:
                return None
        else:
            corners = [n / d for n in (num_lo, num_hi) for d in (den_lo, den_hi) if d != 0]
        if not corners or not all(_is_finite(c) for c in corners):
            return None
        observed_rate = num_obs / den_obs if den_obs != 0 else None
        if observed_rate is None or not _is_finite(observed_rate):
            return None
        return {
            "observed": observed_rate,
            "lower": min(corners),
            "upper": max(corners),
            "numerator_bounds": (num_lo, num_hi),
            "denominator_bounds": (den_lo, den_hi),
        }

    # -- Section 8: metric-aware bound dispatch --------------------------
    @classmethod
    def _bound_for_metric(
        cls,
        df: pd.DataFrame,
        metric_def: Optional[MetricDefinition],
        target_col: str,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], str, Dict[str, Any], Optional[Tuple[float, float]]]:
        """Returns (observed, lower, upper, assumption_type, assumption_params, reference_range)
        or (None, None, None, ...) if no defensible bound exists for this metric."""
        if target_col not in df.columns:
            return None, None, None, "none", {}, None

        agg = metric_def.aggregation_type if metric_def else None
        series = df[target_col]

        if agg == AggregationType.COUNT:
            observed, lower, upper = cls._count_bounds(series)
            return (
                observed, lower, upper,
                "worst_case_missing_as_occurrence",
                {"note": "lower=observed non-null count; upper=every missing row assumed a real occurrence"},
                None,
            )

        if agg == AggregationType.COUNT_DISTINCT:
            # BUGFIX (DEFECT-005): previously absent -- fell through to the
            # SUM branch below and crashed calling float() on non-numeric
            # identifier values (e.g. customer_id).
            observed, lower, upper = cls._distinct_count_bounds(series)
            return (
                observed, lower, upper,
                "worst_case_missing_as_new_distinct_entity",
                {"note": "lower=observed distinct count; upper=every missing row assumed a new distinct entity"},
                None,
            )

        if agg in _RATE_LIKE and metric_def and metric_def.numerator_column and metric_def.denominator_column:
            rb = cls._rate_bounds(df, metric_def.numerator_column, metric_def.denominator_column)
            if rb is None:
                return None, None, None, "none", {}, None
            return (
                rb["observed"], rb["lower"], rb["upper"],
                "interval_arithmetic_numerator_denominator_reallocation",
                {
                    "numerator_column": metric_def.numerator_column,
                    "denominator_column": metric_def.denominator_column,
                    "numerator_bounds": rb["numerator_bounds"],
                    "denominator_bounds": rb["denominator_bounds"],
                },
                None,
            )

        if agg == AggregationType.MEAN or agg == AggregationType.WEIGHTED_MEAN or (
            agg in _RATE_LIKE and not (metric_def and metric_def.numerator_column and metric_def.denominator_column)
        ):
            mb = cls._mean_bounds(series)
            if mb is None:
                return None, None, None, "none", {}, None
            observed, lower, upper, obs_min, obs_max = mb
            note = "" if agg == AggregationType.MEAN or agg == AggregationType.WEIGHTED_MEAN else (
                " (rate/ratio degraded to MEAN bound: no raw numerator/denominator pair, "
                "mirroring MetricSemanticsResolver's own honest degradation policy)"
            )
            return (
                observed, lower, upper,
                "observed_range_reallocation_mean",
                {"reallocation_note": "each missing value assumed within observed [min, max]" + note},
                (obs_min, obs_max),
            )

        # SUM (RESOLVED, UNRESOLVED_DEFAULT_SUM, LEGACY_FALLBACK, or event_volume_sum)
        sb = cls._sum_bounds(series)
        if sb is None:
            return None, None, None, "none", {}, None
        observed, lower, upper, obs_min, obs_max = sb
        return (
            observed, lower, upper,
            "observed_range_reallocation_sum",
            {"reallocation_note": "each missing value assumed within observed [min, max] of non-null values"},
            (obs_min, obs_max),
        )

    # -- Main entry point -------------------------------------------------
    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        metric_definition: Optional[MetricDefinition],
        target_col: str,
        group_col: Optional[str] = None,
        decision_boundary: Optional[float] = None,
        claim_relevant_columns: Optional[List[str]] = None,
    ) -> SensitivityResult:
        """Single-estimate sensitivity analysis (Sections 5-10).

        decision_boundary: an optional fixed value (e.g. 0.5 for "majority
        share") the caller's conclusion depends on. If the [lower, upper]
        interval straddles it, the boundary is CROSSED. If omitted, a fixed
        relative-width materiality default is used instead (documented at
        module level, not tuned per call).
        """
        if target_col not in df.columns:
            return _insufficient(target_col, group_col, metric_definition, f"Column '{target_col}' not present in dataset.")

        char = cls.characterize_missingness(df, target_col, group_col)
        missing_rate = char["overall_rate"] or 0.0
        total_obs = char["total"]
        affected = char["missing"]

        # Claim relevance (Section 10): only the target column and, if
        # given, the columns the caller names are considered relevant here.
        # Missingness in an unrelated column never reaches this analysis
        # because the caller would not pass it as target_col/group_col.
        relevant_cols = set(claim_relevant_columns or []) | {target_col}
        if group_col:
            relevant_cols.add(group_col)
        claim_relevant = target_col in relevant_cols

        if not claim_relevant:
            return SensitivityResult(
                target_column=target_col, grouping_column=group_col,
                metric_name=metric_definition.name if metric_definition else target_col,
                aggregation_type=metric_definition.aggregation_type.value if metric_definition else "unknown",
                semantic_resolution_status=metric_definition.semantic_resolution_status if metric_definition else "LEGACY_FALLBACK",
                claim_relevant=False, missingness_rate=missing_rate,
                missingness_by_group=char["by_group"], affected_observations=affected,
                total_observations=total_obs, assumption_type="none",
                classification=ROBUST, decision_boundary_status="NOT_EVALUATED",
                rationale="Missingness is present but not in a column relevant to the claim being evaluated; no downgrade applied.",
            )

        if metric_definition is not None and metric_definition.semantic_resolution_status == "LEGACY_FALLBACK":
            return _insufficient(
                target_col, group_col, metric_definition,
                "MetricDefinition is a LEGACY_FALLBACK stand-in (no real semantic resolution occurred); "
                "sensitivity cannot safely assume a metric it was never confidently given.",
                missing_rate, total_obs, affected,
            )

        if affected == 0:
            # BUGFIX (DEFECT-005): this used to call cls._sum_bounds(df[target_col])
            # unconditionally regardless of the metric's actual aggregation
            # type -- crashing with "could not convert string to float" for
            # any COUNT_DISTINCT/COUNT metric over a non-numeric identifier
            # column (e.g. customer_id), even though there was nothing wrong
            # with the data (zero missing values). Route through the same
            # metric-aware dispatcher used in the affected>0 path below so
            # the "observed" estimate always matches how the metric is
            # actually aggregated (SUM/MEAN/COUNT/COUNT_DISTINCT/RATE...).
            observed, _, _, _, _, _ = cls._bound_for_metric(df, metric_definition, target_col)
            if observed is None:
                observed = float(len(df[target_col].dropna()))
            return SensitivityResult(
                target_column=target_col, grouping_column=group_col,
                metric_name=metric_definition.name if metric_definition else target_col,
                aggregation_type=metric_definition.aggregation_type.value if metric_definition else "unknown",
                semantic_resolution_status=metric_definition.semantic_resolution_status if metric_definition else "RESOLVED",
                claim_relevant=True, missingness_rate=0.0, missingness_by_group=char["by_group"],
                affected_observations=0, total_observations=total_obs,
                observed_estimate=observed, lower_bound=observed, upper_bound=observed,
                sensitivity_width=0.0, relative_sensitivity=0.0,
                assumption_type="none_needed", classification=ROBUST,
                decision_boundary_status="NOT_CROSSED",
                rationale="No missing values in the claim-relevant column; sensitivity analysis is trivially ROBUST.",
            )

        observed, lower, upper, assumption_type, assumption_params, ref_range = cls._bound_for_metric(
            df, metric_definition, target_col
        )
        if observed is None or lower is None or upper is None:
            return _insufficient(
                target_col, group_col, metric_definition,
                "Fewer than 2 non-null observations (or an unresolvable metric type) -- no defensible "
                "sensitivity bound can be established from observed data.",
                missing_rate, total_obs, affected,
            )

        lo, hi = min(lower, upper), max(lower, upper)
        width = hi - lo
        rel = (width / abs(observed)) if abs(observed) > 1e-9 else (0.0 if width == 0 else float("inf"))

        if decision_boundary is not None:
            crossed = lo <= decision_boundary <= hi
        else:
            crossed = rel > _DEFAULT_MATERIALITY_RELATIVE_WIDTH

        if not crossed:
            classification = ROBUST
            rationale = (
                f"Observed estimate {observed:.4g} with sensitivity interval [{lo:.4g}, {hi:.4g}] under the "
                f"stated assumption ({assumption_type}); the relevant decision boundary is not crossed, so the "
                f"conclusion is materially unchanged across the plausible range."
            )
        elif missing_rate >= _UNIDENTIFIABLE_MISSING_RATE:
            classification = UNIDENTIFIABLE
            rationale = (
                f"Sensitivity interval [{lo:.4g}, {hi:.4g}] crosses the relevant decision boundary AND "
                f"{missing_rate:.0%} of the claim-relevant data is missing -- the available observations plus "
                f"the stated bounding assumption do not permit a defensible determination."
            )
        else:
            classification = SENSITIVE
            rationale = (
                f"Sensitivity interval [{lo:.4g}, {hi:.4g}] crosses the relevant decision boundary under the "
                f"stated assumption ({assumption_type}); the conclusion is not invariant to plausible missing-value "
                f"assignments and must not be presented as such."
            )

        return SensitivityResult(
            target_column=target_col, grouping_column=group_col,
            metric_name=metric_definition.name if metric_definition else target_col,
            aggregation_type=metric_definition.aggregation_type.value if metric_definition else "unknown",
            semantic_resolution_status=metric_definition.semantic_resolution_status if metric_definition else "RESOLVED",
            claim_relevant=True, missingness_rate=missing_rate, missingness_by_group=char["by_group"],
            affected_observations=affected, total_observations=total_obs,
            observed_estimate=observed, lower_bound=lo, upper_bound=hi,
            sensitivity_width=width, relative_sensitivity=(None if math.isinf(rel) else rel),
            assumption_type=assumption_type, assumption_parameters=assumption_params,
            reference_range=ref_range, decision_boundary_status=("CROSSED" if crossed else "NOT_CROSSED"),
            classification=classification, rationale=rationale,
        )

    # -- Section 24: competing-hypothesis / ranking-stability analysis ----
    @classmethod
    def analyze_group_ranking(
        cls,
        df: pd.DataFrame,
        metric_definition: Optional[MetricDefinition],
        target_col: str,
        group_col: str,
        group_a: Any,
        group_b: Any,
    ) -> SensitivityResult:
        """Determines whether the ranking between two groups (which one has
        the higher/leading metric value) is stable across the plausible
        missingness range for EACH group independently -- the standard shape
        of a leading-vs-counter-hypothesis comparison (Section 24)."""
        if target_col not in df.columns or group_col not in df.columns:
            return _insufficient(target_col, group_col, metric_definition, "Target or grouping column not present in dataset.")

        sub_a = df[df[group_col] == group_a]
        sub_b = df[df[group_col] == group_b]
        if len(sub_a) == 0 or len(sub_b) == 0:
            return _insufficient(target_col, group_col, metric_definition, f"One of the compared groups ({group_a!r}, {group_b!r}) has no rows.")

        if metric_definition is not None and metric_definition.semantic_resolution_status == "LEGACY_FALLBACK":
            return _insufficient(target_col, group_col, metric_definition, "MetricDefinition is a LEGACY_FALLBACK stand-in; cannot safely bound a ranking comparison.")

        obs_a, lo_a, hi_a, assumption_type, params_a, _ = cls._bound_for_metric(sub_a, metric_definition, target_col)
        obs_b, lo_b, hi_b, _, params_b, _ = cls._bound_for_metric(sub_b, metric_definition, target_col)

        char = cls.characterize_missingness(df, target_col, group_col)
        missing_rate = char["overall_rate"] or 0.0

        if obs_a is None or obs_b is None:
            return _insufficient(
                target_col, group_col, metric_definition,
                "Fewer than 2 non-null observations in at least one compared group -- ranking stability "
                "cannot be defensibly bounded.",
                missing_rate, char["total"], char["missing"],
            )

        observed_leader = group_a if obs_a >= obs_b else group_b
        # Worst case for the observed leader: its own lower bound vs. the
        # other group's upper bound.
        if observed_leader == group_a:
            worst_case_leader_still_leads = lo_a >= hi_b
        else:
            worst_case_leader_still_leads = lo_b >= hi_a
        worst_case_leader = observed_leader if worst_case_leader_still_leads else (group_b if observed_leader == group_a else group_a)

        ranking_stable = bool(worst_case_leader_still_leads)
        rel_rate_a = char["by_group"].get(str(group_a), 0.0)
        rel_rate_b = char["by_group"].get(str(group_b), 0.0)
        max_relevant_rate = max(rel_rate_a, rel_rate_b)

        if ranking_stable:
            classification = ROBUST
            rationale = (
                f"{group_a!r} vs {group_b!r} on '{target_col}': observed leader {observed_leader!r} remains "
                f"ahead even in the worst case ([{lo_a:.4g},{hi_a:.4g}] vs [{lo_b:.4g},{hi_b:.4g}]) -- ranking "
                f"is robust to plausible missing-value assignment."
            )
        elif max_relevant_rate >= _UNIDENTIFIABLE_MISSING_RATE:
            classification = UNIDENTIFIABLE
            rationale = (
                f"{group_a!r} vs {group_b!r} on '{target_col}': the worst-case intervals overlap/flip AND "
                f"missingness in the deciding group reaches {max_relevant_rate:.0%} -- the ranking between "
                f"these hypotheses cannot be defensibly determined from observed data plus the stated assumption."
            )
        else:
            classification = SENSITIVE
            rationale = (
                f"{group_a!r} vs {group_b!r} on '{target_col}': observed leader is {observed_leader!r}, but "
                f"plausible missing-value assignment ([{lo_a:.4g},{hi_a:.4g}] vs [{lo_b:.4g},{hi_b:.4g}]) can "
                f"reverse the ranking -- this instability must be represented, not hidden behind the "
                f"observed-data winner."
            )

        return SensitivityResult(
            target_column=target_col, grouping_column=group_col,
            metric_name=metric_definition.name if metric_definition else target_col,
            aggregation_type=metric_definition.aggregation_type.value if metric_definition else "unknown",
            semantic_resolution_status=metric_definition.semantic_resolution_status if metric_definition else "RESOLVED",
            claim_relevant=True, missingness_rate=missing_rate, missingness_by_group=char["by_group"],
            affected_observations=char["missing"], total_observations=char["total"],
            observed_estimate=obs_a - obs_b, lower_bound=min(lo_a - hi_b, lo_b - hi_a),
            upper_bound=max(hi_a - lo_b, hi_b - lo_a),
            sensitivity_width=abs((hi_a - lo_b) - (lo_a - hi_b)), relative_sensitivity=None,
            assumption_type=assumption_type, assumption_parameters={"group_a": params_a, "group_b": params_b},
            decision_boundary_status=("NOT_CROSSED" if ranking_stable else "CROSSED"),
            classification=classification, rationale=rationale,
            ranking_stable=ranking_stable,
            leading_group_observed=str(observed_leader), leading_group_worst_case=str(worst_case_leader),
        )
