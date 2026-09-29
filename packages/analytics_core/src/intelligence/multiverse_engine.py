"""MultiverseEngine: Specification Curve Analysis evaluating analytical robustness across defensible specifications."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class SpecificationOutcome:
    """Individual analytical specification outcome."""
    specification_name: str
    trimming: str        # NONE, 1PCT_TRIM, 5PCT_TRIM
    aggregation: str     # SUM, MEAN, MEDIAN
    measured_effect: float
    is_concordant: bool


@dataclass
class MultiverseRobustnessReport:
    """Multiverse specification curve sensitivity outcome."""
    total_specifications: int
    concordant_specifications: int
    # BUGFIX: when the multiverse analysis is not applicable (no dimension/
    # target column, non-numeric target, fewer than two groups), this must
    # be None rather than a numeric value. Previously these cases still
    # populated robustness_pct=100.0, which meant "we couldn't run the
    # analysis" was indistinguishable downstream from "the finding is 100%
    # robust" -- a false-confidence bug. `applicable=False` is the
    # authoritative signal; robustness_pct/robustness_score are None
    # whenever applicable is False.
    robustness_pct: Optional[float]
    specification_curve: List[Dict[str, Any]]
    epistemic_summary: str
    applicable: bool = True

    @property
    def robustness_score(self) -> Optional[float]:
        if not self.applicable or self.robustness_pct is None:
            return None
        return float(self.robustness_pct / 100.0)


class MultiverseEngine:
    """Evaluates analytical stability across a combinatorial grid of defensible specifications."""

    @staticmethod
    def evaluate_specification_curve(
        df: pd.DataFrame,
        dimension_col: str,
        target_metric_col: str,
    ) -> MultiverseRobustnessReport:
        if dimension_col not in df.columns or target_metric_col not in df.columns or df.empty:
            return MultiverseRobustnessReport(
                total_specifications=0,
                concordant_specifications=0,
                robustness_pct=None,
                specification_curve=[],
                epistemic_summary=(
                    "Multiverse analysis: not applicable (missing dimension/target column "
                    "or empty dataset); no specification curve was evaluated."
                ),
                applicable=False,
            )

        # BUGFIX (DEFECT-005): the specification grid below is built entirely
        # around SUM/MEAN/MEDIAN of a numeric target metric (trimming by
        # quantile, .std() of grouped aggregates). For a COUNT_DISTINCT-style
        # metric over a non-numeric identifier column (e.g. customer_id),
        # every one of those pandas calls raises TypeError and crashes the
        # investigation. A SUM/MEAN/MEDIAN specification curve is not a
        # meaningful robustness check for a count of distinct entities in
        # the first place, so treat it the same as the "insufficient data"
        # case above rather than assuming applicability.
        if not pd.api.types.is_numeric_dtype(df[target_metric_col]):
            return MultiverseRobustnessReport(
                total_specifications=0,
                concordant_specifications=0,
                robustness_pct=None,
                specification_curve=[],
                epistemic_summary=(
                    "Multiverse analysis: SUM/MEAN/MEDIAN specification curve is not applicable to "
                    f"non-numeric target column '{target_metric_col}'; no specification curve was evaluated."
                ),
                applicable=False,
            )

        specifications: List[SpecificationOutcome] = []
        series = df[target_metric_col].dropna()

        # BUGFIX (Multiverse robustness): the previous implementation used
        # std() of grouped aggregates as "the effect" and called a spec
        # concordant whenever that std was positive in both baseline and
        # variant. A standard deviation across 2+ groups is virtually
        # always positive (it's zero only if every group is exactly tied),
        # so that check was close to a tautology -- it could never actually
        # detect a fragile finding. The substantive question a specification
        # curve is supposed to answer is: does the specific group driving
        # the finding, and the direction it deviates in, stay the same
        # across defensible re-specifications of the analysis? We compute
        # that directly: for each specification, find the group whose value
        # deviates most from the mean of the other groups, and check that
        # both the identified group and the sign of its deviation match
        # the baseline.
        def _extreme_group_effect(values: pd.Series):
            """Return (group_label, signed_deviation_from_rest_mean) for the
            group whose aggregate deviates most from the mean of the other
            groups, or (None, 0.0) if fewer than 2 groups are present."""
            if values.shape[0] < 2:
                return None, 0.0
            best_group, best_dev = None, 0.0
            for g in values.index:
                rest_mean = values.drop(g).mean()
                dev = values[g] - rest_mean
                if abs(dev) > abs(best_dev) or best_group is None:
                    best_group, best_dev = g, dev
            return best_group, float(best_dev)

        # Baseline: Raw data, SUM aggregation
        baseline_values = df.groupby(dimension_col)[target_metric_col].sum()
        baseline_group, baseline_dev = _extreme_group_effect(baseline_values)

        if baseline_group is None:
            return MultiverseRobustnessReport(
                total_specifications=0,
                concordant_specifications=0,
                robustness_pct=None,
                specification_curve=[],
                epistemic_summary=(
                    f"Multiverse analysis: fewer than two groups present in '{dimension_col}'; "
                    "not applicable, no specification curve was evaluated."
                ),
                applicable=False,
            )

        base_dir_positive = baseline_dev > 0.0

        # Grid of defensible analytical specifications
        trimmings = [("NONE", 0.0), ("1PCT_TRIM", 0.01), ("5PCT_TRIM", 0.05)]
        aggregations = ["SUM", "MEAN", "MEDIAN"]

        for trim_name, trim_q in trimmings:
            if trim_q > 0.0:
                q_low = series.quantile(trim_q)
                q_high = series.quantile(1.0 - trim_q)
                sub_df = df[(df[target_metric_col] >= q_low) & (df[target_metric_col] <= q_high)]
            else:
                sub_df = df

            if sub_df.empty:
                continue

            for agg in aggregations:
                spec_name = f"Spec_{trim_name}_{agg}"
                if agg == "SUM":
                    grouped = sub_df.groupby(dimension_col)[target_metric_col].sum()
                elif agg == "MEAN":
                    grouped = sub_df.groupby(dimension_col)[target_metric_col].mean()
                else:
                    grouped = sub_df.groupby(dimension_col)[target_metric_col].median()

                spec_group, spec_dev = _extreme_group_effect(grouped)
                is_concordant = (
                    spec_group is not None
                    and spec_group == baseline_group
                    and (spec_dev > 0.0) == base_dir_positive
                )
                specifications.append(
                    SpecificationOutcome(
                        specification_name=spec_name,
                        trimming=trim_name,
                        aggregation=agg,
                        measured_effect=float(spec_dev) if not np.isnan(spec_dev) else 0.0,
                        is_concordant=is_concordant,
                    )
                )

        total_specs = max(1, len(specifications))
        concordant_cnt = sum(1 for s in specifications if s.is_concordant)
        robustness_pct = (concordant_cnt / total_specs) * 100.0

        summary = (
            f"Multiverse Robustness: Finding is robust across {concordant_cnt}/{total_specs} "
            f"({robustness_pct:.1f}%) of defensible analytical specifications."
        )

        return MultiverseRobustnessReport(
            total_specifications=total_specs,
            concordant_specifications=concordant_cnt,
            robustness_pct=float(robustness_pct),
            specification_curve=[s.__dict__ for s in specifications],
            epistemic_summary=summary,
        )
