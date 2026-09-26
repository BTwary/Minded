"""
data_quality_gate.py: First-Class Pre-Investigation Dataset Fitness and Data Quality Gate.

Evaluates datasets before analytical reasoning to ensure mathematical fitness.
If data quality is insufficient, the gate fails closed with an explicit INSUFFICIENT_EVIDENCE status.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import re
import numpy as np
import pandas as pd


@dataclass
class DataQualityFinding:
    """Contextual, claim-oriented data-quality finding.

    Unlike a raw warning, a finding records why an observed data-quality
    pattern matters, which analysis families it can affect, and what should
    happen next. It is deterministic and derived only from observed data.
    """
    finding_id: str
    severity: str
    issue: str
    scope: Dict[str, Any]
    evidence: Dict[str, Any]
    potential_bias: str
    affected_methods: List[str]
    recommended_action: str


@dataclass
class DataReadinessConstraint:
    """Explicit constraint produced by pre-analysis data quality gate."""
    check_id: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    finding: str
    evidence: Dict[str, Any]
    affected_methods: List[str]
    required_action: str
    can_proceed: bool


@dataclass
class DataQualityAssessment:
    """Structured assessment of dataset fitness for autonomous investigation."""
    dataset_name: str
    row_count: int
    column_count: int
    overall_quality_score: float
    fitness_verdict: str  # "FIT", "CAUTION", "UNFIT"
    can_proceed: bool
    missingness_summary: Dict[str, float] = field(default_factory=dict)
    constant_columns: List[str] = field(default_factory=list)
    near_constant_columns: List[str] = field(default_factory=list)
    duplicate_rows_count: int = 0
    duplicate_rows_percentage: float = 0.0
    sparse_segments: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    duplicate_key_columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    outlier_columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    class_imbalance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    temporal_issues: List[str] = field(default_factory=list)
    selection_bias_indicators: List[str] = field(default_factory=list)
    leakage_indicators: List[str] = field(default_factory=list)
    unit_consistency_indicators: List[str] = field(default_factory=list)
    # Contextual, claim-oriented findings
    contextual_findings: List[DataQualityFinding] = field(default_factory=list)
    # Formal Constraint System
    constraints: List[DataReadinessConstraint] = field(default_factory=list)
    method_allowlist: List[str] = field(default_factory=list)
    method_blocklist: Dict[str, str] = field(default_factory=dict)

    def to_provenance_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "overall_quality_score": self.overall_quality_score,
            "fitness_verdict": self.fitness_verdict,
            "can_proceed": self.can_proceed,
            "contextual_findings": [
                {
                    "finding_id": f.finding_id,
                    "severity": f.severity,
                    "issue": f.issue,
                    "scope": f.scope,
                    "evidence": f.evidence,
                    "potential_bias": f.potential_bias,
                    "affected_methods": f.affected_methods,
                    "recommended_action": f.recommended_action,
                }
                for f in self.contextual_findings
            ],
            "constraints": [
                {
                    "check_id": c.check_id,
                    "severity": c.severity,
                    "finding": c.finding,
                    "evidence": c.evidence,
                    "affected_methods": c.affected_methods,
                    "required_action": c.required_action,
                }
                for c in self.constraints
            ],
            "method_allowlist": self.method_allowlist,
            "method_blocklist": self.method_blocklist,
        }


class DataQualityGate:
    """Evaluates dataset fitness and enforces fail-closed data quality barriers."""

    # Duplicated business-key rows (% of rows) at/above which the dataset is UNFIT.
    KEY_DUPLICATION_CRITICAL_PCT = 1.0

    ALL_METHODS: Set[str] = {
        "DESCRIPTIVE",
        "ASSOCIATION",
        "COMPARISON",
        "DIAGNOSTIC",
        "PREDICTION",
        "FORECASTING",
        "CAUSAL",
        "SEGMENTATION",
        "RECONCILIATION",
        "GOVERNANCE",
    }

    @staticmethod
    def evaluate_fitness(
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        min_sample_size: int = 5,
        time_col: Optional[str] = None,
        metric_col: Optional[str] = None,
    ) -> DataQualityAssessment:
        n_rows, n_cols = df.shape
        critical_issues: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []
        constraints: List[DataReadinessConstraint] = []
        method_blocklist: Dict[str, str] = {}

        # 1. Empty dataset check
        if n_rows == 0 or n_cols == 0:
            c = DataReadinessConstraint(
                check_id="EMPTY_DATASET",
                severity="CRITICAL",
                finding="Dataset is completely empty (0 rows or 0 columns).",
                evidence={"row_count": n_rows, "column_count": n_cols},
                affected_methods=list(DataQualityGate.ALL_METHODS),
                required_action="Provide a non-empty dataset before requesting analysis.",
                can_proceed=False,
            )
            return DataQualityAssessment(
                dataset_name=dataset_name,
                row_count=n_rows,
                column_count=n_cols,
                overall_quality_score=0.0,
                fitness_verdict="UNFIT",
                can_proceed=False,
                critical_issues=[c.finding],
                constraints=[c],
                method_allowlist=[],
                method_blocklist={m: "EMPTY_DATASET" for m in DataQualityGate.ALL_METHODS},
            )

        # 2. Minimum Sample Size for Inferential Statistics
        if n_rows < min_sample_size:
            msg = (
                f"Sample size ({n_rows} rows) is below the minimum mathematical threshold ({min_sample_size} rows) for statistical inference."
            )
            critical_issues.append(msg)
            c = DataReadinessConstraint(
                check_id="INSUFFICIENT_SAMPLE_SIZE",
                severity="CRITICAL",
                finding=msg,
                evidence={"row_count": n_rows, "min_sample_size": min_sample_size},
                affected_methods=["ASSOCIATION", "COMPARISON", "DIAGNOSTIC", "PREDICTION", "FORECASTING", "CAUSAL", "SEGMENTATION"],
                required_action="Acquire additional observations or restrict questions to descriptive summaries.",
                can_proceed=False,
            )
            constraints.append(c)
            for m in c.affected_methods:
                method_blocklist[m] = "INSUFFICIENT_SAMPLE_SIZE"

        # 3. Duplicate Rows Check
        dup_count = int(df.duplicated().sum())
        dup_pct = (dup_count / n_rows) * 100.0 if n_rows > 0 else 0.0
        if dup_pct > 25.0:
            warnings.append(f"High duplicate record rate detected: {dup_count:,} duplicate rows ({dup_pct:.1f}%).")
            recommendations.append("Consider deduplicating records on primary business entity keys.")

        # 3b. Business-Key (Primary Identifier) Uniqueness Check.
        # Distinct from the exact-row duplicate check above: this catches a
        # column that is NAMED like a unique business identifier (account_id,
        # customer_id, order_id, etc.) but whose values are not actually
        # unique -- whether or not the surrounding row data also happens to
        # match. Silently aggregating on a violated key double-counts the
        # affected entities and inflates any SUM/COUNT built on that grain.
        duplicate_key_columns: Dict[str, Dict[str, Any]] = {}
        id_name_pattern = ("_id", "_key", "_pk", "id", "key")
        for col in df.columns:
            col_lower = col.lower().strip()
            looks_like_identifier = (
                col_lower == "id"
                or col_lower.endswith(("_id", "_key", "_pk"))
                or col_lower in ("key", "pk", "account_id", "customer_id")
            )
            if not looks_like_identifier:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue
            n_unique = series.nunique()
            n_total = len(series)
            cardinality_ratio = n_unique / n_total
            # Only treat this as a genuine row-level identifier if MOST values
            # are already unique. A column named "category_id"/"region_id"
            # with only a handful of distinct values repeated many times is a
            # categorical dimension/foreign key, not a violated primary key --
            # without this gate, any low-cardinality "*_id" grouping column
            # would be wrongly flagged as a corrupted identifier.
            if cardinality_ratio < 0.8:
                continue
            # A genuine identifier column should be ~100% unique. Anything
            # short of that on a column named like a key is a uniqueness
            # violation, not a stylistic quirk.
            if n_unique < n_total:
                dup_value_count = n_total - n_unique
                dup_row_pct = (dup_value_count / n_total) * 100.0
                duplicate_key_columns[col] = {
                    "distinct_values": int(n_unique),
                    "total_rows": int(n_total),
                    "duplicated_row_count": int(dup_value_count),
                    "duplicated_row_pct": round(dup_row_pct, 2),
                }
                key_msg = (
                    f"Column '{col}' looks like a unique business identifier but contains "
                    f"{dup_value_count:,} duplicated key value(s) across {n_total:,} rows "
                    f"({dup_row_pct:.1f}%). Aggregating on this grain without resolving the "
                    f"duplication risk will double-count the affected entities."
                )
                # Materiality: duplicated rows can inflate a SUM/COUNT by at most the
                # duplicated-row percentage. Below the threshold the dataset stays usable
                # (CAUTION, with the bound stated); at/above it, fail closed as before.
                if dup_row_pct >= DataQualityGate.KEY_DUPLICATION_CRITICAL_PCT:
                    critical_issues.append(key_msg)
                else:
                    warnings.append(
                        key_msg + f" Immaterial below {DataQualityGate.KEY_DUPLICATION_CRITICAL_PCT:g}%: "
                        f"any SUM/COUNT over this grain is inflated by at most {dup_row_pct:.2f}%."
                    )
                recommendations.append(
                    f"Resolve whether duplicate '{col}' values represent legitimate repeated "
                    f"events or erroneous duplication before trusting any SUM/COUNT built on this grain."
                )

        # 3c. Temporal Magnitude Discontinuity Check (silent unit/scale change).
        # A metric that jumps by an extreme multiplicative factor between
        # adjacent time buckets is far more often a units artifact (dollars
        # vs cents, kg vs g) than a genuine business event. This does not
        # try to detect ordinary growth/decline -- only implausible
        # order-of-magnitude jumps -- and fails closed rather than let a
        # scale artifact be reported as a confident business finding.
        #
        # This is only a meaningful signal when each "period" median
        # actually aggregates multiple observations -- that's what makes an
        # extreme jump *between periods* implausible as anything but a
        # systemic scale change. At row-level granularity (e.g. one row per
        # day, common for daily transactional feeds), a "period" is a
        # single raw value, its "median" is just that one value, and a
        # >=20x jump against a neighboring single row is exactly what one
        # ordinary outlier or data-entry error looks like -- not evidence
        # of a dataset-wide unit/scale change. Treating it as the latter
        # hard-blocks the entire dataset (see the critical_issues gate
        # below) over a single anomalous row that univariate outlier
        # detection (elsewhere in this profiler) already flags on its own
        # merits, without conflating it with a structural scale defect.
        # Require at least a few observations per bucket before trusting
        # its median as representative of a real period.
        if time_col and metric_col and time_col in df.columns and metric_col in df.columns:
            try:
                tmp = df[[time_col, metric_col]].dropna()
                bucket_sizes = tmp.groupby(time_col)[metric_col].size()
                min_bucket_n = 3
                well_populated_buckets = bucket_sizes[bucket_sizes >= min_bucket_n].index
                # Bucket by the raw value of time_col (works for strings like
                # "2026-01" as well as actual dates truncated by the caller).
                period_medians = (
                    tmp[tmp[time_col].isin(well_populated_buckets)]
                    .groupby(time_col)[metric_col].median().sort_index()
                )
                abs_medians = period_medians.abs()
                nonzero = abs_medians[abs_medians > 1e-9]
                if len(nonzero) >= 2:
                    ordered = list(nonzero.items())
                    for (p1, m1), (p2, m2) in zip(ordered, ordered[1:]):
                        ratio = max(m1, m2) / min(m1, m2)
                        if ratio >= 20.0:
                            critical_issues.append(
                                f"Column '{metric_col}' median jumps {ratio:.1f}x between "
                                f"'{time_col}'={p1!r} and '{time_col}'={p2!r} (medians {m1:,.2f} vs {m2:,.2f}). "
                                f"This magnitude is far more consistent with a silent unit/scale change "
                                f"(e.g. dollars vs cents) than a genuine business event and must be "
                                f"confirmed before being reported as a finding."
                            )
            except Exception:
                pass

        # 4. Column Missingness & Constant Columns
        missingness: Dict[str, float] = {}
        constant_cols: List[str] = []
        near_constant_cols: List[str] = []

        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            null_pct = (null_count / n_rows) * 100.0
            missingness[col] = round(null_pct, 2)

            if null_pct == 100.0:
                # A column that is entirely null is only a *dataset-blocking*
                # problem when the current question actually needs it (it's
                # the resolved metric or time column). An unrelated, unused
                # column being empty (a deprecated/optional field that real
                # datasets commonly carry) previously hard-failed EVERY
                # question against the whole dataset via `critical_issues`,
                # even ones that never reference that column -- the same
                # over-broad failure mode this gate's other checks (constant
                # columns, outliers) already avoid by only warning. Scope the
                # hard failure to columns actually required for this
                # analysis; still surface it as a warning otherwise so it's
                # visible without blocking unrelated questions.
                if col in (metric_col, time_col):
                    critical_issues.append(
                        f"Column '{col}' is 100% missing (all nulls) and is required "
                        f"for this analysis (resolved as {'metric' if col == metric_col else 'time'} column)."
                    )
                else:
                    warnings.append(f"Column '{col}' is 100% missing (all nulls) and cannot be used in any analysis.")
            elif null_pct > 60.0:
                warnings.append(f"Column '{col}' has severe missingness ({null_pct:.1f}% null).")

            # Check distinct non-null values
            valid_vals = series.dropna()
            n_unique = valid_vals.nunique()

            if n_unique <= 1 and len(valid_vals) > 0:
                constant_cols.append(col)
            elif n_unique == 2 and len(valid_vals) > 50:
                top_freq = valid_vals.value_counts(normalize=True).iloc[0]
                if top_freq > 0.98:
                    near_constant_cols.append(col)

        if constant_cols:
            warnings.append(f"Zero-variance constant columns detected: {constant_cols}. These provide zero information gain.")

        # 4b. Numeric outlier scan. Outliers are evidence, not automatically bad data.
        outlier_columns: Dict[str, Dict[str, Any]] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) < 8:
                continue
            q1, q3 = vals.quantile([0.25, 0.75])
            iqr = float(q3 - q1)
            if iqr <= 0:
                continue
            lo, hi = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
            count = int(((vals < lo) | (vals > hi)).sum())
            if count:
                outlier_columns[str(col)] = {
                    "method": "IQR_1.5",
                    "outlier_count": count,
                    "outlier_pct": round(count / len(vals) * 100.0, 3),
                    "lower_bound": lo,
                    "upper_bound": hi,
                }

        # 4c. Binary/class-imbalance scan for low-cardinality columns.
        class_imbalance: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            vals = df[col].dropna()
            unique = vals.nunique()
            if unique != 2 or len(vals) < 20:
                continue
            counts = vals.value_counts(dropna=False)
            minority = int(counts.min())
            majority = int(counts.max())
            ratio = minority / majority if majority else 0.0
            if ratio < 0.20:
                class_imbalance[str(col)] = {
                    "classes": {str(k): int(v) for k, v in counts.items()},
                    "minority_majority_ratio": round(ratio, 6),
                    "minority_pct": round(minority / len(vals) * 100.0, 3),
                }
                warnings.append(f"Strong class imbalance in '{col}': minority class is {minority / len(vals) * 100.0:.2f}%.")

        # Record SEVERE_CLASS_IMBALANCE constraint for predictions if minority < 5%
        for bcol, detail in class_imbalance.items():
            min_pct = detail.get("minority_pct", 0.0)
            if min_pct < 5.0 and n_rows >= 20:
                msg = f"Severe class imbalance in '{bcol}': minority class is only {min_pct:.1f}%."
                constraints.append(DataReadinessConstraint(
                    check_id="SEVERE_CLASS_IMBALANCE",
                    severity="HIGH",
                    finding=msg,
                    evidence={"column": bcol, "minority_pct": min_pct},
                    affected_methods=["PREDICTION"],
                    required_action="Use stratified resampling, SMOTE, or precision-recall metrics instead of raw accuracy.",
                    can_proceed=True,
                ))

        # 4d. Temporal validity/leakage indicators.
        temporal_issues: List[str] = []
        if time_col and time_col in df.columns:
            parsed = pd.to_datetime(df[time_col], errors="coerce")
            bad_dates = int(parsed.isna().sum())
            if bad_dates:
                temporal_issues.append(f"{bad_dates:,} rows have unparseable values in time column '{time_col}'.")
            if parsed.notna().any():
                parsed_tz = getattr(parsed.dt, "tz", None)
                now = pd.Timestamp.utcnow().tz_convert(parsed_tz) if parsed_tz is not None else pd.Timestamp.utcnow().tz_localize(None)
                future = int((parsed.dropna() > now).sum())
                if future:
                    temporal_issues.append(f"{future:,} rows in '{time_col}' are future-dated relative to local execution time.")
                    warnings.append(temporal_issues[-1])

        # 4e. Selection/survivorship indicators. Name matching is only a candidate
        # generator. Where a target metric is available, require an observed
        # selection-dependent outcome difference before emitting a measured flag.
        selection_bias_indicators: List[str] = []
        col_names = {str(c).lower(): c for c in df.columns}
        selection_candidates = [
            c for lc, c in col_names.items()
            if re.search(r"(^|_)(active|current|eligible|surviv)(_|$)", lc)
        ]
        for c in selection_candidates:
            vals = df[c].dropna()
            if vals.nunique() != 2 or metric_col not in df.columns:
                continue
            metric = pd.to_numeric(df[metric_col], errors="coerce")
            groups = [v for v in vals.unique()]
            a = metric[df[c] == groups[0]].dropna()
            b = metric[df[c] == groups[1]].dropna()
            if len(a) >= 5 and len(b) >= 5 and np.isfinite(a.mean()) and np.isfinite(b.mean()):
                pooled = max(float(np.nanstd(metric)), 1e-12)
                standardized_gap = abs(float(a.mean()) - float(b.mean())) / pooled
                if standardized_gap >= 0.20:
                    selection_bias_indicators.append(
                        f"Measured selection-dependent metric difference for '{c}': standardized mean gap {standardized_gap:.3f} (n={len(a)}/{len(b)})."
                    )
        if not selection_bias_indicators and any("cancel" in c or "churn" in c for c in col_names) and any("current" in c or "active" in c for c in col_names):
            selection_bias_indicators.append(
                "Potential eligibility/survivorship mechanism detected from status fields; no measured target difference was established, so this remains a review warning."
            )

        # 4f. Leakage indicators. Distinguish:
        #   (1) Structural post-outcome / future temporal role: features that explicitly
        #       denote post-event or future state (e.g. post_*, future_*, cancellation_reason,
        #       refund_after, churn_date, cancel_date) are structurally unavailable at
        #       prediction time and represent hard admissibility barriers for prediction tasks,
        #       even when constant or zero-variance in the sample.
        #   (2) Measured near-deterministic relationship: candidate features whose empirical
        #       association with the target reaches near-deterministic predictability (>=0.995).
        leakage_indicators: List[str] = []
        target_col = None
        if metric_col and metric_col in df.columns:
            target_col = metric_col
        else:
            # Fallback target candidate scan
            for c in df.columns:
                cl = str(c).lower().strip()
                if cl in ("target", "outcome", "label", "churn", "churn_event", "converted", "is_churn"):
                    target_col = c
                    break

        structural_post_outcome_tokens = ("post_", "future_", "cancellation_reason", "refund_after", "churn_date", "cancel_date")
        measured_candidate_tokens = ("outcome", "churn_date", "cancel_date", "cancellation_reason", "refund_after", "post_", "future_", "target_", "label")

        for c in df.columns:
            if target_col and c == target_col:
                continue
            cl = str(c).lower().strip()
            # 1. Structural temporal role analysis:
            is_structural_post_outcome = (
                cl.startswith("post_")
                or "post_outcome" in cl
                or "_post_" in cl
                or cl.startswith("future_")
                or "_future_" in cl
                or cl in ("cancellation_reason", "refund_after", "churn_date", "cancel_date")
            )
            if is_structural_post_outcome:
                leakage_indicators.append(
                    f"Structural post-outcome temporal feature detected for '{c}'; post-event/future-horizon fields are inadmissible for prediction tasks."
                )

            # 2. Measured evidence against target:
            if target_col and target_col in df.columns and any(tok in cl for tok in measured_candidate_tokens):
                pair = pd.DataFrame({"x": df[target_col], "y": df[c]}).dropna()
                if len(pair) >= 10:
                    measured = None
                    if pd.api.types.is_numeric_dtype(pair["x"]) and pd.api.types.is_numeric_dtype(pair["y"]):
                        x = pair["x"].to_numpy(dtype=float)
                        y = pair["y"].to_numpy(dtype=float)
                        if np.std(x) > 0 and np.std(y) > 0:
                            measured = abs(float(np.corrcoef(x, y)[0, 1]))
                    else:
                        rates = pair.groupby("y")["x"].agg(lambda z: z.nunique() == 1)
                        measured = float(rates.mean()) if len(rates) else 0.0
                    if measured is not None and measured >= 0.995:
                        msg = f"Measured near-deterministic target relationship for '{c}' (score={measured:.3f}); temporal/post-outcome leakage review required."
                        if msg not in leakage_indicators:
                            leakage_indicators.append(msg)

        if leakage_indicators:
            warnings.extend(leakage_indicators)
            constraints.append(DataReadinessConstraint(
                check_id="TEMPORAL_LEAKAGE_RISK",
                severity="HIGH",
                finding="; ".join(leakage_indicators),
                evidence={"leakage_indicators": leakage_indicators},
                affected_methods=["PREDICTION"],
                required_action="Exclude post-outcome or target-leaking fields prior to training or prediction.",
                can_proceed=True,
            ))

        unit_consistency_indicators: List[str] = []
        for col in df.select_dtypes(include=[np.number]).columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 10 and vals.abs().median() > 0:
                q99 = float(vals.abs().quantile(0.99))
                med = float(vals.abs().median())
                if med > 0 and q99 / med >= 1_000:
                    unit_consistency_indicators.append(f"Column '{col}' spans >1,000x from median magnitude to p99; inspect units/scales.")

        # 4g. Contextual data-quality reasoning.
        # These findings deliberately connect a measured data-quality pattern
        # to the analyses it can distort. They do not infer an unobservable
        # missingness mechanism or declare a causal bias from metadata alone.
        contextual_findings: List[DataQualityFinding] = []

        # Group-concentrated missingness: overall missingness can look harmless
        # while being highly concentrated in one segment. Only emit when a
        # real grouping dimension is available and each compared group has
        # enough observations to make the comparison meaningful.
        if metric_col and metric_col in df.columns:
            for group_col in df.columns:
                if group_col == metric_col:
                    continue
                if not (pd.api.types.is_object_dtype(df[group_col])
                        or isinstance(df[group_col].dtype, pd.CategoricalDtype)
                        or pd.api.types.is_bool_dtype(df[group_col])):
                    continue
                grouped = df.groupby(group_col, dropna=False)[metric_col].apply(lambda s: float(s.isna().mean()))
                sizes = df.groupby(group_col, dropna=False).size()
                eligible = grouped.index[sizes >= 5]
                if len(eligible) < 2:
                    continue
                rates = grouped.loc[eligible]
                spread = float(rates.max() - rates.min())
                overall = float(df[metric_col].isna().mean())
                if spread >= 0.20 and overall >= 0.05:
                    high_group = rates.idxmax()
                    low_group = rates.idxmin()
                    contextual_findings.append(DataQualityFinding(
                        finding_id="MISSINGNESS_CONCENTRATED_BY_GROUP",
                        severity="HIGH",
                        issue=f"Missingness in '{metric_col}' is materially concentrated across '{group_col}' groups.",
                        scope={"metric_column": str(metric_col), "grouping_column": str(group_col)},
                        evidence={
                            "overall_missing_rate": round(overall, 6),
                            "highest_missing_group": str(high_group),
                            "highest_missing_rate": round(float(rates.max()), 6),
                            "lowest_missing_group": str(low_group),
                            "lowest_missing_rate": round(float(rates.min()), 6),
                            "rate_spread": round(spread, 6),
                            "eligible_group_count": int(len(eligible)),
                        },
                        potential_bias="Group comparisons or models using the affected metric may be distorted if missingness is related to group membership.",
                        affected_methods=["COMPARISON", "DIAGNOSTIC", "PREDICTION", "CAUSAL"],
                        recommended_action="Run claim-specific missingness sensitivity analysis and avoid treating group differences as robust until the result is stable under the documented bounds.",
                    ))
                    break

        # Outlier concentration by a categorical segment. An overall outlier
        # count alone is not enough to infer a problem; concentration in one
        # segment is the contextual signal worth surfacing.
        for metric_name, detail in outlier_columns.items():
            if detail.get("outlier_pct", 0.0) < 1.0:
                continue
            vals = pd.to_numeric(df[metric_name], errors="coerce")
            q1, q3 = vals.dropna().quantile([0.25, 0.75]) if vals.notna().sum() else (np.nan, np.nan)
            iqr = float(q3 - q1) if np.isfinite(q1) and np.isfinite(q3) else 0.0
            if iqr <= 0:
                continue
            lo, hi = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
            outlier_mask = (vals < lo) | (vals > hi)
            for group_col in df.columns:
                if group_col == metric_name:
                    continue
                if not (pd.api.types.is_object_dtype(df[group_col]) or isinstance(df[group_col].dtype, pd.CategoricalDtype)):
                    continue
                group_rates = outlier_mask.groupby(df[group_col], dropna=False).mean()
                sizes = df.groupby(group_col, dropna=False).size()
                eligible = group_rates.index[sizes >= 5]
                if len(eligible) < 2:
                    continue
                rates = group_rates.loc[eligible]
                if float(rates.max() - rates.min()) >= 0.10 and float(rates.max()) >= 0.15:
                    contextual_findings.append(DataQualityFinding(
                        finding_id="OUTLIERS_CONCENTRATED_BY_GROUP",
                        severity="MEDIUM",
                        issue=f"Outliers in '{metric_name}' are concentrated in at least one '{group_col}' segment.",
                        scope={"metric_column": str(metric_name), "grouping_column": str(group_col)},
                        evidence={"highest_group_outlier_rate": round(float(rates.max()), 6), "lowest_group_outlier_rate": round(float(rates.min()), 6), "rate_spread": round(float(rates.max()-rates.min()), 6)},
                        potential_bias="Means, regressions, and anomaly-driven conclusions may be disproportionately influenced by one segment.",
                        affected_methods=["COMPARISON", "ASSOCIATION", "DIAGNOSTIC", "PREDICTION"],
                        recommended_action="Inspect the affected segment and repeat material analyses with robust or sensitivity-based methods before treating the aggregate result as representative.",
                    ))
                    break

        # Selection indicators are already measured above. Convert them into
        # structured findings so downstream report/recommendation layers do not
        # have to parse prose.
        if selection_bias_indicators:
            contextual_findings.append(DataQualityFinding(
                finding_id="SELECTION_BIAS_INDICATOR",
                severity="HIGH",
                issue="Observed selection/survivorship indicators may affect representativeness.",
                scope={"indicators": list(selection_bias_indicators)},
                evidence={"indicator_count": len(selection_bias_indicators)},
                potential_bias="Observed differences may partly reflect who entered or remained in the dataset rather than the underlying population process.",
                affected_methods=["COMPARISON", "DIAGNOSTIC", "PREDICTION", "CAUSAL"],
                recommended_action="Define the target population explicitly and test whether the conclusion remains stable after accounting for the observed selection mechanism.",
            ))

        if temporal_issues:
            contextual_findings.append(DataQualityFinding(
                finding_id="TEMPORAL_DATA_QUALITY",
                severity="MEDIUM",
                issue="Temporal validity issues were detected in the analysis time field.",
                scope={"time_column": str(time_col) if time_col else None},
                evidence={"issues": list(temporal_issues)},
                potential_bias="Trend, forecasting, cohort, and pre/post comparisons may use observations outside the intended time horizon.",
                affected_methods=["COMPARISON", "DIAGNOSTIC", "FORECASTING", "CAUSAL"],
                recommended_action="Resolve or explicitly exclude invalid time records before interpreting temporal effects.",
            ))

        # 5. Composite Quality Scoring (0.0 to 100.0)
        total_cells = n_rows * n_cols
        total_nulls = sum(df[c].isnull().sum() for c in df.columns)
        null_penalty = (total_nulls / total_cells) * 40.0 if total_cells > 0 else 0.0
        dup_penalty = min(20.0, dup_pct * 0.5)
        constant_penalty = min(15.0, len(constant_cols) * 5.0)
        sample_penalty = 25.0 if n_rows < 15 else 0.0
        key_penalty = min(30.0, sum(v["duplicated_row_pct"] for v in duplicate_key_columns.values()) * 0.5)

        overall_score = max(0.0, min(100.0, 100.0 - (null_penalty + dup_penalty + constant_penalty + sample_penalty + key_penalty)))

        # 6. Fitness Verdict
        if critical_issues or overall_score < 30.0 or n_rows < min_sample_size:
            verdict = "UNFIT"
            can_proceed = False
        elif warnings or overall_score < 70.0:
            verdict = "CAUTION"
            can_proceed = True
        else:
            verdict = "FIT"
            can_proceed = True

        allowed = [m for m in DataQualityGate.ALL_METHODS if m not in method_blocklist]
        if not can_proceed:
            allowed = ["DESCRIPTIVE"] if n_rows > 0 else []

        return DataQualityAssessment(
            dataset_name=dataset_name,
            row_count=n_rows,
            column_count=n_cols,
            overall_quality_score=round(overall_score, 1),
            fitness_verdict=verdict,
            can_proceed=can_proceed,
            missingness_summary=missingness,
            constant_columns=constant_cols,
            near_constant_columns=near_constant_cols,
            duplicate_rows_count=dup_count,
            duplicate_rows_percentage=round(dup_pct, 2),
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
            duplicate_key_columns=duplicate_key_columns,
            outlier_columns=outlier_columns,
            class_imbalance=class_imbalance,
            temporal_issues=temporal_issues,
            selection_bias_indicators=selection_bias_indicators,
            leakage_indicators=leakage_indicators,
            unit_consistency_indicators=unit_consistency_indicators,
            contextual_findings=contextual_findings,
            constraints=constraints,
            method_allowlist=allowed,
            method_blocklist=method_blocklist,
        )
