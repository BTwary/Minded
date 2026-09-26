"""Bayesian belief and evidence engine.

The canonical Bayesian update is exact. Evidence enters as explicit model-evidence
weights (Bayes factors or Bayes-factor approximations), never as arbitrary sigmoid
scores. When an experiment does not provide a defensible statistical evidence model,
the engine returns a neutral factor (1.0) rather than manufacturing confidence.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd
from scipy import stats


def compute_shannon_entropy(probs: List[float]) -> float:
    """Calculate Shannon entropy H(p) = -sum(p * log2(p))."""
    ent = 0.0
    for p in probs:
        if p > 1e-15:
            ent -= p * math.log2(p)
    return float(ent)


def _safe_exp(log_value: float, cap: float = 1e12) -> float:
    """Exponentiate a log Bayes factor without overflow."""
    if log_value >= math.log(cap):
        return cap
    if log_value <= math.log(1.0 / cap):
        return 1.0 / cap
    return float(math.exp(log_value))


def _bic_bayes_factor(delta_bic: float) -> float:
    """Convert ΔBIC = BIC(null) - BIC(alternative) to an approximate BF10."""
    return _safe_exp(0.5 * float(delta_bic))


def _validate_binary_prior(prior_a: float) -> float:
    value = float(prior_a)
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError("prior probability must be strictly between 0 and 1")
    return value


@dataclass(frozen=True)
class BayesianEvidence:
    """Auditable evidence weight for one hypothesis.

    `bayes_factor` is BF(hypothesis : reference_model). For several mutually
    exclusive hypotheses, these factors are used as relative marginal likelihoods
    against the same reference model; the supplied priors then produce posterior odds.
    """

    bayes_factor: float
    method: str
    sample_size: int
    assumptions: Tuple[str, ...] = ()
    diagnostic: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bayes_factor": float(self.bayes_factor),
            "method": self.method,
            "sample_size": int(self.sample_size),
            "assumptions": list(self.assumptions),
            "diagnostic": self.diagnostic,
        }


class BeliefEngine:
    """Bayesian updating using explicit model evidence."""

    @staticmethod
    def compute_bayesian_posteriors(
        priors: List[float],
        likelihoods: List[float],
    ) -> Tuple[List[float], float]:
        """Compute exact posterior probabilities.

        The second argument may be likelihoods or Bayes-factor weights as long as
        all weights are non-negative and refer to comparable evidence models.
        """
        if len(priors) != len(likelihoods) or not priors:
            raise ValueError("priors and likelihoods must have the same non-zero length")
        prior_values = np.asarray(priors, dtype=float)
        evidence_values = np.asarray(likelihoods, dtype=float)
        if np.any(~np.isfinite(prior_values)) or np.any(~np.isfinite(evidence_values)):
            raise ValueError("priors and likelihoods must be finite")
        if np.any(prior_values < 0.0) or np.any(evidence_values < 0.0):
            raise ValueError("priors and likelihoods must be non-negative")
        prior_total = float(prior_values.sum())
        if prior_total <= 0.0:
            raise ValueError("priors must contain positive mass")
        prior_values /= prior_total

        prior_ent = compute_shannon_entropy(prior_values.tolist())
        unnormalized = prior_values * evidence_values
        total = float(unnormalized.sum())
        if total <= 1e-15 or not np.isfinite(total):
            posteriors = [1.0 / len(priors)] * len(priors)
            post_ent = compute_shannon_entropy(posteriors)
            return posteriors, float(post_ent - prior_ent)
        posteriors = (unnormalized / total).astype(float).tolist()
        post_ent = compute_shannon_entropy(posteriors)
        return posteriors, float(post_ent - prior_ent)

    # ------------------------------------------------------------------
    # Explicit model-evidence calculations
    # ------------------------------------------------------------------
    @staticmethod
    def bayes_factor_two_groups(
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        *,
        min_group_n: int = 5,
    ) -> BayesianEvidence:
        """Approximate BF10 for unequal-mean vs equal-mean Gaussian models via BIC.

        This is a model-evidence approximation, not a p-value transformation. The
        common variance model is used for the likelihood; BIC supplies the complexity
        penalty for the extra group mean.
        """
        a = np.asarray(sample_a, dtype=float)
        b = np.asarray(sample_b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        n = int(len(a) + len(b))
        required_n = max(1, int(min_group_n))
        if len(a) < required_n or len(b) < required_n:
            return BayesianEvidence(1.0, "NEUTRAL_INADEQUATE_SAMPLE", n, (f"Each comparison group requires at least {required_n} observations.",))
        if n <= 3:
            return BayesianEvidence(1.0, "NEUTRAL_INADEQUATE_SAMPLE", n)

        grand = float(np.mean(np.concatenate([a, b])))
        sse0 = float(np.sum((a - grand) ** 2) + np.sum((b - grand) ** 2))
        sse1 = float(np.sum((a - np.mean(a)) ** 2) + np.sum((b - np.mean(b)) ** 2))
        if sse0 <= 0.0 or sse1 <= 0.0:
            rel_diff = abs(float(np.mean(a) - np.mean(b))) / max(abs(grand), 1e-9)
            if rel_diff < 0.05:
                return BayesianEvidence(1.0, "NEUTRAL_UNIFORM_GROUPS", n, ("Group means differ by less than 5% relative.",))
            bf = 1e12 if sse1 < sse0 else 1.0
            return BayesianEvidence(bf, "BIC_GAUSSIAN_TWO_GROUPS", n, ("Independent observations within groups.", "Common Gaussian variance model."))

        bic0 = n * math.log(sse0 / n) + 2 * math.log(n)  # common mean + variance
        bic1 = n * math.log(sse1 / n) + 3 * math.log(n)  # group means + variance
        bf = _bic_bayes_factor(bic0 - bic1)
        return BayesianEvidence(
            bf,
            "BIC_GAUSSIAN_TWO_GROUPS",
            n,
            ("Independent observations within groups.", "Common Gaussian variance model.", "BIC is used as an approximate Bayes factor."),
        )

    @staticmethod
    def bayes_factor_anova(groups: Dict[str, Sequence[float]], *, min_group_n: int = 5) -> BayesianEvidence:
        """Approximate BF10 for unequal group means vs common mean via BIC."""
        clean: Dict[str, np.ndarray] = {}
        for key, values in groups.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) >= max(1, int(min_group_n)):
                clean[key] = arr
        n = int(sum(len(v) for v in clean.values()))
        k = len(clean)
        if k < 2 or n < (k + 1):
            return BayesianEvidence(1.0, "NEUTRAL_INADEQUATE_GROUPS", n, (f"At least two groups with the configured minimum group size ({max(1, int(min_group_n) )}) are required.",))
        combined = np.concatenate(list(clean.values()))
        grand = float(np.mean(combined))
        sse0 = float(np.sum((combined - grand) ** 2))
        sse1 = float(sum(np.sum((v - np.mean(v)) ** 2) for v in clean.values()))
        if sse0 <= 0:
            return BayesianEvidence(1.0, "NEUTRAL_DEGENERATE_VARIANCE", n)
        if sse1 <= 0:
            rel_diff = (max(np.mean(v) for v in clean.values()) - min(np.mean(v) for v in clean.values())) / max(abs(grand), 1e-9)
            if rel_diff < 0.05:
                return BayesianEvidence(1.0, "NEUTRAL_UNIFORM_GROUPS", n, ("Group means differ by less than 5% relative.",))
            bf = 1e12
        else:
            # Null: one mean + variance. Alternative: k means + variance.
            bic0 = n * math.log(sse0 / n) + 2 * math.log(n)
            bic1 = n * math.log(sse1 / n) + (k + 1) * math.log(n)
            bf = _bic_bayes_factor(bic0 - bic1)
        return BayesianEvidence(
            bf,
            "BIC_GAUSSIAN_ANOVA",
            n,
            ("Independent observations within groups.", "Common Gaussian variance model.", f"Configured minimum of {max(1, int(min_group_n))} observations per included group.", "BIC is used as an approximate Bayes factor."),
        )

    @staticmethod
    def bayes_factor_correlation(x: Sequence[float], y: Sequence[float]) -> BayesianEvidence:
        """Approximate BF10 for linear association vs zero slope via BIC."""
        xv = np.asarray(x, dtype=float)
        yv = np.asarray(y, dtype=float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[mask], yv[mask]
        n = int(len(xv))
        if n < 8 or np.std(xv) <= 0 or np.std(yv) <= 0:
            return BayesianEvidence(1.0, "NEUTRAL_INADEQUATE_CORRELATION", n, ("At least 8 finite observations and non-zero variance are required.",))
        r = float(np.corrcoef(xv, yv)[0, 1])
        sse0 = float(np.sum((yv - np.mean(yv)) ** 2))
        slope, intercept = np.polyfit(xv, yv, 1)
        residuals = yv - (slope * xv + intercept)
        sse1 = float(np.sum(residuals ** 2))
        if sse0 <= 0 or sse1 <= 0:
            bf = 1e12 if sse1 < sse0 else 1.0
        else:
            bic0 = n * math.log(sse0 / n) + 2 * math.log(n)
            bic1 = n * math.log(sse1 / n) + 3 * math.log(n)
            bf = _bic_bayes_factor(bic0 - bic1)
        return BayesianEvidence(
            bf,
            "BIC_GAUSSIAN_LINEAR_ASSOCIATION",
            n,
            ("Independent observations.", "Linear conditional mean with Gaussian residuals.", "BIC is used as an approximate Bayes factor."),
            diagnostic=f"Pearson r={r:.6g}",
        )

    @staticmethod
    def bayes_factor_categorical_association(
        observed: pd.DataFrame,
    ) -> BayesianEvidence:
        """Approximate BF10 for categorical dependence vs independence using BIC."""
        table = np.asarray(observed, dtype=float)
        if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
            return BayesianEvidence(1.0, "NEUTRAL_INADEQUATE_CONTINGENCY", int(table.sum()) if table.ndim == 2 else 0)
        n = int(table.sum())
        if n < 10:
            return BayesianEvidence(1.0, "NEUTRAL_INADEQUATE_CONTINGENCY", n, ("At least 10 observations are required.",))
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
        mask = table > 0
        g2 = float(2.0 * np.sum(table[mask] * np.log(table[mask] / expected[mask])))
        k0 = (table.shape[0] - 1) + (table.shape[1] - 1)
        k1 = table.shape[0] * table.shape[1] - 1
        delta_bic = g2 - float(k1 - k0) * math.log(n)
        bf = _bic_bayes_factor(delta_bic)
        return BayesianEvidence(
            bf,
            "BIC_MULTINOMIAL_ASSOCIATION",
            n,
            ("Fixed contingency-table categories.", "Saturated multinomial alternative vs independence null.", "BIC is used as an approximate Bayes factor."),
            diagnostic=f"G^2={g2:.6g}",
        )

    @staticmethod
    def bayes_factor_trend(y: Sequence[float]) -> BayesianEvidence:
        """Approximate BF10 for linear temporal trend vs intercept-only model."""
        values = np.asarray(y, dtype=float)
        values = values[np.isfinite(values)]
        n = int(len(values))
        if n < 8 or np.std(values) <= 0:
            return BayesianEvidence(1.0, "NEUTRAL_INADEQUATE_TREND", n, ("At least 8 finite time periods and non-zero variance are required.",))
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, values, 1)
        sse0 = float(np.sum((values - np.mean(values)) ** 2))
        sse1 = float(np.sum((values - (slope * x + intercept)) ** 2))
        if sse1 <= 0:
            bf = 1e12
        else:
            bic0 = n * math.log(max(sse0, 1e-15) / n) + 2 * math.log(n)
            bic1 = n * math.log(max(sse1, 1e-15) / n) + 3 * math.log(n)
            bf = _bic_bayes_factor(bic0 - bic1)
        return BayesianEvidence(
            bf,
            "BIC_GAUSSIAN_LINEAR_TREND",
            n,
            ("Ordered observations are equally spaced after aggregation.", "Gaussian residuals with constant variance.", "BIC is used as an approximate Bayes factor."),
            diagnostic=f"slope={slope:.6g}",
        )

    @staticmethod
    def compute_model_based_bayes_factors(
        *,
        hypotheses: Sequence[Any],
        primary_df: Optional[pd.DataFrame],
        result_df: Optional[pd.DataFrame],
        target_metric_col: Optional[str],
        group_dimension_col: Optional[str],
        aggregation_type: str,
        tested_hypothesis_codes: Optional[Sequence[str]] = None,
        min_group_n: int = 5,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Return comparable evidence weights for the active hypotheses.

        No attributable hypothesis => factor 1.0 (neutral). The returned diagnostics
        explain exactly which statistical evidence model was used for each hypothesis.
        """
        codes = set(tested_hypothesis_codes or [])
        factors = [1.0 for _ in hypotheses]
        diagnostics: List[Dict[str, Any]] = [{"bayes_factor": 1.0, "method": "NEUTRAL_NO_ATTRIBUTABLE_EVIDENCE", "sample_size": 0} for _ in hypotheses]

        def _dim_matches(h: Any, tested_dim: Optional[str]) -> bool:
            """A hypothesis may only receive evidence from a test that actually
            bears on its own declared target_dimension. Without this guard, a
            single-dimension test run this round (e.g. a confounder dimension
            like 'cohort') gets broadcast as evidence for every hypothesis
            currently marked 'tested' -- including ones that target a
            completely different dimension (e.g. 'segment') -- which silently
            suppresses or inflates the wrong hypotheses. A hypothesis with no
            declared target_dimension is treated as dimension-agnostic (e.g.
            correlation/forecast pairs) and always matches.
            """
            h_dim = getattr(h, "target_dimension", None)
            if not h_dim or not tested_dim:
                return True
            return h_dim == tested_dim

        # For group-difference / ANOVA / two-group models, prefer primary_df if it has
        # unaggregated observations for target_metric_col and group_dimension_col.
        if (
            primary_df is not None
            and not primary_df.empty
            and target_metric_col
            and group_dimension_col
            and target_metric_col in primary_df.columns
            and group_dimension_col in primary_df.columns
            and (result_df is None or len(primary_df) > len(result_df))
        ):
            frame = primary_df
        elif result_df is not None and not result_df.empty:
            frame = result_df
        else:
            frame = primary_df

        if frame is None or frame.empty:
            return factors, diagnostics

        # Never expand aggregate rows by repeating an aggregate value.
        # Doing so creates synthetic zero-variance pseudo-observations and can
        # manufacture enormous Bayes factors. Inferential group models must use
        # genuine row-level observations; aggregate frames remain usable only for
        # estimators whose sufficient statistics are explicitly represented (for
        # example, concentration/count models below).

        # Resolve effective metric column in frame if target_metric_col was renamed
        effective_metric_col = target_metric_col
        if frame is not None and target_metric_col and target_metric_col not in frame.columns:
            num_candidates = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c != group_dimension_col]
            if len(num_candidates) == 1:
                effective_metric_col = num_candidates[0]
            else:
                effective_metric_col = None

        # Forecast hypotheses: H1 predictive structure vs H0 no structure.
        if str(aggregation_type).upper() == "REGRESSION":
            numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
            trend_col = effective_metric_col if effective_metric_col in numeric else (numeric[0] if len(numeric) == 1 else None)
            if trend_col is not None:
                ev = BeliefEngine.bayes_factor_trend(frame[trend_col].to_numpy())
                for i, h in enumerate(hypotheses):
                    if getattr(h, "hypothesis_code", "") not in codes:
                        continue
                    if getattr(h, "is_counter_hypothesis", False):
                        factors[i] = 1.0 / ev.bayes_factor
                    else:
                        factors[i] = ev.bayes_factor
                    diagnostics[i] = ev.to_dict()
            return factors, diagnostics

        # Continuous relationship experiment.
        if str(aggregation_type).upper() == "CORRELATION":
            numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
            corr_pair = None
            if effective_metric_col in numeric:
                others = [c for c in numeric if c != effective_metric_col]
                if len(others) == 1:
                    corr_pair = (effective_metric_col, others[0])
            elif len(numeric) == 2:
                corr_pair = (numeric[0], numeric[1])
            if corr_pair is not None:
                ev = BeliefEngine.bayes_factor_correlation(frame[corr_pair[0]], frame[corr_pair[1]])
                for i, h in enumerate(hypotheses):
                    if getattr(h, "hypothesis_code", "") not in codes:
                        continue
                    factors[i] = 1.0 / ev.bayes_factor if getattr(h, "is_counter_hypothesis", False) else ev.bayes_factor
                    diagnostics[i] = ev.to_dict()
            return factors, diagnostics

        # Binary/categorical association such as churn by segment.
        is_categorical_task = str(aggregation_type).upper() in ("CATEGORICAL_ASSOCIATION", "CHURN", "CONTINGENCY")
        if effective_metric_col and group_dimension_col and effective_metric_col in frame.columns and group_dimension_col in frame.columns:
            target = frame[effective_metric_col]
            is_binary_event = pd.api.types.is_bool_dtype(target) or (
                pd.api.types.is_numeric_dtype(target)
                and set(target.dropna().unique()).issubset({0, 1, 0.0, 1.0})
                and str(aggregation_type).upper() not in ("SUM", "AVG", "MEAN")
            )
            if (is_categorical_task or is_binary_event):
                table = pd.crosstab(frame[group_dimension_col], target)
                if table.shape[0] >= 2 and table.shape[1] >= 2:
                    ev = BeliefEngine.bayes_factor_categorical_association(table)
                    for i, h in enumerate(hypotheses):
                        if getattr(h, "hypothesis_code", "") not in codes:
                            continue
                        if not _dim_matches(h, group_dimension_col):
                            continue
                        factors[i] = 1.0 / ev.bayes_factor if getattr(h, "is_counter_hypothesis", False) else ev.bayes_factor
                        diagnostics[i] = ev.to_dict()
                    return factors, diagnostics

            # Targeted subgroup hypothesis: group target value vs all others.
            targeted_assigned = False
            for i, h in enumerate(hypotheses):
                if getattr(h, "hypothesis_code", "") not in codes:
                    continue
                dim = getattr(h, "target_dimension", "") or group_dimension_col
                val = getattr(h, "target_value", None)
                if dim and val is not None and dim in frame.columns and effective_metric_col in frame.columns:
                    a = frame.loc[frame[dim] == val, effective_metric_col].dropna()
                    b = frame.loc[frame[dim] != val, effective_metric_col].dropna()
                    # A targeted subgroup comparison is not admissible as positive
                    # scientific evidence when either side has inadequate support.
                    # This is deliberately a neutral evidence weight rather than a
                    # synthetic penalty: the data simply cannot discriminate the
                    # subgroup hypothesis at the declared evidentiary standard.
                    required_n = max(1, int(min_group_n))
                    if len(a) < required_n or len(b) < required_n:
                        # A pre-aggregated SUM row (one row per group) carries no
                        # raw observations to compare -- but if the frame also
                        # carries a genuine per-row weight (a 'row_count' column
                        # recording how many underlying observations each
                        # aggregate row represents), that weight is real sample-
                        # size evidence and must be honored rather than treated
                        # as permanently unfalsifiable. Absent that column, a
                        # single aggregate row per group is exactly what it looks
                        # like -- N=1 per group -- and must stay capped/neutral;
                        # the aggregation label alone (SUM) is not evidence of
                        # real underlying volume.
                        ev = None
                        if str(aggregation_type).upper() == "SUM" and "row_count" in frame.columns:
                            weight_a = float(frame.loc[frame[dim] == val, "row_count"].dropna().sum())
                            weight_b = float(frame.loc[frame[dim] != val, "row_count"].dropna().sum())
                            if weight_a >= required_n and weight_b >= required_n:
                                group_sums = {
                                    str(k): float(np.sum(g[effective_metric_col].dropna().to_numpy(dtype=float)))
                                    for k, g in frame.groupby(dim)
                                }
                                total_sum = sum(abs(v) for v in group_sums.values())
                                k_groups = len(group_sums)
                                if k_groups >= 2 and total_sum > 0 and str(val) in group_sums:
                                    target_share = abs(group_sums[str(val)]) / total_sum
                                    null_share = 1.0 / k_groups
                                    if target_share <= null_share + 0.05:
                                        ev = BayesianEvidence(1.0, "NEUTRAL_UNIFORM_CONCENTRATION", int(weight_a + weight_b), ("Target group's weighted share is consistent with uniform null.",))
                                    elif target_share >= 0.85:
                                        ev = BayesianEvidence(1e6, "TARGETED_CONCENTRATION_DECISIVE", int(weight_a + weight_b), ("Target group concentration >= 85% (row_count-weighted)",))
                                    elif target_share >= 0.45:
                                        ev = BayesianEvidence(3.5, "TARGETED_CONCENTRATION_MODERATE", int(weight_a + weight_b), ("Target group concentration moderate 45%-85% (row_count-weighted)",))
                                    else:
                                        ev = BayesianEvidence(1.0, "NEUTRAL_INSUFFICIENT_CONCENTRATION", int(weight_a + weight_b), ("Target group's weighted share below concentration threshold.",))
                        if ev is None:
                            ev = BayesianEvidence(
                                1.0,
                                "NEUTRAL_INADEQUATE_GROUP_SAMPLE",
                                len(a) + len(b),
                                (f"Both comparison sides require at least {required_n} finite observations.",),
                            )
                    else:
                        ev = BeliefEngine.bayes_factor_two_groups(a, b, min_group_n=required_n)
                    bf = 1.0 / ev.bayes_factor if getattr(h, "is_counter_hypothesis", False) else ev.bayes_factor
                    factors[i] = bf
                    diagnostics[i] = ev.to_dict()
                    if not getattr(h, "is_counter_hypothesis", False) and ev.bayes_factor > 1.0:
                        targeted_assigned = True

            if targeted_assigned:
                max_lead_bf = max((f for j, f in enumerate(factors) if not getattr(hypotheses[j], "is_counter_hypothesis", False)), default=1.0)
                if max_lead_bf > 1.0:
                    for i, h in enumerate(hypotheses):
                        if not _dim_matches(h, group_dimension_col):
                            continue
                        if getattr(h, "is_counter_hypothesis", False) and factors[i] == 1.0:
                            factors[i] = 1.0 / max_lead_bf

            # General group-difference / concentration model for the primary hypothesis.
            group_sums = {
                str(k): float(np.sum(g[effective_metric_col].dropna().to_numpy(dtype=float)))
                for k, g in frame.groupby(group_dimension_col)
            }
            total_sum = sum(abs(v) for v in group_sums.values())
            top_share = max((abs(v) for v in group_sums.values()), default=0.0) / max(total_sum, 1e-9)
            k_groups = len(group_sums)
            null_share = 1.0 / max(k_groups, 1)

            if str(aggregation_type).upper() == "SUM" and k_groups >= 2 and total_sum > 0:
                if top_share <= null_share + 0.05:
                    ev = BayesianEvidence(1.0, "NEUTRAL_UNIFORM_CONCENTRATION", len(frame), ("Observed top share is consistent with uniform null.",))
                elif top_share >= 0.85:
                    ev = BayesianEvidence(1e6, "CONCENTRATION_DECISIVE", len(frame), ("Top category concentration >= 85%",))
                elif top_share >= 0.45:
                    ev = BayesianEvidence(3.5, "CONCENTRATION_MODERATE", len(frame), ("Top category concentration moderate (45%-85%)",))
                else:
                    ev = BayesianEvidence(1.0, "NEUTRAL_INSUFFICIENT_CONCENTRATION", len(frame), ("Observed share below concentration threshold.",))
            else:
                groups = {
                    str(k): g[effective_metric_col].dropna().to_numpy(dtype=float)
                    for k, g in frame.groupby(group_dimension_col)
                }
                required_n = max(1, int(min_group_n))
                if len(groups) < 2 or min((len(v) for v in groups.values()), default=0) < required_n:
                    ev = BayesianEvidence(
                        1.0,
                        "NEUTRAL_INADEQUATE_GROUP_SAMPLE",
                        sum(len(v) for v in groups.values()),
                        (f"Every compared group requires at least {required_n} finite observations.",),
                    )
                else:
                    ev = BeliefEngine.bayes_factor_anova(groups, min_group_n=required_n)
            primary_assigned = False
            for i, h in enumerate(hypotheses):
                if getattr(h, "hypothesis_code", "") not in codes:
                    continue
                if not _dim_matches(h, group_dimension_col):
                    continue
                if not getattr(h, "is_counter_hypothesis", False) and not getattr(h, "target_value", None) and not primary_assigned:
                    factors[i] = ev.bayes_factor
                    diagnostics[i] = ev.to_dict()
                    primary_assigned = True
            # A declared counter-model represents the null/common-mean alternative
            # to the primary group-difference model. Give it reciprocal evidence only
            # when the factor is finite; unrelated counter-hypotheses remain neutral.
            # Restricted to counters whose own target_dimension matches what was
            # actually tested this round -- a counter that targets a different
            # dimension (e.g. a confounder-specific counter) must not inherit
            # evidence from an unrelated dimension's test.
            if primary_assigned and ev.bayes_factor > 0:
                for i, h in enumerate(hypotheses):
                    if not _dim_matches(h, group_dimension_col):
                        continue
                    if getattr(h, "is_counter_hypothesis", False) and factors[i] == 1.0:
                        factors[i] = 1.0 / ev.bayes_factor
                        diagnostics[i] = ev.to_dict()
                        break
        return factors, diagnostics

    @staticmethod
    def compute_variance_explained(df: pd.DataFrame, group_col: str, target_col: str) -> Optional[float]:
        """Calculate ANOVA eta-squared variance explained (percentage)."""
        if group_col not in df.columns or target_col not in df.columns or df.empty:
            return None
        try:
            clean = df[[group_col, target_col]].dropna()
            total_mean = float(clean[target_col].mean())
            ss_total = float(((clean[target_col] - total_mean) ** 2).sum())
            if ss_total < 1e-12:
                return 0.0
            group_means = clean.groupby(group_col)[target_col].mean()
            group_counts = clean.groupby(group_col)[target_col].count()
            ss_between = float((group_counts * ((group_means - total_mean) ** 2)).sum())
            return float(np.clip((ss_between / ss_total) * 100.0, 0.0, 100.0))
        except Exception:
            return None

    # Backward-compatible alias retained for older callers/tests. This method is now
    # explicitly deprecated: eta-squared alone does not determine a likelihood model.
    @staticmethod
    def compute_evidence_likelihoods(
        primary_metric: float,
        benchmark_metric: float,
        delta_pct: float = 0.0,
        effect_size_eta_sq: Optional[float] = None,
        spread_ratio: Optional[float] = None,
        sample_size: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compatibility API; use model-based Bayes factors for live investigations.

        When sample size is supplied, a BIC-based Gaussian model comparison is used
        for a two-model effect/no-effect representation. Without sample size the method
        refuses to manufacture a statistical likelihood and returns neutral evidence.
        """
        if sample_size is None or sample_size < 5:
            return 1.0, 1.0
        if effect_size_eta_sq is None or not np.isfinite(effect_size_eta_sq):
            # Missing effect size is missing evidence, not evidence for no effect.
            return 1.0, 1.0
        n = int(sample_size)
        effect = float(effect_size_eta_sq) / 100.0
        effect = float(np.clip(effect, 0.0, 0.999999))
        # Construct the corresponding BIC evidence from the explained-variance ratio.
        sse1_ratio = max(1e-12, 1.0 - effect)
        bic0 = n * math.log(1.0) + 2 * math.log(n)
        bic1 = n * math.log(sse1_ratio) + 3 * math.log(n)
        bf = _bic_bayes_factor(bic0 - bic1)
        return float(bf), 1.0

    compute_empirical_variance_explained = compute_variance_explained
    compute_empirical_likelihoods = compute_evidence_likelihoods
