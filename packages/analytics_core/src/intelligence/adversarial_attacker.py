"""AdversarialAttacker: Rigorous multi-angle adversarial falsification engine for leading hypotheses."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis


@dataclass
class AdversarialAttackResult:
    """Outcome of an active adversarial attack on the leading hypothesis."""
    leading_hypothesis_code: str
    counter_hypothesis_code: str
    attack_mechanism: str
    discriminating_test_sql: str
    # SURVIVED, WEAKENED, REFUTED, CONTRADICTED, or NOT_APPLICABLE. NOT_APPLICABLE means
    # no attack could be constructed (no resolved dimension/metric, missing columns,
    # non-numeric metric): it is NOT evidence that the hypothesis survived.
    attack_status: str
    epistemic_impact: str
    is_falsified: bool
    simpsons_paradox_detected: bool = False
    outlier_sensitivity_high: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


def detect_simpsons_reversal(
    df: pd.DataFrame,
    dim: str,
    sec_dim: str,
    metric: str,
    min_cell_n: int = 5,
    min_effect_sd: float = 0.2,
    max_groups: int = 12,
) -> Optional[Dict[str, Any]]:
    """Detect a genuine Simpson's reversal of a group comparison across strata.

    A reversal for groups (a, b) requires ALL of:
      * every stratum with adequate data (>= min_cell_n rows in both groups) shows the
        same sign of difference (mixed signs are effect modification, not a reversal);
      * the stratum-size-standardised (adjusted) difference has the opposite sign of the
        marginal difference;
      * both the marginal and adjusted differences are material (>= min_effect_sd
        within-cell standard deviations), so noise in tiny cells cannot trigger it;
      * at least two adequate strata.
    All group pairs are examined (up to max_groups largest groups), not only the first two.
    """
    sub = df[[dim, sec_dim, metric]].dropna()
    if sub.empty:
        return None
    cell = sub.groupby([dim, sec_dim])[metric]
    cell_n = cell.count()
    cell_mean = cell.mean()
    resid = sub[metric] - cell.transform("mean")
    dof = max(1, len(sub) - int((cell_n > 0).sum()))
    within_sd = float((resid.pow(2).sum() / dof) ** 0.5)
    if not np.isfinite(within_sd) or within_sd <= 0:
        return None
    sizes = sub.groupby(dim)[metric].size().sort_values(ascending=False)
    groups = list(sizes.index[:max_groups])
    marg = sub.groupby(dim)[metric].mean()
    strata = list(sub[sec_dim].unique())
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            diffs, weights = [], []
            for st in strata:
                na, nb = cell_n.get((a, st), 0), cell_n.get((b, st), 0)
                if na >= min_cell_n and nb >= min_cell_n:
                    diffs.append(float(cell_mean[(a, st)] - cell_mean[(b, st)]))
                    weights.append(float(na + nb))
            if len(diffs) < 2:
                continue
            adjusted = float(np.average(diffs, weights=weights))
            marginal = float(marg[a] - marg[b])
            same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
            if (
                same_sign
                and marginal * adjusted < 0
                and abs(marginal) >= min_effect_sd * within_sd
                and abs(adjusted) >= min_effect_sd * within_sd
            ):
                return {
                    "groups": [str(a), str(b)],
                    "marginal_difference": round(marginal, 6),
                    "adjusted_difference": round(adjusted, 6),
                    "strata_used": len(diffs),
                    "secondary_dimension": sec_dim,
                }
    return None


class AdversarialAttacker:
    """
    Multi-angle adversarial falsification engine searching for:
    1. Confounding and Simpson's paradox across secondary partitions
    2. Extreme outlier/leverage sensitivity (top 1% artifact check)
    3. Missingness & selection bias
    4. Aggregation & sample imbalance artifacts
    """

    @staticmethod
    def design_attack(
        leading_hyp: PredictiveHypothesis,
        counter_hyps: List[PredictiveHypothesis],
        df: pd.DataFrame,
        semantic: Optional[Any] = None,
        execution_provider: Optional[Any] = None,
    ) -> AdversarialAttackResult:
        counter = counter_hyps[0] if counter_hyps else leading_hyp
        dim = leading_hyp.target_dimension or ""
        metric = leading_hyp.target_metric or ""
        mechanism = f"Multi-angle falsification test (Simpson's paradox, outlier leverage, sample imbalance) for {leading_hyp.hypothesis_code}."

        # P0 (defect019 audit): a hypothesis with no resolved target_dimension
        # (e.g. status UNRESOLVED/AMBIGUOUS from resolve_group_dimension, or
        # a non-comparison hypothesis) must not be attacked against a
        # fabricated "segment" column. If a real dataset happens to contain
        # a column literally named "segment" that is unrelated to this
        # hypothesis, the old fallback would silently attack against it and
        # produce a misleading result. Fail closed instead: this attack
        # class is not applicable.
        if not dim:
            return AdversarialAttackResult(
                leading_hypothesis_code=leading_hyp.hypothesis_code,
                counter_hypothesis_code=counter.hypothesis_code,
                attack_mechanism=mechanism,
                discriminating_test_sql="",
                attack_status="NOT_APPLICABLE",
                epistemic_impact=(
                    "Adversarial attack not applicable: hypothesis has no resolved target_dimension "
                    f"(dimension_resolution_status={getattr(leading_hyp, 'dimension_resolution_status', 'UNKNOWN')}); "
                    "a group-comparison attack cannot be constructed without fabricating a grouping column."
                ),
                is_falsified=False,
            )
        if not metric:
            return AdversarialAttackResult(
                leading_hypothesis_code=leading_hyp.hypothesis_code,
                counter_hypothesis_code=counter.hypothesis_code,
                attack_mechanism=mechanism,
                discriminating_test_sql="",
                attack_status="NOT_APPLICABLE",
                epistemic_impact="Adversarial attack not applicable: hypothesis has no resolved target_metric.",
                is_falsified=False,
            )

        attack_sql = f"SELECT {dim}, COUNT(*) AS group_size, AVG({metric}) AS group_mean FROM data_table GROUP BY {dim} HAVING COUNT(*) >= 5"

        if dim not in df.columns or metric not in df.columns or df.empty:
            return AdversarialAttackResult(
                leading_hypothesis_code=leading_hyp.hypothesis_code,
                counter_hypothesis_code=counter.hypothesis_code,
                attack_mechanism=mechanism,
                discriminating_test_sql=attack_sql,
                attack_status="NOT_APPLICABLE",
                epistemic_impact="Adversarial attack not applicable: required columns are missing or the dataset is empty; the leader was not tested.",
                is_falsified=False,
            )

        # BUGFIX (DEFECT-005): this engine unconditionally computed
        # group-wise .mean()/.quantile() on the target metric column,
        # assuming it is always numeric/additive. For a COUNT_DISTINCT-style
        # metric over a non-numeric identifier column (e.g. customer_id),
        # pandas raises TypeError ("dtype 'str' does not support operation
        # 'mean'") and the whole investigation crashes uncaught. Mean/
        # quantile-based spread and outlier-leverage attacks are only
        # meaningful for a genuinely numeric metric column in the first
        # place, so skip them (SURVIVED, not falsified) rather than assuming
        # applicability -- mirrors the existing "insufficient data" early
        # return above.
        if not pd.api.types.is_numeric_dtype(df[metric]):
            return AdversarialAttackResult(
                leading_hypothesis_code=leading_hyp.hypothesis_code,
                counter_hypothesis_code=counter.hypothesis_code,
                attack_mechanism=mechanism,
                discriminating_test_sql=attack_sql,
                attack_status="NOT_APPLICABLE",
                epistemic_impact=(
                    "Adversarial attack completed: mean/quantile-based spread and outlier-leverage "
                    f"tests are not applicable to non-numeric target column '{metric}'."
                ),
                is_falsified=False,
            )

        # 1. Primary Spread & Imbalance Test
        group_means = df.groupby(dim)[metric].mean()
        max_mean = float(group_means.max())
        min_mean = float(group_means.min())
        overall_mean = float(group_means.mean())
        spread_ratio = (max_mean - min_mean) / max(1e-9, abs(overall_mean))

        # 2. Outlier Leverage Attack (Top 1% trimming)
        q99 = df[metric].quantile(0.99)
        trimmed_df = df[df[metric] <= q99]
        trimmed_means = trimmed_df.groupby(dim)[metric].mean()
        trimmed_spread = (float(trimmed_means.max()) - float(trimmed_means.min())) / max(1e-9, abs(float(trimmed_means.mean()))) if not trimmed_means.empty else spread_ratio
        outlier_sensitive = (spread_ratio > 0.30 and trimmed_spread < 0.10)

        # 3. Simpson's Paradox / Secondary Partition Confounding Check
        simpsons_paradox = False
        other_dims = [c for c in df.select_dtypes(include=["object", "category", "string"]).columns if c != dim and 2 <= df[c].nunique() <= 20]
        # Prefer semantically informative secondary partitions; never use
        # physical column order as an adversarial choice. Ties are ambiguous
        # and therefore do not trigger a confounding claim.
        scored_dims = []
        for c in other_dims:
            low = str(c).lower()
            score = 0.0
            if any(tok in low for tok in ("region", "segment", "channel", "tier", "plan", "cohort", "tenure", "status", "category")):
                score += 2.0
            score += min(float(df[c].nunique()) / 10.0, 1.0)
            scored_dims.append((score, str(c)))
        scored_dims.sort(key=lambda x: (-x[0], x[1]))
        sec_dim = scored_dims[0][1] if scored_dims and (len(scored_dims) == 1 or scored_dims[0][0] > scored_dims[1][0]) else None
        simpsons_detail = None
        if sec_dim:
            try:
                simpsons_detail = detect_simpsons_reversal(df, dim, sec_dim, metric)
            except Exception:
                simpsons_detail = None
            simpsons_paradox = simpsons_detail is not None

        # 4. Measured selection/survivorship attack. Status-like fields are only
        # candidates; require an observed target difference before treating the
        # mechanism as an adversarial finding.
        selection_attack = None
        status_cols = [c for c in df.columns if any(tok in str(c).lower().split("_") for tok in ("active", "current", "eligible", "surviv"))]
        for c in status_cols:
            vals = df[c].dropna()
            if vals.nunique() != 2:
                continue
            a, b = list(vals.unique())[:2]
            av = pd.to_numeric(df.loc[df[c] == a, metric], errors="coerce").dropna()
            bv = pd.to_numeric(df.loc[df[c] == b, metric], errors="coerce").dropna()
            if len(av) >= 5 and len(bv) >= 5:
                pooled = max(float(np.nanstd(df[metric])), 1e-12)
                gap = abs(float(av.mean()) - float(bv.mean())) / pooled
                if gap >= 0.20:
                    selection_attack = {"field": c, "standardized_mean_gap": round(gap, 6), "group_sizes": [len(av), len(bv)]}
                    break

        # 5. Measured target/post-treatment leakage attack. A name creates a
        # candidate, but only near-deterministic observed dependence is recorded.
        leakage_attack = None
        if metric in df.columns:
            leak_tokens = ("outcome", "churn_date", "cancel_date", "cancellation_reason", "refund_after", "post_", "future_", "target_", "label")
            for c in df.columns:
                if c == metric or not any(tok in str(c).lower() for tok in leak_tokens):
                    continue
                pair = pd.DataFrame({"target": df[metric], "feature": df[c]}).dropna()
                if len(pair) < 10:
                    continue
                score = None
                if pd.api.types.is_numeric_dtype(pair["target"]) and pd.api.types.is_numeric_dtype(pair["feature"]):
                    if pair["target"].nunique() > 1 and pair["feature"].nunique() > 1:
                        score = abs(float(pair["target"].corr(pair["feature"])))
                if score is not None and score >= 0.995:
                    leakage_attack = {"field": c, "absolute_correlation": round(score, 6), "sample_size": len(pair)}
                    break

        # Synthesize Attack Status & Epistemic Impact
        #
        # DEFECT-021 (vision audit O3/decision #2): detect_simpsons_reversal()
        # is already a materiality-gated, same-sign, min-cell-size-checked
        # test (not a raw two-group peek), so a hit from it is a genuine
        # falsifying finding, not merely a "weakening" caveat -- the
        # marginal comparison the leading hypothesis rests on reverses sign
        # once a real confound is stratified out. That is direct evidence
        # against the hypothesis as stated, so it is scored REFUTED /
        # is_falsified=True, distinct from the softer WEAKENED findings
        # below (leakage/selection/outlier-sensitivity/low-dispersion),
        # which flag risk without demonstrating the claimed direction is
        # actually wrong.
        if simpsons_paradox:
            status = "REFUTED"
            falsified = True
            impact = (
                f"Adversarial attack succeeded: a genuine Simpson's reversal was found when conditioning on "
                f"secondary partition '{sec_dim}' -- the marginal difference between "
                f"{simpsons_detail['groups'][0]!r} and {simpsons_detail['groups'][1]!r} "
                f"({simpsons_detail['marginal_difference']:+.4g}) flips sign once stratified "
                f"({simpsons_detail['adjusted_difference']:+.4g} across {simpsons_detail['strata_used']} adequate strata). "
                f"The unconditional effect of {dim} is confounded by '{sec_dim}' and does not support the hypothesis as stated."
            )
        elif leakage_attack:
            status = "WEAKENED"
            falsified = False
            impact = f"Adversarial Warning: measured near-deterministic relationship with potential post-outcome field '{leakage_attack['field']}' requires leakage exclusion before a positive claim."
        elif selection_attack:
            status = "WEAKENED"
            falsified = False
            impact = f"Adversarial Warning: observed metric differs by selection/status field '{selection_attack['field']}' (standardized gap {selection_attack['standardized_mean_gap']:.3f}); survivorship/eligibility bias may explain the effect."
        elif outlier_sensitive:
            status = "WEAKENED"
            falsified = False
            impact = f"Adversarial Warning: High outlier sensitivity detected. Trimming top 1% outliers reduces {dim} effect spread from {spread_ratio:.1%} to {trimmed_spread:.1%}."
        elif spread_ratio > 0.40:
            status = "SURVIVED"
            falsified = False
            impact = f"Adversarial attack failed: Leading hypothesis robust against subgroup repartitioning and outlier trimming (Spread: {spread_ratio:.1%})."
        else:
            status = "WEAKENED"
            falsified = False
            impact = f"Adversarial challenge: {dim} categories show low relative dispersion (Spread: {spread_ratio:.1%}), weakening localized hypothesis."

        details = {}
        if simpsons_paradox and other_dims:
            details["secondary_dimension"] = sec_dim
            details["confounding_dimension"] = sec_dim
            details["simpsons_reversal"] = {**(simpsons_detail or {}), "primary_dimension": dim, "metric": metric}
        if outlier_sensitive:
            details["outlier_trimming"] = "1PCT"
        if selection_attack:
            details["selection_bias_attack"] = selection_attack
        if leakage_attack:
            details["target_leakage_attack"] = leakage_attack

        return AdversarialAttackResult(
            leading_hypothesis_code=leading_hyp.hypothesis_code,
            counter_hypothesis_code=counter.hypothesis_code,
            attack_mechanism=mechanism,
            discriminating_test_sql=attack_sql,
            attack_status=status,
            epistemic_impact=impact,
            is_falsified=falsified,
            simpsons_paradox_detected=simpsons_paradox,
            outlier_sensitivity_high=outlier_sensitive,
            details=details,
        )

    @classmethod
    def evaluate_candidate_for_falsification(
        cls,
        hypothesis: PredictiveHypothesis,
        df: pd.DataFrame,
        semantic: Optional[Any] = None,
        counter_hypotheses: Optional[List[PredictiveHypothesis]] = None,
    ) -> AdversarialAttackResult:
        """Alias for evaluating adversarial attacks against a hypothesis."""
        counters = counter_hypotheses or []
        return cls.design_attack(leading_hyp=hypothesis, counter_hyps=counters, df=df, semantic=semantic)

    @classmethod
    def execute_adversarial_attack(
        cls,
        df: pd.DataFrame,
        leading_hypothesis: PredictiveHypothesis,
        counter_hypothesis: PredictiveHypothesis,
        semantic: Optional[Any] = None,
        execution_provider: Optional[Any] = None,
        **kwargs,
    ) -> AdversarialAttackResult:
        return cls.design_attack(
            leading_hyp=leading_hypothesis,
            counter_hyps=[counter_hypothesis],
            df=df,
            semantic=semantic,
            execution_provider=execution_provider,
        )
