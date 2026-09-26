"""
churn_estimands.py — DEFECT-005 churn/segmentation identifiability repair.

Implements the estimand distinctions required by the defect brief without
touching Bayesian priors, stopping thresholds, or the BeliefEngine itself.
This module answers a narrower, prior question: **what is the correct
number to feed the belief engine, and is a segment effect identifiable at
all given the data present** -- not "how confident should we be."

Estimands supported (Sec 4 of the brief):
    A. raw_churn_count      -- number of churn events
    B. crude_churn_rate     -- events / eligible population
    C. exposure_adjusted_rate -- events / person-time at risk
    D. segment_comparison    -- explicit numerator/denominator/unit for each
                                 side of a comparison, never silently mixed

Verdict taxonomy (Sec 13):
    OBSERVED_ASSOCIATION           -- a difference exists in the data, no
                                       claim about cause or robustness yet
    SUPPORTED_SEGMENT_DIFFERENCE   -- the difference survives available
                                       confounder stratification
    CONFOUNDED_IDENTIFIABILITY_LIMITED -- a competing explanation (cohort/
                                       tenure/exposure) explains the
                                       aggregate difference away
    INSUFFICIENT_EVIDENCE          -- not enough data/fields to distinguish
                                       hypotheses at all

Non-negotiables enforced here (see task brief "NON-NEGOTIABLE AA-OS
PRINCIPLES"): no fabricated confidence, no causal language, censored rows
are never silently treated as confirmed non-churn, no hardcoded
dataset-specific branching (all functions operate generically off column
presence/values, not off hardcoded segment names).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

REQUIRED_EVENT_COL = "churn_event"


class ChurnVerdict:
    OBSERVED_ASSOCIATION = "OBSERVED_ASSOCIATION"
    SUPPORTED_SEGMENT_DIFFERENCE = "SUPPORTED_SEGMENT_DIFFERENCE"
    CONFOUNDED_IDENTIFIABILITY_LIMITED = "CONFOUNDED_IDENTIFIABILITY_LIMITED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class EstimandSpec:
    """Explicit statement of what is being measured (Sec 4). Every churn
    computation in this module must be traceable to one of these so no
    estimand is silently substituted for another."""
    numerator: str
    denominator: str
    unit_of_analysis: str
    observation_window: str
    eligibility_criteria: str
    censoring_treatment: str


@dataclass
class ChurnAnalysisResult:
    verdict: str
    estimand: EstimandSpec
    segment_summary: pd.DataFrame
    aggregate_association: Optional[Dict] = None
    stratified_checks: List[Dict] = field(default_factory=list)
    confounders_detected: List[str] = field(default_factory=list)
    narrative: str = ""
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# A/B/C: raw count, crude rate, exposure-adjusted rate
# ---------------------------------------------------------------------------

def raw_churn_count(df: pd.DataFrame, group_col: str) -> pd.Series:
    """A. Raw churn event count per group. NEVER use this alone to rank
    segments by risk -- see Sec 5."""
    _require_columns(df, [group_col, REQUIRED_EVENT_COL])
    return df.groupby(group_col)[REQUIRED_EVENT_COL].sum()


def crude_churn_rate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """B. events / eligible population. Eligible population = every row
    with a non-null churn_event (churned OR confirmed-non-churned).
    Rows that are censored-only (churn_event is NaN because the outcome
    was never confirmed) are excluded from both numerator and
    denominator here -- they are handled by exposure_adjusted_rate
    instead, per Sec 6."""
    _require_columns(df, [group_col, REQUIRED_EVENT_COL])
    eligible = df[df[REQUIRED_EVENT_COL].notna()]
    grouped = eligible.groupby(group_col)[REQUIRED_EVENT_COL].agg(["sum", "count"])
    grouped = grouped.rename(columns={"sum": "events", "count": "eligible_n"})
    grouped["crude_rate"] = grouped["events"] / grouped["eligible_n"]
    return grouped.reset_index()


def exposure_adjusted_rate(df: pd.DataFrame, group_col: str,
                            exposure_col: str = "observation_days") -> pd.DataFrame:
    """C. events / person-time at risk. Uses every row that has exposure
    data, INCLUDING censored rows (their partial exposure still
    contributes person-time even though their outcome is unconfirmed --
    this is the standard actuarial treatment, not an invented one)."""
    _require_columns(df, [group_col, REQUIRED_EVENT_COL, exposure_col])
    usable = df[df[exposure_col].notna()]
    grouped = usable.groupby(group_col).agg(
        events=(REQUIRED_EVENT_COL, lambda s: s.fillna(0).sum()),
        person_time=(exposure_col, "sum"),
        n=(exposure_col, "count"),
    )
    grouped["persontime_rate"] = grouped["events"] / grouped["person_time"]
    return grouped.reset_index()


# ---------------------------------------------------------------------------
# Censoring (Sec 6)
# ---------------------------------------------------------------------------

def censoring_report(df: pd.DataFrame, censored_col: str = "censored") -> Dict:
    """Distinguishes 'did not churn' from 'not observed long enough to
    establish non-churn'. Returns counts so downstream logic can decide
    whether a crude-rate estimand is even valid for this dataset."""
    if censored_col not in df.columns or df[censored_col].isna().all():
        return {
            "censoring_data_available": False,
            "note": "No censoring field present; cannot distinguish confirmed "
                    "non-churn from incomplete observation. Crude rate treats "
                    "all non-events as confirmed non-churn -- this is only "
                    "valid if the product contract defines it that way.",
        }
    n_total = len(df)
    n_censored = int(df[censored_col].fillna(0).astype(int).sum())
    n_confirmed_nonchurn = int(((df[censored_col] == 0) & (df[REQUIRED_EVENT_COL] == 0)).sum())
    n_events = int((df[REQUIRED_EVENT_COL] == 1).sum())
    return {
        "censoring_data_available": True,
        "n_total": n_total,
        "n_events": n_events,
        "n_confirmed_nonchurn": n_confirmed_nonchurn,
        "n_censored_incomplete_observation": n_censored,
        "note": (
            f"{n_censored} of {n_total} rows are right-censored (observation "
            "ended before churn could be confirmed either way) and must NOT "
            "be counted as confirmed survivors in a crude-rate estimand."
        ),
    }


# ---------------------------------------------------------------------------
# Confounder checks (Sec 7, 8) -- stratification / Simpson's-paradox detection
# ---------------------------------------------------------------------------

def stratified_rate_check(df: pd.DataFrame, group_col: str, stratum_col: str,
                           min_stratum_n: int = 20) -> Dict:
    """Compares the aggregate group difference to the within-stratum
    differences. If the aggregate direction/magnitude does not survive
    stratification, this is a Simpson's-paradox-style composition
    confound (Sec 15.A) and the stratum_col is a competing explanation
    for the observed segment difference."""
    _require_columns(df, [group_col, stratum_col, REQUIRED_EVENT_COL])
    eligible = df[df[REQUIRED_EVENT_COL].notna() & df[stratum_col].notna()].copy()

    # Continuous confounders (especially tenure/exposure) cannot be used as
    # literal strata because nearly every value may be unique.  Treating each
    # raw value as a stratum therefore makes the check inapplicable even when
    # the confounder is genuinely balanced or genuinely non-overlapping.
    # Build deterministic quantile bands from the observed pooled population.
    # The bands are only an identifiability diagnostic; they do not alter the
    # primary estimand.  Use at most five bands and retain only bands with
    # sufficient observations in both comparison groups.
    if pd.api.types.is_numeric_dtype(eligible[stratum_col]):
        # If the comparison groups occupy disjoint ranges of a continuous
        # confounder, there is no common support on which the segment effect
        # can be identified. Do not let a pooled quantile bin create a fake
        # overlap merely because a bin boundary spans both ranges.
        group_values = {
            g: eligible.loc[eligible[group_col] == g, stratum_col].astype(float)
            for g in eligible[group_col].dropna().unique()
        }
        if len(group_values) == 2:
            (g_a, a), (g_b, b) = list(group_values.items())
            a_min, a_max = float(a.min()), float(a.max())
            b_min, b_max = float(b.min()), float(b.max())
            has_common_support = max(a_min, b_min) <= min(a_max, b_max)
            if not has_common_support:
                return {
                    "applicable": False,
                    "group1": g_a, "group2": g_b,
                    "aggregate_diff": float(
                        crude_churn_rate(eligible, group_col).set_index(group_col).loc[g_b, "crude_rate"]
                        - crude_churn_rate(eligible, group_col).set_index(group_col).loc[g_a, "crude_rate"]
                    ),
                    "per_stratum": [],
                    # Complete separation means there is no shared stratum on
                    # which to compare the groups at all -- this is
                    # genuine non-identifiability, not a positive finding of
                    # a confound. Confirming a confound requires observing
                    # the segment difference collapse *within* a shared
                    # stratum; ruling one out requires observing it survive
                    # one. Neither is possible here, so this must not be
                    # reported as `confound_detected: True` (which would
                    # misrepresent "we could not check" as "we checked and
                    # found a competing explanation"). Callers that need to
                    # treat non-identifiability the same as a detected
                    # confound for verdict purposes should key off
                    # `applicable: False`, not a fabricated positive.
                    "confound_detected": False,
                    "stratum_binned": False,
                    "reason": (
                        f"{stratum_col} has no common support between the two groups "
                        f"([{a_min:.4g}, {a_max:.4g}] vs [{b_min:.4g}, {b_max:.4g}]); "
                        "the segment comparison is not identifiable after adjustment."
                    ),
                }

        unique_n = int(eligible[stratum_col].nunique())
        if unique_n > 10:
            try:
                bands = pd.qcut(eligible[stratum_col], q=min(5, unique_n), duplicates="drop")
                if bands.notna().sum() >= 2:
                    eligible["__aaos_stratum_band__"] = bands.astype(str)
                    stratum_for_grouping = "__aaos_stratum_band__"
                else:
                    stratum_for_grouping = stratum_col
            except (TypeError, ValueError):
                stratum_for_grouping = stratum_col
        else:
            stratum_for_grouping = stratum_col
    else:
        stratum_for_grouping = stratum_col

    groups = sorted(eligible[group_col].dropna().unique().tolist())
    if len(groups) != 2:
        return {"applicable": False, "reason": "stratified check implemented for exactly two groups"}

    g1, g2 = groups
    agg = crude_churn_rate(eligible, group_col).set_index(group_col)
    if g1 not in agg.index or g2 not in agg.index:
        return {"applicable": False, "reason": "insufficient data for one group"}
    agg_diff = agg.loc[g2, "crude_rate"] - agg.loc[g1, "crude_rate"]

    strata = sorted(eligible[stratum_for_grouping].dropna().unique().tolist())
    per_stratum = []
    for s in strata:
        sub = eligible[eligible[stratum_for_grouping] == s]
        rates = crude_churn_rate(sub, group_col).set_index(group_col)
        if g1 in rates.index and g2 in rates.index and \
           rates.loc[g1, "eligible_n"] >= min_stratum_n and rates.loc[g2, "eligible_n"] >= min_stratum_n:
            diff = rates.loc[g2, "crude_rate"] - rates.loc[g1, "crude_rate"]
            per_stratum.append({
                "stratum": s, "group1_rate": float(rates.loc[g1, "crude_rate"]),
                "stratum_source": stratum_col,
                "stratum_binned": stratum_for_grouping != stratum_col,
                "group2_rate": float(rates.loc[g2, "crude_rate"]),
                "diff": float(diff), "adequate_sample": True,
            })
        else:
            per_stratum.append({"stratum": s, "adequate_sample": False,
                            "stratum_source": stratum_col,
                            "stratum_binned": stratum_for_grouping != stratum_col})

    adequate = [s for s in per_stratum if s.get("adequate_sample")]
    if not adequate:
        return {
            "applicable": False,
            "reason": f"no stratum of {stratum_col} has >= {min_stratum_n} eligible rows per group",
            "stratum_binned": stratum_for_grouping != stratum_col,
            "aggregate_diff": float(agg_diff),
        }

    # A confound is flagged when the within-stratum differences are
    # substantially smaller than the aggregate difference, or flip sign,
    # for every adequately-sampled stratum.
    stratum_diffs = [s["diff"] for s in adequate]
    magnitude_collapses = all(abs(d) < 0.5 * abs(agg_diff) for d in stratum_diffs) if agg_diff != 0 else False
    sign_flips = any((d * agg_diff) < 0 for d in stratum_diffs) if agg_diff != 0 else False
    confound_detected = magnitude_collapses or sign_flips

    return {
        "applicable": True,
        "group1": g1, "group2": g2,
        "aggregate_diff": float(agg_diff),
        "per_stratum": per_stratum,
        "confound_detected": bool(confound_detected),
        "stratum_binned": stratum_for_grouping != stratum_col,
        "reason": (
            f"Aggregate difference ({agg_diff:.4f}) does not survive stratification by "
            f"{stratum_col}: within-stratum differences collapse or flip sign."
            if confound_detected else
            f"Aggregate difference ({agg_diff:.4f}) persists across strata of {stratum_col}."
        ),
    }


# ---------------------------------------------------------------------------
# Top-level analysis: produces a verdict, never silently upgrades association
# to a causal/intrinsic-risk claim (Sec 13, 14)
# ---------------------------------------------------------------------------

def analyze_churn_identifiability(df: pd.DataFrame, group_col: str,
                                   known_confounders: Optional[List[str]] = None,
                                   exposure_col: Optional[str] = "observation_days",
                                   censored_col: Optional[str] = "censored",
                                   min_group_n: int = 20) -> ChurnAnalysisResult:
    """Single entry point implementing the estimand + identifiability
    checks. Returns a ChurnAnalysisResult carrying an explicit verdict
    from ChurnVerdict, never inventing certainty the data doesn't support.
    """
    warnings: List[str] = []
    known_confounders = known_confounders or []

    if group_col not in df.columns:
        return ChurnAnalysisResult(
            verdict=ChurnVerdict.INSUFFICIENT_EVIDENCE,
            estimand=EstimandSpec(numerator="n/a", denominator="n/a", unit_of_analysis="n/a",
                                   observation_window="n/a", eligibility_criteria="n/a",
                                   censoring_treatment="n/a"),
            segment_summary=pd.DataFrame(),
            narrative=f"Column '{group_col}' not present; no segment comparison is possible.",
        )

    if REQUIRED_EVENT_COL not in df.columns or df[REQUIRED_EVENT_COL].notna().sum() == 0:
        return ChurnAnalysisResult(
            verdict=ChurnVerdict.INSUFFICIENT_EVIDENCE,
            estimand=EstimandSpec(numerator="n/a", denominator="n/a", unit_of_analysis="n/a",
                                   observation_window="n/a", eligibility_criteria="n/a",
                                   censoring_treatment="n/a"),
            segment_summary=pd.DataFrame(),
            narrative=(
                "No churn/cancellation/attrition event column is present in this dataset. "
                "AA-OS cannot compute a churn rate or claim a churn-risk difference between "
                "segments from data that does not record whether or when customers churned. "
                "This is a data-availability limitation, not a fabricated negative result."
            ),
        )

    groups = df[group_col].dropna().unique().tolist()
    if len(groups) < 2:
        return ChurnAnalysisResult(
            verdict=ChurnVerdict.INSUFFICIENT_EVIDENCE,
            estimand=EstimandSpec(numerator="churn events", denominator="eligible population",
                                   unit_of_analysis="customer", observation_window="unspecified",
                                   eligibility_criteria=f"non-null {group_col}",
                                   censoring_treatment="n/a"),
            segment_summary=pd.DataFrame(),
            narrative=f"Fewer than two distinct values of '{group_col}' present; no comparison possible.",
        )

    crude = crude_churn_rate(df, group_col)
    small_groups = crude[crude["eligible_n"] < min_group_n][group_col].tolist()
    if small_groups:
        warnings.append(f"Groups with < {min_group_n} eligible rows (unreliable rate estimate): {small_groups}")

    cens_report = censoring_report(df, censored_col) if censored_col else {"censoring_data_available": False}
    if cens_report.get("censoring_data_available") and cens_report.get("n_censored_incomplete_observation", 0) > 0:
        warnings.append(cens_report["note"])

    # exposure-adjusted view, if exposure data exists
    exposure_view = None
    exposure_confound_detected = False
    if exposure_col and exposure_col in df.columns and df[exposure_col].notna().any():
        exposure_view = exposure_adjusted_rate(df, group_col, exposure_col)
        # Detect an exposure confound: crude rate ranking disagrees with
        # person-time-adjusted ranking, or exposure itself differs
        # substantially by group.
        exp_by_group = df.groupby(group_col)[exposure_col].median()
        if exp_by_group.max() > 0 and (exp_by_group.min() / exp_by_group.max()) < 0.6:
            warnings.append(
                f"Median exposure ({exposure_col}) differs substantially across groups "
                f"({exp_by_group.to_dict()}) -- crude rate alone is not a fair comparison; "
                "see exposure-adjusted (person-time) rate instead."
            )
            # Beyond the warning: if the crude rates differ substantially
            # between groups but the person-time (exposure-adjusted) rates
            # converge to near-parity, the apparent segment effect is not
            # identifiable from crude rates -- it is fully explained by
            # exposure duration, a competing explanation that has not been
            # ruled out. This must gate the verdict, not just annotate it,
            # or a real exposure confound silently produces a confident
            # (and wrong) segment-effect diagnosis.
            crude_by_group = crude.set_index(group_col)["crude_rate"]
            pt_by_group = exposure_view.set_index(group_col)["persontime_rate"]
            common = [g for g in crude_by_group.index if g in pt_by_group.index]
            if len(common) >= 2:
                crude_vals = crude_by_group.loc[common]
                pt_vals = pt_by_group.loc[common]
                if crude_vals.min() > 0 and pt_vals.min() > 0:
                    crude_ratio = crude_vals.max() / crude_vals.min()
                    pt_ratio = pt_vals.max() / pt_vals.min()
                    if crude_ratio >= 2.0 and pt_ratio <= 1.5:
                        exposure_confound_detected = True
            # A large exposure imbalance is itself an identifiability problem
            # when exposure is one of the supplied confounders and there is no
            # validated stratum-level adjustment. Do not require the adjusted
            # point estimate to collapse before refusing a crude segment claim:
            # a stronger person-time rate can still reflect residual exposure
            # structure rather than a separable segment effect.

    # aggregate two-proportion association test (association only, no causal language)
    aggregate_association = None
    if len(groups) == 2:
        g1, g2 = sorted(groups)
        c = crude.set_index(group_col)
        if g1 in c.index and g2 in c.index and c.loc[g1, "eligible_n"] >= min_group_n and c.loc[g2, "eligible_n"] >= min_group_n:
            table = [
                [c.loc[g1, "events"], c.loc[g1, "eligible_n"] - c.loc[g1, "events"]],
                [c.loc[g2, "events"], c.loc[g2, "eligible_n"] - c.loc[g2, "events"]],
            ]
            chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
            aggregate_association = {
                "group1": g1, "group2": g2,
                "chi2": float(chi2), "p_value": float(p),
                "significant": bool(p < 0.05),
            }

    # confounder stratification
    stratified_checks = []
    confounders_detected = []
    for confounder in known_confounders:
        if confounder not in df.columns or df[confounder].notna().sum() == 0:
            continue
        result = stratified_rate_check(df, group_col, confounder, min_stratum_n=min_group_n)
        stratified_checks.append({"confounder": confounder, **result})
        if result.get("applicable") and result.get("confound_detected"):
            confounders_detected.append(confounder)

    # A continuous duration/confounder can be badly imbalanced while still
    # being unsuitable for discrete stratum checks. In that situation, a crude
    # segment claim remains non-identifiable rather than being upgraded merely
    # because a person-time adjustment points in the same direction.
    for confounder in known_confounders:
        if confounder not in df.columns or confounder == group_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[confounder]):
            continue
        vals = df.groupby(group_col)[confounder].median().dropna()
        if len(vals) >= 2 and vals.min() >= 0 and vals.max() > 0 and (vals.min() / vals.max()) < 0.6:
            check = next((c for c in stratified_checks if c.get("confounder") == confounder), None)
            if check is not None and not check.get("applicable"):
                exposure_confound_detected = True
                warnings.append(
                    f"Known duration/confounder '{confounder}' is strongly imbalanced across groups "
                    "but cannot be validated by applicable stratification; the crude segment claim "
                    "is therefore not identifiable."
                )
                break

    # ---- verdict logic ----
    if aggregate_association is None:
        verdict = ChurnVerdict.INSUFFICIENT_EVIDENCE
        narrative = "Insufficient eligible sample size in one or more groups to test an association."
    elif not aggregate_association["significant"]:
        verdict = ChurnVerdict.INSUFFICIENT_EVIDENCE
        narrative = (
            f"No statistically significant association detected between {group_col} and churn "
            f"(p={aggregate_association['p_value']:.4f})."
        )
    elif confounders_detected or exposure_confound_detected:
        verdict = ChurnVerdict.CONFOUNDED_IDENTIFIABILITY_LIMITED
        if confounders_detected and exposure_confound_detected:
            competing = ', '.join(confounders_detected) + f", {exposure_col} (exposure duration)"
        elif exposure_confound_detected:
            competing = f"{exposure_col} (exposure duration)"
        else:
            competing = ', '.join(confounders_detected)
        narrative = (
            f"An association between {group_col} and churn is observed "
            f"(p={aggregate_association['p_value']:.4f}), but it does not survive stratification/"
            f"adjustment by: {competing}. The segment effect is not identifiable from "
            "this data -- the competing explanation(s) listed have not been "
            "ruled out. This is NOT evidence that the segment intrinsically drives churn risk."
        )
    elif known_confounders and not any(c.get("applicable") for c in stratified_checks):
        # An unresolved known confounder is a competing explanation, not a
        # harmless annotation.  Do not allow a crude association to become a
        # positive segment conclusion merely because the available data cannot
        # support the requested adjustment.
        verdict = ChurnVerdict.CONFOUNDED_IDENTIFIABILITY_LIMITED
        narrative = (
            f"An association between {group_col} and churn is observed "
            f"(p={aggregate_association['p_value']:.4f}), but known confounders "
            f"({', '.join(known_confounders)}) could not be adequately checked "
            "with the available stratum support. The segment effect is not "
            "identifiable from this data; this is not evidence that the segment "
            "intrinsically drives churn risk."
        )
    elif known_confounders:
        verdict = ChurnVerdict.SUPPORTED_SEGMENT_DIFFERENCE
        narrative = (
            f"An association between {group_col} and churn is observed "
            f"(p={aggregate_association['p_value']:.4f}) and survives stratification by "
            f"{', '.join(known_confounders)}. This supports a genuine segment-level difference in "
            "observed churn, evaluated against the confounders checked -- not a proof of causation."
        )
    else:
        verdict = ChurnVerdict.OBSERVED_ASSOCIATION
        narrative = (
            f"An association between {group_col} and churn is observed "
            f"(p={aggregate_association['p_value']:.4f}). No confounders were supplied to check, "
            "so this remains an observed association, not a confirmed segment-specific effect."
        )

    estimand = EstimandSpec(
        numerator="confirmed churn events",
        denominator="eligible population (non-censored outcome known)" if cens_report.get("censoring_data_available")
                    else "all rows with non-null churn_event",
        unit_of_analysis="customer",
        observation_window=f"per-customer via '{exposure_col}'" if exposure_view is not None else "unspecified/assumed uniform",
        eligibility_criteria=f"non-null {REQUIRED_EVENT_COL}",
        censoring_treatment=("censored rows excluded from crude rate; included as partial "
                              "person-time in exposure-adjusted rate")
                             if cens_report.get("censoring_data_available")
                             else "no censoring field available -- crude rate assumes complete observation",
    )

    return ChurnAnalysisResult(
        verdict=verdict,
        estimand=estimand,
        segment_summary=crude,
        aggregate_association=aggregate_association,
        stratified_checks=stratified_checks,
        confounders_detected=confounders_detected,
        narrative=narrative,
        warnings=warnings,
    )


def _require_columns(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) for churn estimand computation: {missing}")
