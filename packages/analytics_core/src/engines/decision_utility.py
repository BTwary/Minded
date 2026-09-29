"""DecisionUtilityEngine: the single authoritative source of expected-utility math.

Audit P1 fix ("DecisionUtilityEngine still is not actually the
controller authority"): the schemas for ExpectedUtilityCalculation /
DecisionRecommendationSchema already existed
(packages.schemas.src.decision, packages.schemas.src.analysis), but no
engine actually owned producing them -- the controller computed
gain/risk/net inline, so the codebase had a conceptual duplication
between "the decision-utility concept" and "the controller's own copy
of the math". Per the one-authoritative-implementation-per-
responsibility principle, this engine is now the ONLY place that
computes expected_gain_metric / downside_risk_metric /
net_expected_utility. The controller orchestrates (decides WHETHER a
recommendation is warranted and what to call it) but delegates the
utility math itself here:

    Evidence -> DecisionUtilityEngine -> ExpectedUtilityCalculation -> DecisionRecommendationRecord
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class UtilityResult:
    expected_gain_metric: float
    downside_risk_metric: float
    net_expected_utility: float
    utility_function_description: str


class DecisionUtilityEngine:
    """Computes grounded expected utility from empirically-observed signals.

    Deliberately does NOT invent a dollar-value gain: no verified
    financial multiplier exists for an arbitrary metric, so utility is
    expressed in the same empirically-grounded unit already computed
    elsewhere in the investigation -- percentage points of outcome
    variance explained (eta-squared) -- weighted by how likely the
    diagnosis actually is (the leading hypothesis's verified posterior
    probability). This is a heuristic OPERATIONAL utility, not a
    general decision-theoretic model, which `utility_function_description`
    states explicitly rather than implying financial precision.
    """

    DESCRIPTION = (
        "Empirical signal-grounded utility, in percentage points of "
        "outcome variance explained (eta-squared) weighted by verified "
        "posterior probability: gain = variance_explained * P(correct), "
        "risk = variance_explained * P(incorrect), net = gain - risk. "
        "No unverified financial multipliers applied."
    )

    @classmethod
    def compute(
        cls,
        variance_explained_pct: Optional[float],
        posterior_probability: float,
    ) -> UtilityResult:
        var_pts = variance_explained_pct if variance_explained_pct is not None else 0.0
        gain = round(var_pts * posterior_probability, 4)
        risk = round(var_pts * (1.0 - posterior_probability), 4)
        net = round(gain - risk, 4)
        return UtilityResult(
            expected_gain_metric=gain,
            downside_risk_metric=risk,
            net_expected_utility=net,
            utility_function_description=cls.DESCRIPTION,
        )
