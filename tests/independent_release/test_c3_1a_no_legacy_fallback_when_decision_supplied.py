"""v20-C3.1a regression suite: decision-supplied legacy fallback closure.

C3.1 (see test_c3_1_canonical_dispatch_hardening.py) hardened the family
dispatch in ``ExperimentSynthesizer.synthesize_candidate_experiments`` so a
canonical decision's ``method_family`` -- not a raw ``semantic.*`` check --
decides whether CORRELATION / FORECAST / CHURN is reached.

A final audit found a *second*, lower leak in the very same function: after
the canonical-family dispatch falls through (e.g. because the decision's
family is DIAGNOSTIC_BATTERY), three defensive hypothesis-based fallback
branches still ran unconditionally:

    if hypotheses and any(...secondary_metric...):
        _synthesize_correlation_experiments(..., )   # no decision=decision
    if hypotheses and any(...PREDICTION...):
        _synthesize_forecast_experiments(..., )      # no decision=decision
    if hypotheses and semantic.churn_event_col:
        _synthesize_churn_experiments(..., )         # no decision=decision

Because these calls omitted ``decision=decision``, the specialized helper
received ``decision=None`` and legally entered its own legacy/no-decision
branch, which reads raw ``semantic.secondary_metric_col`` /
``semantic.time_col`` / ``semantic.churn_event_col`` directly -- silently
reinterpreting a hypothesis using semantic fields even though a canonical
decision was supplied and had already selected a *different* analytical
family. This suite proves that leak is closed:

1. When a decision is supplied, none of the three fallbacks may run at all
   (they must fail closed, or route through the generic decision path
   instead) -- even when the stale semantic field and a matching
   hypothesis shape are both present.
2. The canonical specialized-family paths (CORRELATION / FORECAST / CHURN)
   are unaffected by this change.
3. The legacy ``decision is None`` compatibility path still works exactly
   as before.
4. The churn presence-check candidate's cosmetic labels no longer read raw
   ``semantic.group_dimension_col`` / ``semantic.target_metric_col`` once a
   decision is supplied.

Run with: python -m unittest tests/independent_release/test_c3_1a_no_legacy_fallback_when_decision_supplied.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.engines.method_selection import (
    MethodFamily, MethodSelectionDecision, MethodSelectionEngine,
    ObjectiveType, ProblemClass,
)
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.intelligence.experiment_synthesizer import ExperimentSynthesizer
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition
from packages.schemas.src.analysis import AggregationType


def _semantic(**overrides):
    base = dict(
        primary_dataset_name="orders",
        target_metric_col="sales",
        group_dimension_col=None,
        time_col=None,
        table_grain="record_level",
        available_numeric_cols=["sales", "price"],
        available_categorical_cols=[],
        world_model=None,
        metric_definition=MetricDefinition(
            name="sales", table_name="orders", source_columns=["sales"],
            semantic_type="sum_measure", aggregation_type=AggregationType.SUM,
            is_additive=True, valid_aggregations=[AggregationType.SUM],
            semantic_resolution_status="RESOLVED",
        ),
        direction_hint="unspecified",
        secondary_metric_col=None,
        churn_event_col=None, churn_exposure_col=None, churn_censored_col=None,
        churn_confounder_cols=[], churn_outcome_available=True,
    )
    base.update(overrides)
    return SemanticResolution(**base)


def _hypothesis(code="HYP-01", secondary_metric="", claim_type="", target_metric="sales",
                is_counter_hypothesis=False):
    return PredictiveHypothesis(
        id=code, hypothesis_code=code, claim="test claim", mechanism="test mechanism",
        predicted_observables_if_true=["obs_true"], predicted_observables_if_false=["obs_false"],
        falsification_criteria="none", required_assumptions=[], prior_probability=0.5,
        posterior_probability=0.5, target_metric=target_metric, secondary_metric=secondary_metric,
        claim_type=claim_type, is_counter_hypothesis=is_counter_hypothesis,
    )


def _diagnostic_decision(target_column, comparison_dimension=None):
    """A canonical decision whose family is DIAGNOSTIC_BATTERY (not
    CORRELATION/FORECAST/CHURN) with a resolved target_column, so the
    dispatch falls through the family-specific branches and reaches the
    three defensive fallbacks under test -- exactly the scenario the C3.1a
    leak allowed to bypass the decision."""
    est = MethodSelectionEngine._estimand(
        _semantic(target_metric_col=target_column, group_dimension_col=comparison_dimension), None,
    )
    est.target_column = target_column
    return MethodSelectionDecision(
        problem_class=ProblemClass.DESCRIPTIVE,
        objective=ObjectiveType.DESCRIBE,
        estimand=est,
        admissible_families={MethodFamily.DIAGNOSTIC_BATTERY},
        max_verdict_tier=None,
        rationale="test",
    )


def _correlation_decision(target_column, predictor_columns):
    est = MethodSelectionEngine._estimand(_semantic(target_metric_col=target_column), None)
    est.target_column = target_column
    est.predictor_columns = list(predictor_columns)
    return MethodSelectionDecision(
        problem_class=ProblemClass.CORRELATIONAL,
        objective=ObjectiveType.COMPARE,
        estimand=est,
        admissible_families={MethodFamily.CORRELATION},
        max_verdict_tier=None,
        rationale="test",
    )


def _forecast_decision(target_column, time_column):
    est = MethodSelectionEngine._estimand(_semantic(target_metric_col=target_column), None)
    est.target_column = target_column
    est.time_column = time_column
    return MethodSelectionDecision(
        problem_class=ProblemClass.FORECASTING,
        objective=ObjectiveType.FORECAST,
        estimand=est,
        admissible_families={MethodFamily.FORECAST},
        max_verdict_tier=None,
        rationale="test",
    )


def _churn_decision(target_column, churn_bindings, comparison_dimension=None):
    est = MethodSelectionEngine._estimand(
        _semantic(target_metric_col=target_column, group_dimension_col=comparison_dimension), None,
    )
    est.target_column = target_column
    est.churn_bindings = churn_bindings
    return MethodSelectionDecision(
        problem_class=ProblemClass.SURVIVAL_CHURN,
        objective=ObjectiveType.PREDICT,
        estimand=est,
        admissible_families={MethodFamily.CHURN},
        max_verdict_tier=None,
        rationale="test",
    )


class TestFallbacksCannotBypassDecision(unittest.TestCase):
    """C3.1a findings 1-3: none of the three defensive hypothesis-shape
    fallbacks may reach a specialized synthesizer once a decision is
    supplied, even when the stale semantic field and a matching hypothesis
    shape are both present and the decision's own family is a non-matching
    one (DIAGNOSTIC_BATTERY)."""

    def test_1_correlation_fallback_cannot_bypass_decision(self):
        decision = _diagnostic_decision("sales")
        semantic = _semantic(target_metric_col="sales", secondary_metric_col="stale_price")
        h1 = _hypothesis(secondary_metric="price", claim_type="")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        # Must not have entered the correlation legacy path.
        self.assertFalse(any(c.code == "EXP-CORR" for c in candidates))
        self.assertFalse(any("stale_price" in (c.query_sql or "") for c in candidates))
        # Must remain on the canonical diagnostic decision path instead.
        self.assertTrue(len(candidates) >= 1)
        self.assertTrue(all(c.target_metric == "sales" for c in candidates))

    def test_2_forecast_fallback_cannot_bypass_decision(self):
        decision = _diagnostic_decision("sales")
        semantic = _semantic(target_metric_col="sales", time_col="stale_date")
        h1 = _hypothesis(claim_type="PREDICTION")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertFalse(any(c.code == "EXP-FORECAST-TREND" for c in candidates))
        self.assertFalse(any("stale_date" in (c.query_sql or "") for c in candidates))
        self.assertTrue(len(candidates) >= 1)
        self.assertTrue(all(c.target_metric == "sales" for c in candidates))

    def test_3_churn_fallback_cannot_bypass_decision(self):
        decision = _diagnostic_decision("sales", comparison_dimension="plan_tier")
        semantic = _semantic(
            target_metric_col="sales", churn_event_col="stale_event",
            group_dimension_col="plan_tier",
            available_categorical_cols=["plan_tier"],
        )
        # A manually constructed churn-like hypothesis: nothing about it
        # names churn explicitly (PredictiveHypothesis has no churn-only
        # field), the old leak was triggered purely by
        # semantic.churn_event_col being set alongside any hypothesis.
        h1 = _hypothesis(code="HYP-01", claim_type="", target_metric="sales")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertFalse(any(str(c.code).startswith("EXP-CHURN") for c in candidates))
        self.assertFalse(any("stale_event" in (c.query_sql or "") for c in candidates))
        self.assertTrue(len(candidates) >= 1)
        self.assertTrue(all(c.target_metric == "sales" for c in candidates))


class TestCanonicalSpecializedDecisionsUnaffected(unittest.TestCase):
    """C3.1a finding 4: the actual canonical CORRELATION / FORECAST / CHURN
    decision paths must remain unchanged by the fallback-gating fix."""

    def test_4a_canonical_correlation_still_works(self):
        decision = _correlation_decision("sales", ["price"])
        semantic = _semantic(target_metric_col="sales", secondary_metric_col=None)
        h1 = _hypothesis(secondary_metric="", claim_type="ASSOCIATION")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].code, "EXP-CORR")
        self.assertEqual(candidates[0].target_metric, "sales")
        self.assertIn("price", candidates[0].metrics)

    def test_4b_canonical_forecast_still_works(self):
        decision = _forecast_decision("revenue", "order_date")
        semantic = _semantic(target_metric_col="revenue", time_col=None)
        h1 = _hypothesis(claim_type="PREDICTION", target_metric="revenue")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].code, "EXP-FORECAST-TREND")
        self.assertEqual(candidates[0].target_metric, "revenue")
        self.assertIn("order_date", candidates[0].query_sql)

    def test_4c_canonical_churn_still_works(self):
        decision = _churn_decision(
            "revenue",
            {
                "event_col": "actual_event", "exposure_col": None, "censored_col": None,
                "confounder_cols": [], "outcome_available": True,
            },
            comparison_dimension="plan_tier",
        )
        semantic = _semantic(
            target_metric_col="revenue", churn_event_col="old_event",
            group_dimension_col="plan_tier",
        )
        h1 = _hypothesis(code="HYP-01", claim_type="")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertTrue(len(candidates) >= 1)
        crude = next(c for c in candidates if c.code == "EXP-CHURN-CRUDE")
        self.assertIn("actual_event", crude.query_sql)
        self.assertNotIn("old_event", crude.query_sql)

    def test_4d_canonical_churn_presence_check_uses_decision_labels_not_stale_semantic(self):
        # decision has no comparison_dimension / target_column mirrored onto
        # its estimand for churn labelling purposes here -- exercise the
        # "canonical values unavailable -> neutral labels" branch.
        est = MethodSelectionEngine._estimand(_semantic(target_metric_col="revenue"), None)
        est.target_column = None
        est.churn_bindings = {
            "event_col": None, "exposure_col": None, "censored_col": None,
            "confounder_cols": [], "outcome_available": False,
        }
        decision = MethodSelectionDecision(
            problem_class=ProblemClass.SURVIVAL_CHURN,
            objective=ObjectiveType.PREDICT,
            estimand=est,
            admissible_families={MethodFamily.CHURN},
            max_verdict_tier=None,
            rationale="test",
        )
        semantic = _semantic(
            target_metric_col="stale_metric", group_dimension_col="stale_dim",
            churn_event_col=None, churn_outcome_available=False,
        )
        h1 = _hypothesis(code="HYP-01", claim_type="")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        presence = next(c for c in candidates if c.code == "EXP-CHURN-PRESENCE-CHECK")
        self.assertNotEqual(presence.target_metric, "stale_metric")
        self.assertNotEqual(presence.target_dimension, "stale_dim")
        self.assertEqual(presence.target_metric, "count")
        self.assertEqual(presence.target_dimension, "")


class TestLegacyCompatibilityWhenDecisionAbsent(unittest.TestCase):
    """C3.1a finding 5: the legacy direct-call behavior (decision is None)
    must remain fully functional -- this fix is scoped to the
    decision-supplied path only, not a compatibility-breaking rewrite."""

    def test_5a_legacy_correlation_fallback_still_works_when_decision_is_none(self):
        semantic = _semantic(target_metric_col="sales", secondary_metric_col="price")
        h1 = _hypothesis(secondary_metric="price", claim_type="ASSOCIATION")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=None,
        )
        self.assertTrue(any(c.code == "EXP-CORR" for c in candidates))

    def test_5b_legacy_forecast_fallback_still_works_when_decision_is_none(self):
        semantic = _semantic(target_metric_col="revenue", time_col="order_date")
        h1 = _hypothesis(claim_type="PREDICTION", target_metric="revenue")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=None,
        )
        self.assertTrue(any(c.code == "EXP-FORECAST-TREND" for c in candidates))

    def test_5c_legacy_churn_fallback_still_works_when_decision_is_none(self):
        semantic = _semantic(
            target_metric_col="revenue", churn_event_col="event_flag",
            group_dimension_col="plan_tier",
        )
        h1 = _hypothesis(code="HYP-01", claim_type="")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=None,
        )
        self.assertTrue(any(str(c.code).startswith("EXP-CHURN") for c in candidates))


if __name__ == "__main__":
    unittest.main()
