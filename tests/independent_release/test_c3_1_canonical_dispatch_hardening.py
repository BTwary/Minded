"""v20-C3.1 regression suite: canonical dispatch hardening.

C3 (see test_c3_canonical_experiment_contract.py) made the individual
``_synthesize_correlation_experiments`` / ``_synthesize_forecast_experiments``
/ ``_synthesize_churn_experiments`` helpers consume
``MethodSelectionDecision.estimand`` instead of re-reading
``semantic.secondary_metric_col`` / ``semantic.time_col`` / ``semantic.churn_*``.

The C3.1 audit found that the *dispatch* site one level up --
``ExperimentSynthesizer.synthesize_candidate_experiments`` -- still gated
whether those canonical-aware helpers were even called on the very same
stale semantic fields:

    if family == MethodFamily.CORRELATION and semantic.secondary_metric_col:
    if family == MethodFamily.FORECAST and semantic.time_col:

So a valid canonical decision (family=CORRELATION/FORECAST, estimand fully
bound) could still be silently rerouted to the generic diagnostic battery
whenever the corresponding semantic.* field happened to be empty or stale
-- exactly the defect class C3 was meant to close, just one call frame
higher up. This suite proves that gap is closed:

1. Dispatch reaches the specialized synthesizer purely off
   ``decision.method_family``, never off a raw ``semantic.*`` check.
2. The remaining generic (diagnostic-battery) path also stops silently
   substituting ``semantic.target_metric_col`` for an unresolved
   ``decision.estimand.target_column`` when a decision is supplied --
   it now fails closed, matching the same "decision supplied -> canonical
   contract only" rule the specialized paths already followed.

Run with: python -m unittest tests/independent_release/test_c3_1_canonical_dispatch_hardening.py
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


def _hypothesis(code="HYP-01", secondary_metric="", claim_type="", target_metric="sales"):
    return PredictiveHypothesis(
        id=code, hypothesis_code=code, claim="test claim", mechanism="test mechanism",
        predicted_observables_if_true=["obs_true"], predicted_observables_if_false=["obs_false"],
        falsification_criteria="none", required_assumptions=[], prior_probability=0.5,
        posterior_probability=0.5, target_metric=target_metric, secondary_metric=secondary_metric,
        claim_type=claim_type,
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


class TestDispatchNotSuppressedByStaleSemantic(unittest.TestCase):
    """DEFECT class (C3.1 finding 1/4): a valid canonical decision must
    reach its specialized synthesizer via synthesize_candidate_experiments
    even when the corresponding semantic.* role field is None/stale --
    dispatch must key off decision.method_family alone."""

    def test_correlation_dispatch_reached_when_semantic_secondary_metric_col_is_none(self):
        # Stale/empty semantic field: the previous dispatch guard
        # (`family == CORRELATION and semantic.secondary_metric_col`) would
        # have silently skipped the specialized synthesizer entirely here.
        semantic = _semantic(target_metric_col="sales", secondary_metric_col=None)
        decision = _correlation_decision("sales", ["price"])
        h1 = _hypothesis(secondary_metric="", claim_type="ASSOCIATION")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].code, "EXP-CORR")
        self.assertEqual(candidates[0].target_metric, "sales")
        self.assertIn("price", candidates[0].metrics)

    def test_forecast_dispatch_reached_when_semantic_time_col_is_none(self):
        semantic = _semantic(target_metric_col="revenue", time_col=None)
        decision = _forecast_decision("revenue", "order_date")
        h1 = _hypothesis(claim_type="PREDICTION", target_metric="revenue")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].code, "EXP-FORECAST-TREND")
        self.assertEqual(candidates[0].target_metric, "revenue")
        self.assertIn("order_date", candidates[0].query_sql)

    def test_churn_dispatch_uses_canonical_binding_not_stale_semantic_event_col(self):
        semantic = _semantic(
            target_metric_col="revenue", churn_event_col="old_event",
            group_dimension_col="plan_tier",
        )
        decision = _churn_decision(
            "revenue",
            {
                "event_col": "actual_event",
                "exposure_col": None,
                "censored_col": None,
                "confounder_cols": [],
                "outcome_available": True,
            },
            comparison_dimension="plan_tier",
        )
        h1 = _hypothesis(code="HYP-01", claim_type="")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertTrue(len(candidates) >= 1)
        crude = next(c for c in candidates if c.code == "EXP-CHURN-CRUDE")
        self.assertIn("actual_event", crude.query_sql)
        self.assertNotIn("old_event", crude.query_sql)

    def test_correlation_dispatch_fails_closed_not_falls_through_to_generic_battery(self):
        # When the canonical decision itself is incomplete, dispatch must
        # still route through the CORRELATION synthesizer (which fails
        # closed and returns []) rather than falling through to the
        # generic diagnostic battery below and silently producing an
        # unrelated concentration/ANOVA-style experiment instead.
        semantic = _semantic(target_metric_col="sales", secondary_metric_col="price")
        decision = _correlation_decision("sales", [])  # predictor unresolved
        h1 = _hypothesis(secondary_metric="price", claim_type="ASSOCIATION")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(candidates, [])


class TestGenericDiagnosticPathCanonicalBinding(unittest.TestCase):
    """C3.1 finding 2: the generic (diagnostic-battery) fallback path must
    also stop substituting semantic.target_metric_col when a decision is
    supplied but its canonical target_column is unresolved."""

    def _generic_decision(self, target_column):
        # Build the estimand from a validly-resolvable semantic (target_metric_col
        # must be a real string for MethodSelectionEngine._metric_ref's MetricRef),
        # then overwrite target_column afterward -- mirrors how the existing C3
        # suite constructs a "canonical contract present but incomplete" decision
        # (see TestFailClosed.test_correlation_fails_closed_when_target_unresolved).
        est = MethodSelectionEngine._estimand(_semantic(target_metric_col="sales"), None)
        est.target_column = target_column
        return MethodSelectionDecision(
            problem_class=ProblemClass.DESCRIPTIVE,
            objective=ObjectiveType.DESCRIBE,
            estimand=est,
            admissible_families={MethodFamily.DIAGNOSTIC_BATTERY},
            max_verdict_tier=None,
            rationale="test",
        )

    def test_generic_path_fails_closed_when_decision_supplied_but_target_column_unresolved(self):
        semantic = _semantic(target_metric_col="sales")
        decision = self._generic_decision(None)
        h1 = _hypothesis(claim_type="")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(candidates, [])

    def test_generic_path_follows_canonical_target_over_stale_semantic(self):
        # semantic says "sales"; the canonical decision (as an independently
        # resolved compiler path might produce) says "revenue" instead.
        semantic = _semantic(target_metric_col="sales")
        decision = self._generic_decision("revenue")
        h1 = _hypothesis(claim_type="", target_metric="revenue")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertTrue(len(candidates) >= 1)
        self.assertTrue(all(c.target_metric == "revenue" for c in candidates))
        self.assertFalse(any(c.target_metric == "sales" for c in candidates))

    def test_generic_path_legacy_no_decision_still_uses_semantic(self):
        semantic = _semantic(target_metric_col="sales")
        h1 = _hypothesis(claim_type="")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=None,
        )
        self.assertTrue(len(candidates) >= 1)
        self.assertTrue(all(c.target_metric == "sales" for c in candidates))


if __name__ == "__main__":
    unittest.main()
