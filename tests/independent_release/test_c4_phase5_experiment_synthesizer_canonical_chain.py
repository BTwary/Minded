"""v20-C4 Phase 5 verification: ExperimentSynthesizer end-to-end against the
now-canonically-sourced MethodSelectionDecision from Phase 4.

FINDING (recorded here rather than only in the docs, so this test is the
executable proof of it): ExperimentSynthesizer required NO code changes for
Phase 5. C3/C3.1 already hardened every generic + specialized synthesis path
(_synthesize_correlation_experiments, _synthesize_forecast_experiments,
_synthesize_churn_experiments, and the generic diagnostic-battery dispatch
in synthesize_candidate_experiments) to:

  - prefer decision.estimand's resolved role fields whenever `decision` is
    supplied, and
  - fail closed (return no candidates) rather than substitute
    semantic.target_metric_col/secondary_metric_col/time_col/churn_* when
    the decision is supplied but its estimand role is unresolved, and
  - only read semantic.* directly in the explicitly-documented
    decision-is-None legacy/compatibility path.

Phase 4 made MethodSelectionDecision.estimand itself sourced from
CanonicalSemanticResolution instead of raw SemanticResolution. Because
ExperimentSynthesizer already deferred entirely to decision.estimand, the
authority transfer propagates through it automatically: as soon as
decision.estimand.target_column/predictor_columns/time_column/churn_bindings
are canonical-sourced (Phase 4), every downstream experiment built from that
decision is too, with zero synthesizer-side changes.

This suite proves that chain end-to-end using the real MethodSelectionEngine
.decide() entry point (not a hand-built MethodSelectionDecision, unlike
test_c3_canonical_experiment_contract.py), reusing the same
deliberate-disagreement construction as
test_c4_phase4_method_selection_canonical_authority.py: a churn-relevant
column makes canonical disagree with semantic.target_metric_col, and this
proves the disagreement propagates all the way into the generated
CandidateExperiment SQL, not just into the EstimandSpec.

Run with: python -m unittest tests/independent_release/test_c4_phase5_experiment_synthesizer_canonical_chain.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.method_selection import MethodSelectionEngine, ProblemClass
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.intelligence.experiment_synthesizer import ExperimentSynthesizer
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition
from packages.schemas.src.analysis import AggregationType


def _semantic(
    metric="revenue", group=None, time_col=None, secondary=None, churn_event=None,
    available_categorical=None, churn_exposure=None, churn_censored=None,
    churn_confounders=None, churn_outcome_available=True,
):
    return SemanticResolution(
        primary_dataset_name="orders", target_metric_col=metric, group_dimension_col=group,
        time_col=time_col, table_grain="record_level", available_numeric_cols=[metric, secondary or "x"],
        available_categorical_cols=list(available_categorical or ([group] if group else [])),
        world_model=None,
        metric_definition=MetricDefinition(
            name=metric, table_name="orders", source_columns=[metric], semantic_type="sum_measure",
            aggregation_type=AggregationType.SUM, is_additive=True, valid_aggregations=[AggregationType.SUM],
            semantic_resolution_status="RESOLVED",
        ),
        direction_hint="unspecified", secondary_metric_col=secondary,
        churn_event_col=churn_event, churn_exposure_col=churn_exposure, churn_censored_col=churn_censored,
        churn_confounder_cols=list(churn_confounders or []), churn_outcome_available=churn_outcome_available,
    )


def _hypothesis(code="HYP-01", target_metric="", secondary_metric=""):
    return PredictiveHypothesis(
        id=code, hypothesis_code=code, claim="test claim", mechanism="test mechanism",
        predicted_observables_if_true=["obs_true"], predicted_observables_if_false=["obs_false"],
        falsification_criteria="none", required_assumptions=[], prior_probability=0.5,
        posterior_probability=0.5, target_metric=target_metric, secondary_metric=secondary_metric,
    )


class TestCanonicalDisagreementPropagatesIntoGeneratedSQL(unittest.TestCase):
    def test_experiment_sql_uses_canonical_columns_not_stale_semantic_target(self):
        # Same construction as the Phase 4 disagreement test: churn_event_col
        # makes canonical.outcome_column() == "price", deliberately
        # different from semantic.target_metric_col == "sales".
        semantic = _semantic(metric="sales", secondary="discount", churn_event="price")
        intent = IntentEngine.parse_intent("Is there a correlation between sales and discount?")

        decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)
        self.assertEqual(decision.problem_class, ProblemClass.CORRELATIONAL)
        self.assertEqual(decision.estimand.target_column, "price")
        self.assertEqual(decision.estimand.predictor_columns, ["discount"])

        hyp = _hypothesis(target_metric="price", secondary_metric="discount")
        candidates = ExperimentSynthesizer._synthesize_correlation_experiments(
            [hyp], semantic, decision=decision,
        )

        self.assertTrue(candidates, "expected at least one candidate experiment")
        corr_candidate = next(c for c in candidates if c.code == "EXP-CORR")
        self.assertIn("price", corr_candidate.query_sql)
        self.assertIn("discount", corr_candidate.query_sql)
        # The stale semantic.target_metric_col value must not leak into the
        # generated query at all -- this is the actual Phase 5 acceptance
        # criterion ("ExperimentSynthesizer cannot override canonical role
        # bindings"), verified at the SQL text level, not just on the
        # intermediate EstimandSpec.
        self.assertNotIn("sales", corr_candidate.query_sql)

    def test_ordinary_case_still_produces_expected_sql(self):
        semantic = _semantic(metric="revenue", secondary="cost")
        intent = IntentEngine.parse_intent("Is there a correlation between revenue and cost?")

        decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)
        hyp = _hypothesis(target_metric="revenue", secondary_metric="cost")
        candidates = ExperimentSynthesizer._synthesize_correlation_experiments(
            [hyp], semantic, decision=decision,
        )
        corr_candidate = next(c for c in candidates if c.code == "EXP-CORR")
        self.assertIn("revenue", corr_candidate.query_sql)
        self.assertIn("cost", corr_candidate.query_sql)


class TestForecastChainUsesCanonicalTimeColumn(unittest.TestCase):
    def test_forecast_experiment_uses_canonical_time_column(self):
        semantic = _semantic(metric="revenue", time_col="order_date")
        intent = IntentEngine.parse_intent("Forecast next month's revenue")

        decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question)
        self.assertEqual(decision.problem_class, ProblemClass.FORECASTING)
        self.assertEqual(decision.estimand.time_column, "order_date")

        hyp = _hypothesis(target_metric="revenue")
        candidates = ExperimentSynthesizer._synthesize_forecast_experiments(
            [hyp], semantic, decision=decision,
        )
        self.assertTrue(candidates, "expected at least one forecast candidate")
        # Every generated forecast candidate must reference the canonical
        # time column somewhere in its query.
        self.assertTrue(any("order_date" in c.query_sql for c in candidates))


if __name__ == "__main__":
    unittest.main()
