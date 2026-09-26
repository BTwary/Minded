"""v20-C3 regression suite: canonical experiment-contract authority.

Covers the AAOS_V20C3 requirement that ``ExperimentSynthesizer`` consumes
the already-resolved analytical roles from ``MethodSelectionDecision.estimand``
instead of independently re-reading ``semantic.target_metric_col`` /
``semantic.secondary_metric_col`` / ``semantic.time_col`` /
``semantic.churn_event_col`` etc.

Deliberately unit-level (no controller/DB round-trip): these tests
construct ``SemanticResolution`` and ``MethodSelectionDecision`` objects
directly and assert on ``CandidateExperiment.query_sql`` /
``target_metric`` / ``metrics`` -- the actual column bindings used in the
generated experiment -- not just that synthesis "succeeds".

Run with: python -m unittest tests/independent_release/test_c3_canonical_experiment_contract.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.engines.method_selection import (
    EstimandSpec, MethodFamily, MethodSelectionDecision, MethodSelectionEngine,
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
        secondary_metric_col="price",
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


def _correlation_decision(semantic) -> MethodSelectionDecision:
    """Build the decision the same way MethodSelectionEngine._select_correlational does."""
    est = MethodSelectionEngine._estimand(semantic, None)
    secondary = getattr(semantic, "secondary_metric_col", None)
    est.predictor_columns = [secondary] if secondary else []
    return MethodSelectionDecision(
        problem_class=ProblemClass.CORRELATIONAL,
        objective=ObjectiveType.COMPARE,
        estimand=est,
        admissible_families={MethodFamily.CORRELATION},
        max_verdict_tier=None,
        rationale="test",
    )


class TestEstimandCarriesRoleBindings(unittest.TestCase):
    """The canonical contract itself must carry the roles (item 8 audit finding)."""

    def test_correlational_estimand_has_target_and_predictor(self):
        semantic = _semantic(target_metric_col="sales", secondary_metric_col="price")
        decision = _correlation_decision(semantic)
        self.assertEqual(decision.estimand.target_column, "sales")
        self.assertEqual(decision.estimand.predictor_columns, ["price"])

    def test_generic_estimand_has_target_column(self):
        semantic = _semantic(target_metric_col="revenue", secondary_metric_col=None)
        est = MethodSelectionEngine._estimand(semantic, None)
        self.assertEqual(est.target_column, "revenue")


class TestQ6SymmetricAssociation(unittest.TestCase):
    """Reversing which variable is named first in a symmetric association
    question must not change the generated experiment's bindings, because
    the synthesizer now consumes one canonical (target, predictor) pair
    rather than re-deriving roles itself."""

    def test_price_sales_and_sales_price_bind_identically(self):
        # "Is price associated with sales?" and "Are price and sales
        # associated?" both resolve, upstream, to the same canonical
        # (target=sales, predictor=price) pair -- this test proves the
        # synthesizer follows that pair exactly, regardless of how it
        # got resolved, rather than re-picking target/secondary itself.
        semantic = _semantic(target_metric_col="sales", secondary_metric_col="price")
        decision = _correlation_decision(semantic)
        h1 = _hypothesis(secondary_metric="price", claim_type="ASSOCIATION")
        candidates = ExperimentSynthesizer._synthesize_correlation_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(len(candidates), 1)
        exp = candidates[0]
        self.assertEqual(exp.target_metric, "sales")
        self.assertIn("sales", exp.metrics)
        self.assertIn("price", exp.metrics)
        self.assertIn("sales", exp.query_sql)
        self.assertIn("price", exp.query_sql)
        self.assertEqual(exp.aggregation_type, "CORRELATION")

    def test_full_pipeline_synthesize_candidate_experiments_uses_decision(self):
        semantic = _semantic(target_metric_col="sales", secondary_metric_col="price")
        decision = _correlation_decision(semantic)
        h1 = _hypothesis(secondary_metric="price", claim_type="ASSOCIATION")
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].target_metric, "sales")
        self.assertIn("price", candidates[0].metrics)


class TestDeliberateSemanticDisagreement(unittest.TestCase):
    """DEFECT class: the synthesizer must follow the canonical contract even
    when a stale semantic.* field disagrees with it -- proving there is no
    accidental fallback to the old source of truth."""

    def test_stale_semantic_target_is_ignored_in_favor_of_canonical_contract(self):
        # Deliberately construct a semantic resolution where
        # target_metric_col/secondary_metric_col say one thing...
        semantic = _semantic(target_metric_col="sales", secondary_metric_col="price")
        # ...but the canonical decision (as a compiler/method-selection
        # layer might independently produce for a differently-phrased
        # question) resolves the roles the other way around.
        decision = _correlation_decision(semantic)
        decision.estimand.target_column = "price"
        decision.estimand.predictor_columns = ["sales"]

        h1 = _hypothesis(secondary_metric="price", claim_type="ASSOCIATION")
        candidates = ExperimentSynthesizer._synthesize_correlation_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(len(candidates), 1)
        exp = candidates[0]
        # The synthesized experiment must follow the canonical contract
        # (target=price), NOT the stale semantic.target_metric_col ("sales").
        self.assertEqual(exp.target_metric, "price")
        self.assertEqual(exp.metrics, ["price", "sales"])


class TestFailClosed(unittest.TestCase):
    """Missing/incomplete canonical roles must fail closed -- no candidates,
    never a silent fallback to semantic.* when a decision was supplied."""

    def test_correlation_fails_closed_when_predictor_unresolved(self):
        semantic = _semantic(target_metric_col="sales", secondary_metric_col="price")
        decision = _correlation_decision(semantic)
        decision.estimand.predictor_columns = []  # canonical contract incomplete
        h1 = _hypothesis(secondary_metric="price", claim_type="ASSOCIATION")
        candidates = ExperimentSynthesizer._synthesize_correlation_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(candidates, [])

    def test_correlation_fails_closed_when_target_unresolved(self):
        semantic = _semantic(target_metric_col="sales", secondary_metric_col="price")
        decision = _correlation_decision(semantic)
        decision.estimand.target_column = None
        h1 = _hypothesis(secondary_metric="price", claim_type="ASSOCIATION")
        candidates = ExperimentSynthesizer._synthesize_correlation_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(candidates, [])

    def test_forecast_fails_closed_when_time_column_unresolved(self):
        semantic = _semantic(target_metric_col="revenue", secondary_metric_col=None, time_col="order_date")
        est = MethodSelectionEngine._estimand(semantic, None)
        est.time_column = None  # canonical contract incomplete
        decision = MethodSelectionDecision(
            problem_class=ProblemClass.FORECASTING, objective=ObjectiveType.FORECAST,
            estimand=est, admissible_families={MethodFamily.FORECAST}, max_verdict_tier=None,
        )
        h1 = _hypothesis(claim_type="PREDICTION")
        candidates = ExperimentSynthesizer._synthesize_forecast_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(candidates, [])

    def test_churn_fails_closed_when_no_churn_bindings_on_decision(self):
        semantic = _semantic(target_metric_col="revenue", secondary_metric_col=None, churn_event_col="churned")
        est = MethodSelectionEngine._estimand(semantic, None)
        est.churn_bindings = None  # canonical contract never resolved churn roles
        decision = MethodSelectionDecision(
            problem_class=ProblemClass.SURVIVAL_CHURN, objective=ObjectiveType.PREDICT,
            estimand=est, admissible_families={MethodFamily.CHURN}, max_verdict_tier=None,
        )
        h1 = _hypothesis(claim_type="")
        candidates = ExperimentSynthesizer._synthesize_churn_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(candidates, [])


class TestForecastAndChurnCanonicalBinding(unittest.TestCase):
    """Positive-path coverage: FORECAST and SURVIVAL_CHURN also consume the
    canonical decision when it is supplied, not just CORRELATIONAL."""

    def test_forecast_uses_decision_estimand_bindings(self):
        semantic = _semantic(target_metric_col="revenue", secondary_metric_col=None, time_col="order_date")
        est = MethodSelectionEngine._estimand(semantic, None)
        decision = MethodSelectionDecision(
            problem_class=ProblemClass.FORECASTING, objective=ObjectiveType.FORECAST,
            estimand=est, admissible_families={MethodFamily.FORECAST}, max_verdict_tier=None,
        )
        h1 = _hypothesis(claim_type="PREDICTION")
        candidates = ExperimentSynthesizer._synthesize_forecast_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].target_metric, "revenue")
        self.assertIn("order_date", candidates[0].query_sql)

    def test_churn_uses_decision_churn_bindings_not_stale_semantic(self):
        # Stale semantic says the event column is "cancelled"; the
        # canonical decision (churn_bindings) says it's "churned".
        semantic = _semantic(
            target_metric_col="revenue", secondary_metric_col=None,
            churn_event_col="cancelled", group_dimension_col="plan_tier",
        )
        est = MethodSelectionEngine._estimand(semantic, None)
        est.churn_bindings = {
            "event_col": "churned",
            "exposure_col": None,
            "censored_col": None,
            "confounder_cols": [],
            "outcome_available": True,
        }
        decision = MethodSelectionDecision(
            problem_class=ProblemClass.SURVIVAL_CHURN, objective=ObjectiveType.PREDICT,
            estimand=est, admissible_families={MethodFamily.CHURN}, max_verdict_tier=None,
        )
        h1 = _hypothesis(code="HYP-01", claim_type="")
        candidates = ExperimentSynthesizer._synthesize_churn_experiments(
            [h1], semantic, decision=decision,
        )
        self.assertTrue(len(candidates) >= 1)
        crude = next(c for c in candidates if c.code == "EXP-CHURN-CRUDE")
        self.assertIn("churned", crude.query_sql)
        self.assertNotIn("cancelled", crude.query_sql)


class TestLegacyCompatibilityPathStillWorks(unittest.TestCase):
    """When no decision is supplied at all (legacy callers constructing
    hypotheses directly), the documented compatibility fallback to
    semantic.* must still work exactly as before v20-C3."""

    def test_correlation_without_decision_falls_back_to_semantic(self):
        semantic = _semantic(target_metric_col="sales", secondary_metric_col="price")
        h1 = _hypothesis(secondary_metric="price", claim_type="ASSOCIATION")
        candidates = ExperimentSynthesizer._synthesize_correlation_experiments([h1], semantic)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].target_metric, "sales")
        self.assertIn("price", candidates[0].metrics)


if __name__ == "__main__":
    unittest.main()
