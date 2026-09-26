import unittest
import pandas as pd

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.method_selection import (
    MethodSelectionEngine, ProblemClass, MethodFamily, VERDICT_TIER_STATISTICALLY_SIGNIFICANT,
)
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition, MetricSemanticsResolver
from packages.schemas.src.analysis import AggregationType, EpistemicClaimType


class TestMethodSelectionSpine(unittest.TestCase):
    def _semantic(self, metric="revenue", group="region", time_col=None, secondary=None, churn_event=None, md=None):
        return SemanticResolution(
            primary_dataset_name="orders", target_metric_col=metric, group_dimension_col=group,
            time_col=time_col, table_grain="record_level", available_numeric_cols=[metric],
            available_categorical_cols=[group] if group else [], world_model=None,
            metric_definition=md or MetricDefinition(
                name=metric, table_name="orders", source_columns=[metric], semantic_type="sum_measure",
                aggregation_type=AggregationType.SUM, is_additive=True, valid_aggregations=[AggregationType.SUM],
                semantic_resolution_status="RESOLVED"),
            direction_hint="unspecified", secondary_metric_col=secondary,
            churn_event_col=churn_event, churn_exposure_col=None, churn_censored_col=None,
            churn_confounder_cols=[], churn_outcome_available=True,
        )

    def test_correlation(self):
        intent = IntentEngine.parse_intent("Is there a correlation between revenue and cost?")
        d = MethodSelectionEngine.decide(intent, self._semantic(secondary="cost"), intent.raw_question)
        self.assertEqual(d.problem_class, ProblemClass.CORRELATIONAL)
        self.assertEqual(d.method_family, MethodFamily.CORRELATION)
        self.assertEqual(d.claim_type, EpistemicClaimType.ASSOCIATION)
        self.assertEqual(d.max_verdict_tier, VERDICT_TIER_STATISTICALLY_SIGNIFICANT)

    def test_forecast(self):
        intent = IntentEngine.parse_intent("Forecast next month's revenue")
        d = MethodSelectionEngine.decide(intent, self._semantic(time_col="order_date"), intent.raw_question)
        self.assertEqual(d.problem_class, ProblemClass.FORECASTING)
        self.assertEqual(d.method_family, MethodFamily.FORECAST)
        self.assertEqual(d.claim_type, EpistemicClaimType.PREDICTION)

    def test_root_cause_is_not_automatically_causal(self):
        intent = IntentEngine.parse_intent("What is the root cause of the revenue drop?")
        d = MethodSelectionEngine.decide(intent, self._semantic(), intent.raw_question)
        self.assertEqual(d.problem_class, ProblemClass.DIAGNOSTIC)
        self.assertIsNone(d.causal_intent)

    def test_explicit_causal_never_invents_graph(self):
        intent = IntentEngine.parse_intent("What is the causal effect of region on revenue?")
        d = MethodSelectionEngine.decide(intent, self._semantic(), intent.raw_question)
        self.assertEqual(d.problem_class, ProblemClass.CAUSAL)
        self.assertIsNotNone(d.causal_intent)
        self.assertTrue(d.causal_eligible)
        self.assertEqual(d.max_verdict_tier, VERDICT_TIER_STATISTICALLY_SIGNIFICANT)

    def test_defect_016_continuous_churn_metric(self):
        md = MetricDefinition(
            name="churn_revenue", table_name="orders", source_columns=["churn_revenue"],
            semantic_type="sum_measure", aggregation_type=AggregationType.SUM,
            is_additive=True, valid_aggregations=[AggregationType.SUM], semantic_resolution_status="RESOLVED",
        )
        intent = IntentEngine.parse_intent("Why did churn_revenue fall?")
        d = MethodSelectionEngine.decide(intent, self._semantic(metric="churn_revenue", md=md), intent.raw_question)
        self.assertEqual(d.problem_class, ProblemClass.DIAGNOSTIC)
        self.assertNotEqual(d.method_family, MethodFamily.CHURN)

    def test_fallback(self):
        d = MethodSelectionEngine.decide(None, None, "anything")
        self.assertTrue(d.fallback_used)
        self.assertEqual(d.problem_class, ProblemClass.FALLBACK)


class TestForecastMetricSemantics(unittest.TestCase):
    def test_rate_expression_is_not_sum(self):
        md = MetricDefinition(
            name="conversion_rate", table_name="orders", source_columns=["conversion_rate", "conversions", "sessions"],
            semantic_type="rate", aggregation_type=AggregationType.RATE,
            numerator_column="conversions", denominator_column="sessions", is_additive=False,
            is_compositional=True, valid_aggregations=[AggregationType.RATE], semantic_resolution_status="RESOLVED",
        )
        expr = MetricSemanticsResolver.sql_aggregation_expression(md, "conversion_rate")
        self.assertIn("SUM(conversions)", expr)
        self.assertIn("NULLIF(SUM(sessions), 0)", expr)


if __name__ == "__main__":
    unittest.main()
