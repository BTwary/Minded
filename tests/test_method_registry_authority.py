import unittest
import pandas as pd

from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.analytics_core.src.engines.method_selection import MethodSelectionEngine, ProblemClass
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition
from packages.schemas.src.analysis import AggregationType


class TestExecutableMethodRegistry(unittest.TestCase):
    def _semantic(self):
        return SemanticResolution(
            primary_dataset_name="customers",
            target_metric_col="churned",
            group_dimension_col="plan_tier",
            time_col=None,
            table_grain="record_level",
            available_numeric_cols=["churned"],
            available_categorical_cols=["plan_tier"],
            world_model=None,
            metric_definition=MetricDefinition(
                name="churned", table_name="customers", source_columns=["churned"],
                semantic_type="binary", aggregation_type=AggregationType.SUM,
                is_additive=True, valid_aggregations=[AggregationType.SUM],
                semantic_resolution_status="RESOLVED",
            ),
            direction_hint="unspecified", secondary_metric_col=None,
            churn_event_col=None, churn_exposure_col="plan_tier", churn_censored_col=None,
            churn_confounder_cols=[], churn_outcome_available=True,
        )

    def test_registry_selects_admissible_association_method(self):
        df = pd.DataFrame({"plan_tier": ["A", "B"] * 20, "churned": [0, 1] * 20})
        semantic = self._semantic()
        plan = UniversalQuestionCompiler.compile(
            "Does plan_tier affect churn?", semantic=semantic, df=df
        )
        intent = IntentEngine.parse_intent("Does plan_tier affect churn?")
        decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question, primary_df=df)
        decision = MethodSelectionEngine.select_for_plan(decision, plan, semantic, None, df)
        self.assertEqual(plan.task, "ASSOCIATION")
        self.assertEqual(decision.selected_method_code, "association_categorical_binary")
        self.assertIn("association_categorical_binary", decision.method_selection_scores)

    def test_registry_blocks_insufficient_prediction_data(self):
        df = pd.DataFrame({"churned": [0, 1] * 5, "age": list(range(10))})
        semantic = self._semantic()
        plan = UniversalQuestionCompiler.compile(
            "Which customers are most likely to churn next month?", semantic=semantic, df=df
        )
        intent = IntentEngine.parse_intent("Which customers are most likely to churn next month?")
        decision = MethodSelectionEngine.decide(intent, semantic, intent.raw_question, primary_df=df)
        decision = MethodSelectionEngine.select_for_plan(decision, plan, semantic, None, df)
        self.assertEqual(plan.task, "PREDICTION")
        self.assertIsNone(decision.selected_method_code)
        self.assertLess(decision.method_selection_scores.get("binary_risk_prediction", 0), 0)


if __name__ == "__main__":
    unittest.main()
