"""Tests verifying canonical contract and question compiler resolution on unseen business questions.

Guarantees:
1. "Do enterprise customers have higher retention than SMB customers?" resolves target retention_rate and group customer_segment.
2. "Why is conversion higher in paid traffic?" resolves target conversion_rate and group traffic_source.
3. "Is profit correlated with advertising_spend?" resolves target and predictor accurately.
4. "What is the average revenue by region?" resolves target and group.
"""
import unittest
import pandas as pd

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler


class TestUnseenBusinessQuestions(unittest.TestCase):
    def test_enterprise_vs_smb_retention(self):
        df = pd.DataFrame({
            "retention_rate": [0.85, 0.92, 0.78, 0.95],
            "customer_segment": ["SMB", "enterprise", "SMB", "enterprise"],
            "revenue": [1000.0, 5000.0, 1200.0, 6000.0],
        })
        q = "Do enterprise customers have higher retention than SMB customers?"
        contract = compile_canonical_question_contract(q, df)
        self.assertEqual(contract.target_column, "retention_rate")
        self.assertEqual(contract.grouping_columns, ("customer_segment",))
        self.assertEqual(contract.task_family, "GROUP_COMPARISON")

        sem = SemanticEngine().resolve_schema(IntentEngine.parse_intent(q, list(df.columns)), {"customers": df})
        plan = UniversalQuestionCompiler.compile(q, semantic=sem, df=df)
        self.assertEqual(plan.semantics.target_column, "retention_rate")
        self.assertEqual(plan.semantics.grouping_columns, ["customer_segment"])
        self.assertEqual(plan.task, "COMPARISON")

    def test_conversion_by_traffic_source(self):
        df = pd.DataFrame({
            "conversion_rate": [0.04, 0.015, 0.05, 0.02],
            "traffic_source": ["paid", "organic", "paid", "organic"],
            "channel": ["google_ads", "seo", "facebook_ads", "direct"],
        })
        q = "Why is conversion higher in paid traffic?"
        contract = compile_canonical_question_contract(q, df)
        self.assertEqual(contract.target_column, "conversion_rate")
        self.assertEqual(contract.grouping_columns, ("traffic_source",))
        self.assertEqual(contract.task_family, "ROOT_CAUSE")

        sem = SemanticEngine().resolve_schema(IntentEngine.parse_intent(q, list(df.columns)), {"traffic": df})
        plan = UniversalQuestionCompiler.compile(q, semantic=sem, df=df)
        self.assertEqual(plan.semantics.target_column, "conversion_rate")
        self.assertEqual(plan.semantics.grouping_columns, ["traffic_source"])
        self.assertEqual(plan.task, "DIAGNOSTIC")

    def test_profit_correlated_with_advertising_spend(self):
        df = pd.DataFrame({
            "profit": [100.0, 150.0, 200.0, 250.0],
            "advertising_spend": [10.0, 20.0, 30.0, 40.0],
        })
        q = "Is profit correlated with advertising_spend?"
        contract = compile_canonical_question_contract(q, df)
        self.assertEqual(contract.target_column, "profit")
        self.assertIn("advertising_spend", contract.explanatory_columns)
        self.assertEqual(contract.task_family, "ASSOCIATION")

    def test_average_revenue_by_region(self):
        df = pd.DataFrame({
            "revenue": [100.0, 200.0, 150.0, 300.0],
            "region": ["North", "South", "North", "South"],
        })
        q = "What is the average revenue by region?"
        contract = compile_canonical_question_contract(q, df)
        self.assertEqual(contract.target_column, "revenue")
        self.assertEqual(contract.grouping_columns, ("region",))


if __name__ == "__main__":
    unittest.main()
