"""Regression suite for P1-3: SemanticBindingSet authority cutover.

Proves that:
1. InvestigationContract's legacy fields (target_column, explanatory_columns,
   time_column, group_dimension) project directly from its canonical
   SemanticBindingSet rather than serving as parallel/competing authorities.
2. AnalysisContractVersionData and AnalysisContractManager project legacy fields
   from semantic_bindings.
3. Replan propagation in contract_authority strictly derives child legacy fields
   from new_semantic_binding_set.
"""
import unittest

from apps.api.src.models.entities import InvestigationContract
from packages.analytics_core.src.runtime.analysis_contract import (
    AnalysisContractManager,
    AnalysisContractVersionData,
)
from packages.schemas.src.semantic_binding import SemanticBinding, SemanticBindingSet
from packages.schemas.src.semantic_role import ResolutionStatus, SemanticRole


class TestSemanticBindingAuthorityCutover(unittest.TestCase):
    def test_investigation_contract_entity_projects_from_bindings(self):
        bindings = SemanticBindingSet(bindings=[
            SemanticBinding(
                column="mrr_loss",
                table="subscriptions",
                role=SemanticRole.TARGET,
                resolution_status=ResolutionStatus.RESOLVED,
            ),
            SemanticBinding(
                column="plan_tier",
                table="subscriptions",
                role=SemanticRole.EXPLANATORY_VARIABLE,
                resolution_status=ResolutionStatus.RESOLVED,
            ),
            SemanticBinding(
                column="event_timestamp",
                table="subscriptions",
                role=SemanticRole.TIME_VARIABLE,
                resolution_status=ResolutionStatus.RESOLVED,
            ),
            SemanticBinding(
                column="cohort",
                table="subscriptions",
                role=SemanticRole.GROUPING_DIMENSION,
                resolution_status=ResolutionStatus.RESOLVED,
            ),
        ])

        contract = InvestigationContract(
            investigation_id="inv-p1-3-test",
            version=1,
            original_question="What drives MRR loss?",
            problem_class="ASSOCIATION",
            claim_type="DESCRIPTIVE",
            semantic_bindings_json=bindings.to_dict_list(),
            target_json=None,
            explanatory_variables_json=[],
            time_window_json=None,
        )

        # Projections must resolve directly from semantic_bindings_json
        self.assertEqual(contract.target_column, "mrr_loss")
        self.assertEqual(contract.explanatory_columns, ["plan_tier"])
        self.assertEqual(contract.time_column, "event_timestamp")
        self.assertEqual(contract.group_dimension, "cohort")

    def test_analysis_contract_version_data_projects_from_bindings(self):
        bindings = [
            {
                "column": "net_revenue",
                "table": "orders",
                "role": "TARGET",
                "resolution_status": "RESOLVED",
                "confidence": 1.0,
            },
            {
                "column": "channel",
                "table": "orders",
                "role": "EXPLANATORY_VARIABLE",
                "resolution_status": "RESOLVED",
                "confidence": 1.0,
            },
            {
                "column": "order_date",
                "table": "orders",
                "role": "TIME_VARIABLE",
                "resolution_status": "RESOLVED",
                "confidence": 1.0,
            },
        ]

        v1 = AnalysisContractManager.create_v1(
            investigation_id="inv-analysis-ctr-test",
            question="Analyze revenue by channel",
            problem_class="ASSOCIATION",
            estimand="channel_effect",
            claim_type="ASSOCIATION",
            claim_ceiling="STATISTICAL_ASSOCIATION",
            primary_dataset="orders",
            grain="order_id",
            unit_of_analysis=["order_id"],
            target_column=None,  # left None to prove projection from bindings
            explanatory_columns=[],
            time_column=None,
            group_dimension=None,
            selected_method="regression",
            candidate_methods=["regression"],
            planned_experiment_codes=["EXP-01"],
            semantic_bindings=bindings,
        )

        self.assertEqual(v1.target_column, "net_revenue")
        self.assertEqual(v1.explanatory_columns, ["channel"])
        self.assertEqual(v1.time_column, "order_date")
        self.assertEqual(v1.semantic_binding_set.projected_target_column(), "net_revenue")

        # Replanning with updated bindings projects new values
        new_bindings = [
            {
                "column": "gross_margin",
                "table": "orders",
                "role": "TARGET",
                "resolution_status": "RESOLVED",
                "confidence": 1.0,
            },
            {
                "column": "region",
                "table": "orders",
                "role": "EXPLANATORY_VARIABLE",
                "resolution_status": "RESOLVED",
                "confidence": 1.0,
            },
        ]

        v2 = AnalysisContractManager.spawn_replan(
            previous=v1,
            trigger_reason="REFINED_TARGET",
            new_semantic_bindings=new_bindings,
        )

        self.assertEqual(v2.target_column, "gross_margin")
        self.assertEqual(v2.explanatory_columns, ["region"])


if __name__ == "__main__":
    unittest.main()
