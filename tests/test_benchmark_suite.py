"""
AA-OS Autonomous Intelligence Benchmark Suite.

Validates the full North-Star vision across:
1. Question understanding & Problem Classification (10 classes)
2. Semantic phrasing invariance (equivalent questions map to compatible contracts)
3. DataReadiness constraint enforcement (fail-closed on adversarial/corrupted data)
4. Formal AnalysisContract versioning & auditability
5. Analytical State Machine progression (100% auditable transitions)
6. Specialist engine execution (Segmentation, Reconciliation, Forecasting)
"""
import pytest
import numpy as np
import pandas as pd

from packages.analytics_core.src.intelligence.universal_analyst import UniversalAnalyst
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate
from packages.analytics_core.src.runtime.analytical_state_machine import (
    AnalysisStateMachine,
    AnalysisState,
    TransitionReason,
    IllegalTransitionError,
)
from packages.analytics_core.src.runtime.analysis_contract import AnalysisContractManager
from packages.analytics_core.src.intelligence.reconciliation_engine import ReconciliationEngine
from packages.analytics_core.src.intelligence.segmentation_engine import SegmentationEngine


@pytest.fixture
def base_semantic():
    return SemanticResolution(
        primary_dataset_name="enterprise_sales",
        target_metric_col="revenue",
        group_dimension_col="plan_tier",
        time_col="order_date",
        table_grain="order_id",
        available_numeric_cols=["revenue", "quantity", "unit_price", "discount"],
        available_categorical_cols=["plan_tier", "region", "customer_segment"],
    )


class TestBenchmarkClassificationMatrix:
    """Matrix testing 10 problem classes with diverse phrasing."""

    CASES = [
        ("Which customer segments exist in our active user base?", "SEGMENTATION"),
        ("Group similar accounts by behavioral patterns", "SEGMENTATION"),
        ("Identify hidden customer personas from product usage", "SEGMENTATION"),

        ("Will total sales likely increase next quarter?", "FORECASTING"),
        ("Forecast future monthly revenue over the next 12 months", "FORECASTING"),
        ("Predict next quarter gross volume trend", "FORECASTING"),

        ("Which customers are most likely to churn next month?", "PREDICTION"),
        ("Calculate probability of churn for each active user", "PREDICTION"),
        ("Predict individual customer conversion risk scores", "PREDICTION"),

        ("Clean this dataset and show exactly what you changed.", "DATA_QUALITY"),
        ("Fix malformed dates and show data cleaning changes", "DATA_QUALITY"),
        ("Remediate data quality issues and report what changed", "DATA_QUALITY"),

        ("Do Quantity, UnitPrice, Discount, and TotalAmount reconcile?", "RECONCILIATION"),
        ("Are there mathematical inconsistencies between subtotal and grand_total?", "RECONCILIATION"),
        ("Verify if quantity times unit_price matches gross_amount", "RECONCILIATION"),

        ("Does subscription tier affect customer churn rate?", "ASSOCIATION"),
        ("Is discount percentage correlated with order margin?", "ASSOCIATION"),
        ("What is the relationship between company size and renewal rate?", "ASSOCIATION"),

        ("Why did Q3 gross margin decline compared to Q2?", "DIAGNOSTIC"),
        ("Explain the key drivers behind last month revenue drop", "DIAGNOSTIC"),
        ("What caused the sudden decline in checkout conversions?", "DIAGNOSTIC"),

        ("Did the pricing change cause the observed drop in signups?", "CAUSAL"),
        ("Estimate the treatment effect of the free trial policy on retention", "CAUSAL"),
        ("What was the causal impact of the redesign intervention?", "CAUSAL"),

        ("Which region has the highest average revenue per account?", "COMPARISON"),
        ("Compare customer lifetime value across acquisition channels", "COMPARISON"),
        ("Rank subscription plans by net retention rate", "COMPARISON"),

        ("Describe the overall revenue distribution across all accounts", "DESCRIPTIVE"),
        ("Summarize baseline transaction metrics and volume", "DESCRIPTIVE"),
    ]

    @pytest.mark.parametrize("question,expected_class", CASES)
    def test_classification_accuracy(self, question, expected_class, base_semantic):
        intent = IntentEngine.parse_intent(question)
        plan = UniversalAnalyst.compile(question, intent, base_semantic)
        assert plan.problem_class == expected_class, (
            f"Classification failed for: '{question}' -> expected {expected_class}, got {plan.problem_class}"
        )


class TestBenchmarkPhrasingInvariance:
    """Verify that different phrasings of the same analytical problem produce compatible contracts."""

    CHURN_ASSOCIATION_PHRASINGS = [
        "Does plan_tier affect churn?",
        "Are customers on different plans churning at different rates?",
        "Is churn associated with subscription tier?",
        "What is the relationship between plan_tier and churn?",
    ]

    def test_phrasing_invariance(self, base_semantic):
        plans = []
        for q in self.CHURN_ASSOCIATION_PHRASINGS:
            intent = IntentEngine.parse_intent(q)
            plan = UniversalAnalyst.compile(q, intent, base_semantic)
            plans.append(plan)

        first = plans[0]
        for idx, other in enumerate(plans[1:], start=2):
            assert other.problem_class == first.problem_class == "ASSOCIATION"
            assert other.claim_type == first.claim_type


class TestBenchmarkDataReadinessConstraints:
    """Verifies that the DataQualityGate functions as an active constraint system."""

    def test_empty_dataset_fails_closed(self):
        empty_df = pd.DataFrame()
        assessment = DataQualityGate.evaluate_fitness(empty_df, dataset_name="empty")
        assert not assessment.can_proceed
        assert assessment.fitness_verdict == "UNFIT"
        assert len(assessment.constraints) > 0
        assert assessment.constraints[0].severity == "CRITICAL"
        assert len(assessment.method_allowlist) == 0

    def test_tiny_sample_blocks_inferential_methods(self):
        tiny_df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        assessment = DataQualityGate.evaluate_fitness(tiny_df, dataset_name="tiny", min_sample_size=5)
        assert not assessment.can_proceed
        assert assessment.fitness_verdict == "UNFIT"
        assert "ASSOCIATION" in assessment.method_blocklist

    def test_class_imbalance_constraint(self):
        np.random.seed(42)
        imbalanced_df = pd.DataFrame({
            "feature": np.random.randn(200),
            "target": [1] * 3 + [0] * 197,  # 1.5% minority
        })
        assessment = DataQualityGate.evaluate_fitness(imbalanced_df, dataset_name="imbalanced")
        assert any(c.check_id == "SEVERE_CLASS_IMBALANCE" for c in assessment.constraints)
        assert assessment.can_proceed  # Cautionary, not blocking entire investigation


class TestBenchmarkReconciliationEngine:
    """Verifies mathematical identity discovery and residual classification."""

    def test_reconciliation_exact_and_error(self):
        df = pd.DataFrame({
            "quantity": [10, 5, 8, 4],
            "unit_price": [100.0, 200.0, 150.0, 250.0],
            "gross_amount": [1000.0, 1050.0, 1200.0, 1000.0],  # row 1 has error (5*200=1000 != 1050)
            "discount_rate": [0.1, 0.1, 0.1, 0.1],
            "net_amount": [900.0, 945.0, 1080.0, 900.0],
        })
        engine = ReconciliationEngine()
        result = engine.discover_and_test(df, "Do Quantity, UnitPrice, and GrossAmount reconcile?", semantic=None)
        assert result.overall_status == "SOME_INCONSISTENT"
        gross_id = next(r for r in result.identity_results if "gross_amount" in r.lhs_column)
        assert gross_id.n_data_error == 1
        assert gross_id.n_exact == 3
        assert gross_id.integrity_score == 0.75


class TestBenchmarkSegmentationEngine:
    """Verifies unsupervised clustering pipeline and noise rejection."""

    def test_well_separated_clusters(self):
        np.random.seed(42)
        c1 = pd.DataFrame({"f1": np.random.normal(10, 1, 60), "f2": np.random.normal(10, 1, 60)})
        c2 = pd.DataFrame({"f1": np.random.normal(100, 5, 60), "f2": np.random.normal(100, 5, 60)})
        df = pd.concat([c1, c2], ignore_index=True)
        res = SegmentationEngine().segment(df)
        assert res.status == "SEGMENTED"
        assert res.n_clusters == 2
        assert res.silhouette_score > 0.50
        assert res.stability_score >= 0.80

    def test_insufficient_sample(self):
        tiny = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        res = SegmentationEngine().segment(tiny)
        assert res.status == "INSUFFICIENT_DATA"


class TestBenchmarkStateMachineAuditability:
    """Verifies that the state machine enforces strict legality and records transitions."""

    def test_happy_path_transitions(self):
        sm = AnalysisStateMachine("inv-bench-01")
        sm.transition(AnalysisState.QUESTION_UNDERSTOOD, TransitionReason.INTENT_CLASSIFIED, "Understood")
        sm.transition(AnalysisState.SEMANTICS_RESOLVED, TransitionReason.SCHEMA_BOUND, "Bound")
        sm.transition(AnalysisState.DATA_READINESS_EVALUATED, TransitionReason.DATA_FIT_CONFIRMED, "Ready")
        sm.transition(AnalysisState.CONTRACT_CREATED, TransitionReason.CONTRACT_COMMITTED, "Contract v1")
        assert len(sm.transitions) == 4
        assert sm.current_state == AnalysisState.CONTRACT_CREATED

    def test_illegal_teleportation_blocked(self):
        sm = AnalysisStateMachine("inv-bench-02")
        with pytest.raises(IllegalTransitionError):
            sm.transition(AnalysisState.FINALIZED, TransitionReason.FINALIZED_COMPLETE, "Skipping everything")


class TestBenchmarkAnalysisContractVersioning:
    """Verifies immutable version progression on replanning."""

    def test_contract_v1_and_replan_v2(self):
        v1 = AnalysisContractManager.create_v1(
            investigation_id="INV-V-01",
            question="Why did churn rise?",
            problem_class="DIAGNOSTIC",
            estimand="explain churn change",
            claim_type="ASSOCIATION",
            claim_ceiling="DIAGNOSED",
            primary_dataset="users",
            grain="user_id",
            unit_of_analysis=["user_id"],
            target_column="churn",
            explanatory_columns=["tenure", "monthly_charges"],
            time_column="month",
            group_dimension="contract_type",
            selected_method="DIAGNOSTIC_BATTERY",
            candidate_methods=["DIAGNOSTIC"],
            planned_experiment_codes=["EXP-01"],
        )
        assert v1.version == 1
        assert v1.parent_version is None

        v2 = AnalysisContractManager.spawn_replan(
            v1,
            trigger_reason="REPLAN_ADVERSARIAL_SIMPSON_PARADOX",
            updated_experiment_codes=["EXP-01", "EXP-02-STRATIFIED"],
        )
        assert v2.version == 2
        assert v2.parent_version == 1
        assert v2.trigger_reason == "REPLAN_ADVERSARIAL_SIMPSON_PARADOX"
        assert "EXP-02-STRATIFIED" in v2.planned_experiment_codes
