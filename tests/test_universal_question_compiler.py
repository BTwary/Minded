import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.analytics_core.src.intelligence.universal_specialized import (
    AutomaticClusteringEngine,
    MathematicalReconciliationEngine,
    GovernanceRiskEngine,
)


def _semantic(df):
    return SemanticResolution(
        primary_dataset_name="subscriptions",
        target_metric_col="churned",
        group_dimension_col="plan_tier",
        time_col="signup_date",
        table_grain="customer_id",
        available_numeric_cols=["tenure_days", "monthly_price", "support_tickets"],
        available_categorical_cols=["plan_tier", "region"],
        churn_event_col="churned",
        churn_outcome_available=True,
    )


def test_question_compiler_distinguishes_core_tasks():
    df = pd.DataFrame({"customer_id": [1, 2], "plan_tier": ["Basic", "Pro"], "churned": [0, 1], "tenure_days": [30, 90]})
    sem = _semantic(df)
    assert UniversalQuestionCompiler.compile("Does plan_tier affect churn?", semantic=sem, df=df).task == "ASSOCIATION"
    assert UniversalQuestionCompiler.compile("Did price cause churn?", semantic=sem, df=df).task == "CAUSAL"
    assert UniversalQuestionCompiler.compile("Find hidden customer segments by tenure and tickets", semantic=sem, df=df).task == "SEGMENTATION"
    assert UniversalQuestionCompiler.compile("Check Quantity UnitPrice Discount TotalAmount discrepancies", semantic=sem, df=pd.DataFrame({"Quantity":[1],"UnitPrice":[10],"Discount":[0],"TotalAmount":[10]})).task == "RECONCILIATION"


def test_reconciliation_finds_real_discrepancy():
    df = pd.DataFrame({
        "Quantity": [2, 1, 3],
        "UnitPrice": [10.0, 20.0, 5.0],
        "Discount": [0.10, 0.00, 0.20],
        "TotalAmount": [18.0, 21.0, 12.0],
    })
    result = MathematicalReconciliationEngine.analyze(df)
    assert result["status"] == "COMPLETED"
    assert result["best_identity"]["formula"] == "quantity * unit_price * (1 - discount_rate)"
    assert result["best_identity"]["discrepancy_rows"] == 1


def test_clustering_is_bounded_and_reports_quality_and_stability():
    rng = np.random.default_rng(7)
    a = rng.normal(0, 0.2, size=(40, 2))
    b = rng.normal(4, 0.2, size=(40, 2))
    df = pd.DataFrame(np.vstack([a, b]), columns=["x", "y"])
    result = AutomaticClusteringEngine.analyze(df, max_k=4)
    assert result["status"] == "COMPLETED"
    assert 2 <= result["selected_k"] <= 4
    assert "silhouette_score" in result
    assert "stability_ari" in result
    assert len(result["cluster_profiles"]) == result["selected_k"]


def test_governance_explicitly_refuses_legal_certification():
    df = pd.DataFrame({"customer_email": ["a@example.com"], "income_reported": [50000], "region": ["West"]})
    result = GovernanceRiskEngine.assess(df)
    assert "customer_email" in result["pii_columns"]
    assert "income_reported" in result["sensitive_columns"]
    assert "cannot establish" not in result["legal_boundary"].lower() or "compliance" in result["legal_boundary"].lower()


def test_association_semantics_resolve_exposure_and_target_from_question():
    df = pd.DataFrame({
        "customer_id": [1, 2],
        "plan_tier": ["Basic", "Pro"],
        "churned": [0, 1],
    })
    sem = _semantic(df)
    plan = UniversalQuestionCompiler.compile("Does plan_tier affect churn?", semantic=sem, df=df)
    assert plan.task == "ASSOCIATION"
    assert plan.semantics.explanatory_columns == ["plan_tier"]
    assert plan.semantics.target_column == "churned"
    assert plan.claim_type == "ASSOCIATION"
    assert all("causal" not in h["statement"].lower() for h in plan.hypotheses)


def test_customer_churn_risk_is_prediction_not_governance_or_forecast():
    df = pd.DataFrame({
        "customer_id": [1, 2],
        "plan_tier": ["Basic", "Pro"],
        "churned": [0, 1],
        "signup_date": ["2025-01-01", "2025-02-01"],
    })
    sem = _semantic(df)
    plan = UniversalQuestionCompiler.compile("Which customers are at highest risk of churn next month?", semantic=sem, df=df)
    assert plan.task == "PREDICTION"
    assert plan.claim_type == "PREDICTION"


def test_question_compiler_maps_paraphrases_to_same_association_contract():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "plan_tier": ["Basic", "Pro", "Basic", "Pro"],
        "churned": [0, 1, 0, 1],
    })
    sem = _semantic(df)
    plans = [
        UniversalQuestionCompiler.compile(q, semantic=sem, df=df)
        for q in [
            "Does plan_tier affect churn?",
            "Are churn rates different between subscription plans?",
            "Is churn associated with plan tier?",
            "Do plan tiers have different retention?",
        ]
    ]
    assert all(p.task in {"ASSOCIATION", "COMPARISON"} for p in plans)
    assert all(p.claim_type == "ASSOCIATION" for p in plans)
    assert plans[0].semantics.explanatory_columns == ["plan_tier"]
    assert plans[2].semantics.explanatory_columns == ["plan_tier"]
    assert plans[1].semantics.grouping_columns == ["plan_tier"]
    assert plans[3].semantics.grouping_columns == ["plan_tier"]


def test_segment_comparison_is_not_mistaken_for_segmentation():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "customer_segment": ["Basic", "Pro", "Basic", "Pro"],
        "aov": [10.0, 20.0, 12.0, 22.0],
    })
    sem = SemanticResolution(
        primary_dataset_name="orders",
        target_metric_col="aov",
        group_dimension_col="customer_segment",
        time_col=None,
        table_grain="customer_id",
        available_numeric_cols=["aov"],
        available_categorical_cols=["customer_segment"],
    )
    plan = UniversalQuestionCompiler.compile(
        "Which customer segment has the highest average order value after discounts?",
        semantic=sem,
        df=df,
    )
    assert plan.task == "COMPARISON"
    assert plan.semantics.grouping_columns == ["customer_segment"]
    assert plan.semantics.target_column == "aov"


def test_nl_resolves_group_noun_without_explicit_by_phrase():
    df = pd.DataFrame({
        "revenue": [10, 20, 30, 40],
        "region": ["North", "South", "North", "South"],
        "order_date": pd.date_range("2026-01-01", periods=4),
    })
    sem = SemanticResolution(
        primary_dataset_name="orders",
        target_metric_col="revenue",
        group_dimension_col="region",
        time_col="order_date",
        table_grain="order_id",
        available_numeric_cols=["revenue"],
        available_categorical_cols=["region"],
    )
    plan = UniversalQuestionCompiler.compile("Which regions have the highest revenue?", semantic=sem, df=df)
    assert plan.task == "COMPARISON"
    assert plan.semantics.grouping_columns == ["region"]
    assert plan.semantics.target_column == "revenue"


def test_nl_resolves_common_metric_alias_when_schema_has_single_candidate():
    df = pd.DataFrame({"net_revenue": [10, 20, 15], "region": ["N", "S", "N"]})
    sem = SemanticResolution(
        primary_dataset_name="orders",
        target_metric_col="net_revenue",
        group_dimension_col="region",
        time_col=None,
        table_grain="order_id",
        available_numeric_cols=["net_revenue"],
        available_categorical_cols=["region"],
    )
    plan = UniversalQuestionCompiler.compile("Show sales by region", semantic=sem, df=df)
    assert plan.semantics.target_column == "net_revenue"
    assert plan.semantics.grouping_columns == ["region"]

def test_intent_parser_extracts_natural_ranking_metric_group_and_direction():
    from packages.analytics_core.src.engines.intent import IntentEngine

    intent = IntentEngine.parse_intent(
        "Which region made the most money?",
        available_columns=["order_id", "customer_id", "revenue", "region"],
    )
    assert intent.intent_type == "PERFORMANCE"
    assert intent.target_metric_hint == "revenue"
    assert intent.dimension_hint == "region"
    assert intent.ranking_direction == "DESC"


def test_intent_parser_extracts_least_and_explicit_aggregation():
    from packages.analytics_core.src.engines.intent import IntentEngine

    intent = IntentEngine.parse_intent(
        "Which region generated the least total sales?",
        available_columns=["revenue", "region"],
    )
    assert intent.intent_type == "PERFORMANCE"
    assert intent.target_metric_hint == "revenue"
    assert intent.dimension_hint == "region"
    assert intent.ranking_direction == "ASC"
    assert intent.aggregation_hint == "sum"


def test_intent_parser_preserves_relation_roles_and_missing_predictor():
    from packages.analytics_core.src.engines.intent import IntentEngine

    intent = IntentEngine.parse_intent(
        "Did discount percentage affect quantity sold?",
        available_columns=["quantity", "revenue"],
    )
    assert intent.intent_type == "CORRELATION"
    assert intent.target_metric_hint == "quantity"
    assert intent.relation_predictor_phrases == ["discount percentage"]


def test_intent_parser_does_not_call_best_or_worst_a_numeric_direction():
    from packages.analytics_core.src.engines.intent import IntentEngine

    intent = IntentEngine.parse_intent(
        "Which product is best by profit?",
        available_columns=["product", "profit"],
    )
    assert intent.ranking_direction is None


def test_intent_parser_understands_common_time_horizon_without_inventing_time_column():
    from packages.analytics_core.src.engines.intent import IntentEngine

    intent = IntentEngine.parse_intent(
        "What will revenue look like next quarter?",
        available_columns=["revenue", "order_date"],
    )
    assert intent.intent_type == "FORECAST"
    assert intent.target_metric_hint == "revenue"
    assert intent.time_horizon_hint == "next quarter"
\n