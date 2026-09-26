import pandas as pd
from packages.analytics_core.src.engines.method_selection import MethodRegistry
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler

class Sem:
    primary_dataset_name = "customers"
    target_metric_col = "churned"
    group_dimension_col = "plan_tier"
    time_col = "signup_date"
    table_grain = "customer_id"
    secondary_metric_col = None
    churn_event_col = "churned"
    metric_definition = None


def test_association_binary_outcome_is_admissible():
    df = pd.DataFrame({"customer_id": range(40), "plan_tier": ["Basic","Pro"]*20, "churned": [0,1]*20})
    plan = UniversalQuestionCompiler.compile("Does plan_tier affect churn?", semantic=Sem(), df=df)
    ok, detail = MethodRegistry.admissibility_for_plan(plan, Sem(), None, df)
    assert ok, detail
    assert detail["method"] == "association_categorical_binary"


def test_prediction_is_blocked_by_severe_readiness_issue():
    df = pd.DataFrame({"customer_id": range(50), "churned": [0]*45+[1]*5, "churn_date": ["2026-01-01"]*50})
    plan = UniversalQuestionCompiler.compile("Which customers are most likely to churn next month?", semantic=Sem(), df=df)
    class Q:
        critical_issues=[]
        leakage_indicators=["Potential target/post-outcome field 'churn_date' requires temporal leakage review before prediction."]
    ok, detail = MethodRegistry.admissibility_for_plan(plan, Sem(), Q(), df)
    assert not ok
    assert "blocked:prediction_leakage_indicators_present" in detail["errors"]


def test_too_small_forecast_is_blocked():
    class F:
        primary_dataset_name="sales"
        target_metric_col="sales"
        group_dimension_col=None
        time_col="date"
        table_grain="row"
        secondary_metric_col=None
        churn_event_col=None
        metric_definition=None
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=6, freq="MS"), "sales": range(6)})
    plan = UniversalQuestionCompiler.compile("Will sales increase next quarter?", semantic=F(), df=df)
    ok, detail = MethodRegistry.admissibility_for_plan(plan, F(), None, df)
    assert not ok
    assert any("sample_size_below_minimum" in e for e in detail["errors"])


def test_finite_entity_catalog_comparison_allows_descriptive_singletons():
    class CatalogSem:
        primary_dataset_name = "products"
        target_metric_col = "unit_cost"
        group_dimension_col = "category"
        time_col = None
        table_grain = "entity_level (product_id)"
        secondary_metric_col = None
        churn_event_col = None
        metric_definition = None

    df = pd.DataFrame({
        "product_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
        "category": ["Laptop", "Tablet", "Accessories", "Accessories", "Accessories", "Audio", "Display", "Enterprise"],
        "unit_cost": [750, 220, 45, 35, 20, 90, 280, 2200],
    })
    plan = UniversalQuestionCompiler.compile("Which products are performing badly?", semantic=CatalogSem(), df=df)
    ok, detail = MethodRegistry.admissibility_for_plan(plan, CatalogSem(), None, df)
    assert ok, detail
    assert detail["method"] == "comparison_group_effect"
    assert not any("sample_size_below_minimum" in e for e in detail["errors"])
    assert not any("statistical_preflight:minimum" in e for e in detail["errors"])
