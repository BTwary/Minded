import pandas as pd
from types import SimpleNamespace

from packages.analytics_core.src.intelligence.universal_analyst import UniversalAnalyst


def sem(target="churned", dim="plan_tier"):
    return SimpleNamespace(
        target_metric_col=target,
        group_dimension_col=dim,
        secondary_metric_col=None,
        time_col="date",
        available_numeric_cols=["revenue", "support_tickets", "quantity"],
        available_categorical_cols=["plan_tier", "region"],
        churn_event_col=target if target == "churned" else None,
        metric_definition=None,
    )


def intent():
    return SimpleNamespace(business_objective="", target_metric_hint=None)


def test_question_driven_association_plan_is_minimal():
    df = pd.DataFrame({"churned": [0,1,0,1], "plan_tier": ["Basic","Pro","Basic","Pro"], "date": pd.date_range("2026-01-01", periods=4)})
    p = UniversalAnalyst.compile("Does plan_tier affect churn?", intent(), sem(), df)
    assert p.problem_class == "ASSOCIATION"
    assert p.target_column == "churned"
    assert p.explanatory_columns == ["plan_tier"]
    assert p.max_experiments == 3
    assert "causal" not in p.claim_type.lower()


def test_prediction_is_distinct_from_forecasting():
    df = pd.DataFrame({"churned": [0,1,0,1,0,1], "plan_tier": ["Basic","Pro"]*3, "customer_id": range(6)})
    p = UniversalAnalyst.compile("Which active customers are most likely to churn next month?", intent(), sem(), df)
    assert p.problem_class == "PREDICTION"
    assert p.objective == "predict_individual_risk"
    assert "leakage-safe features" in p.primary_evidence


def test_reconciliation_is_first_class():
    df = pd.DataFrame({"Quantity": [2,3], "UnitPrice": [10,10], "Discount": [0.0,0.1], "TotalAmount": [20,25]})
    s = sem(target="TotalAmount", dim=None)
    p = UniversalAnalyst.compile("Find mathematical discrepancies between Quantity, UnitPrice, Discount, and TotalAmount", intent(), s, df)
    assert p.problem_class == "RECONCILIATION"
    assert "explicit mathematical identity" in p.primary_evidence


def test_governance_never_claims_legal_compliance():
    df = pd.DataFrame({"email": ["x@y.com"], "income": [100], "churned": [1]})
    p = UniversalAnalyst.compile("Is this data GDPR compliant and could the analysis cause harm?", intent(), sem(), df)
    assert p.problem_class == "GOVERNANCE"
    assert any("jurisdiction" in x for x in p.conditional_checks)


def test_question_variables_override_generic_semantic_target_when_explicit():
    df = pd.DataFrame({
        "churned": [0, 1, 0, 1],
        "plan_tier": ["Basic", "Pro", "Basic", "Pro"],
        "revenue": [10, 20, 11, 22],
    })
    semantic = sem(target="revenue", dim="plan_tier")
    semantic.churn_event_col = "churned"
    p = UniversalAnalyst.compile("Does plan_tier affect churn?", intent(), semantic, df)
    assert p.problem_class == "ASSOCIATION"
    assert p.target_column == "churned"
    assert p.explanatory_columns[0] == "plan_tier"
