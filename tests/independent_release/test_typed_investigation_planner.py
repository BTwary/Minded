from packages.analytics_core.src.intelligence.typed_investigation_planner import (
    compile_typed_investigation, evaluate_condition, resolve_conditional_step,
)


def test_question_types_route_to_broader_analytical_families():
    cases = {
        "What is the skewness and distribution of revenue?": "DISTRIBUTION",
        "What percentage of age values are missing?": "MISSINGNESS",
        "Is price correlated with profit?": "ASSOCIATION",
        "Forecast revenue for next 3 periods": "FORECAST",
        "Did the campaign cause higher conversion?": "CAUSAL_EFFECT",
        "Find unusual transactions": "ANOMALY",
        "Reconcile subtotal and total": "RECONCILIATION",
        "What should we do to reduce cost?": "DECISION_ANALYSIS",
    }
    for question, expected in cases.items():
        plan = compile_typed_investigation(question, "DESCRIPTIVE")
        assert plan["task_type"] == expected, (question, plan["task_type"])
        assert plan["execution_authorized"] is False
        assert plan["branches"]


def test_typed_schema_distinguishes_numeric_datetime_boolean_and_text():
    import pandas as pd
    df = pd.DataFrame({"amount": [1.0], "when": pd.to_datetime(["2024-01-01"]), "active": [True], "label": ["a"]})
    plan = compile_typed_investigation("Describe this dataset", "DESCRIPTIVE", df)
    got = {x["column"]: x["value_type"] for x in plan["typed_schema"]}
    assert got == {"amount": "numeric", "when": "datetime", "active": "boolean", "label": "text_or_mixed"}


def test_conditionals_true_false_and_unknown_are_distinct_and_fail_closed():
    condition = {"fact": "causal_identification.status", "op": "eq", "value": "IDENTIFIED"}
    assert evaluate_condition(condition, {"causal_identification": {"status": "IDENTIFIED"}}) == "TRUE"
    assert evaluate_condition(condition, {"causal_identification": {"status": "FAILED"}}) == "FALSE"
    assert evaluate_condition(condition, {}) == "UNKNOWN"
    assert evaluate_condition({"fact": "x", "op": "python_eval", "value": "True"}, {"x": True}) == "UNKNOWN"


def test_conditional_step_runs_skips_or_blocks_based_on_evidence_and_dependencies():
    step = {"id": "effect", "prerequisites": ["identify"], "when": {"fact": "gate", "op": "eq", "value": "PASS"}, "otherwise": "REPORT_NON_CAUSAL"}
    assert resolve_conditional_step(step, {"gate": "PASS"}, {"identify": "COMPLETED"})["status"] == "RUN"
    skipped = resolve_conditional_step(step, {"gate": "FAIL"}, {"identify": "COMPLETED"})
    assert skipped["status"] == "SKIP" and skipped["fallback"] == "REPORT_NON_CAUSAL"
    assert resolve_conditional_step(step, {}, {"identify": "COMPLETED"})["status"] == "BLOCKED"
    assert resolve_conditional_step(step, {"gate": "PASS"}, {})["status"] == "BLOCKED"


def test_plan_exposes_missing_required_roles_instead_of_implying_readiness():
    plan = compile_typed_investigation("Forecast revenue", "FORECAST")
    assert "time" in plan["unresolved_roles"]
    assert plan["readiness"] == "NEEDS_ROLE_RESOLUTION"


def test_universal_compiler_serializes_typed_investigation_layer():
    import pandas as pd
    from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler

    class Sem:
        primary_dataset_name = "sales"
        target_metric_col = "revenue"
        group_dimension_col = "region"
        time_col = "date"
        table_grain = "row"
        secondary_metric_col = None
        churn_event_col = None
        metric_definition = None

    df = pd.DataFrame({"revenue": [10, 20, 30], "region": ["N", "S", "N"], "date": pd.date_range("2024-01-01", periods=3)})
    plan = UniversalQuestionCompiler.compile("What is the trend in revenue over time?", semantic=Sem(), df=df)
    serialized = plan.to_dict()
    assert serialized["typed_investigation"]["task_type"] == "TEMPORAL_TREND"
    assert serialized["typed_investigation"]["typed_schema"]
    assert serialized["typed_investigation"]["execution_authorized"] is False
    assert "typed_investigation_plan" in serialized["semantics"]["semantic_evidence"]


def test_runtime_policy_blocks_specialist_task_when_required_branch_is_missing_or_blocked():
    from packages.analytics_core.src.intelligence.typed_investigation_planner import conditional_execution_policy

    for task, branch in [
        ("FORECAST", "B-TIME"),
        ("PREDICTION", "B-LEAK"),
        ("COMPARISON", "B-GROUP"),
        ("ASSOCIATION", "B-TYPE"),
        ("SEGMENTATION", "B-STABILITY"),
        ("DECISION_ANALYSIS", "B-CONTEXT"),
        ("RECONCILIATION", "B-FORMULA"),
    ]:
        missing = conditional_execution_policy(task, [])
        assert missing["authorized"] is False, (task, missing)
        blocked = conditional_execution_policy(task, [{"branch_id": branch, "status": "BLOCKED"}])
        assert blocked["authorized"] is False, (task, blocked)
        allowed = conditional_execution_policy(task, [{"branch_id": branch, "status": "RUN"}])
        assert allowed["authorized"] is True, (task, allowed)


def test_causal_policy_allows_noncausal_investigation_but_never_authorizes_effect_without_gate():
    from packages.analytics_core.src.intelligence.typed_investigation_planner import conditional_execution_policy

    denied = conditional_execution_policy("CAUSAL_EFFECT", [{"branch_id": "B-ID", "status": "SKIP"}])
    assert denied["authorized"] is True
    assert denied["causal_effect_authorized"] is False
    assert denied["action"] == "ALLOW_NON_CAUSAL_ONLY"
    allowed = conditional_execution_policy("CAUSAL_EFFECT", [{"branch_id": "B-ID", "status": "RUN"}])
    assert allowed["causal_effect_authorized"] is True


def test_runtime_condition_resolver_does_not_treat_empty_leakage_indicators_as_pass():
    from packages.analytics_core.src.intelligence.typed_investigation_planner import resolve_plan_branches

    plan = {"branches": [{"id": "B-LEAK", "when": {"fact": "prediction.leakage_status", "op": "eq", "value": "PASS"},
                          "then": "HELD_OUT_EVALUATION", "otherwise": "BLOCK_MODEL_FIT"}]}
    decisions = resolve_plan_branches(plan, {"prediction": {}})
    assert decisions[0]["status"] == "BLOCKED"
    assert decisions[0]["condition_result"] == "UNKNOWN"


def test_condition_type_mismatch_is_unknown_not_false():
    assert evaluate_condition({"fact": "x", "op": "eq", "value": True}, {"x": 1}) == "UNKNOWN"
    assert evaluate_condition({"fact": "x", "op": "ne", "value": False}, {"x": 0}) == "UNKNOWN"
    assert evaluate_condition({"fact": "x", "op": "in", "value": 3}, {"x": 1}) == "UNKNOWN"


def test_real_world_question_contract_extracts_grouping_filters_time_and_ranking_without_authorizing_execution():
    import pandas as pd
    from packages.analytics_core.src.intelligence.typed_investigation_planner import compile_typed_investigation

    df = pd.DataFrame({
        "revenue": [10.0, 20.0], "profit": [2.0, 5.0], "region": ["East", "West"],
        "order_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "status": ["active", "inactive"], "customer_id": [1, 2],
    })
    cases = [
        ("What is the average revenue by region?", "AGGREGATION", "MEAN"),
        ("Show total revenue per month", "AGGREGATION", "SUM"),
        ("Compare revenue between regions", "COMPARISON", None),
        ("What is the correlation between revenue and profit?", "ASSOCIATION", "CORRELATION"),
        ("Show the monthly trend in revenue", "TEMPORAL_TREND", None),
        ("Forecast revenue for the next 3 months", "FORECAST", None),
        ("Which are the top 5 customers by revenue?", "RANKING", None),
        ("What percentage of revenue values are missing?", "MISSINGNESS", "MISSING_PERCENT"),
        ("Find unusual transactions", "ANOMALY", None),
        ("Cluster customers into segments", "SEGMENTATION", None),
        ("Did the campaign cause revenue to increase?", "CAUSAL_EFFECT", None),
        ("Predict which customers will churn", "PREDICTION", None),
        ("Analyze the sentiment of customer reviews", "TEXT_ANALYSIS", None),
        ("Reconcile subtotal and total where status is active", "RECONCILIATION", None),
    ]
    for question, expected_task, expected_op in cases:
        plan = compile_typed_investigation(question, "DESCRIPTIVE", df=df)
        assert plan["task_type"] == expected_task, (question, plan["task_type"])
        assert plan["execution_authorized"] is False
        contract = plan["natural_language_contract"]
        assert contract["candidate_only_not_execution_contract"] is True
        if expected_op:
            assert expected_op in contract["operations"], (question, contract["operations"])
    grouped = compile_typed_investigation("What is the average revenue by region?", "DESCRIPTIVE", df=df)
    assert grouped["natural_language_contract"]["grouping_candidates"]
    assert "revenue" in grouped["natural_language_contract"]["schema_column_mentions"]
    filtered = compile_typed_investigation("Reconcile subtotal and total where status is active", "DESCRIPTIVE", df=df)
    assert filtered["natural_language_contract"]["filter_candidates"]
    ranked = compile_typed_investigation("Which are the top 5 customers by revenue?", "DESCRIPTIVE", df=df)
    assert ranked["natural_language_contract"]["ranking"] == {"direction": "TOP", "n": 5}


def test_semantic_aliases_from_universal_compiler_populate_typed_roles():
    from packages.analytics_core.src.intelligence.typed_investigation_planner import compile_typed_investigation

    class Sem:
        target_metric_col = "revenue"
        group_dimension_col = "region"
        time_col = "order_date"
        explanatory_columns = ["profit"]

    import pandas as pd
    df = pd.DataFrame({"revenue": [1.0], "profit": [0.2], "region": ["East"], "order_date": pd.to_datetime(["2024-01-01"])})
    plan = compile_typed_investigation("What is the trend in revenue over time?", "DESCRIPTIVE", df=df, semantics=Sem())
    assert plan["semantic_roles"]["measure"] == "revenue"
    assert plan["semantic_roles"]["time"] == "order_date"
    assert plan["semantic_roles"]["group"] == "region"
    assert plan["semantic_role_types"]["measure"] == "numeric"
    assert plan["semantic_role_types"]["time"] == "datetime"


def test_compound_conditions_require_all_observed_roles_and_propagate_unknown():
    from packages.analytics_core.src.intelligence.typed_investigation_planner import evaluate_condition
    condition = {"all": [
        {"fact": "roles.x_type", "op": "in", "value": ["numeric", "categorical"]},
        {"fact": "roles.y_type", "op": "in", "value": ["numeric", "categorical"]},
    ]}
    assert evaluate_condition(condition, {"roles": {"x_type": "numeric", "y_type": "categorical"}}) == "TRUE"
    assert evaluate_condition(condition, {"roles": {"x_type": "numeric", "y_type": "text_or_mixed"}}) == "FALSE"
    assert evaluate_condition(condition, {"roles": {"x_type": "numeric"}}) == "UNKNOWN"
    assert evaluate_condition({"all": []}, {}) == "UNKNOWN"


def test_association_branch_requires_both_x_and_y_type_evidence():
    from packages.analytics_core.src.intelligence.typed_investigation_planner import compile_typed_investigation, resolve_plan_branches
    plan = compile_typed_investigation("What is the correlation between revenue and profit?", "ASSOCIATION")
    branch = next(x for x in plan["branches"] if x["id"] == "B-TYPE")
    assert resolve_plan_branches({"branches": [branch]}, {"roles": {"x_type": "numeric"}})[0]["status"] == "BLOCKED"
    assert resolve_plan_branches({"branches": [branch]}, {"roles": {"x_type": "numeric", "y_type": "numeric"}})[0]["status"] == "RUN"


def test_forecast_branch_requires_both_monotonic_time_and_explicit_horizon():
    from packages.analytics_core.src.intelligence.typed_investigation_planner import compile_typed_investigation, resolve_plan_branches
    plan = compile_typed_investigation("Forecast revenue monthly", "FORECAST")
    branch = next(x for x in plan["branches"] if x["id"] == "B-TIME")
    missing_horizon = resolve_plan_branches({"branches": [branch]}, {"time_index": {"monotonic": True}, "forecast": {"horizon_status": "UNKNOWN"}})[0]
    assert missing_horizon["status"] in {"BLOCKED", "SKIP"}
    assert missing_horizon["status"] != "RUN"
    valid = resolve_plan_branches({"branches": [branch]}, {"time_index": {"monotonic": True}, "forecast": {"horizon_status": "SPECIFIED"}})[0]
    assert valid["status"] == "RUN"
    assert "forecast_horizon_not_specified" in plan["natural_language_contract"]["ambiguities"]


def test_semantic_role_pointing_to_absent_column_is_not_marked_ready():
    import pandas as pd
    from packages.analytics_core.src.intelligence.typed_investigation_planner import compile_typed_investigation

    class Sem:
        target_metric_col = "revenue_total"

    df = pd.DataFrame({"revenue": [1.0], "region": ["East"]})
    plan = compile_typed_investigation("What is the average revenue?", "DESCRIPTIVE", df=df, semantics=Sem())
    assert plan["readiness"] == "NEEDS_ROLE_RESOLUTION"
    assert any(x["value"] == "revenue_total" for x in plan["invalid_role_bindings"])
    assert plan["execution_authorized"] is False


def test_broad_real_world_task_routing_matrix():
    from packages.analytics_core.src.intelligence.typed_investigation_planner import compile_typed_investigation
    cases = [
        ("Summarize this dataset", "PROFILE"),
        ("What is the total sales by region?", "AGGREGATION"),
        ("What is the average order value?", "AGGREGATION"),
        ("What percentage of customer ages are missing?", "MISSINGNESS"),
        ("Show distribution of order values", "DISTRIBUTION"),
        ("Compare conversion rate across channels", "COMPARISON"),
        ("Is discount associated with profit?", "ASSOCIATION"),
        ("Show revenue trend over time", "TEMPORAL_TREND"),
        ("Forecast monthly revenue for next 6 months", "FORECAST"),
        ("Predict customer churn", "PREDICTION"),
        ("Did discount cause higher sales?", "CAUSAL_EFFECT"),
        ("Cluster customers by purchase behavior", "SEGMENTATION"),
        ("Rank products by margin", "RANKING"),
        ("Decompose the change in revenue by region", "CONTRIBUTION"),
        ("Reconcile subtotal and total", "RECONCILIATION"),
        ("Find unusual transactions", "ANOMALY"),
        ("Estimate survival time after signup", "SURVIVAL"),
        ("Analyze sentiment in customer feedback", "TEXT_ANALYSIS"),
        ("Audit PII and fairness risks", "GOVERNANCE"),
        ("What should we do to reduce costs?", "DECISION_ANALYSIS"),
        ("Which are the top 10 products by profit?", "RANKING"),
        ("Show monthly average profit by product category", "AGGREGATION"),
        ("Compare last quarter with previous quarter", "COMPARISON"),
        ("What is the median delivery time?", "AGGREGATION"),
        ("Calculate standard deviation of price", "DISTRIBUTION"),
        ("How many unique customers purchased?", "AGGREGATION"),
        ("What is the retention rate by cohort?", "AGGREGATION"),
        ("Identify the main drivers of churn", "CONTRIBUTION"),
        ("Find themes in customer reviews", "TEXT_ANALYSIS"),
        ("Check whether subtotal + tax equals invoice total", "RECONCILIATION"),
    ]
    for question, expected in cases:
        plan = compile_typed_investigation(question, "DESCRIPTIVE")
        assert plan["task_type"] == expected, (question, plan["task_type"], expected)
        assert plan["execution_authorized"] is False
