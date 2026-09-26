from packages.analytics_core.src.engines.local_conversational_analyst import LocalConversationalAnalyst


def test_forecast_question_routes_locally():
    plan = LocalConversationalAnalyst.plan("Will sales likely increase next quarter?")
    assert plan.problem_class == "forecasting"
    assert plan.user_intent == "forecast_future_value_or_direction"
    assert plan.target_horizon == "quarter"
    assert plan.requested_metric_hint == "sales"
    assert len(plan.follow_ups) >= 2


def test_causal_precedes_diagnostic():
    plan = LocalConversationalAnalyst.plan("Why did the price increase cause sales to fall?")
    assert plan.problem_class == "causal"
    assert "causal" in plan.explanation.lower()


def test_local_general_question_has_safe_fallback():
    plan = LocalConversationalAnalyst.plan("What is happening in the data?")
    assert plan.problem_class in {"descriptive", "general"}
    assert plan.user_intent
    assert plan.explanation


def test_survival_churn_routes_to_retention_analysis():
    plan = LocalConversationalAnalyst.plan("Which customer segment has the highest churn?")
    assert plan.problem_class == "survival_churn"
    assert plan.requested_metric_hint == "churn"


def test_followup_resolver_expands_clear_fragment_only():
    expanded, used = LocalConversationalAnalyst.resolve_followup("by region", ["Show revenue trend"])
    assert used is True
    assert expanded.startswith("Show revenue trend")
    assert "follow-up: by region" in expanded


def test_followup_resolver_does_not_rewrite_standalone_question():
    expanded, used = LocalConversationalAnalyst.resolve_followup("Why did revenue fall last month?", ["Show revenue trend"])
    assert used is False
    assert expanded == "Why did revenue fall last month?"
