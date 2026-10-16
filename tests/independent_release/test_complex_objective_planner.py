from packages.analytics_core.src.intelligence.complex_objective_planner import plan_compound_question


def test_multiple_metrics_by_dimension_expand_into_independent_objectives():
    plan = plan_compound_question("What are revenue and profit by region?")
    assert plan.is_compound
    assert [o.statement for o in plan.objectives] == [
        "What are revenue by region?",
        "What are profit by region?",
    ]
    assert not plan.dependency_blocked


def test_different_task_families_are_not_collapsed_into_one_objective():
    plan = plan_compound_question("Compare revenue across regions and forecast next month")
    assert plan.is_compound
    assert [o.statement for o in plan.objectives] == [
        "Compare revenue across regions",
        "forecast next month",
    ]
    assert not plan.dependency_blocked


def test_dependent_follow_up_is_blocked_from_false_independent_execution():
    plan = plan_compound_question(
        "Why did revenue fall in March, which customer segments were affected, and did support activity increase for those customers?"
    )
    assert plan.is_compound
    assert plan.dependency_blocked
    assert len(plan.objectives) == 3
    assert any(o.depends_on_prior_objective for o in plan.objectives)


def test_multi_predictor_control_wording_is_not_marked_compound():
    plan = plan_compound_question(
        "Do annual_sales depend on price and marketing_spend, controlling for discount?"
    )
    assert not plan.is_compound
    assert not plan.dependency_blocked


def test_business_metric_enumeration_expands_common_financial_metrics():
    plan = plan_compound_question("Compare revenue, gross profit and margin by region")
    assert plan.is_compound
    assert [o.statement for o in plan.objectives] == [
        "Compare revenue by region", "Compare gross profit by region", "Compare margin by region"
    ]


def test_metric_enumeration_after_grouping_expands_lifetime_value_and_returns():
    plan = plan_compound_question("Rank the top 5 segments by lifetime value and returns")
    assert plan.is_compound
    assert [o.statement for o in plan.objectives] == [
        "Rank the top 5 segments by lifetime value", "Rank the top 5 segments by returns"
    ]


def test_excessive_compound_question_is_blocked_instead_of_unbounded_plan():
    question = ", ".join([f"compare revenue in segment {i}" for i in range(13)])
    plan = plan_compound_question(question)
    assert plan.is_compound
    assert plan.dependency_blocked
    assert "12-objective" in plan.dependency_reason


def test_four_metric_list_preserves_every_metric_and_grouping():
    plan = plan_compound_question("Compare revenue, gross profit, margin and returns by region")
    assert [o.statement for o in plan.objectives] == [
        "Compare revenue by region", "Compare gross profit by region",
        "Compare margin by region", "Compare returns by region",
    ]
    assert not plan.dependency_blocked


def test_metric_list_after_by_handles_four_metrics_and_multiword_aliases():
    plan = plan_compound_question(
        "Rank the top 5 products by revenue, profit, margin and return rate"
    )
    assert [o.statement for o in plan.objectives] == [
        "Rank the top 5 products by revenue", "Rank the top 5 products by profit",
        "Rank the top 5 products by margin", "Rank the top 5 products by return rate",
    ]


def test_operational_and_marketing_metric_vocabulary_expands():
    plan = plan_compound_question("Show revenue and customer acquisition cost by channel")
    assert [o.statement for o in plan.objectives] == [
        "Show revenue by channel", "Show customer acquisition cost by channel"
    ]


def test_user_and_product_lifecycle_metrics_expand_by_month():
    plan = plan_compound_question("Compare active users, signups and cancellations by month")
    assert [o.statement for o in plan.objectives] == [
        "Compare active users by month", "Compare signups by month",
        "Compare cancellations by month",
    ]


def test_subscription_and_retention_metrics_expand_without_losing_grouping():
    plan = plan_compound_question(
        "Compare net revenue retention, cohort retention and revenue per user by month"
    )
    assert [o.statement for o in plan.objectives] == [
        "Compare net revenue retention by month",
        "Compare cohort retention by month",
        "Compare revenue per user by month",
    ]
    assert not plan.dependency_blocked


def test_commerce_funnel_metrics_expand_by_channel():
    plan = plan_compound_question(
        "Show cart abandonment, checkout completion and payment failure rate by channel"
    )
    assert [o.statement for o in plan.objectives] == [
        "Show cart abandonment by channel",
        "Show checkout completion by channel",
        "Show payment failure rate by channel",
    ]


def test_operations_and_quality_metrics_expand_by_supplier():
    plan = plan_compound_question(
        "Compare lead time, supplier defect rate and on time in full by supplier"
    )
    assert [o.statement for o in plan.objectives] == [
        "Compare lead time by supplier",
        "Compare supplier defect rate by supplier",
        "Compare on time in full by supplier",
    ]


def test_data_quality_metrics_expand_by_dataset():
    plan = plan_compound_question(
        "Compare completeness, duplicate rate and data freshness by dataset"
    )
    assert [o.statement for o in plan.objectives] == [
        "Compare completeness by dataset",
        "Compare duplicate rate by dataset",
        "Compare data freshness by dataset",
    ]


def test_finance_metrics_with_multiword_names_expand_by_quarter():
    plan = plan_compound_question(
        "Compare contribution margin, working capital and cash conversion cycle by quarter"
    )
    assert [o.statement for o in plan.objectives] == [
        "Compare contribution margin by quarter",
        "Compare working capital by quarter",
        "Compare cash conversion cycle by quarter",
    ]
