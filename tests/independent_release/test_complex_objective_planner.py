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
