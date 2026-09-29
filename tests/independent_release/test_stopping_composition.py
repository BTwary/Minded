from packages.analytics_core.src.engines.stopping import StoppingEngine


def test_missingness_gate_survives_max_iteration_stop():
    d = StoppingEngine.evaluate_stopping(
        leading_posterior=0.95,
        experiments_completed=5,
        max_iterations=5,
        all_verifications_passed=True,
        counter_hypothesis_evaluated=True,
        missingness_classification="SENSITIVE",
    )
    assert d.should_stop is True
    assert d.reason == "SAFETY_LIMIT_WITH_UNRESOLVED_GATES"
    assert d.criteria_breakdown["missingness_resolved"] is False


def test_adversarial_gate_survives_max_iteration_stop():
    d = StoppingEngine.evaluate_stopping(
        leading_posterior=0.99,
        experiments_completed=5,
        max_iterations=5,
        all_verifications_passed=True,
        counter_hypothesis_evaluated=True,
        unresolved_adversarial_issues=["confounding"],
    )
    assert d.should_stop is True
    assert d.reason == "SAFETY_LIMIT_WITH_UNRESOLVED_GATES"
    assert d.criteria_breakdown["adversarial_resolved"] is False
