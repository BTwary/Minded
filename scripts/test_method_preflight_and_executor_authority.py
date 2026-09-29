import os
import sys
sys.path.insert(0, os.path.abspath("."))

"""Regression tests for statistical preflight and executable method authority."""
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd

from packages.analytics_core.src.statistics.preflight import numeric_pair, independent_groups
from packages.analytics_core.src.statistics.method_selection import select_correlation, select_two_group
from packages.analytics_core.src.engines.method_selection import MethodRegistry, MethodSelectionEngine, MethodSelectionDecision, ProblemClass
from packages.analytics_core.src.intelligence.experiment_synthesizer import CandidateExperiment
from packages.schemas.src.analysis import ObjectiveType


def test_constant_range_is_blocked_before_scipy():
    x = np.ones(50)
    y = np.linspace(1.0, 5.0, 50)
    pf = numeric_pair(x, y, method="correlation", min_n=3)
    assert not pf.allowed
    assert "exposure_constant_or_near_constant" in pf.reasons
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decision = select_correlation(x, y)
    assert decision.method == "NOT_APPLICABLE"
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_all_constant_groups_are_blocked_without_scipy_omnibus_call():
    groups = {"A": [2.0] * 10, "B": [2.0] * 10, "C": [2.0] * 10}
    pf = independent_groups(groups, method="multi_group_independent", min_group_n=2, min_groups=3)
    assert pf.allowed is False
    assert "constant_or_near_constant_group_outcome" in pf.reasons


def test_two_group_constant_ranges_do_not_enter_inferential_scipy_paths():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decision = select_two_group([5.0] * 20, [5.0] * 20)
    assert decision.method == "NOT_APPLICABLE"
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def _plan(task: str, target: str = "y", expl=None, group=None):
    sem = SimpleNamespace(
        target_column=target,
        explanatory_columns=expl or [],
        grouping_columns=group or [],
        group_dimension_col=(group or [None])[0],
        time_column=None,
    )
    return SimpleNamespace(task=task, semantics=sem, claim_type="PREDICTION" if task in {"PREDICTION", "FORECAST"} else "ASSOCIATION")


def test_registry_binds_prediction_to_exact_specialist_executor():
    df = pd.DataFrame({"y": [0, 1] * 30, "x": np.arange(60)})
    plan = _plan("PREDICTION")
    quality = SimpleNamespace(critical_issues=[], leakage_indicators=[])
    ok, gate = MethodRegistry.admissibility_for_plan(plan, None, quality, df)
    assert ok
    assert gate["method"] == "binary_risk_prediction"
    assert gate["executor_id"] == "specialized:binary_risk_prediction"


def test_candidate_binding_is_fail_closed_on_mismatch():
    c = CandidateExperiment(code="EXP", target_hypothesis_code="H", tool_name="duckdb_sql", query_sql="SELECT 1", description="d", aggregation_type="CORRELATION")
    MethodSelectionEngine.bind_candidates([c], "association_numeric")
    ok, detail = MethodSelectionEngine.assert_candidate_binding(c, "association_numeric")
    assert ok and detail == "scientific_loop:association_numeric"
    c.method_code = "association_categorical_binary"
    ok, detail = MethodSelectionEngine.assert_candidate_binding(c, "association_numeric")
    assert not ok
    assert "candidate_method_mismatch" in detail


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
