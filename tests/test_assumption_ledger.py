from types import SimpleNamespace

from packages.analytics_core.src.engines.assumption_ledger import AssumptionLedgerEngine


def _semantic(aggregation="SUM"):
    return SimpleNamespace(
        metric_definition=SimpleNamespace(aggregation_type=aggregation, is_additive=(aggregation == "SUM")),
        target_metric_col="sales",
        time_col="date",
    )


def _method(problem_class="DESCRIPTIVE"):
    estimand = SimpleNamespace(unit_of_analysis=SimpleNamespace(keys=["order_id"]))
    return SimpleNamespace(problem_class=SimpleNamespace(value=problem_class), estimand=estimand)


def _quality(score=92, warnings=None, issues=None):
    return SimpleNamespace(overall_quality_score=score, warnings=warnings or [], critical_issues=issues or [])


def test_assumption_ledger_captures_metric_and_execution_contract():
    items, card = AssumptionLedgerEngine.build(
        question="What were sales?",
        method_decision=_method(),
        semantic=_semantic(),
        quality_assessment=_quality(),
    )
    codes = {x.code for x in items}
    assert "METRIC_SEMANTICS" in codes
    assert "UNIT_OF_ANALYSIS" in codes
    assert "NO_SILENT_SAMPLING" in codes
    assert "INDEPENDENT_VERIFICATION" in codes
    assert card.assumption_count == len(items)


def test_forecast_and_causal_assumptions_are_explicit():
    causal_gate = SimpleNamespace(status=SimpleNamespace(value="OBSERVATIONAL_ONLY"))
    items, card = AssumptionLedgerEngine.build(
        question="Will sales increase next quarter?",
        method_decision=_method("FORECASTING"),
        semantic=_semantic(),
        quality_assessment=_quality(),
        temporal_scope=SimpleNamespace(found=True, ambiguous=False, ambiguity_note=None),
        causal_gate_result=causal_gate,
    )
    codes = {x.code for x in items}
    assert "FORECAST_GENERALIZATION" in codes
    assert "FORECAST_BACKTESTING" in codes

    items2, _ = AssumptionLedgerEngine.build(
        question="Did price cause sales to fall?",
        method_decision=_method("CAUSAL"),
        semantic=_semantic(),
        quality_assessment=_quality(),
        causal_gate_result=causal_gate,
    )
    assert "CAUSAL_IDENTIFICATION" in {x.code for x in items2}
