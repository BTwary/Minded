from packages.analytics_core.src.engines.calculation_lineage import CalculationLineageEngine
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition
from packages.schemas.src.analysis import AggregationType, CalculationTraceSchema, EvidenceSchema, ValidationStatus
from packages.analytics_core.src.engines.local_explanation import LocalExplanationEngine


def _metric():
    return MetricDefinition(
        name="revenue",
        table_name="sales",
        source_columns=["revenue"],
        semantic_type="sum_measure",
        aggregation_type=AggregationType.SUM,
        unit="currency",
        is_additive=True,
        rationale="Revenue is an additive monetary measure at row grain.",
    )


def test_trace_contains_formula_sql_and_verification_chain():
    trace = CalculationLineageEngine.build_experiment_trace(
        investigation_id="INV-1",
        experiment_id="EXP-1",
        effective_sql="SELECT SUM(revenue) AS total_metric FROM data_table WHERE region = 'North'",
        metric_definition=_metric(),
        dataset_fingerprints={"sales": "abc123"},
        input_row_count=100,
        output_row_count=1,
        primary_value=1250.0,
        result_columns=["total_metric"],
        verification=type("V", (), {
            "primary_metric_val": 1250.0,
            "secondary_metric_val": 1250.0,
            "observed_delta_pct": 0.0,
            "tolerance_threshold": 1e-4,
            "is_mathematically_identical": True,
            "status": "VERIFIED",
        })(),
        belief_update={"priors": [0.5, 0.5], "likelihoods": [0.9, 0.1], "posteriors": [0.9, 0.1], "delta_entropy": 0.5},
    )
    payload = trace.to_dict()
    assert payload["canonical_hash"]
    assert payload["dataset_fingerprints"] == {"sales": "abc123"}
    assert any(s["kind"] == "METRIC_FORMULA" and "SUM(revenue)" in (s["executable_expression"] or "") for s in payload["steps"])
    assert any(s["kind"] == "INDEPENDENT_VERIFICATION" for s in payload["steps"])
    assert any(s["kind"] == "BELIEF_UPDATE" and "posterior_i" in (s["formula"] or "") for s in payload["steps"])
    CalculationTraceSchema.model_validate(payload)


def test_local_explanation_exposes_calculation_trace():
    trace = {"trace_id": "TRACE-EXP-1", "investigation_id": "INV-1", "experiment_id": "EXP-1", "canonical_hash": "hash", "steps": [{"step_id": "1", "kind": "METRIC_FORMULA", "title": "Metric", "formula": "SUM(revenue)"}]}
    evidence = EvidenceSchema(
        id="EV-1",
        statement="Revenue was 1250.",
        calculation_summary="SUM(revenue)",
        dataset_version="1",
        row_count_analyzed=100,
        validation_status=ValidationStatus.PASSED,
        calculation_trace=trace,
    )
    result = LocalExplanationEngine.build(
        question="What is revenue?",
        verdict_type="INCONCLUSIVE",
        direct_answer="No decisive finding.",
        justification="Insufficient evidence.",
        evidence_rows=[evidence],
    )
    out = result.to_dict()
    assert len(out["calculation_trace"]) == 1
    assert out["calculation_trace"][0]["trace_id"] == "TRACE-EXP-1"
