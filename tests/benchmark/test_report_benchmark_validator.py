"""Regression tests for the machine-checkable real-world report benchmark."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from packages.analytics_core.src.reporting.report_benchmark import (
    report_benchmark_json_schema,
    validate_report_contract,
)


def valid_report() -> dict:
    return {
        "schema_version": "1.0.0",
        "metadata": {
            "title": "Q3 Revenue Analysis", "author_or_system": "MindEd AA-OS",
            "generated_at": "2026-10-20T12:00:00Z", "version": "1.0.0",
            "data_as_of": "2026-10-15", "request_id": "REQ-001", "stakeholder": "Operations lead",
        },
        "executive_summary": {
            "text": "Q3 revenue was 100 units. Audit repeat purchases next quarter.",
            "question": "Which region had the highest Q3 revenue?",
            "answer": "West had the highest revenue.",
            "key_numbers": [{"literal": "100", "fact_id": "fact-west-revenue", "semantic_role": "revenue"}],
            "non_metric_numeric_tokens": ["3"],
            "recommended_action": "Audit repeat purchases in West.",
        },
        "objective": [{"question_id": "q1", "question": "Which region had the highest Q3 revenue?", "decision_to_inform": "Prioritize regional review."}],
        "scope": {
            "time_range": "Q3 2026", "population": "Delivered, non-returned orders",
            "exclusions": ["Orders without delivery completion"],
            "metric_definitions": [{"name": "revenue", "definition": "Sum of recorded order revenue", "unit": "units"}],
            "glossary": [{"term": "Q3", "definition": "Third calendar quarter"}],
        },
        "data_sources": [{"name": "orders.csv", "extraction_time": "2026-10-20T11:00:00Z", "row_count": 1000, "known_limitations": ["None identified"], "owner": "Operations"}],
        "methodology": {
            "tools": ["DuckDB", "Python"], "cleaning_steps": [{"operation": "None", "rows_affected": 0, "reason": "No cleaning required", "disclosed_in_report": True}],
            "statistical_methods": [], "assumptions": ["Revenue field is recorded in consistent units."],
            "metrics_computation": "SUM(revenue) grouped by region after the approved filters.",
        },
        "data_quality": {
            "missing_value_percentages": {"revenue": 0.0}, "duplicate_count": 0,
            "duplicate_handling": "No duplicates found.", "outlier_handling": "No outliers removed.",
            "reconciliation_checks": [{"name": "Input row count", "observed": 1000, "expected": 1000, "tolerance": 0, "passed": True}],
        },
        "evidence": [{"evidence_id": "ev1", "status": "VERIFIED", "source": "execution-1", "calculation_id": "calc-1", "description": "Executed aggregate query"}],
        "facts": [{"fact_id": "fact-west-revenue", "value": 100, "unit": "units", "lineage_id": "lineage-1", "source_path": "result.groups.West.revenue", "calculation_id": "calc-1", "evidence_refs": ["ev1"], "verified": True}],
        "claims": [{"claim_id": "claim1", "text": "West revenue was 100 units.", "claim_type": "descriptive", "fact_ids": ["fact-west-revenue"], "evidence_refs": ["ev1"], "numeric_refs": [{"literal": "100", "fact_id": "fact-west-revenue", "semantic_role": "revenue"}]}],
        "findings": [{"finding_id": "f1", "headline": "West had the highest Q3 revenue.", "analysis_type": "comparative", "interpretation": "West leads the compared groups.", "confidence": "high", "confidence_explanation": "All eligible rows were included in the approved aggregation.", "metric_names": ["revenue"], "fact_ids": ["fact-west-revenue"], "evidence_refs": ["ev1"], "chart_refs": [], "table_refs": ["t1"], "sample_size": 100}],
        "result_tables": [{"table_id": "t1", "title": "Revenue by region", "rows": [{"label": "West", "value": 100, "fact_id": "fact-west-revenue", "n": 100}], "source_note": "Saved group-level execution results"}],
        "visualizations": [{"chart_id": "c1", "chart_type": "bar", "title": "West leads regional Q3 revenue", "x_axis": {"label": "Region", "unit": "category"}, "y_axis": {"label": "Revenue", "unit": "units"}, "source_note": "Saved group-level execution results", "data": [{"label": "West", "value": 100, "fact_id": "fact-west-revenue", "n": 100}]}],
        "conclusions": [{"question_id": "q1", "answer": "West had the highest Q3 revenue.", "fact_ids": ["fact-west-revenue"], "evidence_refs": ["ev1"]}],
        "recommendations": [{"action": "Audit repeat-purchase trends in West.", "priority": "medium", "expected_impact": "May identify a further analysis opportunity; impact is not quantified.", "finding_refs": ["f1"], "owner": "Operations"}],
        "limitations": ["This is a descriptive comparison and may not generalize beyond the selected population."],
        "causation_caveat": "The observed regional difference does not establish a causal mechanism.",
        "unanswered_questions": ["The data does not identify the reason for the regional difference."],
        "appendix": {"queries_or_code": ["SELECT region, SUM(revenue) ... GROUP BY region"], "data_dictionary": [{"field": "revenue", "definition": "Recorded order revenue", "data_type": "number", "units_or_values": "units"}], "full_tables": [], "version_history": ["v1: initial report"]},
    }


def test_complete_report_passes_bronze_gate():
    result = validate_report_contract(valid_report())
    assert result.valid
    assert result.publishable
    assert result.grade == "BRONZE"
    assert result.completeness_rate == 1.0
    assert result.numeric_traceability_rate == 1.0


def test_question_conclusion_count_and_ids_must_match():
    data = valid_report()
    data["conclusions"] = []
    result = validate_report_contract(data)
    assert not result.publishable
    assert any("Question-answer mismatch" in err or "conclusion" in err.lower() for err in result.errors)


def test_untraceable_numeric_claim_is_hard_fail():
    data = valid_report()
    data["claims"][0]["text"] = "West revenue was 100 units and increased by 20 units."
    result = validate_report_contract(data)
    assert not result.publishable
    assert any("untraceable numeric tokens" in err for err in result.errors)


def test_chart_or_table_mismatch_with_fact_is_hard_fail():
    data = valid_report()
    data["visualizations"][0]["data"][0]["value"] = 999
    result = validate_report_contract(data)
    assert not result.publishable
    assert any("does not match fact" in err for err in result.errors)


def test_unhedged_causal_claim_without_admissible_design_is_hard_fail():
    data = valid_report()
    data["claims"][0]["text"] = "The discount caused revenue to increase."
    result = validate_report_contract(data)
    assert not result.publishable
    assert any("causal language" in err or "Causal claim" in err for err in result.errors)


def test_undisclosed_removed_rows_and_failed_reconciliation_are_hard_failures():
    data = valid_report()
    data["methodology"]["cleaning_steps"] = [{"operation": "Drop rows", "rows_affected": 12, "reason": "Invalid records", "disclosed_in_report": False}]
    data["data_quality"]["reconciliation_checks"] = [{"name": "Revenue total", "observed": 900, "expected": 1000, "tolerance": 0, "passed": False}]
    result = validate_report_contract(data)
    assert not result.publishable
    assert any("not disclosed" in err for err in result.errors)
    assert any("without an explained exception" in err for err in result.errors)


def test_forecast_requires_uncertainty_interval():
    data = valid_report()
    data["findings"][0]["analysis_type"] = "predictive"
    data["findings"][0]["predictive_output"] = {"point_forecast": 110, "uncertainty_explanation": "Interval omitted intentionally for test."}
    result = validate_report_contract(data)
    assert not result.publishable
    assert any("uncertainty interval" in err for err in result.errors)


def test_undefined_metric_and_missing_definitions_fail():
    data = valid_report()
    data["findings"][0]["metric_names"] = ["unregistered_metric"]
    result = validate_report_contract(data)
    assert not result.publishable
    assert any("undefined metrics" in err for err in result.errors)


def test_schema_snapshot_is_valid_and_matches_model_generation():
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).parents[2] / "packages" / "schemas" / "src" / "report_benchmark.schema.json"
    stored = json.loads(schema_path.read_text(encoding="utf-8"))
    generated = report_benchmark_json_schema()
    assert stored == generated
    jsonschema.Draft202012Validator.check_schema(stored)
    jsonschema.Draft202012Validator(stored).validate(valid_report())


def test_silver_and_gold_require_independent_golden_set_metrics():
    metrics = {
        "rerun_exact_match_rate": 1.0, "key_figure_traceability_rate": 1.0,
        "planted_issue_detection_rate": 0.95, "false_insight_rate_on_no_signal": 0,
        "quality_rubric_average": 1.8, "reconciliation_pass_rate": 1.0,
        "statistical_validity_passed": True, "golden_dataset_count": 24,
        "human_reviewers": 3, "inter_rater_agreement": 0.82,
    }
    result = validate_report_contract(valid_report(), metrics)
    assert result.grade == "GOLD"
    assert result.quality_average >= 1.7


def test_draft_report_can_never_be_published_when_contract_invalid():
    data = deepcopy(valid_report())
    data["executive_summary"]["text"] = "A report summary with seven lines.\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7"
    result = validate_report_contract(data)
    assert not result.publishable
    assert any("six-line maximum" in err for err in result.errors)


def test_golden_reproducibility_failure_blocks_publishability():
    metrics = {
        "rerun_exact_match_rate": 0.99, "key_figure_traceability_rate": 1.0,
        "planted_issue_detection_rate": 0.95, "false_insight_rate_on_no_signal": 0,
        "quality_rubric_average": 1.8, "reconciliation_pass_rate": 1.0,
        "statistical_validity_passed": True, "golden_dataset_count": 24,
        "human_reviewers": 3, "inter_rater_agreement": 0.82,
    }
    result = validate_report_contract(valid_report(), metrics)
    assert not result.publishable
    assert result.grade == "FAIL"
    assert any("Bronze gate failed" in err for err in result.errors)


def test_recommendation_cannot_invent_numeric_impact():
    data = valid_report()
    data["recommendations"][0]["expected_impact"] = "Expected recovery of 40 units per quarter."
    result = validate_report_contract(data)
    assert not result.publishable
    assert any("Recommendation expected impact contains untraceable numeric token '40'" in err for err in result.errors)


def test_reconciliation_pass_flag_is_recomputed_against_tolerance():
    data = valid_report()
    data["data_quality"]["reconciliation_checks"] = [{"name": "Revenue total", "observed": 900, "expected": 1000, "tolerance": 1, "passed": True}]
    result = validate_report_contract(data)
    assert not result.publishable
    assert any("marked passed but observed and expected values exceed tolerance" in err for err in result.errors)


def test_inferential_test_requires_p_value_effect_size_and_interval():
    data = valid_report()
    data["methodology"]["statistical_methods"] = [{
        "name": "Two-proportion z-test", "rationale": "Compare conversion rates between randomized arms.",
        "inference_type": "inferential", "assumptions_checked": ["Independence checked"],
    }]
    result = validate_report_contract(data)
    assert not result.publishable
    assert any("must report p-value, effect size, and confidence interval" in err for err in result.errors)
