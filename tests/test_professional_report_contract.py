"""Regression tests for the general-purpose report contract and document exports."""
from io import BytesIO

import docx
from pypdf import PdfReader

from packages.analytics_core.src.reporting.exporter import (
    _build_report_contract,
    canonical_report_payload,
    compute_canonical_report_hash,
    export_investigation_to_docx,
    export_investigation_to_pdf,
    _render_report_chart_png,
)
from packages.analytics_core.src.reporting.narrative_composer import validate_narrative_claim, compose_report_narrative


def _sample(verified=True):
    return {
        "id": "report-contract-fixture",
        "question": "Which region had the highest total revenue in 2025?",
        "status": "COMPLETED",
        "verdict_type": "OBSERVED",
        "analysis_plan": {
            "estimand": {"target": "revenue", "requested_aggregation": "SUM"},
            "semantics": {"grouping_columns": ["region"]},
        },
        "execution_record": {
            "target_column": "revenue",
            "aggregation": "SUM",
            "grouping_columns": ["region"],
            "filters": [{"column": "order_date", "operator": "BETWEEN", "value": ["2025-01-01", "2025-12-31"]}],
            "time_range": {"start": "2025-01-01", "end": "2025-12-31", "end_exclusive": False},
            "population_row_count": 36,
            "result_row_count": 3,
            "contract_status": "MATCHED" if verified else "MISMATCH",
            "verification_status": "VERIFIED" if verified else "FAILED",
        },
        "analyst_result": {
            "kind": "RANKING",
            "headline": "Region C recorded the highest revenue.",
            "numbers": {
                "operation": "SUM",
                "target_column": "revenue",
                "grouping_columns": ["region"],
                "table": {
                    "A": {"value": 1278.0, "n": 12},
                    "B": {"value": 1878.0, "n": 12},
                    "C": {"value": 2478.0, "n": 12},
                },
            },
            "details": ["Region C leads by total revenue in the requested period."],
            "caveats": ["Descriptive result; this does not establish causation."],
            "next_steps": ["Inspect month-level patterns before deciding on an intervention."],
        },
        "evidence": [{"id": "e1", "statement": "Aggregated revenue by region", "calculation_summary": "SUM(revenue) GROUP BY region", "row_count_analyzed": 36, "validation_status": "VERIFIED" if verified else "FAILED"}],
        "reproducible_manifest_hash": "fixture-manifest",
    }


def test_report_contract_explains_interpretation_calculation_and_selects_charts():
    contract = _build_report_contract(_sample())
    assert contract["schema_version"] == "1.1"
    assert any("Target metric / outcome: revenue" in x for x in contract["interpreted"])
    assert any("Grouping dimension(s): region" in x for x in contract["interpreted"])
    assert any("Executed calculation: SUM(revenue)" in x for x in contract["calculation"])
    assert any("Eligible input records: 36" in x for x in contract["calculation"])
    assert [chart["chart_type"] for chart in contract["visualizations"]] == ["bar", "donut"]
    assert contract["visualizations"][0]["data"][-1]["revenue"] == 1278.0
    narrative = contract["narrative"]
    assert narrative["family"] == "comparison"
    assert narrative["status"] == "READY"
    assert "C recorded the highest sum of revenue" in narrative["summary"]
    assert any("93.9%" in text for text in narrative["paragraphs"])
    assert narrative["supported_claim_count"] >= 4
    assert narrative["validation"]["status"] == "PASSED"
    assert all(c["validation_status"] == "SUPPORTED" for c in narrative["claims"])
    assert all(c["evidence_refs"] for c in narrative["claims"])


def test_report_contract_does_not_visualize_unverified_execution():
    contract = _build_report_contract(_sample(verified=False))
    assert contract["visualizations"] == []
    assert contract["narrative"]["status"] == "WITHHELD"
    assert contract["narrative"]["paragraphs"] == []
    assert contract["narrative"]["supported_claim_count"] == 0


def test_claim_validator_withholds_fabricated_number_even_when_other_facts_are_verified():
    facts = {
        "metric": {"value": 1278.0, "source_paths": ["analyst_result.numbers.value"], "evidence_refs": ["e1"]},
        "group": {"value": "Region C", "source_paths": ["analyst_result.numbers.group"], "evidence_refs": ["e1"]},
    }
    claim = {
        "text": "Region C recorded revenue of 99,999.",
        "fact_ids": ["metric", "group"],
        "evidence_refs": ["e1"],
        "source_paths": ["analyst_result.numbers.value", "analyst_result.numbers.group"],
        "numeric_literals": ["99999"],
    }
    ok, issues = validate_narrative_claim(claim, facts, ["e1"], ["analyst_result.numbers.value", "analyst_result.numbers.group"])
    assert not ok
    assert any("unsupported numeric literals" in issue for issue in issues)


def test_trend_narrative_uses_saved_slope_and_interval_only():
    result = {
        "kind": "TREND",
        "headline": "Revenue increased.",
        "numbers": {"target_column": "revenue", "grain": "year", "slope_per_period": 0.5, "start_level": 10, "end_level": 13, "ci_low": 0.2, "ci_high": 0.8},
        "caveats": ["Historical trend is not causal evidence."],
    }
    evidence = [{"id": "trend-e1", "validation_status": "VERIFIED", "statement": "Trend estimate", "calculation_summary": "OLS trend"}]
    execution = {"target_column": "revenue", "aggregation": "TREND", "contract_status": "MATCHED", "verification_status": "VERIFIED"}
    narrative = compose_report_narrative(question="How did revenue trend?", analyst_result=result, execution_record=execution, evidence=evidence, status="COMPLETED")
    assert narrative["family"] == "trend"
    assert narrative["status"] == "READY"
    assert "upward" in narrative["summary"]
    assert any("0.2 to 0.8" in text for text in narrative["paragraphs"])
    assert narrative["validation"]["status"] == "PASSED"


def test_pdf_contains_transparent_report_sections_and_chart_title():
    sample = _sample()
    pdf = export_investigation_to_pdf(sample)
    assert len(pdf) > 1000
    text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(pdf)).pages)
    assert "WHAT MINDED INTERPRETED" in text
    assert "WHAT THE COMPARISON SHOWS" in text
    assert "highest sum of revenue" in text
    assert "Evidence support: e1" in text
    assert "HOW THE RESULT WAS CALCULATED" in text
    assert "VISUALIZATIONS" in text
    assert "Share of returned total revenue by region" in text
    assert "Eligible input records: 36" in text


def test_docx_contains_report_sections_and_embedded_chart_images():
    doc = docx.Document(BytesIO(export_investigation_to_docx(_sample())))
    paragraphs = "\n".join(p.text for p in doc.paragraphs)
    assert "What MindEd interpreted" in paragraphs
    assert "What the comparison shows" in paragraphs
    assert "highest sum of revenue" in paragraphs
    assert "Evidence support: e1" in paragraphs
    assert "How the result was calculated" in paragraphs
    assert "Visualizations" in paragraphs
    assert "Share of returned total revenue by region" in paragraphs
    assert len(doc.inline_shapes) >= 2  # branded logo + charts


def test_report_hash_covers_execution_record_and_derived_report_content():
    base = _sample()
    original = compute_canonical_report_hash(base)
    changed = _sample()
    changed["execution_record"]["population_row_count"] = 35
    assert compute_canonical_report_hash(changed) != original
    payload = canonical_report_payload(base)
    assert payload["execution_record"]["population_row_count"] == 36
    assert len(payload["report_contract"]["visualizations"]) == 2


def test_saved_composition_over_time_becomes_embeddable_stacked_chart():
    payload = _sample()
    payload["analyst_result"]["numbers"]["composition_over_time"] = {
        "time_field": "quarter", "series_field": "region", "value_field": "revenue",
        "data": [
            {"quarter": "2025-Q1", "region": "A", "revenue": 100},
            {"quarter": "2025-Q1", "region": "B", "revenue": 80},
            {"quarter": "2025-Q2", "region": "A", "revenue": 120},
            {"quarter": "2025-Q2", "region": "B", "revenue": 90},
        ],
    }
    contract = _build_report_contract(payload)
    chart = contract["visualizations"][0]
    assert chart["chart_type"] == "stacked_bar"
    assert len(chart["data"]) == 4
    png = _render_report_chart_png(chart)
    assert png and png.startswith(b"\x89PNG\r\n\x1a\n")


def test_saved_paired_observations_become_scatter_chart():
    payload = _sample()
    payload["analyst_result"]["kind"] = "RELATIONSHIP"
    payload["analyst_result"]["numbers"] = {
        "target_column": "revenue",
        "scatter_data": {"x_field": "marketing_spend", "y_field": "revenue", "data": [
            {"marketing_spend": 10, "revenue": 20},
            {"marketing_spend": 15, "revenue": 27},
            {"marketing_spend": 20, "revenue": 31},
        ]},
    }
    contract = _build_report_contract(payload)
    assert any(chart["chart_type"] == "scatter" for chart in contract["visualizations"])
    scatter = next(chart for chart in contract["visualizations"] if chart["chart_type"] == "scatter")
    assert len(scatter["data"]) == 3
    assert _render_report_chart_png(scatter).startswith(b"\x89PNG\r\n\x1a\n")


def test_saved_histogram_bins_become_distribution_chart():
    payload = _sample()
    payload["question"] = "Show the distribution of revenue"
    payload["analyst_result"]["numbers"]["histogram_bins"] = [
        {"bin_start": 0, "bin_end": 10, "count": 3},
        {"bin_start": 10, "bin_end": 20, "count": 7},
    ]
    contract = _build_report_contract(payload)
    histogram = next(chart for chart in contract["visualizations"] if chart["chart_type"] == "histogram")
    assert len(histogram["data"]) == 2
    assert _render_report_chart_png(histogram).startswith(b"\x89PNG\r\n\x1a\n")


def test_line_report_chart_renders_saved_uncertainty_interval():
    from packages.analytics_core.src.reporting.exporter import _normalise_chart_candidate
    chart = _normalise_chart_candidate({
        "chart_id": "forecast", "chart_type": "line", "title": "Actual versus forecast",
        "x_axis": {"field": "period", "type": "nominal"}, "y_axis": {"field": "value"},
        "data": [
            {"period": "2025-Q1", "value": 10},
            {"period": "2025-Q2", "value": 12, "ci_low": 9, "ci_high": 15},
            {"period": "2025-Q3", "value": 14, "ci_low": 10, "ci_high": 18},
        ],
    })
    assert chart is not None
    png = _render_report_chart_png(chart)
    assert png and png.startswith(b"\x89PNG\r\n\x1a\n")



def test_combined_diagnostic_comparison_trend_uses_multiple_blueprints():
    result = {
        "kind": "ROOT_CAUSE",
        "headline": "Revenue declined.",
        "numbers": {
            "target_column": "revenue", "operation": "SUM", "grain": "year",
            "table": {"Region A": {"value": 100, "n": 10}, "Region B": {"value": 80, "n": 10}, "Region C": {"value": 60, "n": 10}},
            "previous": 120, "current": 100, "top_contributor": "Region B", "top_contribution_share": 0.4,
            "slope_per_period": -2.5,
        },
        "caveats": ["Observed changes do not establish cause."],
        "next_steps": ["Inspect sales volume and unit price separately."],
    }
    execution = {"target_column": "revenue", "aggregation": "SUM", "grouping_columns": ["region"], "contract_status": "MATCHED", "verification_status": "VERIFIED"}
    evidence = [{"id": "E1", "validation_status": "VERIFIED", "statement": "Verified revenue values", "calculation_summary": "SUM(revenue) GROUP BY region"}]
    narrative = compose_report_narrative(
        question="Why did revenue decline, and which regions contributed most over the last five years?",
        analyst_result=result, execution_record=execution, evidence=evidence, status="COMPLETED",
    )
    assert narrative["families"] == ["diagnostic", "comparison", "trend"]
    assert narrative["status"] == "READY"
    assert any("Region A recorded the highest sum" in text for text in narrative["paragraphs"])
    assert any("downward" in text and "slope" in text for text in narrative["paragraphs"])
    assert all(claim["validation_status"] == "SUPPORTED" for claim in narrative["claims"])
    assert all(claim["evidence_refs"] for claim in narrative["claims"])


def test_relationship_narrative_uses_recorded_correlations_without_claiming_causality():
    result = {
        "kind": "CORRELATION",
        "headline": "Variables are associated.",
        "numbers": {"target_column": "revenue", "correlations": {"marketing_spend": 0.82, "price": -0.4}},
        "caveats": [],
    }
    evidence = [{"id": "E-rel", "validation_status": "VERIFIED", "statement": "Verified correlation results"}]
    execution = {"target_column": "revenue", "aggregation": "PEARSON", "contract_status": "MATCHED", "verification_status": "VERIFIED"}
    narrative = compose_report_narrative(question="What variables are correlated with revenue?", analyst_result=result, execution_record=execution, evidence=evidence, status="COMPLETED")
    assert narrative["family"] == "relationship"
    assert "marketing_spend" in narrative["summary"]
    assert any("does not by itself establish" in text for text in narrative["paragraphs"])
    assert narrative["validation"]["status"] == "PASSED"


def test_distribution_narrative_reports_iqr_from_saved_quartiles():
    result = {
        "kind": "DISTRIBUTION",
        "headline": "Revenue spread summarized.",
        "numbers": {"target_column": "revenue", "distribution_profile": {"observed": {"p25": 10, "p50": 15, "p75": 24}}},
        "caveats": [],
    }
    evidence = [{"id": "E-dist", "validation_status": "VERIFIED", "statement": "Verified distribution profile"}]
    execution = {"target_column": "revenue", "aggregation": "DISTRIBUTION_SUMMARY", "contract_status": "MATCHED", "verification_status": "VERIFIED"}
    narrative = compose_report_narrative(question="Show the distribution of revenue.", analyst_result=result, execution_record=execution, evidence=evidence, status="COMPLETED")
    assert narrative["family"] == "distribution"
    assert "10 to 24" in narrative["summary"]
    assert "14" in narrative["summary"]
    assert narrative["validation"]["status"] == "PASSED"


def test_forecast_narrative_requires_a_saved_projection_and_interval():
    result = {
        "kind": "FORECAST",
        "headline": "Projection available.",
        "numbers": {"target_column": "revenue", "grain": "month", "run_rate_projection": {"projected_value": 110, "pi_low": 95, "pi_high": 130}},
        "caveats": [],
    }
    evidence = [{"id": "E-fc", "validation_status": "VERIFIED", "statement": "Verified forecast output"}]
    execution = {"target_column": "revenue", "aggregation": "FORECAST", "contract_status": "MATCHED", "verification_status": "VERIFIED"}
    narrative = compose_report_narrative(question="Forecast revenue for next month.", analyst_result=result, execution_record=execution, evidence=evidence, status="COMPLETED")
    assert narrative["family"] == "forecast"
    assert "110" in narrative["summary"]
    assert any("95 to 130" in text for text in narrative["paragraphs"])
    assert any("not a guaranteed outcome" in text for text in narrative["paragraphs"])
    assert narrative["validation"]["status"] == "PASSED"
