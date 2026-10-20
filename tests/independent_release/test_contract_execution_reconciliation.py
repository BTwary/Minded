"""Regression tests for explicit approved-plan versus execution-receipt reconciliation."""
from types import SimpleNamespace

import pandas as pd

from packages.analytics_core.src.runtime.execution_reconciliation import (
    reconcile_approved_execution,
)

def _case():
    contract = SimpleNamespace(
        target_column="revenue", requested_aggregation="SUM", grouping_columns=("region",),
        filter_conditions=({"column": "status", "operator": "!=", "value": "refunded"},),
        time_range=None, time_column=None, comparison_values=(), numerator_column=None,
        denominator_column=None, share_of_total=False, requested_outputs=("table",),
        ranking_direction=None,
    )
    plan = SimpleNamespace(to_sql=lambda: 'SELECT "region", SUM("revenue") FROM "data_table" WHERE "status" != ? GROUP BY "region"')
    binding = SimpleNamespace(plan=plan, specification={
        "target_column": "revenue", "aggregation": "SUM", "grouping_columns": ["region"],
        "filters": [{"column": "status", "operator": "!=", "value": "refunded"}],
        "share_of_total": False, "time_range": None, "comparison_values": [],
        "requested_outputs": ["table"], "ranking_direction": None,
    })
    fingerprints = {"sales": "fp-sales"}
    versions = {"sales": {"dataset_id": "ds-1", "dataset_version_id": "dsv-1", "dataset_version": 3}}
    inputs = {"sales": "content-hash", "data_table": "content-hash"}
    result = SimpleNamespace(
        source_datasets=["sales"], source_dataset_fingerprints=fingerprints,
        source_dataset_versions=versions, input_content_fingerprints=inputs,
        executed_query=plan.to_sql(),
    )
    return contract, binding, result, fingerprints, versions, inputs


def _reconcile(case):
    contract, binding, result, fingerprints, versions, inputs = case
    return reconcile_approved_execution(
        contract, binding, result, dataset_name="sales",
        expected_fingerprints=fingerprints, expected_versions=versions,
        expected_inputs=inputs,
    )


def test_reconciliation_passes_only_when_contract_and_receipt_agree():
    report = _reconcile(_case())
    assert report["status"] == "MATCHED"
    assert all(report["checks"].values())
    assert report["failed_checks"] == []


def test_reconciliation_fails_closed_when_actual_sql_differs():
    case = _case()
    case[2].executed_query += " /* altered */"
    report = _reconcile(case)
    assert report["status"] == "MISMATCH"
    assert "executed_sql_matches" in report["failed_checks"]


def test_reconciliation_fails_closed_when_dataset_version_changes():
    case = _case()
    case[2].source_dataset_versions = {"sales": {"dataset_id": "ds-1", "dataset_version_id": "dsv-2", "dataset_version": 4}}
    report = _reconcile(case)
    assert report["status"] == "MISMATCH"
    assert "dataset_version_matches" in report["failed_checks"]


def test_reconciliation_fails_closed_when_source_fingerprint_changes():
    case = _case()
    case[2].source_dataset_fingerprints = {"sales": "different-fingerprint"}
    report = _reconcile(case)
    assert report["status"] == "MISMATCH"
    assert "dataset_fingerprint_matches" in report["failed_checks"]


def test_reconciliation_fails_closed_when_applied_filter_or_metric_differs():
    case = _case()
    case[1].specification["filters"] = [{"column": "status", "operator": "=", "value": "paid"}]
    case[1].specification["aggregation"] = "MEAN"
    report = _reconcile(case)
    assert report["status"] == "MISMATCH"
    assert "filters_match" in report["failed_checks"]
    assert "aggregation_matches" in report["failed_checks"]
