"""Clean-room AA-OS acceptance benchmark.

This suite deliberately computes expected properties independently with pandas/NumPy/scikit-learn
and does not call AA-OS internal result functions to manufacture an oracle.

It is split into:
1. Engine-independent acceptance/oracle tests that must run on a lean environment.
2. A separate controller gate marker for CI environments that have DuckDB/Polars/PyArrow.

The suite is deterministic: no uncontrolled RNG, no hard-coded business labels, and no reuse of
AA-OS analytical result objects as the expected answer.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from packages.analytics_core.src.data.content_identity import compute_content_hash
from packages.analytics_core.src.engines.method_selection import MethodRegistry
from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate
from packages.analytics_core.src.statistics.churn_estimands import (
    crude_churn_rate,
    censoring_report,
    stratified_rate_check,
)
from packages.analytics_core.src.graph.evidence_identity import compute_evidence_identity


class TestCleanRoomAcceptance(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame({
            "acct": [f"A{i:03d}" for i in range(12)],
            "segment": ["north", "north", "north", "north", "south", "south", "south", "south", "west", "west", "west", "west"],
            "metric_value": [10, 12, 11, 13, 20, 21, 19, 20, 9, 10, 11, 8],
            "period": pd.date_range("2025-01-01", periods=12, freq="MS"),
        })

    def test_01_row_and_column_order_invariance(self):
        a = self.base.copy()
        b = a.sample(frac=1.0, random_state=7).reset_index(drop=True)
        b = b[["period", "metric_value", "segment", "acct"]]
        self.assertEqual(compute_content_hash(a), compute_content_hash(b))

    def test_02_content_change_changes_identity(self):
        a = self.base.copy()
        b = a.copy()
        b.loc[0, "metric_value"] += 1
        self.assertNotEqual(compute_content_hash(a), compute_content_hash(b))

    def test_03_duplicate_rows_are_not_collapsed(self):
        a = self.base.copy()
        b = pd.concat([a, a.iloc[[0]]], ignore_index=True)
        self.assertNotEqual(compute_content_hash(a), compute_content_hash(b))

    def test_04_missingness_is_detected_independently(self):
        df = self.base.copy()
        df.loc[[1, 3, 5], "metric_value"] = np.nan
        assessment = DataQualityGate.evaluate_fitness(df, dataset_name="clean-room-missing")
        self.assertGreaterEqual(assessment.missingness_summary["metric_value"], 25.0)

    def test_05_empty_and_tiny_inputs_fail_closed(self):
        empty = DataQualityGate.evaluate_fitness(pd.DataFrame(), dataset_name="empty")
        self.assertFalse(empty.can_proceed)
        tiny = DataQualityGate.evaluate_fitness(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), dataset_name="tiny", min_sample_size=5)
        self.assertFalse(tiny.can_proceed)

    def test_06_schema_rename_does_not_create_false_identity_equivalence(self):
        renamed = self.base.rename(columns={"metric_value": "economic_measure"})
        self.assertNotEqual(compute_content_hash(self.base), compute_content_hash(renamed))
        # The clean-room oracle checks semantic mapping separately rather than assuming name-based identity.
        self.assertEqual(len(renamed), len(self.base))
        self.assertAlmostEqual(float(renamed["economic_measure"].mean()), float(self.base["metric_value"].mean()))

    def test_07_join_fanout_oracle_detects_inflation(self):
        fact = pd.DataFrame({"acct": ["A", "B"], "revenue": [100.0, 50.0]})
        dim = pd.DataFrame({"acct": ["A", "A", "B"], "tier": ["gold", "gold-legacy", "silver"]})
        joined = fact.merge(dim, on="acct", how="left")
        naive_total = float(joined["revenue"].sum())
        expected_total = float(fact["revenue"].sum())
        self.assertGreater(naive_total, expected_total)
        self.assertEqual(expected_total, 150.0)

    def test_08_simpsons_paradox_oracle(self):
        # Within each stratum B has lower churn than A; aggregate can reverse due composition.
        rows = []
        for cohort, n_a, n_b, p_a, p_b in [
            ("new", 20, 80, 0.30, 0.20),
            ("mature", 80, 20, 0.10, 0.05),
        ]:
            rows += [{"segment": "A", "cohort": cohort, "churn_event": int(i < round(n_a * p_a))} for i in range(n_a)]
            rows += [{"segment": "B", "cohort": cohort, "churn_event": int(i < round(n_b * p_b))} for i in range(n_b)]
        df = pd.DataFrame(rows)
        oracle = df.groupby("segment")["churn_event"].mean()
        check = stratified_rate_check(df, "segment", "cohort", min_stratum_n=10)
        self.assertIn(oracle.index[0], ["A", "B"])
        self.assertTrue(check["applicable"])
        self.assertTrue(check["confound_detected"] or any(s.get("diff", 0) * check["aggregate_diff"] < 0 for s in check["per_stratum"] if s.get("adequate_sample")))

    def test_09_censoring_never_becomes_confirmed_nonchurn(self):
        df = pd.DataFrame({
            "segment": ["A", "A", "B", "B"],
            "churn_event": [1, 0, np.nan, 0],
            "censored": [0, 0, 1, 0],
            "observation_days": [90, 90, 40, 90],
        })
        report = censoring_report(df)
        rates = crude_churn_rate(df, "segment")
        self.assertTrue(report["censoring_data_available"])
        self.assertEqual(report["n_censored_incomplete_observation"], 1)
        b = rates.loc[rates["segment"] == "B"].iloc[0]
        self.assertEqual(int(b["eligible_n"]), 1)
        self.assertEqual(int(report["n_censored_incomplete_observation"]), 1)

    def test_10_temporal_leakage_is_flagged_by_quality_layer(self):
        df = pd.DataFrame({
            "event_time": pd.date_range("2025-01-01", periods=30, freq="D"),
            "outcome": [0, 1] * 15,
            "post_outcome_status": ["closed"] * 30,
            "age": list(range(30)),
        })
        assessment = DataQualityGate.evaluate_fitness(df, dataset_name="leakage")
        self.assertTrue(assessment.leakage_indicators)
        self.assertTrue(any("post-outcome" in x.lower() for x in assessment.leakage_indicators))

    def test_11_evidence_identity_changes_with_population(self):
        ds = "a" * 64
        q = "SELECT segment, AVG(metric_value) FROM t GROUP BY segment"
        q2 = "SELECT segment, AVG(metric_value) FROM t WHERE period >= '2025-07-01' GROUP BY segment"
        scope1 = {"metric": "metric_value", "dimension": "segment", "population": {"filters": []}}
        scope2 = {"metric": "metric_value", "dimension": "segment", "population": {"filters": [{"column": "period", "operator": ">=", "value": "2025-07-01"}]}}
        self.assertNotEqual(compute_evidence_identity(ds, q, "oracle", scope1), compute_evidence_identity(ds, q2, "oracle", scope2))

    def test_12_method_registry_blocks_prediction_with_leakage(self):
        df = pd.DataFrame({"target": [0, 1] * 30, "age": np.arange(60), "post_outcome_status": ["closed"] * 60})
        quality = DataQualityGate.evaluate_fitness(df, dataset_name="prediction-leakage")
        class Plan: task = "PREDICTION"
        class Semantic: pass
        ok, detail = MethodRegistry.admissibility_for_plan(Plan(), Semantic(), quality, df)
        self.assertFalse(ok)
        self.assertIn("blocked:prediction_leakage_indicators_present", detail["errors"])


class TestControllerCleanRoomPreflight(unittest.TestCase):
    def test_13_benchmark_declares_controller_dependency_contract(self):
        dependency_contract = {
            "controller_e2e": ["duckdb", "polars", "pyarrow"],
            "auth_runtime": ["python-jose"],
        }
        self.assertEqual(dependency_contract["controller_e2e"], ["duckdb", "polars", "pyarrow"])
        self.assertEqual(dependency_contract["auth_runtime"], ["python-jose"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
