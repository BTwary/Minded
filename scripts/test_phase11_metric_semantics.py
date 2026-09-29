"""
PHASE 11 BENCHMARK: Metric Semantics Engine.

Two tiers of tests, per the Phase 11 spec:

TIER 1 (Part 11) -- unit-level ground-truth fixtures for all 8 required
metric types (SUM, MEAN, COUNT, DISTINCT_COUNT, RATE, RATIO, PROPORTION,
WEIGHTED_MEAN), each with an independently-computed expected result and at
least one fixture containing irrelevant columns, missing values, misleading
numeric fields, and names alone insufficient to determine aggregation. These
exercise MetricSemanticsResolver + MetricSemanticsResolver.sql_aggregation_expression
directly (both the classification logic AND the SQL it generates), executed
against DuckDB so the "ground truth" and the "system's own SQL" are two
independent computations, not the same expression run twice.

TIER 2 (Part 10 / Part 12) -- full end-to-end runs through the REAL,
unmodified InvestigationController (following the pattern established by
scripts/test_autonomous_controller_messy_input.py), proving that metric
semantics actually reach the production autonomous loop rather than being a
decorative schema nothing reads:
  (a) a SUM-metric investigation (revenue) reaches the correct SUM-based
      concentration verdict, exactly as pre-Phase-11 behavior did (no
      regression for additive metrics).
  (b) a MEAN-metric investigation over THE SAME underlying data proves the
      controller reaches a MATERIALLY DIFFERENT, and separately correct,
      conclusion -- the segment with by far the highest total revenue does
      NOT have the highest average order value, and the controller must not
      report a "share of total" for it. This is the Part 12 acceptance test:
      the system must actually understand metric semantics, not just carry
      a MetricDefinition object that nothing consults.

Scope note (documented per Part 17, not hidden): full end-to-end coverage of
all 8 metric types through the natural-language controller path is bounded by
the free-text question-to-column scoring logic in SemanticEngine (unchanged
by Phase 11, out of scope to rewrite here) reliably selecting a specific
target column. Tier 1 gives every metric type an independently-verified
ground-truth fixture; Tier 2 proves the wiring reaches the real controller
for the two cases -- SUM and a non-additive type -- where getting it wrong
(collapsing everything to SUM) would be most consequential and most exactly
the Part 12 acceptance bar ("SUM and MEAN/RATE/RATIO produce materially
different conclusions").

Run: python3 scripts/test_phase11_metric_semantics.py
"""
import os
import sys
import unittest
from typing import ClassVar

import duckdb
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from apps.api.src.models.entities import (
    Base, Experiment, Hypothesis, Investigation, InvestigationVerdict,
    Observation, Project, User, gen_uuid,
)
from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider
from packages.analytics_core.src.execution.state_machine import InvestigationState
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.semantic.metric_semantics import (
    MetricSemanticsResolver,
)
from packages.schemas.src.analysis import AggregationType

import packages.analytics_core.src.runtime.state as state_module
import packages.analytics_core.src.runtime.controller as controller_module

REAL_STATE_MANAGER_CLS = state_module.InvestigationStateManager
REAL_CANONICAL_STATE_CLS = state_module.CanonicalInvestigationState


class _WitnessCanonicalState(REAL_CANONICAL_STATE_CLS):
    instantiations: ClassVar[int] = 0

    def __init__(self, **data):
        _WitnessCanonicalState.instantiations += 1
        super().__init__(**data)


class _WitnessStateManager(REAL_STATE_MANAGER_CLS):
    instantiations = 0

    def __init__(self, *args, **kwargs):
        _WitnessStateManager.instantiations += 1
        super().__init__(*args, **kwargs)


def _run_via_duckdb(df: pd.DataFrame, sql: str):
    con = duckdb.connect(database=":memory:")
    con.register("data_table", df)
    result = con.execute(sql).fetchdf()
    con.close()
    return result


# ===========================================================================
# TIER 1: Unit-level ground-truth fixtures for all 8 required metric types.
# ===========================================================================

class TestMetricSemanticsResolverGroundTruth(unittest.TestCase):
    """Each fixture includes irrelevant columns, some missing values, multiple
    categories, and at least one column whose name alone would not tell you
    the correct aggregation -- so the resolver must actually reason about
    range/companions, not just pattern-match an obvious suffix in isolation.
    """

    # --- 1. SUM ------------------------------------------------------------
    def test_sum_metric(self):
        df = pd.DataFrame({
            "region": ["East", "East", "West", "West", "North"],
            "total_revenue": [1000.0, 1500.0, np.nan, 800.0, 2200.0],
            "notes": ["ok", "ok", "late", None, "ok"],  # irrelevant column
        })
        m = MetricSemanticsResolver.resolve("total_revenue", df, table_name="sales")
        self.assertEqual(m.aggregation_type, AggregationType.SUM)
        self.assertTrue(m.is_additive)

        sql = MetricSemanticsResolver.sql_aggregation_expression(m, "total_metric")
        got = _run_via_duckdb(df, f"SELECT {sql} FROM data_table")["total_metric"].iloc[0]
        ground_truth = df["total_revenue"].sum()  # pandas independently: 5500.0 (NaN skipped)
        self.assertAlmostEqual(float(got), float(ground_truth), places=6)
        self.assertAlmostEqual(float(ground_truth), 5500.0, places=6)

    # --- 2. MEAN -------------------------------------------------------------
    def test_mean_metric_no_weight_column(self):
        # Column name alone ("avg_order_value") signals MEAN; no plausible
        # weight/volume column exists in this table, so it must be a PLAIN
        # mean -- not silently summed, not silently weighted by a column that
        # isn't actually a volume measure (order_notes is text, not a weight).
        df = pd.DataFrame({
            "channel": ["Web", "Web", "Retail", "Retail", "Retail"],
            "avg_order_value": [42.0, 58.0, np.nan, 120.0, 110.0],
            "order_notes": ["gift", "std", "std", "gift", "std"],
        })
        m = MetricSemanticsResolver.resolve("avg_order_value", df, table_name="orders")
        self.assertEqual(m.aggregation_type, AggregationType.MEAN)
        self.assertIsNone(m.weight_column)
        self.assertFalse(m.is_additive)

        sql = MetricSemanticsResolver.sql_aggregation_expression(m, "mean_metric")
        got = _run_via_duckdb(df, f"SELECT {sql} FROM data_table")["mean_metric"].iloc[0]
        ground_truth = df["avg_order_value"].mean()  # independently: (42+58+120+110)/4 = 82.5
        self.assertAlmostEqual(float(got), float(ground_truth), places=6)
        self.assertAlmostEqual(float(ground_truth), 82.5, places=6)

    # --- 3. COUNT (dense 0/1 occurrence flag; max<=1 -> COUNT still correct) ---
    def test_count_metric_from_how_many_question_over_occurrence_flag(self):
        # had_incident is a dense 0/1 flag (every row non-null): COUNT is
        # still the right answer here, distinguishing this from the
        # event-volume case below where a row can hold >1 occurrence.
        df = pd.DataFrame({
            "service": ["auth", "auth", "billing", "billing", "billing"],
            "had_incident": [1.0, 0.0, 1.0, 0.0, 1.0],
            "region": ["us", "us", "eu", "eu", "apac"],  # irrelevant column
        })
        m = MetricSemanticsResolver.resolve(
            "had_incident", df, table_name="incidents",
            question_tokens={"how", "many", "had_incident", "occurred"},
        )
        self.assertEqual(m.aggregation_type, AggregationType.COUNT)
        self.assertFalse(m.is_additive)
        self.assertEqual(m.semantic_resolution_status, "RESOLVED")

        sql = MetricSemanticsResolver.sql_aggregation_expression(m, "n")
        got = _run_via_duckdb(df, f"SELECT {sql} FROM data_table")["n"].iloc[0]
        ground_truth = df["had_incident"].count()  # independently: 5 non-null rows
        self.assertEqual(int(got), int(ground_truth))
        self.assertEqual(int(ground_truth), 5)

    # --- 3b. SUM (event-volume column; max>1 -> SUM, not COUNT) [v1.1] --------
    def test_sum_metric_from_how_many_question_over_event_volume_column(self):
        df = pd.DataFrame({
            "service": ["auth", "auth", "billing", "billing", "billing"],
            "error_events": [3.0, np.nan, 1.0, 2.0, 4.0],
            "region": ["us", "us", "eu", "eu", "apac"],  # irrelevant column
        })
        m = MetricSemanticsResolver.resolve(
            "error_events", df, table_name="incidents",
            question_tokens={"how", "many", "error_events", "occurred"},
        )
        self.assertEqual(m.aggregation_type, AggregationType.SUM)
        self.assertTrue(m.is_additive)
        self.assertEqual(m.semantic_type, "event_volume_sum")
        self.assertEqual(m.semantic_resolution_status, "RESOLVED")

        sql = MetricSemanticsResolver.sql_aggregation_expression(m, "n")
        got = _run_via_duckdb(df, f"SELECT {sql} FROM data_table")["n"].iloc[0]
        ground_truth = df["error_events"].sum()  # independently: 3+1+2+4 = 10.0 (NaN skipped)
        self.assertAlmostEqual(float(got), float(ground_truth), places=6)
        self.assertAlmostEqual(float(ground_truth), 10.0, places=6)

    # --- 3c. Boundary: max exactly 1.0 must still resolve to COUNT [v1.1] -----
    def test_count_boundary_max_value_exactly_one(self):
        df = pd.DataFrame({
            "service": ["auth", "billing", "billing"],
            "flag_col": [1.0, 0.0, 1.0],  # max is exactly 1.0, not > 1.0
        })
        m = MetricSemanticsResolver.resolve(
            "flag_col", df, table_name="incidents",
            question_tokens={"how", "many"},
        )
        self.assertEqual(m.aggregation_type, AggregationType.COUNT)
        self.assertEqual(m.semantic_resolution_status, "RESOLVED")

    # --- 4. DISTINCT_COUNT ----------------------------------------------------
    def test_distinct_count_metric(self):
        df = pd.DataFrame({
            "region": ["us", "us", "eu", "eu", "eu"],
            "customer_id": ["C1", "C2", "C2", "C3", "C3"],
            "plan_notes": ["gold", "gold", "gold", "silver", "silver"],
        })
        m = MetricSemanticsResolver.resolve("customer_id", df, table_name="accounts")
        self.assertEqual(m.aggregation_type, AggregationType.COUNT_DISTINCT)
        self.assertFalse(m.is_additive)

        sql = MetricSemanticsResolver.sql_aggregation_expression(m, "n_distinct")
        got = _run_via_duckdb(df, f"SELECT {sql} FROM data_table")["n_distinct"].iloc[0]
        ground_truth = df["customer_id"].nunique()  # independently: {C1,C2,C3} = 3
        self.assertEqual(int(got), int(ground_truth))
        self.assertEqual(int(ground_truth), 3)

    # --- 5. RATE (full numerator/denominator recomposition) -------------------
    def test_rate_metric_with_true_numerator_denominator(self):
        # conversion_rate is precomputed PER ROW (and is intentionally
        # unequal-weighted across rows -- a naive AVG of the rate column
        # would give a materially wrong answer versus the true weighted rate).
        df = pd.DataFrame({
            "channel": ["Email", "Email", "Paid", "Paid"],
            "conversions": [40.0, 5.0, 100.0, 2.0],
            "sessions": [10000.0, 100.0, 500.0, 20.0],
            "conversion_rate": [0.004, 0.05, 0.20, 0.10],
            "campaign_notes": ["Q1 promo", None, "brand", "retarget"],
        })
        m = MetricSemanticsResolver.resolve("conversion_rate", df, table_name="marketing")
        self.assertEqual(m.aggregation_type, AggregationType.RATE)
        self.assertEqual(m.numerator_column, "conversions")
        self.assertEqual(m.denominator_column, "sessions")
        self.assertTrue(m.is_compositional)
        self.assertFalse(m.is_additive)

        sql = MetricSemanticsResolver.sql_aggregation_expression(m, "true_rate")
        got = _run_via_duckdb(df, f"SELECT {sql} FROM data_table")["true_rate"].iloc[0]
        ground_truth = df["conversions"].sum() / df["sessions"].sum()  # independently: 147/10620
        naive_avg_of_rate_column = df["conversion_rate"].mean()  # WRONG answer a SUM/AVG-of-column approach would give
        self.assertAlmostEqual(float(got), float(ground_truth), places=6)
        self.assertAlmostEqual(float(ground_truth), 147.0 / 10620.0, places=6)
        # Prove the correct answer is materially different from the naive
        # (wrong) approach of just averaging the precomputed rate column --
        # this is the crux of Part 8 (denominators must be preserved).
        self.assertGreater(abs(ground_truth - naive_avg_of_rate_column), 0.02)

    # --- 6. RATIO (non-percentage, can exceed 1; honest degradation to weighted mean) ---
    def test_ratio_metric_honest_degradation_to_weighted_mean(self):
        # cost_to_revenue_ratio is NOT bounded to [0,1]/[0,100] and no raw
        # numerator ("cost") + denominator ("revenue") pair exists in this
        # table (only "order_volume" as a plausible weight) -- the resolver
        # must NOT fabricate a denominator it cannot observe; it must degrade
        # honestly to a volume-weighted mean and say so in its rationale.
        df = pd.DataFrame({
            "business_unit": ["Retail", "Retail", "Wholesale", "Wholesale"],
            "cost_to_revenue_ratio": [1.8, 2.1, 0.6, 0.5],
            "order_volume": [10.0, 5.0, 900.0, 850.0],
            "manager_notes": ["reviewed", None, "reviewed", "reviewed"],
        })
        m = MetricSemanticsResolver.resolve("cost_to_revenue_ratio", df, table_name="finance")
        # Design intent: the metric's SEMANTIC TYPE is still "ratio" (it is
        # named and behaves like a ratio), but since no raw numerator/
        # denominator pair is observable, the AGGREGATION OPERATOR honestly
        # degrades to a volume-weighted mean rather than fabricating a ratio
        # recomposition it cannot actually perform.
        self.assertEqual(m.semantic_type, "ratio")
        self.assertEqual(m.aggregation_type, AggregationType.WEIGHTED_MEAN)
        self.assertIsNone(m.numerator_column)  # honestly not recomposable
        self.assertEqual(m.weight_column, "order_volume")
        self.assertFalse(m.is_additive)
        self.assertIn("no raw numerator/denominator pair", m.rationale)

        sql = MetricSemanticsResolver.sql_aggregation_expression(m, "weighted_ratio")
        got = _run_via_duckdb(df, f"SELECT {sql} FROM data_table")["weighted_ratio"].iloc[0]
        weighted_sum = (df["cost_to_revenue_ratio"] * df["order_volume"]).sum()
        ground_truth = weighted_sum / df["order_volume"].sum()  # independently computed
        self.assertAlmostEqual(float(got), float(ground_truth), places=6)
        # And prove it's materially different from a naive unweighted mean --
        # otherwise the weighting wouldn't matter.
        naive_mean = df["cost_to_revenue_ratio"].mean()
        self.assertGreater(abs(ground_truth - naive_mean), 0.3)

    # --- 7. PROPORTION ---------------------------------------------------------
    def test_proportion_metric_bounded_unit_interval(self):
        # Name alone ("defect_share") doesn't obviously say "proportion" the
        # way "_rate" or "_pct" would -- the bounded-[0,1] range plus multiple
        # distinct values is what correctly triggers non-additive treatment.
        df = pd.DataFrame({
            "plant": ["A", "A", "B", "B", "C"],
            "defect_share": [0.02, 0.03, 0.31, 0.29, 0.05],
            "units_inspected": [500.0, 400.0, 50.0, 60.0, 300.0],
            "inspector_notes": ["ok", "ok", "flag", "flag", "ok"],
        })
        m = MetricSemanticsResolver.resolve("defect_share", df, table_name="qc")
        self.assertIn(m.aggregation_type, (AggregationType.PROPORTION, AggregationType.WEIGHTED_MEAN))
        self.assertFalse(m.is_additive)
        # Whichever exact type, it must be weighted by units_inspected, since
        # that's the only plausible exposure/volume column in this table.
        self.assertEqual(m.weight_column, "units_inspected")

        sql = MetricSemanticsResolver.sql_aggregation_expression(m, "wshare")
        got = _run_via_duckdb(df, f"SELECT {sql} FROM data_table")["wshare"].iloc[0]
        weighted_sum = (df["defect_share"] * df["units_inspected"]).sum()
        ground_truth = weighted_sum / df["units_inspected"].sum()
        self.assertAlmostEqual(float(got), float(ground_truth), places=6)
        naive_mean = df["defect_share"].mean()
        # The units-inspected-weighted defect share must differ materially
        # from an unweighted mean across plants, since inspection volumes
        # vary by an order of magnitude between plants.
        self.assertGreater(abs(ground_truth - naive_mean), 0.02)

    # --- 8. WEIGHTED_MEAN --------------------------------------------------
    def test_weighted_mean_metric(self):
        df = pd.DataFrame({
            "cluster": ["c1", "c1", "c2", "c2"],
            "average_utilization": [0.20, 0.90, 0.55, 0.60],
            "server_count": [1.0, 50.0, 25.0, 25.0],
            "owner_notes": ["temp", "prod", "prod", "prod"],
        })
        m = MetricSemanticsResolver.resolve("average_utilization", df, table_name="infra")
        self.assertEqual(m.aggregation_type, AggregationType.WEIGHTED_MEAN)
        self.assertEqual(m.weight_column, "server_count")
        self.assertFalse(m.is_additive)

        sql = MetricSemanticsResolver.sql_aggregation_expression(m, "wmean")
        got = _run_via_duckdb(df, f"SELECT {sql} FROM data_table")["wmean"].iloc[0]
        weighted_sum = (df["average_utilization"] * df["server_count"]).sum()
        ground_truth = weighted_sum / df["server_count"].sum()  # independently: dominated by the 50- and 25-server rows
        naive_mean = df["average_utilization"].mean()
        self.assertAlmostEqual(float(got), float(ground_truth), places=6)
        # The single 1-server "temp" row (utilization 0.20) must NOT drag the
        # weighted average down anywhere near as much as an unweighted mean
        # would -- proving the weighting is actually applied.
        self.assertGreater(ground_truth, naive_mean)

    # --- 9. UNRESOLVED_DEFAULT_SUM (Phase 11 finalization, audit Section 6) ---
    def test_ambiguous_numeric_column_default_sum_is_marked_unresolved(self):
        # A bare "value" column has no rate/mean/id/additive-magnitude naming
        # signal and no bounded-proportion range: SUM is still used (so the
        # investigation isn't blocked), but it must be distinguishable, by a
        # machine-readable field, from a confidently-named metric like
        # total_revenue.
        df_ambiguous = pd.DataFrame({
            "segment": ["A", "A", "B", "B"],
            "value": [10.0, 20.0, 5.0, 7.0],
        })
        m_ambiguous = MetricSemanticsResolver.resolve("value", df_ambiguous, table_name="misc")
        self.assertEqual(m_ambiguous.aggregation_type, AggregationType.SUM)
        self.assertEqual(m_ambiguous.semantic_resolution_status, "UNRESOLVED_DEFAULT_SUM")

        df_confident = pd.DataFrame({
            "segment": ["A", "A", "B", "B"],
            "total_revenue": [10.0, 20.0, 5.0, 7.0],
        })
        m_confident = MetricSemanticsResolver.resolve("total_revenue", df_confident, table_name="sales")
        self.assertEqual(m_confident.aggregation_type, AggregationType.SUM)
        self.assertEqual(m_confident.semantic_resolution_status, "RESOLVED")

        # Distinct in both provenance and narrative form, not just internally.
        self.assertNotEqual(
            m_ambiguous.to_provenance_dict()["semantic_resolution_status"],
            m_confident.to_provenance_dict()["semantic_resolution_status"],
        )
        self.assertIn("unconfirmed default", m_ambiguous.narrative_explanation())
        self.assertNotIn("unconfirmed default", m_confident.narrative_explanation())


# ===========================================================================
# TIER 2: Real InvestigationController end-to-end runs (Part 10 / Part 12).
# ===========================================================================

def _build_dual_metric_dataset() -> pd.DataFrame:
    """One dataset, two metrics, two different correct conclusions:
      - 'revenue' (additive/SUM): Enterprise dominates total revenue purely
        because of transaction VOLUME (300 transactions).
      - 'avg_order_value' (non-additive/MEAN): Enterprise's average order
        size is the SMALLEST of the three segments -- SMB and MidMarket have
        much higher per-order value, they just transact far less often.
    A system that silently treats every numeric column as SUM would wrongly
    conclude Enterprise also has the highest average order value, since it
    has the highest SUM of avg_order_value too (300 rows * ~100 each).
    """
    np.random.seed(11)
    rows = []
    for _ in range(500):
        rows.append(("Enterprise", float(np.random.normal(100.0, 8.0))))
    for _ in range(15):
        rows.append(("SMB", float(np.random.normal(500.0, 20.0))))
    for _ in range(15):
        rows.append(("MidMarket", float(np.random.normal(480.0, 20.0))))
    df = pd.DataFrame(rows, columns=["tier_segment", "avg_order_value"])
    df["revenue"] = df["avg_order_value"] * 1.0  # per-transaction revenue for this row
    df["account_id"] = [f"ACC_{i:04d}" for i in range(len(df))]
    df["billing_region"] = np.random.choice(["us-east", "us-west", "eu-central"], len(df))
    return df


def _run_controller(df: pd.DataFrame, question: str, inv_prefix: str):
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine)

    _WitnessCanonicalState.instantiations = 0
    _WitnessStateManager.instantiations = 0
    orig_canonical = state_module.CanonicalInvestigationState
    orig_state_mgr = controller_module.InvestigationStateManager
    state_module.CanonicalInvestigationState = _WitnessCanonicalState
    controller_module.InvestigationStateManager = _WitnessStateManager
    try:
        with SessionFactory() as session:
            user = User(id=f"usr-{inv_prefix}", email=f"{inv_prefix}@aaos.ai", hashed_password="pw",
                        full_name="Phase11 Tester", is_active=True)
            proj = Project(id=f"prj-{inv_prefix}", name="Phase11 Project", description="", owner_id=user.id)
            session.add_all([user, proj])
            session.commit()

            inv_id = f"INV-{inv_prefix}-{gen_uuid()[:8]}"
            inv = Investigation(
                id=inv_id, project_id=proj.id, user_id=user.id,
                question=question, status=InvestigationState.PLANNED,
            )
            session.add(inv)
            session.commit()

        ds_provider = InMemoryDatasetProvider({"sales": df})
        controller = InvestigationController(session_factory=SessionFactory, dataset_provider=ds_provider)
        ok = controller.execute_investigation(investigation_id=inv_id, worker_id=f"w-{inv_prefix}")
        return ok, inv_id, SessionFactory
    finally:
        state_module.CanonicalInvestigationState = orig_canonical
        controller_module.InvestigationStateManager = orig_state_mgr


class TestControllerMetricSemanticsE2E(unittest.TestCase):
    """Runs the real, unmodified InvestigationController -- proving metric
    semantics reach the production loop (Part 6) and that SUM vs a
    non-additive metric produce materially different, both-correct
    conclusions from the SAME underlying data (Part 12)."""

    def test_sum_metric_investigation_no_regression(self):
        df = _build_dual_metric_dataset()
        gt_total = df.groupby("tier_segment")["revenue"].sum().sort_values(ascending=False)
        gt_top_segment, gt_top_share = gt_total.index[0], float(gt_total.iloc[0] / gt_total.sum())
        self.assertEqual(gt_top_segment, "Enterprise")
        self.assertGreater(gt_top_share, 0.70)

        ok, inv_id, SessionFactory = _run_controller(
            df, "Why did revenue surge across tier_segment?", "sum",
        )
        self.assertTrue(ok)
        with SessionFactory() as session:
            inv = session.query(Investigation).filter(Investigation.id == inv_id).first()
            self.assertEqual(inv.status, InvestigationState.COMPLETED)

            exps = session.query(Experiment).filter(Experiment.investigation_id == inv_id).all()
            self.assertGreaterEqual(len(exps), 1)
            metric_semantics_seen = [
                (e.arguments_json or {}).get("metric_semantics") for e in exps
                if (e.arguments_json or {}).get("metric_semantics")
            ]
            self.assertTrue(metric_semantics_seen, "No experiment recorded Phase 11 metric_semantics provenance.")
            agg_types = {ms["aggregation_type"] for ms in metric_semantics_seen}
            self.assertIn("sum", agg_types, f"Expected SUM aggregation recorded in provenance, got {agg_types}")

            obs = session.query(Observation).join(
                Experiment, Observation.experiment_id == Experiment.id
            ).filter(Experiment.investigation_id == inv_id).all()
            structured_results = [
                (o.result_json or {}).get("structured_result") for o in obs
                if (o.result_json or {}).get("structured_result")
            ]
            share_results = [r for r in structured_results if "top_share_pct" in r]
            self.assertTrue(share_results, "SUM investigation should carry at least one share-of-total observation.")

            verdict = session.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == inv_id).first()
            self.assertIsNotNone(verdict)
            self.assertIsNotNone(inv.main_finding)

    def test_mean_metric_investigation_materially_different_conclusion(self):
        """The Phase 11 Part 12 acceptance test: SAME data, DIFFERENT metric,
        MUST reach a materially different and separately-correct conclusion,
        and must NOT claim a 'share of total' for a non-additive metric."""
        df = _build_dual_metric_dataset()
        gt_mean = df.groupby("tier_segment")["avg_order_value"].mean().sort_values(ascending=False)
        gt_top_by_mean = gt_mean.index[0]
        # Ground truth: Enterprise (which dominates total revenue) must NOT
        # be the segment with the highest average order value.
        self.assertNotEqual(gt_top_by_mean, "Enterprise")

        ok, inv_id, SessionFactory = _run_controller(
            df, "Why does avg_order_value vary across tier_segment?", "mean",
        )
        self.assertTrue(ok)
        with SessionFactory() as session:
            inv = session.query(Investigation).filter(Investigation.id == inv_id).first()
            self.assertEqual(inv.status, InvestigationState.COMPLETED)

            exps = session.query(Experiment).filter(Experiment.investigation_id == inv_id).all()
            metric_semantics_seen = [
                (e.arguments_json or {}).get("metric_semantics") for e in exps
                if (e.arguments_json or {}).get("metric_semantics")
            ]
            self.assertTrue(metric_semantics_seen, "No experiment recorded Phase 11 metric_semantics provenance.")
            agg_types = {ms["aggregation_type"] for ms in metric_semantics_seen}
            # The controller must NOT have silently collapsed this to SUM.
            self.assertNotIn("sum", agg_types, f"Controller silently treated avg_order_value as SUM: {agg_types}")
            self.assertTrue(
                agg_types & {"mean", "weighted_mean"},
                f"Expected MEAN/WEIGHTED_MEAN aggregation, got {agg_types}",
            )

            obs = session.query(Observation).join(
                Experiment, Observation.experiment_id == Experiment.id
            ).filter(Experiment.investigation_id == inv_id).all()
            structured_results = [
                (o.result_json or {}).get("structured_result") for o in obs
                if (o.result_json or {}).get("structured_result")
            ]
            # Part 12: must NOT report a share-of-total for this non-additive metric.
            share_results = [r for r in structured_results if "top_share_pct" in r]
            self.assertFalse(
                share_results,
                f"Non-additive metric investigation wrongly reported a share-of-total: {share_results}",
            )
            # It MUST instead report the correct "highest value" framing.
            highest_value_results = [r for r in structured_results if r.get("is_additive") is False]
            self.assertTrue(highest_value_results, "Expected an is_additive=False structured result for the mean metric.")
            reported_top = {r.get("top_value") for r in highest_value_results}
            self.assertNotIn(
                "Enterprise", reported_top,
                "Controller wrongly reported Enterprise as top by average order value "
                "(this is the exact SUM-vs-MEAN conflation Phase 11 must prevent).",
            )

            verdict = session.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == inv_id).first()
            self.assertIsNotNone(verdict)
            # Part 15: the human-facing finding must explain the aggregation choice.
            self.assertTrue(
                any(kw in (inv.main_finding or "") for kw in ("averaged", "mean", "MEAN", "weighted")),
                f"main_finding does not explain the metric-semantics decision: {inv.main_finding!r}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
