"""Session 11: Real Human Data Analyst Level Intelligence Verification Suite.

Validates human-analyst depth capabilities addressing DEFECT-026:
1. Price-Volume-Mix (PVM) mathematical decomposition (Rate vs Volume).
2. Secondary dimension drilldown within primary change drivers.
3. Intra-period run-rate trajectory dynamics (detecting mid-period step-changes).
4. Epistemic verdict accuracy: NO_DETECTABLE_EFFECT with CI bounds on powered null tests.
5. Pareto (80/20) concentration and breakdown question semantics.
6. Metric aggregation intelligence (intensive vs extensive metrics).
7. Actionable operational next steps tailored to findings.
8. Real-world messy dataset robustness.
9. Mutation pinning for per-day normalization across calendar periods.

Run with: python -m pytest tests/independent_release/test_session11_human_analyst_level.py -v
"""
import hashlib
import os
import sys
import unittest
import uuid

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Investigation, InvestigationVerdict
from packages.analytics_core.src.engines.analyst_answer import (
    build_analyst_result,
    classify_question,
    _choose_agg,
)
from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
from packages.analytics_core.src.runtime.controller import InvestigationController


def _txt(r):
    return r.to_text() if r else ""


class InMemoryProvider(BaseDatasetProvider):
    def __init__(self, d):
        self.d = d

    def acquire_context(self, project_id, dataset_ids=None):
        return InvestigationDataContext(
            project_id=project_id,
            datasets_map=self.d,
            dataset_fingerprints={n: hashlib.sha256(x.to_json().encode()).hexdigest() for n, x in self.d.items()},
            requested_dataset_ids=dataset_ids,
        )


def _run(df, question):
    db = SessionLocal()
    iid = f"INV-S11-{uuid.uuid4().hex[:8]}"
    db.add(Investigation(id=iid, project_id=f"p-{uuid.uuid4().hex[:6]}", question=question, status="PLANNED"))
    db.commit()
    db.close()
    InvestigationController(
        session_factory=SessionLocal,
        dataset_provider=InMemoryProvider({"t": df}),
    ).execute_investigation(investigation_id=iid, worker_id="w-s11")
    db = SessionLocal()
    inv = db.query(Investigation).filter(Investigation.id == iid).first()
    verdict_record = db.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == iid).first()
    out = {
        "verdict_type": str(inv.verdict_type),
        "confidence_score": float(inv.confidence_score) if inv.confidence_score is not None else 0.0,
        "direct_answer": inv.direct_answer or "",
        "main_finding": inv.main_finding or "",
        "verdict_record_type": str(verdict_record.verdict_type) if verdict_record else None,
        "verdict_record_confidence": float(verdict_record.confidence_score) if verdict_record else 0.0,
    }
    db.close()
    return out


class TestPriceVolumeMixDecomposition(unittest.TestCase):
    """Decomposition of period change into Volume effect vs Rate effect."""

    def test_volume_driven_drop_decomposition(self):
        # 2025-05: 100 txns at avg price $50 = $5,000
        # 2025-06: 50 txns at avg price $50 = $2,500
        # Change: -$2,500 entirely driven by volume (-50 txns * $50 = -$2500)
        dates_may = pd.date_range("2025-05-01", "2025-05-31", periods=100)
        dates_jun = pd.date_range("2025-06-01", "2025-06-30", periods=50)
        df = pd.DataFrame({
            "date": list(dates_may) + list(dates_jun),
            "revenue": [50.0] * 150,
            "region": ["North"] * 150,
        })
        r = build_analyst_result("Why did revenue drop in June?", df, target="revenue", group="region", time_col="date")
        self.assertIsNotNone(r)
        self.assertIn("pvm_decomposition", r.numbers)
        pvm = r.numbers["pvm_decomposition"]
        self.assertAlmostEqual(pvm["volume_effect"], -2500.0, delta=100.0)
        self.assertAlmostEqual(pvm["rate_effect"], 0.0, delta=10.0)
        self.assertIn("volume-driven", _txt(r).lower())

    def test_rate_driven_drop_decomposition(self):
        # 2025-05: 100 txns at $100 = $10,000
        # 2025-06: 100 txns at $50 = $5,000
        # Change: -$5,000 entirely driven by rate effect
        dates_may = pd.date_range("2025-05-01", "2025-05-31", periods=100)
        dates_jun = pd.date_range("2025-06-01", "2025-06-30", periods=100)
        df = pd.DataFrame({
            "date": list(dates_may) + list(dates_jun),
            "revenue": [100.0] * 100 + [50.0] * 100,
            "region": ["West"] * 200,
        })
        r = build_analyst_result("Why did revenue drop in June?", df, target="revenue", group="region", time_col="date")
        self.assertIsNotNone(r)
        self.assertIn("pvm_decomposition", r.numbers)
        pvm = r.numbers["pvm_decomposition"]
        self.assertAlmostEqual(pvm["volume_effect"], 0.0, delta=10.0)
        self.assertAlmostEqual(pvm["rate_effect"], -5000.0, delta=100.0)
        self.assertIn("rate-driven", _txt(r).lower())


class TestSecondaryDimensionDrilldown(unittest.TestCase):
    """Testing drilldown into sub-segments of the primary driver segment."""

    def test_secondary_subsegment_drilldown(self):
        rng = np.random.RandomState(42)
        # Create May and June data
        # Region South drops dramatically, driven specifically by sub-category 'Enterprise'
        records = []
        for d in pd.date_range("2025-05-01", "2025-05-31", freq="D"):
            # South May: 20 per day (15 Enterprise, 5 SMB)
            for _ in range(15):
                records.append({"date": d, "region": "South", "tier": "Enterprise", "sales": 100.0})
            for _ in range(5):
                records.append({"date": d, "region": "South", "tier": "SMB", "sales": 50.0})
            # North May: 10 per day
            for _ in range(10):
                records.append({"date": d, "region": "North", "tier": "SMB", "sales": 60.0})

        for d in pd.date_range("2025-06-01", "2025-06-30", freq="D"):
            # South June: Enterprise drops from 15 to 2 per day! SMB stays 5
            for _ in range(2):
                records.append({"date": d, "region": "South", "tier": "Enterprise", "sales": 100.0})
            for _ in range(5):
                records.append({"date": d, "region": "South", "tier": "SMB", "sales": 50.0})
            # North June: stays 10
            for _ in range(10):
                records.append({"date": d, "region": "North", "tier": "SMB", "sales": 60.0})

        df = pd.DataFrame(records)
        r = build_analyst_result(
            "Why did sales drop in June?",
            df,
            target="sales",
            group="region",
            time_col="date",
            candidate_dimensions=["region", "tier"],
        )
        self.assertIsNotNone(r)
        self.assertIn("South", _txt(r))
        self.assertIn("sub_segment_findings", r.numbers)
        sub_findings = r.numbers["sub_segment_findings"]
        self.assertEqual(sub_findings["dimension"], "tier")
        self.assertEqual(sub_findings["top_subsegment"], "Enterprise")
        self.assertIn("Enterprise", _txt(r))


class TestIntraPeriodRunRateTrajectory(unittest.TestCase):
    """Detecting mid-period inflection points and step-changes."""

    def test_sharp_step_change_detected(self):
        records = []
        # May: steady $1,000/day
        for d in pd.date_range("2025-05-01", "2025-05-31", freq="D"):
            records.append({"date": d, "revenue": 1000.0, "market": "US"})
        # June 1-15: steady $1,000/day
        for d in pd.date_range("2025-06-01", "2025-06-15", freq="D"):
            records.append({"date": d, "revenue": 1000.0, "market": "US"})
        # June 16-30: sharp drop to $100/day
        for d in pd.date_range("2025-06-16", "2025-06-30", freq="D"):
            records.append({"date": d, "revenue": 100.0, "market": "US"})

        df = pd.DataFrame(records)
        r = build_analyst_result("Why did revenue drop in June?", df, target="revenue", group="market", time_col="date")
        self.assertIsNotNone(r)
        self.assertIn("intra_period_trajectory", r.numbers)
        traj = r.numbers["intra_period_trajectory"]
        self.assertTrue(traj["step_change_detected"])
        self.assertTrue("2025-06-15" in str(traj["inflection_date"]) or "2025-06-16" in str(traj["inflection_date"]))
        self.assertIn("trajectory", _txt(r).lower())
        self.assertIn("sharp run-rate drop", _txt(r).lower())


class TestParetoConcentrationAndBreakdown(unittest.TestCase):
    """Pareto 80/20 distribution and composition question semantics."""

    def test_pareto_concentration_computed_correctly(self):
        # 100 entities: top 15 have 85% of volume, bottom 85 have 15%
        entities = [f"Client_{i:03d}" for i in range(100)]
        values = [5000.0] * 15 + [30.0] * 85
        df = pd.DataFrame({"client": entities, "revenue": values})

        r = build_analyst_result("What is the breakdown of revenue by client?", df, target="revenue", group="client")
        self.assertIsNotNone(r)
        self.assertTrue(r.descriptive)
        self.assertIn("pareto_top_share", r.numbers)
        self.assertGreaterEqual(r.numbers["pareto_top_share"], 0.80)
        self.assertLessEqual(r.numbers["pareto_80_count"], 20)
        self.assertIn("Pareto concentration (80/20)", _txt(r))

    def test_breakdown_question_classification(self):
        q = "What is the breakdown of sales by region?"
        self.assertEqual(classify_question(q), "RANKING")


class TestMetricAggregationIntelligence(unittest.TestCase):
    """Limit #4: Intelligent aggregation choice for intensive vs extensive metrics."""

    def test_intensive_metrics_default_to_mean(self):
        # Price, rate, latency, score should average, not sum
        self.assertEqual(_choose_agg("Breakdown by region", None, False, target="unit_price"), "mean")
        self.assertEqual(_choose_agg("Breakdown by department", None, False, target="satisfaction_score"), "mean")
        self.assertEqual(_choose_agg("By server", None, False, target="latency_ms"), "mean")
        self.assertEqual(_choose_agg("By channel", None, False, target="churn_rate"), "mean")
        self.assertEqual(_choose_agg("By customer segment", None, False, target="aov"), "mean")

    def test_extensive_metrics_default_to_sum(self):
        # Revenue, volume, spend, sales, cost should sum
        self.assertEqual(_choose_agg("Breakdown by region", None, False, target="revenue"), "sum")
        self.assertEqual(_choose_agg("By product", None, False, target="units_sold"), "sum")
        self.assertEqual(_choose_agg("By campaign", None, False, target="marketing_spend"), "sum")


class TestActionableNextSteps(unittest.TestCase):
    """Ensuring analytical results provide concrete, human-analyst operational next steps."""

    def test_next_steps_present_across_analysis_types(self):
        rng = np.random.RandomState(99)
        df_ranking = pd.DataFrame({"product": ["A", "B", "C"], "sales": [500, 300, 100]})
        r_rank = build_analyst_result("Which product has highest sales?", df_ranking, target="sales", group="product")
        self.assertGreater(len(r_rank.next_steps), 0)
        self.assertIn("Recommended next steps:", _txt(r_rank))

        df_comp = pd.DataFrame({
            "grp": ["A"] * 50 + ["B"] * 50,
            "metric": list(rng.normal(10, 2, 50)) + list(rng.normal(20, 2, 50)),
        })
        r_comp = build_analyst_result("Is metric higher for B than A?", df_comp, target="metric", group="grp")
        self.assertGreater(len(r_comp.next_steps), 0)
        self.assertIn("Recommended next steps:", _txt(r_comp))


class TestEpistemicVerdictAccuracyEndToEnd(unittest.TestCase):
    """Limit #1: NO_DETECTABLE_EFFECT certified on powered null test."""

    def test_powered_null_yields_no_detectable_effect_with_confidence_one(self):
        rng = np.random.RandomState(123)
        n = 500
        df = pd.DataFrame({
            "id": range(2 * n),
            "variant": ["A"] * n + ["B"] * n,
            "score": list(rng.normal(50.0, 5.0, n)) + list(rng.normal(50.0, 5.0, n)),
        })
        res = _run(df, "Does score differ between variant A and variant B?")
        self.assertEqual(res["verdict_type"], "NO_DETECTABLE_EFFECT")
        self.assertGreaterEqual(res["confidence_score"], 0.80)
        self.assertLessEqual(res["confidence_score"], 0.95)
        self.assertEqual(res["verdict_record_type"], "NO_DETECTABLE_EFFECT")
        self.assertGreaterEqual(res["verdict_record_confidence"], 0.80)
        self.assertLessEqual(res["verdict_record_confidence"], 0.95)
        self.assertNotEqual(res["confidence_score"], 1.0)  # Epistemic calibration: never 100% omniscience
        self.assertIn("95% CI", res["direct_answer"])


    def test_underpowered_null_remains_inconclusive(self):
        rng = np.random.RandomState(456)
        n = 4  # Severely underpowered sample
        df = pd.DataFrame({
            "id": range(2 * n),
            "treatment": ["Control"] * n + ["Test"] * n,
            "response": list(rng.normal(10.0, 2.0, n)) + list(rng.normal(10.0, 2.0, n)),
        })
        res = _run(df, "Does response differ between Test and Control?")
        # An underpowered null sample cannot certify NO_DETECTABLE_EFFECT
        self.assertNotEqual(res["verdict_type"], "NO_DETECTABLE_EFFECT")


class TestPerDayNormalizationPin(unittest.TestCase):
    """Limit #2: Mutation pinning for per-day rate normalization across calendar periods."""

    def test_calendar_month_length_difference_properly_normalized(self):
        # Jan 2025: 31 days, exactly 100/day -> total 3,100
        # Feb 2025: 28 days, exactly 100/day -> total 2,800
        # Total drops by -300 (-9.7%), but per-day rate change is 0.0%!
        dates_jan = pd.date_range("2025-01-01", "2025-01-31", freq="D")
        dates_feb = pd.date_range("2025-02-01", "2025-02-28", freq="D")
        df = pd.DataFrame({
            "date": list(dates_jan) + list(dates_feb),
            "sales": [100.0] * len(dates_jan) + [100.0] * len(dates_feb),
            "region": ["HQ"] * (len(dates_jan) + len(dates_feb)),
        })
        r = build_analyst_result("Why did sales drop in February?", df, target="sales", group="region", time_col="date")
        self.assertIsNotNone(r)
        self.assertIn("per_day_change_pct", r.numbers)
        self.assertAlmostEqual(r.numbers["per_day_change_pct"], 0.0, delta=0.01)
        self.assertIn("fewer calendar days", _txt(r).lower())


class TestMessyDataRobustness(unittest.TestCase):
    """Limit #5: Robustness against non-standard, messy, or adversarial datasets."""

    def test_handles_nan_inf_and_strings(self):
        df = pd.DataFrame({
            "segment": ["A", "B", None, "A", "B", float("nan"), "A"],
            "metric": [10.0, np.inf, 20.0, -np.inf, np.nan, 30.0, 40.0],
        })
        r = build_analyst_result("Which segment has higher metric?", df, target="metric", group="segment")
        # Must execute cleanly without unhandled exceptions
        self.assertTrue(r is None or r.kind in ("GROUP_COMPARISON", "RANKING"))

    def test_handles_single_value_group(self):
        df = pd.DataFrame({
            "segment": ["AllSame"] * 10,
            "metric": [10.0] * 10,
        })
        r = build_analyst_result("Which segment is highest?", df, target="metric", group="segment")
        self.assertIsNone(r)


class TestSession11APIAnalystResultSurface(unittest.TestCase):
    """Verifies that human-analyst structured intelligence is surfaced by the API."""

    def setUp(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from apps.api.src.core.database import Base
        from apps.api.src.models.entities import User, Project, gen_uuid

        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

        self.user = User(id=gen_uuid(), email="analyst@example.com", hashed_password="x", full_name="Analyst", role="analyst")
        self.db.add(self.user)
        self.project = Project(id=gen_uuid(), owner_id=self.user.id, name="Analyst Project")
        self.db.add(self.project)
        self.db.flush()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_get_investigation_surfaces_analyst_result_when_present(self):
        from apps.api.src.models.entities import InvestigationEvent
        from apps.api.src.api.v1.investigations import get_investigation, get_investigation_explanation

        inv = Investigation(
            id="INV-ANALYST-SURFACE-001",
            project_id=self.project.id,
            user_id=self.user.id,
            question="Why did revenue drop in June?",
            status="COMPLETED",
            verdict_type="OBSERVED",
            direct_answer="Revenue fell $50k driven by West region volume decline.",
            main_finding="Revenue fell $50k driven by West region volume decline.",
        )
        self.db.add(inv)
        self.db.flush()

        # Add analyst result event
        mock_analyst_payload = {
            "kind": "TIME_SERIES_DRIVER",
            "headline": "Revenue fell $50,000 (-12.5%) from May to June",
            "pvm_decomposition": {
                "total_delta": -50000.0,
                "price_effect": 10000.0,
                "volume_effect": -60000.0,
                "primary_effect": "volume",
            },
            "sub_segment_findings": [
                {"segment": "West", "sub_dimension": "product_line", "top_sub_segment": "Enterprise", "sub_delta": -45000.0}
            ],
            "next_steps": [
                "Investigate enterprise churn in West region",
            ],
        }
        event = InvestigationEvent(
            investigation_id=inv.id,
            event_type="investigation.analyst_result",
            sequence=1,
            event_payload_json=mock_analyst_payload,
        )
        self.db.add(event)
        self.db.flush()

        resp = get_investigation(inv.id, current_user=self.user, db=self.db)
        self.assertIn("analyst_result", resp)
        self.assertIsNotNone(resp["analyst_result"])
        self.assertEqual(resp["analyst_result"]["kind"], "TIME_SERIES_DRIVER")
        self.assertEqual(resp["analyst_result"]["pvm_decomposition"]["primary_effect"], "volume")

        exp_resp = get_investigation_explanation(inv.id, mode="DETERMINISTIC", current_user=self.user, db=self.db)
        self.assertIn("analyst_result", exp_resp)
        self.assertIsNotNone(exp_resp["analyst_result"])
        self.assertEqual(exp_resp["analyst_result"]["sub_segment_findings"][0]["segment"], "West")

    def test_get_investigation_handles_absent_analyst_result(self):
        from apps.api.src.api.v1.investigations import get_investigation, get_investigation_explanation

        inv = Investigation(
            id="INV-ANALYST-SURFACE-002",
            project_id=self.project.id,
            user_id=self.user.id,
            question="What is total volume?",
            status="COMPLETED",
        )
        self.db.add(inv)
        self.db.flush()

        resp = get_investigation(inv.id, current_user=self.user, db=self.db)
        self.assertIn("analyst_result", resp)
        self.assertIsNone(resp["analyst_result"])

        exp_resp = get_investigation_explanation(inv.id, mode="DETERMINISTIC", current_user=self.user, db=self.db)
        self.assertIn("analyst_result", exp_resp)
        self.assertIsNone(exp_resp["analyst_result"])


if __name__ == "__main__":
    unittest.main()
