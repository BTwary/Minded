"""Session 10: the numbers-first analyst answer layer and the relation-regex fix.

Ground truth is planted, so each assertion checks a number/direction the data really has.
Run with: python -m pytest tests/independent_release/test_session10_analyst_answer.py
"""
import hashlib
import os
import sys
import unittest
import uuid

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("AAOS_BUSINESS_TIMEZONE", "UTC")

from apps.api.src.core.database import SessionLocal, engine
from apps.api.src.models.entities import Base, Investigation
from packages.analytics_core.src.engines.analyst_answer import build_analyst_result, parse_period
from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
from packages.analytics_core.src.runtime.controller import InvestigationController

Base.metadata.create_all(bind=engine)


def _txt(r):
    return r.to_text() if r else ""


class TestGroupComparison(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(11)

    def test_signal_reports_both_means_direction_and_interval(self):
        n = 800
        g = self.rng.choice(["A", "B"], n)
        y = self.rng.normal(100, 20, n) + np.where(g == "B", 15, 0)
        r = build_analyst_result("Is average spend higher for B than A?", pd.DataFrame({"g": g, "spend": y}), target="spend", group="g")
        self.assertEqual(r.kind, "GROUP_COMPARISON")
        self.assertTrue(_txt(r).startswith("Yes"))
        self.assertGreater(r.numbers["diff"], 10)
        self.assertLess(r.numbers["ci_low"], r.numbers["diff"])
        self.assertGreater(r.numbers["ci_high"], r.numbers["diff"])
        self.assertGreater(r.numbers["ci_low"], 0)
        self.assertLess(r.numbers["p_value"], 0.001)
        self.assertEqual(r.finding, "positive")

    def test_opposite_direction_is_not_confirmed(self):
        n = 800
        g = self.rng.choice(["A", "B"], n)
        y = self.rng.normal(100, 20, n) + np.where(g == "A", 15, 0)
        r = build_analyst_result("Is average spend higher for B than A?", pd.DataFrame({"g": g, "spend": y}), target="spend", group="g")
        self.assertTrue(_txt(r).startswith("No, the opposite"))

    def test_null_is_reported_with_interval_not_as_a_finding(self):
        n = 800
        g = self.rng.choice(["A", "B"], n)
        r = build_analyst_result("Does spend differ by g?", pd.DataFrame({"g": g, "spend": self.rng.normal(100, 20, n)}), target="spend", group="g")
        self.assertIn("no statistically detectable difference", _txt(r).lower())
        self.assertIn("95% CI", _txt(r))
        self.assertEqual(r.finding, "none")

    def test_outlier_driven_mean_difference_is_flagged_not_robust(self):
        n = 400
        g = np.array(["X"] * n + ["Y"] * n)
        y = self.rng.normal(50, 10, 2 * n)
        y[:6] = 20000  # six absurd values in X
        r = build_analyst_result("Do X customers spend more than Y?", pd.DataFrame({"g": g, "spend": y}), target="spend", group="g")
        self.assertIn("extreme value", _txt(r))
        self.assertNotEqual(r.finding, "positive")

    def test_rate_comparison_uses_rates_not_shares(self):
        n = 6000
        v = self.rng.choice(["A", "B"], n)
        c = (self.rng.rand(n) < np.where(v == "B", 0.15, 0.10)).astype(int)
        r = build_analyst_result("Is the conversion rate higher for B than A?", pd.DataFrame({"v": v, "conv": c}), target="conv", group="v")
        self.assertEqual(r.kind, "RATE_COMPARISON")
        self.assertGreater(r.numbers["rates"]["B"], r.numbers["rates"]["A"])
        self.assertIn("percentage points", _txt(r))
        self.assertTrue(_txt(r).startswith("Yes"))

    def test_rate_null(self):
        n = 3000
        v = self.rng.choice(["A", "B"], n)
        c = (self.rng.rand(n) < 0.1).astype(int)
        r = build_analyst_result("Is the conversion rate higher for B than A?", pd.DataFrame({"v": v, "conv": c}), target="conv", group="v")
        self.assertTrue(_txt(r).startswith("No --"))


class TestRanking(unittest.TestCase):
    def test_ranking_is_descriptive_and_correct(self):
        rng = np.random.RandomState(3)
        cat = rng.choice(["Books", "Toys", "Games"], 900, p=[0.1, 0.3, 0.6])
        df = pd.DataFrame({"cat": cat, "sales": rng.gamma(3, 30, 900)})
        truth = df.groupby("cat").sales.sum().idxmax()
        r = build_analyst_result("Which category had the highest total sales?", df, target="sales", group="cat")
        self.assertTrue(r.descriptive)
        self.assertEqual(r.numbers["top"], truth)
        self.assertTrue(_txt(r).startswith(f"{truth} has the highest"))

    def test_lowest_reverses_order(self):
        df = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 10, "v": [1.0] * 10 + [5.0] * 10})
        r = build_analyst_result("Which c has the lowest average v?", df, target="v", group="c")
        self.assertEqual(r.numbers["top"], "a")


class TestAssociationAndTrend(unittest.TestCase):
    def test_correlation_reports_r_and_refuses_causal_language(self):
        rng = np.random.RandomState(4)
        x = rng.gamma(4, 50, 900)
        df = pd.DataFrame({"x": x, "y": 3 * x + rng.normal(0, 80, 900)})
        r = build_analyst_result("Is x related to y?", df, target="y", explanatory=["x"])
        self.assertGreater(r.numbers["pearson_r"], 0.9)
        self.assertIn("associated", _txt(r))
        self.assertNotIn("causes", _txt(r))

    def test_correlation_null(self):
        rng = np.random.RandomState(5)
        df = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
        r = build_analyst_result("Is x related to y?", df, target="y", explanatory=["x"])
        self.assertEqual(r.finding, "none")

    def test_trend_up_and_partial_last_month_dropped(self):
        rng = np.random.RandomState(6)
        days = pd.date_range("2024-01-01", "2025-12-10", freq="D")
        df = pd.DataFrame({"d": days, "rev": 100 + np.arange(len(days)) * 0.1 + rng.normal(0, 5, len(days))})
        r = build_analyst_result("Has revenue grown over time?", df, target="rev", time_col="d")
        self.assertEqual(r.kind, "TREND")
        self.assertEqual(r.finding, "positive")
        self.assertTrue(any("excluded from the trend" in c for c in r.caveats))

    def test_monthly_table_is_not_mistaken_for_partial_periods(self):
        m = pd.date_range("2023-01-01", periods=36, freq="MS")
        df = pd.DataFrame({"m": m, "rev": np.linspace(100, 160, 36)})
        r = build_analyst_result("Has revenue grown over time?", df, target="rev", time_col="m")
        self.assertFalse(any("excluded" in c for c in r.caveats))
        self.assertEqual(r.numbers["periods"], 36)


class TestPeriodChange(unittest.TestCase):
    def _df(self, seed=1):
        rng = np.random.RandomState(seed)
        rows = []
        for d in pd.date_range("2025-01-01", periods=365, freq="D"):
            for reg in ["North", "South", "East", "West"]:
                base = 1000 + rng.normal(0, 40)
                if d.month == 6 and reg == "South":
                    base *= 0.55
                rows.append((d, reg, base))
        return pd.DataFrame(rows, columns=["date", "region", "revenue"])

    def test_locates_the_true_segment_not_the_biggest_one(self):
        r = build_analyst_result("Why did revenue drop in June?", self._df(), target="revenue", group="region", time_col="date")
        self.assertEqual(r.kind, "PERIOD_CHANGE")
        self.assertEqual(r.numbers["top_contributor"], "South")
        self.assertGreater(r.numbers["top_contribution_share"], 0.7)
        self.assertLess(r.numbers["delta"], 0)
        self.assertTrue(r.supersedes_loop)

    def test_month_length_does_not_fake_an_unusual_change(self):
        # Flat per-day series: February is lower in *totals* only because it is short.
        rng = np.random.RandomState(9)
        days = pd.date_range("2024-01-01", "2025-06-30", freq="D")
        df = pd.DataFrame({"date": days, "revenue": 100.0 + rng.normal(0, 1, len(days))})
        df["region"] = np.where(np.arange(len(df)) % 2 == 0, "N", "S")
        r = build_analyst_result("Why did revenue drop in February 2025?", df, target="revenue", group="region", time_col="date")
        self.assertLess(abs(r.numbers["change_z_vs_history"]), 2.5)
        self.assertIn("within normal", _txt(r))
        self.assertIn("Calendar effect", _txt(r))

    def test_real_per_day_drop_is_flagged_even_though_short_month_hides_it_in_totals(self):
        rng = np.random.RandomState(9)
        days = pd.date_range("2024-01-01", "2025-06-30", freq="D")
        rev = 100.0 + rng.normal(0, 1, len(days))
        rev = np.where((days.year == 2025) & (days.month == 2), rev * 0.94, rev)  # true 6% per-day drop
        df = pd.DataFrame({"date": days, "revenue": rev})
        df["region"] = np.where(np.arange(len(df)) % 2 == 0, "N", "S")
        r = build_analyst_result("Why did revenue drop in February 2025?", df, target="revenue", group="region", time_col="date")
        self.assertIn("unusual for this series", _txt(r))

    def test_missing_prior_period_is_stated_not_guessed(self):
        df = self._df()
        df = df[df["date"] >= "2025-06-01"]
        r = build_analyst_result("Why did revenue drop in June?", df, target="revenue", group="region", time_col="date")
        self.assertIn("Cannot compare", _txt(r))

    def test_period_parsing_uses_data_calendar(self):
        dates = pd.Series(pd.date_range("2023-01-01", "2024-12-31", freq="D"))
        p = parse_period("why did it fall in march", dates)
        self.assertEqual((p.start.year, p.start.month), (2024, 3))
        self.assertIn("most recent", p.resolved_year_note)
        self.assertEqual(parse_period("in march 2023", dates).start.year, 2023)
        self.assertIsNone(parse_period("what is the weather", dates))


class TestSafety(unittest.TestCase):
    def test_never_raises_and_returns_none_on_unusable_input(self):
        self.assertIsNone(build_analyst_result("q", pd.DataFrame(), target="x"))
        self.assertIsNone(build_analyst_result("q", pd.DataFrame({"a": [1, 2]}), target="missing"))
        self.assertIsNone(build_analyst_result("Which g is highest?", pd.DataFrame({"g": ["a", "a"], "v": [1, 2]}), target="v", group="g"))


class InMemoryProvider(BaseDatasetProvider):
    def __init__(self, d):
        self.d = d

    def acquire_context(self, project_id, dataset_ids=None):
        return InvestigationDataContext(
            project_id=project_id, datasets_map=self.d,
            dataset_fingerprints={n: hashlib.sha256(x.to_json().encode()).hexdigest() for n, x in self.d.items()},
            requested_dataset_ids=dataset_ids)


def _run(df, question):
    db = SessionLocal()
    iid = f"INV-S10-{uuid.uuid4().hex[:8]}"
    db.add(Investigation(id=iid, project_id=f"p-{uuid.uuid4().hex[:6]}", question=question, status="PLANNED"))
    db.commit()
    db.close()
    InvestigationController(session_factory=SessionLocal, dataset_provider=InMemoryProvider({"t": df})).execute_investigation(
        investigation_id=iid, worker_id="w")
    db = SessionLocal()
    inv = db.query(Investigation).filter(Investigation.id == iid).first()
    out = (str(inv.verdict_type), inv.direct_answer or "", inv.status)
    db.close()
    return out


class TestEndToEnd(unittest.TestCase):
    def test_plain_correlation_question_runs_instead_of_authority_conflict(self):
        rng = np.random.RandomState(1)
        n = 900
        s = rng.gamma(4, 50, n)
        df = pd.DataFrame({"week": range(n), "ad_spend": s, "revenue": 3 * s + rng.normal(0, 80, n), "channel": rng.choice(list("abc"), n)})
        verdict, answer, _ = _run(df, "Is ad spend related to revenue?")
        self.assertNotIn("authority conflict", answer.lower())
        self.assertIn("r=", answer)

    def test_ranking_is_answered_and_observed(self):
        rng = np.random.RandomState(2)
        cat = rng.choice(["Books", "Toys", "Games", "Tools"], 900, p=[0.1, 0.2, 0.5, 0.2])
        df = pd.DataFrame({"sale_id": range(900), "category": cat, "sales": rng.gamma(3, 30, 900)})
        truth = df.groupby("category").sales.sum().idxmax()
        verdict, answer, _ = _run(df, "Which product category had the highest total sales?")
        self.assertIn(f"{truth} has the highest", answer)
        self.assertEqual(verdict, "OBSERVED")

    def test_why_drop_names_the_true_segment(self):
        df = TestPeriodChange()._df(seed=7)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        verdict, answer, _ = _run(df, "Why did revenue drop in June?")
        self.assertIn("Biggest contributor by region: South", answer)
        self.assertNotIn("East accounts", answer)


class TestMultiGroupPowerRegression(unittest.TestCase):
    """Session 11 introduced an adequate_power heuristic that used dict-style
    ``.values()`` on a pandas Series count -- TypeError on every >2-group
    numeric or categorical-rate comparison, silently swallowed by
    build_analyst_result's outer try/except, so questions with more than two
    groups (the DEFECT-026 headline case: "does X differ by region?" with
    4 regions) silently fell back to the old, pre-fix loop output. Pins the
    fix and fails again if the same mistake is reintroduced."""

    def test_multi_group_numeric_comparison_does_not_silently_return_none(self):
        rng = np.random.RandomState(21)
        n = 1200
        reg = rng.choice(["North", "South", "East", "West"], n)
        val = rng.normal(100, 20, n) + np.where(reg == "West", 15, 0)
        df = pd.DataFrame({"region": reg, "order_value": val})
        r = build_analyst_result("Does average order value differ by region?", df, target="order_value", group="region")
        self.assertIsNotNone(r)
        self.assertEqual(r.kind, "GROUP_COMPARISON")
        self.assertEqual(r.numbers["highest"], "West")
        self.assertIn("adequate_power", r.numbers)

    def test_multi_group_numeric_comparison_end_to_end_controller(self):
        rng = np.random.RandomState(21)
        n = 1200
        reg = rng.choice(["North", "South", "East", "West"], n)
        val = rng.normal(100, 20, n) + np.where(reg == "West", 15, 0)
        df = pd.DataFrame({"region": reg, "order_value": val})
        verdict, answer, _ = _run(df, "Does average order value differ by region?")
        self.assertIn("West", answer)
        self.assertIn("order_value", answer)

    def test_multi_group_rate_comparison_does_not_silently_return_none(self):
        rng = np.random.RandomState(22)
        n = 3000
        seg = rng.choice(["Gold", "Silver", "Bronze"], n)
        conv = (rng.rand(n) < np.where(seg == "Gold", 0.2, 0.1)).astype(int)
        df = pd.DataFrame({"segment": seg, "converted": conv})
        r = build_analyst_result("Does conversion rate differ by segment?", df, target="converted", group="segment")
        self.assertIsNotNone(r)
        self.assertEqual(r.kind, "RATE_COMPARISON")
        self.assertEqual(r.numbers["highest"], "Gold")
        self.assertIn("adequate_power", r.numbers)


class TestConfoundGuardIsNarrow(unittest.TestCase):
    """Session 11's ZIP guarded the analyst-answer layer against overriding genuine
    churn/Simpson's-paradox findings (DEFECT-017/018/019) by checking whether any
    *candidate* hypothesis text contained "confound". The synthesizer routinely offers
    a "confounded by <dimension>" candidate alongside every categorical breakdown
    question, whether or not it is ever adopted -- so that guard silently disabled the
    period-change layer on ordinary "why did revenue drop" questions (the DEFECT-026
    headline case). The guard must key off the loop's own asserted answer, not off
    every rejected candidate hypothesis."""

    def test_why_question_with_a_rejected_confound_candidate_still_gets_the_analyst_answer(self):
        rng = np.random.RandomState(37)
        rows = []
        for d in pd.date_range("2025-01-01", periods=365, freq="D"):
            for reg in ["North", "South", "East", "West"]:
                base = 1000 + rng.normal(0, 40)
                if d.month == 6 and reg == "South":
                    base *= 0.55
                rows.append((d.strftime("%Y-%m-%d"), reg, base))
        df = pd.DataFrame(rows, columns=["date", "region", "revenue"])
        verdict, answer, _ = _run(df, "Why did revenue drop in June?")
        self.assertIn("Biggest contributor by region: South", answer)


if __name__ == "__main__":
    unittest.main()

