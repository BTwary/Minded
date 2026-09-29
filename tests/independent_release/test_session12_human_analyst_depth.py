"""Session 12: Principal Human Data Analyst Depth & Rigor Verification Suite.

Validates the full mathematical and analytical depth of an elite Principal Data Analyst:
1. Kitagawa mix-shift vs rate decomposition & Simpson's Paradox detection.
2. Multilevel waterfall bridge exact arithmetic reconciliation (V0 -> V1).
3. Cohort & account lifecycle bridge (New, Retained, Lost accounts).
4. Gini inequality coefficient, HHI, and Pareto concentration curve.
5. Common Language Effect Size (CLES) and full parametric/non-parametric distribution profiles.
6. Multivariable regression covariate control, elasticity at means, and confounder detection.
7. Predictive run-rate extrapolation with 95% prediction intervals, volatility, and autocorrelation.
8. Counterfactual opportunity sizing across ranking, comparison, and period change.
9. Categorized 3-tier actionable strategic playbooks (Triage 24-48h, Diagnostic 1-2w, Strategic 30-90d).
10. End-to-end integration and backward compatibility.

Run with: python -m pytest tests/independent_release/test_session12_human_analyst_depth.py -v
"""
import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.engines.analyst_answer import (
    AnalystResult,
    build_analyst_result,
    _gini_and_concentration,
    _distribution_profile,
    _kitagawa_decomposition,
    _cohort_lifecycle_bridge,
    _multivariate_ols_context,
    _build_strategic_playbook,
)


class TestKitagawaDecompositionAndSimpsonsParadox(unittest.TestCase):
    """Test Kitagawa decomposition and automated Simpson's Paradox detection."""

    def test_kitagawa_identity_exact_reconciliation(self):
        # Construct period 0 and period 1 data across two segments
        # Delta y_bar must exactly equal Rate Effect + Mix Shift Effect
        prev = pd.DataFrame({
            "segment": ["Enterprise"] * 60 + ["SMB"] * 40,
            "revenue": [100.0] * 60 + [20.0] * 40,
        })
        cur = pd.DataFrame({
            "segment": ["Enterprise"] * 30 + ["SMB"] * 70,
            "revenue": [110.0] * 30 + [25.0] * 70,
        })
        decomp = _kitagawa_decomposition(prev, cur, "segment", "revenue", agg="mean")
        self.assertIsNotNone(decomp)
        delta_avg = decomp["delta_average"]
        rate_eff = decomp["pure_rate_effect"]
        mix_eff = decomp["mix_shift_effect"]
        self.assertAlmostEqual(delta_avg, rate_eff + mix_eff, places=5)
        self.assertEqual(decomp["dimension"], "segment")

    def test_simpsons_paradox_detection(self):
        # Simpson's Paradox:
        # Segment A: Rate increases from 10 to 12 (+2)
        # Segment B: Rate increases from 2 to 3 (+1)
        # Period 0: 80% Segment A (mean = 0.8*10 + 0.2*2 = 8.4)
        # Period 1: 20% Segment A (mean = 0.2*12 + 0.8*3 = 2.4 + 2.4 = 4.8)
        # Overall average dropped from 8.4 to 4.8 (-3.6) despite every segment improving!
        dates_may = pd.date_range("2025-05-01", "2025-05-31", periods=100)
        dates_jun = pd.date_range("2025-06-01", "2025-06-30", periods=100)

        # May: 80 A at 10, 20 B at 2
        may_data = [("A", 10.0)] * 80 + [("B", 2.0)] * 20
        # June: 20 A at 12, 80 B at 3
        jun_data = [("A", 12.0)] * 20 + [("B", 3.0)] * 80

        df = pd.DataFrame({
            "date": list(dates_may) + list(dates_jun),
            "segment": [x[0] for x in may_data] + [x[0] for x in jun_data],
            "score": [x[1] for x in may_data] + [x[1] for x in jun_data],
        })

        res = build_analyst_result(
            "Why did score change in June 2025?",
            df,
            target="score",
            time_col="date",
            candidate_dimensions=["segment"],
            default_aggregation="mean",
        )
        self.assertIsNotNone(res)
        self.assertIsNotNone(res.mix_shift_decomposition)
        self.assertTrue(res.mix_shift_decomposition["simpsons_paradox"])
        full_text = res.to_text().lower()
        self.assertIn("simpson's paradox", full_text)
        self.assertIn("mix-shift", full_text)

    def test_flat_within_rates_with_mix_shift_does_not_trigger_simpsons_paradox(self):
        # Segment A: Rate is 10.0 in both periods
        # Segment B: Rate is 2.0 in both periods
        # Period 0: 80% A, 20% B (mean = 8.4)
        # Period 1: 20% A, 80% B (mean = 4.8)
        # delta_avg = -3.6, but within rates did NOT move in the opposite direction (they didn't move at all)
        prev = pd.DataFrame({
            "segment": ["A"] * 80 + ["B"] * 20,
            "revenue": [10.0] * 80 + [2.0] * 20,
        })
        cur = pd.DataFrame({
            "segment": ["A"] * 20 + ["B"] * 80,
            "revenue": [10.0] * 20 + [2.0] * 80,
        })
        decomp = _kitagawa_decomposition(prev, cur, "segment", "revenue", agg="mean")
        self.assertIsNotNone(decomp)
        self.assertFalse(decomp["simpsons_paradox"])
        self.assertEqual(decomp["primary_mechanism"], "mix_shift")
        self.assertAlmostEqual(decomp["pure_rate_effect"], 0.0, places=5)
        self.assertAlmostEqual(decomp["mix_shift_effect"], -4.8, places=5)

        # End-to-end through build_analyst_result
        dates_may = pd.date_range("2025-05-01", "2025-05-31", periods=100)
        dates_jun = pd.date_range("2025-06-01", "2025-06-30", periods=100)
        df = pd.DataFrame({
            "date": list(dates_may) + list(dates_jun),
            "segment": (["A"] * 80 + ["B"] * 20) + (["A"] * 20 + ["B"] * 80),
            "score": ([10.0] * 80 + [2.0] * 20) + ([10.0] * 20 + [2.0] * 80),
        })
        res = build_analyst_result(
            "Why did score change in June 2025?",
            df,
            target="score",
            time_col="date",
            candidate_dimensions=["segment"],
            default_aggregation="mean",
        )
        self.assertIsNotNone(res)
        self.assertIsNotNone(res.mix_shift_decomposition)
        self.assertFalse(res.mix_shift_decomposition["simpsons_paradox"])
        self.assertNotIn("simpson's paradox", res.to_text().lower())


class TestWaterfallBridgeReconciliation(unittest.TestCase):
    """Test multilevel waterfall bridge exact arithmetic reconciliation."""

    def test_waterfall_exact_reconciliation_period_change(self):
        dates_may = pd.date_range("2025-05-01", "2025-05-31", periods=120)
        dates_jun = pd.date_range("2025-06-01", "2025-06-30", periods=80)

        # Segments: North, South, East, West
        df = pd.DataFrame({
            "date": list(dates_may) + list(dates_jun),
            "sales": [100.0] * 120 + [80.0] * 80,
            "region": (["North"] * 50 + ["South"] * 40 + ["East"] * 20 + ["West"] * 10) +
                      (["North"] * 20 + ["South"] * 30 + ["East"] * 20 + ["West"] * 10),
        })

        res = build_analyst_result(
            "Why did sales drop in June 2025?",
            df,
            target="sales",
            time_col="date",
            group="region",
            default_aggregation="sum",
        )
        self.assertIsNotNone(res)
        self.assertIsNotNone(res.waterfall_bridge)
        bridge = res.waterfall_bridge
        v0 = bridge["initial_value"]
        v1 = bridge["final_value"]
        steps = bridge["steps"]

        self.assertGreater(len(steps), 1)
        self.assertEqual(steps[0]["step"], "Prior Base")
        self.assertAlmostEqual(steps[0]["subtotal"], v0, places=4)
        self.assertAlmostEqual(steps[-1]["subtotal"], v1, places=4)
        self.assertAlmostEqual(v1 - v0, bridge["net_delta"], places=4)

        full_text = res.to_text()
        self.assertIn("Reconciliation: Prior Base:", full_text)
        self.assertIn("[Final:", full_text)
        self.assertIn("Strategic Playbook:", full_text)


class TestCohortLifecycleBridge(unittest.TestCase):
    """Test account/customer cohort lifecycle decomposition (New, Retained, Churned)."""

    def test_cohort_reconciliation_identity(self):
        prev = pd.DataFrame({
            "account_id": ["acc_1", "acc_2", "acc_3", "acc_4"],
            "mrr": [1000.0, 2000.0, 1500.0, 500.0],  # Total = 5000
        })
        cur = pd.DataFrame({
            "account_id": ["acc_1", "acc_2", "acc_5", "acc_6"],
            "mrr": [1200.0, 1800.0, 800.0, 700.0],   # Total = 4500 (acc_3 & acc_4 churned, acc_5 & acc_6 new)
        })

        cl = _cohort_lifecycle_bridge(prev, cur, "account_id", "mrr", agg="sum")
        self.assertIsNotNone(cl)
        # Expected:
        # Total Delta = 4500 - 5000 = -500
        # Retained (acc_1, acc_2): cur=3000, prev=3000, delta=0
        # New (acc_5, acc_6): +1500
        # Churned (acc_3, acc_4): 2000
        # Net identity: Total Delta == Retained Delta + New - Lost
        # -500 == 0 + 1500 - 2000 = -500
        self.assertEqual(cl["total_delta"], -500.0)
        self.assertEqual(cl["retained_account_delta"], 0.0)
        self.assertEqual(cl["new_account_acquisition"], 1500.0)
        self.assertEqual(cl["churned_account_loss"], 2000.0)
        self.assertEqual(cl["retained_account_count"], 2)
        self.assertEqual(cl["new_account_count"], 2)
        self.assertEqual(cl["churned_account_count"], 2)
        self.assertAlmostEqual(
            cl["total_delta"],
            cl["retained_account_delta"] + cl["new_account_acquisition"] - cl["churned_account_loss"],
            places=5,
        )

    def test_cohort_lifecycle_in_period_change_integration(self):
        dates_may = pd.date_range("2025-05-01", "2025-05-31", periods=10)
        dates_jun = pd.date_range("2025-06-01", "2025-06-30", periods=10)

        df = pd.DataFrame({
            "date": list(dates_may) + list(dates_jun),
            "customer_id": [f"c_{i}" for i in range(1, 11)] + [f"c_{i}" for i in [1, 2, 3, 4, 11, 12, 13, 14, 15, 16]],
            "revenue": [100.0] * 20,
        })

        res = build_analyst_result(
            "Why did revenue change in June 2025?",
            df,
            target="revenue",
            time_col="date",
            default_aggregation="sum",
        )
        self.assertIsNotNone(res)
        self.assertIsNotNone(res.cohort_lifecycle)
        self.assertIn("cohort lifecycle bridge", res.to_text().lower())


class TestConcentrationAndTailFragility(unittest.TestCase):
    """Test Gini coefficient, HHI, and Pareto concentration calculations."""

    def test_perfectly_equal_distribution(self):
        s = pd.Series([100.0] * 50)
        conc = _gini_and_concentration(s)
        self.assertAlmostEqual(conc["gini_coefficient"], 0.0, places=2)
        self.assertIn("low concentration", conc["assessment"].lower())

    def test_severe_pareto_concentration(self):
        # 1 whale account with 90%, 99 accounts with remaining 10%
        vals = [9000.0] + [10.0] * 99
        s = pd.Series(vals)
        conc = _gini_and_concentration(s)
        self.assertGreaterEqual(conc["gini_coefficient"], 0.85)
        self.assertGreaterEqual(conc["top_1pct_share"], 0.85)
        self.assertIn("Severe concentration", conc["assessment"])

    def test_ranking_enriches_concentration(self):
        df = pd.DataFrame({
            "product": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
            "revenue": [50000.0, 1000.0, 800.0, 600.0, 400.0, 300.0, 200.0, 100.0],
        })
        res = build_analyst_result(
            "Which product had the highest revenue?",
            df,
            target="revenue",
            group="product",
            default_aggregation="sum",
        )
        self.assertIsNotNone(res)
        self.assertIsNotNone(res.concentration_diagnostics)
        self.assertGreaterEqual(res.concentration_diagnostics["gini_coefficient"], 0.60)
        self.assertIsNotNone(res.scenario_sensitivity)
        self.assertIn("opportunity_gap_to_median", res.scenario_sensitivity)


class TestCommonLanguageEffectSizeAndProfiles(unittest.TestCase):
    """Test Common Language Effect Size (CLES) and distribution percentiles."""

    def test_distribution_profile_percentiles(self):
        np.random.seed(42)
        data = np.random.normal(100.0, 15.0, size=200)
        prof = _distribution_profile(pd.Series(data))
        self.assertEqual(prof["n"], 200)
        self.assertAlmostEqual(prof["mean"], 100.0, delta=3.0)
        self.assertAlmostEqual(prof["median"], 100.0, delta=3.0)
        self.assertTrue(prof["p10"] < prof["p25"] < prof["p50"] < prof["p75"] < prof["p90"])

    def test_cles_in_numeric_comparison(self):
        np.random.seed(42)
        # Group A clearly outperforms Group B
        a = np.random.normal(120.0, 10.0, size=50)
        b = np.random.normal(80.0, 10.0, size=50)
        df = pd.DataFrame({
            "variant": ["A"] * 50 + ["B"] * 50,
            "latency": list(a) + list(b),
        })
        res = build_analyst_result(
            "Is latency different between A and B?",
            df,
            target="latency",
            group="variant",
        )
        self.assertIsNotNone(res)
        self.assertIn("common_language_effect_size", res.numbers)
        cles = res.numbers["common_language_effect_size"]
        self.assertGreater(cles, 0.90)
        self.assertIn("common language effect size", res.to_text().lower())
        self.assertIsNotNone(res.distribution_profile)
        self.assertIn("subject", res.distribution_profile)
        self.assertIn("reference", res.distribution_profile)


class TestMultivariateOLSAndElasticity(unittest.TestCase):
    """Test multivariable regression covariate control and elasticity at means."""

    def test_multivariate_context_and_elasticity(self):
        np.random.seed(42)
        n = 100
        spend = np.random.uniform(50, 500, size=n)
        tenure = np.random.uniform(1, 10, size=n)
        # Revenue = 50 + 2.5 * spend + 10 * tenure + noise
        revenue = 50.0 + 2.5 * spend + 10.0 * tenure + np.random.normal(0, 10, size=n)

        df = pd.DataFrame({"spend": spend, "tenure": tenure, "revenue": revenue})
        res = build_analyst_result(
            "Is spend associated with revenue?",
            df,
            target="revenue",
            explanatory=["spend"],
        )
        self.assertIsNotNone(res)
        self.assertIn("elasticity_at_mean", res.numbers)
        self.assertIsNotNone(res.multivariate_context)
        multiv = res.multivariate_context
        self.assertIn("tenure", multiv["controlled_covariates"])
        self.assertAlmostEqual(multiv["controlled_slope"], 2.5, delta=0.3)
        self.assertTrue(multiv["is_significant_after_control"])
        self.assertIn("elasticity", res.to_text().lower())

    def test_confounding_risk_detection(self):
        np.random.seed(42)
        n = 100
        z = np.random.normal(100, 15, size=n)
        # X and Y are both driven by Z (spurious correlation)
        x = 0.8 * z + np.random.normal(0, 5, size=n)
        y = 0.8 * z + np.random.normal(0, 5, size=n)

        df = pd.DataFrame({"marketing": x, "sales": y, "market_size": z})
        res = build_analyst_result(
            "Is marketing associated with sales?",
            df,
            target="sales",
            explanatory=["marketing"],
        )
        self.assertIsNotNone(res)
        # When market_size is controlled, marketing's slope should attenuate
        self.assertIsNotNone(res.multivariate_context)
        multiv = res.multivariate_context
        self.assertIn("market_size", multiv["controlled_covariates"])


class TestPredictiveRunRateExtrapolation(unittest.TestCase):
    """Test predictive run-rate extrapolations and time-series volatility."""

    def test_trend_prediction_interval_and_autocorrelation(self):
        # 12 monthly periods with upward trend
        dates = pd.date_range("2024-01-01", "2024-12-31", periods=120)
        # Add linear slope + noise
        base_y = np.linspace(100, 300, 120) + np.random.normal(0, 10, 120)
        df = pd.DataFrame({"date": dates, "mrr": base_y})

        res = build_analyst_result(
            "What is the trend of mrr over time?",
            df,
            target="mrr",
            time_col="date",
        )
        self.assertIsNotNone(res)
        self.assertIn("run_rate_projection", res.numbers)
        proj = res.numbers["run_rate_projection"]
        self.assertGreater(proj["projected_value"], 250.0)
        self.assertLess(proj["pi_low"], proj["projected_value"])
        self.assertGreater(proj["pi_high"], proj["projected_value"])
        self.assertIn("run-rate extrapolation", res.to_text().lower())
        self.assertIn("coefficient_of_variation", res.numbers)


class TestCategorizedStrategicPlaybook(unittest.TestCase):
    """Test categorized 3-tier actionable playbooks."""

    def test_playbook_structure_and_tiers(self):
        playbook = _build_strategic_playbook(
            "PERIOD_CHANGE",
            "negative",
            "Enterprise Tier",
            {"pvm_decomposition": {"volume_effect_share": 0.85}},
        )
        self.assertEqual(len(playbook), 3)
        tiers = [item["tier"] for item in playbook]
        self.assertIn("Immediate Operational Triage (24-48h)", tiers[0])
        self.assertIn("Diagnostic Deep-Dive (1-2 weeks)", tiers[1])
        self.assertIn("Strategic & Commercial Mitigation (30-90 days)", tiers[2])
        for item in playbook:
            self.assertIn(item["priority"], ("HIGH", "MEDIUM", "LOW"))
            self.assertTrue(len(item["action"]) > 10)
            self.assertTrue(len(item["focus_area"]) > 3)


class TestAnalystResultBackwardCompatibility(unittest.TestCase):
    """Ensure existing contracts, serialization, and text rendering remain 100% stable."""

    def test_to_dict_preserves_all_keys(self):
        r = AnalystResult(
            kind="TEST",
            headline="Sample headline",
            details=["Detail 1"],
            caveats=["Caveat 1"],
            numbers={"key": 123},
        )
        d = r.to_dict()
        self.assertEqual(d["kind"], "TEST")
        self.assertEqual(d["headline"], "Sample headline")
        self.assertEqual(d["executive_summary"], "Sample headline")
        self.assertIn("waterfall_bridge", d)
        self.assertIn("mix_shift_decomposition", d)
        self.assertIn("concentration_diagnostics", d)
        self.assertIn("scenario_sensitivity", d)
        self.assertIn("distribution_profile", d)
        self.assertIn("multivariate_context", d)
        self.assertIn("cohort_lifecycle", d)
        self.assertIn("strategic_playbook", d)

    def test_to_text_stability(self):
        r = AnalystResult(
            kind="TEST",
            headline="Headline text.",
            details=["Detail text."],
            caveats=["Check A."],
            next_steps=["Step 1."],
        )
        txt = r.to_text()
        self.assertIn("Headline text.", txt)
        self.assertIn("Detail text.", txt)
        self.assertIn("Checks: Check A.", txt)
        self.assertIn("Recommended next steps: Step 1.", txt)


class TestMultiGroupComparisonDefectFix(unittest.TestCase):
    """Pinned regression test for min(ns.values()) bug in multi-group comparisons (k > 2)."""

    def test_multi_group_numeric_comparison_k_greater_than_2(self):
        # 4 regions: North, South, East, West with distinct order values
        df = pd.DataFrame({
            "region": ["North"] * 25 + ["South"] * 30 + ["East"] * 20 + ["West"] * 25,
            "aov": [120.0] * 25 + [85.0] * 30 + [150.0] * 20 + [95.0] * 25,
        })
        res = build_analyst_result(
            "Does average order value differ by region?",
            df,
            target="aov",
            group="region",
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.kind, "GROUP_COMPARISON")
        self.assertIn("anova_f", res.numbers)
        self.assertIn("eta_squared", res.numbers)
        self.assertTrue(res.numbers["adequate_power"])
        self.assertIn("differs across region", res.headline.lower())

    def test_multi_group_rate_comparison_k_greater_than_2(self):
        # 4 regions: North, South, East, West with binary conversions
        df = pd.DataFrame({
            "region": ["North"] * 30 + ["South"] * 30 + ["East"] * 30 + ["West"] * 30,
            "converted": [1] * 15 + [0] * 15 + [1] * 6 + [0] * 24 + [1] * 20 + [0] * 10 + [1] * 10 + [0] * 20,
        })
        res = build_analyst_result(
            "Does conversion rate differ by region?",
            df,
            target="converted",
            group="region",
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.kind, "RATE_COMPARISON")
        self.assertIn("chi2_p", res.numbers)
        self.assertIn("cramers_v", res.numbers)
        self.assertTrue(res.numbers["adequate_power"])
        self.assertIn("differs across region", res.headline.lower())

    def test_multi_group_powered_null_no_detectable_effect(self):
        # 3 groups with identical mean & variance -> ANOVA should find no detectable difference
        np.random.seed(42)
        n_per_group = 40
        df = pd.DataFrame({
            "region": ["A"] * n_per_group + ["B"] * n_per_group + ["C"] * n_per_group,
            "metric": list(np.random.normal(100, 10, n_per_group)) +
                      list(np.random.normal(100, 10, n_per_group)) +
                      list(np.random.normal(100, 10, n_per_group)),
        })
        res = build_analyst_result(
            "Does metric differ across region?",
            df,
            target="metric",
            group="region",
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.finding, "none")
        self.assertTrue(res.numbers["adequate_power"])
        self.assertIn("no statistically detectable difference", res.headline.lower())


class TestConfoundingGuardAndWhyDidRevenueDropEndToEnd(unittest.TestCase):
    """Pin that candidate hypotheses containing 'confound' do NOT silence the analyst layer."""

    def test_why_revenue_drop_names_true_driver_without_confounding_suppression(self):
        import uuid
        import hashlib
        from apps.api.src.core.database import SessionLocal
        from apps.api.src.models.entities import Investigation
        from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
        from packages.analytics_core.src.runtime.controller import InvestigationController

        class LocalProvider(BaseDatasetProvider):
            def __init__(self, d):
                self.d = d
            def acquire_context(self, project_id, dataset_ids=None):
                return InvestigationDataContext(
                    project_id=project_id,
                    datasets_map=self.d,
                    dataset_fingerprints={n: hashlib.sha256(x.to_json().encode()).hexdigest() for n, x in self.d.items()},
                    requested_dataset_ids=dataset_ids,
                )

        dates_may = pd.date_range("2025-05-01", "2025-05-31", periods=100)
        dates_jun = pd.date_range("2025-06-01", "2025-06-30", periods=70)
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "date": list(dates_may) + list(dates_jun),
            "region": (["North"] * 50 + ["South"] * 50) + (["North"] * 50 + ["South"] * 20),
            "revenue": 100.0 + rng.normal(0, 5, 170),
        })
        iid = f"INV-CONF-{uuid.uuid4().hex[:8]}"
        db = SessionLocal()
        db.add(Investigation(id=iid, project_id=f"p-{uuid.uuid4().hex[:6]}", question="Why did revenue drop in June?", status="PLANNED"))
        db.commit()
        db.close()

        ctrl = InvestigationController(
            session_factory=SessionLocal,
            dataset_provider=LocalProvider({"sales": df}),
        )
        ctrl.execute_investigation(investigation_id=iid, worker_id="w-conf")

        db = SessionLocal()
        inv = db.query(Investigation).filter(Investigation.id == iid).first()
        ans = inv.direct_answer or ""
        main = inv.main_finding or ""
        db.close()

        self.assertIn("South", ans + main)
        self.assertIn("revenue", ans.lower())


class TestStatisticalCurrencyAndFormattedStrings(unittest.TestCase):
    """Test robustness of statistical preflight, selection, and inference on currency/formatted strings."""

    def test_preflight_and_method_selection_with_currency(self):
        from packages.analytics_core.src.statistics.preflight import _finite, numeric_pair
        from packages.analytics_core.src.statistics.method_selection import select_two_group
        from packages.analytics_core.src.statistics.inference import execute_two_group, execute_correlation
        from packages.analytics_core.src.statistics.analytical_math import descriptive_summary, welch_t_test

        arr = _finite(["$94.08", "€1,234.50", "¥500", "₹75.25", "25.5%", "N/A"])
        self.assertEqual(len(arr), 5)
        self.assertAlmostEqual(arr[0], 94.08, places=2)
        self.assertAlmostEqual(arr[1], 1234.50, places=2)
        self.assertAlmostEqual(arr[2], 500.0, places=2)
        self.assertAlmostEqual(arr[3], 75.25, places=2)
        self.assertAlmostEqual(arr[4], 25.5, places=2)

        # Preflight numeric_pair
        x = ["$10.00", "$20.00", "$30.00", "$40.00"]
        y = ["€15.00", "€25.00", "€35.00", "€45.00"]
        pf = numeric_pair(x, y, method="correlation")
        self.assertEqual(pf.status, "ALLOWED")

        # Two group selection & execution
        g1 = ["$100.00", "$110.50", "$105.00", "$115.00"]
        g2 = ["$80.00", "$85.00", "$82.50", "$88.00"]
        dec = select_two_group(g1, g2)
        self.assertIn(dec.method, ("Welch t-test", "Mann-Whitney U"))
        res = execute_two_group(g1, g2)
        self.assertIn("p_value", res)
        self.assertLess(res["p_value"], 0.05)

        # Correlation execution
        c_res = execute_correlation(x, y)
        self.assertAlmostEqual(c_res["coefficient"], 1.0, places=3)

        # Analytical math descriptive
        desc = descriptive_summary(["$10.00", "$20.00", "$30.00"])
        self.assertAlmostEqual(desc["mean"], 20.0, places=2)

        # Welch t-test
        wt = welch_t_test(g1, g2)
        self.assertTrue(wt["is_significant"])

    def test_analyst_answer_end_to_end_with_currency_column(self):
        dates_may = pd.date_range("2025-05-01", "2025-05-31", periods=100)
        dates_jun = pd.date_range("2025-06-01", "2025-06-30", periods=100)
        df = pd.DataFrame({
            "date": list(dates_may) + list(dates_jun),
            "region": (["North"] * 60 + ["South"] * 40) + (["North"] * 30 + ["South"] * 70),
            "revenue": [f"${x:.2f}" for x in ([100.0] * 60 + [50.0] * 40) + ([80.0] * 30 + [40.0] * 70)],
        })
        res = build_analyst_result(
            "Why did revenue drop in June 2025?",
            df,
            target="revenue",
            time_col="date",
            group="region",
            default_aggregation="sum",
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.kind, "PERIOD_CHANGE")
        self.assertIn("revenue", res.headline.lower())
        self.assertIn("Reconciliation: Prior Base:", res.to_text())
        self.assertIn("Strategic Playbook:", res.to_text())


if __name__ == "__main__":
    unittest.main()

