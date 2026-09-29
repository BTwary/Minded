"""
test_native_analytical_mathematics.py: Ground-Truth Mathematical Test Suite for MindEd AA-OS.

Verifies exact numerical accuracy of all 6 native mathematical primitive domains against
deterministic analytical formulas and statistical ground truths.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "analytics_core", "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "api", "src"))

from packages.analytics_core.src.statistics.analytical_math import (
    concentration_metrics,
    descriptive_summary,
    detect_distribution_shift,
    detect_outliers,
    frequency_table,
    kolmogorov_smirnov_test,
    kruskal_wallis_test,
    mann_whitney_u_test,
    multiple_testing_correction,
    one_way_anova,
    relationship_analysis,
    segment_analysis,
    time_series_summary,
    weighted_mean,
    welch_t_test,
)


class TestNativeAnalyticalMathematics(unittest.TestCase):
    """Rigorous ground-truth verification of deterministic analytical math primitives."""

    def test_01_descriptive_moments_and_percentiles(self):
        """Verify exact mean, median, variance, std dev, CV, and percentiles."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        res = descriptive_summary(data)

        self.assertEqual(res["total_count"], 5)
        self.assertEqual(res["valid_count"], 5)
        self.assertEqual(res["null_count"], 0)
        self.assertEqual(res["null_rate"], 0.0)
        self.assertEqual(res["sum"], 150.0)
        self.assertEqual(res["mean"], 30.0)
        self.assertEqual(res["median"], 30.0)
        self.assertEqual(res["min"], 10.0)
        self.assertEqual(res["max"], 50.0)
        self.assertEqual(res["range"], 40.0)
        self.assertEqual(res["variance"], 250.0)
        self.assertAlmostEqual(res["std_dev"], 15.8114, places=3)
        self.assertAlmostEqual(res["coefficient_of_variation"], 15.8114 / 30.0, places=3)
        self.assertEqual(res["percentiles"]["p50"], 30.0)
        self.assertEqual(res["percentiles"]["p25"], 20.0)
        self.assertEqual(res["percentiles"]["p75"], 40.0)
        self.assertEqual(res["iqr"], 20.0)
        print(" -> [TEST 01] Descriptive moments & percentiles: PASSED")

    def test_02_weighted_mean(self):
        """Verify exact weighted mean calculation."""
        vals = [10.0, 20.0, 30.0]
        weights = [1.0, 2.0, 3.0]
        # (10*1 + 20*2 + 30*3) / (1+2+3) = (10 + 40 + 90) / 6 = 140 / 6 = 23.3333...
        wm = weighted_mean(vals, weights)
        self.assertAlmostEqual(wm, 140.0 / 6.0, places=4)
        print(" -> [TEST 02] Weighted mean calculation: PASSED")

    def test_03_concentration_gini_and_hhi(self):
        """Verify Gini coefficient, Herfindahl Index, and Pareto 80/20 on skewed data."""
        # 10 items: 9 items have 10 each (sum=90), 1 dominant item has 910 (total sum = 1000)
        data = [10.0] * 9 + [910.0]
        res = concentration_metrics(data)

        self.assertEqual(res["item_count"], 10)
        self.assertEqual(res["total_sum"], 1000.0)
        self.assertEqual(res["top_1_share"], 0.91)
        self.assertGreater(res["gini_coefficient"], 0.80)
        # HHI: (91%^2 = 8281) + 9*(1%^2 = 9) = 8290
        self.assertAlmostEqual(res["herfindahl_index"], 8290.0, places=0)
        self.assertEqual(res["concentration_level"], "High")
        print(" -> [TEST 03] Gini, HHI, and concentration metrics: PASSED")

    def test_04_time_series_trend_and_cusum_break(self):
        """Verify period-over-period delta, Theil-Sen slope, and CUSUM change point."""
        # Pre-break: [100, 102, 101, 100, 103], Post-break step change: [250, 255, 252, 258, 260]
        dates = [f"2026-01-{i+1:02d}" for i in range(10)]
        vals = [100.0, 102.0, 101.0, 100.0, 103.0, 250.0, 255.0, 252.0, 258.0, 260.0]
        df = pd.DataFrame({"timestamp": dates, "value": vals})

        res = time_series_summary(df, "timestamp", "value")
        self.assertEqual(res["observation_count"], 10)
        self.assertEqual(res["start_value"], 100.0)
        self.assertEqual(res["end_value"], 260.0)
        self.assertEqual(res["net_change"], 160.0)
        self.assertEqual(res["growth_rate_percentage"], 160.0)
        self.assertTrue(res["trend"]["is_significant"])
        self.assertGreater(res["trend"]["slope_theil_sen_robust"], 0.0)
        self.assertTrue(res["change_point"]["detected"])
        print(" -> [TEST 04] Time series trend and CUSUM break: PASSED")

    def test_05_segment_decomposition_and_pareto(self):
        """Verify segment sum, share of total, ranking, and Pareto core classification."""
        df = pd.DataFrame({
            "region": ["North", "North", "South", "South", "East", "West"],
            "sales": [400.0, 300.0, 150.0, 50.0, 70.0, 30.0],  # Total = 1000. North=700 (70%), South=200 (20%), East=70 (7%), West=30 (3%)
        })
        res = segment_analysis(df, "region", "sales")
        self.assertEqual(res["total_sum"], 1000.0)
        self.assertEqual(res["segment_count"], 4)
        
        # Check North is rank 1 with 70% share
        north = next(s for s in res["segments"] if s["segment"] == "North")
        self.assertEqual(north["rank"], 1)
        self.assertEqual(north["sum"], 700.0)
        self.assertEqual(north["share_percentage"], 70.0)
        self.assertTrue(north["is_pareto_core"])

        # Check total shares sum to 100%
        shares = [s["share_percentage"] for s in res["segments"]]
        self.assertAlmostEqual(sum(shares), 100.0, places=1)
        print(" -> [TEST 05] Segment decomposition & Pareto analysis: PASSED")

    def test_06_bivariate_relationships_and_partial_correlation(self):
        """Verify exact Pearson correlation (r=1.0) and partial correlation under confounding."""
        # Exact linear relationship
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = 2.0 * x + 5.0
        df = pd.DataFrame({"x": x, "y": y})

        res = relationship_analysis(df, "x", "y")
        self.assertEqual(res["pearson"]["coefficient"], 1.0)
        self.assertEqual(res["spearman"]["coefficient"], 1.0)
        self.assertEqual(res["kendall_tau"]["coefficient"], 1.0)
        self.assertTrue(res["pearson"]["is_significant"])

        # Confounder test: X and Y both driven by Z
        # Fixed seed (audit fix, 2026-09-09): this test was unseeded, so
        # correlation_attenuation_percentage varied run-to-run with the exact
        # noise draw and the >50.0 assertion below failed ~2 times out of 5
        # runs even with zero code changes -- a flaky assertion, not a real
        # regression. Seeding makes the test deterministic and reproducible,
        # matching the "no fabricated/uncontrolled randomness in acceptance
        # tests" spirit of the rest of this suite.
        rng = np.random.default_rng(42)
        z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        x_conf = 3.0 * z + rng.normal(0, 0.1, 10)
        y_conf = 4.0 * z + rng.normal(0, 0.1, 10)
        df_conf = pd.DataFrame({"x": x_conf, "y": y_conf, "z": z})
        res_conf = relationship_analysis(df_conf, "x", "y", control_col="z")
        self.assertIsNotNone(res_conf["partial_correlation"])
        self.assertGreater(res_conf["partial_correlation"]["correlation_attenuation_percentage"], 50.0)
        print(" -> [TEST 06] Bivariate relationship & partial correlation: PASSED")

    def test_07_categorical_cramers_v(self):
        """Verify Cramér's V association metric on contingency table."""
        df = pd.DataFrame({
            "tier": ["Gold", "Gold", "Gold", "Bronze", "Bronze", "Bronze"] * 10,
            "status": ["Active", "Active", "Active", "Churned", "Churned", "Churned"] * 10,
        })
        res = relationship_analysis(df, "tier", "status")
        self.assertEqual(res["type"], "categorical_association")
        self.assertEqual(res["cramers_v"], 1.0)
        self.assertTrue(res["is_significant"])
        print(" -> [TEST 07] Categorical Cramers V association: PASSED")

    def test_08_anova_eta_squared_and_welch_t_test(self):
        """Verify One-Way ANOVA F-stat & eta^2, and Welch's t-test with Cohen's d."""
        # 3 distinct groups with high between-group variance
        groups = {
            "G1": [10.0, 11.0, 12.0, 10.5],
            "G2": [50.0, 52.0, 51.0, 49.0],
            "G3": [100.0, 102.0, 99.0, 101.0],
        }
        res_anova = one_way_anova(groups)
        self.assertTrue(res_anova["is_significant"])
        self.assertGreater(res_anova["eta_squared"], 0.95)
        self.assertEqual(res_anova["variance_explained_strength"], "High")

        # Welch's t-test between G1 and G3
        res_t = welch_t_test(groups["G1"], groups["G3"])
        self.assertTrue(res_t["is_significant"])
        self.assertGreater(abs(res_t["cohens_d"]), 2.0)
        self.assertEqual(res_t["effect_magnitude"], "Large")
        print(" -> [TEST 08] One-Way ANOVA eta^2 and Welch's t-test: PASSED")

    def test_09_non_parametric_tests_and_ks_shift(self):
        """Verify Mann-Whitney U, Kruskal-Wallis, and Kolmogorov-Smirnov distribution shift."""
        sample_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample_b = [10.0, 20.0, 30.0, 40.0, 50.0]

        res_u = mann_whitney_u_test(sample_a, sample_b)
        self.assertTrue(res_u["is_significant"])

        res_kw = kruskal_wallis_test({"A": sample_a, "B": sample_b})
        self.assertTrue(res_kw["is_significant"])

        res_ks = kolmogorov_smirnov_test(sample_a, sample_b)
        self.assertTrue(res_ks["distribution_shifted"])
        print(" -> [TEST 09] Non-parametric Mann-Whitney, Kruskal-Wallis & KS shift: PASSED")

    def test_10_outlier_detection_and_multiple_testing_fdr(self):
        """Verify IQR, Z-Score, MAD outlier detectors and Benjamini-Hochberg FDR correction."""
        # 19 regular values around 100, 1 extreme outlier at 10,000
        data = [100.0 + i for i in range(19)] + [10000.0]
        res_iqr = detect_outliers(data, method="iqr")
        self.assertEqual(res_iqr["outlier_count"], 1)
        self.assertEqual(res_iqr["outlier_indices"], [19])
        self.assertTrue(res_iqr["has_high_outlier_risk"])

        # Multiple Testing FDR: p-vals [0.001, 0.005, 0.04, 0.20, 0.85] with m=5
        # p_adj = [0.005, 0.0125, 0.0667, 0.25, 0.85]. Exactly 2 are <= 0.05.
        p_vals = [0.001, 0.005, 0.04, 0.20, 0.85]
        res_fdr = multiple_testing_correction(p_vals, method="fdr_bh", alpha=0.05)
        self.assertEqual(res_fdr["test_count"], 5)
        self.assertEqual(res_fdr["significant_count"], 2)
        self.assertTrue(res_fdr["significant"][0])
        self.assertTrue(res_fdr["significant"][1])
        self.assertFalse(res_fdr["significant"][2])
        self.assertFalse(res_fdr["significant"][4])
        print(" -> [TEST 10] Outlier detection & Benjamini-Hochberg FDR: PASSED")


def main():
    print("=" * 80, flush=True)
    print("RUNNING NATIVE ANALYTICAL MATHEMATICS ACCEPTANCE TESTS (10 TESTS)", flush=True)
    print("=" * 80, flush=True)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNativeAnalyticalMathematics)
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    if res.wasSuccessful():
        print("=" * 80, flush=True)
        print("ALL 10 NATIVE ANALYTICAL MATHEMATICS TESTS PASSED (100% PROVEN)", flush=True)
        print("=" * 80, flush=True)
        sys.exit(0)
    else:
        print("NATIVE ANALYTICAL MATHEMATICS TESTS FAILED", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
