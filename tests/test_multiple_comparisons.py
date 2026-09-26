import os, sys, unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from packages.analytics_core.src.statistics.multiple_comparisons import pairwise_posthoc
from packages.analytics_core.src.statistics.inference import execute_multi_group
from packages.analytics_core.src.statistics.engine import StatisticalEngine


class TestMultipleComparisons(unittest.TestCase):
    def test_anova_significant_runs_tukey_family(self):
        groups = {"A":[1,2,3,2,1], "B":[5,6,5,7,6], "C":[10,11,9,10,12]}
        result = execute_multi_group(groups)
        self.assertEqual(result["method"], "One-way ANOVA")
        self.assertEqual(result["posthoc"]["status"], "completed")
        self.assertEqual(result["posthoc"]["family_size"], 3)
        self.assertTrue(all("adjusted_p_value" in x for x in result["posthoc"]["comparisons"]))

    def test_non_significant_omnibus_gates_pairwise_search(self):
        groups = {"A":[1,2,3,4,5], "B":[1,2,3,4,5], "C":[1,2,3,4,5]}
        result = execute_multi_group(groups)
        self.assertEqual(result["posthoc"]["status"], "not_run")
        self.assertEqual(result["posthoc"]["comparisons"], [])

    def test_welch_uses_pairwise_welch_plus_holm(self):
        groups = {"A":[1,2,3,4,5], "B":[3,7,12,20,35], "C":[30,31,29,32,28]}
        result = pairwise_posthoc(groups, omnibus_method="Welch ANOVA", omnibus_p_value=0.001)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["correction"], "holm")
        self.assertTrue(all(x["method"] == "Welch pairwise t-test" for x in result["comparisons"]))

    def test_kruskal_uses_dunn_and_adjustment(self):
        groups = {"A":[1,1,2,2,1], "B":[5,6,7,5,6], "C":[10,11,9,12,10]}
        result = pairwise_posthoc(groups, omnibus_method="Kruskal-Wallis", omnibus_p_value=0.001)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(all(x["method"] == "Dunn pairwise rank test" for x in result["comparisons"]))
        self.assertTrue(all(x["effect_size_name"] == "rank-biserial correlation" for x in result["comparisons"]))

    def test_holm_is_monotone_and_never_below_raw(self):
        groups = {"A":[1,2,3], "B":[4,5,6], "C":[8,9,10], "D":[12,13,14]}
        result = pairwise_posthoc(groups, omnibus_method="One-way ANOVA", omnibus_p_value=0.001, correction="holm")
        for x in result["comparisons"]:
            self.assertGreaterEqual(x["adjusted_p_value"], x["raw_p_value"])

    def test_bonferroni_and_fdr_supported(self):
        groups = {"A":[1,2,3], "B":[4,5,6], "C":[8,9,10]}
        for corr in ("bonferroni", "fdr_bh"):
            result = pairwise_posthoc(groups, omnibus_method="One-way ANOVA", omnibus_p_value=0.001, correction=corr)
            self.assertEqual(result["correction"], corr)
            self.assertEqual(len(result["comparisons"]), 3)

    def test_effect_and_ci_are_traceable_for_tukey_path(self):
        groups = {"A":[1,2,3,2,1], "B":[5,6,5,7,6], "C":[10,11,9,10,12]}
        result = pairwise_posthoc(groups, omnibus_method="One-way ANOVA", omnibus_p_value=0.001, correction="tukey-hsd")
        for x in result["comparisons"]:
            self.assertEqual(x["correction"], "tukey-hsd")
            self.assertIn("effect_size", x)
            self.assertIn("effect_size_name", x)
            self.assertIsNotNone(x["confidence_interval"])

    def test_engine_exposes_posthoc(self):
        e = StatisticalEngine()
        result = e.posthoc_multi_group({"A":[1,2,3], "B":[6,7,8], "C":[11,12,13]}, omnibus_method="One-way ANOVA", omnibus_p_value=0.001)
        self.assertEqual(result["status"], "completed")

    def test_correlation_family_uses_adjusted_significance(self):
        e = StatisticalEngine()
        rng = np.random.default_rng(7)
        x = rng.normal(size=120)
        df = __import__("pandas").DataFrame({"x": x, "y": x + rng.normal(scale=0.2, size=120), "z": rng.normal(size=120)})
        result = e.correlation_analysis(df, ["x", "y", "z"], correction="holm")
        self.assertEqual(result["comparison_count"], 3)
        self.assertEqual(result["multiple_testing_correction"], "holm")
        self.assertIn("adjusted_p_value_matrix", result)
        for item in result["strong_correlations"]:
            self.assertLess(item["adjusted_p_value"], 0.05)

    def test_omnibus_gate_can_be_explicitly_disabled(self):
        groups = {"A":[1,2,3], "B":[1,2,4], "C":[1,2,5]}
        result = pairwise_posthoc(groups, omnibus_method="One-way ANOVA", omnibus_p_value=0.9, require_significant_omnibus=False)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["familywise_search_policy"], "ungated")


if __name__ == '__main__':
    unittest.main()
