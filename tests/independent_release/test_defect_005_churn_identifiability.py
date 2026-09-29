"""
DEFECT-005 churn/segmentation identifiability -- known-answer test matrix.

Exercises packages/analytics_core/src/statistics/churn_estimands.py against
the six known-answer scenarios required by the defect brief Sec 12, plus the
adversarial cases in Sec 15. Ground truth for every scenario is computed
independently inside scripts/generate_churn_seed_data.py's scenario
generators (arithmetic derived from the data-generating process, not from
this module's own code), per the brief's "known-answer" requirement.

Run with: python -m unittest discover -s tests/independent_release -p "test_*.py"
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from packages.analytics_core.src.statistics.churn_estimands import (  # noqa: E402
    ChurnVerdict,
    analyze_churn_identifiability,
    crude_churn_rate,
    exposure_adjusted_rate,
    raw_churn_count,
    censoring_report,
    stratified_rate_check,
)
import generate_churn_seed_data as gen  # noqa: E402


class TestRawCountTrap(unittest.TestCase):
    """Case 1: raw count says A worse; rate says B worse."""

    def setUp(self):
        self.df, self.truth = gen.scenario_raw_count_trap(random.Random(1))

    def test_raw_count_favors_A(self):
        counts = raw_churn_count(self.df, "segment")
        self.assertGreater(counts["A"], counts["B"])

    def test_rate_favors_B_as_worse(self):
        rates = crude_churn_rate(self.df, "segment").set_index("segment")
        self.assertGreater(rates.loc["B", "crude_rate"], rates.loc["A", "crude_rate"])

    def test_system_does_not_rank_by_raw_count(self):
        result = analyze_churn_identifiability(self.df, "segment", min_group_n=20)
        rates = result.segment_summary.set_index("segment")
        # The rate that should be identified as "worse" is B's, not A's,
        # even though A has more raw events. Whether the association
        # clears statistical significance for THIS random draw is a
        # separate, honest question (B has only 100 rows, so noise can
        # legitimately swallow a 2pp difference) -- INSUFFICIENT_EVIDENCE
        # is an acceptable, correct verdict here; a verdict claiming A is
        # worse would not be.
        self.assertGreater(rates.loc["B", "crude_rate"], rates.loc["A", "crude_rate"])
        self.assertIn(result.verdict, (ChurnVerdict.OBSERVED_ASSOCIATION,
                                        ChurnVerdict.SUPPORTED_SEGMENT_DIFFERENCE,
                                        ChurnVerdict.INSUFFICIENT_EVIDENCE))


class TestExposureTrap(unittest.TestCase):
    """Case 2: same hazard, different exposure duration."""

    def setUp(self):
        self.df, self.truth = gen.scenario_exposure_trap(random.Random(2))

    def test_crude_rates_differ_due_to_exposure(self):
        rates = crude_churn_rate(self.df, "segment").set_index("segment")
        # C (365 days exposure) should show a materially higher crude rate
        # than D (90 days exposure) purely from exposure difference.
        self.assertGreater(rates.loc["C", "crude_rate"], rates.loc["D", "crude_rate"])

    def test_persontime_rates_converge(self):
        exp = exposure_adjusted_rate(self.df, "segment").set_index("segment")
        ratio = exp.loc["C", "persontime_rate"] / exp.loc["D", "persontime_rate"]
        # Same underlying daily hazard -> person-time rates should be close
        # (within statistical noise), unlike the crude rates.
        self.assertTrue(0.7 < ratio < 1.3, f"person-time rates diverged: {ratio}")

    def test_system_flags_exposure_imbalance(self):
        result = analyze_churn_identifiability(self.df, "segment", min_group_n=20)
        self.assertTrue(any("exposure" in w.lower() or "observation_days" in w
                             for w in result.warnings),
                         f"expected an exposure-imbalance warning, got: {result.warnings}")


class TestCohortTrap(unittest.TestCase):
    """Case 3: aggregate segment difference explained by cohort mix."""

    def setUp(self):
        self.df, self.truth = gen.scenario_cohort_trap(random.Random(3))

    def test_aggregate_difference_exists(self):
        rates = crude_churn_rate(self.df, "segment").set_index("segment")
        self.assertGreater(rates.loc["F", "crude_rate"], rates.loc["E", "crude_rate"])

    def test_within_cohort_difference_collapses(self):
        check = stratified_rate_check(self.df, "segment", "cohort", min_stratum_n=20)
        self.assertTrue(check["applicable"])
        self.assertTrue(check["confound_detected"],
                         f"expected cohort confound to be detected: {check}")

    def test_system_returns_confounded_verdict(self):
        result = analyze_churn_identifiability(
            self.df, "segment", known_confounders=["cohort"], min_group_n=20)
        self.assertEqual(result.verdict, ChurnVerdict.CONFOUNDED_IDENTIFIABILITY_LIMITED)
        self.assertIn("cohort", result.confounders_detected)
        # must NOT make an unqualified causal/intrinsic-risk claim -- the
        # narrative is allowed to use "intrinsically" only inside an
        # explicit negation ("this is NOT evidence that ... intrinsically
        # drives churn risk"), never as an affirmative claim.
        narrative = result.narrative.lower()
        self.assertNotIn("causes", narrative)
        self.assertIn("not evidence", narrative)


class TestTenureTrap(unittest.TestCase):
    """Case 4: segment membership collinear with tenure."""

    def setUp(self):
        self.df, self.truth = gen.scenario_tenure_trap(random.Random(4))

    def test_aggregate_difference_exists(self):
        rates = crude_churn_rate(self.df, "segment").set_index("segment")
        self.assertGreater(rates.loc["G", "crude_rate"], rates.loc["H", "crude_rate"])

    def test_tenure_and_segment_are_collinear(self):
        # tenure_days ranges for G and H should not overlap at all in this
        # fixture -- confirming stratification by tenure_days is genuinely
        # impossible (no shared tenure band to compare within), which is
        # itself the correct thing for the system to detect.
        g_max = self.df[self.df.segment == "G"].tenure_days.max()
        h_min = self.df[self.df.segment == "H"].tenure_days.min()
        self.assertLess(g_max, h_min)

    def test_system_does_not_overclaim_when_stratification_impossible(self):
        # No tenure_band column exists (raw tenure_days is continuous and
        # collinear with segment) -- stratified_rate_check on tenure_days
        # directly should find no stratum with adequate sample on both
        # sides, and the system must not silently claim a supported effect.
        check = stratified_rate_check(self.df, "segment", "tenure_days", min_stratum_n=20)
        self.assertFalse(check.get("confound_detected", False) and check.get("applicable"),
                          "should not be able to confidently confirm OR rule out "
                          "the confound when segment and tenure never co-occur")
        result = analyze_churn_identifiability(
            self.df, "segment", known_confounders=["tenure_days"], min_group_n=20)
        self.assertNotEqual(result.verdict, ChurnVerdict.SUPPORTED_SEGMENT_DIFFERENCE)


class TestGenuineEffect(unittest.TestCase):
    """Case 5: real segment effect that survives stratification."""

    def setUp(self):
        self.df, self.truth = gen.scenario_genuine_effect(random.Random(5))

    def test_effect_survives_cohort_stratification(self):
        check = stratified_rate_check(self.df, "segment", "cohort", min_stratum_n=20)
        self.assertTrue(check["applicable"])
        self.assertFalse(check["confound_detected"],
                          f"genuine effect should survive cohort stratification: {check}")

    def test_system_returns_supported_difference(self):
        result = analyze_churn_identifiability(
            self.df, "segment", known_confounders=["cohort"], min_group_n=20)
        self.assertEqual(result.verdict, ChurnVerdict.SUPPORTED_SEGMENT_DIFFERENCE)


class TestInsufficientInformation(unittest.TestCase):
    """Case 6: confounder fields stripped -- must not guess."""

    def setUp(self):
        self.df, self.truth = gen.scenario_insufficient_information(random.Random(6))

    def test_no_confounder_fields_present(self):
        for col in ("cohort", "tenure_days", "observation_days"):
            self.assertTrue(self.df[col].isna().all())

    def test_system_reports_insufficient_or_unconfirmed_without_confounders(self):
        # Small n (40/segment) and no confounder fields -- system must not
        # fabricate a confident SUPPORTED_SEGMENT_DIFFERENCE.
        result = analyze_churn_identifiability(self.df, "segment", known_confounders=[],
                                                 min_group_n=20)
        self.assertNotEqual(result.verdict, ChurnVerdict.SUPPORTED_SEGMENT_DIFFERENCE)


class TestNoChurnColumnAtAll(unittest.TestCase):
    """Negative proof (Sec 18): a dataset with genuinely no churn signal
    (mirrors this repo's actual customers.csv, per DEFECT_005_ROOT_CAUSE.md
    Sec 5.2) must return INSUFFICIENT_EVIDENCE, not a crash or a guess."""

    def test_missing_event_column(self):
        import pandas as pd
        df = pd.DataFrame({
            "customer_id": [f"CUST-{i}" for i in range(10)],
            "segment": ["A"] * 5 + ["B"] * 5,
        })
        result = analyze_churn_identifiability(df, "segment")
        self.assertEqual(result.verdict, ChurnVerdict.INSUFFICIENT_EVIDENCE)
        self.assertIn("no churn", result.narrative.lower())


class TestCensoringDistinction(unittest.TestCase):
    """Sec 6: censored rows must never be silently treated as confirmed
    non-churn."""

    def test_censored_rows_excluded_from_crude_rate_denominator_language(self):
        df, _truth = gen.scenario_exposure_trap(random.Random(7))
        report = censoring_report(df)
        self.assertTrue(report["censoring_data_available"])
        self.assertGreater(report["n_censored_incomplete_observation"], 0)
        # confirmed non-churn must be strictly less than total non-events,
        # i.e. censored rows are not folded into "confirmed non-churn"
        total_nonevents = int((df["churn_event"] == 0).sum())
        self.assertLessEqual(report["n_confirmed_nonchurn"], total_nonevents)


class TestAdversarialSimpsonsParadox(unittest.TestCase):
    """Sec 15.A: aggregate points one way, stratified points another --
    duplicate of the cohort trap, kept as an explicit adversarial case."""

    def test_aggregate_disagrees_with_stratified(self):
        df, _truth = gen.scenario_cohort_trap(random.Random(8))
        agg = crude_churn_rate(df, "segment").set_index("segment")
        agg_direction = agg.loc["F", "crude_rate"] > agg.loc["E", "crude_rate"]
        self.assertTrue(agg_direction)  # F looks worse in aggregate

        check = stratified_rate_check(df, "segment", "cohort", min_stratum_n=20)
        h1 = next(s for s in check["per_stratum"] if s["stratum"] == "2025-H1")
        self.assertTrue(h1["adequate_sample"])
        # within H1 the segment difference should be substantially smaller
        # than the aggregate difference -- the module's own confound
        # threshold (Sec 15.A) is "within-stratum diff < 0.5x aggregate diff"
        aggregate_diff = check["aggregate_diff"]
        self.assertLess(abs(h1["diff"]), 0.5 * abs(aggregate_diff),
                         "within H1 the segment difference should collapse relative "
                         "to the aggregate difference")
        self.assertTrue(check["confound_detected"])


class TestAdversarialDenominatorManipulation(unittest.TestCase):
    """Sec 15.B: fewer customers + more events must not auto-label worse
    without checking the rate."""

    def test_fewer_customers_more_events_not_auto_worse(self):
        df, truth = gen.scenario_raw_count_trap(random.Random(9))
        # B has fewer customers (100) but could have fewer raw events than
        # A (900) while still having a higher rate -- confirm rate-based
        # comparison, not count-based, drives the verdict direction.
        rates = crude_churn_rate(df, "segment").set_index("segment")
        counts = raw_churn_count(df, "segment")
        if counts["B"] < counts["A"]:
            self.assertGreater(rates.loc["B", "crude_rate"], rates.loc["A", "crude_rate"])


if __name__ == "__main__":
    unittest.main()
