"""Acceptance tests for ROOT_CAUSE analytical rigor: Confounder Screening, Adjustment, and Attenuation.

Verifies:
1. True Confounder with Substantial Attenuation:
   Candidate factor correlates with outcome AND differs across groups;
   multiple regression adjustment attenuates the group difference by >= 30%;
   evidence classified as CANDIDATE_CONFOUNDER_SUBSTANTIAL_ATTENUATION with nuanced epistemic language.
2. Effect Reversal (Simpson's Paradox):
   Unadjusted group difference reverses sign post-adjustment;
   evidence classified as CONFOUNDER_EFFECT_REVERSAL.
3. Persistent Group Difference:
   Adjusting for candidate factor does not attenuate the group difference (attenuation < 10%);
   evidence classified as PERSISTENT_GROUP_DIFFERENCE.
4. Confounder Screening Disqualification:
   Variables that do not correlate with outcome or do not differ across groups are disqualified from
   being candidate confounders.
5. Observational Ceiling Enforcement:
   Observational boundaries are explicitly reported, prohibiting unhedged mechanistic causal assertions.
"""
import unittest
import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.analyst_answer import build_analyst_result


class TestRootCauseConfounderRigor(unittest.TestCase):
    def test_unseen_true_confounder_substantial_attenuation(self):
        """Unseen test case 1: Group difference is substantially explained by confounder Z."""
        rng = np.random.RandomState(42)
        n = 400
        # Group: region A vs region B
        region = np.array(["Region_A"] * (n // 2) + ["Region_B"] * (n // 2))
        # Confounder Z: seniority (higher in Region A)
        seniority = np.where(region == "Region_A", rng.normal(8.0, 1.5, n), rng.normal(2.0, 1.5, n))
        # Target: productivity is driven almost entirely by seniority, not region directly
        productivity = 20.0 + 3.0 * seniority + rng.normal(0, 2.0, n)

        df = pd.DataFrame({"region": region, "seniority": seniority, "productivity": productivity})
        res = build_analyst_result("Why does Region_A have higher productivity than Region_B?", df, target="productivity", group="region")

        self.assertIsNotNone(res)
        self.assertEqual(res.kind, "ROOT_CAUSE")
        self.assertEqual(res.numbers["top_candidate"], "seniority")
        self.assertGreater(abs(res.numbers["top_candidate_r"]), 0.7)
        self.assertGreaterEqual(res.numbers["attenuation_pct"], 30.0)
        self.assertIn("CANDIDATE_CONFOUNDER_SUBSTANTIAL_ATTENUATION", res.numbers["confounding_classification"])

        # Nuanced epistemic language: must state consistent with confounding, not dogmatic proof
        details_txt = " ".join(res.details)
        self.assertIn("candidate confounder", details_txt.lower())
        self.assertIn("consistent with confounding", details_txt.lower())
        self.assertNotIn("is confounded by", details_txt.lower())
        self.assertIn("claim ceiling", details_txt.lower())

    def test_unseen_simpsons_paradox_effect_reversal(self):
        """Unseen test case 2: Unadjusted effect reverses sign post-adjustment (Simpson's Paradox)."""
        rng = np.random.RandomState(101)
        n = 500
        # Product type: Premium vs Basic
        tier = np.array(["Premium"] * (n // 2) + ["Basic"] * (n // 2))
        # Defect rate appears higher in Premium unadjusted because Premium is used in heavy-duty environments (severity)
        severity = np.where(tier == "Premium", rng.normal(10.0, 1.0, n), rng.normal(2.0, 1.0, n))
        # True defect rate given same severity is LOWER for Premium (-5 * is_premium + 2 * severity)
        is_prem = (tier == "Premium").astype(float)
        defects = 10.0 - 4.0 * is_prem + 1.8 * severity + rng.normal(0, 0.5, n)

        df = pd.DataFrame({"tier": tier, "severity": severity, "defects": defects})
        res = build_analyst_result("Why does Premium have higher defects than Basic?", df, target="defects", group="tier")

        self.assertIsNotNone(res)
        self.assertEqual(res.kind, "ROOT_CAUSE")
        self.assertEqual(res.numbers["top_candidate"], "severity")
        self.assertEqual(res.numbers["confounding_classification"], "CONFOUNDER_EFFECT_REVERSAL")
        self.assertTrue(res.numbers["unadjusted_diff"] * res.numbers["adjusted_diff"] < 0)

        details_txt = " ".join(res.details)
        self.assertIn("reverses direction", details_txt.lower())
        self.assertIn("candidate confounder", details_txt.lower())

    def test_unseen_persistent_difference_no_attenuation(self):
        """Unseen test case 3: Candidate covariate is correlated with outcome but does not attenuate group difference."""
        rng = np.random.RandomState(77)
        n = 400
        division = np.array(["North"] * (n // 2) + ["South"] * (n // 2))
        is_north = (division == "North").astype(float)
        # Covariate Z: market_size differs between North and South, and has slight correlation with sales
        market_size = np.where(division == "North", rng.normal(50.0, 5.0, n), rng.normal(48.0, 5.0, n))
        # Sales has a strong direct division effect of +25, and only tiny +0.05 * market_size
        sales = 100.0 + 25.0 * is_north + 0.05 * market_size + rng.normal(0, 2.0, n)

        df = pd.DataFrame({"division": division, "market_size": market_size, "sales": sales})
        res = build_analyst_result("Why does North have higher sales than South?", df, target="sales", group="division")

        self.assertIsNotNone(res)
        self.assertEqual(res.kind, "ROOT_CAUSE")
        # Attenuation should be negligible (< 10%)
        self.assertLess(res.numbers["attenuation_pct"], 10.0)
        self.assertEqual(res.numbers["confounding_classification"], "PERSISTENT_GROUP_DIFFERENCE")

        details_txt = " ".join(res.details)
        self.assertIn("persistent difference", details_txt.lower())
        self.assertIn("persists independently", details_txt.lower())

    def test_confounder_screening_rejects_uncorrelated_variables(self):
        """Unseen test case 4: Variables uncorrelated with target cannot be named as candidate confounders."""
        rng = np.random.RandomState(88)
        n = 300
        group = np.array(["GrpA"] * 150 + ["GrpB"] * 150)
        score = np.where(group == "GrpA", rng.normal(80, 5, n), rng.normal(60, 5, n))
        noise = rng.normal(0, 10, n)  # completely independent noise

        df = pd.DataFrame({"group": group, "score": score, "noise_var": noise})
        res = build_analyst_result("Why does GrpA have higher score than GrpB?", df, target="score", group="group")

        self.assertIsNotNone(res)
        self.assertEqual(res.kind, "ROOT_CAUSE")
        # noise_var must NOT be qualified as a candidate confounder
        self.assertNotEqual(res.numbers.get("top_candidate"), "noise_var")
        details_txt = " ".join(res.details)
        self.assertIn("none of the scanned factors met the dual criteria", details_txt.lower())

    def test_categorical_confounder_discovery_and_dummy_adjustment(self):
        """Unseen test case 5: Categorical confounder is discovered, evaluated via ANOVA/TVD, and dummy-adjusted."""
        from packages.analytics_core.src.data.certification_data import load_dataset
        df = load_dataset("titanic")
        res = build_analyst_result("Why did First class have higher survival than Third class?", df, target="survived", group="class")

        self.assertIsNotNone(res)
        self.assertEqual(res.kind, "ROOT_CAUSE")
        # Surrogates pclass and alive must not be selected as top_candidate
        self.assertNotIn(res.numbers["top_candidate"], ["pclass", "alive", "class", "survived"])
        # Top candidate must be a valid covariate such as who or sex or fare
        self.assertIn(res.numbers["top_candidate"], ["who", "sex", "adult_male", "fare"])
        # Attenuation should be evaluated
        self.assertGreater(res.numbers["attenuation_pct"], 0.0)
        self.assertIn("candidate confounder", " ".join(res.details).lower())

    def test_surrogate_exclusion_prevents_recoded_duplicate_confounders(self):
        """Unseen test case 6: Exposure and target surrogates (e.g. pclass for class, alive for survived) are excluded."""
        from packages.analytics_core.src.data.certification_data import load_dataset
        df = load_dataset("titanic")
        res = build_analyst_result("Why did First class have higher survival than Third class?", df, target="survived", group="class")
        scanned = res.numbers.get("candidate_factors_scanned", [])
        self.assertNotIn("pclass", scanned)
        self.assertNotIn("alive", scanned)
        self.assertNotIn("class", scanned)
        self.assertNotIn("survived", scanned)


if __name__ == "__main__":
    unittest.main()

