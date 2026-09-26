"""Verification of Hugging Face Space App and Scenarios."""
import unittest
from pathlib import Path

from hf_space.app import (
    load_scenario_b2b,
    load_scenario_financials,
    load_scenario_cohorts,
    _load_files,
    run_minded_investigation,
)


class TestHFSpaceApp(unittest.TestCase):

    def test_scenario_files_exist(self):
        b2b_files, q1 = load_scenario_b2b()
        self.assertEqual(len(b2b_files), 3)
        for f in b2b_files:
            self.assertTrue(Path(f).is_file(), f"File {f} does not exist")
        self.assertIn("premium", q1.lower())

        fin_files, q2 = load_scenario_financials()
        self.assertEqual(len(fin_files), 1)
        self.assertTrue(Path(fin_files[0]).is_file())
        self.assertIn("division", q2.lower())

        coh_files, q3 = load_scenario_cohorts()
        self.assertEqual(len(coh_files), 1)
        self.assertTrue(Path(coh_files[0]).is_file())
        self.assertIn("retention", q3.lower())

    def test_load_files_robust(self):
        fin_files, _ = load_scenario_financials()
        datasets = _load_files(fin_files)
        self.assertEqual(len(datasets), 1)
        df = list(datasets.values())[0]
        # Verify gross_revenue was coerced to numeric float by RobustFileLoader
        self.assertTrue(df["gross_revenue"].dtype.kind in ("f", "i"), f"Expected numeric, got {df['gross_revenue'].dtype}")


    def test_run_minded_investigation_b2b_scenario(self):
        b2b_files, question = load_scenario_b2b()
        res = run_minded_investigation(b2b_files, question)
        self.assertEqual(len(res), 8)
        report, plot, recs, hypotheses, evidence, suggested_q, claim_gate, manifest = res
        self.assertIn("Minded Autonomous Analytical Verdict", report)
        self.assertIn("Enterprise", report)
        self.assertIsNotNone(plot)
        self.assertFalse(hypotheses.empty)
        self.assertFalse(evidence.empty)
        self.assertIn("analysis_id", manifest)


if __name__ == "__main__":
    unittest.main()
