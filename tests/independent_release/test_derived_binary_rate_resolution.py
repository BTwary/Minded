"""Regression tests for conservative binary-indicator-to-rate resolution."""
import os, sys, unittest
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from packages.analytics_core.src.engines.variable_resolution_gate import infer_binary_rate_indicator

class TestDerivedBinaryRateResolution(unittest.TestCase):
    def test_unique_defect_indicator_resolves(self):
        df = pd.DataFrame({"is_defective": [0, 1, 0, 0, 1], "temperature": [20, 25, 21, 22, 27]})
        self.assertEqual(infer_binary_rate_indicator("What is the defect rate?", df), "is_defective")

    def test_ambiguous_binary_indicators_fail_closed(self):
        df = pd.DataFrame({"is_defective": [0, 1, 0], "defect_flag": [0, 1, 0]})
        self.assertIsNone(infer_binary_rate_indicator("What is the defect rate?", df))

    def test_nonbinary_measure_does_not_resolve_as_rate_flag(self):
        df = pd.DataFrame({"defect_count": [0, 1, 3, 2], "temperature": [20, 21, 22, 23]})
        self.assertIsNone(infer_binary_rate_indicator("What is the defect rate?", df))

    def test_unrelated_binary_column_is_not_substituted(self):
        df = pd.DataFrame({"is_late": [0, 1, 0], "temperature": [20, 21, 22]})
        self.assertIsNone(infer_binary_rate_indicator("What is the defect rate?", df))

if __name__ == "__main__":
    unittest.main()
