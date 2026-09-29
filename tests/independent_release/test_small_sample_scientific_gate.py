import unittest
import pandas as pd

from packages.analytics_core.src.engines.belief import BeliefEngine
from packages.analytics_core.src.intelligence.evidence_patterns import SegmentDifferenceDetector
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis


class SmallSampleScientificGateTests(unittest.TestCase):
    def test_targeted_subgroup_evidence_is_neutral_when_group_is_too_small(self):
        # Deliberately sparse data: 8 rows across 6 groups. A visually different
        # subgroup must not manufacture a positive Bayes factor.
        df = pd.DataFrame({
            "category": ["A", "B", "C", "D", "E", "F", "A", "B"],
            "stock_level": [80, 75, 70, 10, 65, 60, 82, 77],
        })
        h = PredictiveHypothesis(
            id="HYP-EMERGENT",
            hypothesis_code="HYP-EMERGENT",
            claim="Category A differs",
            mechanism="",
            predicted_observables_if_true=[],
            predicted_observables_if_false=[],
            falsification_criteria="",
            required_assumptions=[],
            prior_probability=0.5,
            posterior_probability=0.5,
            belief_state="PROPOSED",
            target_metric="stock_level",
            target_dimension="category",
            target_value="A",
        )
        factors, diagnostics = BeliefEngine.compute_model_based_bayes_factors(
            hypotheses=[h],
            primary_df=df,
            result_df=df,
            target_metric_col="stock_level",
            group_dimension_col="category",
            aggregation_type="MEAN",
            tested_hypothesis_codes=["HYP-EMERGENT"],
        )
        self.assertEqual(factors, [1.0])
        self.assertEqual(diagnostics[0]["method"], "NEUTRAL_INADEQUATE_GROUP_SAMPLE")

    def test_detector_also_refuses_sparse_group_partition(self):
        df = pd.DataFrame({
            "category": ["A", "B", "C", "D", "E", "F", "A", "B"],
            "stock_level": [80, 75, 70, 10, 65, 60, 82, 77],
        })
        self.assertIsNone(SegmentDifferenceDetector().detect(df, {}))

    def test_adequate_groups_still_receive_non_neutral_evidence(self):
        df = pd.DataFrame({
            "category": ["A"] * 8 + ["B"] * 8,
            "stock_level": [10, 11, 9, 10, 11, 10, 9, 10] + [50, 51, 49, 50, 52, 48, 50, 51],
        })
        h = PredictiveHypothesis(
            id="HYP-01",
            hypothesis_code="HYP-01",
            claim="Categories differ",
            mechanism="",
            predicted_observables_if_true=[],
            predicted_observables_if_false=[],
            falsification_criteria="",
            required_assumptions=[],
            prior_probability=0.5,
            posterior_probability=0.5,
            belief_state="PROPOSED",
        )
        factors, diagnostics = BeliefEngine.compute_model_based_bayes_factors(
            hypotheses=[h], primary_df=df, result_df=df,
            target_metric_col="stock_level", group_dimension_col="category",
            aggregation_type="MEAN", tested_hypothesis_codes=["HYP-01"],
        )
        self.assertGreater(factors[0], 1.0)
        self.assertNotEqual(diagnostics[0]["method"], "NEUTRAL_INADEQUATE_GROUP_SAMPLE")


if __name__ == "__main__":
    unittest.main()
