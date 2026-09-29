"""P1-12 Regression: Formal synthesis-time VERIFICATION experiment role.

Proves that ExperimentSynthesizer formally produces VERIFICATION candidate
experiments paired to primary candidates with replication_of populated,
unifying the scientific provenance lifecycle.
"""
import sys
import unittest
sys.path.insert(0, '.')

from packages.analytics_core.src.intelligence.experiment_synthesizer import ExperimentSynthesizer
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.schemas.src.semantic_role import ExperimentRole


class TestVerificationExperimentRole(unittest.TestCase):
    def test_synthesis_produces_verification_candidate_paired_to_primary(self):
        hyp = PredictiveHypothesis(
            id='hyp-1',
            hypothesis_code='HYP-01',
            claim='us-east segment drove sales drop',
            mechanism='regional concentration',
            predicted_observables_if_true=['Concentration in us-east > 45%'],
            predicted_observables_if_false=['Uniform drop across regions'],
            falsification_criteria='Concentration < 20%',
            required_assumptions=['Stable recording'],
            prior_probability=0.5,
            posterior_probability=0.5,
            target_metric='sales_amount',
            target_dimension='region',
        )
        semantic = SemanticResolution(
            primary_dataset_name='sales_data',
            target_metric_col='sales_amount',
            group_dimension_col='region',
            time_col='order_date',
            table_grain='order_id',
            available_numeric_cols=['sales_amount'],
            available_categorical_cols=['region'],
        )

        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(
            hypotheses=[hyp],
            semantic=semantic,
            predictions_by_hyp={'HYP-01': 'PRED-01'},
        )

        roles = [c.experiment_role for c in candidates]
        self.assertIn(ExperimentRole.PRIMARY.value, roles, 'PRIMARY candidate must exist')
        self.assertIn(ExperimentRole.VERIFICATION.value, roles, 'VERIFICATION candidate must be formally synthesized')

        verify_candidates = [c for c in candidates if c.experiment_role == ExperimentRole.VERIFICATION.value]
        self.assertTrue(len(verify_candidates) >= 1)
        vc = verify_candidates[0]

        # Verify pairing to primary
        primary = next(c for c in candidates if c.experiment_role == ExperimentRole.PRIMARY.value)
        self.assertEqual(vc.replication_of, primary.code)
        self.assertEqual(vc.provenance.get('verification_target'), primary.code)
        self.assertEqual(vc.target_hypothesis_code, primary.target_hypothesis_code)
        self.assertEqual(vc.aggregation_type, primary.aggregation_type)


if __name__ == '__main__':
    unittest.main()
