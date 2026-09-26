import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from packages.analytics_core.src.statistics.probability_calibration import (
    fit_isotonic_calibration,
)
from packages.analytics_core.src.intelligence.epistemic_calibration import EpistemicCalibrationEngine
from packages.analytics_core.src.intelligence.evidence_patterns import SegmentDifferenceDetector
from packages.analytics_core.src.governance.claim_gate import admit_positive_claim
from packages.analytics_core.src.intelligence.nl_semantic_interpreter import interpret_with_schema
from packages.analytics_core.src.providers.manager import InfrastructureManager
from packages.analytics_core.src.providers.ai import BaseAIProvider


class P0ClosureTests(unittest.TestCase):
    def test_empirical_calibration_is_monotone_and_explicit(self):
        p = np.linspace(0.05, 0.95, 40)
        y = (p + np.array([0.18 if i % 3 == 0 else -0.05 for i in range(40)]) > 0.55).astype(int)
        profile = fit_isotonic_calibration(
            y, p, scope="p0_test", population_scope="p0_fixture",
            out_of_sample_verified=True,
            calibration_source_fingerprint="p0-fixture-source",
            fitting_protocol="labeled_holdout_out_of_sample_v1",
        )
        self.assertTrue(profile.is_usable)
        calibrated = [profile.calibrate(x) for x in [0.1, 0.3, 0.5, 0.7, 0.9]]
        self.assertEqual(calibrated, sorted(calibrated))
        v = EpistemicCalibrationEngine.calibrate_epistemic_state(
            0.8, sample_size=100, calibration_profile=profile,
            runtime_model_id="AAOS_BELIEF_V1", runtime_scope="p0_test",
            runtime_population_scope="p0_fixture",
        )
        self.assertEqual(v.calibration_status, "CALIBRATED")
        self.assertEqual(v.probability_semantics, "EMPIRICALLY_CALIBRATED")

    def test_no_profile_is_explicitly_uncalibrated(self):
        v = EpistemicCalibrationEngine.calibrate_epistemic_state(0.8, sample_size=100, calibration_profile=None)
        self.assertEqual(v.calibration_status, "UNAVAILABLE")
        self.assertEqual(v.probability_semantics, "MODEL_BASED_BELIEF")

    def test_segment_detector_does_not_promote_one_row_per_group_cv(self):
        df = pd.DataFrame({"segment": ["A", "B", "C", "D"], "value": [100.0, 1.0, 1.0, 1.0]})
        self.assertIsNone(SegmentDifferenceDetector().detect(df, {}))

    def test_segment_detector_requires_inferential_evidence(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "segment": np.repeat(["A", "B", "C"], 20),
            "value": np.concatenate([rng.normal(10, 1, 20), rng.normal(10, 1, 20), rng.normal(10, 1, 20)]),
        })
        self.assertIsNone(SegmentDifferenceDetector().detect(df, {}))
        strong = pd.DataFrame({
            "segment": np.repeat(["A", "B", "C"], 20),
            "value": np.concatenate([rng.normal(1, .1, 20), rng.normal(5, .1, 20), rng.normal(10, .1, 20)]),
        })
        self.assertIsNotNone(SegmentDifferenceDetector().detect(strong, {}))

    def test_nl_interpreter_never_accepts_unknown_column(self):
        # Deterministic fallback path: any candidate must map to physical schema.
        out = interpret_with_schema("Why did imaginary_profit fall?", ["revenue", "cost", "region"])
        self.assertNotIn("imaginary_profit", [out.target, out.group, out.time, out.secondary_target])
        self.assertTrue(out.source in {"deterministic", "byom_validated"})


    def test_byom_semantics_are_schema_validated_before_compilation(self):
        class FakeProvider(BaseAIProvider):
            @property
            def provider_type(self): return "fake"
            @property
            def is_local(self): return True
            @property
            def model_name(self): return "fake"
            def generate_response(self, prompt, system_prompt=None):
                return '{"task":"COMPARISON","target":"revenue","group":"region"}'
            def test_connection(self): return True, "ok"
        previous = InfrastructureManager._ai_provider
        try:
            InfrastructureManager.set_ai_provider(FakeProvider())
            out = interpret_with_schema("Which territory has more revenue?", ["revenue", "region", "cost"])
            self.assertEqual(out.target, "revenue")
            self.assertEqual(out.group, "region")
            self.assertEqual(out.source, "byom_validated")
        finally:
            InfrastructureManager._ai_provider = previous

    def test_claim_gate_preserves_positive_claim_but_does_not_conflate_calibration(self):
        admission = admit_positive_claim(
            verdict_type="DIAGNOSED", directly_tested=True, verified_evidence=True,
            probability_calibrated=False,
        )
        self.assertTrue(admission.allowed)


if __name__ == "__main__":
    unittest.main()
