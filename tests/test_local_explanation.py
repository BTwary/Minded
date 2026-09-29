import unittest
from types import SimpleNamespace

from packages.analytics_core.src.engines.local_explanation import LocalExplanationEngine


class TestLocalExplanationEngine(unittest.TestCase):
    def test_explanation_is_deterministic_and_uses_persisted_facts(self):
        hypotheses = [
            SimpleNamespace(status="supported", statement="Region A explains most of the decline.", posterior_probability=0.84),
            SimpleNamespace(status="refuted", statement="Systemic decline explains the change.", posterior_probability=0.16),
        ]
        experiments = [SimpleNamespace() for _ in range(3)]
        observations = [SimpleNamespace(), SimpleNamespace()]
        evidence = [
            SimpleNamespace(validation_status="VERIFIED", statement="Region A contributed 78% of the observed decline."),
            SimpleNamespace(validation_status="FAILED", statement="A second specification could not be independently verified."),
        ]
        result = LocalExplanationEngine.build(
            question="Why did revenue fall?",
            verdict_type="STATISTICALLY_SIGNIFICANT",
            direct_answer="Region A was the main contributor.",
            justification="Verified evidence supports the leading diagnostic explanation.",
            hypotheses=hypotheses,
            experiments=experiments,
            observations=observations,
            evidence_rows=evidence,
            method_selection={
                "problem_class": "diagnostic",
                "objective": "root_cause",
                "method_family": "diagnostic_battery",
            },
        )
        self.assertEqual(result.mode, "DETERMINISTIC")
        self.assertIn("diagnostic", result.what_was_analyzed.lower())
        self.assertEqual(result.conclusion, "Region A explains most of the decline.")
        self.assertEqual(len(result.evidence), 2)
        self.assertIn("1 of 2", result.verification)
        self.assertTrue(any("not independently verified" in x for x in result.limitations))

    def test_local_mode_never_requires_ai(self):
        result = LocalExplanationEngine.build(
            question="What happened?",
            verdict_type="INCONCLUSIVE",
            direct_answer="Insufficient evidence.",
            justification="The evidence did not support a reliable conclusion.",
        )
        self.assertEqual(result.mode, "DETERMINISTIC")
        self.assertIn("inconclusive", result.summary.lower())
        self.assertTrue(any("inconclusive" in x.lower() for x in result.limitations))


if __name__ == "__main__":
    unittest.main()
