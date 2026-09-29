import unittest

from packages.analytics_core.src.governance.recommendation_grounding import (
    evaluate_recommendation_grounding,
)


class TestRecommendationGrounding(unittest.TestCase):
    def _evaluate(self, support, contradict=(), validation=None, verification=None):
        return evaluate_recommendation_grounding(
            support,
            contradict,
            validation or {},
            verification or {},
        )

    def test_no_support_is_unsupported(self):
        result = self._evaluate([])
        self.assertEqual(result.status, "UNSUPPORTED")
        self.assertEqual(result.supporting_evidence_ids, ())

    def test_unverified_support_is_not_grounded(self):
        result = self._evaluate(
            ["EV1"],
            validation={"EV1": "VERIFIED"},
            verification={"EV1": "UNVERIFIED"},
        )
        self.assertEqual(result.status, "UNSUPPORTED")
        self.assertEqual(result.unverified_supporting_evidence_ids, ("EV1",))

    def test_verified_support_is_grounded(self):
        result = self._evaluate(
            ["EV1", "EV2"],
            validation={"EV1": "VERIFIED", "EV2": "PASSED"},
            verification={"EV1": "VERIFIED", "EV2": "PASSED"},
        )
        self.assertEqual(result.status, "GROUNDED")
        self.assertEqual(result.verified_supporting_evidence_ids, ("EV1", "EV2"))

    def test_verified_contradiction_blocks_grounding(self):
        result = self._evaluate(
            ["EV1"],
            contradict=["EV9"],
            validation={"EV1": "VERIFIED", "EV9": "VERIFIED"},
            verification={"EV1": "VERIFIED", "EV9": "VERIFIED"},
        )
        self.assertEqual(result.status, "INCONCLUSIVE")
        self.assertEqual(result.verified_contradicting_evidence_ids, ("EV9",))

    def test_latest_failed_verification_blocks_prior_pass(self):
        result = self._evaluate(
            ["EV1"],
            validation={"EV1": "VERIFIED"},
            verification={"EV1": "FAILED"},
        )
        self.assertEqual(result.status, "UNSUPPORTED")

    def test_duplicate_ids_are_deduplicated(self):
        result = self._evaluate(
            ["EV1", "EV1"],
            validation={"EV1": "VERIFIED"},
            verification={"EV1": "VERIFIED"},
        )
        self.assertEqual(result.status, "GROUNDED")
        self.assertEqual(result.supporting_evidence_ids, ("EV1",))


if __name__ == "__main__":
    unittest.main()
