import unittest

from packages.analytics_core.src.intelligence.eig_optimizer import EIGOptimizer


class TestDeterministicExperimentRanking(unittest.TestCase):
    def test_equal_utility_is_independent_of_arrival_order(self):
        candidates = [
            {"code": "EXP-B", "target_hypothesis_code": "H2", "computed_utility": 1.0, "fingerprint": "fp-b"},
            {"code": "EXP-A", "target_hypothesis_code": "H1", "computed_utility": 1.0, "fingerprint": "fp-a"},
        ]
        forward = EIGOptimizer.rank_candidates(candidates)
        reverse = EIGOptimizer.rank_candidates(list(reversed(candidates)))
        self.assertEqual(
            [c["code"] for c in forward],
            [c["code"] for c in reverse],
        )
        self.assertEqual([c["code"] for c in forward], ["EXP-A", "EXP-B"])

    def test_near_equal_utility_uses_stable_scientific_key(self):
        candidates = [
            {"code": "EXP-Z", "target_hypothesis_code": "H2", "computed_utility": 1.0000000000001, "fingerprint": "z"},
            {"code": "EXP-A", "target_hypothesis_code": "H1", "computed_utility": 1.0, "fingerprint": "a"},
        ]
        forward = EIGOptimizer.rank_candidates(candidates)
        reverse = EIGOptimizer.rank_candidates(list(reversed(candidates)))
        self.assertEqual([c["code"] for c in forward], [c["code"] for c in reverse])


if __name__ == "__main__":
    unittest.main()
