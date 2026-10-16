"""Regression tests for final verification and evidence-lineage policy."""
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.governance.verification_gate import (
    has_current_verified_supporting_evidence,
    verification_gate_reason,
)


class TestFinalVerificationOverlayGate(unittest.TestCase):
    def test_failed_experiments_block_significance_even_if_leading_evidence_flag_is_true(self):
        self.assertIn("one or more experiments", verification_gate_reason(
            "STATISTICALLY_SIGNIFICANT", all_verifications_passed=False, has_verified_evidence=True,
        ))

    def test_missing_leading_evidence_blocks_significance(self):
        self.assertIn("no independently verified supporting evidence", verification_gate_reason(
            "STATISTICALLY_SIGNIFICANT", all_verifications_passed=True, has_verified_evidence=False,
        ))

    def test_failed_verification_blocks_all_hypothesis_verdicts(self):
        for verdict in ("DIAGNOSED", "STATISTICALLY_SIGNIFICANT", "NO_DETECTABLE_EFFECT"):
            with self.subTest(verdict=verdict):
                self.assertIsNotNone(verification_gate_reason(
                    verdict, all_verifications_passed=False, has_verified_evidence=False,
                ))

    def test_verified_hypothesis_verdict_is_not_blocked(self):
        for verdict in ("DIAGNOSED", "STATISTICALLY_SIGNIFICANT", "NO_DETECTABLE_EFFECT"):
            with self.subTest(verdict=verdict):
                self.assertIsNone(verification_gate_reason(
                    verdict, all_verifications_passed=True, has_verified_evidence=True,
                ))

    def test_descriptive_observed_is_not_mistaken_for_hypothesis_verdict(self):
        self.assertIsNone(verification_gate_reason(
            "OBSERVED", all_verifications_passed=False, has_verified_evidence=False,
        ))

    def test_string_false_cannot_pass_gate_by_truthiness(self):
        self.assertIsNotNone(verification_gate_reason(
            "STATISTICALLY_SIGNIFICANT", all_verifications_passed="false", has_verified_evidence=True,
        ))
        self.assertIsNotNone(verification_gate_reason(
            "STATISTICALLY_SIGNIFICANT", all_verifications_passed=True, has_verified_evidence="false",
        ))

    def test_only_currently_verified_supporting_evidence_counts(self):
        self.assertTrue(has_current_verified_supporting_evidence(
            ["support-1"],
            evidence_validation_status_by_id={"support-1": "VERIFIED"},
            latest_verification_status_by_id={"support-1": "VERIFIED"},
        ))
        # Newest independent verification failed: old VERIFIED must not count.
        self.assertFalse(has_current_verified_supporting_evidence(
            ["support-1"],
            evidence_validation_status_by_id={"support-1": "VERIFIED"},
            latest_verification_status_by_id={"support-1": "FAILED"},
        ))
        # A verification record alone is insufficient if the evidence itself is partial.
        self.assertFalse(has_current_verified_supporting_evidence(
            ["support-1"],
            evidence_validation_status_by_id={"support-1": "PARTIALLY_VERIFIED"},
            latest_verification_status_by_id={"support-1": "VERIFIED"},
        ))

    def test_contradictory_evidence_cannot_be_substituted_for_support(self):
        self.assertFalse(has_current_verified_supporting_evidence(
            [],
            evidence_validation_status_by_id={"contradiction-1": "VERIFIED"},
            latest_verification_status_by_id={"contradiction-1": "VERIFIED"},
        ))

    def test_missing_or_empty_lineage_fails_closed(self):
        self.assertFalse(has_current_verified_supporting_evidence(
            None, evidence_validation_status_by_id={}, latest_verification_status_by_id={},
        ))

    def test_controller_places_final_gate_after_overlay_and_before_manifest(self):
        repo_root = Path(__file__).resolve().parents[2]
        source = (repo_root / "packages/analytics_core/src/runtime/controller.py").read_text(encoding="utf-8")
        gate = source.index("# FINAL CLAIM-INTEGRITY GATE (P0)")
        overlay = source.index("# Analyst-result overlay", source.index("# 8. Epistemic Calibration")) if "# Analyst-result overlay" in source else source.index("if analyst_result is not None:", source.index("# 8. Epistemic Calibration"))
        manifest = source.index("manifest_hash = ProvenanceEngine.compute_reproducible_manifest_hash(", gate)
        self.assertLess(overlay, gate)
        self.assertLess(gate, manifest)
        # No later branch may upgrade an inferential verdict after the final gate.
        later_assignments = re.findall(r'verdict_eval\.verdict_type\s*=\s*["\']([^"\']+)["\']', source[gate:])
        self.assertTrue(all(value == "INCONCLUSIVE" for value in later_assignments), later_assignments)


if __name__ == "__main__":
    unittest.main()
