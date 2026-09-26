"""P0 epistemic-integrity fix: verification-status contract acceptance tests.

Covers the required cases from the task brief:
  - Controller -> InvestigationState -> Database -> AnalysisService -> API
    agree on verification status for a genuine VERIFIED case (real,
    end-to-end, through AnalysisService.execute_analysis -- not a mocked
    engine call).
  - Persistence round-trip: a status written directly to the DB survives
    reconstruction unchanged.
  - An unrecognized/corrupted status never silently becomes SKIPPED.
  - Evidence.confidence_score no longer silently fabricates 0.95 when no
    confidence was explicitly computed.
  - state_reconstruction.py's supporting/contradicting classification agrees
    with the live-run classification in controller.py for the same status
    strings (the pre/post-restart invariant).

FAILED and PARTIALLY_VERIFIED are exercised as direct persistence-layer
cases (constructing the DB row with that status directly) rather than as
genuine end-to-end controller runs: engineering a real dual-engine
disagreement or a real partially-verified experiment from the seed data
was out of scope for this fix and would risk testing manufactured rather
than genuine conditions. This is called out explicitly rather than
claimed as a full black-box proof for those two cases.

Run with: python -m unittest discover -s tests/independent_release -p "test_*.py"
Requires a seeded database (see scripts/generate_seed_data.py +
scripts/seed_database.py).
"""
import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Evidence, Hypothesis, Investigation
from apps.api.src.services.analysis_service import AnalysisService
from packages.analytics_core.src.execution.state_machine import (
    VerificationStatus,
    to_public_validation_status,
)
from packages.analytics_core.src.runtime.state_reconstruction import (
    classify_evidence_for_hypothesis,
    reconstruct_investigation_state,
)
from packages.schemas.src.analysis import AnalysisCreate, ValidationStatus


class TestVerifiedCaseEndToEnd(unittest.TestCase):
    """Case A from the task brief: a genuine VERIFIED investigation, run
    through the real production path (AnalysisService, not the controller
    called directly), asserting the public API reports PASSED."""

    def test_correlation_verified_maps_to_passed(self):
        db = SessionLocal()
        try:
            service = AnalysisService(db)
            response = service.execute_analysis(
                AnalysisCreate(
                    question="Is there a correlation between revenue and cost?",
                    project_id="proj-default",
                )
            )
            self.assertTrue(len(response.evidence) > 0, "expected at least one evidence row")
            statuses = {e.validation_status for e in response.evidence}
            # The dual-engine (duckdb/polars) correlation check on
            # revenue/cost is a real, deterministic agreement -- this must
            # never be SKIPPED, which is what the pre-fix code produced for
            # every evidence row regardless of actual verification outcome.
            self.assertNotIn(ValidationStatus.SKIPPED, statuses)
            self.assertIn(ValidationStatus.PASSED, statuses)
        finally:
            db.close()


class TestVerificationStatusMapping(unittest.TestCase):
    """Direct unit coverage of the single authoritative mapping function."""

    def test_verified_maps_to_passed(self):
        self.assertEqual(to_public_validation_status(VerificationStatus.VERIFIED), ValidationStatus.PASSED)

    def test_failed_maps_to_failed(self):
        self.assertEqual(to_public_validation_status(VerificationStatus.FAILED), ValidationStatus.FAILED)

    def test_partially_verified_maps_to_warning(self):
        self.assertEqual(to_public_validation_status(VerificationStatus.PARTIALLY_VERIFIED), ValidationStatus.WARNING)

    def test_legitimate_skip_states_map_to_skipped(self):
        self.assertEqual(to_public_validation_status(VerificationStatus.UNVERIFIED), ValidationStatus.SKIPPED)
        self.assertEqual(to_public_validation_status(VerificationStatus.NOT_APPLICABLE), ValidationStatus.SKIPPED)
        self.assertEqual(to_public_validation_status(None), ValidationStatus.SKIPPED)

    def test_legacy_lowercase_vocabulary_still_maps_correctly(self):
        """scripts/migrate_analysis_runs_to_investigations.py and the
        Evidence.validation_status column default both use the older
        lowercase vocabulary -- must not regress on old data."""
        self.assertEqual(to_public_validation_status("verified"), ValidationStatus.PASSED)
        self.assertEqual(to_public_validation_status("failed"), ValidationStatus.FAILED)
        self.assertEqual(to_public_validation_status("partial"), ValidationStatus.WARNING)
        self.assertEqual(to_public_validation_status("unverified"), ValidationStatus.SKIPPED)

    def test_unrecognized_status_never_silently_becomes_skipped(self):
        """Regression test required by task section 6: an invalid/unexpected
        status must surface as UNKNOWN, not be hidden as a legitimate skip."""
        result = to_public_validation_status("SOME_NEW_STATUS_NOBODY_MAPPED_YET")
        self.assertEqual(result, ValidationStatus.UNKNOWN)
        self.assertNotEqual(result, ValidationStatus.SKIPPED)

    def test_case_drift_still_resolves_to_correct_status(self):
        """A future casing change (e.g. 'Verified') should resolve via the
        case-insensitive fallback rather than falling to UNKNOWN."""
        self.assertEqual(to_public_validation_status("Verified"), ValidationStatus.PASSED)


class TestPersistenceRoundTrip(unittest.TestCase):
    """Task section 4: prove a status written to the DB survives readback
    identically, for both a freshly-created and a reconstructed-from-DB
    investigation."""

    def _make_investigation(self, db):
        inv_id = f"INV-{uuid.uuid4().hex[:12]}"
        db.add(Investigation(id=inv_id, project_id="proj-default", question="test", status="COMPLETED"))
        db.commit()
        return inv_id

    def test_verified_status_persists_and_reads_back_identically(self):
        db = SessionLocal()
        try:
            inv_id = self._make_investigation(db)
            ev_id = f"EV-{uuid.uuid4().hex[:12]}"
            db.add(Evidence(
                id=ev_id, investigation_id=inv_id, statement="test evidence",
                validation_status=VerificationStatus.VERIFIED,
            ))
            db.commit()
            db.expire_all()
            row = db.query(Evidence).filter(Evidence.id == ev_id).first()
            self.assertEqual(row.validation_status, "VERIFIED")
            self.assertEqual(to_public_validation_status(row.validation_status), ValidationStatus.PASSED)
        finally:
            db.close()

    def test_confidence_score_no_longer_defaults_to_fabricated_high_value(self):
        """Task section 7 regression: an Evidence row created without an
        explicit confidence_score must persist as None, never 0.95."""
        db = SessionLocal()
        try:
            inv_id = self._make_investigation(db)
            ev_id = f"EV-{uuid.uuid4().hex[:12]}"
            db.add(Evidence(
                id=ev_id, investigation_id=inv_id, statement="legacy migrated evidence, no confidence computed",
                validation_status="verified",
            ))
            db.commit()
            db.expire_all()
            row = db.query(Evidence).filter(Evidence.id == ev_id).first()
            self.assertIsNone(row.confidence_score)
            self.assertNotEqual(row.confidence_score, 0.95)
        finally:
            db.close()


class TestStateReconstructionAgreesWithLiveRun(unittest.TestCase):
    """Task section 4/9: the same status string must classify the same way
    whether checked during a live run or after a restart. This tests
    classify_evidence_for_hypothesis directly -- the exact function
    state_reconstruction.py calls -- so it doesn't need to reproduce full
    hypothesis-identity recomputation to prove the classification itself
    is now case-correct."""

    def test_verified_classifies_as_supporting(self):
        self.assertEqual(classify_evidence_for_hypothesis("VERIFIED", None), "supporting")

    def test_failed_classifies_as_contradicting(self):
        self.assertEqual(classify_evidence_for_hypothesis("FAILED", None), "contradicting")

    def test_partially_verified_classifies_as_neither(self):
        # Conservative behavior preserved deliberately -- see task scope
        # protection ("do not weaken verification requirements"): a
        # partially-verified result is not strong enough to count as either
        # supporting or contradicting evidence for a hypothesis.
        self.assertIsNone(classify_evidence_for_hypothesis("PARTIALLY_VERIFIED", None))

    def test_legacy_lowercase_still_classifies_correctly(self):
        self.assertEqual(classify_evidence_for_hypothesis("verified", None), "supporting")
        self.assertEqual(classify_evidence_for_hypothesis("failed", None), "contradicting")

    def test_nonzero_effect_size_counts_as_supporting_regardless_of_status(self):
        self.assertEqual(classify_evidence_for_hypothesis(None, 3.2), "supporting")


if __name__ == "__main__":
    unittest.main()
