"""P1 Regression Suite: Security Role Scoping Fail-Closed and Verdict Empirical Refutation Rigor.

Tests:
1. Admin / Organization Admin with no organization_id fails closed:
   - get_user_authorized_project_ids returns only user-owned projects, never global or cross-tenant projects.
   - verify_project_ownership on other users' projects raises 403 Forbidden.
2. Verdict Engine Counter-Hypothesis Refutation Rigor:
   - Counter-hypothesis refutation requires explicit empirical refutation.
   - Fallback when counter_hypothesis_empirically_refuted is None must be False (fail-closed),
     even when variance explained is high or primary hypothesis is diagnosed.
"""
import sys
import unittest

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '.')

from apps.api.src.models.entities import Base, User, Project
from apps.api.src.core.security import get_user_authorized_project_ids, verify_project_ownership
from packages.analytics_core.src.engines.verdict import VerdictEngine


class TestSecurityRoleScopingFailClosed(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.db = self.SessionFactory()

        # Admin user with NO organization_id
        self.admin_no_org = User(
            id="usr-admin-no-org",
            email="admin_no_org@test.com",
            full_name="Admin No Org",
            hashed_password="pw",
            role="organization_admin",
            organization_id=None,
        )
        # Regular user
        self.regular_user = User(
            id="usr-bob",
            email="bob@test.com",
            full_name="Bob User",
            hashed_password="pw",
            role="analyst",
            organization_id="org-bob",
        )

        # Projects
        self.admin_project = Project(
            id="proj-admin-owned",
            name="Admin Personal Project",
            owner_id=self.admin_no_org.id,
            org_id=None,
        )
        self.bob_project = Project(
            id="proj-bob-owned",
            name="Bob Secret Project",
            owner_id=self.regular_user.id,
            org_id="org-bob",
        )
        self.third_project = Project(
            id="proj-other",
            name="Other Project",
            owner_id="usr-charlie",
            org_id="org-other",
        )

        self.db.add_all([self.admin_no_org, self.regular_user, self.admin_project, self.bob_project, self.third_project])
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_admin_with_no_org_scoped_to_owned_projects_only(self):
        """Organization admin without organization_id must fail closed to only their own projects."""
        authorized_ids = get_user_authorized_project_ids(self.admin_no_org, self.db)
        self.assertIn("proj-admin-owned", authorized_ids)
        self.assertNotIn("proj-bob-owned", authorized_ids, "Admin with no org must NOT access Bob's project")
        self.assertNotIn("proj-other", authorized_ids, "Admin with no org must NOT access Charlie's project")
        self.assertEqual(authorized_ids, ["proj-admin-owned"])

    def test_admin_with_no_org_denied_access_to_other_projects(self):
        """verify_project_ownership must raise 403 when admin has no organization_id and project is not owned."""
        with self.assertRaises(HTTPException) as ctx:
            verify_project_ownership(self.bob_project.id, self.admin_no_org, self.db)
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("Unauthorized", ctx.exception.detail)


class TestVerdictCounterHypothesisRefutationRigor(unittest.TestCase):
    def setUp(self):
        self.engine = VerdictEngine()

    def test_counter_refuted_defaults_to_false_without_explicit_refutation(self):
        """Even with high posterior and variance explained, counter_refuted must fail closed to False."""
        verdict = VerdictEngine.evaluate_verdict(
            leading_hypothesis_code="HYP-01",
            leading_hypothesis_posterior=0.92,
            all_verifications_passed=True,
            variance_explained_pct=85.0,
            is_categorical_diagnostic=True,
            adversarial_attack_survived=True,
            leading_hypothesis_directly_tested=True,
            counter_hypothesis_empirically_refuted=None,  # Not explicitly refuted
        )
        self.assertTrue(verdict.primary_hypothesis_supported)
        self.assertFalse(
            verdict.counter_hypothesis_refuted,
            "counter_hypothesis_refuted must be False when empirical refutation was not performed"
        )

    def test_counter_refuted_explicit_true(self):
        """When counter_hypothesis_empirically_refuted is explicitly True, counter_refuted is True."""
        verdict = VerdictEngine.evaluate_verdict(
            leading_hypothesis_code="HYP-01",
            leading_hypothesis_posterior=0.92,
            all_verifications_passed=True,
            variance_explained_pct=85.0,
            is_categorical_diagnostic=True,
            adversarial_attack_survived=True,
            leading_hypothesis_directly_tested=True,
            counter_hypothesis_empirically_refuted=True,
        )
        self.assertTrue(verdict.primary_hypothesis_supported)
        self.assertTrue(verdict.counter_hypothesis_refuted)

    def test_counter_refuted_explicit_false(self):
        """When counter_hypothesis_empirically_refuted is explicitly False, counter_refuted is False."""
        verdict = VerdictEngine.evaluate_verdict(
            leading_hypothesis_code="HYP-01",
            leading_hypothesis_posterior=0.92,
            all_verifications_passed=True,
            variance_explained_pct=85.0,
            is_categorical_diagnostic=True,
            adversarial_attack_survived=True,
            leading_hypothesis_directly_tested=True,
            counter_hypothesis_empirically_refuted=False,
        )
        self.assertTrue(verdict.primary_hypothesis_supported)
        self.assertFalse(verdict.counter_hypothesis_refuted)


if __name__ == "__main__":
    unittest.main()
