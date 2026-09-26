"""P1-16 Regression: Cross-project report analysis_id isolation.

Proves that attempting to create a report in Project A referencing an
analysis_id from Project B raises a 403 Forbidden cross-project violation.
"""
import sys
import unittest
sys.path.insert(0, '.')

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.src.models.entities import Base, User, Project, AnalysisRun, Report
from apps.api.src.api.v1.reports import create_report
from packages.schemas.src.report import ReportCreate


class TestCrossProjectReportIsolation(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.db = self.SessionFactory()

        self.user = User(id='usr-alice', email='alice@test.com', full_name='Alice', hashed_password='x', role='analyst')
        self.proj_a = Project(id='proj-a', name='Project A', owner_id=self.user.id, org_id='org-1')
        self.proj_b = Project(id='proj-b', name='Project B', owner_id=self.user.id, org_id='org-1')

        # Analysis run in Project B
        self.run_b = AnalysisRun(
            id='run-in-proj-b',
            project_id=self.proj_b.id,
            question='What drove churn?',
            status='COMPLETED',
            direct_answer='Churn was driven by latency.',
            main_finding='High latency in eu-central.',
            confidence=0.92,
        )

        self.db.add_all([self.user, self.proj_a, self.proj_b, self.run_b])
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_cross_project_analysis_id_rejected(self):
        """Creating a report in proj-a referencing an analysis in proj-b must raise 403."""
        payload = ReportCreate(
            project_id=self.proj_a.id,
            analysis_id=self.run_b.id,
            title='Alice Cross-Project Attempt',
            report_type='executive',
        )

        with self.assertRaises(HTTPException) as ctx:
            create_report(payload=payload, current_user=self.user, db=self.db)

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn('Cross-project isolation violation', ctx.exception.detail)

    def test_same_project_analysis_id_succeeds(self):
        """Creating a report in proj-b referencing an analysis in proj-b must succeed."""
        payload = ReportCreate(
            project_id=self.proj_b.id,
            analysis_id=self.run_b.id,
            title='Legitimate Project B Report',
            report_type='executive',
        )

        resp = create_report(payload=payload, current_user=self.user, db=self.db)
        self.assertEqual(resp.project_id, self.proj_b.id)
        self.assertEqual(resp.analysis_id, self.run_b.id)
        self.assertIn('Churn was driven by latency', resp.markdown_content)
        self.assertIn('PENDING_REVIEW', resp.markdown_content)

    def test_report_carries_signed_off_status(self):
        """When an investigation has been signed off, the report markdown must carry the signoff details."""
        from apps.api.src.models.entities import InvestigationEvent
        from datetime import datetime, timezone
        ev = InvestigationEvent(
            id='ev-signoff-1',
            investigation_id=self.run_b.id,
            execution_id='exec-1',
            event_type='verification.signed_off',
            timestamp=datetime.now(timezone.utc),
            event_payload_json={
                'outcome': 'ACCEPTED',
                'comment': 'All 3 items verified against source database.',
                'reviewer': {'email': 'lead_verifier@enterprise.com'},
                'packet_fingerprint': 'abc1234567890def1234567890',
            },
        )
        self.db.add(ev)
        self.db.commit()

        payload = ReportCreate(
            project_id=self.proj_b.id,
            analysis_id=self.run_b.id,
            title='Verified Project B Report',
            report_type='executive',
        )

        resp = create_report(payload=payload, current_user=self.user, db=self.db)
        self.assertIn('SIGNED_OFF (ACCEPTED)', resp.markdown_content)
        self.assertIn('lead_verifier@enterprise.com', resp.markdown_content)
        self.assertIn('All 3 items verified against source database', resp.markdown_content)


if __name__ == '__main__':
    unittest.main()
