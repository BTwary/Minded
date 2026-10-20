from pathlib import Path

ROOT = Path(__file__).parents[2]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_sessions_are_persisted_as_parent_records_and_questions_link_back_to_them():
    models = source("apps/api/src/models/entities.py")
    migration = source("migrations/versions/ab42d0e9c771_multi_question_analysis_sessions.py")
    assert "class AnalysisSession(Base)" in models
    assert "accepted_investigation_ids_json" in models
    assert "report_revision" in models
    assert "analysis_session_id = Column(String(36), ForeignKey(\"analysis_sessions.id\")" in models
    assert 'op.create_table(\n        "analysis_sessions"' in migration
    assert 'b.add_column(sa.Column("analysis_session_id"' in migration


def test_closed_sessions_reject_new_questions_and_cannot_close_while_workers_are_active():
    investigation_api = source("apps/api/src/api/v1/investigations.py")
    sessions_api = source("apps/api/src/api/v1/analysis_sessions.py")
    assert 'if str(session.status or "OPEN").upper() != "OPEN"' in investigation_api
    assert "This analysis is closed. Start a new analysis to ask another question." in investigation_api
    assert 'session.status = "CLOSED"' in sessions_api
    assert '"CANCEL_REQUESTED"' in sessions_api
    assert "Cannot close this analysis while" in sessions_api


def test_cumulative_exports_include_only_accepted_verified_questions():
    sessions_api = source("apps/api/src/api/v1/analysis_sessions.py")
    exporter = source("packages/analytics_core/src/reporting/session_exporter.py")
    assert "accepted_investigation_ids_json" in sessions_api
    assert 'row.get("review_status") or ""' in sessions_api
    assert '!= "ACCEPTED_AND_VERIFIED"' in sessions_api
    assert "export_analysis_session_to_pdf" in sessions_api
    assert "export_analysis_session_to_docx" in sessions_api
    assert "Each question section retains its own executed calculation" in exporter
    assert "Questions included" in exporter


def test_session_questions_are_not_duplicated_as_separate_normal_history_entries():
    api = source("apps/api/src/api/v1/investigations.py")
    history_route = api[api.index('@router.get("/history")'):api.index('@router.delete("/history")')]
    assert "Investigation.analysis_session_id.is_(None)" in history_route
    assert 'payload["questions"] = [' in source("apps/api/src/api/v1/analysis_sessions.py")
    assert '"accepted_into_report": row.id in (session.accepted_investigation_ids_json or [])' in source("apps/api/src/api/v1/analysis_sessions.py")


def test_frontend_sends_session_id_and_makes_acceptance_add_a_report_section():
    analyst = source("apps/web/src/app/analyst/page.tsx")
    review_ui = source("apps/web/src/components/StructuredAnalysisAnswer.tsx")
    history_ui = source("apps/web/src/app/history/page.tsx")
    assert "analysis_session_id: activeSession.id" in analyst
    assert "analysis_session_id: (await ensureOpenAnalysisSession()).id" in analyst
    assert '"Accept & Add to Report"' in review_ui
    assert "Export cumulative PDF" in analyst
    assert "Close Analysis" in analyst
    assert "fetchAnalysisSessions" in history_ui
    assert "Continue analysis" in history_ui


def test_closed_session_freezes_question_dispatch_guidance_decision_and_verification_mutations():
    api = source("apps/api/src/api/v1/investigations.py")
    assert "def _assert_analysis_session_open_for_child_mutation" in api
    for route in (
        '@router.post("/{investigation_id}/dispatch")',
        '@router.post("/{investigation_id}/cancel")',
        '@router.post("/{investigation_id}/decision")',
        '@router.post("/{investigation_id}/verification/decisions")',
        '@router.post("/{investigation_id}/verification/signoff")',
        '@router.post("/{investigation_id}/guidance")',
    ):
        start = api.index(route)
        next_route = api.find('@router.', start + len(route))
        section = api[start: next_route if next_route >= 0 else len(api)]
        assert "_assert_analysis_session_open_for_child_mutation(db, investigation_id)" in section, route
