import os
from types import SimpleNamespace

from apps.api.src.core.project_visibility import filter_user_facing_projects, is_diagnostic_project_name


def test_generated_stress_and_debug_projects_are_identified():
    assert is_diagnostic_project_name("Stress Project U8R27")
    assert is_diagnostic_project_name("Debug Project")
    assert not is_diagnostic_project_name("Regional Sales")
    assert not is_diagnostic_project_name("Stress testing operations")


def test_diagnostic_projects_are_hidden_from_normal_lists_without_deleting_them(monkeypatch):
    monkeypatch.delenv("AAOS_SHOW_DIAGNOSTIC_PROJECTS", raising=False)
    rows = [
        SimpleNamespace(id="real", name="Regional Sales"),
        SimpleNamespace(id="stress", name="Stress Project U8R27"),
        SimpleNamespace(id="debug", name="Debug Project U1R37"),
    ]
    visible = filter_user_facing_projects(rows)
    assert [row.id for row in visible] == ["real"]
    assert len(rows) == 3  # visibility is non-destructive


def test_diagnostic_projects_can_be_explicitly_shown_for_debugging(monkeypatch):
    monkeypatch.setenv("AAOS_SHOW_DIAGNOSTIC_PROJECTS", "1")
    rows = [SimpleNamespace(id="stress", name="Stress Project U8R27")]
    assert filter_user_facing_projects(rows) == rows
