"""User-facing visibility rules for generated diagnostic workspaces.

Stress/debug harness projects are test artifacts, not ordinary workspace history.
They remain persisted and can be made visible for debugging by setting
AAOS_SHOW_DIAGNOSTIC_PROJECTS=1; this module never deletes or mutates them.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, TypeVar

ProjectT = TypeVar("ProjectT")
_DIAGNOSTIC_NAME = re.compile(r"^(?:stress project|debug project)(?:\b|[_\s-])", re.IGNORECASE)


def is_diagnostic_project_name(name: object) -> bool:
    return bool(_DIAGNOSTIC_NAME.match(str(name or "").strip()))


def filter_user_facing_projects(projects: Iterable[ProjectT]) -> list[ProjectT]:
    """Hide generated Stress/Debug Project workspaces from normal screens by default.

    Data is preserved. Set AAOS_SHOW_DIAGNOSTIC_PROJECTS=1 to expose diagnostic
    workspaces during development or support investigations.
    """
    if os.getenv("AAOS_SHOW_DIAGNOSTIC_PROJECTS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return list(projects)
    return [
        project for project in projects
        if not is_diagnostic_project_name(getattr(project, "name", ""))
    ]
