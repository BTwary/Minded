"""Shared vocabulary for semantic_binding_builder.py and
semantic_resolution_builder.py (v20-C1 hardening item A).

Both builders classify the same raw ``SemanticResolution`` fields into two
independent output shapes (a flat ``SemanticBindingSet`` vs. the richer
``CanonicalSemanticResolution``), but they were making the same two
classification calls -- "is this task TARGET-style?" and "what
ResolutionStatus does this churn status string map to?" -- from two
separately-declared, textually-identical constants. That was a drift risk:
the two literals agreed by construction at v20-C1 but had nothing enforcing
they stay in sync as task/status vocabulary evolves.

Both builders now import these two constants from here instead of
declaring their own copies. This is the same "one shared vocabulary, no
duplicate authority" principle SemanticRole/ResolutionStatus already follow
in packages/schemas/src/semantic_role.py, applied to these two
builder-local classification tables.
"""
from __future__ import annotations

from packages.schemas.src.semantic_role import ResolutionStatus

# Problem classes (UniversalQuestionCompiler task vocabulary) for which the
# primary resolved metric is best described as TARGET (something to be
# forecast/predicted) rather than OUTCOME (something to be explained).
TARGET_STYLE_TASKS = frozenset({"FORECAST", "PREDICTION"})

# Maps the raw (string) churn_event_resolution_status value that
# SemanticResolution exposes onto the canonical ResolutionStatus enum.
CHURN_STATUS_MAP = {
    "RESOLVED": ResolutionStatus.RESOLVED,
    "AMBIGUOUS": ResolutionStatus.AMBIGUOUS,
    "UNRESOLVED": ResolutionStatus.UNRESOLVED,
}
