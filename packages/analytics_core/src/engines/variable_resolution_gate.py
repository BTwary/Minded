"""
variable_resolution_gate.py

Centralized pre-analysis variable-resolution gate (VARIABLE_NOT_FOUND).

Context (see docs/... question-invariance workstream, Phase 1 baseline):
four independent estimand families (correlation, rate/proportion, trend,
ranking) each had their own ad-hoc column-resolution code, and none of them
distinguished "the question named no specific variable" from "the question
named a specific variable that does not exist in this dataset". When the
requested variable did not exist, every family except the churn-specific
path silently fell back to whatever column happened to be available
(usually the first categorical/numeric pair) and reported a confident
answer to a question that was never asked -- in two observed cases at
confidence 1.0.

This module is the single, family-agnostic check that must run before any
of those synthesizers get a chance to do that fallback binding. It does not
try to resolve what the user *meant* (that remains each family's job); it
only answers one narrow, high-precision question: does the raw question
text contain what looks like a specific physical-column reference (a
snake_case identifier, e.g. "marketing_spend", "widget_count",
"blorptastic_index") that does not correspond -- exactly, or under simple
case/separator normalization -- to any column in the available schema?

Deliberately NOT in scope here (by design, to keep this check high
precision and low false-positive):
  - Bare English words with no underscore ("returned", "orders") are not
    flagged. Those questions are still protected against fabrication by
    each family's own admissibility gate; this module only owns the
    narrower "the user typed an identifier that isn't a real column"
    signal, which is the one no family was handling generically.
  - Churn/attrition/retention-flavored tokens are excluded here entirely.
    That family already has its own dedicated, tested fail-closed
    resolution (SemanticEngine.resolve_schema's churn_event_col ->
    churn_outcome_available -> "no identifiable churn outcome" path) and
    this gate must not preempt it with a different, less specific message.
"""
from __future__ import annotations

import re
from typing import Iterable, List

# A "physical-identifier-style" token: starts with a letter, contains at
# least one underscore-joined segment. This deliberately matches the shape
# of an actual database/dataframe column name (snake_case) rather than
# ordinary English phrasing, which keeps false positives low: everyday
# words and multi-word phrases ("average order value") never match this
# pattern, only tokens that look like the user is naming a column directly.
_IDENTIFIER_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b")

# Tokens touching this vocabulary are owned by the existing, dedicated
# churn-identifiability path (see semantic.py) and must not be re-flagged
# here with a different, generic message.
_CHURN_OWNED_SUBSTRINGS = (
    "churn", "attrition", "cancel", "retention", "dropout", "dropoff",
)


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[\s-]+", "_", str(name).strip().lower())


# A second, narrower detection rule for the rate/proportion family
# specifically: a question phrased around "proportion"/"rate"/"percentage"
# that names NO column from the available schema at all (not even a
# synonym token, not even partially) is asking about an outcome this
# dataset simply does not contain. Scoped tightly to this vocabulary (as
# opposed to a blanket "zero column tokens" rule for every question,
# which would be far too broad and would misfire on legitimate
# alias-resolved questions in other task families) because this is the
# one documented case where the rate family currently substitutes an
# unrelated ranking/aggregation instead of reporting the missing outcome.
_RATE_PROPORTION_RE = re.compile(r"\b(proportion|percentage|percent|rate)\b", re.I)


def _question_references_any_column(question: str, columns: Iterable[str]) -> bool:
    q_norm = re.sub(r"[^a-z0-9]+", " ", (question or "").lower())
    q_tokens = set(q_norm.split())
    for c in columns:
        col_phrase = _normalize_column_name(c).replace("_", " ").strip()
        if not col_phrase:
            continue
        col_words = col_phrase.split()
        if len(col_words) == 1:
            if col_words[0] in q_tokens:
                return True
        elif col_phrase in q_norm:
            return True
    return False


def detect_rate_question_with_no_column_reference(question: str, columns: Iterable[str]) -> bool:
    """True when `question` uses rate/proportion vocabulary but names no
    column from `columns` anywhere -- the outcome it asks about isn't in
    this dataset at all, as opposed to merely being phrased with a synonym
    of some column that IS present."""
    if not _RATE_PROPORTION_RE.search(question or ""):
        return False
    ql = (question or "").lower()
    if any(sub in ql for sub in _CHURN_OWNED_SUBSTRINGS):
        # Churn-flavored rate questions ("churn rate by segment") are owned
        # by the existing dedicated churn-identifiability path, which gives
        # a more specific, already-correct answer than this generic gate
        # would ("no identifiable churn outcome" vs. a generic missing-
        # variable message) -- do not preempt it.
        return False
    return not _question_references_any_column(question, columns)


def detect_unresolved_requested_variables(question: str, columns: Iterable[str]) -> List[str]:
    """Return the distinct identifier-style tokens named in `question` that
    do not correspond to any column in `columns` (case/separator-insensitive).

    Returns an empty list when nothing that looks like an explicitly named,
    nonexistent variable is found -- callers should treat that as "this
    gate has no objection", not as "the question is fully resolved".
    """
    normalized_cols = {_normalize_column_name(c) for c in columns}
    found: List[str] = []
    for tok in _IDENTIFIER_TOKEN_RE.findall(question or ""):
        tok_norm = tok.lower()
        if any(sub in tok_norm for sub in _CHURN_OWNED_SUBSTRINGS):
            continue
        if tok_norm in normalized_cols:
            continue
        if tok not in found:
            found.append(tok)
    return found
