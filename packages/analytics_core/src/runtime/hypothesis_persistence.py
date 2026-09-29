"""Database-level idempotent persistence for canonical hypotheses.

The in-memory ``HypothesisConsolidator`` / ``InvestigationStateManager`` path
(see ``hypothesis_consolidation.py``) is what makes a *single controller
process* agree with itself about which hypothesis a given canonical identity
belongs to. It cannot, by itself, protect against two different processes
(a genuine concurrent-worker race, a replay racing a still-running worker,
etc.) each deciding independently that a given proposition is "new" and
attempting to insert two separate hypothesis rows for it.

That guarantee has to live at the database: ``hypotheses`` carries a
``UNIQUE(investigation_id, canonical_identity)`` constraint, and this module
is the ONLY place that writes a hypothesis row. It always resolves through
the constraint rather than assuming the caller's in-memory view is correct:

    1. Look for an existing row for this (investigation_id, canonical_identity).
       If found, that row is authoritative -- return it unchanged.
    2. Otherwise attempt to insert the caller's candidate row.
    3. If the insert loses a race (IntegrityError on the uniqueness
       constraint), roll back the failed insert and re-read: some other
       writer won the race, and their row is the canonical one.

Callers must never fall back to ``session.add``/``session.merge`` on the
``Hypothesis`` model directly -- that would reopen exactly the gap this
module exists to close.
"""
from __future__ import annotations

from typing import Any, Callable, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session


def get_or_create_hypothesis_row(
    session: Session,
    investigation_id: str,
    canonical_identity: str,
    build_entity: Callable[[], Any],
    update_existing: Callable[[Any], None] | None = None,
) -> Tuple[Any, bool]:
    """Idempotent get-or-create for a ``Hypothesis`` row, keyed by the
    database's own ``UNIQUE(investigation_id, canonical_identity)``
    constraint rather than by any in-memory registry.

    ``build_entity`` is called at most once, to construct the row to insert
    if no canonical row exists yet. ``update_existing``, if given, is called
    on an already-existing row every time one is found (created this call or
    not) so callers can apply idempotent field refreshes (e.g. folding in
    newly-observed provenance) without ever inserting a second row.

    Returns ``(row, created)``.
    """
    from apps.api.src.models.entities import Hypothesis  # local import: avoid import cycles

    def _lookup():
        return (
            session.query(Hypothesis)
            .filter(
                Hypothesis.investigation_id == investigation_id,
                Hypothesis.canonical_identity == canonical_identity,
            )
            .first()
        )

    existing = _lookup()
    if existing is not None:
        if update_existing is not None:
            update_existing(existing)
        return existing, False

    entity = build_entity()
    session.add(entity)
    try:
        session.flush()
        return entity, True
    except IntegrityError:
        # Lost a genuine concurrent-creation race: some other writer
        # committed the canonical row for this identity between our lookup
        # and our insert. Discard our attempted insert and defer to theirs
        # -- do NOT retry the insert, and do NOT create a second row.
        session.rollback()
        survivor = _lookup()
        if survivor is None:
            # The constraint fired for a reason other than this identity
            # (or the survivor is not yet visible under this isolation
            # level) -- surface the original failure rather than silently
            # dropping the hypothesis.
            raise
        if update_existing is not None:
            update_existing(survivor)
        return survivor, False
