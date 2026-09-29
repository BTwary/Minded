"""Canonical semantic-resolution model (v20-C1).

v20-A/v20-B1 gave AA-OS a canonical ``SemanticBindingSet``: a flat list of
(column, role, status) facts, plus four compatibility projections
(``projected_target_column`` etc.) for reading a single resolved value back
out of it. That is sufficient for the execution gate
(``validate_experiment_against_contract``), which only ever needs to ask
"is this specific column bound, executable, and role-compatible?".

It is NOT sufficient for hypothesis/experiment synthesis
(``predictive_hypothesis.py``, ``experiment_synthesizer.py``), which need
to ask richer questions the flat binding list cannot represent:

  - "was this field ambiguous, and if so, among which candidates?" (a
    SemanticBinding only records the WINNING resolution status; it has
    nowhere to put the candidate list that made a churn-event or
    grouping-dimension resolution AMBIGUOUS rather than RESOLVED).
  - "what is the full pool of columns that COULD have been used here?"
    (e.g. every categorical column in the dataset, most of which were
    never bound to anything -- ``available_categorical_cols``).
  - "what collection of columns plays this analytical PURPOSE?" (e.g. the
    churn confounder set is not one binding, it's a set of
    EXPLANATORY_VARIABLE bindings all serving the same discovery purpose,
    and that purpose would otherwise be lost the moment they're flattened
    into the same binding list as everything else).

Forcing all of that through more single-value projection methods (as
v20-B1's four `projected_*` methods do) would silently throw away exactly
the scientific provenance ``SemanticResolution`` currently carries -- e.g.
collapsing "AMBIGUOUS between 'churn' and 'cancelled'" down to a bare
``None`` loses the candidate list a human (or a later stage) would need to
resolve the ambiguity. Hence this richer, explicit model instead:

    CanonicalSemanticResolution
        bindings: SemanticBindingSet          (unchanged, v20-A/B1)
        outcome / dimension / time /
          secondary_metric / exposure /
          censored: ResolvedSemanticField     (single-value + status + candidates)
        confounders: SemanticCandidateSet     (a collection, not a single value)
        available_categorical_candidates:     (the raw dataset-wide pool,
          List[str]                            distinct from any one
                                                field's candidate list)

``SemanticBindingSet`` is NOT replaced -- it remains the execution gate's
source of truth (validate_experiment_against_contract, the persisted
contract's semantic_bindings_json) and is embedded here unchanged. This
model sits ABOVE it, for consumers (hypothesis/experiment synthesis) that
need the resolution story the flat binding list intentionally discards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Optional

from packages.schemas.src.semantic_binding import SemanticBindingSet
from packages.schemas.src.semantic_role import ResolutionStatus, SemanticRole


@dataclass(frozen=True)
class ResolvedSemanticField:
    """One analytical field's full resolution story: not just the winning
    value, but the status that produced it and (when status is not a clean
    single RESOLVED) the candidate columns that were in play.

    Invariants (mirrors resolve_group_dimension's existing fail-closed
    contract, generalized to every field built this way):
      RESOLVED    -> value is set, candidates == [value]
      AMBIGUOUS   -> value is None, candidates has >= 2 entries
      UNRESOLVED  -> value is None, candidates == []
    A field may also be genuinely not-applicable to this investigation
    (e.g. secondary_metric on a non-correlation question) -- that is
    UNRESOLVED with an empty candidate list, identical in shape to "no
    candidates existed", since downstream consumers must treat both the
    same way (nothing safe to read).
    """
    role: SemanticRole
    value: Optional[str]
    status: ResolutionStatus
    candidates: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce the RESOLVED/AMBIGUOUS/UNRESOLVED shape invariants
        documented above at construction time, rather than trusting every
        call site (builder, test fixture, future consumer) to keep them by
        convention. A field built outside semantic_resolution_builder.py
        that violates its own declared status is a bug we want to surface
        immediately, not silently propagate downstream."""
        if self.status == ResolutionStatus.RESOLVED:
            if not self.value:
                raise ValueError(
                    f"ResolvedSemanticField({self.role}) is RESOLVED but has no value"
                )
            if list(self.candidates) != [self.value]:
                raise ValueError(
                    f"ResolvedSemanticField({self.role}) is RESOLVED for "
                    f"{self.value!r} but candidates are {self.candidates!r} "
                    f"(expected [{self.value!r}])"
                )
        elif self.status == ResolutionStatus.AMBIGUOUS:
            if self.value is not None:
                raise ValueError(
                    f"ResolvedSemanticField({self.role}) is AMBIGUOUS but has a value ({self.value!r})"
                )
            if len(self.candidates) < 2:
                raise ValueError(
                    f"ResolvedSemanticField({self.role}) is AMBIGUOUS but has "
                    f"{len(self.candidates)} candidate(s); AMBIGUOUS requires >= 2"
                )
        elif self.status == ResolutionStatus.UNRESOLVED:
            if self.value is not None:
                raise ValueError(
                    f"ResolvedSemanticField({self.role}) is UNRESOLVED but has a value ({self.value!r})"
                )
            if self.candidates:
                raise ValueError(
                    f"ResolvedSemanticField({self.role}) is UNRESOLVED but has "
                    f"candidates {self.candidates!r}; UNRESOLVED requires none"
                )
        # INFERRED/USER_SPECIFIED and any other ResolutionStatus values are
        # not part of this field's documented contract (see module
        # docstring) -- left unvalidated rather than guessed at.

        # v20-C2.1 hardening: the shape checks above ran against the
        # caller's own list/dict objects. Freeze AFTER validating so a
        # later in-place mutation of that same list/dict (e.g.
        # `field.candidates.append(...)`) can no longer desynchronize the
        # already-validated RESOLVED/AMBIGUOUS/UNRESOLVED invariant from
        # what a downstream reader actually sees -- frozen=True alone only
        # stopped `field.candidates = [...]`, not this.
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def is_resolved(self) -> bool:
        return self.status == ResolutionStatus.RESOLVED and bool(self.value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value if isinstance(self.role, SemanticRole) else self.role,
            "value": self.value,
            "status": self.status.value if isinstance(self.status, ResolutionStatus) else self.status,
            "candidates": list(self.candidates),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class SemanticCandidateSet:
    """A COLLECTION of columns serving one analytical purpose (e.g. "the
    confounder set"), as distinct from ResolvedSemanticField which is
    exactly one winning value. There is no "ambiguous" state here -- an
    empty ``columns`` list simply means no columns were identified for
    this purpose, which is a normal, common outcome (most investigations
    have no confounder candidates at all).
    """
    role: SemanticRole
    purpose: str
    columns: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # v20-C2.1 hardening: same in-place-mutation gap as
        # ResolvedSemanticField -- freeze the collection and provenance
        # after construction rather than leaving them as plain list/dict.
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def is_empty(self) -> bool:
        return len(self.columns) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value if isinstance(self.role, SemanticRole) else self.role,
            "purpose": self.purpose,
            "columns": list(self.columns),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class QuestionRoleProposal:
    """v20-C4.2.1: the explicit analytical roles the UniversalQuestionCompiler
    grounded in the user's own question, handed to canonical resolution as
    ONE argument instead of a growing list of loose keyword arguments.

    Three distinct ideas that must never be treated as synonyms:

      REQUESTED variable   -- named by the question ("plan tier" in "Does
                              plan tier affect cancellation rate?").
      DISCOVERED variable  -- found by SemanticEngine from the data
                              (secondary_metric_col, e.g. tenure_months).
      USEFUL variable      -- worth exploring experimentally (a candidate
                              confounder); the adaptive loop may pursue
                              it, but it is not the answer to the question.

    ``explanatory_columns`` is everything the compiler put in its
    explanatory role, but that list is NOT all question-explicit: the
    compiler also falls back to semantic bindings (group/secondary metric)
    when the question names nothing. Only columns that are ALSO in
    ``referenced_columns`` (the compiler's own record of columns the
    question actually referenced or confidently resolved) are treated as
    explicit; the rest are exposed separately as inferred and are never
    authoritative.
    """
    explanatory_columns: List[str] = field(default_factory=list)
    referenced_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    requested_explanatory: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        if self.requested_explanatory is not None:
            req = tuple(self.requested_explanatory)
            expl = tuple(self.explanatory_columns) if self.explanatory_columns else req
            ref = tuple(self.referenced_columns) if self.referenced_columns else req
            object.__setattr__(self, "requested_explanatory", req)
            object.__setattr__(self, "explanatory_columns", expl)
            object.__setattr__(self, "referenced_columns", ref)
        else:
            object.__setattr__(self, "explanatory_columns", tuple(self.explanatory_columns))
            object.__setattr__(self, "referenced_columns", tuple(self.referenced_columns))
            object.__setattr__(self, "requested_explanatory", tuple(self.explicit_explanatory_columns()))

    @classmethod
    def from_plan_semantics(cls, plan_semantics: Any) -> "QuestionRoleProposal":
        """Build from a compiled plan's SemanticContract (``plan.semantics``).
        Tolerates a missing/partial contract by producing an empty proposal
        (which resolves nothing), never by raising."""
        return cls(
            explanatory_columns=list(getattr(plan_semantics, "explanatory_columns", None) or []),
            referenced_columns=list(getattr(plan_semantics, "referenced_columns", None) or []),
            target_column=getattr(plan_semantics, "target_column", None),
        )

    def explicit_explanatory_columns(self) -> List[str]:
        referenced = set(self.referenced_columns)
        out: List[str] = []
        for col in self.explanatory_columns:
            if col and col in referenced and col != self.target_column and col not in out:
                out.append(col)
        return out

    def inferred_explanatory_columns(self) -> List[str]:
        explicit = set(self.explicit_explanatory_columns())
        return [c for c in self.explanatory_columns if c and c not in explicit and c != self.target_column]

    def deterministic_requested_explanatory(self) -> List[str]:
        return sorted(dict.fromkeys(self.explicit_explanatory_columns()))


@dataclass(frozen=True)
class CanonicalSemanticResolution:
    """The full canonical semantic-resolution result for one contract
    version: the flat execution-facing SemanticBindingSet, PLUS the richer
    resolution story each of the underlying SemanticResolution's fields
    represents (single-field ambiguity/candidates, and the specialized
    survival-analysis / correlation collections that never fit a single
    binding).

    Frozen (v20-C1 hardening item B): this object is meant to be handed,
    as a single authoritative snapshot, across hypothesis/experiment/
    controller boundaries starting with C2. Making it immutable after
    construction means no downstream consumer can quietly rebind e.g.
    `.outcome` mid-investigation and have that drift silently from the
    persisted contract it was built from.
    """
    bindings: SemanticBindingSet
    outcome: ResolvedSemanticField
    dimension: ResolvedSemanticField
    time: ResolvedSemanticField
    secondary_metric: ResolvedSemanticField
    exposure: ResolvedSemanticField
    censored: ResolvedSemanticField
    confounders: SemanticCandidateSet
    # The raw dataset-wide categorical-column pool this resolution drew
    # from -- distinct from `dimension.candidates`, which is only
    # populated when the dimension itself was ambiguous/unresolved. Kept
    # here because several consumers (e.g. alternative-hypothesis
    # synthesis after a refutation) need "every OTHER categorical column",
    # not just the ones considered for the primary dimension decision.
    available_categorical_candidates: List[str] = field(default_factory=list)
    # v20-C4.2.1: the explanatory variable(s) the QUESTION itself asked
    # about (see QuestionRoleProposal) -- deliberately separate from
    # `secondary_metric`, which is whatever SemanticEngine discovered.
    # None means canonical was built without question roles (legacy
    # callers); an empty set means the question named no explanatory
    # variable. Both read back as [] via requested_explanatory_columns().
    requested_explanatory: Optional[SemanticCandidateSet] = None

    def __post_init__(self) -> None:
        """Defensive structural validation (v20-C1 hardening item B): the
        per-field invariants are already enforced by ResolvedSemanticField/
        SemanticCandidateSet themselves at their own construction time, so
        this only guards against the wrong *kind* of object being passed
        for a given attribute -- e.g. a plain string handed to `outcome`
        instead of a ResolvedSemanticField -- which the dataclass's type
        hints alone do not catch at runtime."""
        single_value_fields = (
            "outcome", "dimension", "time",
            "secondary_metric", "exposure", "censored",
        )
        for name in single_value_fields:
            val = getattr(self, name)
            if not isinstance(val, ResolvedSemanticField):
                raise TypeError(
                    f"CanonicalSemanticResolution.{name} must be a "
                    f"ResolvedSemanticField, got {type(val).__name__}"
                )
        if not isinstance(self.bindings, SemanticBindingSet):
            raise TypeError(
                "CanonicalSemanticResolution.bindings must be a "
                f"SemanticBindingSet, got {type(self.bindings).__name__}"
            )
        if not isinstance(self.confounders, SemanticCandidateSet):
            raise TypeError(
                "CanonicalSemanticResolution.confounders must be a "
                f"SemanticCandidateSet, got {type(self.confounders).__name__}"
            )
        if self.requested_explanatory is not None and not isinstance(self.requested_explanatory, SemanticCandidateSet):
            raise TypeError(
                "CanonicalSemanticResolution.requested_explanatory must be a "
                f"SemanticCandidateSet or None, got {type(self.requested_explanatory).__name__}"
            )
        # v20-C2.1 hardening: nested dataclasses (ResolvedSemanticField,
        # SemanticCandidateSet, SemanticBindingSet) now freeze their own
        # collections in their own __post_init__, but this container's own
        # directly-held list still needed the same treatment -- it is not
        # covered by any of the nested types' construction.
        object.__setattr__(
            self, "available_categorical_candidates", tuple(self.available_categorical_candidates)
        )

    # -- Convenience accessors, per v20-C1's proposed consumer-facing API --

    def outcome_column(self) -> Optional[str]:
        return self.outcome.value if self.outcome.is_resolved() else None

    def primary_dimension(self) -> Optional[str]:
        return self.dimension.value if self.dimension.is_resolved() else None

    def time_variable(self) -> Optional[str]:
        return self.time.value if self.time.is_resolved() else None

    def explanatory_variables(self) -> List[str]:
        return self.bindings.projected_explanatory_columns()

    def candidate_dimensions(self) -> List[str]:
        """The candidates behind the dimension decision itself (populated
        when AMBIGUOUS), NOT the full dataset-wide pool -- see
        available_categorical_candidates for that."""
        return list(self.dimension.candidates)

    def churn_event_resolution(self) -> ResolvedSemanticField:
        return self.outcome

    def confounder_set(self) -> SemanticCandidateSet:
        return self.confounders

    def secondary_metric_column(self) -> Optional[str]:
        """The second numeric variable tested for association (v20-C2
        section 7) -- an EXPLANATORY_VARIABLE-role field, deliberately
        distinct from outcome()/primary_dimension(): a correlation
        investigation's secondary metric is never the same analytical
        concept as its outcome, so this must not be collapsed into
        outcome_column()."""
        return self.secondary_metric.value if self.secondary_metric.is_resolved() else None

    def requested_explanatory_columns(self) -> List[str]:
        """Columns the question itself named as explanatory (REQUESTED, not
        discovered). Empty when none, or when canonical was built without
        question roles."""
        return list(self.requested_explanatory.columns) if self.requested_explanatory is not None else []

    def deterministic_requested_explanatory(self) -> List[str]:
        """Return requested explanatory columns in deterministic (sorted, deduplicated) order."""
        return sorted(dict.fromkeys(self.requested_explanatory_columns()))

    def requested_categorical_columns(self) -> List[str]:
        """The subset of requested_explanatory_columns() that are
        categorical, i.e. columns a Pearson/Spearman correlation is
        undefined for. A column counts as categorical when it is the
        resolved primary dimension or is in the dataset's categorical pool."""
        pool = set(self.available_categorical_candidates)
        dim = self.primary_dimension()
        return [c for c in self.requested_explanatory_columns() if c in pool or c == dim]

    def exposure_column(self) -> Optional[str]:
        return self.exposure.value if self.exposure.is_resolved() else None

    def censored_column(self) -> Optional[str]:
        return self.censored.value if self.censored.is_resolved() else None

    def is_churn_relevant(self) -> bool:
        """True iff `outcome` was populated from the churn-event
        resolution path (semantic.churn_event_col plus its ambiguity/
        availability signals -- see semantic_resolution_builder.
        _resolve_outcome_field), rather than from an ordinary
        semantic.target_metric_col.

        `outcome` is deliberately overloaded: the same field carries
        either a resolved/ambiguous/unresolved churn event OR an ordinary
        resolved target metric, depending on which branch built it. That
        means `outcome.is_resolved()` cannot be used to tell "this is a
        churn question" from "this is any other resolved-target question"
        -- both look identical (RESOLVED, non-empty value, single-entry
        candidates) once resolved, and an *unresolved* churn outcome would
        otherwise be indistinguishable from a field that was simply never
        populated. This checks the field's provenance instead, which
        records which source branch actually produced it, and so stays
        correct across all three churn states (RESOLVED / AMBIGUOUS /
        UNRESOLVED) as well as the non-churn case.
        """
        return self.outcome.provenance.get("source") == "semantic.churn_event_col"
