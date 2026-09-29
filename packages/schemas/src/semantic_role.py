"""Canonical semantic-role vocabulary for AA-OS analytical fields (v20-A).

This is the single shared enum set for what an analytical column *means* to
an executable experiment, independent of its physical type. Every other
module (contract, binding builder, experiment validator, controller gate)
imports from here rather than declaring its own role vocabulary -- see
MindEd_AAOS_v20-A section 2 ("Do not create duplicate role enums in
multiple modules").
"""
from __future__ import annotations

from enum import Enum


class SemanticRole(str, Enum):
    TARGET = "TARGET"
    OUTCOME = "OUTCOME"
    EXPLANATORY_VARIABLE = "EXPLANATORY_VARIABLE"
    GROUPING_DIMENSION = "GROUPING_DIMENSION"
    TIME_VARIABLE = "TIME_VARIABLE"
    ENTITY_KEY = "ENTITY_KEY"
    WEIGHT = "WEIGHT"
    CENSORED_TIME = "CENSORED_TIME"
    EXPOSURE = "EXPOSURE"
    TREATMENT = "TREATMENT"
    UNKNOWN = "UNKNOWN"


class ResolutionStatus(str, Enum):
    """How confidently a binding's (column, role) pair was determined.

    Execution eligibility (section 7):
      RESOLVED, USER_SPECIFIED       -> may execute
      AMBIGUOUS, UNRESOLVED          -> must not execute
      INFERRED                       -> may execute only when an explicit
                                         safety policy allows it and the
                                         binding's confidence threshold is
                                         satisfied (see EXECUTABLE_STATUSES /
                                         INFERRED_EXECUTION_CONFIDENCE_FLOOR
                                         below).
    """
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"
    INFERRED = "INFERRED"
    USER_SPECIFIED = "USER_SPECIFIED"


# Statuses that may execute unconditionally.
UNCONDITIONALLY_EXECUTABLE_STATUSES = frozenset({
    ResolutionStatus.RESOLVED,
    ResolutionStatus.USER_SPECIFIED,
})

# Statuses that must never execute, regardless of confidence.
NEVER_EXECUTABLE_STATUSES = frozenset({
    ResolutionStatus.AMBIGUOUS,
    ResolutionStatus.UNRESOLVED,
})

# INFERRED bindings require at least this confidence to be considered
# executable. This is a conservative default; callers may only tighten it,
# never loosen it, without an explicit contract-level override.
INFERRED_EXECUTION_CONFIDENCE_FLOOR = 0.75


class RoleConflictType(str, Enum):
    """Classification of a multi-role assignment on the same column."""
    LEGITIMATE_MULTIROLE = "LEGITIMATE_MULTIROLE"
    CONFLICTING_ROLE = "CONFLICTING_ROLE"


# Multi-role columns are the COMMON case in real analytical contracts (a
# comparison's grouping dimension IS its explanatory variable; a survival
# model's exposure/duration column is routinely also an explanatory
# confounder; a treatment column is routinely also an explanatory variable).
# So, per v20-A section 6 ("do not prohibit legitimate multi-role semantics
# blindly"), the default is LEGITIMATE_MULTIROLE and only the SPECIFIC,
# named-in-spec combinations below -- where the same column would have to
# simultaneously be the thing being explained/predicted and something
# explaining/predicting it, or a grouping identity mixed with a row-identity
# key -- are treated as a genuine conflict.
_DEFAULT_CONFLICTING_ROLE_PAIRS = frozenset({
    frozenset({SemanticRole.OUTCOME, SemanticRole.EXPLANATORY_VARIABLE}),
    frozenset({SemanticRole.OUTCOME, SemanticRole.TIME_VARIABLE}),
    frozenset({SemanticRole.TARGET, SemanticRole.EXPLANATORY_VARIABLE}),
    frozenset({SemanticRole.TARGET, SemanticRole.TIME_VARIABLE}),
    frozenset({SemanticRole.GROUPING_DIMENSION, SemanticRole.ENTITY_KEY}),
})


def classify_role_pair(role_a: SemanticRole, role_b: SemanticRole) -> RoleConflictType:
    """Classify whether two roles assigned to the SAME column are a
    legitimate multi-role semantic (e.g. a grouping dimension that is also
    the explanatory variable of a comparison) or a genuine conflict (e.g. a
    column being both the OUTCOME and the thing explaining the OUTCOME).
    """
    if role_a == role_b:
        return RoleConflictType.LEGITIMATE_MULTIROLE
    pair = frozenset({role_a, role_b})
    if pair in _DEFAULT_CONFLICTING_ROLE_PAIRS:
        return RoleConflictType.CONFLICTING_ROLE
    return RoleConflictType.LEGITIMATE_MULTIROLE


class ExperimentRole(str, Enum):
    PRIMARY = "PRIMARY"
    SUPPORTING = "SUPPORTING"
    ADVERSARIAL = "ADVERSARIAL"
    VERIFICATION = "VERIFICATION"
    # Explicit non-role for legacy/newly-constructed candidates that have not
    # yet been classified. Distinct from PRIMARY -- an UNASSIGNED experiment
    # is not automatically treated as the contract's main experiment.
    UNASSIGNED = "UNASSIGNED"
