"""Typed semantic-role binding structures for AA-OS analytical contracts
(v20-A section 3/4).

A SemanticBinding is the formal statement "this column, in this table,
plays this analytical role, at this confidence, with this resolution
status" -- replacing implicit, per-engine reads of raw ``semantic.*``
fields with one canonical, validated representation that the analytical
contract carries and every downstream consumer (experiment synthesis,
execution, verification, verdict, provenance) can query the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Optional

from packages.schemas.src.semantic_role import (
    RoleConflictType,
    SemanticRole,
    ResolutionStatus,
    UNCONDITIONALLY_EXECUTABLE_STATUSES,
    NEVER_EXECUTABLE_STATUSES,
    INFERRED_EXECUTION_CONFIDENCE_FLOOR,
    classify_role_pair,
)


@dataclass(frozen=True)
class SemanticBinding:
    """One (column, role) analytical binding.

    Minimum required fields: column, table, role, resolution_status,
    confidence. Everything else is populated when known.
    """
    column: str
    table: str
    role: SemanticRole
    resolution_status: ResolutionStatus
    confidence: float = 1.0
    semantic_type: Optional[str] = None
    physical_type: Optional[str] = None
    grain: Optional[str] = None
    unit: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # v20-C2.1 hardening: frozen=True only blocks reassigning
        # `self.provenance = ...`; the dict it points to was still mutable
        # in place (`binding.provenance["source"] = "spoofed"`). Wrap it in
        # a read-only view so the whole binding is actually immutable, not
        # just its top-level attributes.
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def is_executable(self) -> bool:
        """Whether an executable experiment may reference this binding."""
        if self.resolution_status in NEVER_EXECUTABLE_STATUSES:
            return False
        if self.resolution_status in UNCONDITIONALLY_EXECUTABLE_STATUSES:
            return True
        if self.resolution_status == ResolutionStatus.INFERRED:
            return self.confidence >= INFERRED_EXECUTION_CONFIDENCE_FLOOR
        return False

    def to_dict(self) -> Dict[str, Any]:
        # Built explicitly rather than via dataclasses.asdict(): asdict()
        # does not know MappingProxyType is meant to become a plain dict
        # again for JSON persistence, and would otherwise leak a
        # non-JSON-serializable value into semantic_bindings_json.
        return {
            "column": self.column,
            "table": self.table,
            "role": self.role.value if isinstance(self.role, SemanticRole) else self.role,
            "resolution_status": (
                self.resolution_status.value
                if isinstance(self.resolution_status, ResolutionStatus)
                else self.resolution_status
            ),
            "confidence": self.confidence,
            "semantic_type": self.semantic_type,
            "physical_type": self.physical_type,
            "grain": self.grain,
            "unit": self.unit,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SemanticBinding":
        role = d.get("role")
        status = d.get("resolution_status")
        return cls(
            column=d["column"],
            table=d.get("table", ""),
            role=SemanticRole(role) if not isinstance(role, SemanticRole) else role,
            resolution_status=(
                ResolutionStatus(status) if not isinstance(status, ResolutionStatus) else status
            ),
            confidence=float(d.get("confidence", 1.0)),
            semantic_type=d.get("semantic_type"),
            physical_type=d.get("physical_type"),
            grain=d.get("grain"),
            unit=d.get("unit"),
            provenance=dict(d.get("provenance") or {}),
        )


@dataclass(frozen=True)
class RoleConflict:
    column: str
    table: str
    roles: List[SemanticRole]
    conflict_type: RoleConflictType
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "table": self.table,
            "roles": [r.value for r in self.roles],
            "conflict_type": self.conflict_type.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SemanticBindingSet:
    """Canonical, queryable collection of semantic bindings for one
    analytical contract version.

    Frozen (v20-C2.1 hardening): ``bindings`` is stored as a tuple, not a
    list, so a downstream consumer cannot ``.bindings.append(...)`` a
    binding into what is meant to be an immutable snapshot handed across
    hypothesis/experiment/controller boundaries. Before this, the class
    itself wasn't even declared frozen -- both the container AND its
    contents were mutable in place.
    """
    bindings: List[SemanticBinding] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", tuple(self.bindings))

    def get_by_role(self, role: SemanticRole) -> List[SemanticBinding]:
        return [b for b in self.bindings if b.role == role]

    def get_by_column(self, column: str, table: Optional[str] = None) -> List[SemanticBinding]:
        return [
            b for b in self.bindings
            if b.column == column and (table is None or b.table == table)
        ]

    def executable_bindings(self) -> List[SemanticBinding]:
        return [b for b in self.bindings if b.is_executable()]

    def validate(self) -> List[RoleConflict]:
        """Detect incompatible role assignments on the same (table, column).

        Two or more DIFFERENT roles on the same column are either an
        explicitly allowed LEGITIMATE_MULTIROLE combination (see
        packages.schemas.src.semantic_role) or a CONFLICTING_ROLE that must
        be surfaced -- never silently permitted.
        """
        conflicts: List[RoleConflict] = []
        by_col: Dict[tuple, List[SemanticBinding]] = {}
        for b in self.bindings:
            by_col.setdefault((b.table, b.column), []).append(b)

        for (table, column), group in by_col.items():
            roles = sorted({b.role for b in group}, key=lambda r: r.value)
            if len(roles) < 2:
                continue
            # Pairwise-classify; if ANY pair is a genuine conflict, the whole
            # multi-role assignment for this column is flagged.
            conflicting_pairs = []
            for i in range(len(roles)):
                for j in range(i + 1, len(roles)):
                    kind = classify_role_pair(roles[i], roles[j])
                    if kind == RoleConflictType.CONFLICTING_ROLE:
                        conflicting_pairs.append((roles[i], roles[j]))
            if conflicting_pairs:
                pair_desc = ", ".join(f"{a.value}+{b.value}" for a, b in conflicting_pairs)
                conflicts.append(RoleConflict(
                    column=column,
                    table=table,
                    roles=list(roles),
                    conflict_type=RoleConflictType.CONFLICTING_ROLE,
                    reason=f"Column '{column}' in '{table}' assigned incompatible roles: {pair_desc}",
                ))
        return conflicts

    def has_conflicts(self) -> bool:
        return len(self.validate()) > 0

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [b.to_dict() for b in self.bindings]

    @classmethod
    def from_dict_list(cls, items: List[Dict[str, Any]]) -> "SemanticBindingSet":
        return cls(bindings=[SemanticBinding.from_dict(d) for d in (items or [])])

    # -- Backward-compatible projections (v20-A section 5) ------------------
    # The canonical string fields (target_column, explanatory_columns,
    # time_column, group_dimension) become compatibility projections
    # DERIVED from the binding set rather than a second, independently
    # maintained truth.

    def projected_target_column(self) -> Optional[str]:
        """v20-B1: fail-closed. 0 executable candidates for a role -> try
        the next role, then None; exactly 1 -> that column; >1 -> None
        (ambiguous), never a first-wins guess. Previously this silently
        returned ``candidates[0].column`` on ties, reintroducing the same
        first-wins semantic assumption section 5 was meant to retire.
        """
        for role in (SemanticRole.TARGET, SemanticRole.OUTCOME):
            candidates = [b for b in self.get_by_role(role) if b.is_executable()]
            if len(candidates) == 1:
                return candidates[0].column
            if len(candidates) > 1:
                return None
        return None

    def projected_explanatory_columns(self) -> List[str]:
        return [b.column for b in self.get_by_role(SemanticRole.EXPLANATORY_VARIABLE)]

    def projected_time_column(self) -> Optional[str]:
        """Fail-closed: see projected_target_column."""
        candidates = self.get_by_role(SemanticRole.TIME_VARIABLE)
        return candidates[0].column if len(candidates) == 1 else None

    def projected_group_dimension(self) -> Optional[str]:
        """Fail-closed: see projected_target_column."""
        candidates = self.get_by_role(SemanticRole.GROUPING_DIMENSION)
        return candidates[0].column if len(candidates) == 1 else None
