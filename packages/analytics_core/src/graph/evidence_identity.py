"""Canonical, persisted scientific evidence identity for AA-OS.

Evidence identity represents *what computation was performed against which
snapshot and scope*, not the experiment UUID that happened to execute it.
This makes retries, replays and independently planned duplicate computations
idempotent for Bayesian evidence accumulation while preserving distinct
engine identities for independent verification.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Tuple


def _normalize_sql(sql: Optional[str]) -> str:
    if not sql:
        return ""
    s = re.sub(r"\s+", " ", str(sql).strip().lower())
    return s.rstrip(";").strip()




def _semantic_sql_normalize(sql: Optional[str]) -> str:
    """Normalize only semantics-preserving SQL spelling differences.

    This is deliberately conservative: whitespace/casing, redundant WHERE TRUE/1=1,
    and redundant parentheses around a complete scalar expression are normalized.
    Type-changing casts are not removed unless the caller provides a structured
    analytical scope, where SQL text is intentionally non-authoritative.
    """
    s = _normalize_sql(sql)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+where\s+(?:true|1\s*=\s*1)\s*$", "", s, flags=re.I)
    s = re.sub(r"\bselect\s*\(\s*(sum\s*\([^)]*\))\s*\)\s+from\b", r"select \1 from", s, flags=re.I)
    return s
def _canonical(value: Any, path: str = "$") -> Any:
    """Convert only explicitly supported values to canonical JSON primitives.

    Unknown objects are rejected instead of stringified. Scientific identities must
    never depend on repr()/memory addresses or incidental formatting.
    """
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"non-finite float at {path}")
        return value
    if isinstance(value, Enum):
        return _canonical(value.value, path)
    if isinstance(value, Decimal):
        return {"__type__": "decimal", "value": format(value, "f")}
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise TypeError(f"naive datetime at {path}")
        return {"__type__": "datetime", "value": value.astimezone(timezone.utc).isoformat()}
    if isinstance(value, date) and not isinstance(value, datetime):
        return {"__type__": "date", "value": value.isoformat()}
    if isinstance(value, time):
        if value.tzinfo is None:
            return {"__type__": "time", "value": value.isoformat()}
        return {"__type__": "time", "value": value.isoformat()}
    if isinstance(value, dict):
        out = {}
        for key in sorted(value, key=lambda x: str(x)):
            if not isinstance(key, str):
                raise TypeError(f"non-string key at {path}: {key!r}")
            out[key] = _canonical(value[key], f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item, f"{path}[]") for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    raise TypeError(f"unsupported canonical type {type(value).__name__} at {path}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def hash_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_dataset_identity(dataset_fingerprints: Optional[Dict[str, str]]) -> str:
    """Hash the complete set of source snapshot fingerprints, by dataset name."""
    return hash_canonical(dataset_fingerprints or {})


@dataclass(frozen=True)
class EvidenceIdentityComponents:
    dataset_identity_hash: str
    query_plan_hash: str
    execution_engine: str
    semantic_scope: str

    def as_key(self) -> str:
        return "|".join([
            self.dataset_identity_hash,
            self.query_plan_hash,
            self.execution_engine,
            self.semantic_scope,
        ])


def compute_evidence_identity_components(
    dataset_identity_hash: str,
    query_or_plan: str,
    execution_engine: str,
    semantic_scope: Optional[str] = None,
) -> EvidenceIdentityComponents:
    return EvidenceIdentityComponents(
        dataset_identity_hash=str(dataset_identity_hash).strip().lower(),
        query_plan_hash=hashlib.sha256(_normalize_sql(query_or_plan).encode("utf-8")).hexdigest(),
        execution_engine=str(execution_engine or "").strip().lower(),
        semantic_scope=str(semantic_scope or "").strip().lower(),
    )


def compute_evidence_identity(
    dataset_identity_hash: str,
    query_or_plan: str,
    execution_engine: str,
    semantic_scope: Optional[str] = None,
    *,
    estimator: Optional[str] = None,
    formula: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    output_columns: Optional[list[str]] = None,
    input_row_count: Optional[int] = None,
    output_row_count: Optional[int] = None,
) -> str:
    """Return the authoritative computation identity.

    Result values are deliberately excluded: a changed result for an identical
    computation is an integrity/determinism problem, not new evidence.
    """
    structured_scope = semantic_scope if isinstance(semantic_scope, (dict, list, tuple, set)) else None
    query_plan = {
        # Structured scope is authoritative only when it fully represents the
        # population/estimand-defining computation.  Persist the normalized SQL
        # alongside it so a missing population predicate cannot collapse two
        # materially different computations into one evidence identity.
        "query_semantic_plan": _canonical(structured_scope) if structured_scope is not None else _semantic_sql_normalize(query_or_plan),
        "normalized_query": _semantic_sql_normalize(query_or_plan),
        "estimator": str(estimator or "").strip().lower(),
        "formula": str(formula or "").strip(),
        "parameters": _canonical(parameters or {}),
        "output_columns": sorted(str(c) for c in (output_columns or [])),
        "input_row_count": input_row_count,
        "output_row_count": output_row_count,
    }
    semantic = canonical_json(semantic_scope) if isinstance(semantic_scope, (dict, list, tuple, set)) else str(semantic_scope or "").strip().lower()
    components = compute_evidence_identity_components(
        dataset_identity_hash,
        canonical_json(query_plan),
        execution_engine,
        semantic,
    )
    return hashlib.sha256(components.as_key().encode("utf-8")).hexdigest()


@dataclass
class EvidenceRegistry:
    """Investigation-scoped in-memory compatibility registry."""
    _seen: Dict[str, str] = field(default_factory=dict)

    def register(self, identity: str, evidence_id: str) -> Tuple[bool, str]:
        existing = self._seen.get(identity)
        if existing is not None:
            return True, existing
        self._seen[identity] = evidence_id
        return False, evidence_id

    def is_known(self, identity: str) -> bool:
        return identity in self._seen
