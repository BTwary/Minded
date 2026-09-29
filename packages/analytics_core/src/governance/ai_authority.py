"""AI proposal firewall: AI may interpret/propose, never bypass the compiler."""
from __future__ import annotations
from typing import Any, Mapping, Sequence


def validate_ai_proposal(proposal: Mapping[str, Any], available_columns: Sequence[str]) -> dict[str, Any]:
    allowed = set(str(c) for c in available_columns)
    referenced = []
    for key in ("target", "target_metric", "group", "group_dimension", "time", "time_column", "secondary_metric"):
        value = proposal.get(key)
        if isinstance(value, str) and value:
            referenced.append((key, value))
    unknown = [f"{k}:{v}" for k, v in referenced if v not in allowed]
    if unknown:
        return {"allowed": False, "reason": "AI_REFERENCED_UNKNOWN_SCHEMA_COLUMN", "unknown": unknown}
    if proposal.get("sql") or proposal.get("query"):
        return {"allowed": False, "reason": "AI_CANNOT_EXECUTE_ARBITRARY_SQL"}
    return {"allowed": True, "reason": "AI_PROPOSAL_ACCEPTED_FOR_DETERMINISTIC_COMPILATION"}
