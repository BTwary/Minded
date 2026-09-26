"""Immutable dataset transformation lineage records."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any
import hashlib, json

from .content_identity import compute_content_hash


@dataclass(frozen=True)
class TransformationRecord:
    transformation_id: str
    operation: str
    input_content_hash: str
    output_content_hash: str
    parameters: dict[str, Any]
    affected_rows: int
    affected_columns: list[str]
    reason: str
    created_at: str
    reversible: bool = False

    @property
    def record_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def make_transformation_record(*, operation: str, input_df, output_df, parameters=None, affected_rows: int | None = None, affected_columns=None, reason: str = "") -> TransformationRecord:
    return TransformationRecord(
        transformation_id=f"TX-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
        operation=operation,
        input_content_hash=compute_content_hash(input_df),
        output_content_hash=compute_content_hash(output_df),
        parameters=parameters or {},
        affected_rows=int(affected_rows if affected_rows is not None else abs(len(output_df)-len(input_df))),
        affected_columns=list(affected_columns or []),
        reason=reason,
        created_at=datetime.now(timezone.utc).isoformat(),
        reversible=False,
    )
