"""Canonical, content-addressed dataset identity for AA-OS.

The identity is derived from the complete logical dataset contents, not IDs,
row counts, or version numbers.  Rows are canonicalized as an unordered multiset
so valid row-order permutations do not change the content hash.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "aaos-dataset-content-v3"


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return {"type": "int", "value": int(value)}
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isnan(v):
            return {"type": "float", "value": "NaN"}
        if math.isinf(v):
            return {"type": "float", "value": "Infinity" if v > 0 else "-Infinity"}
        return {"type": "float", "value": format(v, ".17g")}
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return {"type": "datetime", "value": pd.Timestamp(value).isoformat()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "value": bytes(value).hex()}
    if pd.isna(value):
        return {"type": "missing"}
    return {"type": type(value).__name__, "value": str(value)}


def canonical_dataset_payload(
    df: pd.DataFrame,
    *,
    source_metadata: dict[str, Any] | None = None,
    transformation_lineage: list[dict[str, Any]] | None = None,
) -> bytes:
    columns = sorted(str(c) for c in df.columns)
    # Reindex columns by canonical names; duplicate column names are represented
    # positionally because pandas cannot otherwise address them unambiguously.
    canonical_columns = []
    for idx, original in enumerate(df.columns):
        canonical_columns.append((str(original), idx, str(df.dtypes.iloc[idx])))
    canonical_columns.sort(key=lambda x: (x[0], x[1]))

    row_tokens = []
    for row in df.itertuples(index=False, name=None):
        cells = [_scalar(row[idx]) for _, idx, _ in canonical_columns]
        row_tokens.append(json.dumps(cells, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    row_tokens.sort()

    payload = {
        "protocol": PROTOCOL_VERSION,
        "schema": [(name, dtype) for name, _, dtype in canonical_columns],
        "row_count": len(df),
        "rows": row_tokens,
        # Context is explicit and canonicalized so persisted dataset identity can
        # be bound to its immutable source and transformation history when the
        # caller has that provenance available. Pure in-memory callers may omit it.
        "source_metadata": source_metadata or {},
        "transformation_lineage": transformation_lineage or [],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_content_hash(
    df: pd.DataFrame,
    *,
    source_metadata: dict[str, Any] | None = None,
    transformation_lineage: list[dict[str, Any]] | None = None,
) -> str:
    """Return the canonical dataset identity hash.

    The logical table content is always included.  Persistence layers may also
    bind stable source metadata and transformation lineage to make provenance
    part of the identity.
    """
    return hashlib.sha256(
        canonical_dataset_payload(
            df, source_metadata=source_metadata, transformation_lineage=transformation_lineage
        )
    ).hexdigest()
