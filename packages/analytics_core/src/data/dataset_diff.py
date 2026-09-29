"""Deterministic dataset-to-dataset forensic diffing for offline AA-OS analysis.

This module deliberately stays independent of persistence and UI.  It compares two
materialized tables and reports schema, row, quality, and distribution changes without
network/AI dependencies.  When stable key columns are supplied it can identify added,
removed, and changed records.  Without keys it falls back to canonical row-multiset
comparison, which is explicit about what can and cannot be inferred.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd


@dataclass
class DatasetDiff:
    schema_added: List[str] = field(default_factory=list)
    schema_removed: List[str] = field(default_factory=list)
    schema_type_changed: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    row_count_before: int = 0
    row_count_after: int = 0
    rows_added: int = 0
    rows_removed: int = 0
    rows_changed: int = 0
    row_identity_mode: str = "ROW_MULTISET"
    key_columns: List[str] = field(default_factory=list)
    duplicate_key_counts: Dict[str, int] = field(default_factory=dict)
    missingness_before: Dict[str, float] = field(default_factory=dict)
    missingness_after: Dict[str, float] = field(default_factory=dict)
    missingness_delta: Dict[str, float] = field(default_factory=dict)
    numeric_distribution_delta: Dict[str, Dict[str, float]] = field(default_factory=dict)
    categorical_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _canonical_scalar(value: Any) -> str:
    if value is None:
        return "<NULL>"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isnan(v):
            return "<NAN>"
        if math.isinf(v):
            return "<INF>" if v > 0 else "<-INF>"
        return f"<FLOAT>{v:.17g}"
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return f"<DATE>{pd.Timestamp(value).isoformat()}"
    if pd.isna(value):
        return "<MISSING>"
    return f"<{type(value).__name__}>{value}"


def _row_token(row: Sequence[Any]) -> str:
    return "\x1f".join(_canonical_scalar(v) for v in row)


def _row_multiset(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in df[list(columns)].itertuples(index=False, name=None):
        token = _row_token(row)
        counts[token] = counts.get(token, 0) + 1
    return counts


def _key_token(row: Sequence[Any]) -> str:
    return _row_token(row)


def _key_index(df: pd.DataFrame, key_columns: Sequence[str]) -> Dict[str, List[int]]:
    """Index rows by canonical key values using positional row numbers only.

    The DataFrame index is not an identity field and may contain arbitrary labels
    (including duplicates).  ``reset_index(drop=True)`` gives us stable positional
    locators for fetching rows later without accidentally treating index labels as
    business-key identity.
    """
    result: Dict[str, List[int]] = {}
    for pos, row in enumerate(df[list(key_columns)].itertuples(index=False, name=None)):
        token = _key_token(row)
        result.setdefault(token, []).append(pos)
    return result


def _key_counts(df: pd.DataFrame, key_columns: Sequence[str]) -> Dict[str, int]:
    """Return counts per canonical key token without using row/index positions."""
    counts: Dict[str, int] = {}
    for row in df[list(key_columns)].itertuples(index=False, name=None):
        token = _key_token(row)
        counts[token] = counts.get(token, 0) + 1
    return counts


def _key_has_missing_value(row: Sequence[Any]) -> bool:
    for value in row:
        if value is None:
            return True
        if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
            return True
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                return True
        except (TypeError, ValueError):
            pass
    return False


def _missingness(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {str(c): 0.0 for c in df.columns}
    return {str(c): float(df[c].isna().mean()) for c in df.columns}


def _distribution_changes(before: pd.DataFrame, after: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    changes: Dict[str, Dict[str, float]] = {}
    common = [c for c in before.columns if c in after.columns]
    for c in common:
        if not (pd.api.types.is_numeric_dtype(before[c]) and pd.api.types.is_numeric_dtype(after[c])):
            continue
        b = pd.to_numeric(before[c], errors="coerce").dropna()
        a = pd.to_numeric(after[c], errors="coerce").dropna()
        if b.empty or a.empty:
            continue
        bmean, amean = float(b.mean()), float(a.mean())
        bmed, amed = float(b.median()), float(a.median())
        bstd, astd = float(b.std(ddof=0)), float(a.std(ddof=0))
        scale = max(abs(bmean), abs(bmed), bstd, 1e-12)
        changes[str(c)] = {
            "mean_before": bmean,
            "mean_after": amean,
            "mean_relative_change": (amean - bmean) / scale,
            "median_before": bmed,
            "median_after": amed,
            "std_before": bstd,
            "std_after": astd,
        }
    return changes


def _categorical_changes(before: pd.DataFrame, after: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    changes: Dict[str, Dict[str, Any]] = {}
    for c in [c for c in before.columns if c in after.columns]:
        if not (pd.api.types.is_object_dtype(before[c]) or isinstance(before[c].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(before[c])):
            continue
        bvals = {str(v) for v in before[c].dropna().unique()}
        avals = {str(v) for v in after[c].dropna().unique()}
        added, removed = sorted(avals - bvals), sorted(bvals - avals)
        if added or removed:
            changes[str(c)] = {"added_categories": added, "removed_categories": removed}
    return changes


def compare_datasets(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    key_columns: Optional[Sequence[str]] = None,
) -> DatasetDiff:
    """Compare two datasets without making semantic or causal claims.

    ``key_columns`` should identify a stable business/entity row.  If omitted,
    row-multiset comparison is used.  This intentionally avoids pretending that
    positional row numbers identify records.
    """
    before = before.copy()
    after = after.copy()
    diff = DatasetDiff(
        row_count_before=len(before),
        row_count_after=len(after),
        schema_added=sorted(set(map(str, after.columns)) - set(map(str, before.columns))),
        schema_removed=sorted(set(map(str, before.columns)) - set(map(str, after.columns))),
    )

    common = sorted(set(map(str, before.columns)) & set(map(str, after.columns)))
    # Resolve columns by their string names.  Duplicate names are intentionally
    # rejected because key/schema semantics would otherwise be ambiguous.
    if len(set(map(str, before.columns))) != len(before.columns) or len(set(map(str, after.columns))) != len(after.columns):
        diff.notes.append("Duplicate column names prevent an unambiguous dataset diff.")
        return diff

    for c in common:
        bt, at = str(before[c].dtype), str(after[c].dtype)
        if bt != at:
            diff.schema_type_changed[c] = (bt, at)

    missing_before = _missingness(before)
    missing_after = _missingness(after)
    diff.missingness_before = missing_before
    diff.missingness_after = missing_after
    diff.missingness_delta = {
        c: missing_after.get(c, 0.0) - missing_before.get(c, 0.0)
        for c in sorted(set(missing_before) | set(missing_after))
        if abs(missing_after.get(c, 0.0) - missing_before.get(c, 0.0)) > 0.0
    }
    diff.numeric_distribution_delta = _distribution_changes(before, after)
    diff.categorical_changes = _categorical_changes(before, after)

    requested_keys = [str(c) for c in (key_columns or [])]
    if requested_keys:
        missing_keys = [c for c in requested_keys if c not in before.columns or c not in after.columns]
        if missing_keys:
            diff.notes.append(f"Requested key columns missing from one dataset: {missing_keys}.")
        else:
            diff.row_identity_mode = "STABLE_KEY"
            diff.key_columns = requested_keys
            # A missing key value is not a stable identity.  Keep those rows out of
            # record-level matching instead of collapsing every NULL/NaN into one
            # artificial business key.  We retain an explicit audit note so the caller
            # knows the row-level diff is partial for those records.
            key_rows_before = before[requested_keys].itertuples(index=False, name=None)
            key_rows_after = after[requested_keys].itertuples(index=False, name=None)
            invalid_before = sum(1 for row in key_rows_before if _key_has_missing_value(row))
            invalid_after = sum(1 for row in key_rows_after if _key_has_missing_value(row))
            if invalid_before or invalid_after:
                diff.notes.append(
                    f"{invalid_before} before-row(s) and {invalid_after} after-row(s) have incomplete stable keys; "
                    "those rows are excluded from record-level identity matching."
                )

            valid_before = before.loc[
                ~before[requested_keys].isna().any(axis=1)
            ].reset_index(drop=True)
            valid_after = after.loc[
                ~after[requested_keys].isna().any(axis=1)
            ].reset_index(drop=True)
            bi = _key_index(valid_before, requested_keys)
            ai = _key_index(valid_after, requested_keys)
            bc = _key_counts(valid_before, requested_keys)
            ac = _key_counts(valid_after, requested_keys)

            # Duplicate identity is determined solely by key-value multiplicity.  Do
            # not use row/index positions, because a pure row reorder must never become
            # a duplicate-key finding.  A duplicated key is not safely record-identifiable,
            # so rows under that key are excluded from changed-row classification.
            duplicate_keys = {
                k: max(bc.get(k, 0), ac.get(k, 0))
                for k in set(bc) | set(ac)
                if bc.get(k, 0) > 1 or ac.get(k, 0) > 1
            }
            if duplicate_keys:
                diff.duplicate_key_counts = duplicate_keys
                diff.notes.append(
                    "Duplicate stable keys detected; those keys are not record-identifiable and are excluded "
                    "from changed-row classification. Added/removed counts reflect key multiplicity differences."
                )

            duplicate_set = set(duplicate_keys)
            added_keys = set(ai) - set(bi)
            removed_keys = set(bi) - set(ai)
            diff.rows_added = sum(ac[k] for k in added_keys)
            diff.rows_removed = sum(bc[k] for k in removed_keys)

            compare_cols = [c for c in common if c not in requested_keys]
            changed = 0
            for key in set(ai) & set(bi):
                before_count = bc[key]
                after_count = ac[key]
                delta = after_count - before_count

                # If identity is duplicated, the key cannot safely identify which
                # physical record was added/removed/changed.  Report only the net
                # multiplicity difference and do not manufacture a record update.
                if key in duplicate_set:
                    if delta > 0:
                        diff.rows_added += delta
                    elif delta < 0:
                        diff.rows_removed += -delta
                    continue

                # A unique key with changed multiplicity is also not safely alignable
                # to a single record, so report the multiplicity delta rather than a
                # spurious ``rows_changed`` event.
                if delta > 0:
                    diff.rows_added += delta
                    continue
                if delta < 0:
                    diff.rows_removed += -delta
                    continue

                before_rows = valid_before.iloc[bi[key]][compare_cols]
                after_rows = valid_after.iloc[ai[key]][compare_cols]
                if not before_rows.reset_index(drop=True).equals(after_rows.reset_index(drop=True)):
                    changed += 1

            diff.rows_changed = changed
            if any(str(before[k].dtype) != str(after[k].dtype) for k in requested_keys):
                diff.notes.append(
                    "One or more stable-key column dtypes changed between versions; identity matching is based "
                    "on canonical typed key values and may therefore classify representation changes as additions/removals."
                )
            return diff

    # No usable stable key: compare logical row multisets, never row positions.
    common_for_rows = common
    if not common_for_rows:
        diff.notes.append("No common columns exist, so row-level comparison is unavailable.")
        diff.rows_added = len(after)
        diff.rows_removed = len(before)
        diff.row_identity_mode = "UNAVAILABLE"
        return diff
    bm = _row_multiset(before, common_for_rows)
    am = _row_multiset(after, common_for_rows)
    diff.rows_added = sum(max(0, am.get(k, 0) - bm.get(k, 0)) for k in set(am) | set(bm))
    diff.rows_removed = sum(max(0, bm.get(k, 0) - am.get(k, 0)) for k in set(am) | set(bm))
    diff.notes.append("No stable key supplied; row changes are reported as multiset additions/removals, not record-level updates.")
    return diff
