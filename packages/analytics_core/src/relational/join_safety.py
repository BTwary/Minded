"""Join safety gate: determines whether a proposed join between two
datasets is safe to execute BEFORE any SQL runs.

This module never converts UNKNOWN into SAFE. If cardinality cannot be
established (missing key, non-unique-but-ambiguous, empty data), the gate
returns UNKNOWN and the caller must block execution unless the join was
explicitly authorized (e.g. an explicit many-to-many fanout the user
requested and the caller pre-aggregates for).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd


class Cardinality(str, Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"
    UNKNOWN = "unknown"


class SafetyStatus(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


@dataclass
class JoinSafetyReport:
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    cardinality: Cardinality
    left_key_unique: Optional[bool]
    right_key_unique: Optional[bool]
    fanout_risk: bool
    key_type_compatible: bool
    keys_exist: bool
    status: SafetyStatus
    reason: str


@dataclass(frozen=True)
class AggregateGrainSafetyReport:
    """Whether raw aggregation can preserve a metric's source-table grain."""
    metric_table: str
    safe: bool
    unsafe_hops: List[JoinSafetyReport]
    reason: str


def _key_present(df: pd.DataFrame, key: str) -> bool:
    return key in df.columns


def _is_unique(df: pd.DataFrame, key: str) -> Optional[bool]:
    if key not in df.columns or df.empty:
        return None
    non_null = df[key].dropna()
    if len(non_null) == 0:
        return None
    return bool(non_null.is_unique)


def _types_compatible(left: pd.Series, right: pd.Series) -> bool:
    # Conservative: numeric-vs-numeric or object/string-vs-object/string is
    # compatible; anything else (e.g. numeric key vs datetime key) is not,
    # since a silently mismatched-type join key is a classic cause of a
    # join that "succeeds" but returns nonsense.
    l_numeric = pd.api.types.is_numeric_dtype(left)
    r_numeric = pd.api.types.is_numeric_dtype(right)
    if l_numeric and r_numeric:
        return True
    l_obj = pd.api.types.is_object_dtype(left) or pd.api.types.is_string_dtype(left)
    r_obj = pd.api.types.is_object_dtype(right) or pd.api.types.is_string_dtype(right)
    if l_obj and r_obj:
        return True
    return False


def assess_join_safety(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    left_table: str = "left",
    right_table: str = "right",
    authorize_many_to_many: bool = False,
) -> JoinSafetyReport:
    """Assess whether a join on (left_key, right_key) is safe to execute.

    Never returns SAFE for a case that could not be positively established
    -- missing keys or indeterminate uniqueness (empty data) resolve to
    UNKNOWN, not SAFE.
    """
    keys_exist = _key_present(left_df, left_key) and _key_present(right_df, right_key)
    if not keys_exist:
        return JoinSafetyReport(
            left_table=left_table, right_table=right_table,
            left_key=left_key, right_key=right_key,
            cardinality=Cardinality.UNKNOWN,
            left_key_unique=None, right_key_unique=None,
            fanout_risk=True, key_type_compatible=False, keys_exist=False,
            status=SafetyStatus.UNSAFE,
            reason=f"Join key missing: left has '{left_key}'={_key_present(left_df, left_key)}, "
                   f"right has '{right_key}'={_key_present(right_df, right_key)}.",
        )

    type_ok = _types_compatible(left_df[left_key], right_df[right_key])
    left_unique = _is_unique(left_df, left_key)
    right_unique = _is_unique(right_df, right_key)

    if left_unique is None or right_unique is None:
        return JoinSafetyReport(
            left_table=left_table, right_table=right_table,
            left_key=left_key, right_key=right_key,
            cardinality=Cardinality.UNKNOWN,
            left_key_unique=left_unique, right_key_unique=right_unique,
            fanout_risk=True, key_type_compatible=type_ok, keys_exist=True,
            status=SafetyStatus.UNKNOWN,
            reason="Key uniqueness could not be established (empty or all-null key column). "
                   "Refusing to assume SAFE.",
        )

    if not type_ok:
        return JoinSafetyReport(
            left_table=left_table, right_table=right_table,
            left_key=left_key, right_key=right_key,
            cardinality=Cardinality.UNKNOWN,
            left_key_unique=left_unique, right_key_unique=right_unique,
            fanout_risk=True, key_type_compatible=False, keys_exist=True,
            status=SafetyStatus.UNSAFE,
            reason=f"Join key types incompatible: {left_df[left_key].dtype} vs {right_df[right_key].dtype}.",
        )

    if left_unique and right_unique:
        cardinality = Cardinality.ONE_TO_ONE
    elif left_unique and not right_unique:
        cardinality = Cardinality.ONE_TO_MANY
    elif not left_unique and right_unique:
        cardinality = Cardinality.MANY_TO_ONE
    else:
        cardinality = Cardinality.MANY_TO_MANY

    fanout_risk = cardinality in (Cardinality.MANY_TO_MANY, Cardinality.ONE_TO_MANY)

    if cardinality == Cardinality.MANY_TO_MANY and not authorize_many_to_many:
        return JoinSafetyReport(
            left_table=left_table, right_table=right_table,
            left_key=left_key, right_key=right_key,
            cardinality=cardinality,
            left_key_unique=left_unique, right_key_unique=right_unique,
            fanout_risk=True, key_type_compatible=True, keys_exist=True,
            status=SafetyStatus.UNSAFE,
            reason="many-to-many join produces a cartesian-style row explosion and was not "
                   "explicitly authorized (authorize_many_to_many=False).",
        )

    return JoinSafetyReport(
        left_table=left_table, right_table=right_table,
        left_key=left_key, right_key=right_key,
        cardinality=cardinality,
        left_key_unique=left_unique, right_key_unique=right_unique,
        fanout_risk=fanout_risk, key_type_compatible=True, keys_exist=True,
        status=SafetyStatus.SAFE,
        reason=f"Cardinality established as {cardinality.value}; "
               f"{'fanout is expected and accounted for' if fanout_risk else 'no fanout expected'}.",
    )


def assess_join_path(
    tables: List[pd.DataFrame],
    table_names: List[str],
    keys: List[tuple],
    authorize_many_to_many: bool = False,
) -> List[JoinSafetyReport]:
    """Assess a strictly LINEAR chain of joins: tables[0] JOIN tables[1]
    JOIN tables[2] ... ``keys`` is a list of (left_key, right_key) pairs,
    one per hop (len(keys) == len(tables) - 1).

    CAUTION: this assumes a linear chain, not an arbitrary join graph. A
    star-shaped plan (e.g. one base table joined independently to two
    others) is NOT representable by positional (tables[i], tables[i+1])
    pairing -- ``RelationalCompiler.compile_and_execute`` does NOT use this
    helper for that reason; it assesses each ``JoinHop``'s own declared
    left_table/right_table directly. Kept here only for genuinely linear
    join-path callers.
    """
    reports = []
    for i, (lk, rk) in enumerate(keys):
        reports.append(assess_join_safety(
            tables[i], tables[i + 1], lk, rk,
            left_table=table_names[i], right_table=table_names[i + 1],
            authorize_many_to_many=authorize_many_to_many,
        ))
    return reports

def assess_declared_join_hops(
    datasets: dict,
    join_hops: list,
    authorize_many_to_many: bool = False,
) -> List[JoinSafetyReport]:
    """Assess the exact join hops declared by an executable experiment.

    ``join_hops`` accepts dictionaries with ``left_table``, ``right_table``,
    ``left_key`` and ``right_key`` keys, matching ``JoinHop`` without creating
    a dependency on the relational compiler module.  This is the canonical
    controller preflight used before a JOIN-bearing experiment is allowed to
    execute.  Missing/ambiguous hops fail closed.
    """
    reports: List[JoinSafetyReport] = []
    for hop in join_hops or []:
        if not isinstance(hop, dict):
            reports.append(JoinSafetyReport(
                left_table="<invalid>", right_table="<invalid>",
                left_key="", right_key="", cardinality=Cardinality.UNKNOWN,
                left_key_unique=None, right_key_unique=None, fanout_risk=True,
                key_type_compatible=False, keys_exist=False,
                status=SafetyStatus.UNSAFE,
                reason="Join hop metadata is not an object; refusing execution.",
            ))
            continue
        lt, rt = hop.get("left_table"), hop.get("right_table")
        lk, rk = hop.get("left_key"), hop.get("right_key")
        if not all(isinstance(v, str) and v for v in (lt, rt, lk, rk)):
            reports.append(JoinSafetyReport(
                left_table=str(lt or "<missing>"), right_table=str(rt or "<missing>"),
                left_key=str(lk or ""), right_key=str(rk or ""),
                cardinality=Cardinality.UNKNOWN, left_key_unique=None,
                right_key_unique=None, fanout_risk=True, key_type_compatible=False,
                keys_exist=False, status=SafetyStatus.UNSAFE,
                reason="Join hop is missing table/key metadata; refusing execution.",
            ))
            continue
        if lt not in datasets or rt not in datasets:
            reports.append(JoinSafetyReport(
                left_table=lt, right_table=rt, left_key=lk, right_key=rk,
                cardinality=Cardinality.UNKNOWN, left_key_unique=None,
                right_key_unique=None, fanout_risk=True, key_type_compatible=False,
                keys_exist=False, status=SafetyStatus.UNSAFE,
                reason=f"Join hop references unavailable table(s): {lt!r}, {rt!r}.",
            ))
            continue
        reports.append(assess_join_safety(
            datasets[lt], datasets[rt], lk, rk,
            left_table=lt, right_table=rt,
            authorize_many_to_many=authorize_many_to_many,
        ))
    return reports


def assess_metric_aggregation_safety(
    metric_table: str,
    reports: List[JoinSafetyReport],
) -> AggregateGrainSafetyReport:
    """Fail closed when a raw join path can duplicate a source-table metric.

    M2-A deliberately has no allocation or pre-aggregation representation.
    Therefore a one-to-many hop in a connected raw plan is not admissible for
    an aggregate sourced from its base relation: it can repeat source rows.
    """
    unsafe = [report for report in reports if report.fanout_risk]
    if unsafe:
        return AggregateGrainSafetyReport(
            metric_table=metric_table,
            safe=False,
            unsafe_hops=unsafe,
            reason=(
                f"Raw aggregation of metric grain {metric_table!r} is blocked: "
                "a join hop can multiply source rows and no pre-aggregation or "
                "allocation semantics are declared."
            ),
        )
    return AggregateGrainSafetyReport(
        metric_table=metric_table,
        safe=True,
        unsafe_hops=[],
        reason="All declared hops are grain-preserving for raw metric aggregation.",
    )
