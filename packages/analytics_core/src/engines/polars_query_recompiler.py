"""Bounded Polars-native re-execution for AA-OS generated SQL.

This module is intentionally a whitelist compiler, not a general SQL parser. It exists
only to independently reproduce the canonical SQL shapes emitted by the AA-OS experiment
synthesizer when Polars SQLContext cannot parse DuckDB-specific syntax. Unknown SQL returns
None so verification fails closed rather than silently changing the population or estimand.
"""
from typing import Any, List, Optional, Tuple
import re
import numpy as np
import pandas as pd
import polars as pl


def _split_top_level(text: str, delimiter: str = ",") -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    i += 2
                    continue
                quote = None
        else:
            if ch in ("'", '"'):
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == delimiter and depth == 0:
                parts.append(text[start:i].strip())
                start = i + 1
        i += 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _split_top_level_and(text: str) -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None
    upper = text.upper()
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    i += 2
                    continue
                quote = None
        else:
            if ch in ("'", '"'):
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and upper.startswith("AND", i):
                before = text[i - 1] if i else " "
                after = text[i + 3] if i + 3 < len(text) else " "
                if before.isspace() and after.isspace():
                    parts.append(text[start:i].strip())
                    start = i + 3
                    i += 3
                    continue
        i += 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _strip_wrapping_parentheses(text: str) -> str:
    value = text.strip()
    while len(value) >= 2 and value[0] == "(" and value[-1] == ")":
        depth = 0
        quote: Optional[str] = None
        balanced = True
        for i, ch in enumerate(value):
            if quote:
                if ch == quote:
                    quote = None
                continue
            if ch in ("'", '"'):
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(value) - 1:
                    balanced = False
                    break
        if balanced and depth == 0:
            value = value[1:-1].strip()
        else:
            break
    return value


def _sql_unquote(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1].replace("''", "'")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
        return float(value) if "." in value else int(value)
    return value


def _apply_known_where(pl_df: pl.DataFrame, where_sql: Optional[str]) -> Optional[pl.DataFrame]:
    if not where_sql:
        return pl_df
    current = pl_df
    pending = list(_split_top_level_and(where_sql))
    while pending:
        condition = _strip_wrapping_parentheses(pending.pop(0))
        nested = _split_top_level_and(condition)
        if len(nested) > 1:
            pending = nested + pending
            continue

        m = re.fullmatch(r"TRY_CAST\(\s*(?P<col>\w+)\s+AS\s+DATE\s*\)\s+IS\s+NOT\s+NULL", condition, re.IGNORECASE)
        if m:
            col = m.group("col")
            if col not in current.columns:
                return None
            try:
                parsed = current.select(pl.col(col).cast(pl.Utf8).str.to_datetime(strict=False).alias("__where_date__"))["__where_date__"]
                current = current.filter(parsed.is_not_null())
                continue
            except Exception:
                return None

        m = re.fullmatch(r"(?P<col>\w+)\s+IS\s+NOT\s+NULL", condition, re.IGNORECASE)
        if m:
            col = m.group("col")
            if col not in current.columns:
                return None
            current = current.filter(pl.col(col).is_not_null())
            continue

        m = re.fullmatch(r"(?P<col>\w+)\s+IS\s+NULL", condition, re.IGNORECASE)
        if m:
            col = m.group("col")
            if col not in current.columns:
                return None
            current = current.filter(pl.col(col).is_null())
            continue

        m = re.fullmatch(r"CAST\(\s*(?P<col>\w+)\s+AS\s+VARCHAR\s*\)\s*(?P<op>=|<>|!=|>=|<=|>|<)\s*(?P<rhs>'.*?')", condition, re.IGNORECASE)
        if m:
            col = m.group("col")
            if col not in current.columns:
                return None
            op = m.group("op")
            rhs = str(_sql_unquote(m.group("rhs")))
            expr = pl.col(col).cast(pl.Utf8)
            try:
                current = current.filter({"=": expr == rhs, "!=": expr != rhs, "<>": expr != rhs, ">": expr > rhs, ">=": expr >= rhs, "<": expr < rhs, "<=": expr <= rhs}[op])
                continue
            except Exception:
                return None

        m = re.fullmatch(r"(?P<col>\w+)\s*(?P<op>=|<>|!=|>=|<=|>|<)\s*(?P<rhs>'.*?'|[-+]?\d+(?:\.\d+)?)", condition, re.IGNORECASE)
        if m:
            col = m.group("col")
            if col not in current.columns:
                return None
            op = m.group("op")
            rhs = _sql_unquote(m.group("rhs"))
            expr = pl.col(col)
            try:
                current = current.filter({"=": expr == rhs, "!=": expr != rhs, "<>": expr != rhs, ">": expr > rhs, ">=": expr >= rhs, "<": expr < rhs, "<=": expr <= rhs}[op])
                continue
            except Exception:
                return None
        return None
    return current


def _strip_alias(expr: str) -> Tuple[str, Optional[str]]:
    m = re.fullmatch(r"(?P<body>.+?)\s+AS\s+(?P<alias>[A-Za-z_]\w*)", expr.strip(), re.IGNORECASE | re.DOTALL)
    return (m.group("body").strip(), m.group("alias")) if m else (expr.strip(), None)


def _strip_outer_numeric_cast(expr: str) -> str:
    m = re.fullmatch(r"CAST\(\s*(.+?)\s+AS\s+(?:DOUBLE|FLOAT|DECIMAL(?:\([^)]*\))?)\s*\)", expr.strip(), re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else expr.strip()


def _sum_expr(inner: str, columns: set[str]) -> Optional[pl.Expr]:
    factors = _split_top_level(inner, "*")
    if len(factors) == 2 and all(f in columns for f in factors):
        return (pl.col(factors[0]) * pl.col(factors[1])).sum()
    if inner in columns:
        return pl.col(inner).sum()
    return None


def _aggregate_expr(expr: str, columns: set[str]) -> Optional[pl.Expr]:
    raw = _strip_outer_numeric_cast(expr.strip())
    if re.fullmatch(r"COUNT\(\s*\*\s*\)", raw, re.IGNORECASE):
        return pl.len()
    m = re.fullmatch(r"COUNT\(\s*(DISTINCT\s+)?(\w+)\s*\)", raw, re.IGNORECASE)
    if m and m.group(2) in columns:
        return pl.col(m.group(2)).n_unique() if m.group(1) else pl.col(m.group(2)).count()
    m = re.fullmatch(r"STDDEV\(\s*(\w+)\s*\)", raw, re.IGNORECASE)
    if m and m.group(1) in columns:
        return pl.col(m.group(1)).std()
    for fn in ("SUM", "AVG", "MEAN"):
        m = re.fullmatch(rf"{fn}\(\s*(.+?)\s*\)", raw, re.IGNORECASE)
        if m:
            inner = m.group(1).strip()
            if fn == "SUM":
                return _sum_expr(inner, columns)
            return pl.col(inner).mean() if inner in columns else None

    ratio = re.fullmatch(r"(?P<num>.+?)\s*/\s*(?:NULLIF\(\s*)?(?P<den>.+?)(?:\s*,\s*0\s*\))?", raw, re.IGNORECASE | re.DOTALL)
    if ratio:
        num = _aggregate_expr(ratio.group("num"), columns)
        den_sql = ratio.group("den").strip()
        den = pl.len() if den_sql.upper() == "COUNT(*)" else _aggregate_expr(den_sql, columns)
        if num is not None and den is not None:
            return pl.when(den != 0).then(num / den).otherwise(None)
    return None


def execute_known_query(query_sql: str, pl_df: pl.DataFrame) -> Optional[pl.DataFrame]:
    q = query_sql.strip().rstrip(";").strip()
    m = re.fullmatch(
        r"SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<source>[A-Za-z_]\w*)"
        r"(?:\s+WHERE\s+(?P<where>.+?))?"
        r"(?:\s+GROUP\s+BY\s+(?P<group>.+?))?"
        r"(?:\s+ORDER\s+BY\s+(?P<order>.+?))?\s*$",
        q, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    current = _apply_known_where(pl_df, m.group("where"))
    if current is None:
        return None
    select_parts = _split_top_level(m.group("select"))
    group_parts = _split_top_level(m.group("group")) if m.group("group") else []
    if not select_parts:
        return None

    for part in select_parts:
        body, alias = _strip_alias(part)
        dm = re.fullmatch(r"DATE_TRUNC\(\s*'day'\s*,\s*TRY_CAST\(\s*(?P<col>\w+)\s+AS\s+DATE\s*\)\s*\)", body, re.IGNORECASE)
        if dm:
            col = dm.group("col")
            if col not in current.columns:
                return None
            try:
                current = current.with_columns(pl.col(col).cast(pl.Utf8).str.to_datetime(strict=False).dt.truncate("1d").alias(alias or "period"))
            except Exception:
                return None

    group_keys: List[str] = []
    for g in group_parts:
        g = g.strip()
        if g in current.columns:
            group_keys.append(g)
        else:
            return None

    if group_keys:
        agg_exprs: List[pl.Expr] = []
        for part in select_parts:
            body, alias = _strip_alias(part)
            out_name = alias or body
            if re.fullmatch(r"\w+", body) and body in group_keys:
                if out_name != body:
                    agg_exprs.append(pl.col(body).first().alias(out_name))
                continue
            if re.fullmatch(r"DATE_TRUNC\(.*\)", body, re.IGNORECASE):
                period_name = alias or "period"
                if period_name not in group_keys:
                    return None
                if out_name != period_name:
                    agg_exprs.append(pl.col(period_name).first().alias(out_name))
                continue
            agg = _aggregate_expr(body, set(current.columns))
            if agg is None:
                return None
            agg_exprs.append(agg.alias(out_name))
        try:
            out = current.group_by(group_keys, maintain_order=True).agg(agg_exprs)
        except Exception:
            return None
    else:
        exprs: List[pl.Expr] = []
        for part in select_parts:
            body, alias = _strip_alias(part)
            out_name = alias or body
            if re.fullmatch(r"\w+", body):
                if body not in current.columns:
                    return None
                exprs.append(pl.col(body).alias(out_name))
            elif len(body) >= 2 and body[0] == "'" and body[-1] == "'":
                exprs.append(pl.lit(_sql_unquote(body)).alias(out_name))
            elif re.fullmatch(r"DATE_TRUNC\(.*\)", body, re.IGNORECASE):
                period_name = alias or "period"
                if period_name not in current.columns:
                    return None
                exprs.append(pl.col(period_name).alias(out_name))
            else:
                agg = _aggregate_expr(body, set(current.columns))
                if agg is None:
                    return None
                exprs.append(agg.alias(out_name))
        try:
            out = current.select(exprs)
        except Exception:
            return None

    order_sql = m.group("order")
    if order_sql and len(out) > 0:
        order_parts = _split_top_level(order_sql)
        by: List[str] = []
        descending: List[bool] = []
        for part in order_parts:
            om = re.fullmatch(r"(?P<col>\w+)(?:\s+(?P<dir>ASC|DESC))?", part.strip(), re.IGNORECASE)
            if not om or om.group("col") not in out.columns:
                return None
            by.append(om.group("col"))
            descending.append((om.group("dir") or "ASC").upper() == "DESC")
        try:
            out = out.sort(by=by, descending=descending)
        except Exception:
            return None
    return out


def _canonical_cell(value: Any) -> str:
    try:
        if pd.isna(value):
            return "<NULL>"
    except Exception:
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.10g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _canonicalize_result_frame(frame: Any) -> pd.DataFrame:
    pdf = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame.copy()
    pdf = pdf.reset_index(drop=True)
    if pdf.empty:
        return pdf
    key = pdf.apply(lambda row: "\x1f".join(_canonical_cell(v) for v in row), axis=1)
    return pdf.loc[key.sort_values(kind="mergesort").index].reset_index(drop=True)


def result_frames_equivalent(primary_result_df: Any, secondary_result_df: Any, tolerance: float) -> Tuple[bool, float]:
    left = _canonicalize_result_frame(primary_result_df)
    right = _canonicalize_result_frame(secondary_result_df)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False, 1.0
    if left.empty:
        return True, 0.0
    max_delta = 0.0
    for col in left.columns:
        lcol, rcol = left[col], right[col]
        if pd.api.types.is_numeric_dtype(lcol) and pd.api.types.is_numeric_dtype(rcol):
            lvals = pd.to_numeric(lcol, errors="coerce").to_numpy(dtype=float)
            rvals = pd.to_numeric(rcol, errors="coerce").to_numpy(dtype=float)
            both_nan = np.isnan(lvals) & np.isnan(rvals)
            either_nan = np.isnan(lvals) | np.isnan(rvals)
            if not np.all(both_nan | ~either_nan):
                return False, 1.0
            finite = ~either_nan
            if finite.any():
                deltas = np.abs(lvals[finite] - rvals[finite])
                max_delta = max(max_delta, float(np.max(deltas)))
                if not np.allclose(lvals[finite], rvals[finite], rtol=0.0, atol=tolerance):
                    return False, max_delta
        else:
            if lcol.map(_canonical_cell).tolist() != rcol.map(_canonical_cell).tolist():
                return False, 1.0
    return True, max_delta
