"""GovernanceEngine: PII masking, token pseudonymization, and AST-only RBAC/RLS injection."""


class RLSQueryRewriteError(RuntimeError):
    """Raised when an RLS-scoped query cannot be safely rewritten."""

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import polars as pl

try:
    import sqlglot
    from sqlglot import exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False


class GovernanceEngine:
    """Enterprise governance, PII protection, and Row-Level Security injection."""

    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d{4}[ -]?){3}\d{4}\b",
        "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    }

    @classmethod
    def mask_pii_columns(
        cls,
        df: Union[pd.DataFrame, pl.DataFrame],
        salt: Optional[str] = None,
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Detects and hashes PII columns with SHA-256 and salt to prevent leakage in Evidence Ledgers."""
        is_polars = isinstance(df, pl.DataFrame)
        pdf = df.to_pandas() if is_polars else df.copy()
        effective_salt = salt or os.environ.get("MIND_PII_SALT", "minded_aaos_default_salt")

        for col in pdf.columns:
            if pd.api.types.is_string_dtype(pdf[col]) or pd.api.types.is_object_dtype(pdf[col]):
                sample = pdf[col].dropna().head(200).astype(str).tolist()
                if not sample:
                    continue

                is_pii = False
                for pattern in cls.PII_PATTERNS.values():
                    matches = sum(1 for val in sample if re.search(pattern, val))
                    if matches >= max(1, int(len(sample) * 0.10)):
                        is_pii = True
                        break

                if is_pii:
                    pdf[col] = pdf[col].apply(
                        lambda x: f"MASKED_{hashlib.sha256(f'{x}_{effective_salt}'.encode('utf-8')).hexdigest()[:12]}"
                        if pd.notnull(x) else x
                    )

        return pl.from_pandas(pdf) if is_polars else pdf

    @classmethod
    def inject_rbac_filters(cls, sql_query: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Safely injects Row-Level Security (RLS) conditions into SQL queries.
        
        Uses sqlglot AST when available, or deterministic clause injection.
        """
        if not user_context:
            return sql_query

        # Check for region or tenant restrictions
        region_val = user_context.get("region")
        tenant_val = user_context.get("tenant_id")

        if not region_val and not tenant_val:
            return sql_query

        def _safe_rbac_literal(value: Any) -> str:
            """Return a strictly validated SQL string literal for trusted RLS values.

            RBAC values are authorization inputs, so fail closed instead of
            attempting to escape arbitrary SQL-shaped strings.
            """
            if not isinstance(value, str) or not re.fullmatch(
                r"[A-Za-z0-9_\-\. ]{1,64}", value
            ):
                raise ValueError(f"Unsafe RBAC value: {value!r}")
            return "'" + value.replace("'", "''") + "'"

        conditions: List[str] = []
        if region_val is not None:
            conditions.append(f"region = {_safe_rbac_literal(region_val)}")
        if tenant_val is not None:
            conditions.append(f"tenant_id = {_safe_rbac_literal(tenant_val)}")

        combined_condition = " AND ".join(conditions)

        # RLS is a security boundary. It is never safe to guess SQL scope with
        # regular expressions. If SQLGlot is unavailable, or parsing/rewrite
        # fails, fail closed and force the caller to abort the query.
        if not _HAS_SQLGLOT:
            raise RLSQueryRewriteError(
                "SQLGlot is required for RBAC/RLS query rewriting; refusing "
                "to execute an unscoped SQL query."
            )

        try:
            parsed = sqlglot.parse_one(sql_query, dialect="duckdb")
            selects = list(parsed.find_all(exp.Select))
            if not selects:
                raise RLSQueryRewriteError(
                    "RBAC/RLS rewriting only supports SELECT statements; no SELECT node found."
                )

            # Every SELECT with a FROM source must receive the RLS predicate.
            # SELECTs with no FROM (e.g. SELECT 1) have no row source to scope.
            scoped_selects = 0
            for select in selects:
                # v20-audit fix (DEFECT-023): sqlglot's internal AST key for
                # the FROM clause changed across versions (it is "from_" in
                # sqlglot 30.x, not "from") -- .args.get("from") silently
                # returned None for every SELECT regardless of whether it
                # actually had a FROM clause, so inject_rbac_filters()
                # treated every scopeable SELECT as unscopeable and failed
                # closed even on ordinary, valid queries (confirmed via a
                # live regression against sqlglot 30.18.0). exp.Select.find
                # (exp.From) is the public, version-stable way to ask "does
                # this SELECT have a FROM clause" and is used everywhere
                # else in this module already.
                if select.find(exp.From) is None:
                    continue
                # Build a fresh expression for each SELECT. SQLGlot expressions
                # are parented nodes and must not be reused across the tree.
                rls_exp = sqlglot.parse_one(combined_condition, dialect="duckdb")
                # v20-audit fix (DEFECT-024, security-relevant): Select.where()
                # defaults to copy=True in this sqlglot version -- it returns
                # a NEW node and leaves the original (the one actually
                # attached to `parsed`, the tree this function serializes and
                # returns) untouched. The return value was previously
                # discarded, so this loop parsed and validated the RLS
                # predicate but never actually attached it anywhere: every
                # call to inject_rbac_filters() silently returned the
                # ORIGINAL, unscoped SQL unchanged (confirmed via a live
                # regression -- the returned SQL contained no injected
                # tenant_id condition at all despite no error being raised).
                # copy=False mutates the node in place so the change is
                # visible through `parsed`.
                if select.args.get("where") is not None:
                    select.where(rls_exp, append=True, copy=False)
                else:
                    select.where(rls_exp, copy=False)
                scoped_selects += 1

            if scoped_selects == 0:
                raise RLSQueryRewriteError(
                    "RBAC/RLS rewriting found no row-producing SELECT scope."
                )

            return parsed.sql(dialect="duckdb")
        except RLSQueryRewriteError:
            raise
        except Exception as exc:
            raise RLSQueryRewriteError(
                f"RBAC/RLS query rewrite failed closed: {type(exc).__name__}"
            ) from exc
