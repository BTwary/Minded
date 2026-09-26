"""
ReconciliationEngine — Semantic Formula Discovery for AA-OS.

Discovers and tests mathematical identities between numeric columns using
semantic role assignment and algebraic candidate generation.  It does NOT
use correlation as a proxy for a mathematical relationship.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Semantic role keyword sets
# ---------------------------------------------------------------------------

SEMANTIC_ROLES: Dict[str, set] = {
    'QUANTITY':        {'qty', 'quantity', 'units', 'count', 'volume', 'num', 'number'},
    'UNIT_PRICE':      {'price', 'unit_price', 'rate', 'cost_per', 'fee', 'tariff'},
    'GROSS_AMOUNT':    {'gross', 'gross_amount', 'gross_value', 'subtotal',
                       'gross_sales', 'list_price'},
    'DISCOUNT_RATE':   {'discount_rate', 'discount_pct', 'discount_percent', 'markdown_rate'},
    'DISCOUNT_AMOUNT': {'discount', 'discount_amount', 'markdown', 'rebate'},
    'NET_AMOUNT':      {'net', 'net_amount', 'net_value', 'net_sales', 'net_revenue'},
    'TAX_RATE':        {'tax_rate', 'tax_pct', 'vat_rate'},
    'TAX_AMOUNT':      {'tax', 'tax_amount', 'vat', 'gst'},
    'TOTAL':           {'total', 'total_amount', 'invoice_total', 'grand_total',
                       'amount_due', 'final_amount'},
    'REVENUE':         {'revenue', 'sales', 'turnover', 'income', 'arr', 'mrr'},
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ResidualClassification:
    row_index: Any
    lhs_value: float
    rhs_value: float
    residual: float
    classification: str  # EXACT | ROUNDING | BUSINESS_EXCEPTION | DATA_ERROR | UNKNOWN


@dataclass
class IdentityTestResult:
    formula_description: str
    lhs_column: str
    rhs_expression: str
    n_total: int
    n_exact: int
    n_rounding: int
    n_business_exception: int
    n_data_error: int
    n_unknown: int
    integrity_score: float          # (n_exact + n_rounding) / n_total
    example_discrepancies: List[ResidualClassification]  # max 5
    status: str  # CONSISTENT | ROUNDING_CONSISTENT | PARTIALLY_CONSISTENT | INCONSISTENT | UNTESTABLE


@dataclass
class ReconciliationResult:
    question: str
    dataset_name: str
    n_rows: int
    candidate_identities_tested: int
    identity_results: List[IdentityTestResult]
    overall_status: str  # ALL_CONSISTENT | SOME_INCONSISTENT | NO_IDENTITIES_FOUND | UNTESTABLE
    summary: str
    claim_ceiling: str = 'OBSERVATION'   # reconciliation is always observational


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ReconciliationEngine:
    """
    Discovers and tests algebraic identities between numeric columns.

    Strategy
    --------
    1. Assign semantic roles to numeric columns via keyword matching.
    2. Generate algebraic candidate identities from known role patterns.
    3. Test each candidate row-by-row: classify residuals as EXACT /
       ROUNDING / BUSINESS_EXCEPTION / DATA_ERROR / UNKNOWN.
    4. Summarise results into a :class:`ReconciliationResult`.

    Correlation is never used as a proxy for a mathematical relationship.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def discover_and_test(
        self,
        df: pd.DataFrame,
        question: str,
        semantic: Any,                  # reserved for future enrichment; unused here
        dataset_name: str = 'dataset',
    ) -> ReconciliationResult:
        """Main entry point — discover and test identities in *df*."""

        # ── edge cases ──────────────────────────────────────────────────
        if df is None or df.empty:
            return ReconciliationResult(
                question=question,
                dataset_name=dataset_name,
                n_rows=0,
                candidate_identities_tested=0,
                identity_results=[],
                overall_status='UNTESTABLE',
                summary='DataFrame is empty; no identities can be tested.',
            )

        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
        ]
        all_null_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().all()
        ]

        if not numeric_cols:
            return ReconciliationResult(
                question=question,
                dataset_name=dataset_name,
                n_rows=len(df),
                candidate_identities_tested=0,
                identity_results=[],
                overall_status='NO_IDENTITIES_FOUND',
                summary='No usable numeric columns found; cannot test identities.',
            )

        # ── role map: role → list of column names ────────────────────────
        question_cols = self._extract_question_columns(question, list(df.columns))
        role_map: Dict[str, List[str]] = {}
        for col in numeric_cols:
            role = self._assign_roles(col)
            if role:
                role_map.setdefault(role, []).append(col)

        # ── generate candidates ───────────────────────────────────────────
        candidates: List[Tuple[str, str, str, Callable]] = \
            self._generate_candidates(role_map, df, question_cols)

        # ── broad fallback when no semantic candidates ────────────────────
        if not candidates:
            candidates = self._generate_broad_candidates(numeric_cols, df)

        if not candidates:
            return ReconciliationResult(
                question=question,
                dataset_name=dataset_name,
                n_rows=len(df),
                candidate_identities_tested=0,
                identity_results=[],
                overall_status='NO_IDENTITIES_FOUND',
                summary=(
                    'No algebraic identity candidates could be generated from the '
                    'available numeric columns.'
                ),
            )

        # ── test each candidate ───────────────────────────────────────────
        results: List[IdentityTestResult] = []
        for lhs_col, rhs_desc, formula_desc, rhs_fn in candidates:
            rhs_series = self._safe_eval_rhs(rhs_fn, df)
            result = self._test_identity(df, lhs_col, rhs_series, formula_desc, rhs_desc)
            results.append(result)

        # ── overall status ────────────────────────────────────────────────
        testable = [r for r in results if r.status != 'UNTESTABLE']
        if not testable:
            overall = 'UNTESTABLE'
        elif all(r.status in ('CONSISTENT', 'ROUNDING_CONSISTENT') for r in testable):
            overall = 'ALL_CONSISTENT'
        elif any(r.status in ('INCONSISTENT', 'PARTIALLY_CONSISTENT') for r in testable):
            overall = 'SOME_INCONSISTENT'
        else:
            overall = 'ALL_CONSISTENT'

        # ── summary text ──────────────────────────────────────────────────
        summary_parts: List[str] = []
        if all_null_cols:
            summary_parts.append(
                f"Skipped all-null columns: {', '.join(all_null_cols)}."
            )
        if len(df) == 1:
            summary_parts.append('Note: only 1 row — statistical patterns cannot be inferred.')
        for r in results:
            summary_parts.append(
                f"[{r.lhs_column}] {r.formula_description} — "
                f"{r.status} (integrity={r.integrity_score:.0%}, "
                f"errors={r.n_data_error}/{r.n_total})"
            )

        return ReconciliationResult(
            question=question,
            dataset_name=dataset_name,
            n_rows=len(df),
            candidate_identities_tested=len(results),
            identity_results=results,
            overall_status=overall,
            summary='\n'.join(summary_parts) if summary_parts else 'No issues found.',
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign_roles(self, col_name: str) -> Optional[str]:
        """Return the best-matching semantic role for *col_name*, or None."""
        normalized = col_name.lower()
        # Split on non-word characters and underscores for token extraction
        tokens = set(re.split(r'[\W_]+', normalized))
        tokens.add(normalized)          # also try the full normalised name

        best_role: Optional[str] = None
        best_score = 0

        for role, keywords in SEMANTIC_ROLES.items():
            score = sum(
                1
                for kw in keywords
                if kw in tokens or any(t in kw or kw in t for t in tokens if len(t) > 2)
            )
            if score > best_score:
                best_score = score
                best_role = role

        return best_role if best_score > 0 else None

    def _extract_question_columns(
        self, question: str, df_columns: List[str]
    ) -> List[str]:
        """Return column names that appear (fuzzy) in the question text."""
        q_lower = question.lower()
        matched: List[str] = []
        for col in df_columns:
            col_norm = col.lower().replace('_', ' ')
            if col_norm in q_lower:
                matched.append(col)
                continue
            tokens = re.split(r'[\W_]+', col.lower())
            if any(t in q_lower for t in tokens if len(t) > 3):
                matched.append(col)
        return matched

    def _generate_candidates(
        self,
        role_map: Dict[str, List[str]],
        df: pd.DataFrame,
        question_cols: List[str],
    ) -> List[Tuple[str, str, str, Callable]]:
        """
        Generate (lhs_col, rhs_expr_str, formula_desc, rhs_callable) tuples
        from known algebraic patterns in *role_map*.

        Question-mentioned columns get priority: if a candidate's LHS column
        appears in *question_cols* it is placed first.
        """
        candidates: List[Tuple[str, str, str, Callable]] = []

        def _cols(role: str) -> List[str]:
            return role_map.get(role, [])

        def _add(lhs_col: str, rhs_desc: str, formula_desc: str, fn: Callable) -> None:
            candidates.append((lhs_col, rhs_desc, formula_desc, fn))

        # ── Pattern 1: GROSS_AMOUNT = QUANTITY × UNIT_PRICE ──────────────
        for qty in _cols('QUANTITY'):
            for price in _cols('UNIT_PRICE'):
                for gross in _cols('GROSS_AMOUNT'):
                    fn = (lambda q=qty, p=price: df[q] * df[p])
                    _add(
                        gross,
                        f'{qty} × {price}',
                        f'{gross} ≈ {qty} × {price}',
                        fn,
                    )

        # ── Pattern 2: NET_AMOUNT = GROSS × (1 − DISCOUNT_RATE) ──────────
        for gross in _cols('GROSS_AMOUNT'):
            for dr in _cols('DISCOUNT_RATE'):
                for net in _cols('NET_AMOUNT'):
                    fn = (lambda g=gross, d=dr: df[g] * (1 - df[d]))
                    _add(
                        net,
                        f'{gross} × (1 − {dr})',
                        f'{net} ≈ {gross} × (1 − {dr})',
                        fn,
                    )

        # ── Pattern 3: NET_AMOUNT = GROSS − DISCOUNT_AMOUNT ──────────────
        for gross in _cols('GROSS_AMOUNT'):
            for disc in _cols('DISCOUNT_AMOUNT'):
                for net in _cols('NET_AMOUNT'):
                    fn = (lambda g=gross, d=disc: df[g] - df[d])
                    _add(
                        net,
                        f'{gross} − {disc}',
                        f'{net} ≈ {gross} − {disc}',
                        fn,
                    )

        # ── Pattern 4: TOTAL = NET × (1 + TAX_RATE) ──────────────────────
        for net in _cols('NET_AMOUNT'):
            for tr in _cols('TAX_RATE'):
                for total in _cols('TOTAL'):
                    fn = (lambda n=net, t=tr: df[n] * (1 + df[t]))
                    _add(
                        total,
                        f'{net} × (1 + {tr})',
                        f'{total} ≈ {net} × (1 + {tr})',
                        fn,
                    )

        # ── Pattern 5: TOTAL = NET + TAX_AMOUNT ──────────────────────────
        for net in _cols('NET_AMOUNT'):
            for tax in _cols('TAX_AMOUNT'):
                for total in _cols('TOTAL'):
                    fn = (lambda n=net, t=tax: df[n] + df[t])
                    _add(
                        total,
                        f'{net} + {tax}',
                        f'{total} ≈ {net} + {tax}',
                        fn,
                    )

        # ── Pattern 6: DISCOUNT_IMPLIED = 1 − NET/GROSS ──────────────────
        for gross in _cols('GROSS_AMOUNT'):
            for net in _cols('NET_AMOUNT'):
                for dr in _cols('DISCOUNT_RATE'):
                    fn = (lambda g=gross, n=net: 1 - df[n] / df[g])
                    _add(
                        dr,
                        f'1 − {net}/{gross}',
                        f'{dr} ≈ 1 − {net}/{gross}',
                        fn,
                    )

        # ── Pattern 7: TOTAL = GROSS − DISCOUNT + TAX ────────────────────
        for gross in _cols('GROSS_AMOUNT'):
            for disc in _cols('DISCOUNT_AMOUNT'):
                for tax in _cols('TAX_AMOUNT'):
                    for total in _cols('TOTAL'):
                        fn = (lambda g=gross, d=disc, t=tax: df[g] - df[d] + df[t])
                        _add(
                            total,
                            f'{gross} − {disc} + {tax}',
                            f'{total} ≈ {gross} − {disc} + {tax}',
                            fn,
                        )

        # ── Pattern 8: REVENUE = QUANTITY × UNIT_PRICE ───────────────────
        for qty in _cols('QUANTITY'):
            for price in _cols('UNIT_PRICE'):
                for rev in _cols('REVENUE'):
                    fn = (lambda q=qty, p=price: df[q] * df[p])
                    _add(
                        rev,
                        f'{qty} × {price}',
                        f'{rev} ≈ {qty} × {price}',
                        fn,
                    )

        # ── Prioritise question-column LHS candidates ─────────────────────
        if question_cols:
            q_set = set(question_cols)
            priority = [c for c in candidates if c[0] in q_set]
            rest = [c for c in candidates if c[0] not in q_set]
            candidates = priority + rest

        return candidates

    def _generate_broad_candidates(
        self, numeric_cols: List[str], df: pd.DataFrame
    ) -> List[Tuple[str, str, str, Callable]]:
        """
        Fallback: generate additive / multiplicative pair candidates from all
        numeric columns, capped at 20 pairs.
        """
        candidates: List[Tuple[str, str, str, Callable]] = []
        cols = numeric_cols[:10]  # cap columns considered
        seen: set = set()

        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                if len(candidates) >= 20:
                    break

                # Additive: does any other column ≈ a + b?
                for c in cols:
                    if c in (a, b):
                        continue
                    key = (c, '+', a, b)
                    if key in seen:
                        continue
                    seen.add(key)
                    fn = (lambda x=a, y=b: df[x] + df[y])
                    candidates.append(
                        (c, f'{a} + {b}', f'{c} ≈ {a} + {b}', fn)
                    )
                    if len(candidates) >= 20:
                        break

                if len(candidates) >= 20:
                    break

                # Multiplicative: does any other column ≈ a × b?
                for c in cols:
                    if c in (a, b):
                        continue
                    key = (c, 'x', a, b)
                    if key in seen:
                        continue
                    seen.add(key)
                    fn = (lambda x=a, y=b: df[x] * df[y])
                    candidates.append(
                        (c, f'{a} × {b}', f'{c} ≈ {a} × {b}', fn)
                    )
                    if len(candidates) >= 20:
                        break

            if len(candidates) >= 20:
                break

        return candidates

    def _safe_eval_rhs(
        self, rhs_fn: Callable, df: pd.DataFrame
    ) -> pd.Series:
        """
        Call *rhs_fn* and return a Series aligned to df.index.
        Catches division-by-zero by replacing ±Inf/NaN with np.nan.
        """
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = rhs_fn()
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=df.index)
            # Replace infinities produced by /0 with NaN
            result = result.replace([np.inf, -np.inf], np.nan)
            return result
        except Exception:
            return pd.Series(np.nan, index=df.index)

    def _classify_residual(self, residual: float, expected_magnitude: float) -> str:
        """Classify a single row residual (preliminary, before one-sided check)."""
        abs_res = abs(residual)
        if abs_res < 1e-9:
            return 'EXACT'
        rel = abs_res / max(abs(expected_magnitude), 1e-12)
        if rel < 0.005:
            return 'ROUNDING'
        return 'DATA_ERROR'   # may be upgraded to BUSINESS_EXCEPTION by _test_identity

    def _test_identity(
        self,
        df: pd.DataFrame,
        lhs_col: str,
        rhs_series: pd.Series,
        formula_desc: str,
        rhs_expr: str,
    ) -> IdentityTestResult:
        """
        Test one algebraic identity row-by-row.

        Residual = lhs − rhs (signed).
        Classification per row:
            EXACT              |residual| < 1e-9
            ROUNDING           |residual| < 0.5 % × |rhs|
            BUSINESS_EXCEPTION residual consistently signed (≥80 % one-sided)
            DATA_ERROR         material, no consistent pattern
            UNKNOWN            cannot compute (NaN on either side)
        """
        lhs_series = df[lhs_col]

        # Both sides must exist (not NaN) for a row to be testable
        valid_mask = lhs_series.notna() & rhs_series.notna()
        n_total = int(valid_mask.sum())

        if n_total == 0:
            return IdentityTestResult(
                formula_description=formula_desc,
                lhs_column=lhs_col,
                rhs_expression=rhs_expr,
                n_total=0,
                n_exact=0,
                n_rounding=0,
                n_business_exception=0,
                n_data_error=0,
                n_unknown=int((~valid_mask).sum()),
                integrity_score=0.0,
                example_discrepancies=[],
                status='UNTESTABLE',
            )

        lhs_v = lhs_series[valid_mask].astype(float)
        rhs_v = rhs_series[valid_mask].astype(float)
        residuals = lhs_v - rhs_v

        # Row-level preliminary classification (EXACT / ROUNDING / provisional DATA_ERROR)
        raw_class: List[str] = []
        for res, rhs_val in zip(residuals, rhs_v):
            raw_class.append(self._classify_residual(float(res), float(rhs_val)))

        # Determine whether material errors are one-sided (BUSINESS_EXCEPTION) or random
        error_indices = [i for i, c in enumerate(raw_class) if c == 'DATA_ERROR']
        error_residuals = residuals.values[error_indices] if error_indices else np.array([])

        if len(error_residuals) > 1:
            positive = int((error_residuals > 0).sum())
            negative = int((error_residuals < 0).sum())
            total_errors = len(error_residuals)
            one_sided = (
                positive / total_errors >= 0.80
                or negative / total_errors >= 0.80
            )
            final_error_label = 'BUSINESS_EXCEPTION' if one_sided else 'DATA_ERROR'
        else:
            # Single error — cannot statistically call it BUSINESS_EXCEPTION
            final_error_label = 'DATA_ERROR'

        # Final class array
        final_class = [
            final_error_label if c == 'DATA_ERROR' else c
            for c in raw_class
        ]

        # Unknown rows (NaN on either side)
        n_unknown = int((~valid_mask).sum())

        # Counts
        n_exact = sum(1 for c in final_class if c == 'EXACT')
        n_rounding = sum(1 for c in final_class if c == 'ROUNDING')
        n_business_exception = sum(1 for c in final_class if c == 'BUSINESS_EXCEPTION')
        n_data_error = sum(1 for c in final_class if c == 'DATA_ERROR')

        integrity_score = (n_exact + n_rounding) / n_total if n_total > 0 else 0.0

        # Status
        if integrity_score == 1.0 and n_total > 0:
            status = 'CONSISTENT' if n_rounding == 0 else 'ROUNDING_CONSISTENT'
        elif integrity_score >= 0.9:
            status = 'ROUNDING_CONSISTENT' if n_rounding > 0 else 'PARTIALLY_CONSISTENT'
        elif integrity_score > 0.0:
            status = 'PARTIALLY_CONSISTENT'
        else:
            status = 'INCONSISTENT'

        # Up to 5 example discrepancies (non-EXACT rows)
        example_discrepancies: List[ResidualClassification] = []
        valid_indices = lhs_series[valid_mask].index.tolist()
        for idx, lhs_val, rhs_val, res, cls in zip(
            valid_indices, lhs_v.tolist(), rhs_v.tolist(),
            residuals.tolist(), final_class
        ):
            if cls != 'EXACT' and len(example_discrepancies) < 5:
                example_discrepancies.append(
                    ResidualClassification(
                        row_index=idx,
                        lhs_value=float(lhs_val),
                        rhs_value=float(rhs_val),
                        residual=float(res),
                        classification=cls,
                    )
                )

        return IdentityTestResult(
            formula_description=formula_desc,
            lhs_column=lhs_col,
            rhs_expression=rhs_expr,
            n_total=n_total,
            n_exact=n_exact,
            n_rounding=n_rounding,
            n_business_exception=n_business_exception,
            n_data_error=n_data_error,
            n_unknown=n_unknown,
            integrity_score=integrity_score,
            example_discrepancies=example_discrepancies,
            status=status,
        )
