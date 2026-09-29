"""TemporalResolver: extracts natural-language temporal intent from a
question and resolves it against the ACTUAL date range present in the
resolved time column (schema-agnostic -- never assumes a column name).

This directly closes DEFECT-002 from CURRENT_DEFECT_REGISTER.md: prior to
this module, `InvestigationIntent.comparison_period_hint` was declared but
never populated, so no diagnostic experiment was ever scoped to the period
actually named in the question -- "why did revenue fall in March" was
answered from whole-history aggregates, diluting a real, sharp, single-month
anomaly into noise.

Design principle: resolution is always anchored to the data's own min/max
dates, never to wall-clock "today" (the dataset's "now" is whatever its
most recent timestamp is -- an investigation asking about "last month"
against a dataset that ends in April 2026 means March 2026, not whatever
month it is when the code runs).
"""
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Tuple

import pandas as pd

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES.keys()) + r")\s*,?\s*(\d{4})\b", re.IGNORECASE
)
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTH_NAMES.keys()) + r")\b", re.IGNORECASE)
_QUARTER_RE = re.compile(r"\bq([1-4])\b(?:\s*,?\s*(\d{4}))?", re.IGNORECASE)
_LAST_N_DAYS_RE = re.compile(r"\blast\s+(\d+)\s+days?\b", re.IGNORECASE)
_BETWEEN_RE = re.compile(
    r"\bbetween\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE
)
_BEFORE_RE = re.compile(r"\bbefore\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_AFTER_RE = re.compile(r"\bafter\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)


def _month_bounds(year: int, month: int) -> Tuple[date, date]:
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start, end


def _quarter_bounds(year: int, quarter: int) -> Tuple[date, date]:
    start_month = (quarter - 1) * 3 + 1
    start = date(year, start_month, 1)
    end_month = start_month + 3
    end = date(year + 1, 1, 1) if end_month > 12 else date(year, end_month, 1)
    return start, end


@dataclass
class ResolvedTemporalScope:
    """Result of resolving a question's temporal language against a dataset's
    actual date range. `matched_expression` and `resolution_method` are
    recorded for provenance (per the brief's requirement that the resolved
    temporal interpretation be traceable, not silently applied)."""
    found: bool
    period_start: Optional[date] = None          # inclusive
    period_end: Optional[date] = None             # exclusive
    baseline_start: Optional[date] = None         # inclusive, for period-over-period comparisons
    baseline_end: Optional[date] = None           # exclusive
    matched_expression: str = ""
    resolution_method: str = ""
    ambiguous: bool = False
    ambiguity_note: str = ""

    def to_temporal_scope(self, table: str, column: str):
        """Convert resolver output to the canonical Pydantic TemporalScope."""
        from packages.schemas.src.analysis import TemporalScope, VariableRef

        if not self.found:
            return None
        return TemporalScope(
            column=VariableRef(table=table, column=column),
            start=self.period_start.isoformat() if self.period_start else None,
            end=self.period_end.isoformat() if self.period_end else None,
            comparison_start=self.baseline_start.isoformat() if self.baseline_start else None,
            comparison_end=self.baseline_end.isoformat() if self.baseline_end else None,
        )

    def sql_filter(self, time_col: str) -> Optional[str]:
        """A dialect-neutral WHERE-clause fragment (DuckDB/SQLite/Postgres all
        accept lexical ISO-date string range comparisons on a text or date
        column). Returns None if nothing was resolved."""
        if not self.found or self.period_start is None or self.period_end is None:
            return None
        return (
            f"CAST({time_col} AS VARCHAR) >= '{self.period_start.isoformat()}' "
            f"AND CAST({time_col} AS VARCHAR) < '{self.period_end.isoformat()}'"
        )

    def baseline_sql_filter(self, time_col: str) -> Optional[str]:
        if self.baseline_start is None or self.baseline_end is None:
            return None
        return (
            f"CAST({time_col} AS VARCHAR) >= '{self.baseline_start.isoformat()}' "
            f"AND CAST({time_col} AS VARCHAR) < '{self.baseline_end.isoformat()}'"
        )


class TemporalResolver:
    """Extracts and resolves temporal intent from a question, anchored to a
    dataset's own observed date range (never wall-clock time)."""

    @staticmethod
    def resolve(question: str, primary_df: Optional[pd.DataFrame], time_col: Optional[str]) -> ResolvedTemporalScope:
        q = question.lower()

        data_min: Optional[date] = None
        data_max: Optional[date] = None
        if primary_df is not None and time_col and time_col in primary_df.columns:
            try:
                parsed = pd.to_datetime(primary_df[time_col], errors="coerce").dropna()
                if len(parsed) > 0:
                    data_min = parsed.min().date()
                    data_max = parsed.max().date()
            except Exception:
                pass

        # Explicit "between A and B"
        m = _BETWEEN_RE.search(q)
        if m:
            start = date.fromisoformat(m.group(1))
            end = date.fromisoformat(m.group(2)) + timedelta(days=1)
            return ResolvedTemporalScope(True, start, end, matched_expression=m.group(0), resolution_method="explicit_between")

        m = _BEFORE_RE.search(q)
        if m:
            end = date.fromisoformat(m.group(1))
            start = data_min or date(1970, 1, 1)
            return ResolvedTemporalScope(True, start, end, matched_expression=m.group(0), resolution_method="explicit_before")

        m = _AFTER_RE.search(q)
        if m:
            start = date.fromisoformat(m.group(1)) + timedelta(days=1)
            end = (data_max + timedelta(days=1)) if data_max else date(2100, 1, 1)
            return ResolvedTemporalScope(True, start, end, matched_expression=m.group(0), resolution_method="explicit_after")

        # "Month Year" e.g. "March 2026"
        m = _MONTH_YEAR_RE.search(q)
        if m:
            month = _MONTH_NAMES[m.group(1).lower()]
            year = int(m.group(2))
            start, end = _month_bounds(year, month)
            bstart, bend = TemporalResolver._prior_month(start)
            return ResolvedTemporalScope(True, start, end, bstart, bend, m.group(0), "explicit_month_year")

        # ISO date literal (rare in NL questions, but explicit)
        m = _ISO_DATE_RE.search(q)
        if m:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            start = date(year, month, day)
            end = start + timedelta(days=1)
            return ResolvedTemporalScope(True, start, end, matched_expression=m.group(0), resolution_method="explicit_iso_date")

        # "Qn" or "Qn YYYY"
        m = _QUARTER_RE.search(q)
        if m:
            quarter = int(m.group(1))
            year = int(m.group(2)) if m.group(2) else (data_max.year if data_max else date.today().year)
            start, end = _quarter_bounds(year, quarter)
            bstart, bend = _quarter_bounds(year, quarter - 1) if quarter > 1 else _quarter_bounds(year - 1, 4)
            return ResolvedTemporalScope(True, start, end, bstart, bend, m.group(0), "explicit_quarter")

        if "last quarter" in q or "previous quarter" in q:
            if data_max:
                cur_q = (data_max.month - 1) // 3 + 1
                if cur_q == 1:
                    start, end = _quarter_bounds(data_max.year - 1, 4)
                    bstart, bend = _quarter_bounds(data_max.year - 1, 3)
                else:
                    start, end = _quarter_bounds(data_max.year, cur_q - 1)
                    bstart, bend = _quarter_bounds(data_max.year, cur_q - 2) if cur_q > 2 else _quarter_bounds(data_max.year - 1, 4)
                return ResolvedTemporalScope(True, start, end, bstart, bend, "last quarter", "relative_last_quarter")
            return ResolvedTemporalScope(False, ambiguous=True, ambiguity_note="No time column / data range available to resolve 'last quarter'.")

        # "year over year" / "yoy"
        if "year over year" in q or "yoy" in q or "y/y" in q:
            if data_max:
                start, end = _month_bounds(data_max.year, data_max.month)
                bstart, bend = _month_bounds(data_max.year - 1, data_max.month)
                return ResolvedTemporalScope(True, start, end, bstart, bend, "year over year", "relative_yoy")
            return ResolvedTemporalScope(False, ambiguous=True, ambiguity_note="No data range available to resolve 'year over year'.")

        # "week over week" / "wow"
        if "week over week" in q or ("wow" in q.split()) or "w/w" in q:
            if data_max:
                end = data_max + timedelta(days=1)
                start = end - timedelta(days=7)
                bend = start
                bstart = bend - timedelta(days=7)
                return ResolvedTemporalScope(True, start, end, bstart, bend, "week over week", "relative_wow")
            return ResolvedTemporalScope(False, ambiguous=True, ambiguity_note="No data range available to resolve 'week over week'.")

        # "last N days"
        m = _LAST_N_DAYS_RE.search(q)
        if m:
            n = int(m.group(1))
            if data_max:
                end = data_max + timedelta(days=1)
                start = end - timedelta(days=n)
                bend = start
                bstart = bend - timedelta(days=n)
                return ResolvedTemporalScope(True, start, end, bstart, bend, m.group(0), "relative_last_n_days")
            return ResolvedTemporalScope(False, ambiguous=True, ambiguity_note=f"No data range available to resolve '{m.group(0)}'.")

        # "trailing 12 months"
        if "trailing 12 months" in q or "trailing twelve months" in q or "last 12 months" in q or "past year" in q:
            if data_max:
                end = data_max + timedelta(days=1)
                start = date(end.year - 1, end.month, 1)
                bend = start
                bstart = date(bend.year - 1, bend.month, 1)
                return ResolvedTemporalScope(True, start, end, bstart, bend, "trailing 12 months", "relative_trailing_12mo")
            return ResolvedTemporalScope(False, ambiguous=True, ambiguity_note="No data range available to resolve 'trailing 12 months'.")

        # "previous year" / "last year"
        if "previous year" in q or "last year" in q:
            if data_max:
                y = data_max.year - 1
                start, end = date(y, 1, 1), date(y + 1, 1, 1)
                bstart, bend = date(y - 1, 1, 1), date(y, 1, 1)
                return ResolvedTemporalScope(True, start, end, bstart, bend, "last year", "relative_last_year")
            return ResolvedTemporalScope(False, ambiguous=True, ambiguity_note="No data range available to resolve 'last year'.")

        # "this month" / "current month"
        if "this month" in q or "current month" in q:
            if data_max:
                start, end = _month_bounds(data_max.year, data_max.month)
                bstart, bend = TemporalResolver._prior_month(start)
                return ResolvedTemporalScope(True, start, end, bstart, bend, "this month", "relative_this_month")
            return ResolvedTemporalScope(False, ambiguous=True, ambiguity_note="No data range available to resolve 'this month'.")

        # "last month" / "previous month"
        if "last month" in q or "previous month" in q:
            if data_max:
                cur_start, _ = _month_bounds(data_max.year, data_max.month)
                start, end = TemporalResolver._prior_month(cur_start)
                bstart, bend = TemporalResolver._prior_month(start)
                return ResolvedTemporalScope(True, start, end, bstart, bend, "last month", "relative_last_month")
            return ResolvedTemporalScope(False, ambiguous=True, ambiguity_note="No data range available to resolve 'last month'.")

        # "yesterday" / "today"
        if "yesterday" in q:
            if data_max:
                start = data_max - timedelta(days=1)
                end = data_max
                bstart, bend = start - timedelta(days=1), start
                return ResolvedTemporalScope(True, start, end, bstart, bend, "yesterday", "relative_yesterday")
            return ResolvedTemporalScope(False, ambiguous=True, ambiguity_note="No data range available to resolve 'yesterday'.")
        if "today" in q:
            if data_max:
                start = data_max
                end = data_max + timedelta(days=1)
                bstart, bend = start - timedelta(days=1), start
                return ResolvedTemporalScope(True, start, end, bstart, bend, "today", "relative_today")
            return ResolvedTemporalScope(False, ambiguous=True, ambiguity_note="No data range available to resolve 'today'.")

        # Bare month name, e.g. "why did revenue fall in March" -- resolve
        # against whichever year the month actually occurs in within the
        # dataset's observed range. If the month appears in more than one
        # year of data, this is genuinely ambiguous and is reported as such
        # rather than silently guessing.
        m = _MONTH_RE.search(q)
        if m:
            month = _MONTH_NAMES[m.group(1).lower()]
            if primary_df is not None and time_col and time_col in primary_df.columns:
                try:
                    parsed = pd.to_datetime(primary_df[time_col], errors="coerce").dropna()
                    years_present = sorted(parsed[parsed.dt.month == month].dt.year.unique().tolist())
                except Exception:
                    years_present = []
            else:
                years_present = []

            if len(years_present) == 1:
                year = years_present[0]
                start, end = _month_bounds(year, month)
                bstart, bend = TemporalResolver._prior_month(start)
                return ResolvedTemporalScope(True, start, end, bstart, bend, m.group(0), "bare_month_resolved_against_data")
            elif len(years_present) > 1:
                # Ambiguous across multiple years -- default to the most
                # recent occurrence but flag it explicitly rather than
                # silently picking one, per the brief's instruction to mark
                # ambiguous temporal interpretations rather than guess
                # quietly.
                year = years_present[-1]
                start, end = _month_bounds(year, month)
                bstart, bend = TemporalResolver._prior_month(start)
                return ResolvedTemporalScope(
                    True, start, end, bstart, bend, m.group(0), "bare_month_resolved_against_data_ambiguous",
                    ambiguous=True,
                    ambiguity_note=(
                        f"'{m.group(0)}' occurs in {len(years_present)} different years in the data "
                        f"({years_present}); defaulted to the most recent ({year}). Specify a year for "
                        f"an unambiguous interpretation."
                    ),
                )
            else:
                return ResolvedTemporalScope(
                    False, ambiguous=True,
                    ambiguity_note=f"'{m.group(0)}' does not appear in the resolved time column's data at all.",
                )

        return ResolvedTemporalScope(False)

    @staticmethod
    def _prior_month(month_start: date) -> Tuple[date, date]:
        if month_start.month == 1:
            return date(month_start.year - 1, 12, 1), month_start
        return date(month_start.year, month_start.month - 1, 1), month_start
