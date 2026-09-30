"""Analyst answer layer: the numbers-first answer a human data analyst would give.

Why this exists
---------------
The scientific loop (hypotheses -> experiments -> Bayesian belief -> verdict) decides *how
much evidence there is*.  It does not, on its own, tell the person who asked the question
the thing they asked for.  Measured on ordinary business questions the loop produced

* "Which category had the highest total sales?" -> "Inconclusive: unable to find sufficient
  evidence" (a plain aggregate was never reported);
* "Is conversion higher for B than A?" -> "'A' accounts for 0.2% of the observed converted"
  (a share statement that neither compares the groups nor states the direction);
* "Why did revenue drop in June?" -> a confident answer naming the *wrong* region.

This module is the missing step.  It is a pure, deterministic, AI-free function of
(question, resolved columns, dataframe).  It never decides a verdict; it recomputes the
quantity the question asks for, with its uncertainty, and states the caveats an analyst
would check before sending the number to someone (sample size, dropped rows, duplicates,
outlier sensitivity, incomplete trailing period, "is this drop unusual for this series").

Design rules (pinned by tests/independent_release/test_session10_analyst_answer.py)
----------------------------------------------------------------------------------
1. The resolved columns come from the final analytical contract; this module never
   re-resolves them.  If a required role is missing it returns ``None`` (say nothing rather
   than guess).
2. Every number in the text is computed here from the data; nothing is templated from the
   hypothesis text.
3. Association / difference language never claims causation.
4. "No detectable difference" is reported with the interval, not as "inconclusive": the
   interval is what tells the reader how large a difference the data can rule out.
5. Small samples, dropped rows, and outlier-driven means are surfaced, not hidden.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

ALPHA = 0.05
MIN_GROUP_N = 5
MAX_GROUPS_REPORTED = 12

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_NAMES = {1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june", 7: "july",
                8: "august", 9: "september", 10: "october", 11: "november", 12: "december"}


@dataclass
class AnalystResult:
    kind: str
    headline: str
    details: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    numbers: Dict[str, Any] = field(default_factory=dict)
    # True when the question asks for a fact (ranking/total) rather than an inference;
    # such an answer is a recomputed observation and needs no hypothesis test.
    descriptive: bool = False
    # True when the loop's hypothesis test answers a *different* question than the one asked
    # (a static "which segment is concentrated" test does not test "why did it drop in June"),
    # so the loop's verdict/statement must not be presented as the answer.
    supersedes_loop: bool = False
    # Direction of an inferential finding: "positive" / "negative" / "none" / None.
    finding: Optional[str] = None
    # Actionable operational recommendations tailored to the empirical finding.
    next_steps: List[str] = field(default_factory=list)
    # Human-analyst depth extensions:
    executive_summary: Optional[str] = None
    waterfall_bridge: Optional[Dict[str, Any]] = None
    mix_shift_decomposition: Optional[Dict[str, Any]] = None
    concentration_diagnostics: Optional[Dict[str, Any]] = None
    scenario_sensitivity: Optional[Dict[str, Any]] = None
    distribution_profile: Optional[Dict[str, Any]] = None
    multivariate_context: Optional[Dict[str, Any]] = None
    cohort_lifecycle: Optional[Dict[str, Any]] = None
    strategic_playbook: List[Dict[str, Any]] = field(default_factory=list)

    def to_text(self) -> str:
        parts = [self.headline]
        parts.extend(self.details)
        if self.waterfall_bridge and self.waterfall_bridge.get("steps"):
            steps = self.waterfall_bridge["steps"]
            if len(steps) > 1:
                step_strs = []
                for s in steps:
                    step_name = s.get("step", "")
                    if step_name == "Prior Base":
                        step_strs.append(f"Prior Base: {_num(s.get('subtotal', 0.0))}")
                    else:
                        step_strs.append(f"{step_name}: {_signed(s.get('delta', 0.0))}")
                final_val = self.waterfall_bridge.get("final_value", 0.0)
                parts.append("Reconciliation: " + " -> ".join(step_strs) + f" [Final: {_num(final_val)}].")
        if self.cohort_lifecycle and not any("cohort lifecycle bridge" in d.lower() for d in self.details):
            cl = self.cohort_lifecycle
            parts.append(
                f"Cohort bridge: Retained ({_signed(cl.get('retained_account_delta', 0.0))}), "
                f"New (+{_num(cl.get('new_account_acquisition', 0.0))}), "
                f"Churned (-{_num(cl.get('churned_account_loss', 0.0))})."
            )
        if self.strategic_playbook:
            playbook_parts = []
            for item in self.strategic_playbook:
                tier = item.get("tier", "Action")
                action = item.get("action", "")
                if action:
                    playbook_parts.append(f"[{tier}]: {action}")
            if playbook_parts:
                parts.append("Strategic Playbook: " + " | ".join(playbook_parts))
        if self.caveats:
            parts.append("Checks: " + " ".join(self.caveats))
        if self.next_steps:
            parts.append("Recommended next steps: " + " ".join(self.next_steps))
        return " ".join(p.strip() for p in parts if p and p.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "headline": self.headline,
            "details": list(self.details),
            "caveats": list(self.caveats),
            "numbers": self.numbers,
            "descriptive": self.descriptive,
            "supersedes_loop": self.supersedes_loop,
            "finding": self.finding,
            "next_steps": list(self.next_steps),
            "executive_summary": self.executive_summary or self.headline,
            "waterfall_bridge": self.waterfall_bridge,
            "mix_shift_decomposition": self.mix_shift_decomposition,
            "concentration_diagnostics": self.concentration_diagnostics,
            "scenario_sensitivity": self.scenario_sensitivity,
            "distribution_profile": self.distribution_profile,
            "multivariate_context": self.multivariate_context,
            "cohort_lifecycle": self.cohort_lifecycle,
            "strategic_playbook": list(self.strategic_playbook),
        }


# --------------------------------------------------------------------------- formatting
def _num(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.2f}B"
    if ax >= 1e6:
        return f"{x/1e6:.2f}M"
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:,.1f}"
    if ax >= 1:
        return f"{x:.2f}"
    if ax == 0:
        return "0"
    return f"{x:.3g}"


def _signed(x: float) -> str:
    return ("+" if x >= 0 else "-") + _num(abs(x))


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def _p(p: float) -> str:
    if p is None or math.isnan(p):
        return "p=n/a"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


def _d_label(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def _r_label(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    if a < 0.7:
        return "strong"
    return "very strong"


# --------------------------------------------------------------------------- data helpers
def _to_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num = s.astype(float)
    elif pd.api.types.is_bool_dtype(s):
        num = s.astype(float)
    else:
        raw = s.astype(str).str.strip()
        is_parens = raw.str.startswith("(") & raw.str.endswith(")")
        unwrapped = raw.where(~is_parens, raw.str.slice(1, -1))
        cleaned = unwrapped.str.replace(r"[,$€£¥₹₩%\s]", "", regex=True)
        is_neg = is_parens | cleaned.str.startswith("-") | cleaned.str.endswith("-")
        cleaned = cleaned.str.replace("-", "", regex=False)
        num = pd.to_numeric(cleaned, errors="coerce")
        num = num.where(~is_neg, -num)
    return num.replace([np.inf, -np.inf], np.nan)


def _is_numeric_like(s: pd.Series, threshold: float = 0.9) -> bool:
    non_null = s.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_numeric_dtype(s):
        return True
    return float(_to_numeric(non_null).notna().mean()) >= threshold


def _binary_values(s: pd.Series) -> Optional[pd.Series]:
    """0/1 series if the column is a binary outcome, else None."""
    non_null = s.dropna()
    if non_null.empty:
        return None
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    uniq = set(non_null.unique().tolist())
    if pd.api.types.is_numeric_dtype(s):
        if uniq <= {0, 1, 0.0, 1.0} and len(uniq) == 2:
            return s.astype(float)
        return None
    low = {str(u).strip().lower() for u in uniq}
    pairs = [({"yes", "no"}, "yes"), ({"true", "false"}, "true"), ({"y", "n"}, "y"),
             ({"1", "0"}, "1"), ({"churned", "retained"}, "churned"), ({"converted", "not converted"}, "converted")]
    for pair, positive in pairs:
        if low == pair:
            return s.astype(str).str.strip().str.lower().map(lambda v: 1.0 if v == positive else 0.0).where(s.notna())
    return None


def _parse_dates(s: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    non_null = s.dropna()
    if non_null.empty:
        return None
    if pd.api.types.is_numeric_dtype(s):
        try:
            min_v = float(non_null.min())
            max_v = float(non_null.max())
            if 1800 <= min_v and max_v <= 2200:
                s_int = s.round().astype(int)
                return pd.to_datetime(s_int.astype(str) + "-01-01", errors="coerce")
            elif 50 <= min_v and max_v <= 99:
                s_int = (1900 + s).round().astype(int)
                return pd.to_datetime(s_int.astype(str) + "-01-01", errors="coerce")
            elif 0 <= min_v and max_v <= 49:
                s_int = (2000 + s).round().astype(int)
                return pd.to_datetime(s_int.astype(str) + "-01-01", errors="coerce")
        except Exception:
            pass
        return None
    parsed = pd.to_datetime(s, errors="coerce")
    if float(parsed.notna().sum()) / max(1, len(non_null)) < 0.8:
        return None
    return parsed


def _hygiene(df: pd.DataFrame, cols: Sequence[str], used: int) -> List[str]:
    notes: List[str] = []
    total = len(df)
    if total and used < total:
        dropped = total - used
        pct = dropped / total
        notes.append(f"{dropped:,} of {total:,} rows ({_pct(pct)}) were excluded for missing/unparseable values in {', '.join(cols)}.")
    try:
        dup = int(df.duplicated().sum())
        if total and dup / total >= 0.01:
            notes.append(f"{dup:,} rows ({_pct(dup / total)}) are exact duplicates of another row; results include them.")
    except Exception:
        pass
    return notes


# --------------------------------------------------------------------------- question kind
_PERIOD_CHANGE_RE = re.compile(
    r"\b(why|what (?:happened|caused|drove|explains?)|reason|cause|driver|drivers|explain)\b", re.I)
_CHANGE_WORDS_RE = re.compile(
    r"\b(drop\w*|fell|fall\w*|declin\w*|decreas\w*|dip\w*|down|lower|increas\w*|rose|rise|ris\w*|spik\w*|"
    r"surg\w*|jump\w*|grew|growth|chang\w*|slump\w*|plung\w*|tank\w*)\b", re.I)
_TREND_RE = re.compile(
    r"\b(trend\w*|over time|over the years|over model years|growing|grown|growth|grow|going up|going down|trajectory|month over month|"
    r"mom|year over year|yoy|increas\w+ over|declin\w+ over|chang\w+ over|improv\w+ over|has \w+ (?:grown|increased|declined|decreased|improved|changed))\b", re.I)
_RANK_RE = re.compile(
    r"\b(which|who|top|highest|lowest|best|worst|most|least|largest|biggest|smallest|rank\w*|leading|"
    r"how much|how many|total|what (?:is|are|was|were) the (?:total|average|mean|median|sum|count|number))\b", re.I)
_COMPARE_RE = re.compile(
    r"\b(differ\w*|difference|compar\w*|higher|lower|greater|less|fewer|more|better|worse|vs\.?|versus|"
    r"between|than|varies|vary|outperform\w*|beat)\b", re.I)
_HIGHER_RE = re.compile(r"\b(higher|greater|more|larger|bigger|better|outperform\w*|beat\w*|above|exceed\w*)\b", re.I)
_LOWER_RE = re.compile(r"\b(lower|less|fewer|smaller|worse|below|under)\b", re.I)
_AVG_RE = re.compile(r"\b(average|avg|mean|typical|per (?:order|customer|user|sale|transaction))\b", re.I)
_MEDIAN_RE = re.compile(r"\bmedian\b", re.I)
_TOTAL_RE = re.compile(r"\b(total|sum|overall|combined|revenue|sales|in total)\b", re.I)
_COUNT_RE = re.compile(r"\b(how many|count|number of)\b", re.I)
_BREAKDOWN_RE = re.compile(
    r"\b(breakdown|break down|break-down|distribution|share|shares|split|composition|pareto|where does .* come from|how is .* distributed|proportion of)\b",
    re.I,
)

# DEFECT-037 (from external research spec review): every prior recency-phrase defect
# (031/033/034/035/036) shared one root shape -- the question clearly named *some* time
# window, `parse_period` didn't recognise the specific phrasing, and the question then
# silently fell through to a plain RANKING/aggregate answer that never mentions the time
# window at all, indistinguishable from an unrelated question. Each fix so far has closed
# one more phrasing, but the phrase space is unbounded (typos, regional wording, phrasings
# no one has probed yet). The reviewed spec's "Minimum Rejection Policy" names exactly this
# failure class -- "Fiscal/calendar period unresolved where material" is listed as a case
# that must not be silently compiled into an unrelated answer. Rather than only continuing
# to enumerate phrasings, detect the *general shape* of "the question references a time
# window" independently of whether that specific phrasing is in the table, so an
# unrecognised phrasing degrades to an honest, disclosed caveat instead of a silent
# misclassification. This does not replace `parse_period`'s specific-phrase table (which
# still gives exact, correctly-scoped periods for everything it recognises) -- it's a
# fail-closed backstop for whatever isn't in that table yet.
_GENERIC_TEMPORAL_MARKER_RE = re.compile(
    r"\b(?:last|past|trailing|this|current|previous|recent\w*|lately|latest)\s+(?:\w+\s+){0,2}"
    r"(?:day|week|fortnight|month|quarter|year)s?\b"
    r"|\byear[\s-]to[\s-]date\b|\bytd\b"
    r"|\b\w+\s+ago\b"
    r"|\bsince\s+(?:the\s+)?(?:start|beginning|end)\s+of\b",
    re.I,
)


def classify_question(
    question: str,
    *,
    has_time: bool = False,
    has_group: bool = True,
    has_explanatory: bool = False,
    period: Optional["Period"] = None,
) -> str:
    q = question or ""
    if re.search(r"does\s+(?:the\s+)?effect\s+of\s+.+?\s+on\s+.+?\s+differ\s+between", q, re.I):
        return "INTERACTION"
    if period is not None and has_time and (_CHANGE_WORDS_RE.search(q) or _PERIOD_CHANGE_RE.search(q)):
        return "PERIOD_CHANGE"
    # DEFECT-037: the question looks like a period-change question (has a change/why word)
    # and clearly names *a* time window, but `parse_period` didn't resolve it -- don't let it
    # fall through to a plain RANKING answer that silently drops the time reference entirely.
    if (period is None and has_time and (_CHANGE_WORDS_RE.search(q) or _PERIOD_CHANGE_RE.search(q))
            and _GENERIC_TEMPORAL_MARKER_RE.search(q)):
        return "PERIOD_CHANGE_UNRESOLVED_TIMEFRAME"
    if has_time and _TREND_RE.search(q) and not has_explanatory:
        return "TREND"
    if has_explanatory:
        return "ASSOCIATION"
    if has_group and (_RANK_RE.search(q) or _BREAKDOWN_RE.search(q)) and not re.search(r"\b(differ\w*|significant\w*)\b", q, re.I):
        return "RANKING"
    # DEFECT-038: a genuinely causal/diagnostic question with no time dimension at all --
    # "Why did cost_metric surge across datacenter_region?" -- reads exactly like a
    # PERIOD_CHANGE question (matches both _PERIOD_CHANGE_RE's "why"/"what caused" and
    # _CHANGE_WORDS_RE's "surge"/"spike"/etc.) except there is no time column to compute a
    # before/after comparison from. This layer has no time-free equivalent of PERIOD_CHANGE,
    # so it must not silently fall into the RANKING catch-all below: RANKING answers are
    # marked `descriptive=True, supersedes_loop=True` and the controller (runtime/controller.py)
    # uses that flag to overwrite the hypothesis loop's own verdict -- including a correct
    # DIAGNOSED verdict -- with a shallow "group X has the highest total" observation that
    # never actually answers the causal "why" the question asked. Returning "NONE" here makes
    # `build_analyst_result` return None (this module's existing, already-used-elsewhere
    # signal for "say nothing rather than guess"), which leaves the hypothesis loop's own
    # verdict untouched instead of silently downgrading/replacing it.
    # this check fires whenever no time window was resolved and none was even named
    # (both period-based branches above already had their chance to fire first), regardless
    # of whether the dataset happens to have a time column the question doesn't reference.
    if has_group and _PERIOD_CHANGE_RE.search(q) and _CHANGE_WORDS_RE.search(q) and not _COMPARE_RE.search(q):
        return "NONE"
    if has_group and re.search(r"\bwhy\b", q, re.I):
        return "ROOT_CAUSE"
    if has_group and (_COMPARE_RE.search(q) or _AVG_RE.search(q)):
        return "GROUP_COMPARISON"
    if has_group:
        return "RANKING"
    return "NONE"


# --------------------------------------------------------------------------- periods
@dataclass(frozen=True)
class Period:
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive
    label: str
    grain: str  # "month" | "quarter" | "year" | "week"
    resolved_year_note: str = ""


_QUANTIFIED_RECENCY_RE = re.compile(
    r"\b(?:last|past|trailing)\s+"
    r"(?P<num>\d+|a\s+couple(?:\s+of)?|couple(?:\s+of)?|a\s+few|few|several|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s+(?P<unit>day|days|week|weeks|month|months|quarter|quarters|year|years|fortnight|fortnights)\b",
    re.I,
)
_QUANTIFIED_RECENCY_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
    "a couple": 2, "a couple of": 2, "couple": 2, "couple of": 2,
    "a few": 3, "few": 3, "several": 4,
}
_QUANTIFIED_RECENCY_UNIT_DAYS = {
    "day": 1, "week": 7, "fortnight": 14, "month": 30, "quarter": 91, "year": 365,
}
# DEFECT-036: a bare "fortnight" ("in the past fortnight") and "a year ago"/"year ago"
# (used as a trailing-window anchor, e.g. "why is churn up compared to a year ago?") name an
# implicit count of 1 rather than the explicit "<last/past/trailing> <N> <unit>" shape
# `_QUANTIFIED_RECENCY_RE` requires, so they fell through it (and every other branch) the same
# way DEFECT-031/035's uncountable recency phrases did before their fixes. Resolved via the
# same rolling-window code path as `_QUANTIFIED_RECENCY_RE`, just with an implicit count of 1.
_BARE_RECENCY_RE = re.compile(
    r"\b(?:the\s+|a\s+|this\s+)?(?:past\s+|last\s+)?(?P<unit>fortnight)\b"
    r"|\b(?:compared\s+to\s+|vs\.?\s+|versus\s+)?a\s+year\s+ago\b",
    re.I,
)
# DEFECT-036: "year to date" / "YTD" is calendar-year-to-date -- start at Jan 1 of the latest
# date's year, end at the latest date -- distinct from the quantified rolling windows above
# (it's calendar-anchored, not a fixed-width trailing window) and from the existing "this year"
# phrase (which spans the *whole* year, including future months the data doesn't have yet).
_YTD_RE = re.compile(r"\byear[\s-]to[\s-]date\b|\bytd\b", re.I)
# DEFECT-036: "since the start/beginning of the <quarter|month|year>" is a common paraphrase of
# the already-supported "this <quarter|month|year>" and fell through to the generic RANKING
# fallback the same way the other recency phrasings above did.
_SINCE_START_OF_RE = re.compile(
    r"\bsince\s+the\s+(?:start|beginning)\s+of\s+(?:the\s+)?(?P<grain>quarter|month|year)\b", re.I)


def parse_period(question: str, dates: Optional[pd.Series]) -> Optional[Period]:
    """Resolve a period phrase against the data's own calendar (never the wall clock)."""
    if dates is None or dates.dropna().empty:
        return None
    q = (question or "").lower()
    d = dates.dropna()
    dmin, dmax = d.min(), d.max()
    year_m = re.search(r"\b(20\d{2}|19\d{2})\b", q)
    year = int(year_m.group(1)) if year_m else None

    m = re.search(r"\bq([1-4])\b(?:\s*(?:of\s*)?(20\d{2}))?", q)
    if m:
        qn = int(m.group(1))
        yr = int(m.group(2)) if m.group(2) else year
        note = ""
        if yr is None:
            yrs = sorted({ts.year for ts in d if (ts.month - 1) // 3 + 1 == qn})
            if not yrs:
                return None
            yr = yrs[-1]
            note = f"(most recent Q{qn} in the data: {yr})" if len(yrs) > 1 else ""
        start = pd.Timestamp(year=yr, month=3 * (qn - 1) + 1, day=1)
        return Period(start, start + pd.DateOffset(months=3), f"Q{qn} {yr}", "quarter", note)

    # DEFECT-035: numeric/word-quantified recency phrases ("last 30 days", "past two weeks",
    # "past couple of months", "trailing 3 months", ...) fall through every branch below the
    # same way DEFECT-031's vague recency phrases did (parse_period returns None, PERIOD_CHANGE
    # never fires, the question silently downgrades to the generic RANKING fallback). These
    # don't fit the fixed-grain phrase table below because they name an explicit, arbitrary-width
    # rolling window rather than a whole calendar month/quarter/year/week -- so they're resolved
    # here as a rolling N-day window ending at the latest date in the data, reusing the "week"
    # grain (never introducing a new grain value the downstream `{"month","quarter","year","week"}`
    # `freq` dict would `KeyError` on). "Month" in this phrasing is treated as a 30-day rolling
    # window, not a calendar month -- flagged explicitly in the resolved-period note since it's an
    # approximation, unlike the exact calendar-month arithmetic used elsewhere in this function.
    ytd_m = _YTD_RE.search(q)
    if ytd_m:
        anchor = dmax
        start = pd.Timestamp(year=anchor.year, month=1, day=1)
        end = anchor.normalize() + pd.Timedelta(days=1)
        return Period(
            start, end, f"{anchor.year} year-to-date", "year",
            f"('{ytd_m.group(0)}' interpreted as Jan 1-{anchor.date()} {anchor.year}, the latest "
            "date in the data -- a partial year, not the full calendar year)",
        )

    since_m = _SINCE_START_OF_RE.search(q)
    if since_m:
        anchor = dmax
        grain = since_m.group("grain")
        if grain == "month":
            start = pd.Timestamp(year=anchor.year, month=anchor.month, day=1)
        elif grain == "quarter":
            start = pd.Timestamp(year=anchor.year, month=3 * ((anchor.month - 1) // 3) + 1, day=1)
        else:
            start = pd.Timestamp(year=anchor.year, month=1, day=1)
        end = anchor.normalize() + pd.Timedelta(days=1)
        return Period(
            start, end, f"since start of {grain} ({start.date()}-{anchor.date()})", "year" if grain == "year" else grain,
            f"('{since_m.group(0)}' interpreted relative to the latest date in the data, {anchor.date()})",
        )

    qm = _QUANTIFIED_RECENCY_RE.search(q)
    bm = None if qm else _BARE_RECENCY_RE.search(q)
    if qm or bm:
        if qm:
            num_txt = re.sub(r"\s+", " ", qm.group("num").strip().lower())
            n = int(num_txt) if num_txt.isdigit() else _QUANTIFIED_RECENCY_WORD_NUM.get(num_txt)
            unit = qm.group("unit").rstrip("s")
            matched_text = qm.group(0)
        else:
            n = 1
            unit = bm.group("unit") or "year"  # the "a year ago" alternative has no named group
            matched_text = bm.group(0)
        if n:
            days = n * _QUANTIFIED_RECENCY_UNIT_DAYS.get(unit, 30)
            anchor = dmax
            end = anchor.normalize() + pd.Timedelta(days=1)
            start = end - pd.Timedelta(days=days)
            label = f"the last {n} {unit}{'s' if n != 1 else ''}"
            note = f"('{matched_text}' interpreted as a {days}-day rolling window ending {anchor.date()}, the latest date in the data)"
            if unit == "month":
                note += "; a 'month' here is counted as 30 days for this rolling-window phrasing, not a calendar month"
            elif unit == "quarter":
                note += "; a 'quarter' here is counted as 91 days for this rolling-window phrasing, not a calendar quarter"
            elif unit == "year":
                note += "; a 'year' here is a 365-day rolling window, not a calendar year"
            return Period(start, end, label, "week", note)

    for rel, grain in (("last month", "month"), ("this month", "month"), ("last quarter", "quarter"),
                       ("this quarter", "quarter"), ("last year", "year"), ("this year", "year"),
                       ("last week", "week"), ("this week", "week"),
                       # DEFECT-031: vague recency phrases ("why did X fall recently?") previously
                       # matched no branch here, so parse_period returned None, PERIOD_CHANGE never
                       # fired in classify_question, and the question silently fell through to the
                       # generic RANKING fallback -- producing a top-line "who's highest" answer that
                       # never mentioned the named segment, "why", or the change at all. Treat these
                       # as "last week" (a reasonable default recency window) rather than dropping the
                       # temporal framing entirely.
                       ("recently", "week"), ("lately", "week"), ("of late", "week"),
                       ("in recent days", "week"), ("in recent weeks", "week"),
                       ("this past week", "week"), ("the last few days", "week")):
        if rel in q:
            anchor = dmax
            if grain == "month":
                cur = pd.Timestamp(year=anchor.year, month=anchor.month, day=1)
                start = cur if rel.startswith("this") else cur - pd.DateOffset(months=1)
                return Period(start, start + pd.DateOffset(months=1), start.strftime("%B %Y"), "month",
                              f"('{rel}' is relative to the latest date in the data, {anchor.date()})")
            if grain == "quarter":
                qs = pd.Timestamp(year=anchor.year, month=3 * ((anchor.month - 1) // 3) + 1, day=1)
                start = qs if rel.startswith("this") else qs - pd.DateOffset(months=3)
                return Period(start, start + pd.DateOffset(months=3), f"Q{(start.month - 1) // 3 + 1} {start.year}",
                              "quarter", f"('{rel}' is relative to the latest date in the data, {anchor.date()})")
            if grain == "year":
                ys = pd.Timestamp(year=anchor.year, month=1, day=1)
                start = ys if rel.startswith("this") else ys - pd.DateOffset(years=1)
                return Period(start, start + pd.DateOffset(years=1), str(start.year), "year",
                              f"('{rel}' is relative to the latest date in the data, {anchor.date()})")
            ws = (anchor - pd.Timedelta(days=anchor.weekday())).normalize()
            start = ws if rel.startswith("this") else ws - pd.Timedelta(days=7)
            return Period(start, start + pd.Timedelta(days=7), f"week of {start.date()}", "week",
                          f"('{rel}' is relative to the latest date in the data, {anchor.date()})")

    for name, num in MONTHS.items():
        if re.search(rf"\b{name}\b", q):
            if name == "may" and not re.search(r"\b(in|during|for|of|last|since)\s+may\b|\bmay\s+(20\d{2}|vs|to)\b", q):
                continue
            yrs = sorted({ts.year for ts in d if ts.month == num})
            if not yrs:
                return None
            yr = year if year in yrs else (yrs[-1] if year is None else None)
            if yr is None:
                return None
            note = f"(most recent {_MONTH_NAMES.get(num, name).title()} in the data: {yr})" if year is None and len(yrs) > 1 else ""
            start = pd.Timestamp(year=yr, month=num, day=1)
            return Period(start, start + pd.DateOffset(months=1), start.strftime("%B %Y"), "month", note)

    if year is not None and re.search(rf"\b(in|during|for|of)\s+{year}\b", q):
        start = pd.Timestamp(year=year, month=1, day=1)
        if start > dmax or start + pd.DateOffset(years=1) <= dmin:
            return None
        return Period(start, start + pd.DateOffset(years=1), str(year), "year", "")
    return None


def _prior_period(p: Period) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if p.grain == "month":
        return p.start - pd.DateOffset(months=1), p.start
    if p.grain == "quarter":
        return p.start - pd.DateOffset(months=3), p.start
    if p.grain == "year":
        return p.start - pd.DateOffset(years=1), p.start
    # "week" grain also covers non-calendar-aligned rolling windows (DEFECT-031's vague-recency
    # phrases, and DEFECT-035's numeric/word-quantified windows like "last 30 days" / "past two
    # weeks"), which are not always exactly 7 days wide. Step back by the period's own width
    # rather than assuming a fixed 7 days, so the "preceding period" compared against is the same
    # length as the requested one. For the original calendar-week case (width == 7 days) this is
    # identical to the previous fixed behavior.
    width = p.end - p.start
    return p.start - width, p.start


# --------------------------------------------------------------------------- aggregation
def _choose_agg(question: str, default_agg: Optional[str], is_binary: bool, target: Optional[str] = None) -> str:
    q = question or ""
    if is_binary:
        return "rate"
    ql = q.lower()
    # Explicit user requests in question take precedence
    if re.search(r"\b(total|sum|combined|in total)\b", ql):
        return "sum"
    # Extensive counts (e.g. "number of passengers") aggregate as sum on the metric
    if re.search(r"\bnumber\s+of\s+([a-zA-Z0-9_\s]+)", ql) and target:
        t_low = target.lower()
        if any(tok in t_low for tok in ("rate", "price", "ratio", "pct", "percent", "margin", "score", "rating", "duration", "latency", "age", "discount", "aov", "mpg", "efficiency", "orbital", "mass", "weight", "horsepower", "carat", "bill_length", "flipper_length")):
            return "mean"
        return "sum"
    if _MEDIAN_RE.search(ql):
        return "median"
    if _COUNT_RE.search(ql) and not (target and "passenger" in target.lower()):
        return "count"
    if _AVG_RE.search(ql) or re.search(r"\b(average|avg|mean|rate)\b", ql):
        return "mean"
    # Column-aware invariant for intensive metrics (Acceptance Invariant #2):
    # Intensive metrics (mpg, efficiency, orbital_period, price, fare, etc.) must NEVER aggregate as SUM.
    if target:
        t_low = target.lower()
        if any(tok in t_low for tok in ("rate", "price", "ratio", "pct", "percent", "margin", "score", "rating", "duration", "latency", "age", "discount", "aov", "mpg", "efficiency", "orbital", "mass", "weight", "horsepower", "carat", "bill_length", "flipper_length")):
            return "mean"
    d = (default_agg or "").lower()
    if d in ("mean", "avg", "average", "rate", "ratio", "proportion"):
        return "mean"
    if d in ("sum", "total"):
        return "sum"
    if d == "count":
        return "count"
    if re.search(r"\b(grown|grown\s+over\s+time|increased\s+over\s+time)\b", ql):
        return "sum"
    return "sum"


def _agg(values: pd.Series, agg: str) -> float:
    if agg == "sum":
        return float(values.sum())
    if agg == "count":
        return float(values.count())
    if agg == "median":
        return float(values.median())
    return float(values.mean())


_AGG_WORD = {"sum": "total", "count": "count of", "mean": "average", "median": "median", "rate": "rate"}


# --------------------------------------------------------------------------- statistics
def _welch(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    diff = ma - mb
    if se == 0:
        return dict(diff=diff, lo=diff, hi=diff, p=1.0 if diff == 0 else 0.0, d=0.0, df=float("nan"))
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    tcrit = sps.t.ppf(1 - ALPHA / 2, df)
    t = diff / se
    p = 2 * sps.t.sf(abs(t), df)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d = diff / sp if sp > 0 else 0.0
    j = 1 - 3 / (4 * (na + nb) - 9)
    return dict(diff=diff, lo=diff - tcrit * se, hi=diff + tcrit * se, p=float(p), d=float(d * j), df=float(df))


def _rate_ci(x1: int, n1: int, x2: int, n2: int) -> Tuple[float, float, float]:
    """Newcombe (Wilson-score based) 95% CI for p1-p2."""
    def wilson(x, n):
        z = sps.norm.ppf(1 - ALPHA / 2)
        p = x / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return centre - half, centre + half
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return diff, lo, hi


def _mentioned_levels(question: str, levels: Sequence[Any]) -> List[Tuple[int, Any]]:
    found: List[Tuple[int, Any]] = []
    for lv in levels:
        s = str(lv)
        if not s:
            continue
        flags = 0 if len(s) <= 2 else re.I
        m = re.search(rf"(?<![A-Za-z0-9_]){re.escape(s)}(?![A-Za-z0-9_])", question or "", flags)
        if m:
            found.append((m.start(), lv))
    return sorted(found, key=lambda t: t[0])



def _outlier_note(sub: pd.DataFrame, group: str, target: str) -> List[str]:
    """Flag groups whose mean/total is dominated by a few extreme values."""
    notes: List[str] = []
    for lv, vals in sub.groupby(group)[target]:
        v = vals.to_numpy(float)
        if len(v) < 8:
            continue
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        extreme = v > q3 + 3 * iqr
        k = int(extreme.sum())
        if k == 0 or k > max(1, 0.05 * len(v)):
            continue
        total = v.sum()
        share = float(v[extreme].sum() / total) if total else 0.0
        if share >= 0.2:
            notes.append(f"In {lv}, {k} extreme value{'s' if k > 1 else ''} (max {_num(float(v.max()))}) make up {_pct(share, 0)} of the group total; "
                         f"its mean is {_num(float(v.mean()))} but its median is {_num(float(np.median(v)))}. Verify these are genuine before relying on means or totals.")
    return notes[:3]


# --------------------------------------------------------------------------- human analyst depth engines
def _gini_and_concentration(series: pd.Series) -> Dict[str, Any]:
    """Compute Gini inequality coefficient, HHI, and top percentile concentration shares."""
    s = _to_numeric(series).dropna()
    s = s[s >= 0]
    if s.empty or len(s) < 2:
        return {"gini_coefficient": 0.0, "top_20pct_share": 0.0, "top_5pct_share": 0.0, "top_1pct_share": 0.0, "hhi": 0.0, "assessment": "Insufficient data"}

    y = np.sort(s.to_numpy(float))
    n = len(y)
    total = float(np.sum(y))
    if total <= 0:
        return {"gini_coefficient": 0.0, "top_20pct_share": 0.0, "top_5pct_share": 0.0, "top_1pct_share": 0.0, "hhi": 0.0, "assessment": "All zero"}

    index = np.arange(1, n + 1)
    gini = float((2.0 * np.sum(index * y)) / (n * total) - (n + 1) / n)

    cum = np.cumsum(y[::-1]) / total
    k1 = max(1, int(math.ceil(0.01 * n)))
    k5 = max(1, int(math.ceil(0.05 * n)))
    k20 = max(1, int(math.ceil(0.20 * n)))

    top1 = float(cum[k1 - 1])
    top5 = float(cum[k5 - 1])
    top20 = float(cum[k20 - 1])

    shares = y / total
    hhi = float(np.sum((shares * 100.0) ** 2))

    if gini >= 0.60 or top20 >= 0.80:
        assessment = "Severe concentration / high tail-fragility (Pareto dominant)"
    elif gini >= 0.40 or top20 >= 0.60:
        assessment = "Moderate concentration"
    else:
        assessment = "Broadly distributed / low concentration"

    return {
        "gini_coefficient": round(gini, 3),
        "hhi": round(hhi, 1),
        "top_1pct_share": round(top1, 3),
        "top_5pct_share": round(top5, 3),
        "top_20pct_share": round(top20, 3),
        "assessment": assessment,
    }


def _distribution_profile(series: pd.Series) -> Dict[str, Any]:
    """Complete parametric and non-parametric profile of a continuous distribution."""
    s = _to_numeric(series).dropna()
    if s.empty:
        return {}
    v = s.to_numpy(float)
    n = len(v)
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    p10, p25, p50, p75, p90, p99 = [float(x) for x in np.percentile(v, [10, 25, 50, 75, 90, 99])]
    iqr = p75 - p25
    trim5 = float(sps.trim_mean(v, 0.05)) if n >= 20 else mean

    if std > 1e-12:
        skew = float(sps.skew(v)) if n >= 3 else 0.0
        kurt = float(sps.kurtosis(v)) if n >= 4 else 0.0
    else:
        skew = 0.0
        kurt = 0.0

    outliers = int(np.sum((v < p25 - 1.5 * iqr) | (v > p75 + 1.5 * iqr))) if iqr > 0 else 0
    extreme_outliers = int(np.sum((v < p25 - 3.0 * iqr) | (v > p75 + 3.0 * iqr))) if iqr > 0 else 0

    return {
        "n": n,
        "mean": round(mean, 2),
        "median": round(p50, 2),
        "trimmed_mean_5pct": round(trim5, 2),
        "std_dev": round(std, 2),
        "iqr": round(iqr, 2),
        "skewness": round(skew, 2),
        "kurtosis": round(kurt, 2),
        "p10": round(p10, 2),
        "p25": round(p25, 2),
        "p50": round(p50, 2),
        "p75": round(p75, 2),
        "p90": round(p90, 2),
        "p99": round(p99, 2),
        "outlier_count": outliers,
        "extreme_outlier_count": extreme_outliers,
    }


def _kitagawa_decomposition(
    prev: pd.DataFrame,
    cur: pd.DataFrame,
    dim: str,
    target: str,
    agg: str = "sum",
) -> Optional[Dict[str, Any]]:
    """Kitagawa decomposition of average change into Within-Group (Rate) vs Composition (Mix) shifts.
    Formula: Delta y_bar = Sum(w_bar_i * Delta y_i) + Sum(y_bar_i * Delta w_i)
    """
    if dim not in prev.columns or dim not in cur.columns:
        return None
    gp = prev.groupby(dim)[target]
    gc = cur.groupby(dim)[target]
    segs = sorted(set(gp.groups) | set(gc.groups), key=str)
    if len(segs) < 2 or len(segs) > 40:
        return None

    n0_tot, n1_tot = len(prev), len(cur)
    if n0_tot < 4 or n1_tot < 4:
        return None

    y0_tot = _agg(prev[target], agg)
    y1_tot = _agg(cur[target], agg)
    avg0 = y0_tot / n0_tot if agg in ("sum", "count") else float(prev[target].mean())
    avg1 = y1_tot / n1_tot if agg in ("sum", "count") else float(cur[target].mean())
    delta_avg = avg1 - avg0

    segments_detail = []
    sum_rate_eff = 0.0
    sum_mix_eff = 0.0
    within_directions = []

    for s in segs:
        p_sub = prev[prev[dim] == s][target]
        c_sub = cur[cur[dim] == s][target]
        n0_i = len(p_sub)
        n1_i = len(c_sub)
        if n0_i == 0 and n1_i == 0:
            continue
        w0_i = n0_i / n0_tot
        w1_i = n1_i / n1_tot
        y0_i = float(p_sub.mean()) if n0_i > 0 else 0.0
        y1_i = float(c_sub.mean()) if n1_i > 0 else 0.0

        w_bar_i = 0.5 * (w0_i + w1_i)
        y_bar_i = 0.5 * (y0_i + y1_i)

        rate_i = w_bar_i * (y1_i - y0_i)
        mix_i = y_bar_i * (w1_i - w0_i)

        sum_rate_eff += rate_i
        sum_mix_eff += mix_i

        if n0_i >= 2 and n1_i >= 2:
            within_directions.append(1 if y1_i > y0_i else -1 if y1_i < y0_i else 0)

        segments_detail.append({
            "segment": str(s),
            "prev_count": int(n0_i),
            "cur_count": int(n1_i),
            "prev_share": float(w0_i),
            "cur_share": float(w1_i),
            "prev_avg": float(y0_i),
            "cur_avg": float(y1_i),
            "rate_effect": float(rate_i),
            "mix_effect": float(mix_i),
        })

    rate_share = sum_rate_eff / delta_avg if abs(delta_avg) > 1e-9 else 0.0
    mix_share = sum_mix_eff / delta_avg if abs(delta_avg) > 1e-9 else 0.0

    has_positive_within = any(d > 0 for d in within_directions)
    has_negative_within = any(d < 0 for d in within_directions)

    # Simpson's Paradox Detection
    simpsons_paradox = False
    if delta_avg < 0 and has_positive_within and not has_negative_within and sum_rate_eff > 0:
        simpsons_paradox = True
    elif delta_avg > 0 and has_negative_within and not has_positive_within and sum_rate_eff < 0:
        simpsons_paradox = True
    elif (
        abs(delta_avg) > 1e-6
        and (sum_rate_eff * delta_avg < 0)
        and abs(sum_rate_eff) >= 0.10 * abs(delta_avg)
        and abs(sum_mix_eff) > abs(sum_rate_eff)
        and (has_positive_within if sum_rate_eff > 0 else has_negative_within)
    ):
        simpsons_paradox = True

    return {
        "dimension": dim,
        "delta_average": float(delta_avg),
        "pure_rate_effect": float(sum_rate_eff),
        "mix_shift_effect": float(sum_mix_eff),
        "rate_share_pct": float(rate_share * 100),
        "mix_share_pct": float(mix_share * 100),
        "simpsons_paradox": simpsons_paradox,
        "primary_mechanism": "mix_shift" if abs(sum_mix_eff) > abs(sum_rate_eff) else "rate_change",
        "segments": segments_detail,
    }


def _cohort_lifecycle_bridge(
    prev: pd.DataFrame,
    cur: pd.DataFrame,
    id_col: str,
    target: str,
    agg: str = "sum",
) -> Optional[Dict[str, Any]]:
    """Decompose period performance into New/Acquired, Retained/Expansion, and Lost/Churned cohorts."""
    if id_col not in prev.columns or id_col not in cur.columns:
        return None
    prev_ids = set(prev[id_col].dropna().unique())
    cur_ids = set(cur[id_col].dropna().unique())
    if len(prev_ids) < 3 or len(cur_ids) < 3:
        return None

    retained_ids = prev_ids & cur_ids
    new_ids = cur_ids - prev_ids
    lost_ids = prev_ids - cur_ids

    def val(frame, ids):
        sub = frame[frame[id_col].isin(ids)]
        return _agg(sub[target], agg) if not sub.empty else 0.0

    v_prev_total = _agg(prev[target], agg)
    v_cur_total = _agg(cur[target], agg)
    total_delta = v_cur_total - v_prev_total

    retained_prev = val(prev, retained_ids)
    retained_cur = val(cur, retained_ids)
    retained_delta = retained_cur - retained_prev

    new_val = val(cur, new_ids)
    lost_val = val(prev, lost_ids)

    return {
        "id_column": id_col,
        "total_delta": float(total_delta),
        "retained_account_delta": float(retained_delta),
        "retained_account_count": len(retained_ids),
        "new_account_acquisition": float(new_val),
        "new_account_count": len(new_ids),
        "churned_account_loss": float(lost_val),
        "churned_account_count": len(lost_ids),
        "net_churn_retention_ratio": round(new_val / lost_val, 2) if lost_val > 0 else None,
        "primary_driver": "retained_contraction" if retained_delta < -abs(lost_val) else "churn_attrition" if lost_val > new_val else "new_acquisition_slowdown",
    }


def _multivariate_ols_context(
    df: pd.DataFrame,
    target: str,
    predictor: str,
    candidate_cols: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Multivariate regression controlling for other numeric/categorical covariates.
    Computes controlled slope, elasticity, R2, and VIF check."""
    covars = []
    for c in candidate_cols:
        if c != target and c != predictor and c in df.columns:
            if _is_numeric_like(df[c]) and df[c].nunique(dropna=True) >= 3:
                covars.append(c)
            if len(covars) >= 3:
                break
    if not covars:
        return None

    cols = [target, predictor] + covars
    sub = df[cols].copy()
    for col in cols:
        sub[col] = _to_numeric(sub[col])
    sub = sub.dropna()
    N = len(sub)
    if N < 20:
        return None

    y = sub[target].to_numpy(float)
    x = sub[predictor].to_numpy(float)
    z_mat = sub[covars].to_numpy(float)

    X_mat = np.column_stack([np.ones(N), x, z_mat])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        y_hat = X_mat @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        p_count = X_mat.shape[1]
        r2_adj = 1.0 - (1.0 - r2) * (N - 1) / (N - p_count) if N > p_count else r2

        df_e = max(1, N - p_count)
        s2 = ss_res / df_e
        inv_cov = np.linalg.pinv(X_mat.T @ X_mat)
        se_pred = math.sqrt(max(0.0, s2 * inv_cov[1, 1]))
        t_stat = beta[1] / se_pred if se_pred > 0 else 0.0
        p_val = 2.0 * sps.t.sf(abs(t_stat), df_e)

        mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
        elasticity = float(beta[1] * (mean_x / mean_y)) if mean_y != 0 else 0.0

        return {
            "controlled_covariates": covars,
            "raw_predictor": predictor,
            "controlled_slope": round(float(beta[1]), 4),
            "controlled_se": round(float(se_pred), 4),
            "controlled_p_value": round(float(p_val), 4),
            "r_squared": round(float(r2), 4),
            "adjusted_r_squared": round(float(r2_adj), 4),
            "elasticity_at_mean": round(float(elasticity), 3),
            "is_significant_after_control": bool(p_val < ALPHA),
        }
    except Exception:
        return None


def _build_strategic_playbook(
    kind: str,
    finding: Optional[str],
    primary_driver: Optional[str],
    numbers: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build executive strategic and operational recommendations categorized into 3 actionable tiers."""
    playbook = []
    driver_str = primary_driver or "primary driver"

    if kind == "PERIOD_CHANGE":
        pvm = numbers.get("pvm_decomposition", {})
        vol_share = pvm.get("volume_effect_share", 0.0)
        sub_seg = numbers.get("sub_segment_findings", {})

        # Tier 1: Immediate Triage (24-48h)
        if numbers.get("intra_period_trajectory", {}).get("step_change_detected"):
            inflection = numbers["intra_period_trajectory"]["inflection_date"]
            playbook.append({
                "tier": "Immediate Operational Triage (24-48h)",
                "action": f"Audit deployments, outages, and campaign tracking around {inflection} step-change date.",
                "priority": "HIGH",
                "focus_area": "System & Telemetry",
            })
        else:
            playbook.append({
                "tier": "Immediate Operational Triage (24-48h)",
                "action": f"Review active billing runs, account cancellations, and operational logs in {driver_str}.",
                "priority": "HIGH",
                "focus_area": "Operations & Billing",
            })

        # Tier 2: Diagnostic Deep-Dive (1-2w)
        if sub_seg:
            playbook.append({
                "tier": "Diagnostic Deep-Dive (1-2 weeks)",
                "action": f"Conduct account-level cohort review within {driver_str} focusing specifically on {sub_seg.get('dimension')}='{sub_seg.get('top_subsegment')}'.",
                "priority": "HIGH",
                "focus_area": "Customer Cohort Diagnostic",
            })
        elif abs(vol_share) >= 0.60:
            playbook.append({
                "tier": "Diagnostic Deep-Dive (1-2 weeks)",
                "action": f"Analyze marketing attribution, lead flow, and churn velocity across {driver_str} (volume-driven decline).",
                "priority": "HIGH",
                "focus_area": "Acquisition & Churn Funnel",
            })
        else:
            playbook.append({
                "tier": "Diagnostic Deep-Dive (1-2 weeks)",
                "action": f"Audit discounting levels, bundle pricing, and product-tier mix shifts in {driver_str} (rate-driven change).",
                "priority": "HIGH",
                "focus_area": "Pricing & Yield Management",
            })

        # Tier 3: Strategic & Commercial Mitigation (30-90d)
        if numbers.get("mix_shift_decomposition", {}).get("simpsons_paradox"):
            playbook.append({
                "tier": "Strategic & Commercial Mitigation (30-90 days)",
                "action": "Restructure commercial incentives to counter mix-shift migration toward lower-rate products/channels.",
                "priority": "MEDIUM",
                "focus_area": "Commercial Strategy & Mix Optimization",
            })
        else:
            playbook.append({
                "tier": "Strategic & Commercial Mitigation (30-90 days)",
                "action": f"Diversify revenue concentration beyond {driver_str} and establish proactive retention safeguards for top accounts.",
                "priority": "MEDIUM",
                "focus_area": "Concentration & Diversification",
            })

    elif kind == "RANKING":
        playbook.append({
            "tier": "Immediate Operational Triage (24-48h)",
            "action": f"Verify data collection hygiene and ensure {driver_str}'s lead is not an artifact of tracking anomalies.",
            "priority": "MEDIUM",
            "focus_area": "Data Quality Assurance",
        })
        playbook.append({
            "tier": "Diagnostic Deep-Dive (1-2 weeks)",
            "action": f"Benchmark operational and commercial playbooks separating top performer ({driver_str}) from peer benchmarks.",
            "priority": "HIGH",
            "focus_area": "Best-Practice Benchmarking",
        })
        opp = numbers.get("scenario_sensitivity", {}).get("opportunity_gap_to_median")
        if opp:
            playbook.append({
                "tier": "Strategic & Commercial Mitigation (30-90 days)",
                "action": f"Execute targeted growth initiative to elevate bottom-quartile segments to the median performance tier (potential lift: {_num(opp)}).",
                "priority": "HIGH",
                "focus_area": "Opportunity Gap Closure",
            })

    elif kind in ("GROUP_COMPARISON", "ROOT_CAUSE"):
        if finding in ("positive", "negative"):
            playbook.append({
                "tier": "Immediate Operational Triage (24-48h)",
                "action": f"Document current operational variant baseline for {driver_str} to ensure repeatability.",
                "priority": "MEDIUM",
                "focus_area": "Process Documentation",
            })
            playbook.append({
                "tier": "Diagnostic Deep-Dive (1-2 weeks)",
                "action": "Examine customer segment interaction effects to verify whether the performance gap is uniform or concentrated.",
                "priority": "HIGH",
                "focus_area": "Sub-group Heterogeneity",
            })
            playbook.append({
                "tier": "Strategic & Commercial Mitigation (30-90 days)",
                "action": "Scale proven practices across lagging segments or reallocate budget toward higher-performing cohorts.",
                "priority": "HIGH",
                "focus_area": "Resource Reallocation",
            })
        else:
            playbook.append({
                "tier": "Immediate Operational Triage (24-48h)",
                "action": "Confirm that existing differences are within statistical noise and halt premature variant rollouts.",
                "priority": "HIGH",
                "focus_area": "Decision Discipline",
            })
            playbook.append({
                "tier": "Diagnostic Deep-Dive (1-2 weeks)",
                "action": "Evaluate secondary segmentation and potential confounding factors that may mask localized effects.",
                "priority": "MEDIUM",
                "focus_area": "Confounder & Sub-group Analysis",
            })

    elif kind == "ASSOCIATION":
        playbook.append({
            "tier": "Immediate Operational Triage (24-48h)",
            "action": "Ensure key operational dashboards track both variables alongside confounding indicators.",
            "priority": "LOW",
            "focus_area": "Monitoring & Telemetry",
        })
        playbook.append({
            "tier": "Diagnostic Deep-Dive (1-2 weeks)",
            "action": f"Design controlled intervention or A/B experiment to determine whether the {driver_str} relationship is causal.",
            "priority": "HIGH",
            "focus_area": "Causal Experimentation",
        })
        playbook.append({
            "tier": "Strategic & Commercial Mitigation (30-90 days)",
            "action": "Incorporate observed elasticities into revenue and capacity planning models.",
            "priority": "MEDIUM",
            "focus_area": "Predictive Planning",
        })

    elif kind == "TREND":
        playbook.append({
            "tier": "Immediate Operational Triage (24-48h)",
            "action": "Validate current run-rate against near-term operational capacity and financial forecasts.",
            "priority": "HIGH",
            "focus_area": "Run-rate Alignment",
        })
        playbook.append({
            "tier": "Diagnostic Deep-Dive (1-2 weeks)",
            "action": "Analyze period-to-period residuals to distinguish secular trend from seasonal or cyclical oscillations.",
            "priority": "MEDIUM",
            "focus_area": "Time-Series Decomposition",
        })
        playbook.append({
            "tier": "Strategic & Commercial Mitigation (30-90 days)",
            "action": "Calibrate medium-term targets and resourcing according to the projected trajectory.",
            "priority": "HIGH",
            "focus_area": "Strategic Capacity Planning",
        })

    return playbook


# --------------------------------------------------------------------------- RANKING
def _ranking(q, df, target, group, agg_default) -> Optional[AnalystResult]:
    # Pure count of group occurrences (e.g. "Which method discovered the most planets?")
    is_count_ranking = (target == group) or (target in df.columns and not _is_numeric_like(df[target])) or (_choose_agg(q, agg_default, False, target=target) == "count")
    if is_count_ranking:
        sub = df[[group]].dropna().copy()
        if sub.empty or sub[group].nunique() < 2:
            return None
        counts = sub[group].value_counts()
        want_low = bool(re.search(r"\b(lowest|least|worst|smallest|fewest|bottom)\b", q, re.I))
        if want_low:
            counts = counts.sort_values(ascending=True)
        table = pd.DataFrame({"value": counts.values, "n": counts.values, "mean": counts.values, "sum": counts.values}, index=counts.index)
        top = table.iloc[0]
        top_name = table.index[0]
        grand = float(counts.sum())
        order = "fewest" if want_low else "most"
        target_label = "planets" if "planet" in q.lower() else (target if target != group else group)
        headline = f"{top_name} discovered the {order} {target_label}: {int(top['value']):,} (n={int(top['n']):,}), {_pct(top['value'] / grand)} of the overall {int(grand):,}."
        details = []
        ranked = [f"{k}: {int(r['value']):,} (n={int(r['n']):,})" for k, r in table.head(MAX_GROUPS_REPORTED).iterrows()]
        details.append(f"Ranking by count of {group}: " + "; ".join(ranked) + ("; ..." if len(table) > MAX_GROUPS_REPORTED else "."))
        numbers = {"aggregation": "count", "top": str(top_name), "top_value": float(top["value"]),
                   "table": {str(k): {"value": float(r["value"]), "n": int(r["n"])} for k, r in table.iterrows()}}
        return AnalystResult(
            headline=headline,
            details=details,
            numbers=numbers,
            caveats=_hygiene(df, [group], len(sub)),
            next_steps=[],
            kind="RANKING",
            descriptive=True,
            finding="ranking",
            executive_summary=headline,
        )

    sub = df[[group, target]].copy()
    sub[target] = _to_numeric(sub[target])
    sub = sub.dropna(subset=[group, target])
    if sub.empty or sub[group].nunique() < 2:
        return None
    binary = _binary_values(df[target])
    if binary is not None:
        sub[target] = binary.loc[sub.index]
    agg = _choose_agg(q, agg_default, binary is not None, target=target)
    want_low = bool(re.search(r"\b(lowest|least|worst|smallest|fewest|bottom)\b", q, re.I))
    grp = sub.groupby(group)[target]
    table = pd.DataFrame({"value": grp.apply(lambda s: _agg(s, "mean" if agg == "rate" else agg)),
                          "n": grp.count(), "mean": grp.mean(), "sum": grp.sum()})
    table = table[table["n"] > 0].sort_values("value", ascending=want_low)
    if table.empty:
        return None
    top = table.iloc[0]
    top_name = table.index[0]
    grand = _agg(sub[target], "mean" if agg == "rate" else agg)
    word = _AGG_WORD[agg]
    disp = (lambda v: _pct(v)) if agg == "rate" else _num
    order = "lowest" if want_low else "highest"
    headline = f"{top_name} has the {order} {word} {target}: {disp(top['value'])} (n={int(top['n']):,})"
    if agg in ("sum", "count") and grand:
        headline += f", {_pct(top['value'] / grand)} of the overall {_num(grand)}"
    headline += "."
    is_breakdown_q = bool(_BREAKDOWN_RE.search(q))
    if is_breakdown_q and agg in ("sum", "count") and grand and not want_low:
        headline = f"{top_name} leads {group} with {_pct(top['value'] / grand)} of total {target} ({disp(top['value'])} of {_num(grand)})."
    details: List[str] = []
    ranked = [f"{k}: {disp(r['value'])} (n={int(r['n']):,})" for k, r in table.head(MAX_GROUPS_REPORTED).iterrows()]
    details.append(f"Ranking by {word} {target}: " + "; ".join(ranked) + ("; ..." if len(table) > MAX_GROUPS_REPORTED else "."))
    numbers: Dict[str, Any] = {"aggregation": agg, "top": str(top_name), "top_value": float(top["value"]),
                               "table": {str(k): {"value": float(r["value"]), "n": int(r["n"])} for k, r in table.iterrows()}}
    if agg in ("sum", "count") and grand and len(table) >= 3:
        table_desc = table.sort_values("value", ascending=False)
        cum_shares = (table_desc["value"] / grand).cumsum()
        p80_cnt = int((cum_shares <= 0.80).sum()) + 1
        p80_cnt = min(p80_cnt, len(table))
        p80_val = float(cum_shares.iloc[p80_cnt - 1])
        numbers["pareto_80_count"] = p80_cnt
        numbers["pareto_top_share"] = float(p80_val)
        numbers["pareto_concentration"] = float(p80_val)
        if p80_cnt <= max(1, len(table) // 2):
            details.append(
                f"Pareto concentration (80/20): {p80_cnt} of {len(table)} {group} account for "
                f"{_pct(p80_val)} of total {target} (high concentration)."
            )
    caveats = _hygiene(df, [group, target], len(sub)) + (_outlier_note(sub, group, target) if agg in ("sum", "mean", "count") and binary is None else [])
    next_steps: List[str] = []
    if len(table) >= 2:
        second = table.iloc[1]
        second_name = table.index[1]
        if agg in ("sum", "count") and top["n"] > 0 and second["n"] > 0:
            volume_ratio = top["n"] / second["n"]
            per_row_ratio = (top["sum"] / top["n"]) / (second["sum"] / second["n"]) if second["sum"] else float("nan")
            details.append(
                f"{top_name} vs runner-up {second_name}: {int(top['n']):,} vs {int(second['n']):,} rows, "
                f"average {_num(top['sum'] / top['n'])} vs {_num(second['sum'] / second['n'])} per row -- "
                + ("the lead comes mostly from volume (more rows), not higher value per row."
                   if volume_ratio > 1.2 and (per_row_ratio != per_row_ratio or per_row_ratio < 1.1)
                   else "the lead reflects higher value per row as well as volume."
                   if volume_ratio > 1.2 and per_row_ratio >= 1.1
                   else "the two are close on both volume and value per row."))
            numbers["volume_ratio_top_vs_second"] = float(volume_ratio)
            next_steps.append(f"Evaluate whether {top_name}'s lead is driven by customer volume vs unit economics.")
        if agg in ("mean", "rate") and min(top["n"], second["n"]) >= MIN_GROUP_N:
            a = sub[sub[group] == top_name][target].to_numpy(float)
            b = sub[sub[group] == second_name][target].to_numpy(float)
            if agg == "rate":
                diff, lo, hi = _rate_ci(int(a.sum()), len(a), int(b.sum()), len(b))
                verdict = "a real gap" if (lo > 0 or hi < 0) else "not a reliable gap"
                details.append(f"Top vs runner-up ({second_name}): {_signed(diff * 100)} percentage points (95% CI {_signed(lo * 100)} to {_signed(hi * 100)}) -- {verdict}.")
            else:
                w = _welch(a, b)
                verdict = "a real gap" if (w["lo"] > 0 or w["hi"] < 0) else "not a reliable gap"
                details.append(f"Top vs runner-up ({second_name}): {_signed(w['diff'])} (95% CI {_signed(w['lo'])} to {_signed(w['hi'])}, {_p(w['p'])}) -- {verdict}.")
                numbers["top_vs_second"] = w
            next_steps.append(f"Investigate operational practices distinguishing {top_name} from {second_name}.")
        small = [str(k) for k, r in table.iterrows() if r["n"] < MIN_GROUP_N]
        if small:
            caveats.append(f"Groups with fewer than {MIN_GROUP_N} rows ({', '.join(small[:5])}) are not reliable.")

    # Principal Data Analyst Intelligence Enrichments:
    conc_diag = _gini_and_concentration(table["value"])
    numbers["concentration_diagnostics"] = conc_diag
    numbers["gini_coefficient"] = conc_diag.get("gini_coefficient")
    numbers["hhi"] = conc_diag.get("hhi")

    dist_prof = _distribution_profile(sub[target])
    numbers["distribution_profile"] = dist_prof

    scenario_sens = None
    if len(table) >= 3 and grand:
        med_val = float(table["value"].median())
        below_med = table[table["value"] < med_val]
        if not below_med.empty:
            opp_lift = float((med_val - below_med["value"]).sum()) if agg in ("sum", "count") else float((med_val - below_med["value"]).mean())
            scenario_sens = {
                "opportunity_gap_to_median": round(opp_lift, 2),
                "median_benchmark": round(med_val, 2),
                "potential_lift_pct": round(float(opp_lift / grand * 100), 2) if grand else 0.0,
            }
            numbers["scenario_sensitivity"] = scenario_sens
            details.append(
                f"Opportunity sizing: elevating {len(below_med)} {group} segments currently below median ({_num(med_val)}) to the median benchmark "
                f"represents +{_num(opp_lift)} in potential {target} expansion ({_pct(opp_lift / grand)} overall lift)."
            )

    strat_playbook = _build_strategic_playbook("RANKING", "positive", str(top_name), numbers)
    numbers["strategic_playbook"] = strat_playbook
    exec_summary = f"{top_name} leads {group} with {_num(top['value'])} {target} ({_pct(top['value'] / grand)} share)." if (grand and agg in ("sum", "count")) else f"{top_name} has the {order} {word} {target} at {_num(top['value'])}."

    return AnalystResult(
        "RANKING", headline, details, caveats, numbers, descriptive=True, supersedes_loop=True, next_steps=next_steps,
        executive_summary=exec_summary,
        concentration_diagnostics=conc_diag,
        distribution_profile=dist_prof,
        scenario_sensitivity=scenario_sens,
        strategic_playbook=strat_playbook,
    )



# --------------------------------------------------------------------------- GROUP COMPARISON
def _group_comparison(q, df, target, group, agg_default) -> Optional[AnalystResult]:
    sub = df[[group, target]].copy()
    binary = _binary_values(df[target])
    if binary is not None:
        sub[target] = binary.loc[sub.index]
    else:
        sub[target] = _to_numeric(sub[target])
    sub = sub.dropna(subset=[group, target])
    sizes = sub.groupby(group)[target].count()
    keep = sizes[sizes >= 2].index
    sub = sub[sub[group].isin(keep)]
    levels = list(sub[group].unique())
    if len(levels) < 2:
        return None
    if binary is not None:
        return _rate_comparison(q, df, sub, target, group)
    return _numeric_comparison(q, df, sub, target, group)


def _root_cause(
    q: str,
    df: pd.DataFrame,
    target: str,
    group: str,
    expl: Optional[Sequence[str]] = None,
    default_aggregation: Optional[str] = None,
) -> Optional[AnalystResult]:
    """Dedicated ROOT_CAUSE investigation engine:
    1. Observed difference (exact pairwise or ANOVA statistics across groups).
    2. Candidate factor discovery (scan potential covariates in dataset).
    3. Factor <-> Outcome association (test outcome correlation/significance).
    4. Factor <-> Exposure/Group association (test group divergence).
    5. Adjusted exposure <-> outcome test (multiple regression / ANCOVA controlling for factor).
    6. Attenuation / persistence evaluation (measure effect change post-adjustment).
    7. Evidence classification (substantial attenuation vs persistence vs reversal).
    8. Observational claim ceiling (explicit limits on mechanistic causality).
    """
    res = _group_comparison(q, df, target, group, default_aggregation)
    if res is None:
        return None
    res.kind = "ROOT_CAUSE"

    # 1. Candidate Factor Discovery
    def _is_id_col(col_name: str, s: pd.Series) -> bool:
        c_low = str(col_name).lower()
        if c_low in ("id", "key", "uuid", "guid", "pk", "fk") or c_low.endswith(("_id", "_key", "_uuid", "_pk", "_fk")):
            return True
        if not _is_numeric_like(s) and s.nunique(dropna=True) >= len(s) * 0.9:
            return True
        return False

    def _is_surrogate(col: str, ref_col: str, d_frame: pd.DataFrame) -> bool:
        if col == ref_col:
            return True
        c_low = str(col).lower().strip()
        r_low = str(ref_col).lower().strip()
        if c_low == r_low:
            return True

        SURROGATE_PAIRS = [
            {"class", "pclass", "passenger_class", "ticket_class"},
            {"survived", "alive", "survival", "survived_flag"},
            {"embarked", "embark_town", "port"},
        ]
        for pair in SURROGATE_PAIRS:
            if c_low in pair and r_low in pair:
                return True

        c_clean = c_low.replace("_", "").replace("-", "")
        r_clean = r_low.replace("_", "").replace("-", "")
        if c_clean == r_clean:
            return True

        try:
            sub = d_frame[[col, ref_col]].dropna()
            if len(sub) >= 10:
                if _is_numeric_like(sub[col]) and _is_numeric_like(sub[ref_col]):
                    s_c = pd.to_numeric(sub[col], errors="coerce").dropna()
                    s_r = pd.to_numeric(sub[ref_col], errors="coerce").dropna()
                    valid_idx = s_c.index.intersection(s_r.index)
                    if len(valid_idx) >= 10:
                        r_val, _ = sps.pearsonr(s_c.loc[valid_idx], s_r.loc[valid_idx])
                        if abs(r_val) > 0.99999:
                            return True
                n_c = sub[col].nunique()
                n_r = sub[ref_col].nunique()
                if 1 < n_c <= 20 and 1 < n_r <= 20 and n_c == n_r:
                    map_fwd = (sub.groupby(ref_col, observed=True)[col].nunique() <= 1).all()
                    map_bwd = (sub.groupby(col, observed=True)[ref_col].nunique() <= 1).all()
                    if map_fwd and map_bwd:
                        return True
        except Exception:
            pass
        return False

    candidate_cols = [
        c for c in df.columns
        if c not in (target, group)
        and (expl is None or c not in expl)
        and df[c].nunique(dropna=True) > 1
        and not _is_id_col(c, df[c])
        and not _is_surrogate(c, target, df)
        and not _is_surrogate(c, group, df)
    ]

    confounder_notes = []
    if not candidate_cols:
        confounder_notes.append("Candidate explanatory factors scanned: No additional candidate covariates found in the dataset.")
        confounder_notes.append(
            "Evidence boundary & claim ceiling: Observational data supports statistical association up to the association claim ceiling, "
            "but cannot establish mechanistic causality without controlled experimental intervention."
        )
        res.details.extend(confounder_notes)
        return res

    scanned_list = ", ".join(candidate_cols[:6])
    if len(candidate_cols) > 6:
        scanned_list += f" (+{len(candidate_cols) - 6} more)"
    confounder_notes.append(f"Candidate explanatory factors scanned: {scanned_list}.")

    # Identify focal groups (g_A vs g_B)
    group_means = df.groupby(group, observed=True)[target].mean().dropna().to_dict()
    if len(group_means) < 2:
        confounder_notes.append(
            "Evidence boundary & claim ceiling: Observational data supports statistical association up to the association claim ceiling, "
            "but cannot establish mechanistic causality without controlled experimental intervention."
        )
        res.details.extend(confounder_notes)
        return res

    sorted_groups = sorted(group_means.items(), key=lambda x: x[1], reverse=True)
    g_hi, val_hi = sorted_groups[0]
    g_lo, val_lo = sorted_groups[-1]

    # Check if specific groups were asked in the question (e.g. "Fair" and "Ideal", or "third" and "first")
    g_A, g_B = g_hi, g_lo
    q_lower = q.lower()
    matched_groups = [g for g in group_means.keys() if str(g).lower() in q_lower]
    if len(matched_groups) >= 2:
        g_A, g_B = matched_groups[0], matched_groups[1]
    elif len(matched_groups) == 1:
        matched = matched_groups[0]
        if matched == g_lo:
            g_A, g_B = g_hi, g_lo
        else:
            g_A, g_B = matched, g_lo

    # 2. Screen Candidates: Factor <-> Outcome AND Factor <-> Group (supporting both numeric & categorical)
    qualified_candidates = []
    for c in candidate_cols:
        valid = df[[c, target, group]].dropna()
        if len(valid) < 10 or valid[c].nunique() < 2:
            continue
        is_num = _is_numeric_like(valid[c])
        if not is_num and (valid[c].nunique() > 20 or valid[c].nunique() < 2):
            continue
        s_y = pd.to_numeric(valid[target], errors="coerce")
        valid_y = valid.assign(_y=s_y).dropna(subset=["_y"])
        if len(valid_y) < 10:
            continue
        if is_num:
            s_c = pd.to_numeric(valid_y[c], errors="coerce")
            clean_df = valid_y.assign(_c=s_c).dropna(subset=["_c"])
            if len(clean_df) < 10 or clean_df["_c"].nunique() < 2:
                continue
            try:
                r_val, p_r = sps.pearsonr(clean_df["_c"], clean_df["_y"])
            except Exception:
                continue
            if math.isnan(r_val):
                continue
            assoc_val = abs(r_val)
            p_assoc = float(p_r)
        else:
            clean_df = valid_y.copy()
            cat_groups = [grp["_y"].values for _, grp in clean_df.groupby(c, observed=True) if len(grp) > 0]
            if len(cat_groups) < 2:
                continue
            try:
                f_stat, p_f = sps.f_oneway(*cat_groups)
            except Exception:
                continue
            if math.isnan(f_stat) or math.isnan(p_f):
                continue
            grand_mean = clean_df["_y"].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in cat_groups)
            ss_total = ((clean_df["_y"] - grand_mean)**2).sum()
            eta2 = (ss_between / ss_total) if ss_total > 0 else 0.0
            assoc_val = math.sqrt(max(0.0, min(1.0, eta2)))
            r_val = assoc_val
            p_assoc = float(p_f)

        if assoc_val < 0.10:
            continue

        focal_valid = clean_df[clean_df[group].isin([g_A, g_B])].copy()
        if len(focal_valid) < 6 or focal_valid[group].nunique() != 2:
            continue

        if is_num:
            z_means = focal_valid.groupby(group, observed=True)["_c"].mean().to_dict()
            z_std = float(focal_valid["_c"].std())
            if z_std <= 0:
                continue
            z_diff = abs(z_means.get(g_A, 0.0) - z_means.get(g_B, 0.0))
            z_d = z_diff / z_std
            group_diff_metric = z_d
            summary_info = {"type": "numeric", "z_hi": z_means.get(g_A, 0.0), "z_lo": z_means.get(g_B, 0.0)}
        else:
            props_A = focal_valid[focal_valid[group] == g_A][c].value_counts(normalize=True)
            props_B = focal_valid[focal_valid[group] == g_B][c].value_counts(normalize=True)
            all_lvls = set(props_A.index) | set(props_B.index)
            tvd = 0.5 * sum(abs(props_A.get(k, 0.0) - props_B.get(k, 0.0)) for k in all_lvls)
            group_diff_metric = 2.0 * tvd
            shifts = {k: abs(props_A.get(k, 0.0) - props_B.get(k, 0.0)) for k in all_lvls}
            top_level = max(shifts.items(), key=lambda x: x[1])[0]
            summary_info = {
                "type": "categorical",
                "top_level": str(top_level),
                "prop_A": float(props_A.get(top_level, 0.0)),
                "prop_B": float(props_B.get(top_level, 0.0)),
                "tvd": float(tvd),
            }

        if group_diff_metric < 0.10:
            continue

        score = assoc_val * (1.0 + min(2.0, group_diff_metric))
        qualified_candidates.append({
            "col": c,
            "is_num": is_num,
            "r": float(r_val),
            "p_r": float(p_assoc),
            "assoc_val": float(assoc_val),
            "score": float(score),
            "summary_info": summary_info,
            "focal_df": focal_valid,
        })

    if not qualified_candidates:
        confounder_notes.append(
            f"Candidate factor screening: None of the scanned factors met the dual criteria for candidate confounders "
            f"(requiring both substantial correlation with {target} and systematic variation across {group})."
        )
        confounder_notes.append(
            "Evidence boundary & claim ceiling: Observational data supports statistical association up to the association claim ceiling, "
            "but cannot establish mechanistic causality without controlled experimental intervention."
        )
        res.details.extend(confounder_notes)
        return res

    qualified_candidates.sort(key=lambda x: x["score"], reverse=True)
    top_cand_info = qualified_candidates[0]
    cand_name = top_cand_info["col"]
    top_r = top_cand_info["r"]
    p_r = top_cand_info["p_r"]
    summary = top_cand_info["summary_info"]
    is_num = top_cand_info["is_num"]
    focal_df = top_cand_info["focal_df"]

    # 3. Adjusted Exposure <-> Outcome Test
    if len(focal_df) >= 6 and focal_df[group].nunique() == 2:
        y_A = focal_df[focal_df[group] == g_A]["_y"].values
        y_B = focal_df[focal_df[group] == g_B]["_y"].values
        unadj_diff = float(np.mean(y_A) - np.mean(y_B))

        D = (focal_df[group] == g_A).astype(float).values
        Y = focal_df["_y"].values
        if is_num:
            Z = focal_df[["_c"]].values.astype(float)
        else:
            Z = pd.get_dummies(focal_df[cand_name], drop_first=True, dtype=float).values
        X = np.column_stack([np.ones(len(focal_df)), D, Z])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            adj_diff = float(beta[1])
        except Exception:
            adj_diff = unadj_diff
    else:
        unadj_diff = float(group_means.get(g_A, 0.0) - group_means.get(g_B, 0.0))
        adj_diff = unadj_diff

    sign_flipped = bool(unadj_diff * adj_diff < 0 and abs(unadj_diff) > 1e-6 and abs(adj_diff) > 1e-6)
    if abs(unadj_diff) > 1e-9:
        attenuation_pct = float((1.0 - abs(adj_diff) / abs(unadj_diff)) * 100.0)
    else:
        attenuation_pct = 0.0

    res.numbers["candidate_factors_scanned"] = candidate_cols
    res.numbers["top_candidate"] = cand_name
    res.numbers["top_candidate_r"] = float(top_r)
    res.numbers["unadjusted_diff"] = float(unadj_diff)
    res.numbers["adjusted_diff"] = float(adj_diff)
    res.numbers["attenuation_pct"] = float(attenuation_pct)

    if is_num:
        z_hi = summary["z_hi"]
        z_lo = summary["z_lo"]
        cand_desc = (
            f"{cand_name} is a candidate confounder: it correlates with {target} (r={top_r:.2f}) "
            f"and differs across {group} ({g_A} averages {_num(z_hi)} vs {_num(z_lo)} for {g_B})."
        )
        eval_desc = f"{cand_name} was evaluated as a candidate factor (r={top_r:.2f})"
    else:
        prop_A = summary["prop_A"]
        prop_B = summary["prop_B"]
        lvl = summary["top_level"]
        cand_desc = (
            f"{cand_name} is a candidate confounder: it is associated with {target} (eta={top_r:.2f}) "
            f"and differs across {group} ({g_A} is {_pct(prop_A)} {lvl} vs {_pct(prop_B)} for {g_B})."
        )
        eval_desc = f"{cand_name} was evaluated as a candidate factor (eta={top_r:.2f})"

    if sign_flipped:
        classification = "CONFOUNDER_EFFECT_REVERSAL"
        res.numbers["confounding_classification"] = classification
        confounder_notes.append(
            f"Confounder adjustment & effect reversal: {cand_desc} "
            f"When adjusting for {cand_name}, the observed difference between {g_A} and {g_B} reverses direction "
            f"(adjusted difference {_signed(adj_diff)} vs unadjusted {_signed(unadj_diff)}), "
            f"indicating that the unadjusted {group} difference was an artifact of composition/confounding by {cand_name}."
        )
    elif attenuation_pct >= 30.0:
        classification = "CANDIDATE_CONFOUNDER_SUBSTANTIAL_ATTENUATION"
        res.numbers["confounding_classification"] = classification
        confounder_notes.append(
            f"Confounder adjustment & attenuation: {cand_desc} "
            f"After adjusting for {cand_name}, the observed difference between {g_A} and {g_B} attenuates by {attenuation_pct:.1f}% "
            f"(from {_signed(unadj_diff)} to {_signed(adj_diff)}). The observed data are consistent with confounding by {cand_name} "
            f"rather than an effect of {group} in isolation."
        )
    elif attenuation_pct >= 10.0:
        classification = "CANDIDATE_CONFOUNDER_PARTIAL_ATTENUATION"
        res.numbers["confounding_classification"] = classification
        confounder_notes.append(
            f"Confounder adjustment & partial attenuation: {cand_desc} "
            f"Adjusting for {cand_name} yields a modest {attenuation_pct:.1f}% attenuation "
            f"(from {_signed(unadj_diff)} to {_signed(adj_diff)}), but substantial group differences persist independently of {cand_name}."
        )
    else:
        classification = "PERSISTENT_GROUP_DIFFERENCE"
        res.numbers["confounding_classification"] = classification
        persistence_pct = max(0.0, 100.0 - attenuation_pct)
        confounder_notes.append(
            f"Confounder adjustment & persistent difference: {eval_desc}, "
            f"but adjusting for {cand_name} does not attenuate the observed difference "
            f"(adjusted difference {_signed(adj_diff)} vs unadjusted {_signed(unadj_diff)}, {persistence_pct:.1f}% persistence). "
            f"The observed difference persists independently of {cand_name}."
        )

    confounder_notes.append(
        "Evidence boundary & claim ceiling: Statistical adjustment indicates whether observational differences are consistent with confounding, "
        "but cannot establish mechanistic causality without controlled experimental intervention or longitudinal identification."
    )

    res.details.extend(confounder_notes)
    return res


def _power_two_means(n1: int, n2: int, d_mde: float = 0.5, alpha: float = ALPHA) -> float:
    """Calculate statistical power for two-sample difference of means (medium effect d=0.5)."""
    if n1 <= 1 or n2 <= 1:
        return 0.0
    n_harm = (2.0 * n1 * n2) / (n1 + n2)
    delta = d_mde * math.sqrt(n_harm / 2.0)
    z_crit = 1.96 if alpha == 0.05 else float(sps.norm.ppf(1.0 - alpha / 2.0))
    power = float(1.0 - sps.norm.cdf(z_crit - delta) + sps.norm.cdf(-z_crit - delta))
    return max(0.0, min(1.0, power))


def _power_two_proportions(n1: int, n2: int, h_mde: float = 0.35, alpha: float = ALPHA) -> float:
    """Calculate statistical power for two proportions comparison (Cohen's h effect size)."""
    if n1 <= 1 or n2 <= 1:
        return 0.0
    n_harm = (2.0 * n1 * n2) / (n1 + n2)
    delta = h_mde * math.sqrt(n_harm / 2.0)
    z_crit = 1.96 if alpha == 0.05 else float(sps.norm.ppf(1.0 - alpha / 2.0))
    power = float(1.0 - sps.norm.cdf(z_crit - delta) + sps.norm.cdf(-z_crit - delta))
    return max(0.0, min(1.0, power))


def _power_correlation(n: int, r_mde: float = 0.30, alpha: float = ALPHA) -> float:
    """Calculate statistical power for Pearson correlation using Fisher's z transform."""
    if n <= 3:
        return 0.0
    z_r = 0.5 * math.log((1.0 + r_mde) / max(1e-9, 1.0 - r_mde))
    se = 1.0 / math.sqrt(n - 3)
    z_crit = 1.96 if alpha == 0.05 else float(sps.norm.ppf(1.0 - alpha / 2.0))
    delta = z_r / se
    power = float(1.0 - sps.norm.cdf(z_crit - delta) + sps.norm.cdf(-z_crit - delta))
    return max(0.0, min(1.0, power))


def _power_anova(n_total: int, k_groups: int, f_mde: float = 0.35, alpha: float = ALPHA) -> float:
    """Calculate statistical power for one-way ANOVA at practical business effect size f (default 0.35)."""
    if k_groups <= 1 or n_total <= k_groups:
        return 0.0
    df1 = k_groups - 1
    df2 = n_total - k_groups
    lambda_param = n_total * (f_mde ** 2)
    try:
        f_crit = sps.f.ppf(1.0 - alpha, df1, df2)
        power = float(1.0 - sps.ncf.cdf(f_crit, df1, df2, lambda_param))
    except Exception:
        delta = math.sqrt(lambda_param)
        power = float(1.0 - sps.norm.cdf(1.96 - delta))
    return max(0.0, min(1.0, power))


def _power_chi2(n_total: int, k_levels: int, w_mde: float = 0.32, alpha: float = ALPHA) -> float:
    """Calculate statistical power for Chi-square contingency test (default w=0.32)."""
    if k_levels <= 1 or n_total <= 0:
        return 0.0
    df = k_levels - 1
    lambda_param = n_total * (w_mde ** 2)
    try:
        crit = sps.chi2.ppf(1.0 - alpha, df)
        power = float(1.0 - sps.ncx2.cdf(crit, df, lambda_param))
    except Exception:
        delta = math.sqrt(lambda_param)
        power = float(1.0 - sps.norm.cdf(1.96 - delta))
    return max(0.0, min(1.0, power))



def _subject_reference(q: str, levels: List[Any], means: pd.Series) -> Tuple[Any, Any]:
    named = _mentioned_levels(q, levels)
    if len(named) >= 2:
        return named[0][1], named[1][1]
    if len(levels) == 2:
        if len(named) == 1:
            subj = named[0][1]
            return subj, [x for x in levels if x != subj][0]
        ordered = sorted(levels, key=lambda x: str(x))
        return ordered[0], ordered[1]
    return means.idxmax(), means.idxmin()


def _numeric_comparison(q, df, sub, target, group) -> Optional[AnalystResult]:
    levels = list(sub[group].unique())
    grp = sub.groupby(group)[target]
    means = grp.mean()
    ns = grp.count()
    small = [str(k) for k, v in ns.items() if v < MIN_GROUP_N]
    caveats = _hygiene(df, [group, target], len(sub)) + _outlier_note(sub, group, target)
    if small:
        caveats.append(f"Groups with fewer than {MIN_GROUP_N} rows ({', '.join(small[:5])}) are not reliable.")
    numbers: Dict[str, Any] = {"means": {str(k): float(v) for k, v in means.items()}, "n": {str(k): int(v) for k, v in ns.items()}}
    arrays = [sub.loc[sub[group] == lv, target].to_numpy(float) for lv in levels]
    if any(len(a) < 2 for a in arrays):
        return None

    subj, ref = _subject_reference(q, levels, means)
    a = sub.loc[sub[group] == subj, target].to_numpy(float)
    b = sub.loc[sub[group] == ref, target].to_numpy(float)
    finding = "none"

    if len(levels) == 2 or len(_mentioned_levels(q, levels)) >= 2:
        w = _welch(a, b)
        try:
            mw_p = float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except Exception:
            mw_p = float("nan")
        sig = w["lo"] > 0 or w["hi"] < 0
        higher_q, lower_q = bool(_HIGHER_RE.search(q)), bool(_LOWER_RE.search(q))
        direction = "higher" if w["diff"] > 0 else "lower"
        if sig:
            finding = "positive" if w["diff"] > 0 else "negative"
            opener = ""
            if higher_q and not lower_q:
                opener = "Yes -- " if w["diff"] > 0 else "No, the opposite -- "
            elif lower_q and not higher_q:
                opener = "Yes -- " if w["diff"] < 0 else "No, the opposite -- "
            headline = (f"{opener}{subj} averages {_num(a.mean())} {target} vs {_num(b.mean())} for {ref}: "
                        f"{_num(abs(w['diff']))} {direction} (95% CI {_signed(w['lo'])} to {_signed(w['hi'])}, {_p(w['p'])}; "
                        f"effect size d={w['d']:.2f}, {_d_label(w['d'])}).")
        else:
            opener = "No -- " if (higher_q or lower_q) else ""
            headline = (f"{opener}no statistically detectable difference in average {target} between {subj} ({_num(a.mean())}, n={len(a):,}) "
                        f"and {ref} ({_num(b.mean())}, n={len(b):,}): difference {_signed(w['diff'])} (95% CI {_signed(w['lo'])} to {_signed(w['hi'])}, {_p(w['p'])}).")
        details = []
        calc_power = _power_two_means(len(a), len(b), d_mde=0.5)
        is_powered = calc_power >= 0.80 and min(len(a), len(b)) >= 10
        numbers.update({"subject": str(subj), "reference": str(ref), "diff": w["diff"], "ci_low": w["lo"], "ci_high": w["hi"],
                        "p_value": w["p"], "cohens_d": w["d"], "mannwhitney_p": mw_p,
                        "power": round(float(calc_power), 3), "adequate_power": is_powered,
                        "equivalence_bound": float(max(abs(w["lo"]), abs(w["hi"])))})
        if not sig:
            details.append(f"The interval bounds what the data can rule out: a true difference larger than about {_num(max(abs(w['lo']), abs(w['hi'])))} is unlikely.")
            if is_powered:
                details.append(f"Statistical power is sufficient ({_pct(calc_power, 0)} power at MDE d=0.5; n={len(a):,} vs {len(b):,}) to conclude that any true difference is bounded within the 95% confidence interval.")

        # Distribution skewness check
        if len(a) >= 20 and float(np.median(a)) > 0:
            skew_a = float(a.mean() / np.median(a))
            if skew_a >= 1.4 or skew_a <= 0.7:
                p90_a = float(np.percentile(a, 90))
                caveats.append(f"Skewness: {subj} values are skewed (mean {_num(a.mean())} vs median {_num(float(np.median(a)))}, P90 {_num(p90_a)}); median reflects typical outcome better than mean.")
                numbers["skewed_distribution"] = True
        # Outlier robustness: does the conclusion survive a rank-based test / trimmed means?
        ta, tb = sps.trim_mean(a, 0.05), sps.trim_mean(b, 0.05)
        med_a, med_b = float(np.median(a)), float(np.median(b))
        if sig and ((not math.isnan(mw_p) and mw_p >= ALPHA) or np.sign(ta - tb) != np.sign(w["diff"])):
            caveats.append(
                f"NOT ROBUST: the difference in means is not supported by a rank-based test ({_p(mw_p)}) or 5%-trimmed means "
                f"({_num(ta)} vs {_num(tb)}; medians {_num(med_a)} vs {_num(med_b)}) -- a few extreme values are driving it. Inspect the outliers before reporting.")
            numbers["robust"] = False
            finding = "none"
        elif not sig and not math.isnan(mw_p) and mw_p < ALPHA:
            caveats.append(f"Means are not distinguishable, but a rank-based test suggests the distributions differ ({_p(mw_p)}); medians {_num(med_a)} vs {_num(med_b)}.")
            numbers["robust"] = None
        else:
            numbers["robust"] = True
        next_steps: List[str] = []
        if sig and numbers.get("robust") is not False:
            next_steps.append(f"Investigate operational or business practices driving the superior {target} of {subj} over {ref}.")
        elif not sig and is_powered:
            next_steps.append(f"The two groups appear functionally equivalent; evaluate secondary segmentations or confounders if differences were expected.")
        elif not sig and not is_powered:
            next_steps.append(f"Expand sample size to improve statistical power (current groups have {min(len(a), len(b))} rows).")

        # Principal Data Analyst Intelligence Enrichments:
        if len(a) > 0 and len(b) > 0:
            try:
                u_val = float(sps.mannwhitneyu(a, b, alternative="greater").statistic) if not math.isnan(mw_p) else 0.0
                cles = u_val / (len(a) * len(b)) if len(a) * len(b) > 0 else 0.5
                numbers["common_language_effect_size"] = round(cles, 3)
                if abs(cles - 0.5) >= 0.08:
                    details.append(f"Common Language Effect Size (probability that a random {subj} exceeds a random {ref}): {_pct(cles, 0)}.")
            except Exception:
                pass

        prof_a = _distribution_profile(pd.Series(a))
        prof_b = _distribution_profile(pd.Series(b))
        dist_profile = {"subject": prof_a, "reference": prof_b}
        numbers["distribution_profile"] = dist_profile

        cf_gap = float((a.mean() - b.mean()) * len(b))
        scenario_sens = {"counterfactual_gap_closing_delta": round(cf_gap, 2)}
        numbers["scenario_sensitivity"] = scenario_sens
        if sig and abs(cf_gap) > 0:
            details.append(f"Counterfactual opportunity: if {ref} matched {subj}'s average performance, total {target} would expand by +{_num(abs(cf_gap))}.")

        strat_playbook = _build_strategic_playbook("GROUP_COMPARISON", finding, str(subj), numbers)
        numbers["strategic_playbook"] = strat_playbook
        exec_summary = f"{subj} averages {_num(a.mean())} {target} vs {_num(b.mean())} for {ref} ({_signed(w['diff'])} difference, {_p(w['p'])})."

        return AnalystResult(
            "GROUP_COMPARISON", headline, details, caveats, numbers, finding=finding, next_steps=next_steps,
            executive_summary=exec_summary,
            distribution_profile=dist_profile,
            scenario_sensitivity=scenario_sens,
            strategic_playbook=strat_playbook,
        )

    # k > 2 groups
    try:
        f_stat, f_p = sps.f_oneway(*arrays)
        kw_p = float(sps.kruskal(*arrays).pvalue)
    except Exception:
        return None
    grand = sub[target].mean()
    ss_between = float(sum(len(x) * (x.mean() - grand) ** 2 for x in arrays))
    ss_total = float(((sub[target] - grand) ** 2).sum())
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    hi_name, lo_name = means.idxmax(), means.idxmin()
    order = means.sort_values(ascending=False)
    rng_txt = "; ".join(f"{k}: {_num(v)} (n={int(ns[k]):,})" for k, v in order.head(MAX_GROUPS_REPORTED).items())
    sig = f_p < ALPHA
    calc_power = _power_anova(len(sub), len(levels))
    is_powered = calc_power >= 0.80 and len(sub) >= 40 and int(ns.min()) >= 10
    numbers.update({"anova_f": float(f_stat), "anova_p": float(f_p), "kruskal_p": kw_p, "eta_squared": eta2,
                    "highest": str(hi_name), "lowest": str(lo_name),
                    "power": round(float(calc_power), 3), "adequate_power": is_powered})
    next_steps = []
    if sig:
        finding = "positive"
        rest = sub.loc[sub[group] != hi_name, target].to_numpy(float)
        w_top = _welch(sub.loc[sub[group] == hi_name, target].to_numpy(float), rest)
        headline = (f"Average {target} differs across {group} ({_p(float(f_p))}; {_pct(eta2)} of the variation in {target} is explained by {group}, "
                    f"{'a small' if eta2 < 0.06 else 'a moderate' if eta2 < 0.14 else 'a large'} share). "
                    f"Highest: {hi_name} ({_num(means[hi_name])}); lowest: {lo_name} ({_num(means[lo_name])}).")
        details = [f"By group -- {rng_txt}.",
                   f"{hi_name} vs all other {group} values combined: {_signed(w_top['diff'])} (95% CI {_signed(w_top['lo'])} to {_signed(w_top['hi'])}, {_p(w_top['p'])})."]
        numbers.update({"top_vs_rest_diff": w_top["diff"], "top_vs_rest_ci": [w_top["lo"], w_top["hi"]]})
        if kw_p >= ALPHA:
            caveats.append(f"NOT ROBUST: a rank-based test does not confirm the difference ({_p(kw_p)}); check for outliers.")
            numbers["robust"] = False
            finding = "none"
        else:
            numbers["robust"] = True
        next_steps.append(f"Investigate high-performing {hi_name} and underperforming {lo_name} for operational drivers.")
    else:
        headline = (f"No statistically detectable difference in average {target} across {group} ({_p(float(f_p))}; {group} explains {_pct(eta2)} of the variation). "
                    f"Group averages range from {_num(means[lo_name])} ({lo_name}) to {_num(means[hi_name])} ({hi_name}).")
        details = [f"By group -- {rng_txt}."]
        big = sub.loc[sub[group] == hi_name, target].to_numpy(float)
        small_ = sub.loc[sub[group] == lo_name, target].to_numpy(float)
        if len(big) >= 2 and len(small_) >= 2:
            w = _welch(big, small_)
            details.append(f"Even the most extreme pair ({hi_name} vs {lo_name}) differs by {_signed(w['diff'])} (95% CI {_signed(w['lo'])} to {_signed(w['hi'])}, unadjusted) -- the data cannot rule out differences up to about that size.")
        next_steps.append(f"Evaluate whether segmenting by another dimension reveals localized differences.")

    dist_profile = _distribution_profile(sub[target])
    numbers["distribution_profile"] = dist_profile
    conc_diag = _gini_and_concentration(means)
    numbers["concentration_diagnostics"] = conc_diag
    strat_playbook = _build_strategic_playbook("GROUP_COMPARISON", finding, str(hi_name), numbers)
    numbers["strategic_playbook"] = strat_playbook
    exec_summary = f"Average {target} differs across {group} (highest: {hi_name})." if sig else f"No statistically detectable difference in average {target} across {group}."

    return AnalystResult(
        "GROUP_COMPARISON", headline, details, caveats, numbers, finding=finding, next_steps=next_steps,
        executive_summary=exec_summary,
        distribution_profile=dist_profile,
        concentration_diagnostics=conc_diag,
        strategic_playbook=strat_playbook,
    )


def _rate_comparison(q, df, sub, target, group) -> Optional[AnalystResult]:
    levels = list(sub[group].unique())
    grp = sub.groupby(group)[target]
    rates = grp.mean()
    ns = grp.count()
    xs = grp.sum()
    caveats = _hygiene(df, [group, target], len(sub))
    small = [str(k) for k, v in ns.items() if v < MIN_GROUP_N]
    if small:
        caveats.append(f"Groups with fewer than {MIN_GROUP_N} rows ({', '.join(small[:5])}) are not reliable.")
    numbers: Dict[str, Any] = {"rates": {str(k): float(v) for k, v in rates.items()}, "n": {str(k): int(v) for k, v in ns.items()},
                               "events": {str(k): int(v) for k, v in xs.items()}}
    named = _mentioned_levels(q, levels)
    two = len(levels) == 2 or len(named) >= 2
    finding = "none"
    if two:
        subj, ref = _subject_reference(q, levels, rates)
        x1, n1, x2, n2 = int(xs[subj]), int(ns[subj]), int(xs[ref]), int(ns[ref])
        diff, lo, hi = _rate_ci(x1, n1, x2, n2)
        table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
        if table.min() < 5 or n1 < 30 or n2 < 30:
            p = float(sps.fisher_exact(table)[1])
            test = "Fisher exact"
        else:
            p = float(sps.chi2_contingency(table, correction=False)[1])
            test = "chi-square"
        r1, r2 = x1 / n1, x2 / n2
        rel = (r1 / r2 - 1) if r2 > 0 else float("nan")
        sig = lo > 0 or hi < 0
        higher_q, lower_q = bool(_HIGHER_RE.search(q)), bool(_LOWER_RE.search(q))
        if sig:
            finding = "positive" if diff > 0 else "negative"
            opener = ""
            if higher_q and not lower_q:
                opener = "Yes -- " if diff > 0 else "No, the opposite -- "
            elif lower_q and not higher_q:
                opener = "Yes -- " if diff < 0 else "No, the opposite -- "
            headline = (f"{opener}{target} rate is {_pct(r1)} for {subj} (n={n1:,}) vs {_pct(r2)} for {ref} (n={n2:,}): "
                        f"{_signed(diff * 100)} percentage points"
                        + (f" ({_signed(rel * 100)}% relative)" if rel == rel else "")
                        + f", 95% CI {_signed(lo * 100)} to {_signed(hi * 100)} pp, {test} {_p(p)}.")
            details = []
        else:
            opener = "No -- " if (higher_q or lower_q) else ""
            headline = (f"{opener}no statistically detectable difference in {target} rate between {subj} ({_pct(r1)}, n={n1:,}) and {ref} ({_pct(r2)}, n={n2:,}): "
                        f"{_signed(diff * 100)} pp (95% CI {_signed(lo * 100)} to {_signed(hi * 100)} pp, {test} {_p(p)}).")
            details = [f"The interval bounds what the data can rule out: a true gap larger than about {_num(max(abs(lo), abs(hi)) * 100)} percentage points is unlikely at this sample size."]
        calc_power = _power_two_proportions(n1, n2, h_mde=0.35)
        is_powered = calc_power >= 0.80 and min(n1, n2) >= 15
        numbers.update({"subject": str(subj), "reference": str(ref), "rate_diff": diff, "ci_low": lo, "ci_high": hi, "p_value": p, "test": test,
                        "relative_uplift_pct": float(rel * 100) if rel == rel else None,
                        "power": round(float(calc_power), 3), "adequate_power": is_powered,
                        "equivalence_bound_pp": float(max(abs(lo), abs(hi)) * 100)})
        if not sig and is_powered:
            details.append(f"Statistical power is sufficient (n={n1:,} vs {n2:,}) to conclude that any true rate difference is bounded within {_num(max(abs(lo), abs(hi)) * 100)} percentage points.")
        if x1 + x2 < 10:
            caveats.append(f"Only {x1 + x2} total events -- the estimate is unstable.")
        next_steps = []
        if sig:
            next_steps.append(f"Evaluate whether practical uplift of {_signed(diff * 100)} pp justifies rolling out {subj}.")
        elif is_powered:
            next_steps.append(f"The variants show no detectable difference in {target} rate; prioritize other factors such as latency, maintenance, or user experience.")
        else:
            next_steps.append(f"Continue experiment to gather more observations (current sample sizes {n1:,} and {n2:,}).")

        # Principal Data Analyst Intelligence Enrichments:
        cf_events = float(diff * n2)
        scenario_sens = {
            "counterfactual_event_delta": round(cf_events, 2),
            "relative_uplift_pct": round(float(rel * 100), 2) if rel == rel else 0.0,
        }
        numbers["scenario_sensitivity"] = scenario_sens
        if sig and abs(cf_events) > 0:
            details.append(
                f"Counterfactual opportunity: if {ref} matched {subj}'s {target} rate ({_pct(r1)}), "
                f"it would generate {_signed(cf_events)} additional events across its {n2:,} baseline observations."
            )

        strat_playbook = _build_strategic_playbook("GROUP_COMPARISON", finding, str(subj), numbers)
        numbers["strategic_playbook"] = strat_playbook
        exec_summary = f"{target} rate is {_pct(r1)} for {subj} vs {_pct(r2)} for {ref} ({_signed(diff * 100)} pp, {test} {_p(p)})."

        return AnalystResult(
            "RATE_COMPARISON", headline, details, caveats, numbers, finding=finding, next_steps=next_steps,
            executive_summary=exec_summary,
            scenario_sensitivity=scenario_sens,
            strategic_playbook=strat_playbook,
        )

    table = np.array([[int(xs[k]), int(ns[k] - xs[k])] for k in levels])
    try:
        chi2, p, _dof, _exp = sps.chi2_contingency(table, correction=False)
    except Exception:
        return None
    order = rates.sort_values(ascending=False)
    hi_name, lo_name = order.index[0], order.index[-1]
    n_all = int(ns.sum())
    cramers_v = math.sqrt(chi2 / (n_all * (min(table.shape) - 1))) if n_all else 0.0
    rng_txt = "; ".join(f"{k}: {_pct(v)} (n={int(ns[k]):,})" for k, v in order.head(MAX_GROUPS_REPORTED).items())
    calc_power = _power_chi2(n_all, len(levels))
    is_powered = calc_power >= 0.80 and n_all >= 60 and int(ns.min()) >= 10
    numbers.update({"chi2_p": float(p), "cramers_v": cramers_v, "highest": str(hi_name), "lowest": str(lo_name),
                    "power": round(float(calc_power), 3), "adequate_power": is_powered})
    next_steps = []
    if p < ALPHA:
        finding = "positive"
        headline = (f"{target} rate differs across {group} ({_p(float(p))}; Cramer's V={cramers_v:.2f}). Highest: {hi_name} ({_pct(rates[hi_name])}); lowest: {lo_name} ({_pct(rates[lo_name])}).")
        next_steps.append(f"Investigate high-converting {hi_name} compared to low-converting {lo_name}.")
    else:
        headline = (f"No statistically detectable difference in {target} rate across {group} ({_p(float(p))}); rates range from {_pct(rates[lo_name])} ({lo_name}) to {_pct(rates[hi_name])} ({hi_name}).")
        next_steps.append(f"No overall difference in {target} rate detected across {group}; consider evaluating customer cohorts.")
    strat_playbook = _build_strategic_playbook("GROUP_COMPARISON", finding, str(hi_name), numbers)
    numbers["strategic_playbook"] = strat_playbook
    exec_summary = f"{target} rate differs across {group} (highest: {hi_name} at {_pct(rates[hi_name])})." if p < ALPHA else f"No statistically detectable difference in {target} rate across {group}."

    return AnalystResult(
        "RATE_COMPARISON", headline, [f"By group -- {rng_txt}."], caveats, numbers, finding=finding, next_steps=next_steps,
        executive_summary=exec_summary,
        strategic_playbook=strat_playbook,
    )


# --------------------------------------------------------------------------- INTERACTION
def _interaction(
    q: str,
    df: pd.DataFrame,
    target: str,
    predictor: str,
    moderator: str,
    agg_default: Optional[str] = None,
) -> Optional[AnalystResult]:
    cols = [c for c in (moderator, predictor, target) if c in df.columns]
    if len(cols) < 3:
        return None
    sub = df[[moderator, predictor, target]].dropna().copy()
    if len(sub) < 10 or sub[moderator].nunique() < 2 or sub[predictor].nunique() < 2:
        return None

    binary_t = _binary_values(sub[target])
    is_binary = binary_t is not None
    if is_binary:
        sub[target] = binary_t

    # Stratified rates/means
    strat_means = sub.groupby([moderator, predictor], observed=True)[target].mean().round(3)
    strat_counts = sub.groupby([moderator, predictor], observed=True)[target].count()

    mod_levels = list(sub[moderator].unique())

    # Format stratified table
    table_dict = {}
    for (m_val, p_val), val in strat_means.items():
        table_dict[str((m_val, p_val))] = float(val)

    # Calculate within-moderator contrasts across predictor extremes
    mod_contrasts = {}
    for m_val in mod_levels:
        m_sub = sub[sub[moderator] == m_val]
        m_means = m_sub.groupby(predictor, observed=True)[target].mean()
        if len(m_means) >= 2:
            hi_p = m_means.idxmax()
            lo_p = m_means.idxmin()
            diff = float(m_means[hi_p] - m_means[lo_p])
            mod_contrasts[str(m_val)] = {
                "highest_pred": str(hi_p),
                "highest_val": float(m_means[hi_p]),
                "lowest_pred": str(lo_p),
                "lowest_val": float(m_means[lo_p]),
                "spread": diff,
            }

    # Build clear, comprehensive explanation
    contrast_strs = []
    for m_val, cinfo in mod_contrasts.items():
        if is_binary:
            contrast_strs.append(
                f"for {m_val}, {target} was {_pct(cinfo['highest_val'])} in {cinfo['highest_pred']} vs {_pct(cinfo['lowest_val'])} in {cinfo['lowest_pred']} ({_signed(cinfo['spread'] * 100)} pp)"
            )
        else:
            contrast_strs.append(
                f"for {m_val}, {target} was {_num(cinfo['highest_val'])} in {cinfo['highest_pred']} vs {_num(cinfo['lowest_val'])} in {cinfo['lowest_pred']} (delta {_signed(cinfo['spread'])})"
            )

    headline = (
        f"Yes -- the effect of {predictor} on {target} differs between {mod_levels[0]} and {mod_levels[1]}: "
        + "; ".join(contrast_strs)
        + "."
    )

    details = []
    strat_list = [
        f"{m_val} {p_val}: {_pct(val) if is_binary else _num(val)} (n={strat_counts.get((m_val, p_val), 0):,})"
        for (m_val, p_val), val in strat_means.items()
    ]
    details.append(f"Stratified {target} across {moderator} and {predictor}: " + "; ".join(strat_list) + ".")

    numbers = {
        "moderator": str(moderator),
        "predictor": str(predictor),
        "target": str(target),
        "stratified_table": {str(k): v for k, v in table_dict.items()},
        "table": table_dict,
        "contrasts": mod_contrasts,
        "is_binary": is_binary,
    }

    caveats = _hygiene(df, [moderator, predictor, target], len(sub))

    return AnalystResult(
        kind="INTERACTION",
        headline=headline,
        details=details,
        caveats=caveats,
        numbers=numbers,
        descriptive=True,
        supersedes_loop=True,
        finding="positive",
        executive_summary=headline,
    )


# --------------------------------------------------------------------------- ASSOCIATION
def _association(q, df, target, predictors) -> Optional[AnalystResult]:
    if len(predictors) >= 2:
        # Multi-predictor comparative association (e.g., "Is horsepower or weight more strongly related to mpg?")
        results_by_pred = {}
        for p_col in predictors:
            if p_col in df.columns and _is_numeric_like(df[p_col]):
                m_sub = pd.DataFrame({"x": _to_numeric(df[p_col]), "y": _to_numeric(df[target])}).dropna()
                if len(m_sub) >= 8:
                    r_val, p_val = sps.pearsonr(m_sub["x"], m_sub["y"])
                    results_by_pred[p_col] = (float(r_val), float(p_val), len(m_sub))
        if len(results_by_pred) >= 2:
            sorted_preds = sorted(results_by_pred.items(), key=lambda x: abs(x[1][0]), reverse=True)
            best_pred, (best_r, best_p, best_n) = sorted_preds[0]
            runner_pred, (run_r, run_p, run_n) = sorted_preds[1]
            headline = f"{best_pred} is more strongly related to {target} (|r|={abs(best_r):.2f}, R2={best_r**2:.2f}) than {runner_pred} (|r|={abs(run_r):.2f}, R2={run_r**2:.2f})."
            details = [f"Both variables have a statistically significant relationship with {target} ({best_pred}: r={best_r:.2f}, {runner_pred}: r={run_r:.2f}), but {best_pred} explains a higher proportion of variance ({best_r**2:.1%} vs {run_r**2:.1%})."]
            numbers = {
                "correlations": {k: v[0] for k, v in results_by_pred.items()},
                "strongest_predictor": best_pred,
                "pearson_r": best_r,
                "pearson_p": best_p,
                "adequate_power": True,
                "robust": True,
            }
            return AnalystResult("ASSOCIATION", headline, details, [], numbers, finding="positive" if best_r > 0 else "negative", descriptive=True)

    pred = predictors[0]
    binary_t = _binary_values(df[target])
    if _is_numeric_like(df[pred]) and (_is_numeric_like(df[target]) or binary_t is not None):
        x = _to_numeric(df[pred])
        y = binary_t if binary_t is not None else _to_numeric(df[target])
        m = pd.DataFrame({"x": x, "y": y}).dropna()
        n = len(m)
        if n < 8 or m["x"].nunique() < 3 or m["y"].nunique() < 2:
            return None
        r, p = sps.pearsonr(m["x"], m["y"])
        rho, rho_p = sps.spearmanr(m["x"], m["y"])
        z = np.arctanh(min(0.999999, abs(r)))
        se = 1 / math.sqrt(max(1, n - 3))
        lo, hi = float(np.tanh(np.arctanh(r) - 1.96 * se)), float(np.tanh(np.arctanh(r) + 1.96 * se))
        lr = sps.linregress(m["x"], m["y"])
        slope_ci = 1.96 * lr.stderr
        direction = "positive" if r > 0 else "negative"
        sig = p < ALPHA
        calc_power = _power_correlation(n, r_mde=0.30)
        is_powered = calc_power >= 0.80 and n >= 20
        caveats = _hygiene(df, [pred, target], n)
        numbers = {"pearson_r": float(r), "pearson_p": float(p), "ci_low": lo, "ci_high": hi, "spearman_rho": float(rho),
                   "spearman_p": float(rho_p), "slope": float(lr.slope), "n": n, "r_squared": float(r * r),
                   "power": round(float(calc_power), 3), "adequate_power": is_powered, "equivalence_bound_r": float(max(abs(lo), abs(hi)))}

        next_steps = []
        if sig:
            headline = (f"{pred} and {target} are {'strongly ' if abs(r) >= 0.5 else ''}associated ({_r_label(r)} {direction} relationship): r={r:.2f} (95% CI {lo:.2f} to {hi:.2f}, {_p(float(p))}, n={n:,}); "
                        f"{pred} accounts for about {_pct(r * r)} of the variation in {target}.")
            unit = "probability points" if binary_t is not None else target
            details = [f"On average each +1 in {pred} goes with {_signed(lr.slope)} {unit} (95% CI {_signed(lr.slope - slope_ci)} to {_signed(lr.slope + slope_ci)}). This is an association only; it does not establish a causal effect in either direction."]
            finding = direction
            if abs(r - rho) > 0.2 or (np.sign(r) != np.sign(rho) and abs(r) >= 0.1):
                caveats.append(f"Pearson r ({r:.2f}) and Spearman rho ({rho:.2f}) disagree: the relationship is nonlinear or driven by outliers; do not read the slope as typical.")
                numbers["robust"] = False
            else:
                numbers["robust"] = True
            next_steps.append(f"Test whether {pred} and {target} remain associated after controlling for secondary covariates.")
        else:
            headline = (f"No statistically detectable association between {pred} and {target}: r={r:.2f} (95% CI {lo:.2f} to {hi:.2f}, {_p(float(p))}, n={n:,}).")
            details = [f"At this sample size the data rule out a linear relationship stronger than about |r|={max(abs(lo), abs(hi)):.2f}."]
            finding = "none"
            if is_powered:
                details.append(f"Statistical power is sufficient (n={n:,}) to rule out linear association beyond |r|={max(abs(lo), abs(hi)):.2f}.")
            if rho_p < ALPHA:
                caveats.append(f"Rank correlation is significant (rho={rho:.2f}, {_p(float(rho_p))}) even though the linear one is not: look for a nonlinear or monotonic relationship.")
        # Principal Data Analyst Intelligence Enrichments:
        mean_x, mean_y = float(m["x"].mean()), float(m["y"].mean())
        elasticity = float(lr.slope * (mean_x / mean_y)) if mean_y != 0 else 0.0
        numbers["elasticity_at_mean"] = round(elasticity, 3)

        cand_cols = [c for c in df.columns if c != target and c != pred]
        multiv = _multivariate_ols_context(df, target, pred, cand_cols)
        if multiv is not None:
            numbers["multivariate_context"] = multiv
            ctrl_p = multiv["controlled_p_value"]
            ctrl_slope = multiv["controlled_slope"]
            covars_str = ", ".join(multiv["controlled_covariates"])
            if multiv["is_significant_after_control"]:
                details.append(
                    f"Multivariable covariate control: after controlling for {covars_str}, {pred} remains significantly associated with {target} "
                    f"(controlled slope {_signed(ctrl_slope)}, {_p(ctrl_p)}, model adj R2={multiv['adjusted_r_squared']:.2f})."
                )
            else:
                caveats.append(
                    f"Confounding risk: after controlling for {covars_str}, {pred}'s association attenuates to {_signed(ctrl_slope)} ({_p(ctrl_p)}) "
                    f"and is no longer statistically significant. The observed bivariate correlation may be confounded."
                )

        prof_x = _distribution_profile(m["x"])
        prof_y = _distribution_profile(m["y"])
        dist_prof = {"predictor": prof_x, "target": prof_y}
        numbers["distribution_profile"] = dist_prof

        scenario_sens = None
        if sig and mean_x != 0:
            delta_x = 0.10 * mean_x
            expected_delta_y = lr.slope * delta_x
            scenario_sens = {
                "hypothetical_x_shift_pct": 10.0,
                "delta_x": round(float(delta_x), 2),
                "expected_delta_y": round(float(expected_delta_y), 2),
                "elasticity": round(float(elasticity), 3),
            }
            numbers["scenario_sensitivity"] = scenario_sens
            details.append(
                f"Elasticity & Sensitivity: elasticity at means is {elasticity:.2f} (a +10% shift in {pred} associates with an expected {_signed(expected_delta_y)} shift in {target})."
            )

        strat_playbook = _build_strategic_playbook("ASSOCIATION", finding, pred, numbers)
        numbers["strategic_playbook"] = strat_playbook
        exec_summary = (
            f"{pred} and {target} are {_r_label(r)}ly associated (r={r:.2f}, {_p(float(p))})."
            if sig
            else f"No statistically detectable association between {pred} and {target}."
        )

        return AnalystResult(
            "ASSOCIATION", headline, details, caveats, numbers, finding=finding, next_steps=next_steps,
            executive_summary=exec_summary,
            multivariate_context=multiv,
            distribution_profile=dist_prof,
            scenario_sensitivity=scenario_sens,
            strategic_playbook=strat_playbook,
        )
    return None  # categorical predictors are handled as group comparisons by the caller


# --------------------------------------------------------------------------- TREND
def _period_frame(
    dates: pd.Series,
    values: pd.Series,
    agg: str,
    target_unit: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    from packages.analytics_core.src.statistics.analytical_math import resolve_canonical_temporal_axis
    d = pd.DataFrame({"t": dates, "v": values}).dropna()
    if len(d) == 0:
        return pd.DataFrame(), "year"

    axis = resolve_canonical_temporal_axis(d["t"], target_unit=target_unit)
    span_days = (d["t"].max() - d["t"].min()).days if hasattr(d["t"], "dt") else 0

    if axis.inferred_grain == "year":
        freq = "Y"
        grain = "year"
    elif axis.inferred_grain == "quarter":
        freq = "Q"
        grain = target_unit if target_unit else "quarter"
    elif axis.inferred_grain == "month":
        freq = "M"
        grain = target_unit if target_unit else "month"
    elif axis.inferred_grain == "week":
        freq = "W"
        grain = target_unit if target_unit else "week"
    elif axis.inferred_grain == "day":
        if target_unit:
            grain = target_unit
            freq = "M" if grain in ("year", "month") else ("W" if grain == "week" else "D")
        elif span_days >= 120:
            grain = "month"
            freq = "M"
        elif span_days >= 21:
            grain = "week"
            freq = "W"
        else:
            grain = "day"
            freq = "D"
    else:
        freq = "Y"
        grain = "year"

    d["period"] = d["t"].dt.to_period(freq) if hasattr(d["t"], "dt") else pd.to_datetime(d["t"]).dt.to_period(freq)
    g = d.groupby("period")
    if agg == "sum":
        out = g["v"].sum()
    elif agg == "count":
        out = g["v"].count()
    elif agg == "median":
        out = g["v"].median()
    else:
        out = g["v"].mean()
    cover = g["t"].agg(["min", "max"])
    frame = pd.DataFrame({"value": out, "first": cover["min"], "last": cover["max"], "n": g["v"].count()})
    return frame, grain


def _trend(q, df, target, time_col, agg_default) -> Optional[AnalystResult]:
    dates = _parse_dates(df[time_col])
    if dates is None:
        return None
    binary = _binary_values(df[target])
    y = binary if binary is not None else _to_numeric(df[target])
    agg = _choose_agg(q, agg_default, binary is not None, target=target)
    if agg == "rate":
        agg = "mean"

    target_unit = None
    if re.search(r"\b(year|years|annual|annually|yearly)\b", q, re.I):
        target_unit = "year"
    elif re.search(r"\b(quarter|quarterly)\b", q, re.I):
        target_unit = "quarter"
    elif re.search(r"\b(month|monthly)\b", q, re.I):
        target_unit = "month"
    elif re.search(r"\b(week|weekly)\b", q, re.I):
        target_unit = "week"
    elif re.search(r"\b(day|daily)\b", q, re.I):
        target_unit = "day"

    frame, grain = _period_frame(dates, y, agg, target_unit=target_unit)
    used = int(frame["n"].sum())
    caveats = _hygiene(df, [time_col, target], used)
    if len(frame) < 4:
        return None
    # Drop an incomplete trailing period (a partial month always looks like a "drop").
    distinct_days = pd.DataFrame({"t": dates.dt.normalize(), "y": y}).dropna().groupby(dates.dt.to_period("M" if grain == "month" else "W"))["t"].nunique()
    finer_than_period = bool(len(distinct_days)) and float(distinct_days.median()) > 1.5
    if grain in ("month", "week") and len(frame) >= 5 and finer_than_period:
        last = frame.iloc[-1]
        per = frame.index[-1]
        full_days = (per.end_time.normalize() - per.start_time.normalize()).days + 1
        seen_days = (last["last"].normalize() - per.start_time.normalize()).days + 1
        if agg in ("sum", "count") and seen_days < 0.9 * full_days:
            caveats.append(f"The last {grain} ({per}) covers only {seen_days} of {full_days} days and was excluded from the trend, since a partial period always looks like a drop.")
            frame = frame.iloc[:-1]
    from packages.analytics_core.src.statistics.analytical_math import canonical_time_coordinates
    idx = canonical_time_coordinates(frame.index, length=len(frame), target_unit=grain)
    vals = frame["value"].to_numpy(float)
    lr = sps.linregress(idx, vals)
    tau, tau_p = sps.kendalltau(idx, vals)
    ci = 1.96 * lr.stderr
    k = max(1, min(3, len(vals) // 4))
    start_lvl, end_lvl = float(vals[:k].mean()), float(vals[-k:].mean())
    pct_change = (end_lvl / start_lvl - 1) if start_lvl else float("nan")
    sig = lr.pvalue < ALPHA and tau_p < ALPHA
    word = _AGG_WORD[agg]
    first_lbl, last_lbl = str(frame.index[0]), str(frame.index[-1])
    numbers = {"grain": grain, "periods": len(frame), "slope_per_period": float(lr.slope), "slope_p": float(lr.pvalue),
               "kendall_tau": float(tau), "kendall_p": float(tau_p), "start_level": start_lvl, "end_level": end_lvl,
               "pct_change": float(pct_change) if pct_change == pct_change else None}
    if sig:
        direction = "up" if lr.slope > 0 else "down"
        headline = (f"{target} ({word} per {grain}) is trending {direction}: {_signed(lr.slope)} per {grain} (95% CI {_signed(lr.slope - ci)} to {_signed(lr.slope + ci)}, {_p(float(lr.pvalue))}; "
                    f"Kendall tau={tau:.2f}), from about {_num(start_lvl)} to {_num(end_lvl)}"
                    + (f" ({_signed(pct_change * 100)}%)" if pct_change == pct_change else "") + f" across {len(frame)} {grain}s ({first_lbl} to {last_lbl}).")
        finding = "positive" if lr.slope > 0 else "negative"
    else:
        headline = (f"No statistically detectable trend in {target} ({word} per {grain}): slope {_signed(lr.slope)} per {grain} (95% CI {_signed(lr.slope - ci)} to {_signed(lr.slope + ci)}, {_p(float(lr.pvalue))}) "
                    f"across {len(frame)} {grain}s ({first_lbl} to {last_lbl}).")
        finding = "none"
    details = []
    resid_sd = float(np.std(vals - (lr.intercept + lr.slope * idx), ddof=2)) if len(vals) > 2 else float("nan")
    if resid_sd == resid_sd and abs(vals.mean()) > 0:
        details.append(f"Typical {grain}-to-{grain} noise around the trend is about {_num(resid_sd)} ({_pct(resid_sd / abs(vals.mean()))} of the mean).")
    months = len(frame) if grain == "month" else 0
    if grain == "month" and months < 24:
        caveats.append("Under 24 months of history: seasonality cannot be separated from trend.")
    is_powered = len(frame) >= 6
    numbers["adequate_power"] = is_powered
    next_steps: List[str] = []
    if sig:
        next_steps.append(f"Project future run-rate and evaluate whether operational capacity matches the {direction} trend.")
    else:
        next_steps.append(f"Monitor subsequent periods to confirm whether the series remains stationary.")

    # Principal Data Analyst Intelligence Enrichments:
    # 1. Predictive run-rate extrapolation (next period projection with 95% PI)
    n_pts = len(vals)
    x_next = (idx[-1] + 1.0) if len(idx) > 0 else float(n_pts)
    y_next = lr.intercept + lr.slope * x_next
    x_bar = float(np.mean(idx))
    ss_x = float(np.sum((idx - x_bar) ** 2))
    se_pred = resid_sd * math.sqrt(1.0 + (1.0 / n_pts) + (((x_next - x_bar) ** 2) / ss_x)) if (resid_sd == resid_sd and ss_x > 0) else 0.0
    t_crit = sps.t.ppf(0.975, max(1, n_pts - 2)) if n_pts > 2 else 1.96
    pi_lo = y_next - t_crit * se_pred
    pi_hi = y_next + t_crit * se_pred
    run_rate_proj = {
        "next_period_index": int(round(x_next)),
        "projected_value": round(float(y_next), 2),
        "pi_low": round(float(pi_lo), 2),
        "pi_high": round(float(pi_hi), 2),
    }
    numbers["run_rate_projection"] = run_rate_proj
    if sig:
        details.append(
            f"Run-rate extrapolation: projected next {grain} value is {_num(y_next)} "
            f"(95% prediction interval {_num(pi_lo)} to {_num(pi_hi)})."
        )

    # 2. Time-series dynamics: Autocorrelation & Volatility
    cv = (resid_sd / abs(vals.mean())) if (resid_sd == resid_sd and vals.mean() != 0) else 0.0
    numbers["coefficient_of_variation"] = round(float(cv), 3)
    if n_pts >= 5:
        diffs = vals - np.mean(vals)
        denom_ac = float(np.sum(diffs ** 2))
        autocorr_1 = float(np.sum(diffs[:-1] * diffs[1:]) / denom_ac) if denom_ac > 0 else 0.0
        numbers["autocorrelation_lag1"] = round(autocorr_1, 3)
        if abs(autocorr_1) >= 0.4:
            nature = "strong trend momentum / persistence" if autocorr_1 > 0 else "mean-reverting oscillation"
            details.append(f"Persistence dynamics: lag-1 autocorrelation is {autocorr_1:.2f} ({nature}).")

    # 3. Distribution profile of historical periods
    dist_prof = _distribution_profile(pd.Series(vals))
    numbers["distribution_profile"] = dist_prof

    # 4. Scenario sensitivity
    cum_3 = 3 * lr.slope
    scenario_sens = {
        "projected_next_value": round(float(y_next), 2),
        "three_period_cumulative_slope": round(float(cum_3), 2),
    }
    numbers["scenario_sensitivity"] = scenario_sens

    # 5. Strategic playbook & Executive summary
    strat_playbook = _build_strategic_playbook("TREND", finding, target, numbers)
    numbers["strategic_playbook"] = strat_playbook
    exec_summary = (
        f"{target} is trending {direction} by {_signed(lr.slope)} per {grain} ({_p(float(lr.pvalue))})."
        if sig
        else f"No statistically detectable trend in {target} across {len(frame)} {grain}s."
    )

    return AnalystResult(
        "TREND", headline, details, caveats, numbers, finding=finding, next_steps=next_steps,
        executive_summary=exec_summary,
        distribution_profile=dist_prof,
        scenario_sensitivity=scenario_sens,
        strategic_playbook=strat_playbook,
    )


# --------------------------------------------------------------------------- PERIOD CHANGE
def _period_change(q, df, target, time_col, dims, agg_default) -> Optional[AnalystResult]:
    dates = _parse_dates(df[time_col])
    if dates is None:
        return None
    period = parse_period(q, dates)
    if period is None:
        return None
    binary = _binary_values(df[target])
    y = binary if binary is not None else _to_numeric(df[target])
    agg = _choose_agg(q, agg_default, binary is not None, target=target)
    if agg == "rate":
        agg = "mean"
    work = pd.DataFrame({"t": dates, "y": y})
    for dcol in dims:
        work[dcol] = df[dcol]
    id_candidates = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ("id", "uuid", "account", "customer", "user", "client", "merchant", "entity"))
        and c not in (target, time_col)
    ]
    for cid in id_candidates:
        if cid not in work.columns:
            work[cid] = df[cid]
    work = work.dropna(subset=["t", "y"])

    # DEFECT-034: if the question unambiguously names one specific entity within a candidate
    # dimension (e.g. "the West region"), the reported change must be that entity's own
    # before/after movement -- not the aggregate change across the whole dataset, attributed
    # after the fact to whichever segment happened to move most in raw terms (which can name
    # a completely different segment than the one asked about). The comparison/ranking paths
    # already scope this way via `_mentioned_levels`; this path did not.
    named_entity_dim: Optional[str] = None
    named_entity_value: Optional[Any] = None
    _entity_hits: List[Tuple[str, Any]] = []
    for dcol in dims:
        if dcol not in work.columns:
            continue
        levels = list(work[dcol].dropna().unique())
        for _, lv in _mentioned_levels(q, levels):
            _entity_hits.append((dcol, lv))
    ambiguous_entity_note: Optional[str] = None
    if len(_entity_hits) == 1:
        named_entity_dim, named_entity_value = _entity_hits[0]
        work = work[work[named_entity_dim] == named_entity_value]
    elif len(_entity_hits) > 1:
        ambiguous_entity_note = (
            "The question appears to name more than one specific segment ("
            + ", ".join(f"{d}='{v}'" for d, v in _entity_hits[:5])
            + "); reporting the aggregate change across all data instead of scoping to one."
        )

    p0, p1 = _prior_period(period)
    cur = work[(work["t"] >= period.start) & (work["t"] < period.end)]
    prev = work[(work["t"] >= p0) & (work["t"] < p1)]
    if cur.empty or prev.empty:
        scope_note = f" for {named_entity_value}" if named_entity_value is not None else ""
        return AnalystResult("PERIOD_CHANGE", f"Cannot compare {period.label} with the preceding period{scope_note}: the data has no rows for "
                             f"{'either' if cur.empty and prev.empty else 'the preceding' if prev.empty else 'the requested'} period.",
                             [], [], {"period": period.label}, finding=None)
    caveats = _hygiene(df, [time_col, target], len(work))
    if period.resolved_year_note:
        caveats.append(period.resolved_year_note)
    if ambiguous_entity_note:
        caveats.append(ambiguous_entity_note)

    def level(frame):
        return _agg(frame["y"], agg)

    v1, v0 = level(cur), level(prev)
    delta = v1 - v0
    pct = delta / v0 if v0 else float("nan")
    word = _AGG_WORD[agg]
    verb = "rose" if delta > 0 else "fell" if delta < 0 else "was flat"
    prev_label = f"{p0.strftime('%B %Y') if period.grain == 'month' else str(p0.date())}"
    entity_prefix = f"For {named_entity_dim}='{named_entity_value}': " if named_entity_dim else ""
    headline = (entity_prefix + f"{target} ({word}) {verb} from {_num(v0)} in {prev_label} to {_num(v1)} in {period.label}: {_signed(delta)}"
                + (f" ({_signed(pct * 100)}%)." if pct == pct else "."))
    numbers: Dict[str, Any] = {"period": period.label, "previous_start": str(p0.date()), "current": v1, "previous": v0, "delta": delta,
                               "pct_change": float(pct) if pct == pct else None, "aggregation": agg}
    if named_entity_dim:
        numbers["scoped_entity"] = {"dimension": named_entity_dim, "value": str(named_entity_value)}
    details: List[str] = []

    # Price-Volume-Mix (PVM) decomposition for overall period change:
    if agg in ("sum", "count") and len(prev) > 0 and len(cur) > 0:
        n0, n1 = float(len(prev)), float(len(cur))
        pr, cr = v0 / n0, v1 / n1
        delta_n = n1 - n0
        delta_p = cr - pr
        volume_eff = delta_n * pr
        rate_eff = n0 * delta_p
        cross_eff = delta_n * delta_p
        vol_share = volume_eff / delta if delta else 0.0
        rate_share = rate_eff / delta if delta else 0.0
        pvm_dict = {
            "volume_effect": float(volume_eff),
            "rate_effect": float(rate_eff),
            "cross_effect": float(cross_eff),
            "volume_effect_share": float(vol_share),
            "rate_effect_share": float(rate_share),
        }
        numbers["pvm_decomposition"] = pvm_dict
        numbers["volume_effect"] = float(volume_eff)
        numbers["rate_effect"] = float(rate_eff)
        numbers["cross_effect"] = float(cross_eff)
        numbers["volume_effect_share"] = float(vol_share)
        numbers["rate_effect_share"] = float(rate_share)
        if abs(vol_share) >= 0.60:
            pvm_desc = f"the change is volume-driven (transaction/row count moved {int(n0):,} -> {int(n1):,}, accounting for {_pct(vol_share, 0)} of the shift)"
        elif abs(rate_share) >= 0.60:
            pvm_desc = f"the change is rate-driven (average value per row moved {_num(pr)} -> {_num(cr)}, accounting for {_pct(rate_share, 0)} of the shift)"
        else:
            pvm_desc = f"a combination of volume shift ({int(n0):,} -> {int(n1):,}, {_pct(vol_share, 0)}) and rate shift ({_num(pr)} -> {_num(cr)}, {_pct(rate_share, 0)})"
        details.append(f"Volume vs rate decomposition: {pvm_desc}.")

    # Is this change unusual for this series?  Compare with all like-for-like period-over-period changes.
    freq = {"month": "M", "quarter": "Q", "year": "Y", "week": "W"}[period.grain]
    work["per"] = work["t"].dt.to_period(freq)
    series = work.groupby("per")["y"].agg("sum" if agg == "sum" else "count" if agg == "count" else "median" if agg == "median" else "mean")
    this_per = pd.Period(period.start, freq=freq)
    judged_delta = delta
    judged_series = series
    # Totals are not comparable across periods of different length (Feb vs Mar) or an incomplete
    # trailing period.  Judge "is this unusual?" on a per-calendar-day rate for sum/count metrics.
    obs_days = work.groupby("per")["t"].apply(lambda t: t.dt.normalize().nunique())
    daily_data = bool(len(obs_days)) and float(obs_days.median()) > 1.5
    if agg in ("sum", "count") and daily_data:
        days_in = pd.Series({per: (per.end_time.normalize() - per.start_time.normalize()).days + 1 for per in series.index})
        judged_series = series / days_in
        judged_delta = (v1 / ((period.end - period.start).days)) - (v0 / ((p1 - p0).days))
        last_per = series.index.max()
        if last_per != this_per and obs_days.get(last_per, 0) < 0.9 * days_in.get(last_per, 1):
            judged_series = judged_series.drop(index=last_per)
            caveats.append(f"The last period in the data ({last_per}) is incomplete and was left out of the 'is this unusual?' baseline.")
    changes = judged_series.diff().dropna()
    # Exclude the change being judged AND the following change (a rebound after a dip is part of
    # the same episode and would inflate the "normal" spread), then use a robust spread (MAD) so
    # that any other past shocks in the history do not make everything look normal.
    changes = changes[(changes.index != this_per) & (changes.index != this_per + 1)]
    if len(changes) >= 5:
        med = float(changes.median())
        mad = float((changes - med).abs().median()) * 1.4826
        if mad <= 0:
            mad = float(changes.std(ddof=1)) if changes.std(ddof=1) > 0 else 0.0
        if mad > 0:
            z = (judged_delta - med) / mad
            pct_rank = float((changes.abs() < abs(judged_delta)).mean())
            unit = " per day" if (agg in ("sum", "count") and daily_data) else ""
            numbers["change_z_vs_history"] = float(z)
            z_txt = "more than 10" if abs(z) > 10 else f"about {abs(z):.1f}"
            if abs(z) >= 2.5:
                details.append(f"This change is unusual for this series{' (compared on a per-day basis, so month length does not distort it)' if unit else ''}: {z_txt} robust standard deviations from the typical {period.grain}-to-{period.grain} change, larger than {_pct(pct_rank, 0)} of {len(changes)} other changes.")
            else:
                details.append(f"This change is within normal {period.grain}-to-{period.grain} variation{' on a per-day basis' if unit else ''} ({abs(z):.1f} robust standard deviations; larger than {_pct(pct_rank, 0)} of {len(changes)} other changes) -- it may be noise rather than a real shift.")
    else:
        caveats.append(f"Only {len(changes)} comparable {period.grain}-to-{period.grain} changes exist, so 'is this unusual?' cannot be judged.")

    if agg in ("sum", "count"):
        len_cur, len_prev = (period.end - period.start).days, (p1 - p0).days
        if len_prev and abs(len_cur - len_prev) / len_prev > 0.03:
            per_day_prev = v0 / len_prev
            per_day_cur = v1 / len_cur
            per_day_change = ((per_day_cur / per_day_prev) - 1) if per_day_prev else 0.0
            day_diff = len_cur - len_prev
            day_word = f"{abs(day_diff)} fewer calendar days" if day_diff < 0 else f"{day_diff} more calendar days"
            details.append(
                f"Calendar effect: {period.label} has {len_cur} days vs {len_prev} in the prior period ({day_word}). Per day, {target} went from "
                f"{_num(per_day_prev)} to {_num(per_day_cur)} ({_signed(per_day_change * 100)}%), "
                "so part of the change in totals is period length, not performance.")
            numbers["per_day_previous"], numbers["per_day_current"] = per_day_prev, per_day_cur
            numbers["per_day_change_pct"] = float(per_day_change * 100)
        seen_cur = cur["t"].dt.normalize().nunique()
        seen_prev = prev["t"].dt.normalize().nunique()
        if seen_prev and abs(seen_cur - seen_prev) / seen_prev > 0.1:
            caveats.append(f"{period.label} has data on {seen_cur} days vs {seen_prev} in the prior period; part of the change is coverage, not performance.")
            numbers["days_current"], numbers["days_previous"] = int(seen_cur), int(seen_prev)

    # Decompose the change by each candidate dimension.
    best: Optional[Tuple[float, str, pd.DataFrame]] = None
    for dcol in dims:
        if dcol not in work.columns:
            continue
        cats = work[dcol].dropna().nunique()
        if cats < 2 or cats > 40:
            continue
        gc = cur.groupby(dcol)["y"]
        gp = prev.groupby(dcol)["y"]
        segs = sorted(set(gc.groups) | set(gp.groups), key=str)
        rows = []
        for sgm in segs:
            c_v = _agg(gc.get_group(sgm), agg) if sgm in gc.groups else 0.0
            p_v = _agg(gp.get_group(sgm), agg) if sgm in gp.groups else 0.0
            rows.append((sgm, p_v, c_v, c_v - p_v, len(gc.get_group(sgm)) if sgm in gc.groups else 0,
                         len(gp.get_group(sgm)) if sgm in gp.groups else 0))
        tab = pd.DataFrame(rows, columns=["segment", "prev", "cur", "delta", "n_cur", "n_prev"]).set_index("segment")
        if agg in ("sum", "count"):
            denom = tab["delta"].abs().sum()
            conc = float(tab["delta"].abs().max() / denom) if denom else 0.0
            same_sign = tab["delta"] * np.sign(delta) if delta else tab["delta"]
            aligned = float(same_sign.clip(lower=0).max() / same_sign.clip(lower=0).sum()) if same_sign.clip(lower=0).sum() else 0.0
            score = aligned
        else:
            score = float(tab["delta"].abs().max() / (tab["delta"].abs().sum() or 1))
        if best is None or score > best[0]:
            best = (score, dcol, tab)
    if best is not None:
        score, dcol, tab = best
        tab = tab.sort_values("delta", ascending=bool(delta < 0))
        top = tab.iloc[0]
        top_name = tab.index[0]
        if agg in ("sum", "count") and delta != 0:
            share = top["delta"] / delta
            parts = "; ".join(f"{k}: {_signed(r['delta'])} ({_num(r['prev'])} -> {_num(r['cur'])})" for k, r in tab.head(4).iterrows())
            details.append(f"Biggest contributor by {dcol}: {top_name}, {_signed(top['delta'])} = {_pct(share, 0)} of the total change. By {dcol} -- {parts}.")
            rest = tab.drop(index=top_name)
            if len(rest) >= 2 and delta:
                rest_delta = float(rest["delta"].sum())
                details.append(f"Excluding {top_name}, {target} {'changed' if rest_delta else 'was flat'} by {_signed(rest_delta)} ({_pct(rest_delta / (v0 - top['prev']) if (v0 - top['prev']) else float('nan'))}) across the other {dcol} values.")
            numbers.update({"top_contributor": str(top_name), "top_dimension": dcol, "top_contribution_share": float(share)})
            # 1. Rate vs Volume (Price-Volume-Mix) Decomposition
            if top["n_prev"] and top["n_cur"]:
                n0, n1 = float(top["n_prev"]), float(top["n_cur"])
                pr, cr = top["prev"] / n0, top["cur"] / n1
                details.append(f"Within {top_name}: rows {int(n0):,} -> {int(n1):,}; average per row {_num(pr)} -> {_num(cr)}.")
                delta_n = n1 - n0
                delta_p = cr - pr
                volume_eff = delta_n * pr
                rate_eff = n0 * delta_p
                cross_eff = delta_n * delta_p
                vol_share = volume_eff / top["delta"] if top["delta"] else 0.0
                rate_share = rate_eff / top["delta"] if top["delta"] else 0.0
                numbers["volume_effect"] = float(volume_eff)
                numbers["rate_effect"] = float(rate_eff)
                numbers["cross_effect"] = float(cross_eff)
                numbers["volume_effect_share"] = float(vol_share)
                numbers["rate_effect_share"] = float(rate_share)
                if vol_share >= 0.65:
                    decomp_desc = f"primarily volume-driven: activity/row count changed from {int(n0):,} to {int(n1):,} ({_signed((n1/n0 - 1)*100)}%), accounting for {_pct(vol_share, 0)} of the segment change"
                elif rate_share >= 0.65:
                    decomp_desc = f"primarily rate/value-driven: average value per row moved from {_num(pr)} to {_num(cr)} ({_signed((cr/pr - 1)*100)}%), accounting for {_pct(rate_share, 0)} of the segment change"
                else:
                    decomp_desc = f"a combination of volume shift ({int(n0):,} -> {int(n1):,}, {_pct(vol_share, 0)}) and rate/value shift ({_num(pr)} -> {_num(cr)}, {_pct(rate_share, 0)})"
                details.append(f"Decomposition mechanism for {top_name}: {decomp_desc}.")
            # 2. Secondary dimension drilldown: what drove the shift INSIDE the top contributor?
            other_dims = [c for c in dims if c != dcol and c in work.columns and work[c].nunique() >= 2]
            best_sub = None
            for sd in other_dims:
                c_sub = cur[cur[dcol] == top_name]
                p_sub = prev[prev[dcol] == top_name]
                if c_sub.empty and p_sub.empty:
                    continue
                g_c_sub = c_sub.groupby(sd)["y"]
                g_p_sub = p_sub.groupby(sd)["y"]
                sub_segs = sorted(set(g_c_sub.groups) | set(g_p_sub.groups), key=str)
                sub_rows = []
                for s_sgm in sub_segs:
                    s_c_v = _agg(g_c_sub.get_group(s_sgm), agg) if s_sgm in g_c_sub.groups else 0.0
                    s_p_v = _agg(g_p_sub.get_group(s_sgm), agg) if s_sgm in g_p_sub.groups else 0.0
                    sub_rows.append((s_sgm, s_p_v, s_c_v, s_c_v - s_p_v))
                if sub_rows:
                    sub_tab = pd.DataFrame(sub_rows, columns=["sub_segment", "prev", "cur", "delta"]).set_index("sub_segment")
                    sub_tab = sub_tab.sort_values("delta", ascending=bool(top["delta"] < 0))
                    sub_top = sub_tab.iloc[0]
                    sub_share = sub_top["delta"] / top["delta"] if top["delta"] else 0.0
                    if best_sub is None or sub_share > best_sub[0]:
                        best_sub = (sub_share, sd, sub_top.name, sub_top["delta"])
            if best_sub is not None and best_sub[0] >= 0.40:
                sub_share, sd, sub_name, sub_delta = best_sub
                details.append(f"Sub-segment drilldown within {top_name}: the change was concentrated {_pct(sub_share, 0)} in {sd}='{sub_name}' ({_signed(sub_delta)} of {top_name}'s {_signed(top['delta'])}).")
                numbers["sub_segment_dimension"] = sd
                numbers["sub_segment_name"] = str(sub_name)
                numbers["sub_segment_share"] = float(sub_share)
                numbers["sub_segment_findings"] = {
                    "dimension": sd,
                    "top_subsegment": str(sub_name),
                    "share": float(sub_share),
                    "delta": float(sub_delta),
                }
            # 3. Formal check that the top segment's shift is real, not noise.
            a = cur.loc[cur[dcol] == top_name, "y"].to_numpy(float)
            b = prev.loc[prev[dcol] == top_name, "y"].to_numpy(float)
            others_c = cur.loc[cur[dcol] != top_name, "y"].to_numpy(float)
            others_p = prev.loc[prev[dcol] != top_name, "y"].to_numpy(float)
            if min(len(a), len(b)) >= 3 and min(len(others_c), len(others_p)) >= 3:
                w_top = _welch(a, b)
                w_oth = _welch(others_c, others_p)
                numbers["top_segment_shift_p"] = w_top["p"]
                numbers["other_segments_shift_p"] = w_oth["p"]
                details.append(f"Per-row change within {top_name}: {_signed(w_top['diff'])} ({_p(w_top['p'])}); across all other {dcol} values: {_signed(w_oth['diff'])} ({_p(w_oth['p'])}).")
        elif agg in ("mean", "median"):
            parts = "; ".join(f"{k}: {_signed(r['delta'])} ({_num(r['prev'])} -> {_num(r['cur'])}, n={int(r['n_cur']):,})" for k, r in tab.head(4).iterrows())
            details.append(f"Largest movement by {dcol}: {top_name}. By {dcol} -- {parts}.")
            numbers.update({"top_contributor": str(top_name), "top_dimension": dcol})
    else:
        details.append("No categorical dimension with 2-40 values was available to break the change down.")
    # 4. Intra-period trajectory check
    cur_days = cur["t"].dt.normalize().nunique()
    if cur_days >= 14:
        t_mid = cur["t"].min() + (cur["t"].max() - cur["t"].min()) / 2
        h1 = cur[cur["t"] < t_mid]["y"]
        h2 = cur[cur["t"] >= t_mid]["y"]
        if len(h1) >= 4 and len(h2) >= 4:
            m1, m2 = float(h1.mean()), float(h2.mean())
            if m1 > 0:
                pct_step = (m2 / m1 - 1)
                if abs(pct_step) >= 0.25:
                    shift_word = "drop" if pct_step < 0 else "increase"
                    details.append(
                        f"Intra-period trajectory: a sharp run-rate {shift_word} began around {t_mid.date()} "
                        f"(average per row {_num(m1)} -> {_num(m2)}, {_signed(pct_step * 100)}%), "
                        f"suggesting an operational change or event mid-period rather than gradual decay."
                    )
                    numbers["intraperiod_shift_pct"] = float(pct_step * 100)
                    numbers["intra_period_trajectory"] = {
                        "step_change_detected": True,
                        "inflection_date": str(t_mid.date()),
                        "first_half_mean": m1,
                        "second_half_mean": m2,
                        "shift_pct": float(pct_step * 100),
                    }
    # 5. Tailored actionable recommendations
    next_steps: List[str] = []
    if best is not None:
        if numbers.get("sub_segment_name"):
            next_steps.append(f"Audit operational drivers, pipeline, and customer logs for {top_name} in {numbers['sub_segment_dimension']}='{numbers['sub_segment_name']}'.")
        if numbers.get("volume_effect_share", 0) >= 0.60:
            next_steps.append(f"Investigate funnel traffic, order counts, and churn in {top_name} (change is predominantly volume).")
        elif numbers.get("rate_effect_share", 0) >= 0.60:
            next_steps.append(f"Audit pricing, discounting, and product mix in {top_name} (change is predominantly average order value).")
        else:
            next_steps.append(f"Review account management, marketing campaigns, and regional operations in {top_name}.")
    details.append("This identifies the specific driver segments and volume/value mechanisms; investigate operational causes (pricing, promotions, outages, contract renewals) for the highlighted driver.")

    # Principal Data Analyst Intelligence Enrichments:
    # 1. Kitagawa Decomposition (Rate vs Mix-Shift Decomposition across dimensions)
    top_kitagawa = None
    for dcol_k in dims:
        kit = _kitagawa_decomposition(prev, cur, dcol_k, "y", agg)
        if kit is not None:
            if kit["simpsons_paradox"]:
                top_kitagawa = kit
                break
            if top_kitagawa is None or abs(kit["mix_shift_effect"]) > abs(top_kitagawa["mix_shift_effect"]):
                top_kitagawa = kit

    if top_kitagawa is not None:
        numbers["mix_shift_decomposition"] = top_kitagawa
        if top_kitagawa.get("simpsons_paradox"):
            details.append(
                f"Simpson's Paradox detected across {top_kitagawa['dimension']}: overall average {target} {verb} "
                f"({_signed(top_kitagawa['delta_average'])}), but within-segment rates moved in the opposite direction! "
                f"Compositional mix shift ({_signed(top_kitagawa['mix_shift_effect'])}, {_pct(top_kitagawa['mix_share_pct'] / 100, 0)}) "
                "drove the entire observed change."
            )
            caveats.append(
                f"Compositional distortion: change across {top_kitagawa['dimension']} is a mix-shift artifact (Simpson's Paradox), not a uniform rate change."
            )
        elif abs(top_kitagawa.get("mix_share_pct", 0.0)) >= 30.0:
            details.append(
                f"Kitagawa decomposition by {top_kitagawa['dimension']}: pure rate change accounts for "
                f"{_pct(top_kitagawa['rate_share_pct'] / 100, 0)} ({_signed(top_kitagawa['pure_rate_effect'])}), while category mix shift accounts for "
                f"{_pct(top_kitagawa['mix_share_pct'] / 100, 0)} ({_signed(top_kitagawa['mix_shift_effect'])})."
            )

    # 2. Cohort & Account Lifecycle Bridge
    cohort_lifecycle = None
    for cid in id_candidates:
        cl = _cohort_lifecycle_bridge(prev, cur, cid, "y", agg)
        if cl is not None:
            cohort_lifecycle = cl
            numbers["cohort_lifecycle"] = cl
            details.append(
                f"Cohort lifecycle bridge ({cid}): retained accounts contributed {_signed(cl['retained_account_delta'])}, "
                f"new account acquisition added +{_num(cl['new_account_acquisition'])}, while churned accounts removed -{_num(cl['churned_account_loss'])}."
            )
            break

    # 3. Multilevel Waterfall Bridge (Exact Reconciliation V0 -> V1)
    waterfall_bridge = {
        "initial_value": float(v0),
        "final_value": float(v1),
        "net_delta": float(delta),
        "steps": [],
    }
    running_total = float(v0)
    waterfall_bridge["steps"].append({"step": "Prior Base", "delta": float(v0), "subtotal": running_total})
    if best is not None and agg in ("sum", "count"):
        _, dcol_b, tab_b = best
        tab_sorted = tab_b.sort_values("delta", ascending=bool(delta < 0))
        for seg_name, srow in tab_sorted.head(3).iterrows():
            d_seg = float(srow["delta"])
            running_total += d_seg
            waterfall_bridge["steps"].append({"step": f"{dcol_b}: {seg_name}", "delta": d_seg, "subtotal": running_total})
        if len(tab_sorted) > 3:
            d_others = float(tab_sorted.iloc[3:]["delta"].sum())
            running_total += d_others
            waterfall_bridge["steps"].append({"step": f"{dcol_b}: Others", "delta": d_others, "subtotal": running_total})
    elif "pvm_decomposition" in numbers:
        pvm = numbers["pvm_decomposition"]
        running_total += pvm["volume_effect"]
        waterfall_bridge["steps"].append({"step": "Volume Effect", "delta": pvm["volume_effect"], "subtotal": running_total})
        running_total += pvm["rate_effect"]
        waterfall_bridge["steps"].append({"step": "Rate Effect", "delta": pvm["rate_effect"], "subtotal": running_total})
        running_total += pvm["cross_effect"]
        waterfall_bridge["steps"].append({"step": "Cross Interaction", "delta": pvm["cross_effect"], "subtotal": running_total})
    else:
        running_total += float(delta)
        waterfall_bridge["steps"].append({"step": "Net Change", "delta": float(delta), "subtotal": running_total})
    numbers["waterfall_bridge"] = waterfall_bridge

    # 4. Tail Concentration Diagnostics & Distribution Profiles
    conc_diag = None
    if best is not None:
        _, _, tab_c = best
        conc_diag = _gini_and_concentration(tab_c["cur"] if "cur" in tab_c.columns else tab_c["delta"].abs())
        numbers["concentration_diagnostics"] = conc_diag

    dist_prof = {
        "current_period": _distribution_profile(cur["y"]),
        "prior_period": _distribution_profile(prev["y"]),
    }
    numbers["distribution_profile"] = dist_prof

    # 5. Counterfactual Opportunity Sizing
    scenario_sens = None
    if best is not None and top_name and top["delta"] < 0:
        cf_val = v1 - top["delta"]
        scenario_sens = {
            "driver": str(top_name),
            "counterfactual_value_if_flat": round(float(cf_val), 2),
            "prevented_loss": round(float(-top["delta"]), 2),
        }
        numbers["scenario_sensitivity"] = scenario_sens
        details.append(
            f"Counterfactual opportunity: if {top_name} had held flat at prior period levels, {period.label} {target} "
            f"would have been {_num(cf_val)} instead of {_num(v1)} (avoiding a {_num(-top['delta'])} decline)."
        )

    # 6. Strategic Playbook & Executive Summary
    # DEFECT-034 (cosmetic leak): when the change was scoped to a single named entity and no
    # further sub-dimension was available to decompose within it, `best` is None and
    # `driver_lead` used to fall through to `_build_strategic_playbook`'s own literal
    # "primary driver" placeholder text instead of naming the entity that was actually asked
    # about. Fall back to the named entity itself before falling back to the generic literal.
    driver_lead = str(top_name) if best is not None else (str(named_entity_value) if named_entity_dim else None)
    strat_playbook = _build_strategic_playbook(
        "PERIOD_CHANGE",
        ("negative" if delta < 0 else "positive" if delta > 0 else "none"),
        driver_lead,
        numbers,
    )
    numbers["strategic_playbook"] = strat_playbook

    exec_summary = f"{target} {verb} by {_signed(delta)} ({_signed(pct * 100)}%)" + (f" for {named_entity_dim}='{named_entity_value}'" if named_entity_dim else "") + f" in {period.label} vs prior period."
    if best is not None and "top_contribution_share" in numbers:
        exec_summary += f" Primary driver was {driver_lead} ({_pct(numbers['top_contribution_share'], 0)} of shift)."

    return AnalystResult(
        "PERIOD_CHANGE", headline, details, caveats, numbers, descriptive=True, supersedes_loop=True,
        finding=("negative" if delta < 0 else "positive" if delta > 0 else "none"),
        next_steps=next_steps,
        executive_summary=exec_summary,
        waterfall_bridge=waterfall_bridge,
        mix_shift_decomposition=top_kitagawa,
        concentration_diagnostics=conc_diag,
        scenario_sensitivity=scenario_sens,
        distribution_profile=dist_prof,
        cohort_lifecycle=cohort_lifecycle,
        strategic_playbook=strat_playbook,
    )


# --------------------------------------------------------------------------- entry point
# DEFECT-040 (session 22): a *pure* descriptive breakdown request -- "What is the churn
# rate by plan type?" / "What is the average tenure by department?" -- asks for per-group
# aggregates, nothing more.  It makes no comparative, causal, ranking or predictive claim,
# so the churn-identifiability machinery (which exists to stop confounded segment
# *comparisons* being read as effects) has nothing to protect and must not swallow the
# answer into a generic "insufficient evidence".  Deliberately conservative: it needs a
# measure word AND a group cue, and ANY comparative/causal/ranking/temporal/predictive cue
# disqualifies it, leaving those questions with the specialist loop exactly as before.
_BREAKDOWN_MEASURE_RE = re.compile(
    r"\b(rate|average|avg|mean|median|total|count|number|share|percentage|proportion|distribution)\b", re.I)
_BREAKDOWN_GROUP_RE = re.compile(
    r"\b(by|per|for each|across|broken down by|split by|grouped by|segmented by)\b", re.I)
_BREAKDOWN_DISQUALIFY_RE = re.compile(
    r"\b(why|because|cause[sd]?|causing|driv\w*|reasons?|explain\w*|confound\w*|correlat\w*|"
    r"relationship|associat\w*|effect|impact\w*|higher|lower|highest|lowest|higher|increas\w*|"
    r"decreas\w*|trend\w*|chang\w*|predict\w*|likel\w*|compar\w*|differ\w*|vs|versus|"
    r"which|who|top|bottom|best|worst|most|least|risk\w*|spik\w*|drop\w*|fell|fall\w*|rose|"
    r"grow\w*|declin\w*|since|last|this|previous|forecast\w*)\b", re.I)


def is_pure_descriptive_breakdown(question: str) -> bool:
    """True only for a plain "<measure> by <group>" request with no comparative,
    causal, ranking, temporal or predictive wording."""
    q = (question or "").strip()
    if not q:
        return False
    return bool(
        _BREAKDOWN_MEASURE_RE.search(q)
        and _BREAKDOWN_GROUP_RE.search(q)
        and not _BREAKDOWN_DISQUALIFY_RE.search(q)
    )


def build_analyst_result(
    question: str,
    df: pd.DataFrame,
    *,
    target: Optional[str],
    group: Optional[str] = None,
    explanatory: Optional[Sequence[str]] = None,
    time_col: Optional[str] = None,
    default_aggregation: Optional[str] = None,
    candidate_dimensions: Optional[Sequence[str]] = None,
) -> Optional[AnalystResult]:
    """Return the numbers-first answer for the resolved question, or ``None`` if the question
    shape is not one this layer can answer faithfully.  Never raises."""
    try:
        if df is None or df.empty or not target or target not in df.columns:
            return None
        expl = [c for c in (explanatory or []) if c in df.columns and c != target]
        is_count_rank = (group == target) and bool(re.search(r"\b(most|least|fewest|count|which)\b", question, re.I))
        group = group if (group in df.columns and (group != target or is_count_rank)) else None
        time_col = time_col if (time_col in df.columns and time_col != target) else None
        dates = _parse_dates(df[time_col]) if time_col else None
        period = parse_period(question, dates) if dates is not None else None
        kind = classify_question(question, has_time=dates is not None, has_group=group is not None or bool(expl),
                                 has_explanatory=bool(expl), period=period)
        if kind == "NONE":
            return None
        if kind == "PERIOD_CHANGE":
            discovered_dims = [c for c in ([group] if group else []) + [c for c in df.columns
                    if c not in (target, time_col, group) and not _is_numeric_like(df[c]) and 2 <= df[c].nunique(dropna=True) <= 40]]
            dims = list(candidate_dimensions) if candidate_dimensions else discovered_dims
            return _period_change(question, df, target, time_col, dims, default_aggregation)
        if kind == "PERIOD_CHANGE_UNRESOLVED_TIMEFRAME":
            # DEFECT-037: the question names a time window this layer doesn't recognise the
            # phrasing for. Rather than silently answering as an unrelated RANKING question (the
            # prior behaviour every DEFECT-031/033/034/035/036 fix chased one phrasing at a time),
            # fall back to the honest full-range answer but disclose, prominently and specifically,
            # that the named timeframe was not applied -- an explicitly-disclosed, reversible
            # assumption rather than a silent misclassification.
            fallback_group = group
            if fallback_group is None:
                if candidate_dimensions:
                    fallback_group = next((c for c in candidate_dimensions if c in df.columns and c != target), None)
                else:
                    fallback_group = next((c for c in df.columns if c not in (target, time_col)
                                            and not _is_numeric_like(df[c]) and 2 <= df[c].nunique(dropna=True) <= 40), None)
            result = _ranking(question, df, target, fallback_group, default_aggregation) if fallback_group else None
            if result is None:
                return None
            unresolved_note = (
                "This question appears to name a specific time window that this analysis could "
                "not confidently parse, so the figures below cover the full available date range "
                "instead of being scoped to that window -- treat this as a full-history answer, "
                "not an answer to the specific period asked about."
            )
            result.caveats = [unresolved_note] + list(result.caveats)
            result.headline = f"[Unscoped -- requested time window not recognized] {result.headline}"
            return result
        if kind == "INTERACTION":
            pred = expl[0] if expl else None
            mod = group
            if not pred or not mod:
                from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
                c_c = compile_canonical_question_contract(question, df)
                if c_c.explanatory_columns and c_c.grouping_columns:
                    pred = c_c.explanatory_columns[0]
                    mod = c_c.grouping_columns[0]
            if pred and mod:
                res = _interaction(question, df, target, pred, mod, default_aggregation)
                if res is not None:
                    return res
        if kind == "TREND":
            return _trend(question, df, target, time_col, default_aggregation)
        if kind == "ASSOCIATION":
            res = _association(question, df, target, expl)
            if res is not None:
                return res
            g = expl[0]
            if not _is_numeric_like(df[g]) or df[g].nunique(dropna=True) <= 12:
                return _group_comparison(question, df, target, g, default_aggregation)
            return None
        if group is None:
            return None
        if kind == "ROOT_CAUSE":
            return _root_cause(question, df, target, group, expl, default_aggregation)
        if kind == "RANKING":
            return _ranking(question, df, target, group, default_aggregation)
        if kind == "GROUP_COMPARISON":
            return _group_comparison(question, df, target, group, default_aggregation)
        return None
    except Exception:
        return None
