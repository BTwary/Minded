"""
variable_resolution_gate.py

Centralized pre-analysis variable-resolution gate (VARIABLE_NOT_FOUND).

Context (see docs/... question-invariance workstream, Phase 1 baseline):
four independent estimand families (correlation, rate/proportion, trend,
ranking) each had their own ad-hoc column-resolution code, and none of them
distinguished "the question named no specific variable" from "the question
named a specific variable that does not exist in this dataset". When the
requested variable did not exist, every family except the churn-specific
path silently fell back to whatever column happened to be available
(usually the first categorical/numeric pair) and reported a confident
answer to a question that was never asked -- in two observed cases at
confidence 1.0.

This module is the single, family-agnostic check that must run before any
of those synthesizers get a chance to do that fallback binding. It does not
try to resolve what the user *meant* (that remains each family's job); it
only answers one narrow, high-precision question: does the raw question
text contain what looks like a specific physical-column reference (a
snake_case identifier, e.g. "marketing_spend", "widget_count",
"blorptastic_index") that does not correspond -- exactly, or under simple
case/separator normalization -- to any column in the available schema?

Deliberately NOT in scope here (by design, to keep this check high
precision and low false-positive):
  - Bare English words with no underscore ("returned", "orders") are not
    flagged. Those questions are still protected against fabrication by
    each family's own admissibility gate; this module only owns the
    narrower "the user typed an identifier that isn't a real column"
    signal, which is the one no family was handling generically.
  - Churn/attrition/retention-flavored tokens are excluded here entirely.
    That family already has its own dedicated, tested fail-closed
    resolution (SemanticEngine.resolve_schema's churn_event_col ->
    churn_outcome_available -> "no identifiable churn outcome" path) and
    this gate must not preempt it with a different, less specific message.
"""
from __future__ import annotations

import re
import pandas as pd
from typing import Iterable, List

# A "physical-identifier-style" token: starts with a letter, contains at
# least one underscore-joined segment. This deliberately matches the shape
# of an actual database/dataframe column name (snake_case) rather than
# ordinary English phrasing, which keeps false positives low: everyday
# words and multi-word phrases ("average order value") never match this
# pattern, only tokens that look like the user is naming a column directly.
_IDENTIFIER_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b")

# Tokens touching this vocabulary are owned by the existing, dedicated
# churn-identifiability path (see semantic.py) and must not be re-flagged
# here with a different, generic message.
_CHURN_OWNED_SUBSTRINGS = (
    "churn", "attrition", "cancel", "retention", "dropout", "dropoff",
)


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[\s-]+", "_", str(name).strip().lower())


# A second, narrower detection rule for the rate/proportion family
# specifically: a question phrased around "proportion"/"rate"/"percentage"
# that names NO column from the available schema at all (not even a
# synonym token, not even partially) is asking about an outcome this
# dataset simply does not contain. Scoped tightly to this vocabulary (as
# opposed to a blanket "zero column tokens" rule for every question,
# which would be far too broad and would misfire on legitimate
# alias-resolved questions in other task families) because this is the
# one documented case where the rate family currently substitutes an
# unrelated ranking/aggregation instead of reporting the missing outcome.
_RATE_PROPORTION_RE = re.compile(r"\b(proportion|percentage|percent|rate)\b", re.I)


def _question_references_any_column(question: str, columns: Iterable[str]) -> bool:
    q_norm = re.sub(r"[^a-z0-9]+", " ", (question or "").lower())
    q_tokens = set(q_norm.split())
    for c in columns:
        col_phrase = _normalize_column_name(c).replace("_", " ").strip()
        if not col_phrase:
            continue
        col_words = col_phrase.split()
        if len(col_words) == 1:
            if col_words[0] in q_tokens:
                return True
        elif col_phrase in q_norm:
            return True
    return False


def detect_rate_question_with_no_column_reference(question: str, columns: Iterable[str]) -> bool:
    """True when `question` uses rate/proportion vocabulary but names no
    column from `columns` anywhere -- the outcome it asks about isn't in
    this dataset at all, as opposed to merely being phrased with a synonym
    of some column that IS present."""
    if not _RATE_PROPORTION_RE.search(question or ""):
        return False
    ql = (question or "").lower()
    if any(sub in ql for sub in _CHURN_OWNED_SUBSTRINGS):
        # Churn-flavored rate questions ("churn rate by segment") are owned
        # by the existing dedicated churn-identifiability path, which gives
        # a more specific, already-correct answer than this generic gate
        # would ("no identifiable churn outcome" vs. a generic missing-
        # variable message) -- do not preempt it.
        return False
    return not _question_references_any_column(question, columns)


# Vocabulary of known aggregate/statistic verbs that prefix a metric noun.
_AGGREGATE_VERB_RE = re.compile(r"\b(average|mean|median|sum)\b", re.I)

# Known business metric words (including common aliases) that the alias system
# can resolve even when the column is named differently (e.g., "sales" → revenue).
# When any of these words appears in the aggregate question, the system has a
# plausible binding path and should NOT be blocked here — the family-level
# resolution will either succeed or return its own specific failure.
_KNOWN_METRIC_WORDS: frozenset = frozenset([
    # revenue / sales family
    "revenue", "sales", "income", "turnover", "proceeds", "takings", "gmv",
    "net", "gross",
    # quantity / volume
    "quantity", "qty", "units", "volume", "items",
    # cost / spend
    "cost", "costs", "expense", "expenses", "spend", "expenditure",
    # profit / margin
    "profit", "profits", "margin", "earnings",
    # discount / price
    "discount", "markdown", "rebate", "price", "pricing",
    # order / transaction
    "value", "order", "orders", "transaction", "transactions", "amount",
    # engagement
    "conversion", "visits", "sessions", "clicks", "impressions", "views",
    # count
    "count", "number",
    # physical / demographic measures
    "weight", "height", "age", "score", "rating", "mass", "bmi", "index",
    # time
    "time", "duration", "days", "hours",
    # compensation
    "salary", "wage", "pay",
    # fuel / logistics
    "fuel", "efficiency", "mileage",
])


def detect_aggregate_metric_not_found(question: str, columns: Iterable[str]) -> bool:
    """Return True when the question asks for an aggregate (average / mean /
    median / sum) of a specific metric that does not correspond to any column
    in ``columns`` *and* does not match any known business-metric alias word.

    This is a fail-closed guard for questions like "What is the average shoe
    size of our customers?" on a dataset that has no shoe-related column: the
    controller must return INCONCLUSIVE rather than silently substituting an
    unrelated column and reporting a confident OBSERVED answer.

    Deliberately conservative (low false-positive):
    - Only fires for aggregate-verb questions (not every descriptive question).
    - Passes through if the question directly references any available column
      (checked by _question_references_any_column).
    - Passes through if any known alias word for a standard business metric
      appears in the question (the alias resolver will handle it).
    - Churn-flavored questions are excluded (owned by a separate path).
    """
    if not _AGGREGATE_VERB_RE.search(question or ""):
        return False
    ql = (question or "").lower()
    if any(sub in ql for sub in _CHURN_OWNED_SUBSTRINGS):
        return False
    # If the question directly names any available column, binding is possible.
    if _question_references_any_column(question, columns):
        return False
    # If any known alias metric word appears, the alias resolver can handle it.
    q_words = set(re.findall(r"\b[a-z]+\b", ql))
    if any(alias in q_words for alias in _KNOWN_METRIC_WORDS):
        return False
    # No column reference and no known alias → the metric is absent from the
    # dataset; return True so the controller can close INCONCLUSIVE.
    return True



def infer_binary_rate_indicator(question: str, df) -> str | None:
    """Resolve a rate/proportion noun to one uniquely named binary indicator.

    This is deliberately conservative: the question must contain rate/proportion
    language, the indicator's name must share a meaningful concept token with the
    noun immediately describing the rate, and exactly one binary-like column may
    match. It returns a physical column name; it does not calculate a rate.
    """
    if df is None or not _RATE_PROPORTION_RE.search(question or ""):
        return None
    q = (question or "").lower()
    match = re.search(r"\b([a-z][a-z0-9_-]*)\s+(?:rate|proportion|percentage|percent)\b", q)
    # Typical phrasing: "defect rate", "conversion rate", "failure rate".
    if not match:
        return None
    concept = match.group(1).replace("-", "_").rstrip("s")
    stop = {"the", "a", "an", "overall", "average", "total", "of", "by", "for"}
    if concept in stop or len(concept) < 3:
        return None

    def stem(token: str) -> str:
        token = token.lower().strip("_")
        for suffix in ("ed", "ing", "ive", "tion", "s"):
            if len(token) > len(suffix) + 3 and token.endswith(suffix):
                token = token[:-len(suffix)]
                break
        return token

    concept_stem = stem(concept)
    candidates = []
    for col in getattr(df, "columns", []):
        series = df[col].dropna()
        if series.empty or series.nunique() > 2:
            continue
        # Downstream execution computes the mean of this indicator. Only native
        # bool and numeric 0/1 series are compatible; categorical labels need an
        # explicit typed transform and must not be approved as executable inputs.
        if pd.api.types.is_bool_dtype(df[col]):
            binary = True
        elif pd.api.types.is_numeric_dtype(df[col]):
            values = set(series.unique().tolist())
            binary = values <= {0, 1, 0.0, 1.0}
        else:
            binary = False
        if not binary:
            continue
        parts = {stem(x) for x in re.findall(r"[a-z0-9]+", str(col).lower())}
        # Require a concept match, not merely a generic flag suffix.
        if concept_stem in parts or any(len(concept_stem) >= 4 and (concept_stem.startswith(x) or x.startswith(concept_stem)) for x in parts if len(x) >= 4):
            candidates.append(str(col))
    return candidates[0] if len(candidates) == 1 else None

def detect_unresolved_requested_variables(question: str, columns: Iterable[str]) -> List[str]:
    """Return the distinct identifier-style tokens named in `question` that
    do not correspond to any column in `columns` (case/separator-insensitive).

    Returns an empty list when nothing that looks like an explicitly named,
    nonexistent variable is found -- callers should treat that as "this
    gate has no objection", not as "the question is fully resolved".
    """
    normalized_cols = {_normalize_column_name(c) for c in columns}
    found: List[str] = []
    for tok in _IDENTIFIER_TOKEN_RE.findall(question or ""):
        tok_norm = tok.lower()
        if any(sub in tok_norm for sub in _CHURN_OWNED_SUBSTRINGS):
            continue
        if tok_norm in normalized_cols:
            continue
        if tok not in found:
            found.append(tok)
    return found


# Business measures that must never be silently replaced by a different column.
# concept word in the question -> substrings that make a column an acceptable binding.
_NAMED_MEASURES = {
    "profit": ("profit", "margin", "earning", "income"),
    "profits": ("profit", "margin", "earning", "income"),
    "margin": ("margin", "profit"),
    "cost": ("cost", "expense", "spend"),
    "costs": ("cost", "expense", "spend"),
    "expense": ("expense", "cost", "spend"),
    "expenses": ("expense", "cost", "spend"),
    "revenue": ("revenue", "sales", "amount", "total", "income", "gmv", "turnover"),
    "salary": ("salary", "wage", "pay", "compensation"),
}


def detect_named_measure_absent(question: str, columns: Iterable[str]) -> str | None:
    """Return the measure word (e.g. ``"profit"``) when the question names a business
    measure that no column plausibly carries, else ``None``.

    Fail-closed guard: the direct controller path must refuse such a question rather than
    bind it to an unrelated column (e.g. profit -> row COUNT).
    """
    ql = (question or "").lower()
    cols = [str(c).lower() for c in columns]
    for word in re.findall(r"\b[a-z]+\b", ql):
        accepted = _NAMED_MEASURES.get(word)
        if accepted and not any(sub in col for col in cols for sub in accepted):
            return word
    return None
