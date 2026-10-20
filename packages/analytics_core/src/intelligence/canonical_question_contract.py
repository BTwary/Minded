"""Canonical Question Contract Engine for AA-OS.

Enforces:
1. One immutable, schema-grounded QuestionContract representation.
2. Deterministic semantic phrase -> physical column resolution order:
   - exact physical column
   - normalized physical column
   - explicit approved semantic aliases
   - schema-derived lexical/stem similarity
   - observed-value-to-column binding
   - otherwise unresolved (never dataframe column order fallback).
3. General analytical outcome aliases (survival, fuel efficiency, tip, etc.).
4. Observed-value -> dimension binding ("male"/"female" -> sex, "credit card"/"cash" -> payment, etc.).
5. Preserves relation grammar (directional exposure -> outcome vs symmetric bivariate).
6. Correct task classification.
7. Multi-objective comparative association.
8. Canonical aggregation ownership.
"""
from __future__ import annotations

import re
import math
from datetime import date, datetime
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


def _canonical_json_value(value: Any) -> Any:
    """Return a strict JSON-native value for identity-bearing hashes.

    Supported scalar conversions are explicit and deterministic.  Unknown values are rejected
    instead of being silently coerced with ``default=str`` (which can collide or vary by type).
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite floats are not valid canonical JSON values.")
        return value
    if isinstance(value, np.generic):
        return _canonical_json_value(value.item())
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Canonical JSON mappings must use string keys.")
        return {key: _canonical_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError(f"Unsupported canonical JSON value type: {type(value).__name__}")


def canonical_json_dumps(value: Any) -> str:
    """Serialize known contract values canonically, rejecting unsupported types."""
    import json
    return json.dumps(
        _canonical_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


@dataclass(frozen=True)
class CanonicalQuestionContract:
    question_text: str
    task_family: str  # DESCRIPTIVE, RANKING, GROUP_COMPARISON, ASSOCIATION, TREND, CAUSAL_REQUEST, ROOT_CAUSE, MULTI_ASSOCIATION, COUNT, INTERACTION, PREDICTION
    claim_type: str   # OBSERVATION, ASSOCIATION, CAUSAL_INFERENCE, PREDICTION
    estimand: str     # mean, rate, count, sum, correlation, group_difference, trend, ranking, multi_correlation, interaction, root_cause
    target_column: Optional[str] = None
    explanatory_columns: Tuple[str, ...] = ()
    grouping_columns: Tuple[str, ...] = ()
    time_column: Optional[str] = None
    filter_conditions: Tuple[Dict[str, Any], ...] = ()
    requested_aggregation: Optional[str] = None  # Registry-backed aggregation name
    comparison_operator: Optional[str] = None
    comparison_values: Tuple[str, ...] = ()
    direction: Optional[str] = None
    ranking_direction: Optional[str] = None      # "DESC", "ASC"
    ranking_limit: Optional[int] = None          # explicit Top-N/Bottom-N bound, included in contract hash
    causal_language: bool = False
    requested_claim: str = "ASSOCIATION"         # CAUSAL, ASSOCIATION, OBSERVATION, PREDICTION
    supported_claim: str = "ASSOCIATION"         # OBSERVATION, ASSOCIATION, CAUSAL_INFERENCE, PREDICTION
    claim_ceiling: str = "ASSOCIATION"           # OBSERVATION, ASSOCIATION, CAUSAL_INFERENCE, PREDICTION
    resolution_confidence: float = 1.0
    unresolved_roles: Tuple[str, ...] = ()
    resolution_evidence: Dict[str, Any] = field(default_factory=dict)
    # Enforcement state (authority firewall, step 1). ``family`` names the enforced family the
    # question attempted (COUNT | SIMPLE_AGGREGATE) or NONE. ``completeness`` is COMPLETE |
    # INCOMPLETE | AMBIGUOUS | NOT_ENFORCED; NOT_ENFORCED (the default) leaves every
    # non-enforced family on its existing path. Do NOT infer a veto from ``target_column is None``.
    family: str = "NONE"
    completeness: str = "NOT_ENFORCED"
    unresolved_semantics: Tuple[str, ...] = ()
    share_of_total: bool = False
    requested_outputs: Tuple[str, ...] = ()
    # Stateful intent IR metadata.  Material values remain in the typed fields above; these
    # fields explain where each value came from and whether every material clause was consumed.
    # The ledger is deliberately evidence-oriented: it can make a plan unresolved, but cannot
    # promote an unresolved plan to executable.
    slot_provenance: Dict[str, str] = field(default_factory=dict)
    slot_confidence: Dict[str, float] = field(default_factory=dict)
    intent_ledger: Tuple[Dict[str, Any], ...] = ()
    # Explicit derived-metric semantics.  These are optional and must be confirmed/resolved
    # before a ratio/rate calculation can execute.
    numerator_column: Optional[str] = None
    denominator_column: Optional[str] = None
    time_range: Optional[Dict[str, Any]] = None
    # Optional typed multi-operation graph. When present, it is part of the canonical hash
    # and is persisted with the locked contract; execution requires the approved contract.
    operation_graph: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_text": self.question_text,
            "task_family": self.task_family,
            "claim_type": self.claim_type,
            "estimand": self.estimand,
            "target_column": self.target_column,
            "explanatory_columns": list(self.explanatory_columns),
            "grouping_columns": list(self.grouping_columns),
            "time_column": self.time_column,
            "filter_conditions": list(self.filter_conditions),
            "requested_aggregation": self.requested_aggregation,
            "comparison_operator": self.comparison_operator,
            "comparison_values": list(self.comparison_values),
            "direction": self.direction,
            "ranking_direction": self.ranking_direction,
            "ranking_limit": self.ranking_limit,
            "causal_language": self.causal_language,
            "requested_claim": self.requested_claim,
            "supported_claim": self.supported_claim,
            "claim_ceiling": self.claim_ceiling,
            "resolution_confidence": float(self.resolution_confidence),
            "unresolved_roles": list(self.unresolved_roles),
            "resolution_evidence": dict(self.resolution_evidence),
            "family": self.family,
            "completeness": self.completeness,
            "unresolved_semantics": list(self.unresolved_semantics),
            "share_of_total": self.share_of_total,
            "requested_outputs": list(self.requested_outputs),
            "slot_provenance": dict(self.slot_provenance),
            "slot_confidence": {str(k): float(v) for k, v in self.slot_confidence.items()},
            "intent_ledger": [dict(item) for item in self.intent_ledger],
            "numerator_column": self.numerator_column,
            "denominator_column": self.denominator_column,
            "time_range": dict(self.time_range) if self.time_range is not None else None,
            "operation_graph": dict(self.operation_graph) if self.operation_graph is not None else None,
        }

    def contract_hash(self) -> str:
        import hashlib
        d = self.to_dict()
        canonical_json = canonical_json_dumps(d)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def canonical_question_contract_from_dict(payload: Dict[str, Any]) -> CanonicalQuestionContract:
    """Reconstruct a canonical contract persisted as an audit-event payload."""
    tuple_fields = {
        "explanatory_columns",
        "grouping_columns",
        "filter_conditions",
        "comparison_values",
        "unresolved_roles",
        "unresolved_semantics",
        "requested_outputs",
        "intent_ledger",
    }
    allowed = set(CanonicalQuestionContract.__dataclass_fields__)
    values = {key: value for key, value in payload.items() if key in allowed}
    for key in tuple_fields:
        if key in values:
            values[key] = tuple(values[key] or ())
    return CanonicalQuestionContract(**values)


# Canonical reusable semantic aliases
APPROVED_SEMANTIC_ALIASES: Dict[str, List[str]] = {
    # Titanic
    "survival rate": ["survived"],
    "survival": ["survived"],
    "survive": ["survived"],
    "survived": ["survived"],
    "passenger class": ["passenger_class", "pclass", "class"],
    "class": ["passenger_class", "pclass", "class"],
    "men and women": ["sex", "who"],
    "men": ["sex", "who"],
    "women": ["sex", "who"],
    "male and female": ["sex"],
    "male": ["sex"],
    "female": ["sex"],
    "gender": ["sex"],
    # MPG
    "fuel efficiency": ["mpg"],
    "fuel economy": ["mpg"],
    "mileage": ["mpg"],
    "mpg": ["mpg"],
    "model year": ["model_year", "year"],
    "model years": ["model_year", "year"],
    "horsepower": ["horsepower"],
    "weight": ["weight", "body_mass_g"],
    "origin": ["origin"],
    # Tips
    "party size": ["size"],
    "size": ["size"],
    "tip amount": ["tip"],
    "tip": ["tip"],
    "tips": ["tip"],
    "gratuity": ["tip"],
    "total bill": ["total_bill", "bill"],
    "bill": ["total_bill", "bill"],
    "day": ["day"],
    "smoker": ["smoker"],
    "smokers": ["smoker"],
    "non smokers": ["smoker"],
    "non-smokers": ["smoker"],
    # Penguins
    "body mass": ["body_mass_g"],
    "weigh": ["body_mass_g"],
    "flipper length": ["flipper_length_mm"],
    "bill length": ["bill_length_mm"],
    "bill depth": ["bill_depth_mm"],
    "species": ["species"],
    "sex": ["sex"],
    # Diamonds
    "price": ["price"],
    "carat": ["carat"],
    "cut": ["cut"],
    "clarity": ["clarity"],
    "color": ["color"],
    # Planets
    "discovery method": ["method"],
    "method": ["method"],
    "planets": ["method"],
    "orbital period": ["orbital_period"],
    "mass": ["mass"],
    "distance": ["distance"],
    "year": ["year"],
    # Taxis
    "pickup borough": ["pickup_borough"],
    "fare": ["fare"],
    "payment": ["payment", "payment_type"],
    # Flights
    "number of passengers": ["passengers"],
    "passenger count": ["passengers"],
    "passengers": ["passengers"],
    "passenger": ["passengers"],
    "month": ["month"],
    # General / eCommerce / Subscriptions
    # Dedicated AOV columns first; otherwise the order-level monetary column (one row per order is
    # verified separately via the proven grain -- an identifier column is NEVER a valid AOV target).
    "average order value": ["aov", "order_value", "total_amount", "order_total", "revenue", "sales_amount", "sales"],
    "aov": ["aov", "order_value", "total_amount", "order_total", "revenue", "sales_amount", "sales"],
    "order value": ["aov", "order_value", "total_amount", "amount", "order_total", "revenue", "sales_amount", "sales"],
    "order value after discounts": ["aov", "order_value", "total_amount", "order_total", "revenue", "sales_amount", "sales"],
    "sales": ["sales", "net_revenue", "gross_revenue", "revenue", "sales_amount", "total_amount"],
    "revenue": ["net_revenue", "gross_revenue", "revenue", "sales_amount", "sales", "total_amount"],
    "customer segment": ["customer_segment", "segment"],
    "customer segments": ["customer_segment", "segment"],
    "segment": ["customer_segment", "segment"],
    "segments": ["customer_segment", "segment"],
    "churn": ["churn_event", "churned", "churn", "is_churned"],
    "churned": ["churn_event", "churned", "churn", "is_churned"],
    "churn rate": ["churn_event", "churned", "churn", "is_churned"],
    "retention": ["retention_rate", "retention", "is_retained", "retained"],
    "return rate": ["returned", "is_returned", "return_flag", "returned_flag"],
    "refund rate": ["refund_flag", "is_refunded", "refunded", "refund"],
    "returns": ["returned", "is_returned", "return_flag"],
    "cancellation rate": ["cancelled", "is_cancelled", "canceled", "cancelled_flag"],
    "late delivery rate": ["late_delivery", "is_late", "late_flag"],
    "retention rate": ["retention_rate", "retention", "is_retained", "retained"],
    "conversion": ["conversion_rate", "conversion", "is_converted", "converted"],
    "conversion rate": ["conversion_rate", "conversion", "is_converted", "converted"],
    "traffic source": ["traffic_source", "source", "channel"],
    "traffic sources": ["traffic_source", "source", "channel"],
    "traffic": ["traffic_source", "source", "channel"],
    "advertising spend": ["advertising_spend", "ad_spend", "spend", "marketing_spend"],
    "ad spend": ["advertising_spend", "ad_spend", "spend", "marketing_spend"],
    "plan tier": ["plan_tier", "plan"],
    "plan tiers": ["plan_tier", "plan"],
    "subscription plan": ["plan_tier", "plan"],
    "subscription plans": ["plan_tier", "plan"],
    "subscription tier": ["plan_tier", "plan"],
    "subscription tiers": ["plan_tier", "plan"],
    "tiers of subscriptions": ["plan_tier", "plan"],
    "tiers": ["plan_tier", "tier", "plan"],
    "tier": ["plan_tier", "tier", "plan"],
    "tenure": ["tenure_days", "tenure"],
    "support tickets": ["support_tickets", "tickets"],
}

# Stem prefixes for invariant lexical root matching
STEM_PREFIXES = {
    "surviv": "survived",
    "passeng": "passengers",
    "orbit": "orbital_period",
    "retent": "retention_rate",
    "convers": "conversion_rate",
}

# Stop words to ignore during category value scanning
_VALUE_STOP_WORDS = {
    "the", "in", "of", "and", "or", "to", "a", "an", "is", "for", "on", "by",
    "with", "it", "at", "as", "be", "do", "does", "did", "no", "yes", "all",
    "any", "some", "both", "each", "more", "less", "than", "between", "over",
    "under", "from", "into", "highest", "lowest", "average", "total", "rate",
    "what", "which", "why", "how", "who", "when", "where", "have", "has", "had",
    "grow", "grown", "change", "changed", "differ", "differs", "affect", "drive",
}


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def strip_aggregation_words(phrase: str) -> str:
    cleaned = re.sub(r"^(?:standard\s+deviation|stddev|stdev|variance|var|average|avg|mean|total|sum|median|mode|most\s+common|most\s+frequent|interquartile\s+range|iqr|range|first\s+quartile|third\s+quartile|lower\s+quartile|upper\s+quartile|missing\s+values?|null\s+values?|blank\s+values?|count\s+of|number\s+of|rate\s+of|the)\s+", "", phrase.strip(), flags=re.I)
    cleaned = re.sub(r"\s+(?:after|before|by|for|during|in)\s+.*$", "", cleaned.strip(), flags=re.I)
    cleaned = re.sub(r"\s+(?:amount|level|rate|score|value|count)$", "", cleaned.strip(), flags=re.I)
    return cleaned.strip()


_MEASURE_TAIL_WORDS = {"value", "amount", "price", "cost", "spend", "revenue", "sales", "total", "income", "worth"}
_IDENTIFIER_TAIL_WORDS = {"id", "key", "uuid", "guid"}


def _is_identifier_column(col: str) -> bool:
    toks = normalize_token(col).split()
    return bool(toks) and toks[-1] in _IDENTIFIER_TAIL_WORDS


def _is_measure_phrase(phrase: str) -> bool:
    toks = normalize_token(phrase).split()
    return bool(toks) and toks[-1] in _MEASURE_TAIL_WORDS


def ambiguous_semantic_candidates(phrase: str, available_cols: Sequence[str]) -> List[str]:
    """Return candidate columns for a deliberately ambiguous high-frequency measure phrase.

    This is used to keep a recognized analytical operation intact while asking the user to
    choose its measure. It is not a resolver: callers must not select a candidate automatically.
    """
    p_norm = normalize_token(phrase)
    if p_norm != "revenue":
        return []
    # An exact physical `revenue` field is resolved before semantic aliases, so only return
    # alternatives when no exact field wins and more than one revenue-named column exists.
    if any(normalize_token(c) == p_norm for c in available_cols):
        return []
    candidates = [
        str(c) for c in available_cols
        if re.search(r"\brevenue\b", normalize_token(str(c)))
    ]
    return candidates if len(candidates) > 1 else []


def resolve_phrase_to_column(phrase: str, available_cols: Sequence[str]) -> Optional[str]:
    """Resolve a single phrase to a physical column using deterministic precedence:
    1. Exact match
    2. Normalized physical match
    3. Approved semantic alias
    4. Lexical/stem similarity
    5. High-confidence fuzzy match (with >= 0.80 and margin >= 0.10)
    """
    p_norm = normalize_token(phrase)
    if not p_norm:
        return None
    # A phrase naming a MEASURE ("order value", "customer spend") must never be loosely matched
    # (containment / fuzzy / stripped-containment) onto an IDENTIFIER column: "order value" -> order_id
    # silently turned "average order value" into a count/mean of order ids. Identifier columns may
    # still match exactly, by normalized name, or through an approved alias.
    measure_phrase = _is_measure_phrase(phrase)

    # 1. Exact physical match
    for c in available_cols:
        if c.lower() == phrase.strip().lower():
            return c

    # 2. Normalized physical column match
    for c in available_cols:
        c_norm = normalize_token(c)
        if c_norm == p_norm:
            return c
        c_snake = re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
        p_snake = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
        if c_snake == p_snake:
            return c

    # 3. Approved semantic aliases
    # "Revenue" is one of the most frequent business terms, but datasets often have
    # several distinct measures (net_revenue, gross_revenue, realized_revenue). When
    # there is no exact physical `revenue` column, bind the generic word only if the
    # schema contains one revenue-named field. If several exist, return unresolved so
    # the user can choose; never let alias-list ordering silently pick a measure.
    if p_norm == "revenue":
        revenue_columns = ambiguous_semantic_candidates(phrase, available_cols)
        if len(revenue_columns) == 0:
            # Zero candidates fall through to the ordinary semantic aliases below; one exact
            # candidate is returned by the helper's caller only when it is uniquely established.
            exact_revenue_columns = [
                c for c in available_cols
                if re.search(r"\brevenue\b", normalize_token(c))
            ]
            if len(exact_revenue_columns) == 1:
                return exact_revenue_columns[0]
        if len(revenue_columns) > 0:
            return None

    aliases = APPROVED_SEMANTIC_ALIASES.get(p_norm, [])
    for target_cand in aliases:
        for c in available_cols:
            if c.lower() == target_cand.lower() or normalize_token(c) == normalize_token(target_cand):
                return c

    # 4. Multi-word phrase containment (e.g. 'flipper length' in 'flipper_length_mm')
    for c in available_cols:
        if measure_phrase and _is_identifier_column(c):
            continue
        c_norm = normalize_token(c)
        if p_norm in c_norm or c_norm in p_norm:
            # Require at least 4 chars and meaningful overlap
            if len(p_norm) >= 4 and len(c_norm) >= 4:
                return c

    # 5. Stem prefix matching
    for prefix, target_cand in STEM_PREFIXES.items():
        if p_norm.startswith(prefix):
            for c in available_cols:
                if c.lower() == target_cand.lower() or c.lower().startswith(prefix):
                    return c

    # 6. High-confidence fuzzy match
    candidates: List[Tuple[str, float]] = []
    for c in available_cols:
        if measure_phrase and _is_identifier_column(c):
            continue
        c_norm = normalize_token(c)
        sim = SequenceMatcher(None, p_norm, c_norm).ratio()
        if sim >= 0.80:
            candidates.append((c, sim))
    candidates.sort(key=lambda x: -x[1])
    if candidates:
        top = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        if second is None or (top[1] - second[1] >= 0.10):
            return top[0]

    # Try stripping aggregation tokens if present
    stripped = strip_aggregation_words(phrase)
    if stripped and stripped.lower() != phrase.lower():
        p_stripped_norm = normalize_token(stripped)
        if p_stripped_norm and p_stripped_norm != p_norm:
            # Re-check exact/normalized/aliases for stripped version
            for c in available_cols:
                if c.lower() == stripped.lower() or normalize_token(c) == p_stripped_norm:
                    return c
            for target_cand in APPROVED_SEMANTIC_ALIASES.get(p_stripped_norm, []):
                for c in available_cols:
                    if c.lower() == target_cand.lower() or normalize_token(c) == normalize_token(target_cand):
                        return c
            for c in available_cols:
                if measure_phrase and _is_identifier_column(c):
                    continue
                c_norm = normalize_token(c)
                if p_stripped_norm in c_norm or c_norm in p_stripped_norm:
                    if len(p_stripped_norm) >= 4 and len(c_norm) >= 4:
                        return c

    # Try stripping prepositional suffixes (e.g. "after discounts", "by day")
    prep_stripped = re.sub(r"\s+(?:after|before|by|for|during|in)\s+.*$", "", phrase.strip(), flags=re.I)
    if prep_stripped and prep_stripped.strip().lower() != phrase.strip().lower():
        cand = resolve_phrase_to_column(prep_stripped.strip(), available_cols)
        if cand:
            return cand

    return None


def resolve_phrase_to_column_list(phrase: str, available_cols: Sequence[str]) -> List[str]:
    """Resolve a compound phrase (e.g. 'price and marketing_spend' or 'price, marketing_spend, and discount') to a list of physical columns."""
    p_str = phrase.strip()
    if not p_str:
        return []

    # If the phrase has conjunctions ('and', '&') or commas, split first!
    if re.search(r"\b(?:and|&)\b|,", p_str, flags=re.I):
        parts = re.split(r",?\s+(?:and|&)\s+|,|\s+and\s+", p_str, flags=re.I)
        cols_found: List[str] = []
        for part in parts:
            clean_part = part.strip()
            if not clean_part:
                continue
            c = resolve_phrase_to_column(clean_part, available_cols)
            if c and c not in cols_found:
                cols_found.append(c)
        if cols_found:
            return cols_found

    single = resolve_phrase_to_column(p_str, available_cols)
    return [single] if single else []



def scan_observed_category_values(
    question: str, df: pd.DataFrame
) -> Tuple[Optional[str], List[str]]:
    """Scan dataframe categorical/object columns for observed values mentioned in the question.
    Returns (grouping_column, [matched_values]) if an unambiguous column match is found.
    """
    ql = question.lower()
    matches_by_col: Dict[str, List[str]] = {}

    for c in df.columns:
        s = df[c]
        # Only inspect low/medium cardinality columns
        is_cat = (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or (pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 30)
        )
        if not is_cat:
            continue
        try:
            unique_vals = [v for v in s.dropna().unique() if pd.notna(v)]
        except Exception:
            continue
        if len(unique_vals) > 60:
            continue

        for val in unique_vals:
            v_str = str(val).strip()
            v_norm = normalize_token(v_str)
            if len(v_norm) < 2 or v_norm in _VALUE_STOP_WORDS:
                continue

            # Look for exact word boundary in question
            pattern = rf"\b{re.escape(v_norm)}\b"
            if re.search(pattern, ql):
                if c not in matches_by_col:
                    matches_by_col[c] = []
                if v_str not in matches_by_col[c]:
                    matches_by_col[c].append(v_str)

    if not matches_by_col:
        return None, []

    # Sort columns by number of matched values descending
    sorted_cols = sorted(matches_by_col.items(), key=lambda x: len(x[1]), reverse=True)
    best_col, best_vals = sorted_cols[0]

    # Check for ambiguity: if top two columns tie on matched values, fail closed
    if len(sorted_cols) > 1 and len(sorted_cols[0][1]) == len(sorted_cols[1][1]):
        if set(sorted_cols[0][1]) == set(sorted_cols[1][1]):
            return None, []

    # If top column has >= 2 matched values, it's uniquely strong (e.g. "male" and "female")
    if len(best_vals) >= 2:
        return best_col, best_vals

    # If only 1 column matched, return it
    if len(sorted_cols) == 1:
        return best_col, best_vals

    # If multiple columns tied with 1 match, check if one matches uniquely on a longer word
    if len(best_vals) == 1 and len(sorted_cols) > 1:
        other_matches = [m for col, m in sorted_cols[1:] if best_vals[0] in m]
        if not other_matches:
            return best_col, best_vals

    return best_col, best_vals


INTENSIVE_METRIC_KEYWORDS = (
    "mpg", "efficiency", "orbital", "period", "price", "fare", "tip",
    "mass", "weight", "rate", "age", "horsepower", "carat", "bill_length", "flipper_length"
)


def resolve_entity_count_column(phrase: str, available_cols: Sequence[str]) -> Optional[str]:
    """Resolve entity noun phrases like 'users', 'customers', 'accounts' to a unique entity ID column."""
    q_norm = normalize_token(phrase)
    if not q_norm:
        return None
    entity_tokens = {
        "user": {"user_id", "customer_id", "account_id", "member_id"},
        "users": {"user_id", "customer_id", "account_id", "member_id"},
        "customer": {"customer_id", "user_id", "account_id", "member_id"},
        "customers": {"customer_id", "user_id", "account_id", "member_id"},
        "account": {"account_id", "customer_id", "user_id", "member_id"},
        "accounts": {"account_id", "customer_id", "user_id", "member_id"},
        "person": {"customer_id", "user_id", "account_id", "member_id"},
        "people": {"customer_id", "user_id", "account_id", "member_id"},
        "client": {"customer_id", "user_id", "account_id", "member_id"},
        "clients": {"customer_id", "user_id", "account_id", "member_id"},
        "order": {"order_id"},
        "orders": {"order_id"},
        "transaction": {"transaction_id", "order_id"},
        "transactions": {"transaction_id", "order_id"},
        "purchase": {"purchase_id", "order_id"},
        "purchases": {"purchase_id", "order_id"},
    }
    recognized_entity = False
    for entity, col_names in entity_tokens.items():
        if q_norm == entity or q_norm.startswith(entity):
            recognized_entity = True
            for c in available_cols:
                cn = normalize_token(c).replace(" ", "_")
                if cn in col_names:
                    return c
    if recognized_entity:
        return None

    # fallback: any *_id column that reads like an entity identifier and is not a numeric measure
    entity_like = []
    for c in available_cols:
        cn = normalize_token(c).replace(" ", "_")
        if cn.endswith("_id") or "_id" in cn:
            if any(token in cn for token in ("customer", "user", "account", "member", "person", "client")):
                # The counted noun must actually name this column (singular/plural tolerant);
                # otherwise "how many zorblats" would silently bind the only *_id column.
                stems = {t[:-1] if t.endswith("s") and len(t) > 3 else t for t in re.split(r"[^a-z0-9]+", q_norm) if t}
                if any(st and st in cn for st in stems):
                    entity_like.append(c)
    if len(entity_like) == 1:
        return entity_like[0]
    return None


_NEGATION_BEFORE_VALUE_RE = re.compile(
    r"\b(?:not|non|never|isn't|aren't|wasn't|weren't|excluding|exclude|except|without|other\s+than|aside\s+from)\s+(?:\w+\s+){0,2}$",
    re.I,
)


def extract_observed_value_filters(
    question: str,
    df: pd.DataFrame,
    existing: Sequence[Mapping[str, Any]] = (),
    skip_columns: Sequence[str] = (),
) -> Tuple[Dict[str, Any], ...]:
    """Population filters from category values the user names verbatim ("Corporate orders",
    "delivered and not returned").

    A value becomes a filter only when it is observed in exactly one categorical column, so no
    column is ever guessed. Negation ("not returned", "excluding X") yields ``!=``. Values that
    are also words of a column name, columns already filtered with the same value, and the
    grouping/target columns are skipped. Two positive values of one column joined only by
    "or"/commas become an IN list; any other multi-value use is left unresolved (not guessed).
    """
    ql = " ".join(str(question or "").split())
    low = ql.casefold()
    normalized_question = normalize_token(ql)
    skip = {str(c).casefold() for c in skip_columns}
    column_words = {w for c in df.columns for w in re.findall(r"[a-z0-9]+", str(c).casefold())}
    seen: Dict[Tuple[int, int], List[Tuple[str, Any]]] = {}
    # Bind an ordinal value to a categorical *band* when the question names the
    # field in the same phrase, e.g. "orders with high support load" ->
    # support_load_band = 'High'.  The generic value scan below intentionally
    # ignores values that are also column-name words; that safety rule correctly
    # avoids guessing, but it also used to drop this explicit field/value phrase
    # because another column was named `high_support_flag`.
    band_filters: List[Dict[str, Any]] = []
    ordinal_values = {"high", "low", "medium", "very high", "very low", "moderate", "elevated"}
    for col in df.columns:
        col_name = str(col)
        if col_name.casefold() in skip:
            continue
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)):
            continue
        normalized_column = normalize_token(col_name)
        if not normalized_column.endswith(" band"):
            continue
        field_phrase = normalized_column[:-len(" band")].strip()
        if not field_phrase:
            continue
        observed_values = [str(value).strip() for value in series.dropna().unique()]
        for observed in observed_values:
            value_phrase = normalize_token(observed)
            if value_phrase not in ordinal_values:
                continue
            value_re = re.escape(value_phrase)
            field_re = re.escape(field_phrase)
            leading = re.search(rf"\b{value_re}\s+{field_re}\b", normalized_question)
            trailing = re.search(rf"\b{field_re}\s+(?:is|was|rated|at)?\s*{value_re}\b", normalized_question)
            match = leading or trailing
            if not match:
                continue
            # A negation attached to the matched qualifier means the opposite band.
            before = normalized_question[max(0, match.start() - 18):match.start()]
            operator = "!=" if re.search(r"\b(?:not|never|without|excluding|exclude|except)\s*$", before) else "="
            value = next((candidate for candidate in observed_values if candidate.casefold() == observed.casefold()), observed)
            already_bound = any(
                str(item.get("column", "")).casefold() == col_name.casefold()
                and str(item.get("operator", "")).upper() == operator
                and str(item.get("value", "")).casefold() == str(value).casefold()
                for item in existing
            )
            if not already_bound and not any(
                item["column"].casefold() == col_name.casefold()
                for item in band_filters
            ):
                band_filters.append({"column": col_name, "operator": operator, "value": value})
            break
    for col in df.columns:
        if str(col).casefold() in skip:
            continue
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)):
            continue
        try:
            values = list(series.dropna().unique())
        except Exception:
            continue
        if not values or len(values) > 60:
            continue
        for value in values:
            text = str(value).strip()
            norm = normalize_token(text)
            if len(norm) < 3 or norm in _VALUE_STOP_WORDS or norm in column_words:
                continue
            for m in re.finditer(rf"(?<![\w]){re.escape(text.casefold())}(?![\w])", low):
                seen.setdefault((m.start(), m.end()), []).append((str(col), value))
    # A phrase observed in several columns is ambiguous: never pick a column for the user.
    resolved: List[Tuple[int, int, str, Any]] = []
    for (start, end), hits in sorted(seen.items()):
        distinct = {(c, str(v).casefold()) for c, v in hits}
        if len({c for c, _ in distinct}) != 1 or len(distinct) != 1:
            continue
        col, value = hits[0]
        resolved.append((start, end, col, value))

    def already(col: str, value: Any) -> bool:
        for f in existing:
            if str(f.get("column", "")).casefold() != col.casefold():
                continue
            fv = f.get("value")
            fv = fv if isinstance(fv, (list, tuple)) else [fv]
            if str(value).casefold() in {str(x).casefold() for x in fv}:
                return True
        return False

    out: List[Dict[str, Any]] = []
    positives: Dict[str, List[Tuple[int, int, Any]]] = {}
    for start, end, col, value in resolved:
        if already(col, value):
            continue
        if _NEGATION_BEFORE_VALUE_RE.search(low[max(0, start - 28):start]):
            out.append({"column": col, "operator": "!=", "value": value})
        else:
            positives.setdefault(col, []).append((start, end, value))
    for col, items in positives.items():
        if any(str(f.get("column", "")).casefold() == col.casefold() and str(f.get("operator", "")).upper() in {"=", "IN"}
               for f in list(existing) + out):
            continue  # the column is already constrained; do not stack a second positive value
        if len(items) == 1:
            out.append({"column": col, "operator": "=", "value": items[0][2]})
        else:
            between = " ".join(low[items[i][1]:items[i + 1][0]] for i in range(len(items) - 1))
            if re.fullmatch(r"(?:\s|,|or|/)*", between):
                out.append({"column": col, "operator": "IN", "value": [v for _, _, v in items]})
    return tuple([*band_filters, *out])



_FLAG_GENERIC_TOKENS = {"delivery", "customer", "order", "support", "flag", "is", "has", "count", "band", "rate", "score", "used", "completed_flag"}


def _flag_population_filters(
    question: str, df: pd.DataFrame, existing: Sequence[Mapping[str, Any]], exclude_cols: Set[str],
) -> List[Dict[str, Any]]:
    """Bind 0/1 flag columns the question states in plain words (\"not returned\" -> returned = 0).

    Only a flag whose every content word is stated is bound; anything ambiguous is left for the human.
    """
    ql = " ".join(str(question).lower().split())
    covered = {str(f.get("column", "")).casefold() for f in existing}
    skip = {str(c).casefold() for c in exclude_cols}
    found: List[Dict[str, Any]] = []
    for c in df.columns:
        if str(c).casefold() in covered or str(c).casefold() in skip:
            continue
        s = df[c]
        if not (pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s)):
            continue
        vals = set(pd.Series(s.dropna().unique()).astype(int).tolist()) if len(s.dropna()) else set()
        if vals != {0, 1}:
            continue
        tokens = [t for t in re.split(r"[^a-z0-9]+", str(c).lower()) if t]
        stems: List[str] = []
        for t in tokens:
            if t in _FLAG_GENERIC_TOKENS or len(t) < 3:
                continue
            stems.append(t)
            for suf in ("ed", "d", "ing", "s"):
                if t.endswith(suf) and len(t) - len(suf) >= 3:
                    stems.append(t[: -len(suf)])
        if len(tokens) > 1:
            stems.append(" ".join(tokens))
        # Every non-generic token of the column name must be stated, else "returned" would
        # also bind "late_and_returned".
        _content = [t for t in tokens if t not in _FLAG_GENERIC_TOKENS and t not in {"and", "or", "of", "the"} and len(t) >= 3]
        if any(not re.search(rf"\b{re.escape(t)}", ql) for t in _content):
            continue
        stems = sorted(set(stems), key=len, reverse=True)
        if not stems:
            continue
        kw = "|".join(re.escape(x) for x in stems)
        neg = re.search(rf"\b(?:not|never|non|without|isn'?t|wasn'?t|weren'?t|didn'?t|no)(?:\s+|-)(?:been\s+)?(?:{kw})\b", ql)
        pos = re.search(
            rf"\b(?:that|which|who)\s+(?:were|was|are|is|had|have|has)\s+(?:{kw})\b",
            ql,
        )
        if neg:
            found.append({"column": c, "operator": "=", "value": 0})
        elif pos:
            found.append({"column": c, "operator": "=", "value": 1})
    return found


def extract_filter_conditions(question: str, available_cols: Sequence[str]) -> Tuple[Dict[str, Any], ...]:
    """Extract a conservative, schema-grounded subset of population predicates.

    Explicit comparisons support arbitrary dataset columns. Common year/date and
    dimension-value forms are supported when the relevant column is unique. This
    function intentionally does not guess among multiple candidate columns; the
    contract compiler marks unresolved filter intent as incomplete instead.
    """
    ql = " ".join((question or "").strip().split())
    conditions: List[Dict[str, Any]] = []
    used: set[tuple[str, str, str]] = set()

    def add(col: str, op: str, value: Any = None) -> None:
        key = (str(col).casefold(), str(op).upper(), repr(value))
        if key not in used:
            conditions.append({"column": col, "operator": str(op).upper(), **({} if value is None and str(op).upper() in {"IS NULL", "IS NOT NULL"} else {"value": value})})
            used.add(key)

    # Explicit comparisons: `discount_rate > 0.1`, `region = 'North'`,
    # `status is not cancelled`. Match against the actual schema, not a token list.
    schema_cols = sorted((str(c) for c in available_cols), key=len, reverse=True)
    col_alt = "|".join(re.escape(c) for c in schema_cols)

    # A same-column alternative in a WHERE clause should become IN/NOT IN, not a
    # single literal that swallows the rest of the question. This is especially
    # important for requests such as "excluding records where order_status is
    # Returned or Cancelled. Return the total for every region." The exclusion
    # language negates the whole set: status NOT IN ('Returned', 'Cancelled').
    # Only consume this high-precision form when the field is an exact schema
    # column and there are at least two explicit alternatives; otherwise the
    # regular resolver below handles the simpler forms or leaves them unresolved.
    filter_parse_question = ql
    handled_alternative_predicate = False
    where_body = re.search(
        r"\bwhere\s+(?P<body>.+?)(?=\.\s*(?:return|show|report|provide|give|list|display|calculate)\b|[.!?;](?:\s|$)|$)",
        ql,
        re.I,
    )
    if where_body:
        body = where_body.group("body")
        body_start = where_body.start("body")
        # A comma / "and" / "then" followed by a task verb starts a new clause
        # ("... is Delivered, calculate the total ..."); it is never another
        # alternative value of the predicate.
        clause_break = re.search(
            r"\s*(?:,|;|\b(?:and|then))\s*(?:then\s+|please\s+)?"
            r"(?:calculate|compute|return|show|report|provide|give|list|display|find|rank|sort|group|count|sum|get|what|how|which|order)\b",
            body,
            re.I,
        )
        if clause_break:
            body = body[:clause_break.start()]
        body_end = body_start + len(body)
        body = body.strip()
        prefix = ql[max(0, where_body.start() - 120):where_body.start()]
        is_exclusion = bool(re.search(r"\b(?:exclude|excluding|without|ignore)\b", prefix, re.I))
        for candidate_column in schema_cols:
            candidate = re.fullmatch(
                rf"\s*{re.escape(candidate_column)}\s+(?:is|equals?|=)\s*(?P<values>.+?)\s*",
                body,
                re.I,
            )
            if not candidate:
                continue
            raw_values = candidate.group("values").strip()
            # Restrict this conversion to clear alternative lists. Keep ordinary
            # predicates and values containing no list conjunction on the regular path.
            if not re.search(r"\s+or\s+|,", raw_values, re.I):
                break
            values = [
                item.strip().strip("\"'").strip()
                for item in re.split(r"\s+or\s+|\s*,\s*", raw_values, flags=re.I)
            ]
            if len(values) < 2 or any(not item for item in values):
                break
            values = [re.sub(r"[.!?]+$", "", item).strip() for item in values]
            if any(not item for item in values):
                break
            # Alternatives are short literals. A value that names another schema
            # column or runs on like a sentence is a swallowed clause: do not
            # convert, leave the predicate to the conservative resolver.
            lowered_cols = {str(c).casefold() for c in available_cols}
            if any(len(item.split()) > 4 or item.casefold() in lowered_cols for item in values):
                break
            add(candidate_column, "NOT IN" if is_exclusion else "IN", values)
            # Keep surrounding task/grouping language visible to other resolvers,
            # but hide the predicate already converted to typed semantics so generic
            # regexes cannot append a contradictory equality/exclusion.
            filter_parse_question = ql[:body_start] + (" " * (body_end - body_start)) + ql[body_end:]
            handled_alternative_predicate = True
            break
    if col_alt:
        explicit = re.compile(
            rf"(?<![\w])(?P<column>{col_alt})(?![\w])\s*(?P<op>>=|<=|!=|<>|=|>|<|\bis\s+not\b|\bis\b|\bequals?\b)\s*(?P<value>\"[^\"]*\"|'[^']*'|[^,;?]+?)(?=\s+(?:and|but)\s+|[,;?]|$)",
            re.I,
        )
        for match in explicit.finditer(filter_parse_question):
            col = next((c for c in available_cols if str(c).casefold() == match.group("column").casefold()), match.group("column"))
            op = match.group("op").casefold()
            raw = match.group("value").strip().strip("\"'").strip()
            if op in {"is", "is not"} and raw.casefold() in {"null", "none", "missing", "blank"}:
                add(col, "IS NULL" if op == "is" else "IS NOT NULL")
                continue
            op = {"equals": "=", "equal": "=", "is": "=" , "is not": "!=" , "<>": "!="}.get(op, op)
            if not raw:
                continue
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", raw):
                value: Any = float(raw) if "." in raw else int(raw)
            elif raw.casefold() in {"true", "false"}:
                value = raw.casefold() == "true"
            else:
                value = raw
            add(str(col), op, value)

    # Schema-grounded calendar year and explicit ISO date range. Use half-open
    # ranges to avoid excluding timestamps later on the final day.
    date_cols = [str(c) for c in available_cols if re.search(r"date|time|timestamp|created|placed|ordered|transaction", str(c), re.I)]
    year_match = re.search(r"\b(?:in|during|for|from)\s+(20\d{2})\b|\b(?:placed|ordered|created|signed\s+up)\s+in\s+(20\d{2})\b", ql, re.I)
    if year_match and len(date_cols) == 1:
        year = int(next(g for g in year_match.groups() if g))
        date_col = date_cols[0]
        add(date_col, ">=", f"{year:04d}-01-01")
        add(date_col, "<", f"{year + 1:04d}-01-01")

    date_range = re.search(r"\b(?:between|from)\s+(\d{4}-\d{2}-\d{2})\s+(?:and|to|through)\s+(\d{4}-\d{2}-\d{2})\b", ql, re.I)
    if date_range and len(date_cols) == 1:
        start, end = date_range.groups()
        try:
            start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
            if start_dt <= end_dt:
                add(date_cols[0], ">=", start)
                add(date_cols[0], "<", (end_dt + pd.Timedelta(days=1)).date().isoformat())
        except (TypeError, ValueError):
            pass

    # Common natural-language dimension filters: `in the North region`,
    # `region North`, `for status completed`. Only resolve a single named dimension.
    # Prevent grouping (`by region`), ranking (`rank regions`), conjunctions (`and channel`),
    # or column names from being swallowed as implicit filter values.
    dimension_names = ("region", "state", "country", "segment", "category", "status", "channel", "tier", "product", "sku")
    disallowed_filter_values = {
        "and", "or", "by", "per", "across", "for", "in", "with", "without", "where",
        "rank", "ranking", "order", "sort", "group", "grouping", "calculate",
        "sum", "total", "avg", "average", "count", "mean", "median", "variance",
        "stddev", "min", "max", "the", "a", "an", "each", "every", "all", "is", "are", "of", "to"
    } | {str(c).casefold() for c in available_cols} | {f"{str(c).casefold()}s" for c in available_cols}

    for dim in dimension_names:
        candidates = [str(c) for c in available_cols if re.sub(r"[^a-z0-9]", "", str(c).casefold()).endswith(dim)]
        if len(candidates) != 1:
            continue
        col = candidates[0]

        # Form 1: for/in (the) <val> <dim> e.g. 'for active status', 'in the North region', 'for East region'
        pat_val_dim = rf"\b(?:for|in)\s+(?:the\s+)?([A-Za-z0-9_-]+)\s+{re.escape(dim)}\b"
        m = re.search(pat_val_dim, filter_parse_question, re.I)
        if m:
            val = m.group(1).strip().strip("'\"")
            if val and val.casefold() not in disallowed_filter_values:
                add(col, "=", val)
                continue

        # Form 2: explicit dim = val or dim is val
        pat_eq = rf"\b{re.escape(dim)}\s+(?:is\s+|equals?\s+|equal\s+to\s+|=)\s*['\"]?([A-Za-z0-9_-]+)['\"]?"
        m = re.search(pat_eq, filter_parse_question, re.I)
        if m:
            val = m.group(1).strip().strip("'\"")
            if val and val.casefold() not in disallowed_filter_values:
                add(col, "=", val)
                continue

        # Form 3: for/in <dim> <val> e.g. 'for region East', 'for status active'
        pat_dim_val_prep = rf"\b(?:for|in)\s+{re.escape(dim)}\s+([A-Za-z0-9_-]+)\b"
        m = re.search(pat_dim_val_prep, filter_parse_question, re.I)
        if m:
            val = m.group(1).strip().strip("'\"")
            if val and val.casefold() not in disallowed_filter_values:
                add(col, "=", val)
                continue

        # Form 4: dim <val> when NOT used purely as grouping or ranking instruction
        # If dim is used in a grouping clause ('by region', 'by region and channel'), only match if an explicit value precedes a secondary clause break
        pat_dim_val = rf"\b{re.escape(dim)}\s+([A-Za-z0-9_-]+)(?=\s+(?:split\s+by|grouped\s+by|broken\s+down\s+by|where\b|[?.!,]|$))"
        m = re.search(pat_dim_val, filter_parse_question, re.I)
        if m:
            val = m.group(1).strip().strip("'\"")
            if val and val.casefold() not in disallowed_filter_values:
                add(col, "=", val)
                continue

    # Common order/subscription lifecycle values often appear without an explicit
    # `status = ...` clause (for example, "completed orders"). Resolve only when
    # there is exactly one status-like field; never bind the word to an arbitrary
    # categorical column.
    status_cols = [str(c) for c in available_cols if any(token in normalize_token(str(c)) for token in ("status", "state"))]
    if re.search(r"\borders?\b", ql):
        order_status_cols = [c for c in status_cols if "order" in normalize_token(c) and "delivery" not in normalize_token(c)]
        if order_status_cols:
            status_cols = order_status_cols
    elif re.search(r"\bsubscriptions?\b", ql):
        subscription_status_cols = [c for c in status_cols if "subscription" in normalize_token(c)]
        if subscription_status_cols:
            status_cols = subscription_status_cols
    if len(status_cols) == 1:
        for lifecycle in ("completed", "cancelled", "canceled", "paid", "refunded", "pending", "shipped", "delivered", "active", "inactive"):
            # An exclusion must not also synthesize the opposite inclusion predicate.
            excluded_lifecycle = re.search(rf"\b(?:exclude|excluding|without|ignore)\s+(?:the\s+)?{lifecycle}\b", ql, re.I)
            if not excluded_lifecycle and re.search(rf"\b{lifecycle}\b", ql, re.I) and not any(x[0] == status_cols[0].casefold() for x in used):
                _neg = re.search(rf"\b(?:not|non|never|isn'?t|wasn'?t|weren'?t|aren'?t|didn'?t)\s+(?:been\s+)?{lifecycle}\b|\bnon-{lifecycle}\b", ql, re.I)
                add(status_cols[0], "!=" if _neg else "=", lifecycle)
                break

    # Legacy signup predicate remains supported, but only when the column is unique.
    signup_cols = [str(c) for c in available_cols if "signup_date" in normalize_token(str(c)) or "sign_up_date" in normalize_token(str(c))]
    if any(token in ql.lower() for token in ("signed up", "signup", "has signed up")) and len(signup_cols) == 1:
        add(signup_cols[0], "IS NOT NULL")

    # Exclusions: 'excluding East', 'excluding East region', 'excluding cancelled orders'.
    # Prefer a semantically explicit lifecycle/status field for lifecycle values. Do not bind
    # "excluding cancelled subscriptions" to a preceding grouping field such as region merely
    # because the question also says "by region". If no status-like field exists, leave the
    # lifecycle filter unresolved rather than inventing a region != cancelled predicate.
    exclusion = None if handled_alternative_predicate else re.search(
        r"\b(?:exclude|excluding|without|ignore)\s+(?:the\s+)?([a-z0-9_ -]+?)(?:\s+(?:subscriptions?|orders?|records?|rows?))?(?:[?.!,]|$)",
        ql,
        re.I,
    )
    if exclusion and not any(c[1] in {"!=", "NOT IN"} for c in used):
        raw_val = exclusion.group(1).strip()
        val = raw_val
        excl_col = None
        explicit_dimension = None
        for dim in dimension_names:
            if raw_val.lower().endswith(f" {dim}"):
                val = raw_val[:-len(dim)-1].strip()
                explicit_dimension = dim
                excl_col = next((c for c in available_cols if dim in str(c).casefold()), None)
                break

        lifecycle_values = {
            "completed", "cancelled", "canceled", "paid", "refunded", "pending",
            "active", "inactive", "returned", "delivered", "shipped", "failed",
        }
        looks_like_lifecycle = val.casefold().strip(" '\"") in lifecycle_values
        status_cols = [
            str(c) for c in available_cols
            if any(token in normalize_token(str(c)) for token in ("status", "state"))
        ]

        # Explicitly named dimensions ("excluding East region") take precedence. Otherwise,
        # a lifecycle status field takes precedence over a preceding grouping dimension.
        if not excl_col and looks_like_lifecycle:
            if len(status_cols) == 1:
                excl_col = status_cols[0]
            # If there is no unique status field, do not fall through and bind the lifecycle
            # value to a previous GROUP BY dimension. The contract remains unresolved below.
        elif not excl_col:
            by_m = re.search(
                r"\b(?:by|across|per)\s+([a-zA-Z0-9_]+)"
                r"(?:\s+and\s+[a-zA-Z0-9_]+)?\s*,?\s+excluding\b",
                ql,
                re.I,
            )
            if by_m:
                excl_col = next((c for c in available_cols if by_m.group(1).casefold() in str(c).casefold()), None)

        if not excl_col and not looks_like_lifecycle:
            # For unqualified lifecycle terms, use status only when it is uniquely identified.
            if len(status_cols) == 1 and any(lc in val.casefold() for lc in lifecycle_values):
                excl_col = status_cols[0]

        if excl_col and val and val.casefold() not in disallowed_filter_values:
            add(excl_col, "!=", val)

    return tuple(conditions)


def extract_aggregation(question: str, target: Optional[str] = None) -> Optional[str]:
    """Determine the canonical requested aggregation from the question text."""
    ql = question.lower()
    # Missingness percentages are a distinct descriptive operation, not a generic RATE.
    if re.search(r"\b(?:percentage|percent|proportion|share)\b.*\b(?:missing|null|blank)\b|\b(?:missing|null|blank)\b.*\b(?:percentage|percent|proportion|share)\b", ql):
        return "MISSING_PERCENT"
    if re.search(r"\b(survival\s+rate|churn\s+rate|rate|rates|proportion|percentage|fraction|share)\b", ql):
        return "RATE"
    if re.search(r"\b(?:variance|var)\b", ql):
        return "VARIANCE"
    if re.search(r"\b(?:mode|most common|most frequent|commonest)\b", ql):
        return "MODE"
    if re.search(r"\binterquartile range|\biqr\b", ql):
        return "IQR"
    if re.search(r"\b(?:first quartile|lower quartile|25th percentile|25th quantile|q1)\b", ql):
        return "Q1"
    if re.search(r"\b(?:third quartile|upper quartile|75th percentile|75th quantile|q3)\b", ql):
        return "Q3"
    if re.search(r"\b(?:missing values?|null values?|blank values?)\b", ql):
        return "MISSING_COUNT"
    if re.search(r"\b(?:range|spread between minimum and maximum)\b", ql):
        return "RANGE"
    if re.search(r"\b(?:standard\s+deviation|stddev|stdev)\b", ql):
        return "STDDEV"
    if re.search(r"\b(average|avg|mean)\b", ql):
        return "MEAN"
    if re.search(r"\bmedian\b", ql):
        return "MEDIAN"
    if re.search(r"\b(total|sum|in\s+total|overall|combined)\b", ql):
        return "SUM"
    # Extensive counts (e.g. "number of passengers") aggregate as SUM on the metric
    num_of_match = re.search(r"\bnumber\s+of\s+([a-zA-Z0-9_\s]+)", ql)
    if num_of_match and target:
        t_low = target.lower()
        if any(k in t_low for k in INTENSIVE_METRIC_KEYWORDS):
            return "MEAN"
        return "SUM"
    if re.search(r"\b(how\s+many|count|discovered\s+the\s+most)\b", ql):
        if re.search(r"\b(users?|customers?|accounts?|people|clients?|orders?|transactions?|purchases?)\b", ql):
            return "COUNT_DISTINCT"
        return "COUNT"
    if re.search(r"\b(grown|grown\s+over\s+time|increased\s+over\s+time)\b", ql):
        if target:
            t_low = target.lower()
            if any(k in t_low for k in INTENSIVE_METRIC_KEYWORDS):
                return "MEAN"
        return "SUM"
    # A ranking request needs a concrete aggregation. In the bounded ranking grammar,
    # ranking a numeric measure defaults to SUM; the chosen default is included in the
    # canonical contract instead of being left only in display/query text.
    if target and re.search(r"^\s*(?:rank|order|sort)\b", ql):
        return "SUM"
    if target:
        t_low = target.lower()
        if any(k in t_low for k in INTENSIVE_METRIC_KEYWORDS):
            return "MEAN"
    return None


_COUNT_ATTEMPT_RE = re.compile(r"\b(how\s+many|count|number\s+of)\b", re.I)
_AGG_ATTEMPT_RE = re.compile(r"\b(total|sum|average|avg|mean|median|variance|var|stddev|stdev)\b|standard\s+deviation", re.I)
# Analytical-intent verbs/constructions only (not business vocabulary such as "churned" or "at risk",
# and not "than"/"different", which can occur in plain counts). When one is present and the cascade
# did not resolve a contract, the question stays NOT_ENFORCED (exploratory).
_NON_ENFORCED_SIGNAL_RE = re.compile(
    r"\b(why|affect\w*|influenc\w*|impact\w*|driv\w*|caus\w*|correlat\w*|relat\w*|associat\w*|"
    r"trend|over\s+time|forecast|predict\w*|compar\w*|versus|vs|difference|"
    # Selection / ranking questions ("which segment generates the highest total ...") are not plain
    # aggregates; they stay on their existing (non-enforced) path, which can plan across tables.
    r"which|highest|lowest|most|least|top|bottom|best|worst|largest|smallest|biggest|greatest|rank\w*)\b",
    re.I,
)


# Group-comparison wording ("does average X differ by region?") is a comparison question, not a plain
# aggregate. Kept out of _NON_ENFORCED_SIGNAL_RE because "different"/"than" can occur in plain counts.
_AGG_COMPARISON_RE = re.compile(r"\b(differ\w*|than|across|between|among|vary|varies|varying)\b", re.I)


def _enforced_family(task_family: str, ql: str) -> str:
    """Return the enforced family a question belongs to (COUNT | SIMPLE_AGGREGATE) or NONE.

    Interim rule (before the completeness ledger): a resolved COUNT/DESCRIPTIVE contract is in its
    family; an unresolved (GENERAL_EXPLORATION) contract is attempted-but-unresolved only when the
    question asks for a plain count/aggregate and carries no non-enforced signal.
    """
    if task_family == "COUNT":
        return "COUNT"
    if task_family in ("DESCRIPTIVE", "RANKING"):
        return "SIMPLE_AGGREGATE"
    if task_family == "GENERAL_EXPLORATION":
        if _NON_ENFORCED_SIGNAL_RE.search(ql):
            return "NONE"
        if _COUNT_ATTEMPT_RE.search(ql):
            return "COUNT"
        if _AGG_ATTEMPT_RE.search(ql) and not _AGG_COMPARISON_RE.search(ql):
            return "SIMPLE_AGGREGATE"
    return "NONE"


def has_multiple_ranking_directives(question: str) -> bool:
    """Return True for genuinely competing ranking directives, not a sort-direction phrase.

    Expressions such as “ranked from highest to lowest” name one descending order.
    Counting the words `highest` and `lowest` independently incorrectly treated that
    standard phrase as two different rankings and prevented schema resolution entirely.
    """
    ql = " ".join(str(question or "").casefold().split())
    # Remove paired endpoints that jointly describe one order. Include common variants
    # where a verb or "from" appears before the two endpoints.
    direction_pairs = (
        r"\b(?:(?:ranked|sorted|ordered)\s+)?(?:from\s+)?highest\s+to\s+lowest\b",
        r"\b(?:(?:ranked|sorted|ordered)\s+)?(?:from\s+)?lowest\s+to\s+highest\b",
        r"\btop\s+to\s+bottom\b",
        r"\bbottom\s+to\s+top\b",
        r"\bhighest\s+first\b",
        r"\blowest\s+first\b",
    )
    for pattern in direction_pairs:
        ql = re.sub(pattern, " ", ql)
    markers = re.findall(r"\b(?:top|bottom|highest|lowest|best|worst)\b", ql)
    limits = re.findall(r"\b(?:top|bottom)\s+(\d+)\b", ql)
    return len(markers) > 1 or len(limits) > 1


def compile_canonical_question_contract(
    question: str, df: pd.DataFrame, semantic: Any = None
) -> CanonicalQuestionContract:
    """Compile a natural-language question into an immutable CanonicalQuestionContract."""
    q_clean = " ".join((question or "").strip().split())
    ql = q_clean.lower()
    cols = [str(c) for c in df.columns]

    target: Optional[str] = None
    explanatory: List[str] = []
    grouping: List[str] = []
    time_col: Optional[str] = None
    comparison_values: List[str] = []
    task_family = "GENERAL_EXPLORATION"
    claim_type = "OBSERVATION"
    requested_claim = "OBSERVATION"
    supported_claim = "OBSERVATION"
    claim_ceiling = "OBSERVATION"
    estimand = "descriptive"
    ranking_direction: Optional[str] = None
    ranking_limit: Optional[int] = None
    if has_multiple_ranking_directives(ql):
        return CanonicalQuestionContract(
            question_text=q_clean, task_family="NONE", claim_type="OBSERVATION",
            estimand="descriptive", completeness="INCOMPLETE", causal_language=False,
            unresolved_roles=("operation_graph",),
            unresolved_semantics=("compound_ranking_not_supported",),
            resolution_evidence={"blocked_compound_ranking": True},
        )
    _limit_match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", ql)
    if _limit_match:
        _parsed_limit = int(_limit_match.group(1))
        if 1 <= _parsed_limit <= 100:
            ranking_limit = _parsed_limit
        else:
            return CanonicalQuestionContract(
                question_text=q_clean, task_family="NONE", claim_type="OBSERVATION",
                estimand="descriptive", completeness="INCOMPLETE", causal_language=False,
                unresolved_roles=("ranking_limit",),
                unresolved_semantics=("ranking_limit_out_of_bounds",),
                resolution_evidence={"ranking_limit_out_of_bounds": _parsed_limit},
            )
    causal_language = False
    evidence: Dict[str, Any] = {}

    # Whole-row duplicate checks do not bind to a single target column. Define duplicates
    # as repeated complete rows beyond the first occurrence; partial/key-based duplicates need
    # an explicit key and are not inferred by this form.
    if re.search(r"\b(?:how many|count|find|identify)\b.*\b(?:duplicate|duplicated|repeated)\s+(?:rows?|records?|entries)\b|\b(?:number of)\s+(?:duplicate|duplicated|repeated)\s+(?:rows?|records?|entries)\b", ql, re.I):
        return CanonicalQuestionContract(
            question_text=q_clean, task_family="DESCRIPTIVE", claim_type="OBSERVATION",
            estimand="duplicate_row_count", requested_aggregation="DUPLICATE_ROW_COUNT",
            family="SIMPLE_AGGREGATE", completeness="COMPLETE", causal_language=False,
            resolution_confidence=1.0,
            resolution_evidence={"common_descriptive_intent": {
                "operation": "DUPLICATE_ROW_COUNT", "duplicate_definition": "identical_complete_rows_beyond_first",
                "parser": "schema_grounded_pattern_v1"
            }},
        )

    # High-frequency distribution/data-quality questions. These forms are compiled into
    # the same canonical contract used by the approval-gated executor; they are not answered
    # by a separate text-only shortcut.
    common_operation_patterns = (
        ("MISSING_PERCENT", r"what percentage of\s+(.+?)\s+(?:values?|entries|cells?)\s+(?:are|is)\s+(?:missing|null|blank)\s*[?.!]*$"),
        ("MISSING_PERCENT", r"(?:percentage|percent|proportion|share) of\s+(.+?)\s+(?:values?|entries|cells?)\s+(?:are|is)\s+(?:missing|null|blank)\s*[?.!]*$"),
        ("SKEWNESS", r"(?:what is the )?skewness\s+(?:of|for|in)\s+(.+?)\s*[?.!]*$"),
        ("KURTOSIS", r"(?:what is the )?kurtosis\s+(?:of|for|in)\s+(.+?)\s*[?.!]*$"),
        ("MAD", r"(?:median absolute deviation|\bmad)\s+(?:of|for|in)\s+(.+?)\s*[?.!]*$"),
        ("COEFFICIENT_OF_VARIATION", r"(?:coefficient of variation|\bcv)\s+(?:of|for|in)\s+(.+?)\s*[?.!]*$"),
        ("ABS_SUM", r"(?:sum of absolute values|absolute sum)\s+(?:of|for|in)\s+(.+?)\s*[?.!]*$"),
        ("POSITIVE_COUNT", r"(?:count|number of)\s+positive\s+(?:values?\s+)?(?:in|for|of)\s+(.+?)\s*[?.!]*$"),
        ("NEGATIVE_COUNT", r"(?:count|number of)\s+negative\s+(?:values?\s+)?(?:in|for|of)\s+(.+?)\s*[?.!]*$"),
        ("MISSING_PERCENT", r"(?:what is the )?(?:percentage|percent|proportion|share) of (?:missing|null|blank) (?:values?|entries|cells?) (?:in|for|of) (.+?)\s*[?.!]*$"),
        ("MISSING_PERCENT", r"(?:how much|what percentage of) (?:the )?(?:data|values?) (?:is|are) missing in (.+?)\s*[?.!]*$"),
        ("MISSING_COUNT", r"(?:count|how many)\s+(?:of the )?(?:missing|null|blank)\s+(?:values?|entries|cells?)\s+(?:in|for|of)\s+(.+?)\s*[?.!]*$"),
        ("MISSING_COUNT", r"(?:missing|null|blank)\s+(?:values?|entries|cells?)\s+(?:in|for|of)\s+(.+?)\s*[?.!]*$"),
        ("MISSING_COUNT", r"(?:how many)\s+(.+?)\s+(?:values?|entries|cells?)\s+(?:are\s+)?(?:missing|null|blank)\s*[?.!]*$"),
        ("COUNT_DISTINCT", r"(?:count|number of|how many) (?:the )?(?:unique|distinct|different) (.+?)\s*[?.!]*$"),
        ("COUNT_DISTINCT", r"(?:how many) (.+?) (?:unique|distinct|different) (?:values?|items?|entries|customers?|users?|orders?)\s*[?.!]*$"),
        ("MIN", r"(?:minimum|min|smallest|lowest) (?:value )?(?:of|for|in) (.+?)\s*[?.!]*$"),
        ("MAX", r"(?:maximum|max|largest|highest) (?:value )?(?:of|for|in) (.+?)\s*[?.!]*$"),
        ("MIN", r"what is the (?:minimum|min|smallest|lowest) (?:value )?(?:of|for|in) (.+?)\s*[?.!]*$"),
        ("MIN", r"what is the (?:minimum|min|smallest|lowest) (?:value of )?(.+?)\s*[?.!]*$"),
        ("MAX", r"what is the (?:maximum|max|largest|highest) (?:value )?(?:of|for|in) (.+?)\s*[?.!]*$"),
        ("MAX", r"what is the (?:maximum|max|largest|highest) (?:value of )?(.+?)\s*[?.!]*$"),
        ("MODE", r"(?:mode|most common|most frequent|commonest)(?:\s+value)?\s+(?:of|for|in)\s+(.+?)\s*[?.!]*$"),
        ("MODE", r"(?:most common|most frequent|commonest)\s+(.+?)\s*[?.!]*$"),
        ("MODE", r"(?:which|what)\s+(.+?)\s+(?:occurs?\s+)?most\s+(?:often|frequently|commonly)\s*[?.!]*$"),
        ("IQR", r"(?:interquartile range|iqr)\s+(?:of|for|in)\s+(.+?)\s*[?.!]*$"),
        ("Q1", r"(?:first quartile|lower quartile|25th percentile|25th quantile|q1)\s+(?:of|for|in)\s+(.+?)\s*[?.!]*$"),
        ("Q3", r"(?:third quartile|upper quartile|75th percentile|75th quantile|q3)\s+(?:of|for|in)\s+(.+?)\s*[?.!]*$"),
        ("RANGE", r"\b(?:range|spread between minimum and maximum)\s+(?:of|for|in)\s+(.+?)\s*[?.!]*$"),
    )
    for common_operation, pattern in common_operation_patterns:
        match = re.search(pattern, ql, re.I)
        if not match:
            continue
        target_phrase = match.group(1).strip()
        # Permit a single explicit grouping clause while keeping metric binding schema-grounded.
        group_phrase = None
        group_match = re.search(r"\s+(?:by|for each|in each)\s+(.+)$", target_phrase, re.I)
        if group_match:
            target_phrase = target_phrase[:group_match.start()].strip()
            group_phrase = group_match.group(1).strip()
        target_col = resolve_phrase_to_column(target_phrase, cols)
        group_col = resolve_phrase_to_column(group_phrase, cols) if group_phrase else None
        if target_col and (not group_phrase or group_col):
            common_groups = (group_col,) if group_col else ()
            return CanonicalQuestionContract(
                question_text=q_clean, task_family="DESCRIPTIVE", claim_type="OBSERVATION",
                estimand=common_operation.lower(), target_column=target_col,
                grouping_columns=common_groups, requested_aggregation=common_operation,
                family="SIMPLE_AGGREGATE", completeness="COMPLETE", causal_language=False,
                resolution_confidence=1.0,
                resolution_evidence={"common_descriptive_intent": {
                    "operation": common_operation, "target_column": target_col,
                    "grouping_columns": list(common_groups), "parser": "schema_grounded_pattern_v1"
                }},
            )
        # A recognized operation with a missing/ambiguous target is not allowed to fall
        # through and accidentally execute a different metric.
        return CanonicalQuestionContract(
            question_text=q_clean, task_family="GENERAL_EXPLORATION", claim_type="OBSERVATION",
            estimand=common_operation.lower(), target_column=None,
            requested_aggregation=common_operation, family="SIMPLE_AGGREGATE",
            completeness="INCOMPLETE", unresolved_roles=("target_column",),
            causal_language=False, resolution_confidence=0.0,
            resolution_evidence={"common_descriptive_intent": {
                "operation": common_operation, "target_phrase": target_phrase,
                "reason": "target_column_unresolved_or_ambiguous"
            }},
        )

    # Check causal vocabulary
    if re.search(r"\b(affect|affects|affected|impact|impacts|impacted|drive|drives|driven|cause|causes|caused|effect\s+of|why)\b", ql):
        causal_language = True

    # 1. Multi-association: "Is X or Z more strongly related to Y?"
    multi_assoc_match = re.search(
        r"(?:is|are)\s+([a-zA-Z0-9_\s]+?)\s+or\s+([a-zA-Z0-9_\s]+?)\s+(?:more\s+)?(?:strongly\s+)?(?:related|correlated|associated)\s+(?:to|with)\s+([a-zA-Z0-9_\s]+?)(?:\?|$)",
        ql,
    )
    if multi_assoc_match:
        cand1 = resolve_phrase_to_column(multi_assoc_match.group(1).strip(), cols)
        cand2 = resolve_phrase_to_column(multi_assoc_match.group(2).strip(), cols)
        cand_target = resolve_phrase_to_column(multi_assoc_match.group(3).strip(), cols)
        if cand1 and cand2 and cand_target:
            explanatory = [cand1, cand2]
            target = cand_target
            task_family = "MULTI_ASSOCIATION"
            claim_type = "ASSOCIATION"
            estimand = "multi_correlation"
            return CanonicalQuestionContract(
                question_text=q_clean,
                task_family=task_family,
                claim_type=claim_type,
                estimand=estimand,
                target_column=target,
                explanatory_columns=tuple(explanatory),
                requested_aggregation="CORRELATION",
                causal_language=False,
                resolution_confidence=1.0,
                resolution_evidence={"multi_association": {"predictors": explanatory, "target": target}},
            )

    # 2. Moderated association: "Does the effect of class on survival differ between men and women?"
    effect_diff_match = re.search(
        r"does\s+(?:the\s+)?effect\s+of\s+(.+?)\s+on\s+(.+?)\s+differ\s+between\s+(.+?)(?:\?|$)",
        ql,
    )
    if effect_diff_match:
        pred_ph = effect_diff_match.group(1).strip()
        tgt_ph = effect_diff_match.group(2).strip()
        mod_ph = effect_diff_match.group(3).strip()
        pred_col = resolve_phrase_to_column(pred_ph, cols)
        tgt_col = resolve_phrase_to_column(tgt_ph, cols)
        mod_col = resolve_phrase_to_column(mod_ph, cols)
        if not mod_col:
            mod_col, _ = scan_observed_category_values(mod_ph, df)
        if pred_col and tgt_col:
            target = tgt_col
            explanatory = [pred_col]
            if mod_col:
                grouping = [mod_col]
            task_family = "INTERACTION"
            claim_type = "ASSOCIATION"
            estimand = "interaction"
            evidence["moderated_association"] = {"target": tgt_col, "predictor": pred_col, "moderator": mod_col}
            return CanonicalQuestionContract(
                question_text=q_clean,
                task_family=task_family,
                claim_type=claim_type,
                estimand=estimand,
                target_column=target,
                explanatory_columns=tuple(explanatory),
                grouping_columns=tuple(grouping),
                requested_aggregation="RATE" if "surviv" in target.lower() else "MEAN",
                causal_language=False,
                resolution_confidence=1.0,
                resolution_evidence=evidence,
            )

    # 3. Directional relation: "Did age affect survival?" / "Does party size drive the tip amount?" / "Did price cause churn?"
    directional_match = re.search(
        r"(?:does|do|did|can|could|will|would)\s+(.+?)\s+(?:affect\w*|influenc\w*|impact\w*|driv\w*|caus\w*)\s+(.+?)(?:\?|$)",
        ql,
    )
    if directional_match and not target:
        pred_phrase = directional_match.group(1).strip()
        target_phrase = directional_match.group(2).strip()
        pred_cols = resolve_phrase_to_column_list(pred_phrase, cols)
        target_col = resolve_phrase_to_column(target_phrase, cols)
        if not target_col and semantic is not None:
            sem_tgt = getattr(semantic, "target_metric_col", None) or getattr(semantic, "churn_event_col", None)
            if sem_tgt and sem_tgt in cols:
                target_col = sem_tgt
        if pred_cols and target_col:
            target = target_col
            explanatory = pred_cols
            task_family = "CAUSAL_REQUEST"
            causal_language = True
            requested_claim = "CAUSAL"
            supported_claim = "ASSOCIATION"
            claim_ceiling = "ASSOCIATION"
            claim_type = "ASSOCIATION"
            estimand = "correlation" if pd.api.types.is_numeric_dtype(df[target_col]) or df[target_col].nunique() == 2 else "group_difference"
            evidence["relation"] = {"type": "directional", "predictor": pred_cols, "target": target_col}
        elif target_col and not pred_cols:
            target = target_col
            task_family = "CAUSAL_REQUEST"
            causal_language = True
            requested_claim = "CAUSAL"
            supported_claim = "ASSOCIATION"
            claim_ceiling = "ASSOCIATION"
            claim_type = "ASSOCIATION"
            estimand = "correlation" if pd.api.types.is_numeric_dtype(df[target_col]) or df[target_col].nunique() == 2 else "group_difference"
            unresolved = ["explanatory"]

    # "effect of X on Y"
    effect_of_match = re.search(r"(?:effect|impact)\s+of\s+(.+?)\s+on\s+(.+?)(?:\s+(?:differ|across)|\?|$)", ql)
    if effect_of_match and not target:
        pred_phrase = effect_of_match.group(1).strip()
        target_phrase = effect_of_match.group(2).strip()
        pred_cols = resolve_phrase_to_column_list(pred_phrase, cols)
        target_col = resolve_phrase_to_column(target_phrase, cols)
        if pred_cols and target_col:
            target = target_col
            explanatory = pred_cols
            task_family = "CAUSAL_REQUEST"
            causal_language = True
            requested_claim = "CAUSAL"
            supported_claim = "ASSOCIATION"
            claim_ceiling = "ASSOCIATION"
            claim_type = "ASSOCIATION"
            estimand = "correlation"
            evidence["relation"] = {"type": "effect_of", "predictor": pred_cols, "target": target_col}

    # Symmetric: "Is survival correlated with fare?" / "Is tip correlated with total bill?" / "Are price and marketing_spend associated with annual_sales?"
    corr_match = re.search(
        r"(?:is|are|was|were)\s+(.+?)\s+(correlated|associated|related)\s+with\s+(.+?)(?:\?|$)",
        ql,
    )
    if corr_match and not target:
        var1_phrase = corr_match.group(1).strip()
        verb = corr_match.group(2).strip().lower()
        var2_phrase = corr_match.group(3).strip()
        var1_cols = resolve_phrase_to_column_list(var1_phrase, cols)
        var2_cols = resolve_phrase_to_column_list(var2_phrase, cols)
        if var1_cols and var2_cols:
            if len(var1_cols) > 1 and len(var2_cols) == 1:
                target = var2_cols[0]
                explanatory = var1_cols
            elif len(var2_cols) > 1 and len(var1_cols) == 1:
                target = var1_cols[0]
                explanatory = var2_cols
            else:
                c1 = var1_cols[0]
                c2 = var2_cols[0]
                sem_tgt = None
                if semantic is not None:
                    sem_tgt = getattr(semantic, "target_metric_col", None) or getattr(semantic, "churn_event_col", None)
                    # A symmetric "X correlated with Y" between two numeric/boolean columns has no
                    # inherent target: the semantic engine's generic metric guess (often just the
                    # second-named column) must not flip the user's stated variable order.
                    if (
                        sem_tgt
                        and sem_tgt != getattr(semantic, "churn_event_col", None)
                        and df is not None
                        and c1 in df.columns and c2 in df.columns
                        and all(pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]) for c in (c1, c2))
                    ):
                        sem_tgt = None
                if sem_tgt and sem_tgt in (c1, c2):
                    if sem_tgt == c1:
                        target = c1
                        explanatory = [c2]
                    else:
                        target = c2
                        explanatory = [c1]
                elif df is not None and c2 in df.columns and (
                    pd.api.types.is_object_dtype(df[c2])
                    or pd.api.types.is_string_dtype(df[c2])
                    or isinstance(df[c2].dtype, pd.CategoricalDtype)
                ) and c1 in df.columns and (
                    pd.api.types.is_numeric_dtype(df[c1]) or pd.api.types.is_bool_dtype(df[c1])
                ):
                    target = c1
                    explanatory = [c2]
                elif df is not None and c1 in df.columns and (
                    pd.api.types.is_object_dtype(df[c1])
                    or pd.api.types.is_string_dtype(df[c1])
                    or isinstance(df[c1].dtype, pd.CategoricalDtype)
                ) and c2 in df.columns and (
                    pd.api.types.is_numeric_dtype(df[c2]) or pd.api.types.is_bool_dtype(df[c2])
                ):
                    target = c2
                    explanatory = [c1]
                elif verb in ("associated", "related"):
                    target = c2
                    explanatory = [c1]
                else:
                    target = c1
                    explanatory = [c2]

            task_family = "MULTI_ASSOCIATION" if len(explanatory) > 1 else "ASSOCIATION"
            requested_claim = "ASSOCIATION"
            supported_claim = "ASSOCIATION"
            claim_ceiling = "ASSOCIATION"
            claim_type = "ASSOCIATION"
            estimand = "correlation"
            evidence["relation"] = {"type": "symmetric_correlation", "var1": target, "var2": explanatory[0]}

    # 4. Trend: "Has X improved/grown/declined over time?" / "Has fuel efficiency improved over model years?" / "Has the number of passengers grown over time?"
    trend_match = re.search(
        r"(?:has|have|is|are)\s+(?:the\s+)?(.+?)\s+(?:improved|grown|declined|increased|decreased|changed)\s+(?:over\s+(?:the\s+)?(.+?)|over\s+time)(?:\?|$)",
        ql,
    )
    if trend_match and not target:
        metric_phrase = trend_match.group(1).strip()
        time_phrase = (trend_match.group(2) or "").strip()
        target_col = resolve_phrase_to_column(metric_phrase, cols)
        time_candidate = resolve_phrase_to_column(time_phrase, cols) if time_phrase else None
        if not time_candidate:
            for tc in ["model_year", "year", "date", "timestamp", "time", "month"]:
                if tc in cols:
                    time_candidate = tc
                    break
        if target_col:
            target = target_col
            time_col = time_candidate
            task_family = "TREND"
            claim_type = "OBSERVATION"
            requested_claim = "OBSERVATION"
            supported_claim = "OBSERVATION"
            claim_ceiling = "OBSERVATION"
            estimand = "trend"
            evidence["trend"] = {"target": target_col, "time_col": time_candidate}

    # 5. Count questions: "How many users have signed up for product_id P0066?"
    # A row count deliberately has no target column.  Treating the grouping
    # column as a numeric target is the historic source of ``SUM(plan_tier)``.
    # ``ROW_COUNT`` is an explicit operation and always compiles to COUNT(*).
    row_count_match = re.search(
        r"\b(?:count|how\s+many|number\s+of)\s+(?:the\s+)?(?:rows?|records?|observations?|subscriptions?)\b",
        ql,
    )
    if row_count_match and not target and not re.search(r"\bwhich\b", ql):
        group_match = re.search(r"\b(?:by|per|across|within|for\s+each|in\s+each)\s+([a-zA-Z0-9_,\s]+?)(?:[?.!]|$)", ql)
        g_phrase = re.split(r"\s*,\s*|\s+(?:including|with|where)\s+", group_match.group(1).strip(), maxsplit=1, flags=re.I)[0] if group_match else ""
        g_col = resolve_phrase_to_column(g_phrase, cols) if group_match else None
        task_family = "COUNT"
        claim_type = requested_claim = supported_claim = claim_ceiling = "OBSERVATION"
        estimand = "count"
        requested_agg = "ROW_COUNT"
        grouping = [g_col] if g_col else []
        evidence["count"] = {
            "entity_phrase": row_count_match.group(0),
            "row_count": True,
            "group": g_col,
        }
        if group_match and not g_col:
            unresolved = ["grouping"]
    count_match = re.search(
        r"\b(?:how\s+many|count|number\s+of)\s+(?:unique\s+)?([a-zA-Z0-9_\s]+?)(?:\s+(?:have|has|had|were|are|is|was|for|with|in|on|by|of))",
        ql,
    )
    if count_match and not target and not re.search(r"\bwhich\b", ql):
        entity_phrase = count_match.group(1).strip()
        entity_col = resolve_entity_count_column(entity_phrase, cols)
        if entity_col:
            target = entity_col
            task_family = "COUNT"
            claim_type = "OBSERVATION"
            requested_claim = "OBSERVATION"
            supported_claim = "OBSERVATION"
            claim_ceiling = "OBSERVATION"
            estimand = "count"
            requested_agg = "COUNT_DISTINCT" if re.search(r"\b(users?|customers?|accounts?|people|clients?|orders?|transactions?|purchases?)\b", ql) else "COUNT"
            evidence["count"] = {
                "entity": entity_col,
                "entity_phrase": entity_phrase,
                "filters": extract_filter_conditions(q_clean, cols),
            }
            # Grouping in count queries: "How many customers are in each region?" / "Count orders by status"
            group_match = re.search(r"\b(?:by|per|across|within|for each|in each)\s+([a-zA-Z0-9_\s]+?)(?:[?.!]|$)", ql)
            if group_match:
                g_phrase = group_match.group(1).strip()
                g_col = resolve_phrase_to_column(g_phrase, cols)
                if g_col and g_col != target:
                    grouping = [g_col]
                    evidence["count"]["group"] = g_col
            filter_conditions = extract_filter_conditions(q_clean, cols)
            if filter_conditions:
                evidence["count"]["filters"] = list(filter_conditions)

    # 5a. Count ranking: "Which region has the most orders?" / "Which channel had the fewest returns?".
    # The entity is something to COUNT per group, not a measure to sum. Resolve both the group and the
    # entity against the schema; if either is unknown fall through (never substitute another measure).
    which_count = re.search(
        r"\b(?:which|what)\s+(?P<grp>[a-z0-9_ ]+?)\s+(?:has|had|have|recorded|generated|placed|made|received)\s+"
        r"(?:the\s+)?(?P<dir>most|fewest|least|(?:highest|largest|greatest|lowest|smallest)\s+(?:number|count)\s+of)\s+"
        r"(?P<ent>[a-z0-9_ ]+?)(?=\s+(?:in|on|for|among|from|during|within|where|that|who)\b|[?.!]|$)",
        ql,
    )
    if which_count and not target:
        count_group = resolve_phrase_to_column(which_count.group("grp").strip(), cols)
        count_entity = resolve_entity_count_column(which_count.group("ent").strip(), cols)
        if count_group and count_entity and count_group != count_entity:
            target = count_entity
            grouping = [count_group]
            task_family = "RANKING"
            claim_type = requested_claim = supported_claim = claim_ceiling = "OBSERVATION"
            estimand = "count"
            requested_agg = "COUNT_DISTINCT" if re.search(r"\b(users?|customers?|accounts?|people|clients?|orders?|transactions?|purchases?)\b", which_count.group("ent")) else "COUNT"
            ranking_direction = "ASC" if re.match(r"fewest|least|lowest|smallest", which_count.group("dir")) else "DESC"
            evidence["count"] = {"entity": count_entity, "entity_phrase": which_count.group("ent").strip(), "group": count_group}
            evidence["ranking"] = {"group": count_group, "target": target, "direction": ranking_direction}

    # 5b. Ranking: "Which X has/had the highest/lowest Y?", "Rank/Order X by Y", or "Top/Bottom [N] X by Y"
    top_rank_match = re.match(
        r"^(?:show\s+|find\s+|identify\s+)?(?P<direction>top|bottom|highest|lowest|best|worst)\s+(?:(?P<limit>\d{1,3})\s+)?(?P<first>.+?)\s+by\s+(?P<second>[A-Za-z_][A-Za-z0-9_ ]*?)(?:\s+(?:for|within|during|over)\s+.+)?[?.!]*$",
        ql,
    )
    if top_rank_match and not target and len(re.findall(r"\b(?:top|bottom|highest|lowest|best|worst)\b", ql)) <= 1:
        _lim_str = top_rank_match.group("limit")
        _lim_int = int(_lim_str) if _lim_str else ranking_limit
        if _lim_int is None or 1 <= _lim_int <= 100:
            _first_p = re.sub(r"^\d{1,3}\s+", "", top_rank_match.group("first").strip())
            _second_p = top_rank_match.group("second").strip()
            _first_c = resolve_phrase_to_column(_first_p, cols)
            _second_c = resolve_phrase_to_column(_second_p, cols)
            if _first_c and _second_c and _first_c != _second_c:
                _first_num = bool(_first_c in df.columns and pd.api.types.is_numeric_dtype(df[_first_c]))
                _second_num = bool(_second_c in df.columns and pd.api.types.is_numeric_dtype(df[_second_c]))
                if not _first_num and _second_num:
                    metric_col, dim_col = _second_c, _first_c
                elif _first_num and not _second_num:
                    metric_col, dim_col = _first_c, _second_c
                else:
                    metric_col, dim_col = _second_c, _first_c
                target = metric_col
                grouping = [dim_col]
                task_family = "RANKING"
                claim_type = requested_claim = supported_claim = claim_ceiling = "OBSERVATION"
                estimand = "ranking"
                ranking_direction = "ASC" if top_rank_match.group("direction").casefold() in {"bottom", "lowest", "worst"} else "DESC"
                ranking_limit = _lim_int
                req_agg = extract_aggregation(q_clean, metric_col) or "SUM"
                evidence["ranking"] = {"group": dim_col, "target": target, "direction": ranking_direction}
                evidence["explicit_rank_roles"] = {"group": dim_col, "measure": target}

    # Permit temporal and population qualifiers after the grouping phrase. The earlier
    # end-anchored pattern failed on otherwise clear requests like "Rank revenue by region
    # for last quarter", allowing the generic descriptive parser's earlier target binding to
    # prevent the ranking branch from ever running.
    rank_imperative_match = re.search(
        r"^\s*(?:rank|order|sort)\s+(.+?)\s+by\s+(.+?)\s*[?.!]*\s*$",
        ql,
    )
    rank_match = re.search(
        r"\bwhich\s+([a-zA-Z0-9_\s]+?)\s+(?:has|had|have|shows?|with|discovered|generated|produced|earned|made|drove|delivered|achieved|recorded|brought\s+in)\s+(?:the\s+)?(highest|lowest|most|least|best|worst|largest|smallest)\s+(.+?)(?:\?|$)",
        ql,
    )
    # The broader verbs ("generated", "produced", ...) can still be parsed when a
    # recognized population/time clause follows the metric, e.g. "Which region generated
    # the most revenue after excluding cancelled orders?" The old guard rejected every
    # such question merely because `excluding` appeared in the metric capture, even though
    # filter extraction below could preserve the clause. Keep the ranking branch only when
    # the text before the trailing scope cue resolves to a real schema column; unresolved
    # scope is still caught by the material-intent ledger and will not be silently dropped.
    if rank_match and re.search(
        r"\b(?:generated|produced|earned|made|drove|delivered|achieved|recorded|brought\s+in)\b", rank_match.group(0)
    ) and re.search(r"\b(?:from|in|for|of|where|among|during|within|across|by|per|excluding|exclude|without|except|after|before|since)\b", rank_match.group(3)):
        _rank_scope_head = re.match(
            r"^(.+?)\s+(?:from|in|for|of|where|among|during|within|across|by|per|excluding|exclude|without|except|after|before|since)\s+.+$",
            rank_match.group(3).strip(), re.I,
        )
        _rank_scope_metric = _rank_scope_head.group(1).strip() if _rank_scope_head else ""
        _rank_scope_resolved = bool(_rank_scope_metric and resolve_phrase_to_column(_rank_scope_metric, cols))
        _rank_scope_ambiguous = bool(_rank_scope_metric and ambiguous_semantic_candidates(_rank_scope_metric, cols))
        if not _rank_scope_resolved and not _rank_scope_ambiguous:
            rank_match = None
    if rank_imperative_match:
        metric_phrase = rank_imperative_match.group(1).strip()
        dim_phrase = rank_imperative_match.group(2).strip()
        direction_match = re.search(r"\s+(asc|desc|ascending|descending)\s*$", dim_phrase, re.I)
        dir_phrase = direction_match.group(1).strip().lower() if direction_match else ""
        if direction_match:
            dim_phrase = dim_phrase[:direction_match.start()].strip()
        # Remove trailing temporal/population clauses from the group phrase, but preserve
        # dimension names such as "region in West" where `in` denotes a category value.
        dim_phrase = re.split(
            r"\s+(?:for|during|within|over)\s+(?=(?:last|past|previous|prior|current|this)\b)|\s+(?:where|excluding|except|without)\b",
            dim_phrase,
            maxsplit=1,
        )[0].strip()

        first_col = resolve_phrase_to_column(metric_phrase, cols) or target
        second_col = resolve_phrase_to_column(dim_phrase, cols)
        # Support both common human phrasings:
        #   "rank revenue by region" (measure first)
        #   "rank regions by total revenue" (entity/dimension first).
        # Physical dtype plus explicit phrase resolution disambiguates them. If both sides
        # have the same broad type, do not silently reverse the user's intended roles.
        first_numeric = bool(first_col and first_col in df.columns and pd.api.types.is_numeric_dtype(df[first_col]))
        second_numeric = bool(second_col and second_col in df.columns and pd.api.types.is_numeric_dtype(df[second_col]))
        if first_col and second_col and not first_numeric and second_numeric:
            metric_col, dim_col = second_col, first_col
        elif first_col and second_col and first_numeric and not second_numeric:
            metric_col, dim_col = first_col, second_col
        else:
            metric_col, dim_col = first_col, second_col
        ranking_direction = "ASC" if dir_phrase in ("asc", "ascending", "lowest", "least") else "DESC"
        req_agg = extract_aggregation(q_clean, metric_col) or "SUM"

        if metric_col and dim_col and metric_col != dim_col:
            target = metric_col
            grouping = [dim_col]
            task_family = "RANKING"
            claim_type = "OBSERVATION"
            requested_claim = "OBSERVATION"
            supported_claim = "OBSERVATION"
            claim_ceiling = "OBSERVATION"
            estimand = "ranking"
            evidence["ranking"] = {"group": dim_col, "target": target, "direction": ranking_direction}
    elif rank_match and not target:
        dim_phrase = rank_match.group(1).strip()
        direction_word = rank_match.group(2).strip().lower()
        metric_phrase = rank_match.group(3).strip()
        # "lowest average discount among non-returned orders": the tail is a population scope,
        # not part of the measure name (it must never pull in a column such as `returned`).
        _tail = re.match(
            r"^(.+?)\s+(?:among|for|within|in|of|where|when|that|which|from|during|excluding|exclude|without|except|after|before|since)\s+(.+)$",
            metric_phrase,
        )
        if _tail and not re.search(r"\b(per|and|or|than)\b", _tail.group(1)):
            _measure_head = _tail.group(1).strip()
            _hc = resolve_phrase_to_column(_measure_head, cols)
            _ambiguous_candidates = ambiguous_semantic_candidates(_measure_head, cols)
            if _hc or _ambiguous_candidates:
                metric_phrase = _measure_head
                evidence["measure_scope_tail"] = _tail.group(2).strip()
                if _ambiguous_candidates:
                    # Keep the operation and all other resolved roles. A metric choice is
                    # required, but the question should not fall back to GENERAL_EXPLORATION.
                    evidence["ambiguous_target_candidates"] = list(_ambiguous_candidates)

        dim_col = resolve_phrase_to_column(dim_phrase, cols)
        metric_col = resolve_phrase_to_column(metric_phrase, cols)
        ranking_direction = "ASC" if direction_word in ("lowest", "least", "worst", "smallest") else "DESC"

        # Special case: "Which method discovered the most planets?"
        if not metric_col and "planet" in metric_phrase:
            metric_col = dim_col
            req_agg = "COUNT"
        else:
            req_agg = extract_aggregation(q_clean, metric_col)

        if not metric_col and semantic is not None:
            sem_tgt = getattr(semantic, "target_metric_col", None)
            if sem_tgt and sem_tgt in cols and sem_tgt != dim_col:
                metric_col = sem_tgt

        if dim_col:
            grouping = [dim_col]
            target = metric_col
            task_family = "RANKING"
            claim_type = "OBSERVATION"
            requested_claim = "OBSERVATION"
            supported_claim = "OBSERVATION"
            claim_ceiling = "OBSERVATION"
            estimand = "ranking"
            if not metric_col:
                ambiguous_candidates = ambiguous_semantic_candidates(metric_phrase, cols)
                if ambiguous_candidates:
                    evidence["ambiguous_target_candidates"] = list(ambiguous_candidates)
                    # A ranking over numeric measures defaults to SUM when no aggregate word
                    # is specified; this default must survive the metric clarification.
                    req_agg = extract_aggregation(q_clean, None) or "SUM"
                    evidence["explicit_rank_roles"] = {
                        "group": dim_col, "measure_phrase": metric_phrase,
                        "candidate_measures": list(ambiguous_candidates),
                    }
            evidence["ranking"] = {"group": dim_col, "target": target, "direction": ranking_direction}

    # Plain-language grouped comparison: "How does revenue compare by region?" or
    # "How does average sales compare across channels?" This is a descriptive aggregate
    # across a schema-grounded dimension, not an unspecified multi-step/statistical test.
    # Keep it on the same canonical contract/execution path as "Compare total revenue by
    # region". A generic measure such as `revenue` must still resolve uniquely; ambiguous
    # metric schemas stay open for clarification rather than selecting a random column.
    compare_by_match = re.fullmatch(
        r"\s*(?:please\s+)?(?:how\s+(?:does|do|did)\s+)?"
        r"(?P<measure>.+?)\s+(?:compare|compares)\s+"
        r"(?:by|across|among)\s+(?P<dimension>.+?)\s*[?.!]*\s*",
        q_clean,
        re.I,
    )
    if compare_by_match and not target:
        measure_phrase = re.sub(r"^(?:the|overall)\s+", "", compare_by_match.group("measure").strip(), flags=re.I)
        dimension_phrase = compare_by_match.group("dimension").strip()
        measure_col = resolve_phrase_to_column(measure_phrase, cols)
        dimension_col = resolve_phrase_to_column(dimension_phrase, cols)
        if (
            measure_col and dimension_col and measure_col != dimension_col
            and measure_col in df.columns and dimension_col in df.columns
            and pd.api.types.is_numeric_dtype(df[measure_col])
            and not _is_identifier_column(measure_col)
            and not _is_identifier_column(dimension_col)
            and (
                pd.api.types.is_object_dtype(df[dimension_col])
                or pd.api.types.is_string_dtype(df[dimension_col])
                or isinstance(df[dimension_col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(df[dimension_col])
                or (pd.api.types.is_integer_dtype(df[dimension_col]) and df[dimension_col].nunique(dropna=True) <= 30)
            )
        ):
            target = measure_col
            grouping = [dimension_col]
            task_family = "DESCRIPTIVE"
            claim_type = requested_claim = supported_claim = claim_ceiling = "OBSERVATION"
            estimand = "mean" if extract_aggregation(q_clean, measure_col) == "MEAN" else "sum"
            evidence["grouped_compare_intent"] = {
                "target": measure_col,
                "group": dimension_col,
                "source_phrase": q_clean,
                "method": "schema_grounded_compare_by_dimension",
            }

    # Compound / Multi-clause Analyst Questions with Ranking or Grouped Aggregation
    # e.g.: "Among orders where order_status is Delivered, calculate the total realized_revenue for each region. Return every region, ranked from highest to lowest total."
    # e.g.: "Calculate total realized_revenue for each region, ranked from highest to lowest"
    # e.g.: "What is the average revenue for each customer segment, ranked highest first?"
    # Causal/diagnostic wording ("why did X surge across region?") is a root-cause request, not a plain
    # grouped aggregate: answering it with a SUM table would silently drop the "why".
    if not target and not re.search(r"\b(why|caus\w*|driv(?:e|es|en|er|ers|ing)|affect\w*|influenc\w*|impact\w*|explain\w*)\b", ql):
        is_ranking_request = bool(re.search(
            r"\b(rank\w*|highest\s+to\s+lowest|lowest\s+to\s+highest|highest\s+first|lowest\s+first|top\s+to\s+bottom|descending|ascending)\b",
            ql,
        ))
        group_m = re.search(
            r"\b(?:for\s+each|in\s+each|for\s+every|in\s+every|by|per|across|grouped\s+by|split\s+by|broken\s+down\s+by)\s+([a-zA-Z0-9_]+)",
            ql,
        )
        if group_m:
            cand_group = resolve_phrase_to_column(group_m.group(1), cols)
            # Grouping by a column that is unique on every row (an identifier such as order_id) yields
            # one-row groups: "average revenue per order" would return the raw column, not an average.
            # Do not bind it; the question stays unresolved and is refused rather than answered wrongly.
            row_grain_entity = None
            if cand_group and df is not None and cand_group in df.columns and len(df) > 1 and df[cand_group].nunique(dropna=False) == len(df):
                # A row-unique identifier is not a real grouping. "average X per order" on an
                # order-level table means the mean of X over rows (the grain is the order), which
                # is unambiguous and recorded as an assumption. Any other use is not bound.
                per_entity = re.search(rf"\bper\s+{re.escape(group_m.group(1))}\b", ql)
                if per_entity and re.search(r"\b(average|avg|mean|median)\b", ql) and not is_ranking_request:
                    row_grain_entity = cand_group
                else:
                    cand_group = None
            if cand_group:
                filter_conds = extract_filter_conditions(q_clean, cols)
                filter_cols = {f["column"] for f in filter_conds}
                cand_metric = None
                if re.search(r"\b(?:average order value|aov|order value)\b", ql):
                    for aov_alias in ("aov", "order_value", "total_amount", "order_total", "order_amount", "revenue", "sales_amount", "gross_sales", "sales", "amount"):
                        cand_metric = next((c for c in cols if re.sub(r"[^a-z0-9]+", "_", c.casefold()).strip("_") == aov_alias and pd.api.types.is_numeric_dtype(df[c])), None)
                        if cand_metric:
                            break
                if not cand_metric:
                    agg_metric_m = re.search(
                        r"\b(total|sum|average|avg|mean|median|minimum|min|maximum|max|count|rate|stddev|variance)\s+(?:of\s+)?(?:the\s+)?([a-zA-Z0-9_]+)",
                        ql,
                    )
                    if agg_metric_m:
                        found_col = resolve_phrase_to_column(agg_metric_m.group(2), cols)
                        if found_col and not _is_identifier_column(found_col) and found_col != cand_group and found_col not in filter_cols:
                            cand_metric = found_col

                if not cand_metric:
                    for c in cols:
                        if c in filter_cols or c == cand_group:
                            continue
                        if df is not None and c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                            if re.search(rf"\b{re.escape(c.lower())}\b|\b{re.escape(c.replace('_', ' ').lower())}\b", ql):
                                cand_metric = c
                                break

                if cand_metric and cand_metric != cand_group:
                    target = cand_metric
                    grouping = [] if row_grain_entity else [cand_group]
                    if row_grain_entity:
                        evidence["row_grain"] = {"entity": group_m.group(1), "identifier_column": row_grain_entity,
                                                 "meaning": "each row is one " + group_m.group(1) + "; per-" + group_m.group(1) + " = per-row"}
                    if is_ranking_request:
                        task_family = "RANKING"
                        claim_type = "OBSERVATION"
                        requested_claim = "OBSERVATION"
                        supported_claim = "OBSERVATION"
                        claim_ceiling = "OBSERVATION"
                        estimand = "ranking"
                        ranking_direction = "ASC" if re.search(r"\b(lowest\s+to\s+highest|lowest\s+first|ascending|\basc\b)\b", ql) else "DESC"
                        evidence["ranking"] = {"group": cand_group, "target": target, "direction": ranking_direction}
                    else:
                        task_family = "DESCRIPTIVE"
                        claim_type = "OBSERVATION"
                        requested_claim = "OBSERVATION"
                        supported_claim = "OBSERVATION"
                        claim_ceiling = "OBSERVATION"
                        estimand = extract_aggregation(q_clean, target) or "sum"

    # Entity-list comparison: "Compare regions A, B and C by total revenue in 2025."
    # The compared entities are the observed levels of ONE categorical dimension and the metric is
    # ONE numeric measure. Both roles are validated against the dataset schema (a numeric column can
    # never be the dimension, a text column can never be the measure) and every named level must
    # exist in the data -- otherwise nothing is bound and the question stays unresolved rather than
    # silently comparing a subset or swapping roles. Single-character levels ("A") are handled here
    # because the generic value scanner deliberately ignores them.
    entity_cmp = re.fullmatch(
        r"(?:please\s+)?compare\s+(?P<head>.+?)\s+by\s+(?P<agg>average|avg|mean|total|sum)\s+(?P<measure>.+?)"
        r"(?:\s+(?:in|for|during)\s+.+?)?[?.!]*",
        q_clean, re.I,
    )
    if entity_cmp and not target and df is not None:
        _head_tokens = entity_cmp.group("head").split()
        _measure_col = resolve_phrase_to_column(entity_cmp.group("measure").strip(), cols)
        if (
            _measure_col and _measure_col in df.columns
            and pd.api.types.is_numeric_dtype(df[_measure_col]) and not _is_identifier_column(_measure_col)
        ):
            for _k in range(1, min(4, len(_head_tokens))):
                _dim_col = resolve_phrase_to_column(" ".join(_head_tokens[:_k]), cols)
                if not _dim_col or _dim_col == _measure_col or _dim_col not in df.columns or pd.api.types.is_numeric_dtype(df[_dim_col]):
                    continue
                _levels_text = " ".join(_head_tokens[_k:])
                _requested = [part.strip().strip("'\"") for part in re.split(r",|\band\b|&|\bversus\b|\bvs\.?", _levels_text, flags=re.I) if part.strip().strip("'\"")]
                _observed = {str(v).strip().casefold(): str(v).strip() for v in df[_dim_col].dropna().unique()}
                if len(_requested) >= 2 and len(set(n.casefold() for n in _requested)) == len(_requested) and all(n.casefold() in _observed for n in _requested):
                    target = _measure_col
                    grouping = [_dim_col]
                    comparison_values = [_observed[n.casefold()] for n in _requested]
                    task_family = "DESCRIPTIVE"
                    estimand = "mean" if entity_cmp.group("agg").lower() in {"average", "avg", "mean"} else "sum"
                    evidence["descriptive_comparison"] = {"values": comparison_values, "group": _dim_col}
                    break

    # Explicit descriptive comparison: aggregate one resolved measure for observed
    # levels of one resolved dimension. No significance or causal claim is implied.
    compare_aggregate = re.fullmatch(
        r"compare\s+((?:average|avg|mean|total|sum|count)\s+.+?)\s+(?:in|for|between)\s+(.+?)[?.!]*", q_clean, re.I,
    )
    if compare_aggregate and not target:
        comparison_target = resolve_phrase_to_column(compare_aggregate.group(1), cols)
        comparison_group, observed_levels = scan_observed_category_values(compare_aggregate.group(2), df)
        requested_levels = [part.strip().strip("'\"").rstrip("?.!") for part in re.split(r"\s+(?:and|versus|vs)\s+", compare_aggregate.group(2), flags=re.I)]
        # Every named level must exist; never silently compare a subset of the request.
        if comparison_target and comparison_group and len(requested_levels) >= 2 and {v.casefold() for v in requested_levels} == {v.casefold() for v in observed_levels}:
            target = comparison_target
            grouping = [comparison_group]
            comparison_values = [next(v for v in observed_levels if v.casefold() == name.casefold()) for name in requested_levels]
            task_family = "DESCRIPTIVE"
            estimand = "mean" if re.match(r"average|avg|mean", compare_aggregate.group(1), re.I) else "sum"
            evidence["descriptive_comparison"] = {"values": comparison_values, "group": comparison_group}

    # Temporal comparison: "Compare revenue this year and last year"
    temporal_compare = re.match(
        r"^(?:(?:please\s+)?compare\s+)?([A-Za-z_][A-Za-z0-9_ ]*?)\s+(?:between\s+)?(?:this|current)\s+year\s+and\s+(?:last|previous|prior)\s+year[?.!]*$",
        ql,
    )
    if temporal_compare and not target:
        t_phrase = temporal_compare.group(1).strip()
        t_col = resolve_phrase_to_column(t_phrase, cols)
        if t_col and t_col in df.columns and pd.api.types.is_numeric_dtype(df[t_col]):
            date_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c]) or re.search(r"(?:date|time|timestamp|_at)$", c, re.I)]
            if len(date_cols) == 1:
                from packages.analytics_core.src.intelligence.intent_state import _relative_time_range
                paired_range = _relative_time_range(ql, df, date_cols[0])
                if paired_range and paired_range.get("type") == "absolute" and paired_range.get("baseline_start"):
                    target = t_col
                    time_col = date_cols[0]
                    task_family = "DESCRIPTIVE"
                    claim_type = requested_claim = supported_claim = claim_ceiling = "OBSERVATION"
                    estimand = "sum"
                    req_agg = "SUM"
                    evidence["paired_time_range"] = paired_range

    # 6. Group Comparison: "Do smokers tip more than non-smokers?" / "Do male penguins weigh more than female penguins?" / "Do credit card payers tip more than cash payers?" / "Do enterprise customers have higher retention than SMB customers?"
    do_compare_match = re.search(
        r"(?:do|does|did|is|are)\s+(.+?)\s+([a-zA-Z0-9_]+)\s+(?:more|less|higher|lower)\s+than\s+(.+?)(?:\?|$)",
        ql,
    )
    do_compare_metric_match = re.search(
        r"(?:do|does|did|is|are)\s+(.+?)\s+(?:have|show|exhibit|experience)?\s*(?:more|less|higher|lower|greater)\s+([a-zA-Z0-9_\s]+?)\s+than\s+(.+?)(?:\?|$)",
        ql,
    )
    if (do_compare_match or do_compare_metric_match) and not target and not re.search(r"\bwhy\b", ql):
        if do_compare_match:
            side_a = do_compare_match.group(1).strip()
            action_metric = do_compare_match.group(2).strip()
            side_b = do_compare_match.group(3).strip()
        else:
            side_a = do_compare_metric_match.group(1).strip()
            action_metric = do_compare_metric_match.group(2).strip()
            side_b = do_compare_metric_match.group(3).strip()

        cand_target = resolve_phrase_to_column(action_metric, cols)
        if not cand_target and do_compare_match and do_compare_metric_match:
            # Fall back to alternative match if Form A failed to resolve metric
            side_a = do_compare_metric_match.group(1).strip()
            action_metric = do_compare_metric_match.group(2).strip()
            side_b = do_compare_metric_match.group(3).strip()
            cand_target = resolve_phrase_to_column(action_metric, cols)

        grp_col_a = resolve_phrase_to_column(side_a, cols)
        grp_col_b = resolve_phrase_to_column(side_b, cols)
        cat_group, cat_vals = scan_observed_category_values(f"{side_a} {side_b}", df)

        chosen_group = grp_col_a if (grp_col_a and grp_col_a == grp_col_b) else (cat_group or grp_col_a or grp_col_b)
        if not chosen_group and semantic is not None:
            sem_grp = getattr(semantic, "group_dimension_col", None)
            if sem_grp and sem_grp in cols:
                chosen_group = sem_grp

        if cand_target and chosen_group and cand_target != chosen_group:
            target = cand_target
            grouping = [chosen_group]
            task_family = "GROUP_COMPARISON"
            claim_type = "ASSOCIATION"
            requested_claim = "ASSOCIATION"
            supported_claim = "ASSOCIATION"
            claim_ceiling = "ASSOCIATION"
            estimand = "group_difference"
            comparison_values = cat_vals or [side_a, side_b]
            evidence["group_comparison"] = {"target": target, "group": chosen_group, "values": comparison_values}

    # 7. "How does bill length differ between species?"
    how_diff_match = re.search(r"how\s+does\s+(.+?)\s+differ\s+between\s+(.+?)(?:\?|$)", ql)
    if how_diff_match and not target:
        m_col = resolve_phrase_to_column(how_diff_match.group(1).strip(), cols)
        g_col = resolve_phrase_to_column(how_diff_match.group(2).strip(), cols)
        if m_col and g_col:
            target = m_col
            grouping = [g_col]
            task_family = "GROUP_COMPARISON"
            claim_type = "ASSOCIATION"
            requested_claim = "ASSOCIATION"
            supported_claim = "ASSOCIATION"
            claim_ceiling = "ASSOCIATION"
            estimand = "group_difference"

    # 7b. "Compare revenue between territory" / "Revenue versus territory": a numeric measure
    # compared across the groups of one categorical column.
    if not target:
        cmp_m = (
            re.search(r"^\s*(?:compare|comparison\s+of)\s+(.+?)\s+(?:between|across|among)\s+(.+?)\s*[?.!]*\s*$", ql)
            or re.search(r"^\s*(.+?)\s+(?:versus|vs\.?)\s+(.+?)\s*[?.!]*\s*$", ql)
        )
        if cmp_m:
            m_col = resolve_phrase_to_column(cmp_m.group(1).strip(), cols)
            g_col = resolve_phrase_to_column(cmp_m.group(2).strip(), cols)
            if (
                m_col and g_col and m_col != g_col
                and pd.api.types.is_numeric_dtype(df[m_col])
                and not pd.api.types.is_numeric_dtype(df[g_col])
            ):
                target = m_col
                grouping = [g_col]
                task_family = "GROUP_COMPARISON"
                claim_type = "ASSOCIATION"
                requested_claim = "ASSOCIATION"
                supported_claim = "ASSOCIATION"
                claim_ceiling = "ASSOCIATION"
                estimand = "group_difference"
                evidence["group_comparison"] = {"target": target, "group": g_col, "values": []}

    # 8. Diagnostic / Why: preserve compound intent such as
    # "which region had the highest return rate and why?". Previously the ranking
    # branch bound target+group first, so this block's `and not target` guard
    # silently dropped the explanatory clause and left a ranking-only contract.
    _why_requested = bool(re.search(
        r"\b(?:why|caus\w*|driv(?:e|es|en|er|ers|ing)|affect\w*|influenc\w*|impact\w*|explain\w*)\b",
        ql,
    ))
    _period_change_owns_question = task_family in ("PERIOD_CHANGE", "PERIOD_CHANGE_UNRESOLVED_TIMEFRAME", "TREND") and not grouping
    if _why_requested and (not target or (not _period_change_owns_question and task_family != "ROOT_CAUSE")):
        causal_language = True
        task_family = "ROOT_CAUSE"
        requested_claim = "CAUSAL"
        supported_claim = "ASSOCIATION"
        claim_ceiling = "ASSOCIATION"
        claim_type = "ASSOCIATION"
        estimand = "root_cause"
        # An outcome explicitly named in the question ("...lower survival than...") takes
        # precedence over the semantic engine's generic metric guess, which can latch onto a
        # grouping word instead (e.g. "class" -> pclass). Preserve an already resolved target
        # from an earlier ranking/comparison clause instead of re-guessing it from a filter value.
        if not target:
            for phrase in ["survival", "price", "mpg", "fare", "tip", "body mass", "weight", "conversion", "retention", "revenue", "sales", "cost", "profit", "churn"]:
                if phrase in ql:
                    c = resolve_phrase_to_column(phrase, cols)
                    if c:
                        target = c
                        break
        if not target and semantic is not None:
            sem_tgt = getattr(semantic, "target_metric_col", None) or getattr(semantic, "churn_event_col", None)
            if sem_tgt and sem_tgt in cols:
                target = sem_tgt
        if not target:
            for c in cols:
                c_norm = normalize_token(c)
                if re.search(rf"\b{re.escape(c_norm)}\b", ql):
                    target = c
                    break
        if not grouping:
            g_cand, g_vals = scan_observed_category_values(q_clean, df)
            if g_cand and g_cand != target:
                grouping = [g_cand]
                comparison_values = g_vals
        if not grouping and semantic is not None:
            # Discover dimensions from the actual dataset schema rather than a list
            # of demonstration-dataset fields (class/species/cut/etc.). Normalize
            # snake_case and spaced names, prefer longer phrases, and only bind a
            # candidate that resolves back to a real physical column.
            schema_phrases = sorted(
                {normalize_token(c) for c in cols if normalize_token(c)},
                key=lambda phrase: (-len(phrase), phrase),
            )
            q_norm = normalize_token(ql)
            for dim_phrase in schema_phrases:
                if re.search(rf"(?<![a-z0-9]){re.escape(dim_phrase)}(?![a-z0-9])", q_norm):
                    c = resolve_phrase_to_column(dim_phrase, cols)
                    if c and c != target:
                        grouping = [c]
                        break
            # A final alias pass supports business synonyms, but only aliases
            # that resolve uniquely against this dataset are eligible.
            if not grouping:
                for dim_phrase in sorted(APPROVED_SEMANTIC_ALIASES, key=lambda phrase: (-len(phrase), phrase)):
                    if re.search(rf"(?<![a-z0-9]){re.escape(dim_phrase)}(?![a-z0-9])", q_norm):
                        c = resolve_phrase_to_column(dim_phrase, cols)
                        if c and c != target:
                            grouping = [c]
                            break

    # 9. Descriptive Breakdown: "What is the survival rate by passenger class?" / "What is the average tip by day?" / "churn rate by plan_tier"
    desc_by_match = re.search(r"^(?:(?:what\s+(?:is|are|was|were)\s+(?:the\s+)?)|calculate\s+|show\s+|find\s+|compute\s+)?(.+?)\s+by\s+(.+?)(?:\?|$)", ql)
    if desc_by_match and not target:
        m_phrase = desc_by_match.group(1).strip()
        g_phrase = desc_by_match.group(2).strip()

        # If g_phrase contains split by / grouped by, prioritize the secondary grouping column
        split_m = re.search(r"\b(?:split\s+by|grouped\s+by|broken\s+down\s+by)\s+(.+)$", g_phrase, re.I)
        if split_m:
            split_candidate = resolve_phrase_to_column(split_m.group(1).strip(), cols)
            if split_candidate:
                g_phrase = split_candidate

        # Strip trailing clause indicators (for ..., where ..., excluding ...) from g_phrase
        g_phrase_clean = re.sub(r"\s+\b(?:for|where|excluding|exclude|without)\b.*$", "", g_phrase, flags=re.I).strip()

        m_col = resolve_phrase_to_column(m_phrase, cols)
        g_col = resolve_phrase_to_column(g_phrase_clean, cols) or resolve_phrase_to_column(g_phrase, cols)
        # Inverted roles (a text/categorical column as the measure with a numeric column as the
        # dimension, e.g. "Compare regions A and B by total revenue") are never bound: fail closed.
        _roles_inverted = bool(
            m_col and g_col and m_col in df.columns and g_col in df.columns
            and not pd.api.types.is_numeric_dtype(df[m_col]) and pd.api.types.is_numeric_dtype(df[g_col])
        )
        if m_col and g_col and not _roles_inverted:
            target = m_col
            grouping = [g_col]
            task_family = "DESCRIPTIVE"
            claim_type = "OBSERVATION"
            requested_claim = "OBSERVATION"
            supported_claim = "OBSERVATION"
            claim_ceiling = "OBSERVATION"
            estimand = "mean" if "average" in m_phrase or "avg" in m_phrase else "rate" if "rate" in m_phrase else "sum"

    # 9a. Distinct Values / Category Exploration: "what are the different tiers of subscriptions?" / "list distinct plan tiers"
    # Aggregate-opening questions like "what is the average region?" are owned by the fail-closed
    # SIMPLE_AGGREGATE enforcement path, not by the categorical-exploration branch.
    if not target and not re.search(r"\b(?:average|avg|mean|median|total|sum)\b", ql):
        dist_match = re.search(
            r"^(?:(?:what\s+(?:are|is)\s+(?:the\s+)?(?:different\s+)?)|list\s+(?:all\s+)?|show\s+(?:all\s+)?|distinct\s+)"
            r"([a-zA-Z0-9_\s]+?)(?:\?|$)",
            ql,
        )
        if dist_match:
            cand_phrase = dist_match.group(1).strip()
            cand_col = resolve_phrase_to_column(cand_phrase, cols)
            if cand_col and (
                pd.api.types.is_object_dtype(df[cand_col])
                or pd.api.types.is_string_dtype(df[cand_col])
                or isinstance(df[cand_col].dtype, pd.CategoricalDtype)
                or df[cand_col].nunique() <= 40
            ):
                target = cand_col
                grouping = [cand_col]
                task_family = "DESCRIPTIVE"
                claim_type = "OBSERVATION"
                requested_claim = "OBSERVATION"
                supported_claim = "OBSERVATION"
                claim_ceiling = "OBSERVATION"
                estimand = "count"
                requested_agg = "COUNT"

    # 9b. Plain simple aggregate: "What is the average revenue?" / "Total revenue by region" /
    # "median quantity". The authority firewall enforces SIMPLE_AGGREGATE, so the contract must
    # resolve these itself rather than leave them to the (now vetoed) semantic-layer fallback.
    # Fail-closed: the WHOLE measure phrase (and group phrase, if any) must resolve to a physical
    # column and the measure must be numeric; anything else (unknown column, "per order", date
    # filters) stays GENERAL_EXPLORATION and is refused as INCOMPLETE.
    if task_family == "GENERAL_EXPLORATION" and not target and not _NON_ENFORCED_SIGNAL_RE.search(ql):
        # Anchored: the aggregate word must OPEN the question (optionally after a plain lead-in such as
        # "what is the"). An unanchored search matched "total order amount" inside "Which customer
        # segment generates the highest total order amount?", resolved it to a confident
        # grouping-free DESCRIPTIVE contract, and authoritatively erased the cross-table dimension.
        agg_m = re.search(
            r"^\s*(?:(?:what\s+(?:is|are|was|were)|what's|show(?:\s+me)?|calculate|compute|give\s+me|find|tell\s+me)\s+)?"
            r"(?:the\s+)?(standard\s+deviation|stddev|stdev|variance|var|average|avg|mean|median|total|sum)\s+(?:of\s+)?(?:the\s+)?(.+?)(?:(?:\s+by\s+|\s+for\s+each\s+)(.+?))?\s*[?.!]?\s*$",
            ql,
        )
        if agg_m:
            agg_word, m_phrase, g_phrase = agg_m.group(1), agg_m.group(2).strip(), (agg_m.group(3) or "").strip()
            # The phrase resolver is fuzzy (token overlap), so a qualified phrase such as
            # "revenue per order" would silently collapse to "revenue" and drop the qualifier
            # (a different metric). Only bare noun phrases may resolve here.
            qualified = re.search(
                r"\b(per|each|every|for|in|on|from|where|when|during|with|without|over|between|among|"
                r"within|after|before|since|than|and|or|of)\b|[,;:()]",
                f"{m_phrase} {g_phrase}",
            )
            # "gross sales for delivered orders": the measure is the head; the tail is a population
            # scope handled by filter binding (and guarded below if nothing consumes it).
            _scope_split = re.match(r"^(.+?)\s+(?:for|among|in|within|of)\s+(.+)$", m_phrase)
            if qualified and _scope_split and not re.search(r"\b(per|each|every)\b|[,;:()]", f"{_scope_split.group(1)} {g_phrase}"):
                _head = _scope_split.group(1).strip()
                _head_col = resolve_phrase_to_column(_head, cols)
                if _head_col and pd.api.types.is_numeric_dtype(df[_head_col]) and not re.search(r"\b(and|or|than|between|after|before|since|over|during|where|when)\b", _head):
                    m_phrase, qualified = _head, None
                    evidence["measure_scope_tail"] = _scope_split.group(2).strip()
            m_col = None if qualified else resolve_phrase_to_column(m_phrase, cols)
            g_col = resolve_phrase_to_column(g_phrase, cols) if (g_phrase and not qualified) else None
            numeric_ok = bool(m_col) and pd.api.types.is_numeric_dtype(df[m_col])
            group_ok = (not g_phrase) or (bool(g_col) and g_col != m_col)
            if numeric_ok and group_ok:
                target = m_col
                grouping = [g_col] if g_col else []
                task_family = "DESCRIPTIVE"
                claim_type = "OBSERVATION"
                requested_claim = "OBSERVATION"
                supported_claim = "OBSERVATION"
                claim_ceiling = "OBSERVATION"
                estimand = {"average": "mean", "avg": "mean", "mean": "mean", "median": "median", "variance": "variance", "var": "variance", "standard deviation": "stddev", "stddev": "stddev", "stdev": "stddev"}.get(agg_word, "sum")
            elif not numeric_ok:
                # Fail closed for invalid aggregate requests such as "What is the average region?".
                # Even when the phrase resolves to a categorical column, it is still not a valid
                # numeric measure for the enforced SIMPLE_AGGREGATE family and must remain incomplete.
                task_family = "DESCRIPTIVE"
                target = None
                grouping = []
                evidence["aggregate_rejection"] = {
                    "agg_word": agg_word,
                    "measure_phrase": m_phrase,
                    "group_phrase": g_phrase,
                    "resolved_measure_column": m_col,
                    "reason": "measure_is_not_numeric",
                }

    # 10. Prediction / Risk: "Which customers are at highest risk of churn next month?" / "Which customers are likely to churn?"
    if (re.search(r"\b(risk|probability|likely|likelihood)\s+of\s+([a-zA-Z0-9_\s]+)", ql) or
        re.search(r"\b(?:which|who)\s+(?:customers?|users?|accounts?)\s+(?:are\s+)?(?:at\s+(?:highest|lowest|top)\s+risk|likely\s+to)\b", ql) or
        re.search(r"\blikely\s+to\s+(?:churn|attrition|cancel|default|convert|leave)\b", ql)) and not target:
        task_family = "PREDICTION"
        claim_type = "PREDICTION"
        requested_claim = "PREDICTION"
        supported_claim = "PREDICTION"
        claim_ceiling = "PREDICTION"
        estimand = "probability"
        for c_name in ["churn", "churned", "attrition", "default"]:
            c = resolve_phrase_to_column(c_name, cols)
            if c:
                target = c
                break
        if not target and semantic is not None:
            target = getattr(semantic, "churn_event_col", None) or getattr(semantic, "target_metric_col", None)

    # Explicit filter-plus-split form:
    # "revenue by region West split by channel" means filter region=West,
    # then aggregate the remaining rows by channel. Do this before the generic
    # grouping-clause parser so it cannot misread "region West" as a dimension.
    # Both columns must resolve against the physical schema; otherwise leave the
    # existing conservative parser/clarification behavior unchanged.
    filtered_split_group = None
    if target and task_family in {"DESCRIPTIVE", "COUNT", "RANKING"}:
        split_match = re.search(
            r"\b(?:by|across|per)\s+(?P<filter_dim>[A-Za-z0-9_]+)"
            r"\s+(?P<filter_value>[A-Za-z0-9_-]+)\s+split\s+by\s+"
            r"(?P<group_phrase>.+?)"
            r"(?=\s+(?:where|during|within|excluding|exclude|without|including|include|"
            r"for\b|over\s+the|sorted\s+by|rank(?:ed)?\s+by)\b|"
            r",\s*(?:where|during|within|excluding|exclude|without|including|include|for\b)|[?.!;]|$)",
            q_clean,
            re.I,
        )
        if split_match:
            filter_dim = resolve_phrase_to_column(split_match.group("filter_dim"), cols)
            group_col = resolve_phrase_to_column(split_match.group("group_phrase").strip(), cols)
            filter_value = split_match.group("filter_value").strip()
            if filter_dim and group_col and filter_dim != group_col and filter_value.casefold() not in {
                str(c).casefold() for c in cols
            }:
                grouping = [group_col]
                filtered_split_group = (filter_dim, filter_value, group_col)
                evidence["filtered_split_group_resolution"] = {
                    "filter_column": filter_dim,
                    "filter_value": filter_value,
                    "grouping_columns": [group_col],
                    "method": "schema_grounded_filter_then_split",
                }

    # Resolve explicit multi-dimensional grouping for descriptive/count/ranking requests.
    # This is deliberately schema-grounded: every dimension must resolve to exactly one
    # physical column, otherwise the existing unresolved/clarification path remains in force.
    if target and task_family in {"DESCRIPTIVE", "COUNT", "RANKING"}:
        multi_group_match = re.search(
            r"\b(?:grouped\s+by|split\s+by|broken\s+down\s+by|by|per|across)\s+(.+?)"
            r"(?=\s+(?:where|during|within|excluding|exclude|without|including|include|for\b|over\s+the|sorted\s+by|rank(?:ed)?\s+by|split\s+by)\b|[?.!;]|$)",
            q_clean, re.I,
        )
        if multi_group_match:
            raw_group_phrase = multi_group_match.group(1).strip()
            group_parts = [part.strip() for part in re.split(r",|\band\b|\bthen\b", raw_group_phrase, flags=re.I) if part.strip()]
            resolved_groups: list[str] = []
            for part in group_parts:
                resolved = resolve_phrase_to_column(part, cols)
                if not resolved:
                    resolved_groups = []
                    break
                if resolved not in resolved_groups:
                    resolved_groups.append(resolved)
            if len(group_parts) > 1 and resolved_groups and len(resolved_groups) == len(group_parts):
                grouping = resolved_groups
                evidence["multi_group_resolution"] = {
                    "source_phrase": raw_group_phrase,
                    "resolved_columns": list(resolved_groups),
                    "method": "schema_grounded_explicit_conjunction",
                }

    # Determine requested aggregation
    req_agg = extract_aggregation(q_clean, target)
    if req_agg is None and evidence.get("grouped_compare_intent") and target and grouping:
        # Explicitly asking how a measure compares across groups implies a side-by-side
        # aggregate. Default to SUM unless the wording specified mean/median/etc.; the
        # preview records that default before approval and execution.
        req_agg = "SUM"
    if (task_family == "COUNT" or evidence.get("count")) and "requested_agg" in locals():
        req_agg = requested_agg
    if estimand in ("correlation", "multi_correlation") and req_agg not in ("CORRELATION", "MEAN", None):
        req_agg = "CORRELATION"

    filter_conditions = extract_filter_conditions(q_clean, cols)
    # Complete two high-value deterministic forms that the legacy pattern cascade can
    # miss: numeric thresholds attached to an explicit column, and exclusions whose
    # surface form is a close inflection of an observed category (refunds/refunded).
    filter_conditions = tuple(filter_conditions)
    known_filter_keys = {(str(x.get("column", "")).casefold(), str(x.get("operator", "").upper()), str(x.get("value", ""))) for x in filter_conditions}
    # Bind EVERY explicit numeric threshold attached to a schema column. Counting
    # threshold clauses prevents one parsed predicate from masking a second dropped one.
    threshold_pattern = re.compile(
        r"(?<![\w])(?P<column>" + "|".join(re.escape(str(c)) for c in sorted(cols, key=len, reverse=True)) +
        r")\s+(?P<cmp>above|over|greater than|more than|exceed(?:s|ed)?|below|under|less than|fewer than|at least|at most)\s+\$?(?P<number>-?\d[\d,]*(?:\.\d+)?)",
        re.I,
    ) if cols else None
    threshold_clauses = list(threshold_pattern.finditer(q_clean)) if threshold_pattern else []
    for threshold in threshold_clauses:
        col = next((str(c) for c in cols if str(c).casefold() == threshold.group("column").casefold()), threshold.group("column"))
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        word = threshold.group("cmp").casefold()
        op = ">=" if word == "at least" else "<=" if word == "at most" else ">" if word in {"above", "over", "greater than", "more than", "exceed", "exceeds", "exceeded"} else "<"
        value = float(threshold.group("number").replace(",", ""))
        if not any(k[0] == col.casefold() and k[1] == op and k[2] == str(value) for k in known_filter_keys):
            filter_conditions += ({"column": col, "operator": op, "value": value},)
            known_filter_keys.add((col.casefold(), op, str(value)))

    # A coordinated lifecycle exclusion such as "excluding returns and cancellations"
    # must bind both observed status categories to one unique status column. Resolving only
    # the first plural noun (returns -> Returned) would silently retain Cancelled records.
    # This is schema- and data-grounded: no NOT IN predicate is emitted unless every term
    # maps to exactly one observed value in the same unique status/state column.
    exclusion_resolved = False
    coordinated_exclusion = re.search(
        r"\b(?:exclude|excluding|except|without|ignore)\s+"
        r"(?P<items>(?:returns?|returned|returning|cancellations?|cancelations?|cancelled|canceled|cancels?)"
        r"(?:\s*(?:,|\band\b|\bor\b)\s*"
        r"(?:returns?|returned|returning|cancellations?|cancelations?|cancelled|canceled|cancels?))+)\b",
        q_clean, re.I,
    )
    if coordinated_exclusion:
        raw_terms = re.findall(
            r"returns?|returned|returning|cancellations?|cancelations?|cancelled|canceled|cancels?",
            coordinated_exclusion.group("items"), re.I,
        )
        status_cols_for_exclusion = [
            str(c) for c in cols
            if any(token in normalize_token(str(c)) for token in ("status", "state"))
        ]
        # Prefer a uniquely identified order-status field when present; otherwise require
        # exactly one status/state field. Never infer a lifecycle filter on a random column.
        order_status_cols = [
            c for c in status_cols_for_exclusion
            if "order" in normalize_token(c) and "delivery" not in normalize_token(c)
        ]
        if order_status_cols:
            status_cols_for_exclusion = order_status_cols
        if len(status_cols_for_exclusion) == 1 and len(raw_terms) >= 2:
            status_col = status_cols_for_exclusion[0]
            observed_values = list(df[status_col].dropna().unique()) if status_col in df.columns else []

            def _lifecycle_kind(term: str) -> Optional[str]:
                token = re.sub(r"[^a-z0-9]+", "", term.casefold())
                if token in {"return", "returns", "returned", "returning"}:
                    return "return"
                if token in {"cancel", "cancels", "cancelled", "canceled", "cancellation", "cancellations", "cancelation", "cancelations"}:
                    return "cancel"
                return None

            def _observed_lifecycle_kind(value: Any) -> Optional[str]:
                token = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
                if token in {"return", "returns", "returned", "returning"}:
                    return "return"
                if token in {"cancel", "cancels", "cancelled", "canceled", "cancellation", "cancellations", "cancelation", "cancelations"}:
                    return "cancel"
                return None

            wanted_kinds = [_lifecycle_kind(term) for term in raw_terms]
            excluded_values: list[Any] = []
            all_terms_resolved = all(kind is not None for kind in wanted_kinds)
            if all_terms_resolved:
                for kind in wanted_kinds:
                    matches = [value for value in observed_values if _observed_lifecycle_kind(value) == kind]
                    if len(matches) != 1:
                        all_terms_resolved = False
                        break
                    if not any(str(matches[0]).casefold() == str(existing).casefold() for existing in excluded_values):
                        excluded_values.append(matches[0])
            if all_terms_resolved and len(excluded_values) == len(set(wanted_kinds)):
                # Replace the parser's partial single-value exclusion for the same status
                # categories with one exact composite predicate; keep unrelated predicates.
                _wanted_strings = {str(value).casefold() for value in excluded_values}
                kept_filters = []
                for item in filter_conditions:
                    same_col = str(item.get("column", "")).casefold() == status_col.casefold()
                    operator = str(item.get("operator", "")).upper()
                    value = item.get("value")
                    existing_values = value if isinstance(value, (list, tuple, set)) else [value]
                    is_matching_exclusion = same_col and operator in {"!=", "NOT IN"} and all(
                        str(v).casefold() in _wanted_strings for v in existing_values
                    )
                    if not is_matching_exclusion:
                        kept_filters.append(item)
                filter_conditions = tuple(kept_filters) + ({
                    "column": status_col, "operator": "NOT IN", "value": excluded_values,
                },)
                exclusion_resolved = True
                known_filter_keys = {(str(x.get("column", "")).casefold(), str(x.get("operator", "").upper()), str(x.get("value", ""))) for x in filter_conditions}

    # Inflection recovery is allowed only when exactly one observed categorical value
    # is a strong lexical match across the entire dataset. Never pick a winner merely
    # because it scored slightly higher than a competing column/value.
    exclusion_match = re.search(r"\b(?:excluding|exclude|except|without)\s+(?:the\s+)?([a-z][a-z0-9_-]*)\b", q_clean, re.I)
    if exclusion_match:
        excluded = exclusion_match.group(1).casefold()
        candidates = []
        def inflection_key(text: str) -> str:
            token = re.sub(r"[^a-z0-9]+", "", text.casefold())
            for suffix in ("ies", "ing", "ed", "es", "s"):
                if token.endswith(suffix) and len(token) > len(suffix) + 2:
                    token = token[:-len(suffix)] + ("y" if suffix == "ies" else "")
                    break
            return token
        excluded_key = inflection_key(excluded)
        for col in df.columns:
            if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
                continue
            for value in df[col].dropna().unique():
                observed_text = str(value).casefold()
                observed_key = inflection_key(observed_text)
                score = 1.0 if excluded_key == observed_key else SequenceMatcher(None, excluded_key, observed_key).ratio()
                # Inflection/near-spelling recovery only, not semantic guessing.
                if score >= 0.82 and (excluded_key == observed_key or score >= 0.90):
                    candidates.append((score, str(col), value))
        # Require a single unique best observed value; if the same best match occurs
        # in multiple dimensions, the intended filter column is ambiguous.
        candidates.sort(key=lambda item: item[0], reverse=True)
        if candidates:
            best_score = candidates[0][0]
            best = [item for item in candidates if abs(item[0] - best_score) < 1e-9]
            unique_best = {(col.casefold(), str(value).casefold()) for _, col, value in best}
            if len(unique_best) == 1:
                _, col, value = best[0]
                if not any(str(f.get("column", "")).casefold() == col.casefold() and str(f.get("operator", "")).upper() in {"!=", "NOT IN"} for f in filter_conditions):
                    filter_conditions += ({"column": col, "operator": "!=", "value": value},)
                exclusion_resolved = True
    # Population qualifiers named by value ("Corporate orders", "delivered and not returned"):
    # bind every value that maps to exactly one column instead of silently dropping it.
    # Diagnostic wording normally avoids treating a comparison cohort as a filter (for
    # example, "why is conversion higher in paid traffic?"). An explicit population
    # preface ("For Small Business orders ...") is different: those conditions constrain
    # every part of the compound question and must survive the diagnostic classification.
    _explicit_population_preface = bool(re.search(r"^\s*(?:for|among|within)\b", ql)) and bool(
        re.search(r"\b(?:orders?|customers?|records?|transactions?|sales)\b", ql)
    )
    if df is not None and (not _why_requested or _explicit_population_preface):
        for extra in extract_observed_value_filters(
            q_clean, df, existing=filter_conditions,
            skip_columns=tuple(grouping or ()) + ((target,) if target else ()),
        ):
            filter_conditions += (extra,)
        if task_family not in ("GENERAL_EXPLORATION",):
            try:
                _flags = _flag_population_filters(
                    q_clean, df, filter_conditions,
                    set(filter(None, [target, *list(grouping or ()), *list(locals().get("explanatory") or ())])),
                )
            except Exception:
                _flags = []
            for extra in _flags:
                filter_conditions += (extra,)
    # Validate categorical predicates against observed values before they can authorize
    # execution. A typo/inflection must never silently turn an exclusion into a no-op.
    invalid_filter_values: List[str] = []
    validated_filters: List[Dict[str, Any]] = []
    for predicate in filter_conditions:
        pred = dict(predicate)
        col = str(pred.get("column", ""))
        op = str(pred.get("operator", "")).upper()
        value = pred.get("value")
        if col in df.columns and op in {"=", "!=", "IN", "NOT IN"} and df[col].dtype == object:
            values = value if isinstance(value, list) else [value]
            observed = [v for v in df[col].dropna().unique()]
            mapped = []
            for item in values:
                exact = next((v for v in observed if str(v).casefold() == str(item).casefold()), None)
                if exact is not None:
                    mapped.append(exact)
                    continue
                near = [(SequenceMatcher(None, str(item).casefold(), str(v).casefold()).ratio(), v) for v in observed]
                near = sorted(near, key=lambda pair: pair[0], reverse=True)
                if near and near[0][0] >= 0.78 and (len(near) == 1 or near[0][0] - near[1][0] >= 0.08):
                    mapped.append(near[0][1])
                else:
                    invalid_filter_values.append(f"{col}:{item}")
                    mapped.append(item)
            pred["value"] = mapped if isinstance(value, list) else mapped[0]
        validated_filters.append(pred)
    filter_conditions = tuple(validated_filters)
    if filtered_split_group is not None:
        filter_col, filter_value, group_col = filtered_split_group
        # Reuse the canonical extracted predicate where available, preserving observed
        # category casing; synthesize it only if the conservative filter parser did not.
        matches_filter = any(
            str(item.get("column", "")).casefold() == filter_col.casefold()
            and str(item.get("operator", "")).upper() == "="
            and str(item.get("value", "")).casefold() == filter_value.casefold()
            for item in filter_conditions
        )
        if not matches_filter:
            observed = next(
                (value for value in df[filter_col].dropna().unique()
                 if str(value).casefold() == filter_value.casefold()),
                filter_value,
            )
            filter_conditions = tuple(filter_conditions) + ({
                "column": filter_col, "operator": "=", "value": observed,
            },)
    if evidence.get("descriptive_comparison"):
        filter_conditions += ({"column": grouping[0], "operator": "IN", "value": list(comparison_values)},)

    # Clean up unresolved roles
    unresolved: List[str] = []
    # "Compare revenue between territory" uses "between" to name the compared groups, not a range filter.
    _ql_filter_scan = (
        re.sub(r"\bbetween\b", " ", ql)
        if (evidence.get("group_comparison") is not None and not (evidence.get("group_comparison") or {}).get("values"))
        else ql
    )
    # In ranking language, "from" is part of a direction phrase, not a population
    # filter. Without this normalization, the ordinary request "ranked from highest
    # to lowest" leaves a false unresolved filter_instruction and blocks an otherwise
    # complete grouped aggregate. Keep genuine source/scope clauses such as "from 2024"
    # untouched.
    _ql_filter_scan = re.sub(
        r"\b(?:(?:ranked|sorted|ordered)\s+)?from\s+(?:highest\s+to\s+lowest|lowest\s+to\s+highest)\b",
        " ",
        _ql_filter_scan,
        flags=re.I,
    )
    filter_intent = re.search(
        r"\b(?:where|only|excluding|exclude|without|except|ignore|during|between|from|greater than|less than|above|below|over|under|at least|at most)\b|\bin\s+20\d{2}\b|\b(?:placed|ordered|created)\s+in\s+20\d{2}\b",
        _ql_filter_scan, re.I,
    )
    if filter_intent and not filter_conditions:
        unresolved.append("filter_instruction")
    if invalid_filter_values:
        unresolved.append("filter_value_unmatched:" + ",".join(invalid_filter_values))
    # A capitalised population word ("Among Enterprise customers") that matches no observed value
    # must not be dropped silently; the human is asked about exactly that word.
    try:
        _bound_text = " ".join(str(f.get("value", "")) for f in filter_conditions).lower()
        _pop = re.search(
            r"(?i:\b(?:among|for|within|in|of))\s+((?:[A-Z][\w&-]*)(?:\s+[A-Z][\w&-]*)?)\s+(?:orders?|customers?|accounts?|users?|clients?|transactions?|segments?)\b",
            str(question),
        )
        if _pop and _pop.start(1) > 0:
            _w = _pop.group(1).strip()
            _known_cols = {re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in df.columns}
            _in_data = any(
                _w.lower() == str(v).lower()
                for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= 200
                for v in df[c].dropna().unique()
            )
            if not _in_data and _w.lower() not in _bound_text and re.sub(r"[^a-z0-9]", "", _w.lower()) not in _known_cols:
                unresolved.append("filter_value_unmatched:" + _w)
    except Exception:
        pass
    # A numeric threshold is a material part of the requested population. If the
    # parser failed to produce a typed predicate, refuse instead of returning the
    # unfiltered aggregate.
    threshold_column_alt = "|".join(re.escape(str(c)) for c in sorted(cols, key=len, reverse=True))
    numeric_threshold_intent_pattern = (
        r"(?<![\w])(?:(?P<column>" + threshold_column_alt + r")\s+)?"
        if threshold_column_alt else r"(?<![\w])(?:(?P<column>))?"
    ) + r"(?P<comparison>above|over|greater than|more than|exceed(?:s|ed)?|below|under|less than|fewer than|at least|at most)\s+\$?(?P<number>\d[\d,]*(?:\.\d+)?)"
    numeric_threshold_intent = list(re.finditer(numeric_threshold_intent_pattern, ql, re.I))
    numeric_predicates = [item for item in filter_conditions if str(item.get("operator", "")).upper() in {">", ">=", "<", "<="}]
    if len(numeric_predicates) < len(numeric_threshold_intent):
        unresolved.append("numeric_threshold_unbound")
    if exclusion_match and not exclusion_resolved and not any(
        str(item.get("operator", "")).upper() in {"!=", "NOT IN"} for item in filter_conditions
    ):
        unresolved.append("exclusion_value_or_column_ambiguous")

    # Explicit intent ledger for filter clauses: each detected material instruction is
    # recorded as consumed only when a corresponding typed predicate exists. This is
    # intentionally scoped to the high-risk filter family; it does not claim a complete
    # natural-language token proof for every task family.
    intent_ledger = []
    # Match each threshold clause to its own typed predicate. A count-only check is
    # unsafe: one unrelated predicate could otherwise make several dropped clauses
    # appear consumed. Preserve duplicate clauses as separate ledger entries.
    def _threshold_operator(text: str) -> str:
        phrase = text.casefold()
        if phrase.startswith(("at least",)):
            return ">="
        if phrase.startswith(("at most",)):
            return "<="
        if phrase.startswith(("above", "over", "greater than", "more than", "exceed")):
            return ">"
        return "<"

    for match in numeric_threshold_intent:
        clause = match.group(0)
        expected_value = float(match.group("number").replace(",", "")) if match.group("number") else None
        expected_op = _threshold_operator(match.group("comparison"))
        expected_column = match.groupdict().get("column")
        # Prefer a predicate whose numeric literal and comparison operator both match.
        # Do not let a different threshold satisfy this clause's ledger entry.
        consumed = expected_value is not None and any(
            str(item.get("operator", "")).upper() == expected_op
            and (not expected_column or str(item.get("column", "")).casefold() == expected_column.casefold())
            and isinstance(item.get("value"), (int, float, np.integer, np.floating))
            and math.isclose(float(item["value"]), expected_value, rel_tol=0.0, abs_tol=1e-9)
            for item in numeric_predicates
        )
        intent_ledger.append({"kind": "numeric_threshold", "text": clause, "status": "CONSUMED" if consumed else "LEFTOVER"})
    if exclusion_match:
        intent_ledger.append({"kind": "categorical_exclusion", "text": exclusion_match.group(0), "status": "CONSUMED" if exclusion_resolved else "LEFTOVER"})
    if filter_intent:
        intent_ledger.append({"kind": "filter_intent", "text": filter_intent.group(0), "status": "CONSUMED" if filter_conditions and not any(x["status"] == "LEFTOVER" for x in intent_ledger) else "LEFTOVER"})
    # A measure is never grouped by itself: "What is the churn rate?" is one overall rate,
    # not a table keyed by the churn flag.
    if target and req_agg and str(req_agg).upper() not in {"COUNT", "ROW_COUNT", "COUNT_DISTINCT"}:
        grouping = [g for g in grouping if g != target]
    if task_family == "COUNT" and evidence.get("count", {}).get("row_count") and re.search(r"\b(?:by|per|across|within|for\s+each|in\s+each)\b", ql) and not grouping:
        unresolved.append("grouping")
    if task_family in ("ASSOCIATION", "CAUSAL_REQUEST") and not target:
        unresolved.append("target")
    if task_family in ("ASSOCIATION", "CAUSAL_REQUEST") and not explanatory:
        unresolved.append("explanatory")
    if task_family in ("RANKING", "GROUP_COMPARISON", "ROOT_CAUSE") and not grouping:
        unresolved.append("grouping")
    # Ranking must have both roles. Previously a question could resolve the dimension
    # (e.g. region) but fail to resolve the measure (e.g. an ambiguous or unknown "revenue")
    # and still leave the contract marked complete. That produced an executable-looking
    # preview with no target metric. Fail closed and request a metric choice instead.
    if task_family == "RANKING" and not target:
        unresolved.append("target")

    # If ranking question with highest/lowest, set ranking_direction
    if not ranking_direction:
        if re.search(r"\b(highest|most|best|largest|top)\b", ql):
            ranking_direction = "DESC"
        elif re.search(r"\b(lowest|least|worst|smallest|bottom)\b", ql):
            ranking_direction = "ASC"

    if task_family == "GENERAL_EXPLORATION":
        res_conf = 0.0
    else:
        res_conf = 1.0 if not unresolved else 0.5

    # Enforcement state. Pre-ledger: COMPLETE only records "the existing cascade resolved this
    # family with no unresolved roles"; it does NOT yet prove every word was consumed (Step 3).
    # AMBIGUOUS is not produced until the ledger can tell it from INCOMPLETE.
    family = _enforced_family(task_family, ql)
    if family == "NONE":
        completeness, unresolved_semantics = "NOT_ENFORCED", ()
    elif family == "SIMPLE_AGGREGATE" and (not target or evidence.get("aggregate_rejection")):
        completeness = "INCOMPLETE"
        if evidence.get("aggregate_rejection"):
            rejection = evidence["aggregate_rejection"]
            reason = rejection.get("reason", "simple_aggregate_not_numeric")
            unresolved_semantics = (f"aggregate_rejection:{reason}",)
        else:
            unresolved_semantics = ("simple_aggregate_request_not_resolved_by_contract",)
    elif task_family == "GENERAL_EXPLORATION":
        completeness = "INCOMPLETE"
        unresolved_semantics = (f"{family.lower()}_request_not_resolved_by_contract",)
    elif unresolved:
        completeness = "INCOMPLETE"
        unresolved_semantics = tuple(f"unresolved_role:{r}" for r in unresolved)
    else:
        completeness, unresolved_semantics = "COMPLETE", ()
        evidence["completeness_basis"] = "resolved_roles_and_filter_integrity_guards"
        evidence["intent_ledger_scope"] = "filter_clauses_only_not_full_token_coverage"

    return CanonicalQuestionContract(
        question_text=q_clean,
        task_family=task_family,
        claim_type=claim_type,
        estimand=estimand,
        target_column=target,
        explanatory_columns=tuple(explanatory),
        grouping_columns=tuple(grouping),
        time_column=time_col,
        time_range=evidence.get("paired_time_range"),
        filter_conditions=tuple(filter_conditions),
        requested_aggregation=req_agg,
        comparison_values=tuple(comparison_values),
        ranking_direction=ranking_direction,
        ranking_limit=ranking_limit,
        causal_language=causal_language,
        requested_claim=requested_claim,
        supported_claim=supported_claim,
        claim_ceiling=claim_ceiling,
        resolution_confidence=res_conf,
        unresolved_roles=tuple(unresolved),
        resolution_evidence=evidence,
        family=family,
        completeness=completeness,
        unresolved_semantics=unresolved_semantics,
        intent_ledger=tuple(intent_ledger),
    )
