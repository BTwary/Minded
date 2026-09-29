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
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


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
    requested_aggregation: Optional[str] = None  # MEAN, RATE, COUNT, SUM, MEDIAN
    comparison_operator: Optional[str] = None
    comparison_values: Tuple[str, ...] = ()
    direction: Optional[str] = None
    ranking_direction: Optional[str] = None      # "DESC", "ASC"
    causal_language: bool = False
    requested_claim: str = "ASSOCIATION"         # CAUSAL, ASSOCIATION, OBSERVATION, PREDICTION
    supported_claim: str = "ASSOCIATION"         # OBSERVATION, ASSOCIATION, CAUSAL_INFERENCE, PREDICTION
    claim_ceiling: str = "ASSOCIATION"           # OBSERVATION, ASSOCIATION, CAUSAL_INFERENCE, PREDICTION
    resolution_confidence: float = 1.0
    unresolved_roles: Tuple[str, ...] = ()
    resolution_evidence: Dict[str, Any] = field(default_factory=dict)

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
            "causal_language": self.causal_language,
            "requested_claim": self.requested_claim,
            "supported_claim": self.supported_claim,
            "claim_ceiling": self.claim_ceiling,
            "resolution_confidence": float(self.resolution_confidence),
            "unresolved_roles": list(self.unresolved_roles),
            "resolution_evidence": dict(self.resolution_evidence),
        }


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
    "average order value": ["aov", "order_value", "total_amount"],
    "aov": ["aov", "order_value", "total_amount"],
    "order value": ["aov", "order_value", "total_amount", "amount"],
    "order value after discounts": ["aov", "order_value", "total_amount"],
    "customer segment": ["customer_segment", "segment"],
    "customer segments": ["customer_segment", "segment"],
    "segment": ["customer_segment", "segment"],
    "segments": ["customer_segment", "segment"],
    "churn": ["churned", "churn", "is_churned"],
    "churned": ["churned", "churn", "is_churned"],
    "churn rate": ["churned", "churn", "is_churned"],
    "plan tier": ["plan_tier", "plan"],
    "subscription plan": ["plan_tier", "plan"],
    "subscription plans": ["plan_tier", "plan"],
    "tenure": ["tenure_days", "tenure"],
    "support tickets": ["support_tickets", "tickets"],
}

# Stem prefixes for invariant lexical root matching
STEM_PREFIXES = {
    "surviv": "survived",
    "passeng": "passengers",
    "orbit": "orbital_period",
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
    cleaned = re.sub(r"^(?:average|avg|mean|total|sum|median|count\s+of|number\s+of|rate\s+of|the)\s+", "", phrase.strip(), flags=re.I)
    cleaned = re.sub(r"\s+(?:after|before|by|for|during|in)\s+.*$", "", cleaned.strip(), flags=re.I)
    cleaned = re.sub(r"\s+(?:amount|level|rate|score|value|count)$", "", cleaned.strip(), flags=re.I)
    return cleaned.strip()


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
    aliases = APPROVED_SEMANTIC_ALIASES.get(p_norm, [])
    for target_cand in aliases:
        for c in available_cols:
            if c.lower() == target_cand.lower() or normalize_token(c) == normalize_token(target_cand):
                return c

    # 4. Multi-word phrase containment (e.g. 'flipper length' in 'flipper_length_mm')
    for c in available_cols:
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
            or pd.api.types.is_categorical_dtype(s)
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


def extract_aggregation(question: str, target: Optional[str] = None) -> Optional[str]:
    """Determine the canonical requested aggregation from the question text."""
    ql = question.lower()
    if re.search(r"\b(survival\s+rate|churn\s+rate|rate|rates|proportion|percentage|fraction|share)\b", ql):
        return "RATE"
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
        return "COUNT"
    if re.search(r"\b(grown|grown\s+over\s+time|increased\s+over\s+time)\b", ql):
        if target:
            t_low = target.lower()
            if any(k in t_low for k in INTENSIVE_METRIC_KEYWORDS):
                return "MEAN"
        return "SUM"
    if target:
        t_low = target.lower()
        if any(k in t_low for k in INTENSIVE_METRIC_KEYWORDS):
            return "MEAN"
    return None


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
    causal_language = False
    evidence: Dict[str, Any] = {}

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
        pred_col = resolve_phrase_to_column(pred_phrase, cols)
        target_col = resolve_phrase_to_column(target_phrase, cols)
        if not target_col and semantic is not None:
            sem_tgt = getattr(semantic, "target_metric_col", None) or getattr(semantic, "churn_event_col", None)
            if sem_tgt and sem_tgt in cols:
                target_col = sem_tgt
        if pred_col and target_col:
            target = target_col
            explanatory = [pred_col]
            task_family = "CAUSAL_REQUEST"
            causal_language = True
            requested_claim = "CAUSAL"
            supported_claim = "ASSOCIATION"
            claim_ceiling = "ASSOCIATION"
            claim_type = "ASSOCIATION"
            estimand = "correlation" if pd.api.types.is_numeric_dtype(df[target_col]) or df[target_col].nunique() == 2 else "group_difference"
            evidence["relation"] = {"type": "directional", "predictor": pred_col, "target": target_col}
        elif target_col and not pred_col:
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
        pred_col = resolve_phrase_to_column(pred_phrase, cols)
        target_col = resolve_phrase_to_column(target_phrase, cols)
        if pred_col and target_col:
            target = target_col
            explanatory = [pred_col]
            task_family = "CAUSAL_REQUEST"
            causal_language = True
            requested_claim = "CAUSAL"
            supported_claim = "ASSOCIATION"
            claim_ceiling = "ASSOCIATION"
            claim_type = "ASSOCIATION"
            estimand = "correlation"
            evidence["relation"] = {"type": "effect_of", "predictor": pred_col, "target": target_col}

    # Symmetric: "Is survival correlated with fare?" / "Is tip correlated with total bill?"
    corr_match = re.search(
        r"(?:is|are|was|were)\s+(.+?)\s+(?:correlated|associated|related)\s+with\s+(.+?)(?:\?|$)",
        ql,
    )
    if corr_match and not target:
        var1_phrase = corr_match.group(1).strip()
        var2_phrase = corr_match.group(2).strip()
        var1 = resolve_phrase_to_column(var1_phrase, cols)
        var2 = resolve_phrase_to_column(var2_phrase, cols)
        if var1 and var2:
            target = var1
            explanatory = [var2]
            task_family = "ASSOCIATION"
            requested_claim = "ASSOCIATION"
            supported_claim = "ASSOCIATION"
            claim_ceiling = "ASSOCIATION"
            claim_type = "ASSOCIATION"
            estimand = "correlation"
            evidence["relation"] = {"type": "symmetric_correlation", "var1": var1, "var2": var2}

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

    # 5. Ranking: "Which X has/had the highest/lowest Y?"
    rank_match = re.search(
        r"\bwhich\s+([a-zA-Z0-9_\s]+?)\s+(?:has|had|have|shows?|with|discovered)\s+(?:the\s+)?(highest|lowest|most|least|best|worst|largest|smallest)\s+(.+?)(?:\?|$)",
        ql,
    )
    if rank_match and not target:
        dim_phrase = rank_match.group(1).strip()
        direction_word = rank_match.group(2).strip().lower()
        metric_phrase = rank_match.group(3).strip()

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
            evidence["ranking"] = {"group": dim_col, "target": target, "direction": ranking_direction}

    # 6. Group Comparison: "Do smokers tip more than non-smokers?" / "Do male penguins weigh more than female penguins?" / "Do credit card payers tip more than cash payers?"
    do_compare_match = re.search(
        r"(?:do|does|did|is|are)\s+(.+?)\s+([a-zA-Z0-9_]+)\s+(?:more|less|higher|lower)\s+than\s+(.+?)(?:\?|$)",
        ql,
    )
    if do_compare_match and not target and not re.search(r"\bwhy\b", ql):
        side_a = do_compare_match.group(1).strip()
        action_metric = do_compare_match.group(2).strip()
        side_b = do_compare_match.group(3).strip()

        cand_target = resolve_phrase_to_column(action_metric, cols)
        grp_col_a = resolve_phrase_to_column(side_a, cols)
        grp_col_b = resolve_phrase_to_column(side_b, cols)
        cat_group, cat_vals = scan_observed_category_values(f"{side_a} {side_b}", df)

        chosen_group = grp_col_a if (grp_col_a and grp_col_a == grp_col_b) else (cat_group or grp_col_a or grp_col_b)
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

    # 8. Diagnostic / Why: "Why did third class passengers have lower survival than first class?" / "Why are Fair cut diamonds priced higher than Ideal cut on average?" / "Why do cars from the usa have lower mpg?"
    if re.search(r"\bwhy\b", ql) and not target:
        causal_language = True
        task_family = "ROOT_CAUSE"
        requested_claim = "CAUSAL"
        supported_claim = "ASSOCIATION"
        claim_ceiling = "ASSOCIATION"
        claim_type = "ASSOCIATION"
        estimand = "root_cause"
        for phrase in ["survival", "price", "mpg", "fare", "tip", "body mass", "weight"]:
            if phrase in ql:
                c = resolve_phrase_to_column(phrase, cols)
                if c:
                    target = c
                    break
        g_cand, g_vals = scan_observed_category_values(q_clean, df)
        if g_cand and g_cand != target:
            grouping = [g_cand]
            comparison_values = g_vals
        if not grouping:
            for dim_phrase in ["passenger class", "class", "cut", "origin", "day", "clarity", "species"]:
                if dim_phrase in ql:
                    c = resolve_phrase_to_column(dim_phrase, cols)
                    if c and c != target:
                        grouping = [c]
                        break

    # 9. Descriptive Breakdown: "What is the survival rate by passenger class?" / "What is the average tip by day?"
    desc_by_match = re.search(r"(?:what\s+is|what\s+are|calculate|show)\s+(?:the\s+)?(.+?)\s+by\s+(.+?)(?:\?|$)", ql)
    if desc_by_match and not target:
        m_phrase = desc_by_match.group(1).strip()
        g_phrase = desc_by_match.group(2).strip()
        m_col = resolve_phrase_to_column(m_phrase, cols)
        g_col = resolve_phrase_to_column(g_phrase, cols)
        if m_col and g_col:
            target = m_col
            grouping = [g_col]
            task_family = "DESCRIPTIVE"
            claim_type = "OBSERVATION"
            requested_claim = "OBSERVATION"
            supported_claim = "OBSERVATION"
            claim_ceiling = "OBSERVATION"
            estimand = "mean" if "average" in m_phrase or "avg" in m_phrase else "rate" if "rate" in m_phrase else "sum"

    # 10. Prediction / Risk: "Which customers are at highest risk of churn next month?"
    if (re.search(r"\b(risk|probability|likely|likelihood)\s+of\s+([a-zA-Z0-9_\s]+)", ql) or
        re.search(r"\b(which|who)\s+(?:customers?|users?|accounts?)\s+are\s+at\s+(?:highest|lowest|top)\s+risk\b", ql)) and not target:
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

    # Determine requested aggregation
    req_agg = extract_aggregation(q_clean, target)
    if estimand in ("correlation", "multi_correlation") and req_agg not in ("CORRELATION", "MEAN", None):
        req_agg = "CORRELATION"

    # Clean up unresolved roles
    unresolved: List[str] = []
    if task_family in ("ASSOCIATION", "CAUSAL_REQUEST") and not target:
        unresolved.append("target")
    if task_family in ("ASSOCIATION", "CAUSAL_REQUEST") and not explanatory:
        unresolved.append("explanatory")
    if task_family in ("RANKING", "GROUP_COMPARISON", "ROOT_CAUSE") and not grouping:
        unresolved.append("grouping")

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

    return CanonicalQuestionContract(
        question_text=q_clean,
        task_family=task_family,
        claim_type=claim_type,
        estimand=estimand,
        target_column=target,
        explanatory_columns=tuple(explanatory),
        grouping_columns=tuple(grouping),
        time_column=time_col,
        filter_conditions=(),
        requested_aggregation=req_agg,
        comparison_values=tuple(comparison_values),
        ranking_direction=ranking_direction,
        causal_language=causal_language,
        requested_claim=requested_claim,
        supported_claim=supported_claim,
        claim_ceiling=claim_ceiling,
        resolution_confidence=res_conf,
        unresolved_roles=tuple(unresolved),
        resolution_evidence=evidence,
    )
