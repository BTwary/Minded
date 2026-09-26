"""
Deterministic natural-language intent parser for AA-OS.

The parser extracts analytical operation, roles, explicit ranking direction,
aggregation hints, temporal phrases, and schema-grounded column hints. It is
deliberately conservative: when two physical columns are equally plausible it
returns no winner rather than guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


_CORRELATION_KEYWORD_RE = re.compile(
    r"\b(correlat\w*|relat\w*|coupl\w*|depend\w*|associat\w*|"
    r"impact\w*|affect\w*|link\w*|influenc\w*|connect\w*)\b"
    r"|\b(tied to|moves? together|effect of)\b",
    re.IGNORECASE,
)

_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "revenue": (
        "revenue", "sales", "net sales", "gross sales", "turnover", "income",
        "money", "proceeds", "takings", "amount", "total amount", "order value",
        "gmv",
    ),
    "aov": (
        "average order value", "aov", "average basket value", "average basket",
    ),
    "quantity": (
        "quantity", "qty", "units", "unit count", "items sold", "volume sold",
    ),
    "cost": (
        "cost", "costs", "expense", "expenses", "spend", "expenditure",
    ),
    "profit": (
        "profit", "profits", "margin", "gross margin", "net margin", "earnings",
    ),
    "discount": (
        "discount", "discount rate", "discount percentage", "discount percent",
        "markdown", "rebate",
    ),
    "price": (
        "price", "unit price", "selling price", "list price",
    ),
    "conversion": (
        "conversion", "conversion rate", "conversion percentage",
    ),
    "delay": (
        "delay", "delivery delay", "latency", "wait time",
    ),
    "churn": (
        "churn", "churn rate", "attrition", "retention", "cancellation",
    ),
}

_DIMENSION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "region": ("region", "regions", "territory", "market", "market area", "zone"),
    "segment": ("segment", "segments", "customer segment", "tier", "plan tier"),
    "category": ("category", "categories", "product category", "product categories"),
    "channel": ("channel", "channels", "acquisition channel", "source", "medium"),
    "product": ("product", "products", "sku", "item"),
    "customer": ("customer", "customers", "client", "clients", "account", "accounts", "user", "users"),
    "order": ("order", "orders", "transaction", "transactions"),
}

_RELATION_BOUNDARY = (
    r"(?=,|;|\bwhile\b|\bwhereas\b|\bcontrolling for\b|\bgiven\b|"
    r"\bafter accounting for\b|\bholding\b|\?|$)"
)

_RANK_HIGH_RE = re.compile(r"\b(highest|most|largest|biggest|top)\b", re.I)
_RANK_LOW_RE = re.compile(r"\b(lowest|least|smallest|bottom|fewest)\b", re.I)
_RANK_AMBIGUOUS_RE = re.compile(r"\b(best|worst)\b", re.I)

_CAUSAL_RE = re.compile(
    r"\b(caus(?:e|ed|es|ing|al|ally)|causal|treatment effect|randomi[sz]ed|"
    r"intervention|counterfactual|what happens if|would .+ if)\b",
    re.I,
)
_PREDICTION_RE = re.compile(
    r"\b(predict\w*|prediction\w*|likely|probabilit\w*|odds|chance|risk|"
    r"at risk|will .*(?:churn|cancel|convert|buy|leave))\b",
    re.I,
)
_FORECAST_RE = re.compile(
    r"\b(forecast\w*|project\w*|projection\w*|outlook|trajectory|"
    r"next month|next quarter|next year|next week|future|going forward|"
    r"this month|this quarter|this year|will .*\b(?:increase|decrease|grow|"
    r"decline|rise|fall)\b)\b",
    re.I,
)
_DIAGNOSTIC_RE = re.compile(
    r"\b(why|driver\w*|root cause|explain\w*|what caused|what drove|"
    r"what is behind|reason behind|account for|what happened to|what changed)\b",
    re.I,
)
_SEGMENTATION_RE = re.compile(
    r"\b(cluster\w*|persona\w*|discover groups?|find groups?|"
    r"identify groups?|build groups?|create groups?)\b",
    re.I,
)
_DATA_QUALITY_RE = re.compile(
    r"\b(data quality|clean|cleaning|missing|duplicate|invalid|malformed|"
    r"outlier\w*|anomal\w*|inconsistent|corrupt|repair|remediat\w*)\b",
    re.I,
)
_RECONCILIATION_RE = re.compile(
    r"\b(discrep\w*|reconcil\w*|formula\w*|mathematical|mismatch\w*|"
    r"tie(?:s|d)? out|add(?:s|ed)? up|reconcile)\b",
    re.I,
)
_GOVERNANCE_RE = re.compile(
    r"\b(privacy|pii|gdpr|ccpa|ethical|ethics|legal|compliance|fairness|"
    r"disparate|harmful|harm|data protection)\b",
    re.I,
)
_PRESCRIPTIVE_RE = re.compile(
    r"\b(what should we do|what should i do|recommend\w*|recommendation\w*|"
    r"action\w*|optimi[sz]\w*|best intervention|how should we|what action)\b",
    re.I,
)
_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|difference|different|higher|lower|greater|less|"
    r"more|fewer|versus|vs\.?|between|vary|varies|outperform\w*|underperform\w*|"
    r"best|worst)\b",
    re.I,
)
_DESCRIPTIVE_RE = re.compile(
    r"\b(total|sum|average|mean|median|count|number of|how many|how much|"
    r"typical|usual|distribution|summary|summar(?:y|ize)|describe|show|list)\b",
    re.I,
)
_CHURN_RE = re.compile(
    r"\b(churn\w*|attrition\w*|cancel\w*|retention\w*|dropoff\w*|"
    r"at risk of leaving)\b",
    re.I,
)


def _normalize_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    p = _normalize_phrase(phrase)
    if not p:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", _normalize_phrase(text)))


def _schema_phrase_matches(phrase: str, columns: Sequence[str]) -> List[str]:
    pn = _normalize_phrase(phrase)
    if not pn:
        return []
    exact = [str(c) for c in columns if _normalize_phrase(str(c)) == pn]
    if exact:
        return exact
    return [
        str(c) for c in columns
        if re.search(rf"(?<!\w){re.escape(pn)}(?!\w)", _normalize_phrase(str(c)))
    ]


def _alias_candidates(phrase: str, columns: Sequence[str], aliases: Dict[str, Tuple[str, ...]]) -> List[str]:
    pn = _normalize_phrase(phrase)
    concepts = [
        concept for concept, values in aliases.items()
        if any(_normalize_phrase(v) == pn for v in values)
    ]
    if not concepts:
        return []

    candidates: List[str] = []
    for col in columns:
        cn = _normalize_phrase(str(col))
        parts = set(cn.split())
        for concept in concepts:
            synonym_set = {_normalize_phrase(v) for v in aliases[concept]} | {concept}
            if cn in synonym_set or parts.intersection(synonym_set):
                candidates.append(str(col))
                break
    return sorted(set(candidates))


def _resolve_phrase(phrase: str, columns: Sequence[str], aliases: Dict[str, Tuple[str, ...]]) -> Optional[str]:
    exact = _schema_phrase_matches(phrase, columns)
    if len(exact) == 1:
        return exact[0]
    alias = _alias_candidates(phrase, columns, aliases)
    if len(alias) == 1:
        return alias[0]
    return None


def _extract_time_hint(question: str) -> Optional[str]:
    patterns = (
        r"\b(?:today|yesterday|this week|last week|next week|this month|last month|next month|"
        r"this quarter|last quarter|next quarter|this year|last year|next year)\b",
        r"\b(?:q[1-4](?:\s+of\s+20\d{2})?)\b",
        r"\b(?:trailing|last)\s+\d+\s+(?:days?|weeks?|months?|quarters?|years?)\b",
        r"\b(?:since|from)\s+[a-z]+(?:\s+20\d{2})?(?:\s+(?:to|through|until)\s+[a-z]+(?:\s+20\d{2})?)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question, re.I)
        if match:
            return match.group(0)
    return None


@dataclass
class InvestigationIntent:
    """Canonical deterministic interpretation of a business question."""
    raw_question: str
    intent_type: str
    comparison_type: str = "GENERAL_INVESTIGATION"
    target_metric_hint: Optional[str] = None
    dimension_hint: Optional[str] = None
    comparison_period_hint: Optional[str] = None
    requested_operations: List[str] = field(default_factory=list)
    candidate_dimensions: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    business_objective: str = ""
    uncertainty_score: float = 0.0
    assumptions: List[str] = field(default_factory=list)
    direction_hint: str = "unspecified"
    ranking_direction: Optional[str] = None
    aggregation_hint: Optional[str] = None
    time_horizon_hint: Optional[str] = None
    relation_target_phrase: Optional[str] = None
    relation_predictor_phrases: List[str] = field(default_factory=list)


class IntentEngine:
    """Deterministic parser; schema evidence wins over linguistic guesses."""

    @staticmethod
    def _relation_sides(question: str) -> Tuple[Optional[str], List[str]]:
        patterns = (
            (r"\b(?:is|are|do|does|can|whether)\s+(.+?)\s+"
             r"(?:affect\w*|influenc\w*|impact\w*)\s+(.+?)" + _RELATION_BOUNDARY, "object"),
            (r"\b(?:is|are|do|does|can|whether)\s+(.+?)\s+"
             r"(?:depend\w*)\s+on\s+(.+?)" + _RELATION_BOUNDARY, "subject"),
            (r"\b(?:correlation|relationship|association)\s+between\s+(.+?)\s+and\s+(.+?)" + _RELATION_BOUNDARY, "symmetric"),
            (r"\b(.+?)\s+(?:associated|correlated|related|linked|connected)\s+with\s+(.+?)" + _RELATION_BOUNDARY, "symmetric"),
            (r"\b(?:effect|impact)\s+of\s+(.+?)\s+on\s+(.+?)" + _RELATION_BOUNDARY, "object"),
        )
        for pattern, mode in patterns:
            match = re.search(pattern, question, re.I)
            if not match:
                continue
            left, right = match.group(1).strip(" ,?"), match.group(2).strip(" ,?")
            if mode == "object":
                return right, [left]
            if mode == "subject":
                return left, [right]
            if mode == "symmetric":
                return None, [left, right]
        return None, []

    @staticmethod
    def parse_intent(question: str, available_columns: Optional[List[str]] = None) -> InvestigationIntent:
        q = " ".join((question or "").split())
        q_lower = q.lower()
        words = re.findall(r"\b[a-zA-Z0-9_]+\b", q_lower)
        columns = [str(c) for c in (available_columns or [])]

        relation_target_phrase, relation_predictor_phrases = IntentEngine._relation_sides(q)
        relation_question = bool(_CORRELATION_KEYWORD_RE.search(q_lower) or relation_predictor_phrases)

        causal = bool(_CAUSAL_RE.search(q_lower))
        prediction = bool(_PREDICTION_RE.search(q_lower))
        forecast = bool(_FORECAST_RE.search(q_lower)) and not prediction
        ranking_high = bool(_RANK_HIGH_RE.search(q_lower))
        ranking_low = bool(_RANK_LOW_RE.search(q_lower))
        ranking_ambiguous = bool(_RANK_AMBIGUOUS_RE.search(q_lower))
        ranking_question = bool(
            ranking_high or ranking_low or ranking_ambiguous
            or re.search(r"\b(rank|ranked|ranking|top|bottom)\b", q_lower, re.I)
        )
        comparison_question = bool(_COMPARISON_RE.search(q_lower))

        if _GOVERNANCE_RE.search(q_lower):
            intent_type, comparison_type, ops = "GENERAL", "GENERAL_INVESTIGATION", ["GOVERNANCE"]
        elif _RECONCILIATION_RE.search(q_lower):
            intent_type, comparison_type, ops = "GENERAL", "GENERAL_INVESTIGATION", ["RECONCILIATION"]
        elif _DATA_QUALITY_RE.search(q_lower) and re.search(
            r"\b(check|find|identify|clean|repair|quality|missing|duplicate|invalid|outlier|anomal)\w*\b",
            q_lower,
            re.I,
        ):
            intent_type, comparison_type, ops = "GENERAL", "GENERAL_INVESTIGATION", ["DATA_QUALITY"]
        elif _PRESCRIPTIVE_RE.search(q_lower):
            intent_type, comparison_type, ops = "GENERAL", "GENERAL_INVESTIGATION", ["PRESCRIPTIVE"]
        elif causal:
            intent_type, comparison_type, ops = "CAUSAL", "ANOMALY_ROOT_CAUSE", ["CAUSAL_EFFECT"]
        elif prediction:
            intent_type, comparison_type, ops = "PREDICTION", "GENERAL_INVESTIGATION", ["RISK_OR_OUTCOME_PREDICTION"]
        elif forecast:
            intent_type, comparison_type, ops = "FORECAST", "PERIOD_OVER_PERIOD", ["TIME_SERIES_FORECAST", "GROWTH_RATE"]
        elif relation_question and not causal:
            intent_type, comparison_type, ops = "CORRELATION", "CORRELATION_SEARCH", ["BIVARIATE_CORRELATION", "PARTIAL_CORRELATION", "CRAMERS_V"]
        elif _SEGMENTATION_RE.search(q_lower) and not ranking_question and not comparison_question:
            intent_type, comparison_type, ops = "SEGMENTATION", "SEGMENT_CONTRAST", ["SEGMENT_DECOMPOSITION", "STABILITY_CHECK"]
        elif _DIAGNOSTIC_RE.search(q_lower):
            intent_type, comparison_type, ops = "ROOT_CAUSE", "ANOMALY_ROOT_CAUSE", ["CONCENTRATION", "VARIANCE_DECOMPOSITION", "ADVERSARIAL_SIMPSON"]
        elif ranking_question or comparison_question:
            intent_type, comparison_type = "PERFORMANCE", "SEGMENT_CONTRAST"
            ops = ["RANKING"] if ranking_question else ["GROUP_COMPARISON"]
        elif _CHURN_RE.search(q_lower):
            intent_type, comparison_type, ops = "CHURN", "SEGMENT_CONTRAST", ["CRUDE_CHURN_RATE", "EXPOSURE_ADJUSTED_RATE", "STRATIFIED_CHURN_CHECK"]
        elif _DESCRIPTIVE_RE.search(q_lower) or re.search(r"\b(by|per|across|within|for each)\b", q_lower):
            intent_type, comparison_type, ops = "GENERAL", "GENERAL_INVESTIGATION", ["DESCRIPTIVE_SUMMARY", "DISTRIBUTION_SCAN"]
        else:
            intent_type, comparison_type, ops = "GENERAL", "GENERAL_INVESTIGATION", ["DESCRIPTIVE_SUMMARY", "DISTRIBUTION_SCAN"]

        target_metric_hint: Optional[str] = None
        dimension_hint: Optional[str] = None
        candidate_dims: List[str] = []

        if relation_target_phrase:
            target_metric_hint = _resolve_phrase(relation_target_phrase, columns, _METRIC_ALIASES)

        explicit_targets = [c for c in columns if _contains_phrase(q, c)]

        if not target_metric_hint:
            metric_mentions: List[Tuple[int, str]] = []
            for _, aliases in _METRIC_ALIASES.items():
                for alias in aliases:
                    if _contains_phrase(q, alias):
                        candidates = _alias_candidates(alias, columns, _METRIC_ALIASES)
                        if len(candidates) == 1:
                            metric_mentions.append((q_lower.find(_normalize_phrase(alias)), candidates[0]))
            metric_mentions.sort(key=lambda x: x[0])
            unique = list(dict.fromkeys(c for _, c in metric_mentions))
            if relation_question and len(unique) >= 2:
                target_metric_hint = None
            elif len(unique) == 1:
                target_metric_hint = unique[0]

        if not target_metric_hint and len(explicit_targets) == 1:
            candidate = explicit_targets[0]
            if not re.search(r"(^|_)(id|key|code|uuid)$", candidate, re.I):
                target_metric_hint = candidate

        group_phrases: List[str] = []
        for match in re.finditer(
            r"\b(?:by|per|across|within|for each)\s+(.+?)(?=\s+(?:after|before|during|since|between|where|when)\b|\?|$)",
            q,
            re.I,
        ):
            group_phrases.append(match.group(1).strip(" ,"))

        top_by = re.search(
            r"\b(?:rank|ranked|top|bottom)\s+(?:\d+\s+)?(?:the\s+)?(.+?)\s+by\s+(.+?)(?:\?|$)",
            q,
            re.I,
        )
        if top_by:
            group_phrases.insert(0, top_by.group(1).strip(" ,"))
            if not target_metric_hint:
                target_metric_hint = _resolve_phrase(top_by.group(2).strip(" ,"), columns, _METRIC_ALIASES)

        which_group = re.search(
            r"\b(?:which|what)\s+(.+?)\s+(?:has|have|shows?|with)\s+(?:the\s+)?"
            r"(?:highest|lowest|largest|smallest|most|least|top|bottom|best|worst)\b",
            q,
            re.I,
        )
        if which_group:
            group_phrases.insert(0, which_group.group(1).strip(" ,"))

        for phrase in group_phrases:
            resolved = _resolve_phrase(phrase, columns, _DIMENSION_ALIASES)
            if resolved and resolved != target_metric_hint:
                dimension_hint = resolved
                break
            matches = _schema_phrase_matches(phrase, columns)
            if len(matches) == 1 and matches[0] != target_metric_hint:
                dimension_hint = matches[0]
                break

        if not dimension_hint:
            dim_mentions: List[str] = []
            for values in _DIMENSION_ALIASES.values():
                for alias in values:
                    if _contains_phrase(q, alias):
                        matches = _alias_candidates(alias, columns, _DIMENSION_ALIASES)
                        if len(matches) == 1:
                            dim_mentions.append(matches[0])
            dim_unique = list(dict.fromkeys(dim_mentions))
            if len(dim_unique) == 1 and dim_unique[0] != target_metric_hint:
                dimension_hint = dim_unique[0]

        if dimension_hint:
            candidate_dims = [dimension_hint]

        resolved_predictors = []
        for phrase in relation_predictor_phrases:
            resolved = _resolve_phrase(phrase, columns, _METRIC_ALIASES)
            resolved_predictors.append(resolved or phrase)
        relation_predictor_phrases = resolved_predictors

        ranking_direction = None
        if ranking_high and not ranking_low and not ranking_ambiguous:
            ranking_direction = "DESC"
        elif ranking_low and not ranking_high and not ranking_ambiguous:
            ranking_direction = "ASC"

        aggregation_hint = None
        if re.search(r"\b(average|mean|avg)\b", q_lower):
            aggregation_hint = "mean"
        elif re.search(r"\bmedian\b", q_lower):
            aggregation_hint = "median"
        elif re.search(r"\b(total|sum)\b", q_lower):
            aggregation_hint = "sum"
        elif re.search(r"\b(how many|number of|count)\b", q_lower):
            aggregation_hint = "count"
        elif re.search(r"\b(rate|percentage|percent|proportion)\b", q_lower):
            aggregation_hint = "rate"

        decrease = bool(re.search(
            r"\b(drop\w*|fell|fall\w*|decreas\w*|declin\w*|down|shrink\w*|"
            r"lower\w*|reduc\w*|dip\w*|worsen\w*|miss\w*)\b", q_lower
        ))
        increase = bool(re.search(
            r"\b(increas\w*|rise\w*|rose|grow\w*|surge\w*|spike\w*|"
            r"gain\w*|higher|improv\w*|jump\w*)\b", q_lower
        ))
        direction_hint = (
            "decrease" if decrease and not increase
            else "increase" if increase and not decrease
            else "unspecified"
        )

        time_horizon_hint = _extract_time_hint(q_lower)

        constraints: Dict[str, Any] = {}
        if relation_predictor_phrases:
            constraints["relation_predictors"] = list(relation_predictor_phrases)
        if relation_target_phrase:
            constraints["relation_target_phrase"] = relation_target_phrase
        if time_horizon_hint:
            constraints["time_horizon"] = time_horizon_hint
        if ranking_ambiguous:
            constraints["ranking_direction_requires_metric_polarity"] = True

        uncertainty = (
            0.10 if relation_question and len(relation_predictor_phrases) >= 2
            else 0.25 if target_metric_hint and dimension_hint
            else 0.35 if target_metric_hint or dimension_hint
            else 0.45 if ranking_question or comparison_question
            else 0.70
        )

        objective = (
            f"Investigate {intent_type.lower().replace('_', ' ')} for "
            f"{target_metric_hint or 'primary metrics'} across "
            f"{dimension_hint or 'key business partitions'}."
        )

        return InvestigationIntent(
            raw_question=q,
            intent_type=intent_type,
            comparison_type=comparison_type,
            target_metric_hint=target_metric_hint,
            dimension_hint=dimension_hint,
            comparison_period_hint=time_horizon_hint,
            requested_operations=ops,
            candidate_dimensions=candidate_dims,
            keywords=words,
            constraints=constraints,
            business_objective=objective,
            uncertainty_score=uncertainty,
            assumptions=[
                "Deterministic execution across available dataset columns.",
                "Schema-grounded aliases are accepted only when uniquely resolvable.",
                "Unresolved or ambiguous variables are not replaced by unrelated columns.",
            ],
            direction_hint=direction_hint,
            ranking_direction=ranking_direction,
            aggregation_hint=aggregation_hint,
            time_horizon_hint=time_horizon_hint,
            relation_target_phrase=relation_target_phrase,
            relation_predictor_phrases=list(relation_predictor_phrases),
        )
