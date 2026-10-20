"""Type-aware analytical planning and fail-closed conditional branch resolution.

This module produces plans, not execution authority. Conditions are a small,
validated data language (never Python/eval) so a runtime can safely resolve
branches from measured evidence and prerequisite statuses.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence
import re

PLAN_VERSION = "2.1"

TASK_TYPES: dict[str, dict[str, Any]] = {
    "PROFILE": {"family": "descriptive", "min_roles": [], "claim_ceiling": "descriptive"},
    "AGGREGATION": {"family": "descriptive", "min_roles": ["measure"], "claim_ceiling": "descriptive"},
    "DISTRIBUTION": {"family": "distribution", "min_roles": ["measure"], "claim_ceiling": "descriptive"},
    "MISSINGNESS": {"family": "data_quality", "min_roles": [], "claim_ceiling": "data_quality"},
    "COMPARISON": {"family": "group_comparison", "min_roles": ["outcome", "group"], "claim_ceiling": "associational"},
    "ASSOCIATION": {"family": "association", "min_roles": ["x", "y"], "claim_ceiling": "associational"},
    "TEMPORAL_TREND": {"family": "temporal", "min_roles": ["time", "measure"], "claim_ceiling": "descriptive_temporal"},
    "FORECAST": {"family": "forecasting", "min_roles": ["time", "measure"], "claim_ceiling": "predictive"},
    "PREDICTION": {"family": "prediction", "min_roles": ["features", "target"], "claim_ceiling": "predictive"},
    "CAUSAL_EFFECT": {"family": "causal", "min_roles": ["treatment", "outcome", "design"], "claim_ceiling": "causal_only_if_identified"},
    "SEGMENTATION": {"family": "segmentation", "min_roles": ["features"], "claim_ceiling": "descriptive_segments"},
    "RANKING": {"family": "ranking", "min_roles": ["entity", "measure"], "claim_ceiling": "descriptive_ranking"},
    "CONTRIBUTION": {"family": "decomposition", "min_roles": ["measure", "dimension", "comparison_window"], "claim_ceiling": "decomposition"},
    "RECONCILIATION": {"family": "reconciliation", "min_roles": ["formula_or_components"], "claim_ceiling": "reconciliation"},
    "ANOMALY": {"family": "anomaly", "min_roles": ["measure_or_features"], "claim_ceiling": "anomaly_candidates"},
    "SURVIVAL": {"family": "time_to_event", "min_roles": ["duration", "event"], "claim_ceiling": "survival_association"},
    "TEXT_ANALYSIS": {"family": "text", "min_roles": ["text"], "claim_ceiling": "descriptive_text"},
    "GOVERNANCE": {"family": "governance", "min_roles": [], "claim_ceiling": "risk_scan_not_legal_certification"},
    "DECISION_ANALYSIS": {"family": "decision", "min_roles": ["objective", "constraints"], "claim_ceiling": "conditional_decision_support"},
}

# Explicit phrase routing wins over broad legacy task classification.
_TASK_PATTERNS: list[tuple[str, str]] = [
    ("MISSINGNESS", r"\b(missingness|missing values?|percentage of .* missing|null rate|completeness)\b"),
    ("DISTRIBUTION", r"\b(distribution|skewness|kurtosis|percentiles?|quartiles?|histogram|spread|variance|standard deviation)\b"),
    ("CAUSAL_EFFECT", r"\b(causal|causation|treatment effect|effect of .* on|impact of .* on|did .* cause)\b"),
    ("SURVIVAL", r"\b(survival|time[- ]to[- ]event|hazard ratio|kaplan[- ]meier)\b"),
    ("FORECAST", r"\b(forecast|forecasting|predict future|next \d+ periods?)\b"),
    ("TEXT_ANALYSIS", r"\b(sentiment|themes? in|topic model|keywords? in|classify text|text analysis|reviews? say|common phrases?)\b"),
    ("PREDICTION", r"\b(predict|prediction|classify|classification|probability that|risk score)\b"),
    ("SEGMENTATION", r"\b(segment customers|customer segments? to discover|cluster|clustering|natural groups?)\b"),
    ("RECONCILIATION", r"\b(reconcile|reconciliation|does .* add up|discrepanc|formula check|check whether .* (?:equals?|matches?|adds? up to)|verify .* (?:equals?|matches?|adds? up to)|validate .* (?:equals?|matches?))\b"),
    ("GOVERNANCE", r"\b(privacy|pii|fairness|bias audit|governance|compliance scan)\b"),
    ("DECISION_ANALYSIS", r"\b(what should we do|recommend|best action|optimi[sz]e|decision analysis|allocate .* budget)\b"),
    ("CONTRIBUTION", r"\b(contribut(?:ion|e|ed|ing)|decompose .* change|drivers of .* (?:change|churn|attrition|revenue|cost|profit)|main drivers of)\b"),
    ("ANOMALY", r"\b(anomal(?:y|ies|ous)|outlier|unusual (?:records?|transactions?)|detect fraud)\b"),
    ("TEMPORAL_TREND", r"\b(trend|over time|month[- ]over[- ]month|year[- ]over[- ]year|changed over)\b"),
    ("ASSOCIATION", r"\b(correlation|correlat\w*|relationship between|association between|associated with|moves together)\b"),
    ("COMPARISON", r"\b(compare|comparison|difference between|higher than|lower than|versus|\bvs\.?\b|gap between|which group performs better)\b"),
    ("RANKING", r"\b(rank|top \d+|bottom \d+|highest|lowest|most profitable|least profitable|best[- ]performing|worst[- ]performing)\b"),
    ("AGGREGATION", r"\b(sum|total|average|mean|median|count|minimum|maximum|retention rate|churn rate|conversion rate|how many|how much|per (?:region|category|customer|product|month|quarter|year)|by (?:region|category|customer|product|month|quarter|year))\b"),
]


def infer_value_type(dtype: Any) -> str:
    """Map pandas/Arrow-like dtype labels to a conservative logical type."""
    s = str(dtype).casefold()
    if any(x in s for x in ("datetime", "timestamp", "date")): return "datetime"
    if "bool" in s: return "boolean"
    if any(x in s for x in ("int", "float", "double", "decimal", "numeric")): return "numeric"
    if any(x in s for x in ("category", "enum")): return "categorical"
    if any(x in s for x in ("string", "object", "utf", "text")): return "text_or_mixed"
    return "unknown"


def _semantic_value(sem: Any, *names: str) -> Any:
    """Read the first populated semantic alias without assuming one schema version."""
    for name in names:
        value = getattr(sem, name, None) if sem is not None else None
        if value not in (None, "", [], {}):
            return value
    return None


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _resolve_column_mentions(question: str, schema: list[dict[str, Any]]) -> list[str]:
    """Return schema-grounded column mentions, longest names first; never invent columns."""
    qn = f" {_normalise_name(question)} "
    found = []
    for item in sorted(schema, key=lambda x: len(_normalise_name(x["column"])), reverse=True):
        name = _normalise_name(item["column"])
        if name and f" {name} " in qn:
            found.append(item["column"])
    return found


def _question_contract(question: str, schema: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract conservative, auditable NL cues; candidates are not execution predicates."""
    q = " ".join(str(question or "").split())
    ql = q.casefold()
    operations = []
    op_patterns = [
        ("COUNT", r"\b(count|how many|number of records|number of rows)\b"),
        ("SUM", r"\b(sum|total|overall amount)\b"),
        ("MEAN", r"\b(mean|average|avg)\b"),
        ("MEDIAN", r"\bmedian\b"),
        ("MODE", r"\bmode|most common\b"),
        ("MIN", r"\b(minimum|min|smallest|lowest value)\b"),
        ("MAX", r"\b(maximum|max|largest|highest value)\b"),
        ("STD", r"\b(standard deviation|std dev|volatility)\b"),
        ("VARIANCE", r"\bvariance\b"),
        ("SKEWNESS", r"\bskew(?:ness)?\b"),
        ("KURTOSIS", r"\bkurtosis\b"),
        ("DISTINCT_COUNT", r"\b(distinct count|unique values|number of unique)\b"),
        ("MISSING_PERCENT", r"\b(percent(?:age)? of .*missing|null rate|missing rate|completeness)\b"),
        ("PERCENT_CHANGE", r"\b(percent(?:age)? change|growth rate|grew by|declined by|decreased by|increased by)\b"),
        ("CORRELATION", r"\b(correlation|correlat\w*|relationship between|moves together)\b"),
        ("RATE", r"\b(rate|share|proportion|conversion|retention rate|churn rate)\b"),
    ]
    for op, pattern in op_patterns:
        if re.search(pattern, q, re.I): operations.append(op)

    grouping = []
    for match in re.finditer(r"\b(?:by|per|for each|within each|across)\s+([\w][\w ./-]{0,48}?)(?=\s+(?:over|during|between|where|with|and|then|while|from|to|in the last)\b|[,;?]|$)", q, re.I):
        raw = match.group(1).strip(" .,-")
        if raw and raw.casefold() not in {"how", "much", "many", "what", "the"}:
            grouping.append(raw)
    grouping = list(dict.fromkeys(grouping))[:5]

    time_grain = None
    grain_patterns = [("day", r"\b(daily|per day|by day|day-over-day)\b"),
                      ("week", r"\b(weekly|per week|by week|week-over-week)\b"),
                      ("month", r"\b(monthly|per month|by month|month-over-month|mom)\b"),
                      ("quarter", r"\b(quarterly|per quarter|by quarter|quarter-over-quarter|qoq)\b"),
                      ("year", r"\b(annually|annual|yearly|per year|by year|year-over-year|yoy)\b")]
    for grain, pattern in grain_patterns:
        if re.search(pattern, q, re.I): time_grain = grain; break

    top_n = None
    rank_match = re.search(r"\b(top|bottom)\s+(\d{1,4})\b", q, re.I)
    if rank_match: top_n = {"direction": rank_match.group(1).upper(), "n": int(rank_match.group(2))}

    comparison = None
    if re.search(r"\b(compare|comparison|versus|\bvs\.?\b|difference between|gap between)\b", q, re.I):
        comparison = "explicit_comparison"
    elif re.search(r"\b(higher than|lower than|more than|less than|increase(?:d)?|decrease(?:d)?|improv(?:e|ed|ement)|worsen(?:ed)?)\b", q, re.I):
        comparison = "directional_comparison"

    filter_candidates = []
    for match in re.finditer(r"\b(where|when|with|without|excluding|except|only)\s+([^,;?]{1,100})", q, re.I):
        clause = match.group(2).strip()
        if clause: filter_candidates.append({"cue": match.group(1).casefold(), "raw_clause": clause, "column_binding": "UNRESOLVED", "execution_safe": False})
    date_scope = None
    m = re.search(r"\b(last|past|previous)\s+(\d+)\s+(days?|weeks?|months?|quarters?|years?)\b", q, re.I)
    if m: date_scope = {"kind": "relative_window_candidate", "direction": m.group(1).casefold(), "amount": int(m.group(2)), "unit": m.group(3).casefold(), "execution_safe": False}
    elif re.search(r"\b(this|last|previous)\s+(month|quarter|year|week)\b", q, re.I):
        m = re.search(r"\b(this|last|previous)\s+(month|quarter|year|week)\b", q, re.I)
        date_scope = {"kind": "calendar_window_candidate", "direction": m.group(1).casefold(), "unit": m.group(2).casefold(), "execution_safe": False}

    mentions = _resolve_column_mentions(q, schema)
    ambiguities = []
    if re.search(r"\b(best|better|worst|improve|optimi[sz]e|effective)\b", q, re.I) and not re.search(r"\b(objective|goal|constraint|cost|revenue|profit|accuracy|conversion|retention)\b", q, re.I):
        ambiguities.append("decision_objective_not_explicit")
    if comparison and len(mentions) < 2 and not grouping:
        ambiguities.append("comparison_entities_or_grouping_not_resolved")
    if re.search(r"\b(cause|caused|causal|impact|effect of)\b", q, re.I) and not re.search(r"\b(experiment|randomi[sz]|difference[- ]in[- ]differences|instrument|regression discontinuity|before and after)\b", q, re.I):
        ambiguities.append("causal_design_not_stated")
    if re.search(r"\b(forecast|forecasting|predict future|next \d+ periods?)\b", q, re.I):
        if not re.search(r"\bnext\s+\d+\s+(?:periods?|days?|weeks?|months?|quarters?|years?)\b|\bfor\s+\d+\s+(?:periods?|days?|weeks?|months?|quarters?|years?)\b", q, re.I):
            ambiguities.append("forecast_horizon_not_specified")
        if not re.search(r"\b(daily|weekly|monthly|quarterly|yearly|per day|per week|per month|per quarter|per year|by day|by week|by month|by quarter|by year)\b", q, re.I):
            ambiguities.append("forecast_grain_not_specified")

    return {"operations": operations, "grouping_candidates": grouping, "time_grain": time_grain,
            "ranking": top_n, "comparison_mode": comparison, "filter_candidates": filter_candidates,
            "date_scope_candidate": date_scope, "schema_column_mentions": mentions,
            "ambiguities": list(dict.fromkeys(ambiguities)),
            "candidate_only_not_execution_contract": True}


def compile_typed_investigation(question: str, task: str, df: Any = None, semantics: Any = None) -> dict[str, Any]:
    """Create a broad, type-aware plan and conservative NL contract proposal."""
    q = " ".join(str(question or "").split())
    selected = None
    for kind, pattern in _TASK_PATTERNS:
        if re.search(pattern, q, re.I):
            selected = kind
            break
    if selected is None:
        selected = {"DATA_QUALITY": "MISSINGNESS", "FORECAST": "FORECAST", "CAUSAL": "CAUSAL_EFFECT", "SEGMENTATION": "SEGMENTATION", "RECONCILIATION": "RECONCILIATION", "GOVERNANCE": "GOVERNANCE", "PRESCRIPTIVE": "DECISION_ANALYSIS", "ASSOCIATION": "ASSOCIATION", "COMPARISON": "COMPARISON", "DIAGNOSTIC": "CONTRIBUTION", "PREDICTION": "PREDICTION"}.get(str(task).upper(), "PROFILE")
    spec = TASK_TYPES[selected]
    def _infer_series_value_type(series: Any) -> str:
        value_type = infer_value_type(getattr(series, "dtype", "unknown"))
        if value_type != "text_or_mixed":
            return value_type
        try:
            observed = series.dropna()
            n = int(len(observed))
            unique_count = int(observed.nunique(dropna=True))
        except Exception:
            return value_type
        if n < 2 or unique_count < 2:
            return value_type
        cardinality_limit = min(50, max(10, int(n ** 0.5 + 0.999999)))
        return "categorical" if unique_count <= cardinality_limit else value_type

    schema = []
    if df is not None and hasattr(df, "columns"):
        for col in list(df.columns):
            try: dtype = str(df[col].dtype)
            except Exception: dtype = "unknown"
            schema.append({"column": str(col), "value_type": _infer_series_value_type(df[col]), "source_dtype": dtype})
    sem = semantics
    target = _semantic_value(sem, "target_column", "target_metric_col", "metric_column", "outcome_column")
    groups = _semantic_value(sem, "grouping_columns", "group_dimension_col", "group_column", "group_by") or []
    if isinstance(groups, str): groups = [groups]
    explanatory = _semantic_value(sem, "explanatory_columns", "feature_columns", "predictor_columns", "x_columns") or []
    if isinstance(explanatory, str): explanatory = [explanatory]
    time_col = _semantic_value(sem, "time_column", "time_col", "date_column", "timestamp_column")
    role_values = {
        "measure": target, "outcome": _semantic_value(sem, "outcome_column", "target_column", "target_metric_col"),
        "target": _semantic_value(sem, "prediction_target", "target_column", "target_metric_col"),
        "x": explanatory[0] if explanatory else None, "y": target,
        "group": groups[0] if groups else None, "time": time_col,
        "features": list(explanatory), "entity": groups[0] if groups else None,
        "treatment": _semantic_value(sem, "treatment_column", "treatment_col"),
        "design": _semantic_value(sem, "causal_design", "identification_strategy"),
        "comparison_window": _semantic_value(sem, "comparison_window", "baseline_window"),
        "formula_or_components": _semantic_value(sem, "reconciliation_formula", "formula", "components"),
        "measure_or_features": target or list(explanatory),
        "duration": _semantic_value(sem, "duration_column", "duration_col"),
        "event": _semantic_value(sem, "event_column", "event_col", "churn_event_col"),
        "text": _semantic_value(sem, "text_column", "text_col", "review_text_column"),
        "objective": _semantic_value(sem, "decision_objective", "objective"),
        "constraints": _semantic_value(sem, "decision_constraints", "constraints"),
    }
    contract = _question_contract(q, schema)
    type_by_column = {item["column"]: item["value_type"] for item in schema}
    mentioned = list(contract["schema_column_mentions"])
    # Bind only unambiguous, schema-grounded mentions. Never manufacture a field name.
    if not role_values.get("measure"):
        numeric_mentions = [c for c in mentioned if type_by_column.get(c) == "numeric"]
        if len(numeric_mentions) == 1:
            role_values["measure"] = numeric_mentions[0]
            role_values["outcome"] = role_values.get("outcome") or numeric_mentions[0]
            role_values["y"] = role_values.get("y") or numeric_mentions[0]
            role_values["measure_or_features"] = role_values.get("measure_or_features") or numeric_mentions[0]
    if not role_values.get("group"):
        resolved_groups = []
        for candidate in contract["grouping_candidates"]:
            candidate_norm = _normalise_name(candidate)
            exact = [c for c in schema if _normalise_name(c["column"]) == candidate_norm]
            if exact: resolved_groups.append(exact[0]["column"])
        resolved_groups = list(dict.fromkeys(resolved_groups))
        if len(resolved_groups) == 1:
            role_values["group"] = resolved_groups[0]
            role_values["entity"] = resolved_groups[0]
            contract["group_binding_status"] = "SCHEMA_RESOLVED"
        elif contract["grouping_candidates"]:
            contract["group_binding_status"] = "UNRESOLVED"
            contract["unresolved_grouping_candidates"] = contract["grouping_candidates"]
    if selected == "ASSOCIATION":
        if not role_values.get("x") and len(mentioned) >= 2:
            role_values["x"] = mentioned[0]
        if not role_values.get("y") and len(mentioned) >= 2:
            role_values["y"] = mentioned[1]
    unresolved_roles = [role for role in spec["min_roles"] if not role_values.get(role)]
    if contract.get("group_binding_status") == "UNRESOLVED" and "group" not in unresolved_roles:
        unresolved_roles.append("group")
    invalid_role_bindings = []
    if schema:
        column_roles = {"measure", "outcome", "target", "x", "y", "group", "time", "entity", "treatment", "duration", "event", "text"}
        for role in column_roles:
            value = role_values.get(role)
            if value and isinstance(value, str) and value not in type_by_column:
                invalid_role_bindings.append({"role": role, "value": value, "reason": "column_not_in_observed_schema"})
                if role not in unresolved_roles: unresolved_roles.append(role)
        for role in ("features", "measure_or_features"):
            value = role_values.get(role)
            if isinstance(value, list):
                for col in value:
                    if col not in type_by_column:
                        invalid_role_bindings.append({"role": role, "value": col, "reason": "column_not_in_observed_schema"})
                        if role not in unresolved_roles: unresolved_roles.append(role)
    typed_roles = {k: v for k, v in role_values.items() if v not in (None, [], {})}
    role_types: dict[str, Any] = {}
    for role, value in typed_roles.items():
        if isinstance(value, str) and value in type_by_column: role_types[role] = type_by_column[value]
        elif isinstance(value, list): role_types[role] = {str(col): type_by_column.get(str(col), "unresolved") for col in value}
    type_warnings: list[str] = []
    if selected in {"AGGREGATION", "DISTRIBUTION", "FORECAST"} and role_types.get("measure") not in (None, "numeric"):
        type_warnings.append("The proposed measure is not confirmed numeric; select a compatible aggregate or resolve the measure type.")
    if invalid_role_bindings:
        type_warnings.append("One or more proposed semantic roles do not match columns in the observed schema; execution must wait for re-binding.")
    if selected == "FORECAST" and role_types.get("time") not in (None, "datetime", "numeric"):
        type_warnings.append("The proposed time index is not confirmed temporal/numeric.")
    if selected == "ASSOCIATION":
        for role in ("x", "y"):
            if role_types.get(role) not in (None, "numeric", "categorical", "boolean", "ordinal"):
                type_warnings.append(f"Role {role} has unsupported or unknown type for association method selection.")
    if selected == "AGGREGATION" and contract["grouping_candidates"] and not role_values.get("group"):
        # Grouped aggregation is still aggregation; missing group binding must be visible.
        contract["group_binding_status"] = "UNRESOLVED"
        if "group" not in unresolved_roles: unresolved_roles.append("group")
    return {
        "plan_version": PLAN_VERSION, "task_type": selected, "task_family": spec["family"],
        "source_question": q, "claim_ceiling": spec["claim_ceiling"], "typed_schema": schema,
        "semantic_roles": typed_roles, "semantic_role_types": role_types, "type_warnings": type_warnings,
        "invalid_role_bindings": invalid_role_bindings,
        "required_roles": list(spec["min_roles"]), "unresolved_roles": unresolved_roles,
        "readiness": "NEEDS_ROLE_RESOLUTION" if unresolved_roles else "ROLE_PROPOSAL_READY",
        "natural_language_contract": contract,
        "execution_authorized": False, "runtime_authorization_required": True,
        "branches": _default_branches(selected),
    }


def _default_branches(task_type: str) -> list[dict[str, Any]]:
    """Conditional checks that are meaningful for the task family; all fail closed."""
    common = [{"id": "B-DQ", "when": {"fact": "data_quality.status", "op": "in", "value": ["PASS", "WARN"]}, "then": "CONTINUE", "otherwise": "BLOCK_OR_CLARIFY", "reason": "Material data quality defects must be addressed before analysis."}]
    branches = {
        "ASSOCIATION": [{"id": "B-TYPE", "when": {"all": [
            {"fact": "roles.x_type", "op": "in", "value": ["numeric", "ordinal", "categorical", "boolean"]},
            {"fact": "roles.y_type", "op": "in", "value": ["numeric", "ordinal", "categorical", "boolean"]},
        ]}, "then": "TYPE_COMPATIBLE_ASSOCIATION_METHOD", "otherwise": "BLOCK_OR_CLARIFY", "reason": "Both variables must have observed, supported types before method selection."}],
        "COMPARISON": [{"id": "B-GROUP", "when": {"fact": "roles.group_cardinality", "op": "gt", "value": 1}, "then": "GROUP_COMPARISON", "otherwise": "BLOCK_OR_CLARIFY", "reason": "A group comparison requires at least two observed groups."}],
        "FORECAST": [{"id": "B-TIME", "when": {"all": [
            {"fact": "time_index.monotonic", "op": "eq", "value": True},
            {"fact": "forecast.horizon_status", "op": "eq", "value": "SPECIFIED"},
        ]}, "then": "TIME_ORDERED_BACKTEST", "otherwise": "BLOCK_OR_CLARIFY_HORIZON_OR_TIME_INDEX", "reason": "Forecasting requires a verified time index and an explicit horizon."}],
        "CAUSAL_EFFECT": [{"id": "B-ID", "when": {"fact": "causal_identification.status", "op": "eq", "value": "IDENTIFIED"}, "then": "ESTIMATE_CAUSAL_EFFECT", "otherwise": "REPORT_NON_CAUSAL_OR_INCONCLUSIVE", "reason": "No causal estimate unless identification is supported."}],
        "PREDICTION": [{"id": "B-LEAK", "when": {"fact": "prediction.leakage_status", "op": "eq", "value": "PASS"}, "then": "HELD_OUT_EVALUATION", "otherwise": "BLOCK_MODEL_FIT", "reason": "Leakage must be cleared before model evaluation."}],
        "SEGMENTATION": [{"id": "B-STABILITY", "when": {"fact": "cluster.stability_status", "op": "eq", "value": "PASS"}, "then": "INTERPRET_SEGMENTS", "otherwise": "REPORT_NO_STABLE_SEGMENTATION", "reason": "Do not present unstable clusters as discovered segments."}],
        "DECISION_ANALYSIS": [{"id": "B-CONTEXT", "when": {"fact": "decision_context.status", "op": "eq", "value": "COMPLETE"}, "then": "COMPARE_FEASIBLE_OPTIONS", "otherwise": "ASK_FOR_MATERIAL_CONSTRAINTS", "reason": "Recommendations require objective and constraints."}],
        "RECONCILIATION": [{"id": "B-FORMULA", "when": {"fact": "reconciliation.formula_status", "op": "eq", "value": "SPECIFIED"}, "then": "TEST_IDENTITY", "otherwise": "ASK_FOR_FORMULA_OR_PROPOSE_CANDIDATES", "reason": "Do not silently invent an accounting identity."}],
    }
    return common + branches.get(task_type, [])


_ALLOWED_OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "truthy", "exists"}


def evaluate_condition(condition: Mapping[str, Any], facts: Mapping[str, Any]) -> str:
    """Evaluate a safe single predicate; UNKNOWN is distinct from FALSE."""
    if not isinstance(condition, Mapping): return "UNKNOWN"
    # Compound boolean predicates are intentionally limited to declarative all/any.
    # Every child is evaluated by this same safe evaluator; no expression execution.
    if "all" in condition or "any" in condition:
        key = "all" if "all" in condition else "any"
        children = condition.get(key)
        if not isinstance(children, (list, tuple)) or not children:
            return "UNKNOWN"
        verdicts = [evaluate_condition(child, facts) for child in children]
        if key == "all":
            if "FALSE" in verdicts: return "FALSE"
            return "TRUE" if all(v == "TRUE" for v in verdicts) else "UNKNOWN"
        if "TRUE" in verdicts: return "TRUE"
        return "FALSE" if all(v == "FALSE" for v in verdicts) else "UNKNOWN"
    fact = condition.get("fact"); op = condition.get("op"); expected = condition.get("value")
    if not isinstance(fact, str) or not fact or op not in _ALLOWED_OPS: return "UNKNOWN"
    current: Any = facts
    for part in fact.split("."):
        if not isinstance(current, Mapping) or part not in current: return "UNKNOWN"
        current = current[part]
    if op == "exists": return "TRUE" if current is not None else "FALSE"
    if current is None: return "UNKNOWN"
    # Type disagreement is not evidence of inequality: it is an invalid or
    # under-specified predicate and therefore UNKNOWN. This avoids Python's
    # bool/int equivalence (True == 1) and accidental coercion in branch gates.
    if op in {"eq", "ne"} and type(current) is not type(expected):
        return "UNKNOWN"
    if op in {"in", "not_in"} and not isinstance(expected, (list, tuple, set, frozenset, str)):
        return "UNKNOWN"
    try:
        result = {
            "eq": lambda: current == expected,
            "ne": lambda: current != expected,
            "gt": lambda: current > expected, "gte": lambda: current >= expected,
            "lt": lambda: current < expected, "lte": lambda: current <= expected,
            "in": lambda: current in expected, "not_in": lambda: current not in expected,
            "truthy": lambda: current is True,
        }[op]()
    except (TypeError, ValueError, KeyError):
        return "UNKNOWN"
    return "TRUE" if result else "FALSE"


def resolve_conditional_step(step: Mapping[str, Any], facts: Mapping[str, Any], prerequisite_statuses: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Resolve a planned step to RUN/SKIP/BLOCKED; missing facts never pass."""
    prerequisites = list(step.get("prerequisites") or [])
    statuses = prerequisite_statuses or {}
    failed = [p for p in prerequisites if statuses.get(str(p)) not in {"COMPLETED", "PASS", "SUCCEEDED"}]
    if failed:
        return {"status": "BLOCKED", "reason": "prerequisites_not_satisfied", "missing_or_failed_prerequisites": failed}
    condition = step.get("when")
    if condition is None:
        return {"status": "RUN", "reason": "unconditional_step"}
    verdict = evaluate_condition(condition, facts)
    if verdict == "TRUE": return {"status": "RUN", "reason": "condition_true", "condition_result": verdict}
    if verdict == "FALSE": return {"status": "SKIP", "reason": "condition_false", "condition_result": verdict, "fallback": step.get("otherwise", "BLOCK")}
    return {"status": "BLOCKED", "reason": "condition_unknown_or_invalid", "condition_result": verdict, "fallback": step.get("otherwise", "BLOCK")}


def resolve_plan_branches(plan: Mapping[str, Any], facts: Mapping[str, Any], prerequisite_statuses: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Resolve every declared branch into an auditable decision record."""
    out = []
    for branch in list(plan.get("branches") or []):
        decision = resolve_conditional_step(branch, facts, prerequisite_statuses)
        out.append({"branch_id": branch.get("id"), "then": branch.get("then"), **decision})
    return out


_REQUIRED_EXECUTION_BRANCHES = {
    "FORECAST": ("B-TIME", "Forecast execution requires a verified time index and an explicit forecast horizon (for example \"next month\" or \"next 3 months\")."),
    "PREDICTION": ("B-LEAK", "Predictive model fitting requires an independent target-leakage check to PASS."),
    "COMPARISON": ("B-GROUP", "Group comparison requires at least two observed groups."),
    "ASSOCIATION": ("B-TYPE", "Association requires a validated method compatible with the resolved variable types."),
    "SEGMENTATION": ("B-STABILITY", "Segment interpretation requires measured cluster stability."),
    "DECISION_ANALYSIS": ("B-CONTEXT", "Decision recommendations require an explicit objective and material constraints."),
    "RECONCILIATION": ("B-FORMULA", "Reconciliation requires a specified or explicitly confirmed identity."),
}

def conditional_execution_policy(task_type: str, decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Convert branch decisions into an enforceable task-level runtime policy.

    This is intentionally conservative: missing branch records are not treated as
    success. Causal tasks may continue non-causal investigation, but causal effect
    estimation is separately authorized only by B-ID.
    """
    task = str(task_type or "UNRESOLVED").upper()
    by_id = {str(item.get("branch_id")): item for item in decisions}
    if task == "CAUSAL_EFFECT":
        authorized = by_id.get("B-ID", {}).get("status") == "RUN"
        return {"authorized": True, "causal_effect_authorized": authorized,
                "action": "ALLOW_CAUSAL_EFFECT" if authorized else "ALLOW_NON_CAUSAL_ONLY",
                "reason": None if authorized else "causal_identification_branch_not_passed"}
    required = _REQUIRED_EXECUTION_BRANCHES.get(task)
    if required is None:
        return {"authorized": True, "causal_effect_authorized": False, "action": "CONTINUE", "reason": None}
    branch_id, reason = required
    decision = by_id.get(branch_id)
    authorized = bool(decision and decision.get("status") == "RUN")
    return {"authorized": authorized, "causal_effect_authorized": False,
            "action": "CONTINUE" if authorized else "BLOCK_SPECIALIST_PATH",
            "branch_id": branch_id, "branch_status": decision.get("status") if decision else "MISSING",
            "reason": None if authorized else reason}
