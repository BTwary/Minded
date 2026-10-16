"""Stateful, schema-grounded intent support for the canonical question contract.

The canonical question contract remains the single source of truth.  This module adds:
- deterministic normalization of user guidance;
- typed, validated patches to material contract slots;
- an evidence ledger for common material clauses that the current parser may not represent.

It intentionally refuses to guess when a material clause cannot be represented in the contract.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd


_GUIDANCE_EMPTY = (None, "", [], {})
_PATCH_ALIASES = {
    "target_column": "target_column",
    "metric": "target_column",
    "analysis.target.measure": "target_column",
    "requested_aggregation": "requested_aggregation",
    "aggregation": "requested_aggregation",
    "analysis.target.agg": "requested_aggregation",
    "grouping_columns": "grouping_columns",
    "group_by": "grouping_columns",
    "analysis.group_by": "grouping_columns",
    "time_column": "time_column",
    "dataset_scope.time_column": "time_column",
    "time_range": "time_range",
    "dataset_scope.time_range": "time_range",
    "filter_conditions": "filter_conditions",
    "filters": "filter_conditions",
    "dataset_scope.filters": "filter_conditions",
    "add_filter_conditions": "add_filter_conditions",
    "dataset_scope.add_filter": "add_filter_conditions",
    "ranking_direction": "ranking_direction",
    "analysis.ranking_direction": "ranking_direction",
    "denominator_column": "denominator_column",
    "analysis.denominator": "denominator_column",
    "numerator_column": "numerator_column",
    "analysis.numerator": "numerator_column",
}


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def normalize_guidance_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize API/GUI guidance identically before preview, create, and revision.

    Empty acknowledgement lists are deliberately discarded: acknowledging no assumptions is
    not an analytical instruction. Non-empty acknowledgements are retained as provenance, but
    the resolver must not interpret them as semantic guidance.
    """
    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in dict(payload or {}).items():
        key = str(raw_key)
        value = _dump(raw_value)
        if key == "confirmed_assumptions" and isinstance(value, Mapping):
            if value.get("replace") is not None:
                value = list(value.get("replace") or [])
            else:
                items = list(value.get("add") or [])
                for remove_item in value.get("remove") or []:
                    items = [item for item in items if item != remove_item]
                value = items
        if value in _GUIDANCE_EMPTY:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        normalized[key] = value

    if "intent_patch" in normalized:
        normalized["intent_patch"] = normalize_intent_patch(normalized["intent_patch"])
        # The option label is display text, not a second instruction source. Once an explicit
        # patch is supplied, the patch is authoritative and the label must not be parsed again.
        normalized.pop("clarification_response", None)
    return normalized


def normalize_intent_patch(raw_patch: Any) -> Dict[str, Any]:
    """Normalize ``{slot, value}`` or ``{set: {slot: value}}`` into typed slot assignments."""
    patch = _dump(raw_patch)
    if not isinstance(patch, Mapping):
        raise ValueError("intent_patch must be an object with 'set' or 'slot'/'value'.")
    if isinstance(patch.get("set"), Mapping):
        raw_set = dict(patch["set"])
    elif "slot" in patch and "value" in patch:
        raw_set = {str(patch["slot"]): patch["value"]}
    else:
        # Also accept the direct map form as a compact API representation.
        raw_set = dict(patch)
    output: Dict[str, Any] = {}
    for raw_slot, value in raw_set.items():
        slot = _PATCH_ALIASES.get(str(raw_slot))
        if slot is None:
            raise ValueError(f"Unsupported intent patch slot: {raw_slot!r}.")
        if slot in output and output[slot] != value:
            raise ValueError(f"Conflicting values supplied for intent slot {slot!r}.")
        output[slot] = value
    if not output:
        raise ValueError("intent_patch must contain at least one slot assignment.")
    return output


def _temporal_columns(frame: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for column in frame.columns:
        series = frame[column]
        name = str(column).casefold()
        # Numeric durations such as ``resolution_time`` are not calendar fields. A name hint
        # is usable only for nonnumeric values that actually parse as dates.
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_datetime64_any_dtype(series) or any(token in name for token in ("date", "time", "timestamp", "_at")):
            parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
            nonnull = int(series.notna().sum())
            if nonnull and int(parsed.notna().sum()) / nonnull >= 0.8:
                candidates.append(str(column))
    return candidates


def _relative_time_range(
    question: str,
    frame: Optional[pd.DataFrame] = None,
    time_column: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    q = " ".join(str(question or "").lower().split())
    now = datetime.now(timezone.utc).date()

    # If a real temporal column is known, reuse the repository's established data-anchored
    # resolver first. This keeps preview/approval consistent with historical/offline datasets;
    # a 2022 dataset's "last quarter" must not silently mean a quarter in the current year.
    if frame is not None and time_column and time_column in frame.columns:
        try:
            from packages.analytics_core.src.engines.temporal import TemporalResolver
            resolved = TemporalResolver.resolve(question, frame, time_column)
            if resolved.found and resolved.period_start and resolved.period_end:
                result = {
                    "type": "absolute", "expression": resolved.matched_expression or q,
                    "start": resolved.period_start.isoformat(), "end": resolved.period_end.isoformat(),
                    "end_exclusive": True, "source": "dataset_temporal_resolver",
                    "resolution_method": resolved.resolution_method,
                }
                if resolved.baseline_start:
                    result["baseline_start"] = resolved.baseline_start.isoformat()
                if resolved.baseline_end:
                    result["baseline_end"] = resolved.baseline_end.isoformat()
                return result
            # If the engine recognized a phrase but could not bind it to the dataset's date
            # range, do not replace that failure with wall-clock dates.
            if resolved.ambiguous:
                return {"type": "unresolved", "expression": resolved.matched_expression or q,
                        "reason": resolved.ambiguity_note or "temporal_scope_ambiguous", "source": "dataset_temporal_resolver"}
        except Exception:
            # The deterministic fallback below is intentionally narrow and still schema-bound.
            pass

        # Generalize data-anchored rolling windows beyond the TemporalResolver's built-ins.
        rolling = re.search(r"\b(?:last|past)\s+(\d{1,4})\s+(day|days|week|weeks|month|months|year|years)\b", q)
        if rolling:
            values = pd.to_datetime(frame[time_column], errors="coerce", utc=True).dropna()
            if values.empty:
                return {"type": "unresolved", "expression": rolling.group(0), "reason": "time_column_has_no_parseable_dates", "source": "dataset_temporal_resolver"}
            from datetime import timedelta
            n = int(rolling.group(1))
            unit = rolling.group(2).rstrip("s")
            days = n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
            end_date = values.max().date() + timedelta(days=1)
            start_date = end_date - timedelta(days=days)
            return {"type": "absolute", "expression": rolling.group(0), "start": start_date.isoformat(),
                    "end": end_date.isoformat(), "end_exclusive": True, "source": "dataset_anchored_rolling_window",
                    "unit_interpretation": f"{n} {unit}(s); months and years use 30/365-day windows"}

        # A requested current quarter is anchored to the most recent observed date, consistent
        # with the existing offline/historical temporal policy.
        if re.search(r"\b(this|current)\s+quarter\b", q):
            values = pd.to_datetime(frame[time_column], errors="coerce", utc=True).dropna()
            if values.empty:
                return {"type": "unresolved", "expression": "current quarter", "reason": "time_column_has_no_parseable_dates", "source": "dataset_temporal_resolver"}
            latest = values.max().date()
            quarter = (latest.month - 1) // 3 + 1
            start = latest.replace(month=(quarter - 1) * 3 + 1, day=1)
            end_month = start.month + 3
            end = latest.replace(year=start.year + (1 if end_month > 12 else 0), month=(end_month - 1) % 12 + 1, day=1)
            return {"type": "absolute", "expression": "current quarter", "start": start.isoformat(),
                    "end": end.isoformat(), "end_exclusive": True, "source": "dataset_temporal_resolver"}
    # Explicit relative windows are anchored to a UTC calendar date so preview/create on the
    # same day hash to the same contract rather than drifting every second.
    match = re.search(r"\b(?:last|past)\s+(\d{1,4})\s+(day|days|week|weeks|month|months|year|years)\b", q)
    if match:
        n = int(match.group(1))
        if n < 1 or n > 5000:
            return {"type": "unresolved", "expression": match.group(0), "reason": "window_out_of_range"}
        unit = match.group(2).rstrip("s")
        multiplier = {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        from datetime import timedelta
        # Treat day-granularity windows as N calendar dates including today. The end is
        # exclusive (tomorrow at midnight) so timestamps throughout today are included.
        end = now + timedelta(days=1)
        start = now - timedelta(days=n * multiplier - 1)
        return {
            "type": "absolute", "expression": match.group(0), "start": start.isoformat(),
            "end": end.isoformat(), "end_exclusive": True, "as_of": now.isoformat(),
            "source": "relative_question_window",
            "unit_interpretation": f"{n} {unit}(s); months and years use 30/365-day windows",
        }

    if re.search(r"\b(last|previous|prior)\s+quarter\b", q):
        quarter = (now.month - 1) // 3 + 1
        this_q_start_month = (quarter - 1) * 3 + 1
        this_q_start = now.replace(month=this_q_start_month, day=1)
        from datetime import timedelta
        prior_day = this_q_start - timedelta(days=1)
        prior_q = (prior_day.month - 1) // 3 + 1
        start = prior_day.replace(month=(prior_q - 1) * 3 + 1, day=1)
        return {"type": "absolute", "expression": "last quarter", "start": start.isoformat(),
                "end": this_q_start.isoformat(), "end_exclusive": True, "as_of": now.isoformat(),
                "source": "relative_question_window"}

    if re.search(r"\b(this|current)\s+quarter\b", q):
        quarter = (now.month - 1) // 3 + 1
        start = now.replace(month=(quarter - 1) * 3 + 1, day=1)
        from datetime import timedelta
        end_month = start.month + 3
        end = start.replace(year=start.year + (1 if end_month > 12 else 0), month=(end_month - 1) % 12 + 1, day=1)
        return {"type": "absolute", "expression": "current quarter", "start": start.isoformat(),
                "end": end.isoformat(), "end_exclusive": True,
                "as_of": now.isoformat(), "source": "relative_question_window"}
    return None


def infer_intent_patch_from_answer(answer: str, pending_slot: str, frame: pd.DataFrame) -> Dict[str, Any]:
    """Turn a short freeform clarification answer into a schema-grounded slot patch.

    This deliberately supports only high-precision deterministic forms. When the answer does not
    identify one unique field/value, it raises instead of inventing a binding. A model-based parser
    may later produce the same patch shape, but the patch validator remains authoritative.
    """
    text = " ".join(str(answer or "").strip().split())
    if not text:
        raise ValueError("Enter an answer for the unresolved intent slot.")
    slot = str(pending_slot or "").strip()
    columns = [str(column) for column in frame.columns]

    def normalize_phrase(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def find_unique_column(candidate_text: str) -> Optional[str]:
        clean = candidate_text.casefold()
        exact = [col for col in columns if normalize_phrase(col) == normalize_phrase(clean)]
        if len(exact) == 1:
            return exact[0]
        contained = [col for col in columns if normalize_phrase(col) and normalize_phrase(col) in normalize_phrase(clean)]
        # Longest match avoids matching `date` when the user supplied `created_date`.
        if contained:
            max_len = max(len(normalize_phrase(col)) for col in contained)
            longest = [col for col in contained if len(normalize_phrase(col)) == max_len]
            if len(longest) == 1:
                return longest[0]
        return None

    if slot in {"target_column", "numerator_column", "denominator_column", "time_column"}:
        answer_text = re.sub(r"^(?:use|choose|select|the|column|field)\s+", "", text, flags=re.I)
        answer_text = re.sub(r"\s+(?:as\s+)?(?:the\s+)?(?:time\s+)?(?:column|field)$", "", answer_text, flags=re.I)
        column = find_unique_column(answer_text)
        if not column:
            raise ValueError(f"Could not identify one unique dataset column from {text!r}. Enter the exact column name or choose a suggested option.")
        return {slot: column}

    if slot == "grouping_columns":
        pieces = [part.strip() for part in re.split(r",|\band\b", text, flags=re.I) if part.strip()]
        groups = [find_unique_column(re.sub(r"^(?:by|group(?:ed)?\s+by|use|choose)\s+", "", piece, flags=re.I)) for piece in pieces]
        if not groups or any(group is None for group in groups):
            raise ValueError("Could not identify the requested grouping column(s). Use the exact dataset column name(s).")
        if len(groups) > 1:
            raise ValueError("The current deterministic plan supports one grouping column. Select one grouping field.")
        return {"grouping_columns": groups}

    if slot == "requested_aggregation":
        q = text.casefold()
        for pattern, value in ((r"\bmedian\b", "MEDIAN"), (r"\b(?:average|avg|mean)\b", "MEAN"),
                               (r"\b(?:sum|total)\b", "SUM"), (r"\bminimum|min\b", "MIN"),
                               (r"\bmaximum|max\b", "MAX"), (r"\b(?:row|record)\s+count\b|\bcount\b", "COUNT"),
                               (r"\bdistinct\b", "COUNT_DISTINCT")):
            if re.search(pattern, q):
                return {"requested_aggregation": value}
        raise ValueError("Specify one supported aggregation: sum, mean, median, min, max, count, or distinct count.")

    if slot == "ranking_direction":
        q = text.casefold()
        if re.search(r"\b(lowest|smallest|least|ascending|asc)\b", q):
            return {"ranking_direction": "ASC"}
        if re.search(r"\b(highest|largest|greatest|descending|desc|top)\b", q):
            return {"ranking_direction": "DESC"}
        raise ValueError("Specify whether to rank highest-to-lowest (DESC) or lowest-to-highest (ASC).")

    if slot == "time_range":
        q = text.casefold()
        if re.search(r"\b(all\s+(?:data|records|time)|entire\s+(?:dataset|history)|no\s+date\s+filter)\b", q):
            return {"time_range": {"type": "all", "expression": text, "source": "clarified"}}
        candidate_times = _temporal_columns(frame)
        relative = _relative_time_range(text, frame, candidate_times[0] if len(candidate_times) == 1 else None)
        if relative and relative.get("type") == "absolute":
            relative = {**relative, "source": "clarified"}
            return {"time_range": relative}
        range_match = re.search(r"\b(?:between|from)\s+(\d{4}-\d{2}-\d{2})\s+(?:and|to|through)\s+(\d{4}-\d{2}-\d{2})\b", q)
        if range_match:
            from datetime import timedelta
            start = pd.Timestamp(range_match.group(1))
            end_inclusive = pd.Timestamp(range_match.group(2))
            if start > end_inclusive:
                raise ValueError("The start date must not be later than the end date.")
            return {"time_range": {
                "type": "absolute", "start": start.date().isoformat(),
                "end": (end_inclusive + pd.Timedelta(days=1)).date().isoformat(),
                "end_exclusive": True, "expression": text, "source": "clarified",
            }}
        raise ValueError("Specify a supported relative period (for example, last 90 days or last quarter), an ISO date range, or all data.")

    if slot == "filter_conditions":
        direct = re.match(r"^\s*([A-Za-z_][\w .-]*)\s*(>=|<=|!=|=|>|<)\s*(.*?)\s*$", text)
        if direct:
            col = find_unique_column(direct.group(1).strip())
            if not col:
                raise ValueError(f"The filter column {direct.group(1).strip()!r} does not uniquely match a dataset column.")
            operator = direct.group(2)
            raw_value = direct.group(3).strip().strip("\"'")
            series = frame[col]
            if pd.api.types.is_bool_dtype(series):
                value: Any = raw_value.casefold() in {"true", "1", "yes", "y"}
            elif pd.api.types.is_numeric_dtype(series):
                value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
                if pd.isna(value):
                    raise ValueError(f"Filter value {raw_value!r} is not numeric for {col!r}.")
                value = value.item() if hasattr(value, "item") else value
            else:
                value = raw_value
            return {"add_filter_conditions": [{"column": col, "operator": operator, "value": value}]}

        q = text.casefold()
        exclude = bool(re.search(r"\b(exclude|excluding|without|ignore|except)\b", q))
        candidates: list[tuple[str, Any]] = []
        for col in columns:
            series = frame[col]
            values = list(series.dropna().unique())
            # Exact observed category/string/flag value appearing as a standalone phrase.
            for value in values:
                value_text = str(value).strip()
                if not value_text:
                    continue
                if re.search(rf"(?<!\w){re.escape(value_text.casefold())}(?!\w)", q):
                    candidates.append((col, value.item() if hasattr(value, "item") else value))
            # If the answer names a status semantically and a flag column has binary values,
            # only infer the conventional positive value when the column name supports it.
            name = col.casefold()
            term = next((term for term in ("active", "inactive", "eligible", "paid", "refunded", "cancelled", "canceled", "completed") if re.search(rf"\b{term}\b", q) and term in name), None)
            if term:
                unique = {str(value).strip().casefold() for value in values}
                positive = next((v for v in ("1", "1.0", "true", "yes", "y", term) if v in unique), None)
                if positive is not None:
                    original = next((value for value in values if str(value).strip().casefold() == positive), None)
                    candidates.append((col, original.item() if hasattr(original, "item") else original))
        # Deduplicate by column + normalized value.
        unique_candidates: dict[tuple[str, str], tuple[str, Any]] = {}
        for col, value in candidates:
            unique_candidates[(col, str(value).casefold())] = (col, value)
        candidates = list(unique_candidates.values())
        if len(candidates) != 1:
            raise ValueError("The answer does not identify one unique filter field/value. Use 'column = value' or choose a suggested filter.")
        col, value = candidates[0]
        op = "!=" if exclude else "="
        # Binary status values must respect the requested polarity: `active_flag != 1`
        # would select inactive rows, and vice versa.
        conditions = [{"column": col, "operator": op, "value": value}]
        return {"add_filter_conditions": conditions}

    raise ValueError(f"Freeform patching is not supported for the pending slot {slot!r}; choose a proposed option or clarify the field explicitly.")


def _question_requirements(question: str, contract: Any) -> list[dict[str, Any]]:
    """Find material language that is not represented by the current contract fields."""
    q = " ".join(str(question or "").split())
    ql = q.casefold()
    requirements: list[dict[str, Any]] = []
    # This contract version is a single-operation IR. Do not approve a compound utterance
    # merely because the single-operation compiler found one plausible clause. The repository
    # has a separate compound splitter, but until its nodes are persisted in the approved contract
    # graph, such questions must be split by the user rather than executed from raw text later.
    try:
        from packages.analytics_core.src.intelligence.complex_objective_planner import plan_compound_question
        compound = plan_compound_question(question)
        if compound.is_compound and not getattr(contract, "operation_graph", ()):
            requirements.append({
                "slot": "operation_graph", "status": "UNRESOLVED",
                "reason": "compound_question_not_bound_to_approved_operation_graph",
                "text": str(question),
            })
    except Exception:
        # Failure to determine compoundness must not crash intent resolution; the normal
        # per-slot completeness checks remain active. Release tests cover the known splitter.
        pass

    agg = str(getattr(contract, "requested_aggregation", None) or "").upper()
    aggregation_provenance = dict(getattr(contract, "slot_provenance", {}) or {}).get("requested_aggregation")
    aggregation_was_clarified = aggregation_provenance == "CLARIFIED"
    aggregation_patterns = (
        (r"\b(average|avg|mean)\b", {"MEAN", "AVERAGE"}, "aggregation"),
        (r"\bmedian\b", {"MEDIAN"}, "aggregation"),
        (r"\b(?:mode|most common|most frequent|commonest)\b", {"MODE"}, "aggregation"),
        (r"\b(?:interquartile range|iqr)\b", {"IQR"}, "aggregation"),
        (r"\b(?:first quartile|lower quartile|25th percentile|25th quantile|q1)\b", {"Q1", "PERCENTILE"}, "aggregation"),
        (r"\b(?:third quartile|upper quartile|75th percentile|75th quantile|q3)\b", {"Q3", "PERCENTILE"}, "aggregation"),
        (r"\b(?:range|spread between minimum and maximum)\b", {"RANGE"}, "aggregation"),
        (r"\b(?:missing values?|null values?|blank values?)\b", {"MISSING_COUNT"}, "aggregation"),
        (r"\b(sum|total|combined)\b", {"SUM"}, "aggregation"),
        (r"\b(minimum|min)\b", {"MIN"}, "aggregation"),
        (r"\b(maximum|max)\b", {"MAX"}, "aggregation"),
        (r"\b(rate|percentage|proportion|fraction)\b", {"RATE"}, "aggregation"),
        (r"\b(count|how many|number of)\b", {"COUNT", "COUNT_DISTINCT", "ROW_COUNT"}, "aggregation"),
    )
    for pattern, accepted, slot in aggregation_patterns:
        if re.search(pattern, ql) and agg and agg not in accepted and not aggregation_was_clarified:
            requirements.append({"slot": slot, "status": "UNRESOLVED", "reason": "requested_aggregation_not_preserved", "text": re.search(pattern, ql).group(0)})
            break
        if re.search(pattern, ql) and not agg:
            requirements.append({"slot": slot, "status": "UNRESOLVED", "reason": "requested_aggregation_not_resolved", "text": re.search(pattern, ql).group(0)})
            break

    time_range = getattr(contract, "time_range", None)
    time_column = getattr(contract, "time_column", None)
    explicit_temporal = bool(re.search(
        r"\b(last|past|previous|prior|this|current)\s+(?:\d+\s+)?(?:day|days|week|weeks|month|months|quarter|quarters|year|years)\b|\b(?:since|between|before|after)\b",
        ql,
    ))
    if time_range and time_range.get("type") == "unresolved":
        requirements.append({
            "slot": "time_range", "status": "UNRESOLVED",
            "reason": str(time_range.get("reason") or "temporal_scope_unresolved"),
            "text": str(time_range.get("expression") or "temporal scope"),
        })
    elif explicit_temporal and (not time_range or (time_range.get("type") != "all" and not time_column)):
        requirements.append({
            "slot": "time_column" if time_range else "time_range",
            "status": "UNRESOLVED",
            "reason": "requested_time_scope_not_preserved",
            "text": re.search(r"\b(?:last|past|previous|prior|this|current)\s+(?:\d+\s+)?(?:day|days|week|weeks|month|months|quarter|quarters|year|years)\b|\b(?:since|between|before|after)\b", ql).group(0),
        })

    if re.search(r"\b(?:exclude|excluding|without|ignore)\b", ql):
        if not any(str(item.get("operator", "")).upper() in {"!=", "NOT IN"} for item in (getattr(contract, "filter_conditions", ()) or ())):
            requirements.append({"slot": "filter_conditions", "status": "UNRESOLVED", "reason": "exclusion_not_preserved"})

    # "active customers only", "only active users", etc. changes the population and cannot be
    # treated as a narrative assumption. Unless an actual population filter exists, ask.
    if re.search(r"\b(?:active|inactive|eligible|paid|refunded|cancelled|canceled|completed)\s+(?:customers?|users?|accounts?|orders?|tickets?|subscriptions?)\s+only\b|\bonly\s+(?:active|inactive|eligible|paid|refunded|cancelled|canceled|completed)\s+(?:customers?|users?|accounts?|orders?|tickets?|subscriptions?)\b", ql):
        population_terms = re.findall(r"\b(active|inactive|eligible|paid|refunded|cancelled|canceled|completed)\b", ql)
        conditions = list(getattr(contract, "filter_conditions", ()) or ())

        def condition_represents_population(item: Mapping[str, Any], term: str) -> bool:
            operator = str(item.get("operator", "")).upper()
            if operator not in {"=", "!="}:
                return False
            column_name = str(item.get("column", "")).casefold()
            value = item.get("value")
            value_text = str(value).strip().casefold()
            # Explicit categorical labels are strong evidence; do not mistake a numeric code
            # in an unrelated column for a semantic status label.
            if value_text == term:
                return operator == "="
            related_name = term in column_name
            inverse_active = term == "inactive" and "active" in column_name and "inactive" not in column_name
            inverse_inactive = term == "active" and "inactive" in column_name
            if not related_name and not inverse_active and not inverse_inactive:
                return False
            truthy = value is True or value_text in {"1", "1.0", "true", "yes", "y"}
            falsy = value is False or value_text in {"0", "0.0", "false", "no", "n"}
            if inverse_active or inverse_inactive:
                term_matches_value = falsy if term == "inactive" or inverse_inactive else truthy
            elif term == "inactive" and "inactive" in column_name:
                term_matches_value = truthy
            else:
                term_matches_value = truthy
            if operator == "!=":
                term_matches_value = not term_matches_value
            return term_matches_value

        represented = any(condition_represents_population(item, term) for item in conditions for term in population_terms)
        if not represented:
            requirements.append({"slot": "filter_conditions", "status": "UNRESOLVED", "reason": "population_qualifier_not_preserved"})

    if re.search(r"\bper\s+(?:active\s+)?(?:customer|user|account|order|ticket|unit|employee|patient|product)\b", ql):
        if not getattr(contract, "denominator_column", None):
            requirements.append({"slot": "denominator_column", "status": "UNRESOLVED", "reason": "derived_metric_denominator_not_resolved"})
        else:
            derived_evidence = dict(getattr(contract, "resolution_evidence", {}) or {}).get("derived_metric_patch", {})
            if derived_evidence.get("unsupported_until_compiled") or not getattr(contract, "numerator_column", None):
                requirements.append({
                    "slot": "derived_metric_execution", "status": "UNRESOLVED",
                    "reason": "ratio_operator_not_compiled_for_deterministic_execution",
                })

    if re.search(r"\b(?:chart|plot|visuali[sz]e)\b", ql):
        outputs = set(getattr(contract, "requested_outputs", ()) or ())
        if not outputs.intersection({"chart", "plot", "visualization"}):
            requirements.append({"slot": "requested_outputs", "status": "UNRESOLVED", "reason": "requested_visual_output_not_preserved"})

    return requirements


def enrich_contract_intent_state(
    question: str,
    contract: Any,
    frame: Optional[pd.DataFrame],
    guidance: Optional[Mapping[str, Any]] = None,
) -> tuple[Any, list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Return contract with provenance, slot confidence and material-clause ledger."""
    from dataclasses import replace
    if contract is None:
        return contract, [], [], []
    guidance = dict(guidance or {})
    values = contract.to_dict() if hasattr(contract, "to_dict") else {}
    explicit_cols = set()
    if frame is not None:
        for column in frame.columns:
            col = str(column)
            if re.search(rf"(?<![\w]){re.escape(col)}(?![\w])", str(question), re.I):
                explicit_cols.add(col)
    prior_provenance = dict(getattr(contract, "slot_provenance", {}) or {})
    provenance: Dict[str, str] = {}
    confidence: Dict[str, float] = {}
    material_slots = {
        "task_family": getattr(contract, "task_family", None),
        "estimand": getattr(contract, "estimand", None),
        "target_column": getattr(contract, "target_column", None),
        "numerator_column": getattr(contract, "numerator_column", None),
        "denominator_column": getattr(contract, "denominator_column", None),
        "explanatory_columns": list(getattr(contract, "explanatory_columns", ()) or ()),
        "grouping_columns": list(getattr(contract, "grouping_columns", ()) or ()),
        "requested_aggregation": getattr(contract, "requested_aggregation", None),
        "time_column": getattr(contract, "time_column", None),
        "time_range": getattr(contract, "time_range", None),
        "filter_conditions": list(getattr(contract, "filter_conditions", ()) or ()),
        "ranking_direction": getattr(contract, "ranking_direction", None),
        "comparison_operator": getattr(contract, "comparison_operator", None),
        "comparison_values": list(getattr(contract, "comparison_values", ()) or ()),
        "requested_outputs": list(getattr(contract, "requested_outputs", ()) or ()),
        "share_of_total": bool(getattr(contract, "share_of_total", False)),
        "claim_type": getattr(contract, "claim_type", None),
    }
    guidance_slot_map = {
        "target_column": "selected_metric", "requested_aggregation": "selected_aggregation",
        "grouping_columns": "selected_group_dimension", "time_column": "selected_time_column",
        "filter_conditions": "scope_filter", "time_range": "intent_patch", "denominator_column": "intent_patch",
    }
    typed_patch = guidance.get("_canonical_intent_patch") or guidance.get("intent_patch") or {}
    typed_patch = dict(typed_patch) if isinstance(typed_patch, Mapping) else {}
    explicit_aggregation = bool(re.search(r"\b(sum|total|average|avg|mean|median|minimum|min|maximum|max|count|rate|percentage|proportion)\b", str(question), re.I))
    explicit_rank_direction = bool(re.search(r"\b(highest|largest|greatest|top|lowest|smallest|ascending|descending)\b", str(question), re.I))
    for slot, value in material_slots.items():
        source_key = guidance_slot_map.get(slot)
        patch_field = {
            "target_column": "target_column", "requested_aggregation": "requested_aggregation",
            "grouping_columns": "grouping_columns", "time_column": "time_column",
            "time_range": "time_range", "filter_conditions": "filter_conditions",
            "denominator_column": "denominator_column", "ranking_direction": "ranking_direction",
        }.get(slot)
        if (patch_field and patch_field in typed_patch) or (source_key and guidance.get(source_key) not in _GUIDANCE_EMPTY):
            source = "CLARIFIED"
            conf = 1.0
        elif slot == "task_family" and value and re.search(
            r"\b(rank|ranked|ranking|compare|comparison|correlat\w*|trend|over time|forecast|predict|count|how many|cluster)\b",
            str(question), re.I,
        ):
            source = "USER_EXPLICIT"
            conf = 1.0
        elif slot == "claim_type" and value and re.search(r"\b(cause|causal|impact|effect|association|correlat\w*)\b", str(question), re.I):
            source = "USER_EXPLICIT"
            conf = 1.0
        elif slot == "estimand" and value and explicit_aggregation:
            source = "USER_EXPLICIT"
            conf = 1.0
        elif slot == "requested_outputs" and value and re.search(r"\b(chart|plot|visuali[sz]e|table)\b", str(question), re.I):
            source = "USER_EXPLICIT"
            conf = 1.0
        elif slot == "target_column" and value in explicit_cols:
            source = "USER_EXPLICIT"
            conf = 1.0
        elif slot == "grouping_columns" and value and all(v in explicit_cols for v in value):
            source = "USER_EXPLICIT"
            conf = 1.0
        elif slot == "time_column" and value in explicit_cols:
            source = "USER_EXPLICIT"
            conf = 1.0
        elif slot == "requested_aggregation" and value and explicit_aggregation:
            source = "USER_EXPLICIT"
            conf = 1.0
        elif slot == "ranking_direction" and value and explicit_rank_direction:
            source = "USER_EXPLICIT"
            conf = 1.0
        elif slot == "filter_conditions" and value and re.search(r"\b(where|only|excluding|exclude|without|except|whose|with)\b", str(question), re.I):
            source = "USER_EXPLICIT"
            conf = 1.0
        elif slot == "time_range" and value and re.search(r"\b(last|past|previous|prior|this|current|since|between|before|after|q[1-4])\b", str(question), re.I):
            source = "USER_EXPLICIT"
            conf = 1.0
        elif value is not None and prior_provenance.get(slot):
            source = prior_provenance[slot]
            conf = float((getattr(contract, "slot_confidence", {}) or {}).get(slot, 0.85))
        elif value is not None:
            source = "INFERRED_DEFAULT"
            conf = 0.75
        else:
            source = "UNRESOLVED"
            conf = 0.0
        provenance[slot] = source
        confidence[slot] = max(0.0, min(1.0, conf))

    current_range = getattr(contract, "time_range", None)
    contract_time_column = getattr(contract, "time_column", None)
    if current_range is None:
        parsed_range = _relative_time_range(question)
        if parsed_range is not None:
            current_range = parsed_range
            provenance["time_range"] = "USER_EXPLICIT"
            confidence["time_range"] = 1.0

    contract_filters = list(getattr(contract, "filter_conditions", ()) or ())
    temporal = _temporal_columns(frame) if frame is not None else []
    if contract_time_column and current_range and current_range.get("source") == "relative_question_window":
        rebound_range = _relative_time_range(question, frame, contract_time_column)
        if rebound_range is not None:
            current_range = rebound_range
            provenance["time_range"] = "USER_EXPLICIT"
            confidence["time_range"] = 1.0
    inferred_time_column = False
    if current_range and not contract_time_column:
        if len(temporal) == 1:
            contract_time_column = temporal[0]
            inferred_time_column = True
            provenance["time_column"] = "INFERRED_DEFAULT"
            confidence["time_column"] = 0.75
        elif len(temporal) > 1:
            provenance["time_column"] = "UNRESOLVED"
            confidence["time_column"] = 0.0

    if contract_time_column and current_range and current_range.get("source") == "relative_question_window":
        # When a previous turn stored a draft window before the user chose among multiple time
        # columns, recalculate that relative phrase against the now-selected dataset/time field.
        rebound_range = _relative_time_range(question, frame, contract_time_column)
        if rebound_range is not None:
            current_range = rebound_range
            provenance["time_range"] = "USER_EXPLICIT"
            confidence["time_range"] = 1.0

    # When a relative period is understood and a unique temporal field is evidenced, compile it
    # into supported deterministic population predicates. The exact boundaries are stored in
    # the contract and hash; downstream execution does not reinterpret relative dates.
    if current_range and contract_time_column and current_range.get("start") and current_range.get("end"):
        contract_filters = [
            item for item in contract_filters
            if not (item.get("column") == contract_time_column and str(item.get("operator", "")).upper() in {">", ">=", "<", "<="})
        ]
        end_operator = "<" if current_range.get("end_exclusive") else "<="
        contract_filters.extend([
            {"column": contract_time_column, "operator": ">=", "value": current_range["start"]},
            {"column": contract_time_column, "operator": end_operator, "value": current_range["end"]},
        ])

    from dataclasses import replace
    probe_contract = replace(
        contract,
        time_column=contract_time_column,
        time_range=current_range,
        filter_conditions=tuple(contract_filters),
        slot_provenance=provenance,
    )
    requirements = _question_requirements(question, probe_contract)
    if current_range and frame is not None and not contract_time_column and not temporal:
        requirements = [r for r in requirements if r.get("slot") != "time_range"]
        requirements.append({"slot": "time_column", "status": "UNRESOLVED", "reason": "no_temporal_column_available"})
    if current_range and frame is not None and not contract_time_column and len(temporal) > 1:
        requirements = [r for r in requirements if r.get("slot") != "time_range"]
        if not any(r.get("slot") == "time_column" for r in requirements):
            requirements.append({"slot": "time_column", "status": "UNRESOLVED", "reason": "multiple_time_columns"})

    material_slots["time_column"] = contract_time_column
    material_slots["time_range"] = current_range
    material_slots["filter_conditions"] = list(contract_filters)

    ledger = []
    for slot, value in material_slots.items():
        ledger.append({
            "slot": slot,
            "status": "RESOLVED" if value is not None and provenance.get(slot) != "UNRESOLVED" else "OPEN",
            "provenance": provenance.get(slot, "UNRESOLVED"),
            "confidence": confidence.get(slot, 0.0),
        })
    for requirement in requirements:
        ledger.append({**requirement, "status": "UNRESOLVED"})

    # Material unresolved requirements become part of the contract and therefore the approval
    # hash. They never disappear as display-only warnings.
    existing_unresolved = list(getattr(contract, "unresolved_semantics", ()) or ())
    old_ledger_entries = [item for item in existing_unresolved if str(item).startswith("intent_ledger:")]
    old_ledger_slots = {str(item).split(":", 2)[1] for item in old_ledger_entries if len(str(item).split(":", 2)) > 1}
    # Rebuild this module's derived ledger on every patch instead of making resolved slots
    # permanently unresolved after a prior clarification.
    existing_unresolved = [item for item in existing_unresolved if not str(item).startswith("intent_ledger:")]
    new_unresolved = [f"intent_ledger:{r.get('slot')}:{r.get('reason')}" for r in requirements]
    unresolved_semantics = tuple(dict.fromkeys([*existing_unresolved, *new_unresolved]))
    unresolved_roles = [
        role for role in (getattr(contract, "unresolved_roles", ()) or ())
        if str(role) not in old_ledger_slots
    ]
    unresolved_roles.extend(r.get("slot", "intent") for r in requirements if r.get("slot") not in unresolved_roles)
    completeness = getattr(contract, "completeness", "NOT_ENFORCED")
    if requirements or unresolved_roles:
        completeness = "INCOMPLETE"
    elif completeness == "INCOMPLETE":
        # Only promote after both the current ledger and the contract's own unresolved roles
        # have cleared. Existing NOT_ENFORCED contracts remain NOT_ENFORCED.
        completeness = "COMPLETE"
    changes: Dict[str, Any] = {
        "slot_provenance": provenance,
        "slot_confidence": confidence,
        "intent_ledger": tuple(ledger),
        "unresolved_semantics": unresolved_semantics,
        "unresolved_roles": tuple(unresolved_roles),
        "completeness": completeness,
        "resolution_confidence": min(float(getattr(contract, "resolution_confidence", 1.0)), 0.25) if requirements else getattr(contract, "resolution_confidence", 1.0),
    }
    changes["time_column"] = contract_time_column
    changes["filter_conditions"] = tuple(contract_filters)
    if current_range is not None:
        changes["time_range"] = current_range
    updated = replace(contract, **changes)
    options: list[dict[str, Any]] = []
    time_column_missing = any(r.get("slot") == "time_column" for r in requirements)
    if time_column_missing and frame is not None:
        for column in temporal:
            options.append({"label": f"Use {column} as the time field", "patch": {"set": {"time_column": column}}})
    filter_requirement = any(r.get("slot") == "filter_conditions" for r in requirements)
    if filter_requirement and frame is not None:
        ql = str(question).casefold()
        qualifiers = [term for term in ("active", "inactive", "eligible", "paid", "refunded", "cancelled", "canceled", "completed") if re.search(rf"\b{term}\b", ql)]
        for column in frame.columns:
            name = str(column).casefold()
            if not qualifiers or not any(term in name for term in qualifiers):
                continue
            series = frame[column]
            unique = {str(v).strip().casefold() for v in series.dropna().unique()}
            positive_candidates = [
                v for v in ("1", "1.0", "true", "yes", "y", "active", "paid", "eligible", "completed", "refunded")
                if v in unique
            ]
            if len(unique) <= 2 and positive_candidates:
                literal: Any = next(v for v in series.dropna().unique() if str(v).strip().casefold() == positive_candidates[0])
                if hasattr(literal, "item"):
                    literal = literal.item()
                options.append({
                    "label": f"Filter {column} = {literal}",
                    "patch": {"set": {"add_filter_conditions": [{"column": str(column), "operator": "=", "value": literal}]}},
                })
    return updated, ledger, requirements, options


def apply_contract_intent_patch(contract: Any, patch: Mapping[str, Any], frame: pd.DataFrame, question: str) -> Any:
    """Apply already-normalized slot assignments to a contract with schema validation.

    Only material fields supported by the current executor are accepted. Derived denominator
    assignment is stored as intent, but it does not itself make a ratio executable; the intent
    ledger continues to block the plan until the runtime has a verified derived-metric operator.
    """
    from dataclasses import replace
    values = dict(patch)
    columns = {str(column) for column in frame.columns}
    updates: Dict[str, Any] = {}
    for field in ("target_column", "numerator_column", "denominator_column", "time_column"):
        if field in values:
            value = values.pop(field)
            if value is not None and str(value) not in columns:
                raise ValueError(f"Intent patch references unavailable column {value!r} for {field}.")
            if field == "time_column" and value is not None and str(value) not in _temporal_columns(frame):
                raise ValueError(f"Column {value!r} is not supported as a temporal field for the requested time filter.")
            updates[field] = str(value) if value is not None else None
    if "requested_aggregation" in values:
        agg = str(values.pop("requested_aggregation")).strip().upper()
        if agg == "AVG":
            agg = "MEAN"
        allowed = {"SUM", "MEAN", "AVERAGE", "MEDIAN", "MIN", "MAX", "COUNT", "COUNT_DISTINCT", "ROW_COUNT", "RATE", "CORRELATION"}
        if agg not in allowed:
            raise ValueError(f"Unsupported requested aggregation in intent patch: {agg}.")
        updates["requested_aggregation"] = "MEAN" if agg == "AVERAGE" else agg
    if "grouping_columns" in values:
        raw = values.pop("grouping_columns")
        groups = [raw] if isinstance(raw, str) else list(raw or [])
        if len(groups) > 1:
            raise ValueError("The current deterministic executor supports one grouping column per contract patch.")
        if any(str(group) not in columns for group in groups):
            raise ValueError("Intent patch references an unavailable grouping column.")
        updates["grouping_columns"] = tuple(str(group) for group in groups)
    if "ranking_direction" in values:
        direction = str(values.pop("ranking_direction")).upper()
        if direction not in {"ASC", "DESC"}:
            raise ValueError("ranking_direction must be ASC or DESC.")
        updates["ranking_direction"] = direction
    if "filter_conditions" in values:
        raw_filters = values.pop("filter_conditions")
        if not isinstance(raw_filters, (list, tuple)):
            raise ValueError("filter_conditions intent patch must be a list.")
        allowed_ops = {"=", "!=", "IN", "NOT IN", "IS NULL", "IS NOT NULL", ">", ">=", "<", "<="}
        clean = []
        for item in raw_filters:
            if not isinstance(item, Mapping):
                raise ValueError("Each filter condition must be an object.")
            col = str(item.get("column", ""))
            op = str(item.get("operator", "")).upper()
            if col not in columns or op not in allowed_ops:
                raise ValueError(f"Invalid filter condition for column/operator: {col!r}/{op!r}.")
            if op not in {"IS NULL", "IS NOT NULL"} and "value" not in item:
                raise ValueError(f"Filter operator {op!r} requires a value.")
            condition = {"column": col, "operator": op}
            if op not in {"IS NULL", "IS NOT NULL"}:
                value = item.get("value")
                if op in {"IN", "NOT IN"}:
                    if not isinstance(value, (list, tuple)) or not value:
                        raise ValueError(f"Filter operator {op!r} requires a non-empty list of values.")
                    if any(not isinstance(item_value, (str, int, float, bool)) and item_value is not None for item_value in value):
                        raise ValueError("Filter values must be JSON scalars.")
                    value = list(value)
                elif not isinstance(value, (str, int, float, bool)) and value is not None:
                    raise ValueError("Filter values must be JSON scalars.")
                condition["value"] = value
            clean.append(condition)
        updates["filter_conditions"] = tuple(clean)
    if "add_filter_conditions" in values:
        raw_filters = values.pop("add_filter_conditions")
        if not isinstance(raw_filters, (list, tuple)):
            raise ValueError("add_filter_conditions intent patch must be a list.")
        allowed_ops = {"=", "!=", "IN", "NOT IN", "IS NULL", "IS NOT NULL", ">", ">=", "<", "<="}
        existing = list(updates.get("filter_conditions", getattr(contract, "filter_conditions", ()) or ()))
        for item in raw_filters:
            if not isinstance(item, Mapping):
                raise ValueError("Each added filter condition must be an object.")
            col = str(item.get("column", ""))
            op = str(item.get("operator", "")).upper()
            if col not in columns or op not in allowed_ops:
                raise ValueError(f"Invalid filter condition for column/operator: {col!r}/{op!r}.")
            if op not in {"IS NULL", "IS NOT NULL"} and "value" not in item:
                raise ValueError(f"Filter operator {op!r} requires a value.")
            value = item.get("value")
            if op in {"IN", "NOT IN"}:
                if not isinstance(value, (list, tuple)) or not value:
                    raise ValueError(f"Filter operator {op!r} requires a non-empty list of values.")
                value = [v.item() if hasattr(v, "item") else v for v in value]
                if any(not isinstance(v, (str, int, float, bool)) and v is not None for v in value):
                    raise ValueError("Filter values must be JSON scalars.")
            elif op not in {"IS NULL", "IS NOT NULL"} and not isinstance(value, (str, int, float, bool)) and value is not None:
                if hasattr(value, "item"):
                    value = value.item()
                else:
                    raise ValueError("Filter values must be JSON scalars.")
            condition = {"column": col, "operator": op}
            if op not in {"IS NULL", "IS NOT NULL"}:
                condition["value"] = value
            for prior in existing:
                if str(prior.get("column")) != col:
                    continue
                prior_op = str(prior.get("operator", "")).upper()
                prior_value = prior.get("value")
                same_literal = str(prior_value).casefold() == str(value).casefold()
                contradictory = (
                    (prior_op == "=" and op == "=" and not same_literal)
                    or (prior_op == "=" and op == "!=" and same_literal)
                    or (prior_op == "!=" and op == "=" and same_literal)
                )
                if contradictory:
                    raise ValueError(f"Clarification conflicts with an existing filter on {col!r}; review the population conditions.")
            if condition not in existing:
                existing.append(condition)
        updates["filter_conditions"] = tuple(existing)
    if "time_range" in values:
        time_range = values.pop("time_range")
        if not isinstance(time_range, Mapping):
            raise ValueError("time_range intent patch must be an object.")
        time_range = {str(k): v for k, v in time_range.items()}
        if str(time_range.get("type", "")).casefold() == "all":
            updates["time_range"] = {"type": "all", "expression": str(time_range.get("expression", "all data")), "source": "clarified"}
            active_time_col = updates.get("time_column", getattr(contract, "time_column", None))
            existing_filters = list(updates.get("filter_conditions", getattr(contract, "filter_conditions", ()) or ()))
            if active_time_col:
                existing_filters = [item for item in existing_filters if not (
                    item.get("column") == active_time_col
                    and str(item.get("operator", "")).upper() in {">", ">=", "<", "<="}
                )]
                updates["filter_conditions"] = tuple(existing_filters)
        else:
            start, end = time_range.get("start"), time_range.get("end")
            if not start or not end:
                raise ValueError("time_range requires explicit 'start' and 'end' boundaries.")
            start_dt, end_dt = pd.to_datetime(start, errors="coerce", utc=True), pd.to_datetime(end, errors="coerce", utc=True)
            if pd.isna(start_dt) or pd.isna(end_dt) or start_dt >= end_dt:
                raise ValueError("time_range boundaries must be valid dates with start < end.")
            time_range["start"] = start_dt.date().isoformat() if len(str(start)) == 10 else start_dt.isoformat()
            time_range["end"] = end_dt.date().isoformat() if len(str(end)) == 10 else end_dt.isoformat()
            time_range.setdefault("type", "absolute")
            updates["time_range"] = time_range
    if values:
        raise ValueError(f"Unsupported intent patch fields: {', '.join(sorted(values))}.")

    if updates.get("time_range") is not None and str(updates["time_range"].get("type", "")).casefold() != "all":
        range_field = updates.get("time_column", getattr(contract, "time_column", None))
        if not range_field:
            temporal = _temporal_columns(frame)
            if len(temporal) == 1:
                range_field = temporal[0]
                updates["time_column"] = range_field
            else:
                raise ValueError("A time column must be selected before applying a time-range patch.")
        if range_field not in columns:
            raise ValueError(f"Time range column {range_field!r} is not present in the dataset.")
        tr = updates["time_range"]
        existing = list(updates.get("filter_conditions", getattr(contract, "filter_conditions", ()) or ()))
        # Replace prior bounds on this column but retain unrelated population filters.
        existing = [x for x in existing if not (x.get("column") == range_field and str(x.get("operator", "")).upper() in {">", ">=", "<", "<="})]
        end_operator = "<" if tr.get("end_exclusive") else "<="
        existing.extend([
            {"column": range_field, "operator": ">=", "value": tr["start"]},
            {"column": range_field, "operator": end_operator, "value": tr["end"]},
        ])
        updates["filter_conditions"] = tuple(existing)
    if "denominator_column" in updates and updates["denominator_column"]:
        # Keep derived-metric semantics explicit. The current simple aggregate executor does not
        # divide numerator by a denominator; the completeness ledger prevents execution.
        updates["resolution_evidence"] = {
            **dict(getattr(contract, "resolution_evidence", {}) or {}),
            "derived_metric_patch": {
                "numerator_column": updates.get("numerator_column", getattr(contract, "numerator_column", None)),
                "denominator_column": updates["denominator_column"],
                "unsupported_until_compiled": True,
            },
        }
    updates["slot_provenance"] = {
        **dict(getattr(contract, "slot_provenance", {}) or {}),
        **{key: "CLARIFIED" for key in updates if key not in {"resolution_evidence"}},
    }
    updates["slot_confidence"] = {
        **dict(getattr(contract, "slot_confidence", {}) or {}),
        **{key: 1.0 for key in updates if key not in {"resolution_evidence"}},
    }
    return replace(contract, **updates)


def filter_condition_mask(series: pd.Series, operator: str, value: Any = None) -> pd.Series:
    """Evaluate a typed population predicate without stringifying ordered comparisons.

    Equality remains case-insensitive for compatibility. Ordered operators are permitted only
    for numeric or parseable temporal values; unsupported comparisons raise and callers fail closed.
    """
    op = str(operator or "").upper()
    if op == "IS NULL":
        return series.isna()
    if op == "IS NOT NULL":
        return series.notna()
    def equality_mask(values: list[Any]) -> pd.Series:
        non_null_values = [item for item in values if item is not None]
        if pd.api.types.is_bool_dtype(series):
            normalized = {str(item).casefold() in {"true", "1", "yes", "y"} for item in non_null_values}
            return series.notna() & series.astype("boolean").isin(normalized)
        if pd.api.types.is_numeric_dtype(series):
            numeric_values = pd.to_numeric(pd.Series(non_null_values), errors="coerce").dropna().tolist()
            left = pd.to_numeric(series, errors="coerce")
            return series.notna() & left.isin(numeric_values)
        allowed = {str(item).casefold() for item in non_null_values}
        return series.notna() & series.astype(str).str.casefold().isin(allowed)

    if op == "=":
        if value is None:
            return series.isna()
        return equality_mask([value])
    if op == "!=":
        if value is None:
            return series.notna()
        return series.notna() & ~equality_mask([value])
    if op in {"IN", "NOT IN"}:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"Filter operator {op!r} requires a non-empty list of values.")
        matches = equality_mask(list(value))
        return matches if op == "IN" else (series.notna() & ~matches)
    if op not in {">", ">=", "<", "<="}:
        raise ValueError(f"Unsupported filter operator {op!r}.")

    if pd.api.types.is_numeric_dtype(series):
        threshold = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(threshold):
            raise ValueError(f"Ordered filter value {value!r} is not numeric for this field.")
        left = pd.to_numeric(series, errors="coerce")
    else:
        # Ordered comparisons on text are accepted only when the column and bound can be treated
        # as dates consistently. We never fall back to lexical string ordering.
        left_dt = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
        right_dt = pd.to_datetime(value, errors="coerce", utc=True)
        nonnull = int(series.notna().sum())
        valid = int(left_dt.notna().sum())
        if pd.isna(right_dt) or nonnull == 0 or valid / nonnull < 0.8:
            raise ValueError(f"Ordered filter requires a numeric or temporal column; cannot compare {value!r} safely.")
        left, threshold = left_dt, right_dt
    if op == ">":
        return left > threshold
    if op == ">=":
        return left >= threshold
    if op == "<":
        return left < threshold
    return left <= threshold
