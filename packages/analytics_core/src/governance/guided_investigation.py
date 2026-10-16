"""Guided Investigation Engine: Pre-analysis clarification and transparent human-in-the-loop governance.

Why this exists (MindEd AA-OS 1.0.0 Product Contract):
MindEd should not treat human involvement as a post-analysis approval rubber stamp.
It involves the analyst at the decisions that shape the analysis, while still allowing
routine, unambiguous work to proceed automatically.

Core Workflow:
1. Understand the question: Identify target metric, population, filters, time period, desired output.
2. Show the proposed interpretation: Display what was understood, columns to use, planned calculation, and assumptions.
3. Involve human judgment when consequential:
   - Clear question & unambiguous columns -> Auto-proceed with option to edit.
   - Multiple plausible metrics -> Ask which metric is intended (e.g. gross vs net revenue).
   - Missing required data -> Explain what is missing; do not invent a substitute; offer defensible alternatives.
   - Ambiguous dates or filters -> Request clarification when choice alters result.
   - Potential join fanout -> Explain record duplication risk and ask for aggregation choice.
   - Causal question on observational data -> Explain association vs causation; offer defensible alternative.
   - Independent verification fails -> Halt claim; report as unverified / inconclusive.
4. Investigate and calculate: Execute verified DuckDB calculation.
5. Review findings together: Accept for workflow, request correction, reject with reason, or redo with revised parameters.
   INVARIANT: Human approval never overrides failed verification or turns an unsupported claim into verified.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple
import pandas as pd
from packages.analytics_core.src.intelligence.canonical_question_contract import canonical_json_dumps
from packages.analytics_core.src.intelligence.intent_state import (
    apply_contract_intent_patch,
    enrich_contract_intent_state,
    infer_intent_patch_from_answer,
    normalize_guidance_payload,
    normalize_intent_patch,
)


class ResolverStatus(str, Enum):
    RESOLVED = "Resolved"
    NEEDS_CLARIFICATION = "NeedsClarification"
    CONFLICT = "Conflict"
    UNSUPPORTED = "Unsupported"


@dataclass(frozen=True)
class GuidanceConflict:
    field: str
    structured_value: Any
    freeform_value: Any
    structured_instruction: str
    freeform_instruction: str


@dataclass(frozen=True)
class ResolverOutcome:
    status: ResolverStatus
    contract: Any = None
    unresolved_instruction: Optional[str] = None
    reason: Optional[str] = None
    suggested_resolutions: Tuple[str, ...] = ()
    schema_information: Dict[str, Any] = field(default_factory=dict)
    conflict: Optional[GuidanceConflict] = None
    original_instruction: Optional[str] = None
    suggested_patches: Tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "contract": self.contract.to_dict() if self.contract is not None else None,
            "unresolved_instruction": self.unresolved_instruction,
            "reason": self.reason,
            "suggested_resolutions": list(self.suggested_resolutions),
            "schema_information": dict(self.schema_information),
            "conflict": (
                {
                    "field": self.conflict.field,
                    "structured_value": self.conflict.structured_value,
                    "freeform_value": self.conflict.freeform_value,
                    "structured_instruction": self.conflict.structured_instruction,
                    "freeform_instruction": self.conflict.freeform_instruction,
                }
                if self.conflict
                else None
            ),
            "original_instruction": self.original_instruction,
            "suggested_patches": [dict(item) for item in self.suggested_patches],
        }


class AmbiguityStatus(str, Enum):
    CLEAR = "CLEAR"
    AMBIGUOUS_METRIC = "AMBIGUOUS_METRIC"
    MISSING_DATA = "MISSING_DATA"
    AMBIGUOUS_SCOPE = "AMBIGUOUS_SCOPE"
    JOIN_FANOUT_RISK = "JOIN_FANOUT_RISK"
    CAUSAL_ON_OBSERVATIONAL = "CAUSAL_ON_OBSERVATIONAL"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    UNEXPECTED_ASSUMPTION = "UNEXPECTED_ASSUMPTION"


class HumanReviewAction(str, Enum):
    ACCEPT = "ACCEPT"
    REQUEST_CORRECTION = "REQUEST_CORRECTION"
    REJECT = "REJECT"
    REDO = "REDO"


import hashlib
import json


def compute_plan_hash(
    contract: Any,
    guidance: Optional[Mapping[str, Any]] = None,
    dataset_fingerprint: Optional[str] = None,
    schema_version: int = 1,
) -> str:
    """Deterministically compute SHA-256 plan hash binding contract, guidance, dataset fingerprint, and version.

    Uses canonical JSON serialization (sorted keys, compact separators) so equivalent inputs produce identical hashes.
    """
    contract_dict = contract.to_dict() if hasattr(contract, "to_dict") else dict(contract or {})
    # Canonicalize guidance: only keep non-empty scalar/list entries, sorted
    canonical_guidance: Dict[str, Any] = {}
    if guidance:
        for k in sorted(guidance.keys()):
            v = guidance[k]
            if v not in (None, "", [], {}):
                if isinstance(v, list):
                    canonical_guidance[k] = list(v)
                elif isinstance(v, dict):
                    canonical_guidance[k] = {dk: v[dk] for dk in sorted(v.keys())}
                else:
                    canonical_guidance[k] = v

    canonical_payload = {
        "contract": contract_dict,
        "guidance": canonical_guidance,
        "dataset_fingerprint": str(dataset_fingerprint or ""),
        "schema_version": int(schema_version),
    }
    canonical_json = canonical_json_dumps(canonical_payload)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_resolved_plan_hash(
    contract: Any,
    dataset_fingerprint: Optional[str] = None,
    schema_version: int = 1,
) -> str:
    """Hash the *resolved* plan: the approved contract plus dataset identity.

    This is the single approval hash used by ``/interpret``, investigation creation and
    guidance revision. Human guidance is an *input* to resolution; its effect is already
    captured by the resolved contract's material fields, so guidance is not hashed
    separately. ``resolution_evidence`` is provenance (it echoes the submitted guidance and
    free text), not a material operation parameter, and is excluded so that the same resolved
    plan always yields the same hash regardless of how it was reached.
    """
    contract_dict = dict(contract.to_dict() if hasattr(contract, "to_dict") else (contract or {}))
    contract_dict.pop("resolution_evidence", None)
    # Provenance helps explain the plan but does not change its material identity. The intent
    # ledger and every material slot remain inside the hash. Never fall back to default=str.
    contract_dict.pop("slot_provenance", None)
    contract_dict.pop("slot_confidence", None)
    if isinstance(contract_dict.get("intent_ledger"), list):
        contract_dict["intent_ledger"] = [
            {key: value for key, value in item.items() if key not in {"provenance", "confidence"}}
            for item in contract_dict["intent_ledger"]
        ]
    canonical_json = canonical_json_dumps(
        {
            "contract": contract_dict,
            "dataset_fingerprint": str(dataset_fingerprint or ""),
            "schema_version": int(schema_version),
        }
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def bind_default_aggregation(contract: Any, default: str = "SUM") -> Any:
    """Make a defaulted aggregation an explicit, hashed part of the contract.

    The preview shows ``SUM(metric)`` for ranking and metric-only descriptive questions. That
    default must live in the approved contract (and thus the plan hash and the controller's
    aggregation check), not only in preview text. The decision is recorded in
    ``resolution_evidence['defaulted_aggregation']`` so it stays visible as a default.
    """
    if contract is None:
        return contract
    task = getattr(contract, "task_family", None)
    target = getattr(contract, "target_column", None)
    groups = getattr(contract, "grouping_columns", ())
    applies = bool(target) and (
        (task == "RANKING" and bool(groups)) or task == "DESCRIPTIVE"
    )
    if not applies:
        return contract

    current = getattr(contract, "requested_aggregation", None)
    if current not in (None, default):
        return contract

    # The compiler may already place SUM in requested_aggregation as its inferred
    # default. Record that default explicitly too, but never relabel an aggregation
    # that the question stated directly.
    question_text = str(getattr(contract, "question_text", "") or "")
    explicit_aggregation = bool(re.search(
        r"\b(sum|total|average|avg|mean|median|minimum|min|maximum|max|count|rate|percentage|proportion)\b",
        question_text,
        re.I,
    ))
    if current == default and explicit_aggregation:
        return contract

    from dataclasses import replace
    return replace(
        contract,
        requested_aggregation=current or default,
        resolution_evidence={
            **dict(getattr(contract, "resolution_evidence", {}) or {}),
            "defaulted_aggregation": default,
        },
    )


@dataclass
class AnalysisPlanPreview:
    question: str
    task: str
    proposed_interpretation: str
    planned_calculation: str
    target_metric: Optional[str] = None
    group_dimension: Optional[str] = None
    time_period: Optional[str] = None
    scope: str = "All available records"
    assumptions: List[str] = field(default_factory=list)
    ambiguity_status: AmbiguityStatus = AmbiguityStatus.CLEAR
    clarification_prompt: Optional[str] = None
    clarification_options: List[str] = field(default_factory=list)
    can_auto_proceed: bool = True
    recommended_action: str = "Proceed with calculation"
    # Canonical contract binding (Milestone M1)
    canonical_contract: Optional[Any] = None
    plan_hash: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_fingerprint: Optional[str] = None
    clarification_option_patches: List[Dict[str, Any]] = field(default_factory=list)
    pending_intent_slot: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "task": self.task,
            "proposed_interpretation": self.proposed_interpretation,
            "planned_calculation": self.planned_calculation,
            "target_metric": self.target_metric,
            "group_dimension": self.group_dimension,
            "time_period": self.time_period,
            "scope": self.scope,
            "assumptions": list(self.assumptions),
            "ambiguity_status": self.ambiguity_status.value if isinstance(self.ambiguity_status, AmbiguityStatus) else str(self.ambiguity_status),
            "clarification_prompt": self.clarification_prompt,
            "clarification_options": list(self.clarification_options),
            "can_auto_proceed": bool(self.can_auto_proceed),
            "recommended_action": self.recommended_action,
            "canonical_contract": self.canonical_contract.to_dict() if hasattr(self.canonical_contract, "to_dict") else None,
            "plan_hash": self.plan_hash,
            "dataset_name": self.dataset_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "clarification_option_patches": [dict(item) for item in self.clarification_option_patches],
            "pending_intent_slot": self.pending_intent_slot,
        }


# Backwards compatibility alias
ProposedInterpretation = AnalysisPlanPreview


_GUIDANCE_AGGREGATIONS = {
    "avg": "MEAN",
    "average": "MEAN",
    "mean": "MEAN",
    "median": "MEDIAN",
    "variance": "VARIANCE",
    "var": "VARIANCE",
    "standard deviation": "STDDEV",
    "stddev": "STDDEV",
    "stdev": "STDDEV",
    "sum": "SUM",
    "total": "SUM",
    "min": "MIN",
    "minimum": "MIN",
    "max": "MAX",
    "maximum": "MAX",
    "count": "COUNT",
    "count rows": "ROW_COUNT",
    "count records": "ROW_COUNT",
    "row count": "ROW_COUNT",
    "count distinct": "COUNT_DISTINCT",
}


def _normalize_guidance_aggregation(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = " ".join(str(value).strip().lower().replace("_", " ").split())
    normalized = {"avg": "average", "count distinct": "count distinct"}.get(normalized, normalized)
    result = _GUIDANCE_AGGREGATIONS.get(normalized)
    return result


def _parse_guidance_instruction(text: str) -> Dict[str, Any]:
    """Parse the bounded, deterministic human-guidance instruction vocabulary."""
    parsed: Dict[str, Any] = {
        "aggregation": None,
        "metric_phrase": None,
        "group_phrase": None,
        "filters": [],
        "ranking_direction": None,
        "share_of_total": False,
        "claim_boundary": None,
        "recognized": False,
    }
    if not text:
        return parsed
    cleaned = " ".join(text.strip().split())
    lower = cleaned.casefold()

    if re.search(r"\b(?:rows?|records?|observations?)\b", lower) and re.search(
        r"\b(?:count|number of|how many)\b", lower
    ):
        parsed["aggregation"] = "ROW_COUNT"
        parsed["recognized"] = True
    elif re.search(r"\bcount\s+(?:the\s+)?non[- ]?null\b", lower):
        parsed["aggregation"] = "COUNT"
        parsed["recognized"] = True
    elif re.search(r"\b(?:distinct|unique)\b", lower) and re.search(
        r"\b(?:count|number of|how many)\b", lower
    ):
        parsed["aggregation"] = "COUNT_DISTINCT"
        parsed["recognized"] = True
    else:
        agg_match = re.search(r"\b(standard\s+deviation|stddev|stdev|variance|var|avg|average|mean|median|sum|total|min(?:imum)?|max(?:imum)?)\b", lower)
        if agg_match:
            parsed["aggregation"] = _GUIDANCE_AGGREGATIONS[" ".join(agg_match.group(1).split())]
            parsed["recognized"] = True

    group_match = re.search(
        r"\b(?:group(?:ed)?\s+by|by|per|across)\s+([a-zA-Z0-9_\s-]+?)(?=\s*(?:,|;|\.|\?|!|$)|\s+(?:exclude|include|rank|sort|show|and\s+show)\b)",
        cleaned,
        re.I,
    )
    if group_match:
        parsed["group_phrase"] = group_match.group(1).strip()
        parsed["recognized"] = True

    metric_match = re.search(
        r"\b(?:use|using|calculate|compute|aggregate|count|sum|total|average|avg|mean|median|variance|var|stddev|stdev|standard\s+deviation|min(?:imum)?|max(?:imum)?)"
        r"\s+(?:the\s+)?(?:of\s+)?([a-zA-Z0-9_][a-zA-Z0-9_\s-]*?)"
        r"(?=\s+(?:(?:group(?:ed)?\s+)?by|per|across|and\s+show|exclude|include|rank|sort|show)\b|[,;.!?]|$)",
        cleaned,
        re.I,
    )
    if metric_match:
        phrase = metric_match.group(1).strip()
        if phrase.casefold() not in {"rows", "records", "observations"}:
            parsed["metric_phrase"] = phrase
            parsed["recognized"] = True

    if re.search(r"\b(?:share|percentage|percent|proportion)\b", lower):
        parsed["share_of_total"] = True
        parsed["recognized"] = True
        if parsed["aggregation"] is None and re.search(r"\b(?:share|percentage|percent|proportion)\s+of\b", lower):
            parsed["aggregation"] = "SUM"

    if re.search(r"\b(?:highest|largest|most|descending|highest first|rank high(?:est)? first)\b", lower):
        parsed["ranking_direction"] = "DESC"
        parsed["recognized"] = True
    elif re.search(r"\b(?:lowest|smallest|least|ascending|lowest first|rank low(?:est)? first)\b", lower):
        parsed["ranking_direction"] = "ASC"
        parsed["recognized"] = True

    if re.search(r"\b(?:association|correlation|non[- ]causal|not\s+causal)\b", lower):
        parsed["claim_boundary"] = "ASSOCIATION"
        parsed["recognized"] = True

    for match in re.finditer(
        r"\b(exclude|excluding|include only|only include)\s+(?:the\s+)?"
        r"([a-zA-Z0-9_\s-]+?)(?=\s*(?:,|;|\.|\?|!|$)|\s+(?:and\s+)?(?:rank|sort|show)\b)",
        cleaned,
        re.I,
    ):
        filter_phrase = re.sub(
            r"\s+(?:customers?|clients?|users?|accounts?|subscriptions?|orders?|records?)$",
            "",
            match.group(2).strip(),
            flags=re.I,
        )
        parsed["filters"].append(
            {"mode": "exclude" if match.group(1).casefold().startswith("excl") else "include", "phrase": filter_phrase}
        )
        parsed["recognized"] = True

    return parsed


def _apply_filter_instruction(
    phrase: str,
    mode: str,
    frame: pd.DataFrame,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Resolve an inclusion/exclusion phrase to one field and one typed value."""
    from packages.analytics_core.src.intelligence.canonical_question_contract import (
        resolve_phrase_to_column,
    )

    columns = [str(column) for column in frame.columns]
    normalized = _norm(phrase)

    # 1. Zero-value exclusions on monetary / quantitative measures (e.g. "zero-dollar", "zero revenue", "0 amount")
    if any(k in normalized for k in ("zero", "0")) and any(k in normalized for k in ("dollar", "price", "revenue", "amount", "cost", "sales", "spend", "value", "order")):
        # Find the target quantitative measure column
        measure_candidates = [
            c for c in columns
            if pd.api.types.is_numeric_dtype(frame[c])
            and any(term in _norm(c) for term in ("amount", "revenue", "price", "sales", "total", "net", "gross", "spend", "cost", "value"))
            and not any(id_k in c.lower() for id_k in ("_id", "id", "key", "pk", "fk", "uuid", "code", "zip"))
        ]
        if measure_candidates:
            target_col = measure_candidates[0]
            op = "!=" if mode == "exclude" else "="
            return {"column": target_col, "operator": op, "value": 0.0}, None
        return None, f"Cannot resolve '{phrase}': no monetary or quantitative measure column found in dataset."

    value_aliases = {
        "cancelled": ("cancelled", "canceled"),
        "canceled": ("cancelled", "canceled"),
        "active": ("active",),
        "returned": ("returned", "return", "returns"),
        "return": ("returned", "return", "returns"),
        "returns": ("returned", "return", "returns"),
    }
    expected_values = value_aliases.get(normalized, (phrase.strip(),))
    candidates: List[Tuple[str, Any]] = []

    for column in columns:
        series = frame[column]
        name = _norm(column)

        if pd.api.types.is_bool_dtype(series):
            if any(term in name for term in ("cancel", "return", "active", "status", normalized)):
                expected = normalized not in {"inactive", "not active", "false", "no", "cancelled", "canceled", "returned", "returns"}
                candidates.append((column, expected if mode == "include" else not expected))
            continue

        if pd.api.types.is_numeric_dtype(series):
            # Boolean flag stored as integer 0/1 (e.g. is_returned, is_cancelled)
            if any(term in name for term in ("is_returned", "returned", "is_cancelled", "cancelled", "is_active", "active")):
                val = 1 if mode == "include" else 0
                candidates.append((column, val))
            continue

        # For string/categorical columns, require the expected value to actually exist in the observed values!
        observed = {str(value).strip().casefold() for value in series.dropna().unique()}
        matching = next((value for value in expected_values if value.casefold() in observed), None)
        if matching is not None:
            # Preserve the exact casing/form present in observed categories
            actual_val = next(val for val in series.dropna().unique() if str(val).strip().casefold() == matching.casefold())
            candidates.append((column, actual_val))

    if not candidates:
        column = resolve_phrase_to_column(phrase, columns)
        if column:
            return None, f"Could not establish the value represented by '{phrase}' for column '{column}'."
        return None, f"Unsupported or unresolvable filter instruction: '{phrase}'. No matching observed category, status, or flag found in dataset."

    unique_columns = list(dict.fromkeys(column for column, _ in candidates))
    if len(unique_columns) != 1:
        return None, f"'{phrase}' can refer to multiple fields: {', '.join(unique_columns)}."

    column = unique_columns[0]
    value = next(val for candidate, val in candidates if candidate == column)
    series = frame[column]

    if pd.api.types.is_bool_dtype(series):
        operator = "="
    elif mode == "exclude":
        operator = "!="
    else:
        operator = "="
    return {"column": column, "operator": operator, "value": value}, None


def _apply_human_guidance_to_contract_impl(
    question: str,
    data_context: Any,
    guidance: Mapping[str, Any],
    previous_contract: Any = None,
) -> ResolverOutcome:
    """Resolve structured and bounded freeform guidance into one validated contract."""
    from dataclasses import replace
    from packages.analytics_core.src.intelligence.canonical_question_contract import (
        CanonicalQuestionContract,
        compile_canonical_question_contract,
        resolve_phrase_to_column,
    )

    instruction = " ".join(
        value.strip()
        for value in (
            str(guidance.get("clarification_response") or ""),
            str(guidance.get("custom_instruction") or ""),
        )
        if value and value.strip()
    )
    parsed = _parse_guidance_instruction(instruction)
    structured_aggregation = _normalize_guidance_aggregation(guidance.get("selected_aggregation"))
    if structured_aggregation == "COUNT":
        # The structured selector is explicitly labelled "COUNT (Rows / Observations)".
        # Counting non-null values is a separate freeform instruction.
        structured_aggregation = "ROW_COUNT"
    if guidance.get("selected_aggregation") and structured_aggregation is None:
        return ResolverOutcome(
            status=ResolverStatus.UNSUPPORTED,
            contract=previous_contract,
            unresolved_instruction=str(guidance["selected_aggregation"]),
            reason="The selected aggregation is not in the supported instruction set.",
            suggested_resolutions=tuple(sorted(_GUIDANCE_AGGREGATIONS.values())),
            original_instruction=str(guidance["selected_aggregation"]),
        )

    frames = getattr(data_context, "datasets_map", None)
    if frames is None and isinstance(data_context, Mapping):
        frames = data_context
    if isinstance(data_context, pd.DataFrame):
        frames = {"data": data_context}
    frames = {str(name): frame for name, frame in (frames or {}).items() if isinstance(frame, pd.DataFrame)}
    if not frames:
        return ResolverOutcome(
            status=ResolverStatus.UNSUPPORTED,
            contract=previous_contract,
            unresolved_instruction=question,
            reason="No usable dataset schema is available to validate the request.",
            original_instruction=question,
        )

    requested_dataset = guidance.get("selected_dataset")
    if requested_dataset:
        if str(requested_dataset) not in frames:
            return ResolverOutcome(
                status=ResolverStatus.NEEDS_CLARIFICATION,
                contract=previous_contract,
                unresolved_instruction=str(requested_dataset),
                reason="The selected dataset is not in the investigation scope.",
                suggested_resolutions=tuple(sorted(frames)),
                schema_information={"datasets": {name: list(map(str, frame.columns)) for name, frame in frames.items()}},
            )
        frames = {str(requested_dataset): frames[str(requested_dataset)]}

    from packages.analytics_core.src.engines.variable_resolution_gate import detect_named_measure_absent
    _all_cols = [str(c) for f in frames.values() for c in f.columns]
    _absent = detect_named_measure_absent(question, _all_cols)
    if _absent and not instruction:
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=previous_contract,
            unresolved_instruction=_absent,
            reason=(
                f"Your question asks about '{_absent}', but none of the selected datasets has a "
                f"column for it. MindEd will not substitute a different measure. Pick an existing "
                f"measure, or add the missing column and re-run."
            ),
            suggested_resolutions=tuple(dict.fromkeys(_all_cols)),
            schema_information={"datasets": {name: list(map(str, frame.columns)) for name, frame in frames.items()}},
            original_instruction=question,
        )
    candidate_results = {
        name: compile_canonical_question_contract(question, frame)
        for name, frame in frames.items()
    }
    structured_metric = guidance.get("selected_metric")
    structured_group = guidance.get("selected_group_dimension")
    candidate_names = []
    for name, frame in frames.items():
        columns = [str(column) for column in frame.columns]
        metrics_ok = not structured_metric or resolve_phrase_to_column(str(structured_metric), columns) is not None
        groups_ok = not structured_group or resolve_phrase_to_column(str(structured_group), columns) is not None
        if metrics_ok and groups_ok:
            candidate_names.append(name)
    auto_selected = None
    if len(frames) > 1 and len(candidate_names) > 1 and not structured_metric and not structured_group:
        auto_selected = unique_best_dataset(question, {n: frames[n] for n in candidate_names})
        if auto_selected:
            candidate_names = [auto_selected]
    if len(frames) > 1 and len(candidate_names) != 1:
        options = tuple(candidate_names or sorted(frames))
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=previous_contract,
            unresolved_instruction=instruction or question,
            reason=(
                "The selected dataset scope contains multiple plausible schemas; this runtime has no "
                "validated relationship metadata to infer a cross-dataset join."
            ),
            suggested_resolutions=options,
            schema_information={"datasets": {name: list(map(str, frame.columns)) for name, frame in frames.items()}},
        )
    dataset_name = candidate_names[0] if len(candidate_names) == 1 else next(iter(frames))
    frame = frames[dataset_name]
    base = previous_contract if isinstance(previous_contract, CanonicalQuestionContract) else candidate_results[dataset_name]
    columns = [str(column) for column in frame.columns]
    comparison_question = bool(
        re.search(r"\b(?:compare|comparison|versus|vs\.?|differ\w*|higher|lower|more|less)\b", question, re.I)
    )
    semantic_guidance = any(
        guidance.get(key) not in (None, "", [])
        for key in (
            "selected_metric",
            "selected_aggregation",
            "selected_group_dimension",
            "selected_dataset",
            "scope_filter",
        )
    ) or bool(instruction)
    if comparison_question and not semantic_guidance:
        return ResolverOutcome(
            status=ResolverStatus.RESOLVED,
            contract=replace(
                base,
                family="NONE",
                completeness="NOT_ENFORCED",
                unresolved_roles=(),
                unresolved_semantics=(),
            ),
            schema_information={"dataset": dataset_name, "columns": columns},
        )
    predictive_question = bool(
        re.search(r"\b(likely to|risk of|probability of|predicted to|likelihood of)\s+(churn|attrition|cancel|default|convert|leave)\b", question, re.I)
        or re.search(r"\b(?:which|who)\s+(?:customers?|users?|accounts?)\s+(?:are\s+)?likely\s+to\b", question, re.I)
    )
    if predictive_question and not semantic_guidance:
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=base,
            unresolved_instruction=instruction or question,
            reason=(
                "The question asks for prospective risk ('likely to churn'), but observational data contains "
                "historical records. Prospective prediction claims require explicit holdout model validation."
            ),
            suggested_resolutions=(
                "Analyze historical churn patterns by segment",
                "List customers already recorded as churned",
                "Refuse predictive claims without validated holdout model",
            ),
            schema_information={"dataset": dataset_name, "columns": columns},
            original_instruction=instruction or question,
        )
    if instruction and not parsed["metric_phrase"]:
        direct_column = re.search(r"`([a-zA-Z0-9_]+)`|\(([a-zA-Z0-9_]+)\)", instruction)
        if direct_column:
            candidate_column = direct_column.group(1) or direct_column.group(2)
            if candidate_column in columns:
                parsed["metric_phrase"] = candidate_column
                parsed["recognized"] = True

    values: Dict[str, Any] = {}
    structured_fields = {
        "selected_metric": "metric",
        "selected_group_dimension": "grouping",
        "selected_aggregation": "aggregation",
    }
    for key, field_name in structured_fields.items():
        if guidance.get(key) not in (None, ""):
            values[field_name] = guidance[key]

    resolved_metric = (
        resolve_phrase_to_column(str(structured_metric), columns)
        if structured_metric not in (None, "")
        else None
    )
    resolved_group = (
        resolve_phrase_to_column(str(structured_group), columns)
        if structured_group not in (None, "")
        else None
    )
    freeform_metric = (
        resolve_phrase_to_column(parsed["metric_phrase"], columns)
        if parsed["metric_phrase"]
        else None
    )
    freeform_groups: list[str] = []
    if parsed["group_phrase"]:
        group_parts = [part.strip() for part in re.split(r",|\band\b|\bthen\b", parsed["group_phrase"], flags=re.I) if part.strip()]
        for part in group_parts:
            resolved_part = resolve_phrase_to_column(part, columns)
            if not resolved_part:
                freeform_groups = []
                break
            if resolved_part not in freeform_groups:
                freeform_groups.append(resolved_part)
    freeform_group = freeform_groups[0] if len(freeform_groups) == 1 else None
    freeform_aggregation = parsed.get("aggregation")

    for field_name, structured_value, freeform_value in (
        ("aggregation", structured_aggregation, freeform_aggregation),
        ("metric", resolved_metric, freeform_metric),
        ("grouping", resolved_group, freeform_group),
    ):
        if structured_value and freeform_value and structured_value != freeform_value:
            return ResolverOutcome(
                status=ResolverStatus.CONFLICT,
                contract=base,
                unresolved_instruction=instruction,
                reason=f"Structured and freeform guidance disagree on {field_name}.",
                conflict=GuidanceConflict(
                    field=field_name,
                    structured_value=structured_value,
                    freeform_value=freeform_value,
                    structured_instruction=str(guidance.get(
                        {"aggregation": "selected_aggregation", "metric": "selected_metric", "grouping": "selected_group_dimension"}[field_name]
                    )),
                    freeform_instruction=instruction,
                ),
                original_instruction=instruction,
            )

    if structured_group not in (None, "") and len(freeform_groups) > 1 and resolved_group not in freeform_groups:
        return ResolverOutcome(
            status=ResolverStatus.CONFLICT, contract=base,
            unresolved_instruction=instruction,
            reason="Structured and freeform guidance disagree on grouping dimensions.",
            original_instruction=instruction,
        )

    aggregation = structured_aggregation or freeform_aggregation or base.requested_aggregation
    if aggregation == "AVG":
        aggregation = "MEAN"
    metric = resolved_metric or freeform_metric or base.target_column
    if parsed.get("group_phrase"):
        groups = list(freeform_groups)
    elif resolved_group:
        groups = [resolved_group]
    else:
        groups = list(base.grouping_columns)
    group = groups[0] if len(groups) == 1 else None

    if parsed["metric_phrase"] and not freeform_metric:
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=base,
            unresolved_instruction=parsed["metric_phrase"],
            reason="The requested measure did not resolve unambiguously to a field in the selected dataset.",
            suggested_resolutions=tuple(columns),
            schema_information={"dataset": dataset_name, "columns": columns},
            original_instruction=instruction,
        )
    if parsed["group_phrase"] and not freeform_groups:
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=base,
            unresolved_instruction=parsed["group_phrase"],
            reason="The requested grouping dimension did not resolve to a field in the selected dataset.",
            suggested_resolutions=tuple(columns),
            schema_information={"dataset": dataset_name, "columns": columns},
            original_instruction=instruction,
        )
    if structured_metric and not resolved_metric:
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=base,
            unresolved_instruction=str(structured_metric),
            reason="The selected measure is not present in the selected dataset.",
            suggested_resolutions=tuple(columns),
            schema_information={"dataset": dataset_name, "columns": columns},
        )
    if structured_group and not resolved_group:
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=base,
            unresolved_instruction=str(structured_group),
            reason="The selected grouping dimension is not present in the selected dataset.",
            suggested_resolutions=tuple(columns),
            schema_information={"dataset": dataset_name, "columns": columns},
        )

    filters = [dict(condition) for condition in base.filter_conditions]
    raw_scope_filter = str(guidance.get("scope_filter") or "").strip()
    if raw_scope_filter:
        filter_match = re.fullmatch(
            r"\s*([a-zA-Z0-9_. -]+?)\s*(=|!=)\s*(?:'([^']*)'|\"([^\"]*)\"|([a-zA-Z0-9_.-]+))\s*",
            raw_scope_filter,
        )
        if not filter_match:
            return ResolverOutcome(
                status=ResolverStatus.UNSUPPORTED,
                contract=base,
                unresolved_instruction=raw_scope_filter,
                reason="Only one equality or exclusion filter with a literal value is supported; SQL expressions are not accepted.",
                original_instruction=raw_scope_filter,
            )
        filter_column = resolve_phrase_to_column(filter_match.group(1).strip(), columns)
        if not filter_column:
            return ResolverOutcome(
                status=ResolverStatus.NEEDS_CLARIFICATION,
                contract=base,
                unresolved_instruction=filter_match.group(1).strip(),
                reason="The filter field did not resolve to a dataset column.",
                suggested_resolutions=tuple(columns),
                schema_information={"dataset": dataset_name, "columns": columns},
                original_instruction=raw_scope_filter,
            )
        filter_value = next(value for value in filter_match.groups()[2:] if value is not None)
        filters.append({"column": filter_column, "operator": filter_match.group(2), "value": filter_value})

    for filter_spec in parsed["filters"]:
        resolved_filter, filter_error = _apply_filter_instruction(filter_spec["phrase"], filter_spec["mode"], frame)
        if filter_error:
            return ResolverOutcome(
                status=ResolverStatus.NEEDS_CLARIFICATION,
                contract=base,
                unresolved_instruction=filter_spec["phrase"],
                reason=filter_error,
                suggested_resolutions=tuple(columns),
                schema_information={"dataset": dataset_name, "columns": columns},
                original_instruction=instruction,
            )
        if resolved_filter:
            filters.append(resolved_filter)

    # Assumption acknowledgements and UI provenance are not semantic instructions. In
    # particular, confirmed_assumptions=[] (or a non-empty acknowledgement list) must not
    # change an otherwise executable contract into NEEDS_CLARIFICATION.
    has_guidance = any(
        guidance.get(key) not in (None, "", [], {})
        for key in (
            "selected_metric", "selected_aggregation", "selected_group_dimension",
            "selected_dataset", "scope_filter", "_canonical_intent_patch",
        )
    ) or bool(instruction)
    if not has_guidance and str(getattr(base, "task_family", "")).upper() == "RANKING":
        return ResolverOutcome(
            status=ResolverStatus.RESOLVED,
            contract=bind_default_aggregation(base),
            schema_information={"dataset": dataset_name, "columns": columns},
        )
    if has_guidance and instruction and not parsed["recognized"]:
        return ResolverOutcome(
            status=ResolverStatus.UNSUPPORTED,
            contract=base,
            unresolved_instruction=instruction,
            reason="The freeform instruction does not match a supported analytical instruction form.",
            suggested_resolutions=(
                "Aggregate a numeric measure by a field",
                "Count rows or distinct values",
                "Apply one equality or exclusion filter",
                "Rank grouped results",
                "Show a share of an additive total",
            ),
            original_instruction=instruction,
        )

    if aggregation is None and (resolved_metric or freeform_metric):
        if parsed.get("share_of_total"):
            aggregation = "SUM"
        else:
            return ResolverOutcome(
                status=ResolverStatus.NEEDS_CLARIFICATION,
                contract=base,
                unresolved_instruction="aggregation",
                reason="The measure and grouping are known, but the requested aggregation is not specified.",
                suggested_resolutions=("SUM", "MEAN", "MEDIAN", "MIN", "MAX"),
                schema_information={"dataset": dataset_name, "metric": metric},
                original_instruction=instruction or question,
            )

    if (
        parsed.get("claim_boundary")
        and aggregation is None
        and not freeform_metric
        and not freeform_group
        and not parsed["filters"]
        and not raw_scope_filter
    ):
        updated_claim = replace(
            base,
            requested_claim="ASSOCIATION",
            supported_claim="ASSOCIATION",
            claim_ceiling="ASSOCIATION",
            claim_type="ASSOCIATION",
            resolution_evidence={
                **dict(base.resolution_evidence),
                "human_claim_boundary": "ASSOCIATION",
                "freeform_instruction": instruction,
            },
        )
        return ResolverOutcome(
            status=ResolverStatus.RESOLVED,
            contract=updated_claim,
            schema_information={"dataset": dataset_name, "columns": columns},
        )

    if aggregation in {"SUM", "MEAN", "MEDIAN", "MIN", "MAX", "VARIANCE", "STDDEV"}:
        if not metric or metric not in frame.columns:
            return ResolverOutcome(
                status=ResolverStatus.NEEDS_CLARIFICATION,
                contract=base,
                unresolved_instruction="metric",
                reason="A valid measure is required for this aggregation.",
                suggested_resolutions=tuple(
                    column for column in columns if pd.api.types.is_numeric_dtype(frame[column])
                ),
                schema_information={"dataset": dataset_name, "columns": columns},
                original_instruction=instruction or question,
            )
        series = frame[metric]
        numeric_aggregation = aggregation in {"SUM", "MEAN", "MEDIAN", "VARIANCE", "STDDEV"}
        compatible = pd.api.types.is_numeric_dtype(series) if numeric_aggregation else (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or series.dtype == object
        )
        if not compatible:
            return ResolverOutcome(
                status=ResolverStatus.NEEDS_CLARIFICATION,
                contract=base,
                unresolved_instruction=f"{aggregation}({metric})",
                reason=f"Column '{metric}' has an incompatible type for {aggregation}.",
                suggested_resolutions=tuple(
                    column for column in columns if pd.api.types.is_numeric_dtype(frame[column])
                ),
                schema_information={"dataset": dataset_name, "metric_dtype": str(series.dtype)},
            )
    if aggregation == "COUNT_DISTINCT" and not metric and not instruction:
        # "How many customers churned?": the verb names a binary flag column, so the
        # question is a count of rows where the flag is positive -- not a distinct count.
        _flag, _positive = _positive_flag_in_question(question, frame)
        if _flag is not None:
            aggregation = "COUNT"
            metric = _flag
            filters = [*filters, {"column": _flag, "operator": "=", "value": _positive}]
    if aggregation == "COUNT_DISTINCT" and not metric:
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=base,
            unresolved_instruction="distinct-count field",
            reason="A field or entity identifier is required for a distinct count.",
            suggested_resolutions=tuple(columns),
            schema_information={"dataset": dataset_name, "columns": columns},
        )
    if group and group not in frame.columns:
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=base,
            unresolved_instruction=group,
            reason="The grouping dimension is not available in the selected dataset.",
            suggested_resolutions=tuple(columns),
            schema_information={"dataset": dataset_name, "columns": columns},
        )
    if parsed["share_of_total"] and (aggregation != "SUM" or not group):
        return ResolverOutcome(
            status=ResolverStatus.UNSUPPORTED,
            contract=base,
            unresolved_instruction=instruction,
            reason="Share-of-total output requires an additive SUM grouped by at least one dimension.",
            suggested_resolutions=("SUM of an additive numeric measure by a dimension",),
            original_instruction=instruction,
        )

    family = "COUNT" if aggregation in {"ROW_COUNT", "COUNT", "COUNT_DISTINCT"} else "SIMPLE_AGGREGATE"
    if aggregation is None:
        family = base.family
    if family == "NONE" and not has_guidance and base.resolution_confidence >= 0.7:
        return ResolverOutcome(
            status=ResolverStatus.RESOLVED,
            contract=bind_default_aggregation(base),
            schema_information={"dataset": dataset_name, "columns": columns},
        )
    if family == "NONE":
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=base,
            unresolved_instruction=instruction or question,
            reason="The request is not yet a complete member of the supported deterministic instruction set.",
            suggested_resolutions=("Choose a supported aggregation, ranking, or filter operation.",),
            schema_information={"dataset": dataset_name, "columns": columns},
        )

    if aggregation == "ROW_COUNT":
        metric = None
    elif aggregation == "COUNT" and not metric:
        metric = None
    unresolved = []
    if aggregation is None:
        unresolved.append("aggregation")
    if aggregation in {"SUM", "MEAN", "MEDIAN", "MIN", "MAX", "VARIANCE", "STDDEV", "PERCENTILE", "COUNT_DISTINCT"} and not metric:
        unresolved.append("target")
    if parsed["group_phrase"] and not groups:
        unresolved.append("grouping")
    requested_aggregation = aggregation or base.requested_aggregation
    task_family = "COUNT" if family == "COUNT" else "DESCRIPTIVE"
    estimand = "count" if family == "COUNT" else {
        "SUM": "sum", "MEAN": "mean", "MEDIAN": "median", "MIN": "minimum", "MAX": "maximum",
        "VARIANCE": "variance", "STDDEV": "stddev", "PERCENTILE": "percentile"
    }.get(aggregation, base.estimand)
    completeness = "INCOMPLETE" if unresolved else "COMPLETE"
    updated = replace(
        base,
        question_text=question,
        task_family=task_family,
        estimand=estimand,
        target_column=metric,
        grouping_columns=tuple(groups),
        filter_conditions=tuple(filters),
        requested_aggregation=requested_aggregation,
        ranking_direction=parsed.get("ranking_direction") or base.ranking_direction,
        share_of_total=bool(parsed["share_of_total"]),
        requested_outputs=(
            tuple(dict.fromkeys((*base.requested_outputs, "group_values", "share_of_total")))
            if parsed["share_of_total"]
            else base.requested_outputs
        ),
        family=family,
        completeness=completeness,
        unresolved_roles=tuple(unresolved),
        unresolved_semantics=tuple(f"unresolved_role:{role}" for role in unresolved),
        resolution_confidence=1.0 if not unresolved else 0.2,
        resolution_evidence={
            **dict(base.resolution_evidence),
            "guidance_resolution": {
                "dataset": dataset_name,
                # Assumption acknowledgements are audit metadata, not calculation semantics;
                # including them here would change the approved plan hash without changing intent.
                "submitted_guidance": {k: v for k, v in dict(guidance).items() if k != "confirmed_assumptions"},
                "freeform_instruction": instruction,
                "resolved_metric": metric,
                "resolved_grouping": group if len(groups) == 1 else list(groups),
                "resolved_aggregation": requested_aggregation,
                "resolved_filters": filters,
            },
        },
    )
    if unresolved:
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=updated,
            unresolved_instruction=", ".join(unresolved),
            reason="The contract is missing one or more required fields.",
            suggested_resolutions=tuple(columns),
            schema_information={"dataset": dataset_name, "columns": columns},
            original_instruction=instruction or question,
        )
    return ResolverOutcome(
        status=ResolverStatus.RESOLVED,
        contract=updated,
        schema_information={"dataset": dataset_name, "columns": columns},
    )


def apply_human_guidance_to_contract(
    question: str,
    data_context: Any,
    guidance: Mapping[str, Any],
    previous_contract: Any = None,
) -> ResolverOutcome:
    """Resolve one normalized guidance payload into the authoritative contract.

    Structured edits and clarification choices are treated as patches to the canonical contract.
    The original question is only used to establish intent and provenance; it is not reinterpreted
    as an alternate execution source after the contract is resolved.
    """
    normalized = normalize_guidance_payload(guidance)
    patch = normalized.pop("intent_patch", None)
    pending_slot = normalized.pop("pending_intent_slot", None)
    freeform_answer = normalized.get("custom_instruction") or normalized.get("clarification_response")
    if pending_slot and freeform_answer:
        if patch:
            return ResolverOutcome(
                status=ResolverStatus.CONFLICT,
                contract=previous_contract,
                unresolved_instruction=str(pending_slot),
                reason="Provide either a selected clarification patch or a freeform answer, not both.",
                original_instruction=str(freeform_answer),
            )
        answer_frames = getattr(data_context, "datasets_map", None)
        if answer_frames is None and isinstance(data_context, Mapping):
            answer_frames = data_context
        if isinstance(data_context, pd.DataFrame):
            answer_frames = {"data": data_context}
        answer_frames = {str(name): frame for name, frame in (answer_frames or {}).items() if isinstance(frame, pd.DataFrame)}
        answer_dataset = str(normalized.get("selected_dataset") or "")
        answer_frame = answer_frames.get(answer_dataset)
        if answer_frame is None and len(answer_frames) == 1:
            answer_frame = next(iter(answer_frames.values()))
        if answer_frame is None and answer_frames:
            _answer_name, answer_frame = select_primary_dataset(question, answer_frames)
        if answer_frame is None:
            return ResolverOutcome(
                status=ResolverStatus.NEEDS_CLARIFICATION,
                contract=previous_contract,
                unresolved_instruction=str(pending_slot),
                reason="The pending clarification cannot be bound because the selected dataset schema is unavailable.",
                original_instruction=str(freeform_answer),
            )
        try:
            patch = normalize_intent_patch(infer_intent_patch_from_answer(str(freeform_answer), str(pending_slot), answer_frame))
        except (TypeError, ValueError) as exc:
            return ResolverOutcome(
                status=ResolverStatus.NEEDS_CLARIFICATION,
                contract=previous_contract,
                unresolved_instruction=str(pending_slot),
                reason=str(exc),
                suggested_resolutions=tuple(map(str, answer_frame.columns)),
                schema_information={"columns": list(map(str, answer_frame.columns))},
                original_instruction=str(freeform_answer),
            )
        # The answer has been compiled to a typed patch. Do not run it through the unrelated
        # general-purpose freeform parser as a second interpretation.
        normalized.pop("custom_instruction", None)
        normalized.pop("clarification_response", None)
    if patch:
        # Simple UI slots also feed the existing resolver so conflicts with freeform instructions
        # are reported by its established conflict machinery. The complete patch is applied again
        # after resolution, guaranteeing that the returned contract contains the exact slot values.
        aliases = {
            "target_column": "selected_metric",
            "requested_aggregation": "selected_aggregation",
        }
        try:
            # Signal to the base resolver that a typed patch was supplied even if it only
            # changes a field (e.g. time_column) that is not a legacy form control.
            normalized["_canonical_intent_patch"] = patch
            for patch_field, guidance_field in aliases.items():
                if patch_field in patch:
                    incoming = patch[patch_field]
                    current = normalized.get(guidance_field)
                    if current not in (None, "") and str(current).casefold() != str(incoming).casefold():
                        raise ValueError(f"intent patch conflicts with {guidance_field}.")
                    normalized[guidance_field] = incoming
            if "grouping_columns" in patch:
                groups = patch["grouping_columns"]
                groups = [groups] if isinstance(groups, str) else list(groups or [])
                if len(groups) > 1:
                    raise ValueError("The current deterministic resolver supports one grouping column per plan.")
                if groups:
                    current = normalized.get("selected_group_dimension")
                    if current not in (None, "") and str(current).casefold() != str(groups[0]).casefold():
                        raise ValueError("intent patch conflicts with selected_group_dimension.")
                    normalized["selected_group_dimension"] = groups[0]
        except ValueError as exc:
            return ResolverOutcome(
                status=ResolverStatus.CONFLICT,
                contract=previous_contract,
                unresolved_instruction=str(exc),
                reason=str(exc),
                original_instruction=str(guidance.get("clarification_response") or ""),
            )

    # UI acknowledgement is provenance only and may be empty after normalization. It is never
    # allowed to make a deterministic question unresolved by itself.
    outcome = _apply_human_guidance_to_contract_impl(
        question=question,
        data_context=data_context,
        guidance=normalized,
        previous_contract=previous_contract,
    )
    if outcome.contract is None:
        return outcome

    frames = getattr(data_context, "datasets_map", None)
    if frames is None and isinstance(data_context, Mapping):
        frames = data_context
    if isinstance(data_context, pd.DataFrame):
        frames = {"data": data_context}
    frames = {str(name): frame for name, frame in (frames or {}).items() if isinstance(frame, pd.DataFrame)}
    dataset_name = (outcome.schema_information or {}).get("dataset")
    frame = frames.get(str(dataset_name)) if dataset_name is not None else None
    if frame is None and len(frames) == 1:
        dataset_name, frame = next(iter(frames.items()))

    contract = outcome.contract
    try:
        if patch:
            if frame is None:
                raise ValueError("The selected dataset is unavailable for applying the intent patch.")
            contract = apply_contract_intent_patch(contract, patch, frame, question)
        # A small, explicitly supported compound shape (aggregate one measure by one dimension,
        # then rank that same dimension by that same measure) can be represented as a persisted
        # graph. Other compound questions remain blocked by the intent-ledger gate below.
        if frame is not None and contract is not None and not getattr(contract, "operation_graph", None):
            try:
                from dataclasses import replace as _replace_contract
                from packages.analytics_core.src.intelligence.analytical_operations import build_supported_aggregate_then_rank_graph
                supported_graph = build_supported_aggregate_then_rank_graph(question, contract, list(map(str, frame.columns)))
                if supported_graph is not None:
                    contract = _replace_contract(contract, operation_graph=supported_graph.to_dict())
            except (ImportError, TypeError, ValueError):
                # The normal compound-intent gate remains authoritative if safe graph compilation fails.
                pass
        contract, ledger, requirements, patches = enrich_contract_intent_state(
            question=question,
            contract=contract,
            frame=frame,
            guidance=normalized,
        )
        contract = bind_default_aggregation(contract)
    except (TypeError, ValueError) as exc:
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=outcome.contract,
            unresolved_instruction=str(exc),
            reason=str(exc),
            suggested_resolutions=tuple(map(str, frame.columns)) if frame is not None else (),
            schema_information={**dict(outcome.schema_information or {}), "intent_patch_error": str(exc)},
            original_instruction=str(guidance.get("clarification_response") or guidance.get("custom_instruction") or ""),
        )

    if dataset_name is not None:
        schema_info = {**dict(outcome.schema_information or {}), "dataset": dataset_name}
    else:
        schema_info = dict(outcome.schema_information or {})
    if frame is not None:
        schema_info.setdefault("columns", list(map(str, frame.columns)))

    if any(item.get("slot") == "operation_graph" for item in requirements):
        reason = (
            "This question contains multiple analytical operations, but the current approved intent contract "
            "does not yet represent them as separate executable nodes. Split it into one analytical question "
            "per operation; AA-OS will not execute only the easiest-to-parse clause."
        )
        return ResolverOutcome(
            status=ResolverStatus.UNSUPPORTED,
            contract=contract,
            unresolved_instruction="operation_graph",
            reason=reason,
            suggested_resolutions=("Split the request into one analytical operation per investigation.",),
            schema_information=schema_info,
            original_instruction=outcome.original_instruction or question,
            suggested_patches=tuple(patches),
        )

    if requirements or (outcome.status == ResolverStatus.RESOLVED and contract.completeness == "INCOMPLETE"):
        open_slots = list(dict.fromkeys(str(item.get("slot")) for item in requirements))
        reason = "Some material parts of the question are not yet represented in the executable contract."
        details = "; ".join(
            f"{item.get('slot')}: {item.get('reason')}" for item in requirements
        )
        if details:
            reason = f"{reason} {details}"
        return ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            contract=contract,
            unresolved_instruction=", ".join(open_slots) or ", ".join(contract.unresolved_roles),
            reason=reason,
            suggested_resolutions=tuple(map(str, frame.columns)) if frame is not None else outcome.suggested_resolutions,
            schema_information=schema_info,
            conflict=outcome.conflict,
            original_instruction=outcome.original_instruction,
            suggested_patches=tuple(patches),
        )

    # Preserve existing fail-closed statuses and enrich successful responses with the same
    # intent ledger used by preview. The plan hash will include this ledger and all material slots.
    return ResolverOutcome(
        status=outcome.status,
        contract=contract,
        unresolved_instruction=outcome.unresolved_instruction,
        reason=outcome.reason,
        suggested_resolutions=outcome.suggested_resolutions,
        schema_information=schema_info,
        conflict=outcome.conflict,
        original_instruction=outcome.original_instruction,
        suggested_patches=tuple(patches),
    )


def _positive_flag_in_question(question: str, frame: Any) -> tuple[Optional[str], Optional[str]]:
    """Return (column, positive_literal) when exactly one binary flag column is named by a
    word of the question (``churned`` -> ``churned``); otherwise (None, None)."""
    words = set(re.findall(r"[a-z]+", str(question).lower()))
    stems = {w[:-2] if w.endswith("ed") else w for w in words if len(w) > 3}
    hits = []
    for column in frame.columns:
        col = str(column).lower()
        if col in words or col.rstrip("_flag") in words or col[:-2] in stems and col.endswith("ed"):
            values = {str(v).strip().lower() for v in frame[column].dropna().unique()}
            if values and values <= {"0", "1", "0.0", "1.0", "true", "false", "yes", "no"} and len(values) <= 2:
                hits.append(column)
    if len(hits) != 1:
        return None, None
    values = {str(v).strip() for v in frame[hits[0]].dropna().unique()}
    for lit in ("1", "1.0", "True", "true", "Yes", "yes"):
        if lit in values:
            return hits[0], lit
    return None, None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _evaluate_guided_investigation_legacy(
    question: str,
    df: Optional[pd.DataFrame] = None,
    semantic: Any = None,
    available_columns: Optional[List[str]] = None,
    tables_info: Optional[Dict[str, List[str]]] = None,
    table_relationships: Optional[List[Dict[str, Any]]] = None,
    is_observational: bool = True,
    verification_passed: Optional[bool] = None,
) -> ProposedInterpretation:
    """Analyze the question against available data contracts and construct a transparent proposed interpretation.

    Detects consequential analytical choices and requests human judgment before execution
    when the interpretation is ambiguous, causal on observational data, has missing measures, or risks join fanout.
    """
    q_clean = " ".join((question or "").strip().split())
    ql = q_clean.lower()

    # Compile underlying canonical contract as single source of truth (Milestone M1)
    canonical_contract = None
    if df is not None:
        try:
            from packages.analytics_core.src.intelligence.canonical_question_contract import (
                compile_canonical_question_contract,
            )
            canonical_contract = compile_canonical_question_contract(q_clean, df)
        except Exception:
            canonical_contract = None
    cols: List[str] = []
    if df is not None and hasattr(df, "columns"):
        cols = [str(c) for c in df.columns]
    elif available_columns:
        cols = list(available_columns)
    elif semantic is not None:
        num = getattr(semantic, "available_numeric_cols", []) or []
        cat = getattr(semantic, "available_categorical_cols", []) or []
        cols = list(num) + list(cat)

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    if df is not None:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in cols if c not in numeric_cols]
    elif semantic is not None:
        numeric_cols = list(getattr(semantic, "available_numeric_cols", []) or [])
        categorical_cols = list(getattr(semantic, "available_categorical_cols", []) or [])
    else:
        # Heuristic based on names
        numeric_cols = [c for c in cols if any(k in c.lower() for k in ["amount", "revenue", "spend", "sales", "price", "count", "qty", "quantity", "cost", "profit", "pct", "rate", "days", "tickets"])]
        categorical_cols = [c for c in cols if c not in numeric_cols]

    # 1. Identify Task Type
    is_predictive = bool(
        re.search(r"\b(likely to|risk of|probability of|predicted to|likelihood of)\s+(churn|attrition|cancel|default|convert|leave)\b", ql)
        or re.search(r"\b(?:which|who)\s+(?:customers?|users?|accounts?)\s+(?:are\s+)?likely\s+to\b", ql)
        or re.search(r"\bpredict\b|\bforecast\b", ql)
    )
    is_ranking = bool(re.search(r"\b(which|who|top|bottom|highest|lowest|best|worst|rank|ranking|most|least)\b", ql))
    is_causal = bool(re.search(r"\b(did|does|do|can)\s+.+\s+(cause|drive|lead to|result in)\b|\bcaus(?:al|ality|e|ed|ing)\b", ql))
    is_diagnostic = bool(re.search(r"\bwhy\s+did\b|\bwhat\s+(caused|drove|led to)\b", ql))
    is_association = bool(re.search(r"\b(affect|influence|impact|associated with|correlated?|relation(?:ship)?|link(?:ed)?)\b", ql))
    is_comparison = bool(re.search(r"\b(compare|difference|higher than|lower than|versus|vs\.?|between)\b", ql))

    task = canonical_contract.task_family if canonical_contract is not None else "DESCRIPTIVE"
    if canonical_contract is None:
        if is_predictive:
            task = "PREDICTION"
        elif is_causal:
            task = "CAUSAL"
        elif is_diagnostic:
            task = "DIAGNOSTIC"
        elif is_ranking:
            task = "RANKING"
        elif is_comparison:
            task = "COMPARISON"
        elif is_association:
            task = "ASSOCIATION"

    # 1a. Handle Prospective / Predictive Questions on Historical Observational Data
    if is_predictive:
        # Check if the target is an observed historical event (like churn/churned)
        event_col = next((c for c in cols if any(k in c.lower() for k in ["churn", "churned", "attrition", "default", "cancel"])), None)
        dim_candidates = [c for c in cols if c != event_col and not any(id_k in c.lower() for id_k in ("_id", "id", "key", "pk", "fk", "uuid"))]
        default_dim = dim_candidates[0] if dim_candidates else (categorical_cols[0] if categorical_cols else "segment")

        return ProposedInterpretation(
            question=q_clean,
            task="PREDICTION",
            proposed_interpretation=(
                f"Question asks for prospective risk ('likely to {event_col or 'churn'}'), but the dataset contains historical observed outcomes. "
                "Predictive claims require verified out-of-sample holdout validation."
            ),
            planned_calculation=(
                f"Clarification required: cannot evaluate individual future risk from historical {event_col or 'binary'} records without predictive holdout validation."
            ),
            target_metric=event_col,
            group_dimension=default_dim,
            assumptions=[
                f"Historical '{event_col or 'target'}' flag records past outcomes, not prospective future probabilities.",
                "Summing binary flags per entity (n=1) produces historical classification (0 or 1), not risk scores.",
                "No out-of-sample validation or trained model exists to warrant a PREDICTION claim.",
            ],
            ambiguity_status=AmbiguityStatus.AMBIGUOUS_SCOPE,
            clarification_prompt=(
                f"This question asks for predictive future risk ('likely to churn'), but '{event_col or 'churned'}' in the dataset is a historical observed outcome. "
                "AA-OS requires genuine out-of-sample validation for prospective risk models. "
                "Would you like to analyze historical churn rates by segment, list customers already labelled as churned, or refuse predictive claims without holdout data?"
            ),
            clarification_options=[
                f"Analyze historical churn patterns by {default_dim} (descriptive breakdown)",
                f"List customers already recorded as churned ({event_col or 'churned'} = 1)",
                "Refuse predictive claims without validated holdout model",
            ],
            can_auto_proceed=False,
            recommended_action="Do not present historical binary records as risk predictions; clarify analytical intent.",
        )

    # 2. Check for Independent Verification Failure if supplied
    if verification_passed is False:
        return ProposedInterpretation(
            question=q_clean,
            task=task,
            proposed_interpretation="Independent recomputation failed to verify this calculation.",
            planned_calculation="Independent DuckDB query reproduced a discrepancy or failed consistency check.",
            assumptions=["Independent verification failed closed."],
            ambiguity_status=AmbiguityStatus.VERIFICATION_FAILED,
            clarification_prompt="Independent verification checks failed. MindEd refuses to certify this claim as verified. Would you like to inspect discrepancies or rework the calculation?",
            clarification_options=["Inspect discrepancies", "Send back for rework", "Reject result"],
            can_auto_proceed=False,
            recommended_action="Refuse verification claim; investigate discrepancies.",
        )

    # 3. Check for Potential Join Fanout Risk
    if table_relationships:
        for rel in table_relationships:
            cardinality = rel.get("cardinality", "")
            if cardinality in ("ONE_TO_MANY", "1:N", "MANY_TO_MANY", "N:M"):
                key = rel.get("join_key", "customer_id")
                metric_name = rel.get("metric", "revenue")
                return ProposedInterpretation(
                    question=q_clean,
                    task=task,
                    proposed_interpretation=f"Multi-table join across {rel.get('left_table')} and {rel.get('right_table')} with 1-to-many relationship.",
                    planned_calculation=f"JOIN {rel.get('left_table')} ON {key}",
                    assumptions=[f"Tables have multiple rows per {key}.", "Direct join risks fanout record duplication."],
                    ambiguity_status=AmbiguityStatus.JOIN_FANOUT_RISK,
                    clarification_prompt=f"Two tables contain multiple rows per {key}. Joining them directly could duplicate {metric_name}. I can aggregate each table to {key} level first, but that changes the analytical grain. Shall I use that approach?",
                    clarification_options=[
                        f"Aggregate to {key} level first (preserve grain)",
                        f"Direct line-item join (accept fanout)",
                        "Refuse join without explicit grain declaration",
                    ],
                    can_auto_proceed=False,
                    recommended_action=f"Aggregate to {key} level first to prevent metric duplication.",
                )

    # 4. Extract Intended Grouping Dimension
    group_col: Optional[str] = None
    group_match = re.search(r"\bwhich\s+([a-zA-Z0-9_ ]+?)\s+(?:has|have|shows?|with|is|had)\b", ql)
    if group_match:
        cand_group = _norm(group_match.group(1))
        for c in categorical_cols:
            if _norm(c) == cand_group or cand_group in _norm(c) or _norm(c) in cand_group:
                group_col = c
                break
    if not group_col and categorical_cols:
        # Check direct column name, token/stem match, or distinct value question
        for c in categorical_cols:
            c_norm = _norm(c)
            c_lower = c.lower()
            if c_lower in ql or c_norm in ql or c_norm in _norm(ql):
                group_col = c
                break
            c_tokens = [t for t in re.findall(r"[a-z0-9]+", c_lower) if len(t) > 3]
            q_tokens = set(re.findall(r"[a-z0-9]+", ql))
            if any(any(qt == ct or (len(qt) > 3 and (qt.startswith(ct) or ct.startswith(qt))) for qt in q_tokens) for ct in c_tokens):
                group_col = c
                break
            # Match semantic phrases such as "tiers" -> "plan_tier"
            for part in c_lower.split("_"):
                if len(part) >= 3 and (part in ql or f"{part}s" in ql):
                    group_col = c
                    break
            if group_col:
                break
    # If the user explicitly requested a breakdown via preposition ("by X", "in each X", "per X")
    # but that dimension could not be resolved in the dataset, fail loudly instead of silently dropping it!
    unresolved_group_phrase: Optional[str] = None
    by_match_unresolved = re.search(r"\b(?:by|per|across|within|for each|in each)\s+([a-zA-Z0-9_ ]+?)(?:\s+after|\s+before|\s+with|\s+had|\s+has|\?|$|\.)", ql)
    if by_match_unresolved and not group_col:
        unresolved_group_phrase = by_match_unresolved.group(1).strip()
        # Filter out common non-column words
        if unresolved_group_phrase not in ("all", "default", "total", "any", "time"):
            found_dims_desc = ", ".join(f"'{c}'" for c in (categorical_cols[:4] if categorical_cols else cols[:4]))
            return ProposedInterpretation(
                question=q_clean,
                task=task,
                proposed_interpretation=f"The question requests a breakdown by '{unresolved_group_phrase}', but no matching dimension column is present in the dataset.",
                planned_calculation=f"Cannot execute grouped calculation: dimension column '{unresolved_group_phrase}' is missing.",
                target_metric=None,
                group_dimension=None,
                assumptions=[f"Available dimensions: {found_dims_desc}", f"Missing requested dimension: '{unresolved_group_phrase}'."],
                ambiguity_status=AmbiguityStatus.MISSING_DATA,
                clarification_prompt=(
                    f"You requested a breakdown by '{unresolved_group_phrase}', but no column matching '{unresolved_group_phrase}' exists in this dataset. "
                    f"Available dimensions are: {found_dims_desc}. "
                    f"Would you like to analyze an available dimension instead, or cancel the analysis?"
                ),
                clarification_options=[
                    f"Group by an available dimension ({categorical_cols[0] if categorical_cols else 'overall'})",
                    "Refuse analysis due to missing measures",
                ],
                can_auto_proceed=False,
                recommended_action=f"Fail loudly: do not silently substitute an overall aggregate when '{unresolved_group_phrase}' is requested.",
            )

    # 5. Extract Intended Target Metric and Check Missing Data vs Multiple Plausible Metrics
    # Check for direct named column in question first (e.g. deals_closed, active_days, net_revenue)
    direct_named_col: Optional[str] = None
    for c in cols:
        cn = _norm(c)
        if cn in ql or c.lower() in ql:
            # Check if this column is a metric or group dimension
            if c != group_col:
                direct_named_col = c
                break

    # Check for requested concepts in question
    metric_concepts = [
        ("profit", ["profit", "profitable", "net_profit", "gross_profit", "operating_profit"]),
        ("cost", ["cost", "total_cost", "cogs", "expense", "expenses"]),
        ("revenue", ["revenue", "net_revenue", "gross_revenue", "total_revenue", "sales"]),
        ("discount", ["discount", "discount_pct", "discount_rate", "discount_amount"]),
        ("margin", ["margin", "profit_margin", "gross_margin", "operating_margin"]),
        ("churn", ["churn", "churned", "attrition", "retention"]),
        ("quantity", ["quantity", "units", "units_sold", "volume", "order_volume"]),
        ("spend", ["spend", "ad_spend", "marketing_spend", "customer_spend"]),
    ]

    requested_concept: Optional[str] = None
    concept_candidates: List[str] = []
    for concept, synonyms in metric_concepts:
        if any(re.search(rf"\b{re.escape(syn)}\b", ql) for syn in synonyms):
            requested_concept = concept
            concept_candidates = synonyms
            break

    # Prioritize Causal on Observational data over ambiguity flags
    if (is_causal or is_diagnostic) and is_observational:
        # Avoid picking identifier columns (order_id, customer_id, etc.) as exposures
        non_id_cats = [c for c in categorical_cols if not any(id_k in c.lower() for id_k in ("_id", "id", "key", "pk", "fk", "uuid"))]
        exp_col = group_col or (non_id_cats[0] if non_id_cats else None)
        # Find best guess target metric
        cand_target = direct_named_col or (numeric_cols[0] if numeric_cols else "outcome")
        for c in numeric_cols:
            if requested_concept and any(syn in _norm(c) for syn in concept_candidates):
                cand_target = c
                break

        if not exp_col:
            # Diagnostic question with no candidate exposure variable
            preview_task = canonical_contract.task_family if canonical_contract else ("DIAGNOSTIC" if is_diagnostic else "CAUSAL")
            return ProposedInterpretation(
                question=q_clean,
                task=preview_task,
                proposed_interpretation=f"Evaluate drivers of {cand_target} using observational data.",
                planned_calculation=f"Cannot execute causal evaluation: no non-identifier explanatory exposure column found.",
                target_metric=cand_target,
                group_dimension=None,
                assumptions=[
                    "Dataset is observational and lacks experimental randomization.",
                    "No candidate explanatory exposure dimension found (identifier columns excluded).",
                ],
                ambiguity_status=AmbiguityStatus.CAUSAL_ON_OBSERVATIONAL,
                clarification_prompt=(
                    f"This question asks what caused or drove changes in {cand_target}, but this observational dataset "
                    f"lacks an identified causal exposure column (identifier columns like order_id/customer_id are excluded). "
                    f"Would you like to select an explanatory dimension to investigate, or proceed with descriptive trend analysis?"
                ),
                clarification_options=[
                    "Select an explanatory dimension to investigate",
                    "Proceed with descriptive trend analysis (non-causal)",
                    "Refuse causal claim without experimental design",
                ],
                can_auto_proceed=False,
                recommended_action="Do not select identifier columns as causal exposures; clarify explanatory dimension.",
                canonical_contract=canonical_contract,
                plan_hash=compute_resolved_plan_hash(contract=canonical_contract, dataset_fingerprint=None, schema_version=1) if canonical_contract else None,
            )

        return ProposedInterpretation(
            question=q_clean,
            task="CAUSAL",
            proposed_interpretation=f"Evaluate relationship between {exp_col} and {cand_target} using observational data.",
            planned_calculation=f"Calculate association metrics and 95% confidence intervals between {exp_col} and {cand_target}.",
            target_metric=cand_target,
            group_dimension=group_col,
            assumptions=[
                "Dataset is observational and lacks experimental randomization.",
                "Causation cannot be mathematically identified without an unconfounded identification design.",
            ],
            ambiguity_status=AmbiguityStatus.CAUSAL_ON_OBSERVATIONAL,
            clarification_prompt=(
                f"This dataset is observational and lacks experimental randomization. "
                f"A statistical analysis can establish association or correlation between {exp_col} and {cand_target}, "
                f"but cannot prove that {exp_col} caused the change. "
                f"Would you like to proceed with a qualified association analysis instead?"
            ),
            clarification_options=[
                f"Proceed with verified association analysis (non-causal)",
                f"Refuse causal claim without experimental design",
            ],
            can_auto_proceed=False,
            recommended_action="Execute qualified association analysis with explicit non-causal ceiling.",
        )

    # Find matching physical columns
    plausible_metric_cols: List[str] = []
    if requested_concept:
        for c in numeric_cols:
            cn = _norm(c)
            if any(syn in cn for syn in concept_candidates):
                plausible_metric_cols.append(c)

    # A. Check for Missing Required Measure
    if requested_concept and not plausible_metric_cols:
        found_measures_desc = ", ".join(f"'{c}'" for c in numeric_cols[:4]) if numeric_cols else "no numeric measures"
        alt_measures = [c for c in numeric_cols if any(k in c.lower() for k in ["revenue", "sales", "discount", "quantity", "units"])]
        alt_desc = " and ".join(f"'{m}'" for m in alt_measures[:2]) if alt_measures else "available numeric columns"

        return ProposedInterpretation(
            question=q_clean,
            task=task,
            proposed_interpretation=f"The question requests an analysis of '{requested_concept}', but no '{requested_concept}' measure is present in the dataset.",
            planned_calculation=f"Cannot execute calculation: missing required column for '{requested_concept}'.",
            target_metric=None,
            group_dimension=group_col,
            assumptions=[f"Found measures: {found_measures_desc}", f"Missing measure: '{requested_concept}'."],
            ambiguity_status=AmbiguityStatus.MISSING_DATA,
            clarification_prompt=(
                f"I found {found_measures_desc}, but no '{requested_concept}' or cost measure. "
                f"I cannot establish whether {requested_concept} changed from these columns alone. "
                f"Would you like me to investigate changes in {alt_desc} instead, or provide the data needed to analyze {requested_concept}?"
            ),
            clarification_options=[
                f"Investigate changes in {alt_desc} instead",
                f"Provide uploaded data containing '{requested_concept}'",
                "Refuse analysis due to missing measures",
            ],
            can_auto_proceed=False,
            recommended_action=f"Do not invent substitute for '{requested_concept}'; propose analyzing {alt_desc} instead.",
        )

    # B. Check for Multiple Plausible Metrics
    if requested_concept and len(plausible_metric_cols) > 1:
        # Check if question specifies an exact physical column name (e.g. "net_revenue")
        exact_matches = [c for c in plausible_metric_cols if c.lower() in ql or _norm(c) in ql]
        if len(exact_matches) == 1:
            # User was explicit! Clear ambiguity
            target_metric = exact_matches[0]
            plausible_metric_cols = [target_metric]
        else:
            default_candidate = plausible_metric_cols[0]
            for c in plausible_metric_cols:
                if "net" in c.lower() or "total" in c.lower():
                    default_candidate = c
                    break

            options = [f"{c.replace('_', ' ').capitalize()} (`{c}`)" for c in plausible_metric_cols]
            options.append("I need more information first")

            return ProposedInterpretation(
                question=q_clean,
                task=task,
                proposed_interpretation=f"Calculate total {requested_concept} grouped by {group_col or 'all records'}.",
                planned_calculation=f"GROUP BY {group_col or 'all'} SUM({default_candidate})",
                target_metric=default_candidate,
                group_dimension=group_col,
                assumptions=[
                    f"Grouping: {group_col or 'Overall'}",
                    f"{requested_concept.capitalize()} definition: `{default_candidate}` was selected. The dataset contains multiple {requested_concept} measures ({', '.join(f'`{c}`' for c in plausible_metric_cols)}). Confirm which measure your question intends.",
                ],
                ambiguity_status=AmbiguityStatus.AMBIGUOUS_METRIC,
                clarification_prompt=f"The dataset contains multiple plausible measures for '{requested_concept}': {', '.join(f'`{c}`' for c in plausible_metric_cols)}. Confirm which measure your question intends.",
                clarification_options=options,
                can_auto_proceed=False,
                recommended_action=f"Confirm intended metric: {', '.join(f'`{c}`' for c in plausible_metric_cols)}.",
            )

    # Target metric selection
    target_metric: Optional[str] = None
    if direct_named_col and direct_named_col in numeric_cols:
        target_metric = direct_named_col
    elif plausible_metric_cols:
        target_metric = plausible_metric_cols[0]
    elif direct_named_col:
        target_metric = direct_named_col
    elif numeric_cols:
        target_metric = numeric_cols[0]

    # D. Unambiguous / Routine Analysis
    is_counting_or_breakdown = bool(re.search(
        r"\b(different|distinct|unique|tiers?|categories|types?|breakdown|distribution|count|counting|how many|frequency|list each)\b",
        ql
    ))
    if is_counting_or_breakdown and group_col and group_col in categorical_cols:
        has_numeric_request = bool(requested_concept or (direct_named_col and direct_named_col in numeric_cols))
        if not has_numeric_request:
            planned_sql = f"SELECT {group_col}, COUNT(*) AS observation_count FROM data GROUP BY {group_col} ORDER BY observation_count DESC"
            interpretation = f"Count observations for each distinct {group_col.replace('_', ' ')} from {group_col} and report observation counts."
            return ProposedInterpretation(
                question=q_clean,
                task="DISTRIBUTION",
                proposed_interpretation=interpretation,
                planned_calculation=planned_sql,
                target_metric=None,
                group_dimension=group_col,
                assumptions=[
                    f"Grouping dimension: `{group_col}`",
                    "Metric: Count of observations (`COUNT(*)`)",
                    "Population: All available records",
                ],
                ambiguity_status=AmbiguityStatus.CLEAR,
                can_auto_proceed=True,
                recommended_action="Auto-proceed with categorical observation count; provide parameter adjustments.",
            )

    is_rate_q = bool(re.search(r"\b(churn rate|rate|percentage|ratio|proportion)\b", ql))
    rate_col = next((c for c in cols if any(k in c.lower() for k in ["churn", "converted", "cancelled", "active", "rate", "event"])), None)
    if is_rate_q and group_col and rate_col:
        planned_sql = f"SELECT {group_col}, AVG({rate_col}) * 100 AS {rate_col}_pct, COUNT(*) AS total_records FROM data GROUP BY {group_col} ORDER BY {rate_col}_pct DESC"
        interpretation = f"Calculate observed rate of {rate_col.replace('_', ' ')} across each {group_col.replace('_', ' ')}."
        return ProposedInterpretation(
            question=q_clean,
            task="DESCRIPTIVE",
            proposed_interpretation=interpretation,
            planned_calculation=planned_sql,
            target_metric=rate_col,
            group_dimension=group_col,
            assumptions=[
                f"Grouping dimension: `{group_col}`",
                f"Metric: `{rate_col}` aggregated with rate/proportion",
                "Observational rates describe group differences; they do not prove causal mechanism.",
            ],
            ambiguity_status=AmbiguityStatus.CLEAR,
            can_auto_proceed=True,
            recommended_action="Auto-proceed with descriptive rate calculation.",
        )

    if is_ranking and group_col and target_metric:
        # Use the contract's aggregation (never a hardcoded SUM) and bind the SUM default into
        # the contract so the approved plan, its hash and the controller gate all carry it.
        canonical_contract = bind_default_aggregation(canonical_contract)
        agg = (canonical_contract.requested_aggregation if canonical_contract is not None else None) or "SUM"
        defaulted = bool(
            canonical_contract is None or canonical_contract.resolution_evidence.get("defaulted_aggregation")
        )
        if agg == "COUNT_DISTINCT":
            agg_sql = f"COUNT(DISTINCT {target_metric})"
        else:
            agg_sql = f"{'AVG' if agg == 'MEAN' else agg}({target_metric})"
        planned_sql = f"SELECT {group_col}, {agg_sql} AS total_{target_metric}, COUNT(*) AS rows FROM data GROUP BY {group_col} ORDER BY total_{target_metric} DESC"
        interpretation = f"Calculate {agg.lower().replace('_', ' ')} of {target_metric.replace('_', ' ')} for each {group_col.replace('_', ' ')}, rank from highest to lowest, and identify the top {group_col.replace('_', ' ')}."
        return ProposedInterpretation(
            question=q_clean,
            task="RANKING",
            proposed_interpretation=interpretation,
            planned_calculation=planned_sql,
            target_metric=target_metric,
            group_dimension=group_col,
            assumptions=[
                f"Grouping dimension: `{group_col}`",
                f"Metric: `{target_metric}` aggregated with {agg}"
                + (" (default; not stated in your question)" if defaulted else ""),
                "Population: All available records with non-null values",
            ],
            ambiguity_status=AmbiguityStatus.CLEAR,
            can_auto_proceed=True,
            recommended_action="Auto-proceed with calculation; provide edit option.",
            canonical_contract=canonical_contract,
            plan_hash=compute_resolved_plan_hash(contract=canonical_contract, dataset_fingerprint=None, schema_version=1) if canonical_contract else None,
        )

    if (is_comparison or is_association) and group_col and target_metric:
        planned_sql = f"SELECT {group_col}, AVG({target_metric}) AS mean_{target_metric}, STDDEV({target_metric}) AS sd, COUNT(*) AS n FROM data GROUP BY {group_col}"
        interpretation = f"Compare {target_metric.replace('_', ' ')} across {group_col.replace('_', ' ')} with two-sample difference, 95% confidence interval, and p-value."
        return ProposedInterpretation(
            question=q_clean,
            task="COMPARISON" if is_comparison else "ASSOCIATION",
            proposed_interpretation=interpretation,
            planned_calculation=planned_sql,
            target_metric=target_metric,
            group_dimension=group_col,
            assumptions=[
                f"Comparison dimension: `{group_col}`",
                f"Target metric: `{target_metric}`",
                "Association/difference language never claims causation.",
            ],
            ambiguity_status=AmbiguityStatus.CLEAR,
            can_auto_proceed=True,
            recommended_action="Auto-proceed with comparison calculation; report confidence intervals.",
            canonical_contract=canonical_contract,
            plan_hash=compute_resolved_plan_hash(contract=canonical_contract, dataset_fingerprint=None, schema_version=1) if canonical_contract else None,
        )

    # Milestone M1: If canonical contract exists, derive the preview directly from the canonical contract
    # to guarantee that the plan shown to the analyst is identical to the canonical execution contract.
    if canonical_contract is not None:
        contract_task = canonical_contract.task_family
        contract_target = canonical_contract.target_column
        contract_groups = canonical_contract.grouping_columns or ()
        contract_group = contract_groups[0] if contract_groups else None

        # Check for unresolved roles in canonical contract
        if canonical_contract.unresolved_roles:
            unresolved_desc = ", ".join(canonical_contract.unresolved_roles)
            amb_status = AmbiguityStatus.MISSING_DATA
            if "explanatory" in canonical_contract.unresolved_roles or contract_task in ("ROOT_CAUSE", "CAUSAL_REQUEST"):
                amb_status = AmbiguityStatus.CAUSAL_ON_OBSERVATIONAL if is_observational else AmbiguityStatus.MISSING_DATA

            return ProposedInterpretation(
                question=q_clean,
                task=contract_task,
                proposed_interpretation=f"The question requires an analysis with analytical role(s) ({unresolved_desc}) that could not be resolved in the dataset.",
                planned_calculation=f"Cannot execute calculation: required analytical variables ({unresolved_desc}) are unresolved in the dataset.",
                target_metric=contract_target,
                group_dimension=contract_group,
                assumptions=[f"Unresolved roles: {unresolved_desc}."],
                ambiguity_status=amb_status,
                clarification_prompt=(
                    f"Your question asks for an analysis involving '{unresolved_desc}', but those roles could not be resolved "
                    f"to valid columns in the dataset. Would you like to select available measures or refine the question?"
                ),
                clarification_options=[
                    "Select an available measure",
                    "Refuse analysis due to missing measures",
                ],
                can_auto_proceed=False,
                recommended_action=f"Clarify unresolved analytical role(s): {unresolved_desc}.",
                canonical_contract=canonical_contract,
                plan_hash=compute_resolved_plan_hash(contract=canonical_contract, dataset_fingerprint=None, schema_version=1),
            )

        # Unambiguous contract: derive displayed plan directly from canonical contract properties
        if contract_task == "RANKING" and contract_group and contract_target:
            canonical_contract = bind_default_aggregation(canonical_contract)
            agg = canonical_contract.requested_aggregation or "SUM"
            planned_sql = f"SELECT {contract_group}, {agg}({contract_target}) AS total_{contract_target}, COUNT(*) AS rows FROM data GROUP BY {contract_group} ORDER BY total_{contract_target} DESC"
            interpretation = f"Calculate total {contract_target.replace('_', ' ')} for each {contract_group.replace('_', ' ')}, rank from highest to lowest, and identify the top {contract_group.replace('_', ' ')}."
            return ProposedInterpretation(
                question=q_clean,
                task="RANKING",
                proposed_interpretation=interpretation,
                planned_calculation=planned_sql,
                target_metric=contract_target,
                group_dimension=contract_group,
                assumptions=[
                    f"Grouping dimension: `{contract_group}`",
                    f"Metric: `{contract_target}` aggregated with {agg}"
                    + (" (default; not stated in your question)" if canonical_contract.resolution_evidence.get("defaulted_aggregation") else ""),
                    "Population: All available records with non-null values",
                ],
                ambiguity_status=AmbiguityStatus.CLEAR,
                can_auto_proceed=True,
                recommended_action="Auto-proceed with calculation; provide edit option.",
                canonical_contract=canonical_contract,
                plan_hash=compute_resolved_plan_hash(contract=canonical_contract, dataset_fingerprint=None, schema_version=1),
            )

        if contract_task in ("COMPARISON", "ASSOCIATION") and contract_group and contract_target:
            planned_sql = f"SELECT {contract_group}, AVG({contract_target}) AS mean_{contract_target}, STDDEV({contract_target}) AS sd, COUNT(*) AS n FROM data GROUP BY {contract_group}"
            interpretation = f"Compare {contract_target.replace('_', ' ')} across {contract_group.replace('_', ' ')} with two-sample difference, 95% confidence interval, and p-value."
            return ProposedInterpretation(
                question=q_clean,
                task=contract_task,
                proposed_interpretation=interpretation,
                planned_calculation=planned_sql,
                target_metric=contract_target,
                group_dimension=contract_group,
                assumptions=[
                    f"Comparison dimension: `{contract_group}`",
                    f"Target metric: `{contract_target}`",
                    "Association/difference language never claims causation.",
                ],
                ambiguity_status=AmbiguityStatus.CLEAR,
                can_auto_proceed=True,
                recommended_action="Auto-proceed with comparison calculation; report confidence intervals.",
                canonical_contract=canonical_contract,
                plan_hash=compute_resolved_plan_hash(contract=canonical_contract, dataset_fingerprint=None, schema_version=1),
            )

        if contract_task == "DESCRIPTIVE":
            canonical_contract = bind_default_aggregation(canonical_contract)
            agg = canonical_contract.requested_aggregation or "SUM"
            if contract_target and contract_group:
                planned_sql = f"SELECT {contract_group}, {agg}({contract_target}) FROM data GROUP BY {contract_group}"
                metric_desc = f"`{contract_target}` aggregated with {agg}"
                eff_metric = contract_target
            elif contract_target:
                planned_sql = f"SELECT {agg}({contract_target}) FROM data"
                metric_desc = f"`{contract_target}` aggregated with {agg}"
                eff_metric = contract_target
            elif contract_group:
                planned_sql = f"SELECT {contract_group}, COUNT(*) FROM data GROUP BY {contract_group}"
                metric_desc = "COUNT(*)"
                eff_metric = None
            else:
                planned_sql = "SELECT COUNT(*) FROM data"
                metric_desc = "COUNT(*)"
                eff_metric = None

            return ProposedInterpretation(
                question=q_clean,
                task="DESCRIPTIVE",
                proposed_interpretation=f"Descriptive analysis of {contract_target or 'records'} grouped by {contract_group or 'overall'}.",
                planned_calculation=planned_sql,
                target_metric=eff_metric,
                group_dimension=contract_group,
                assumptions=[f"Metric: {metric_desc}", f"Grouping: `{contract_group or 'overall'}`"],
                ambiguity_status=AmbiguityStatus.CLEAR,
                can_auto_proceed=True,
                recommended_action="Proceed with descriptive calculation.",
                canonical_contract=canonical_contract,
                plan_hash=compute_resolved_plan_hash(contract=canonical_contract, dataset_fingerprint=None, schema_version=1),
            )

    # General descriptive fallback
    if target_metric and target_metric in categorical_cols:
        calc_str = f"COUNT({target_metric})"
        metric_desc = f"`{target_metric}` (count)"
        eff_metric = None
    elif target_metric:
        calc_str = f"SUM({target_metric})"
        metric_desc = f"`{target_metric}`"
        eff_metric = target_metric
    else:
        calc_str = "COUNT(*)"
        metric_desc = "rows"
        eff_metric = None
    planned_sql = f"SELECT {group_col or 'all'}, {calc_str} FROM data"
    preview = ProposedInterpretation(
        question=q_clean,
        task=task,
        proposed_interpretation=f"Analyze {q_clean} using available data.",
        planned_calculation=planned_sql,
        target_metric=eff_metric,
        group_dimension=group_col,
        assumptions=[f"Metric: {metric_desc}", f"Grouping: `{group_col or 'overall'}`"],
        ambiguity_status=AmbiguityStatus.CLEAR,
        can_auto_proceed=True,
        recommended_action="Proceed with calculation.",
    )
    if canonical_contract is not None:
        preview.canonical_contract = canonical_contract
        preview.plan_hash = compute_resolved_plan_hash(contract=canonical_contract, dataset_fingerprint=None, schema_version=1)
    return preview


def apply_human_review_decision(
    investigation_id: str,
    action: str,
    actor: str,
    comments: Optional[str] = None,
    modified_parameters: Optional[Dict[str, Any]] = None,
    verification_passed: bool = False,
) -> Dict[str, Any]:
    """Record human review decision for an investigation while maintaining verification invariants.

    INVARIANT (MindEd 1.0.0):
    Human approval NEVER overrides a failed or inconclusive verification check.
    If verification did not pass, an 'ACCEPT' action accepts the result only for the
    analyst's local workflow and records that the underlying verification remains unverified.
    """
    action_upper = action.upper()
    if action_upper not in {a.value for a in HumanReviewAction}:
        raise ValueError(f"Invalid review action: '{action}'. Must be one of {[a.value for a in HumanReviewAction]}.")

    record: Dict[str, Any] = {
        "investigation_id": investigation_id,
        "action": action_upper,
        "actor": actor,
        "comments": comments or "",
        "modified_parameters": dict(modified_parameters or {}),
        "machine_verification_passed": bool(verification_passed),
    }

    if action_upper == HumanReviewAction.ACCEPT.value:
        if not verification_passed:
            record["review_status"] = "ACCEPTED_FOR_WORKFLOW_UNVERIFIED"
            record["notice"] = (
                "Accepted for analyst's workflow. Note: Machine verification did not pass; "
                "human approval does not override verification checks or certify mathematical validity."
            )
            record["effective_verification_status"] = "UNVERIFIED"
        else:
            record["review_status"] = "ACCEPTED_AND_VERIFIED"
            record["notice"] = "Accepted for workflow with verified mathematical calculations."
            record["effective_verification_status"] = "VERIFIED"

    elif action_upper == HumanReviewAction.REQUEST_CORRECTION.value:
        record["review_status"] = "CORRECTION_REQUESTED"
        record["notice"] = f"Correction requested by {actor}: '{comments}'."
        record["effective_verification_status"] = "NEEDS_REWORK"

    elif action_upper == HumanReviewAction.REJECT.value:
        if not comments or len(comments.strip()) < 3:
            raise ValueError("Rejection requires a documented reason.")
        record["review_status"] = "REJECTED"
        record["notice"] = f"Analysis rejected by {actor}. Reason: '{comments}'."
        record["effective_verification_status"] = "REJECTED"

    elif action_upper == HumanReviewAction.REDO.value:
        record["review_status"] = "REDO_REQUESTED"
        record["notice"] = f"Redo requested with revised parameters: {modified_parameters}."
        record["effective_verification_status"] = "NOT_RUN"

    return record



def render_contract_preview_from_contract(contract: Any) -> Optional[str]:
    """Render a review-only deterministic calculation from canonical contract fields."""
    graph_payload = getattr(contract, "operation_graph", None)
    if isinstance(graph_payload, dict) and graph_payload.get("nodes"):
        from dataclasses import replace as _replace_contract
        from packages.analytics_core.src.intelligence.analytical_operations import OperationGraph, OperationNode
        try:
            graph = OperationGraph(
                version=int(graph_payload.get("version", 0)),
                nodes=tuple(OperationNode(
                    node_id=str(item["node_id"]), operation=str(item["operation"]),
                    params=dict(item.get("params") or {}), depends_on=tuple(item.get("depends_on") or ()),
                ) for item in graph_payload.get("nodes", [])),
                requested_outputs=tuple(graph_payload.get("requested_outputs") or ()),
            )
            base_contract = _replace_contract(contract, operation_graph=None)
            base_sql = render_contract_preview_from_contract(base_contract)
            if not base_sql:
                return None
            steps = [f"STEP 1 [{graph.nodes[0].node_id}]: {base_sql}"]
            for index, node in enumerate(graph.nodes[1:], start=2):
                if node.operation == "RANK":
                    direction = str(node.params.get("direction", "DESC")).upper()
                    steps.append(f"STEP {index} [{node.node_id}]: RANK the prior grouped result {direction} (depends on {', '.join(node.depends_on)})")
                else:
                    steps.append(f"STEP {index} [{node.node_id}]: {node.operation} (depends on {', '.join(node.depends_on)})")
            return "OPERATION GRAPH " + graph.graph_hash() + " :: " + " ; ".join(steps)
        except (ImportError, KeyError, TypeError, ValueError):
            return None
    target = getattr(contract, "target_column", None)
    groups = tuple(getattr(contract, "grouping_columns", ()) or ())
    agg = str(getattr(contract, "requested_aggregation", None) or "").upper()
    task = str(getattr(contract, "task_family", ""))
    if not agg and target and (task == "RANKING" or (task == "DESCRIPTIVE" and not groups)):
        agg = "SUM"
    if agg in {"AVG", "AVERAGE"}:
        agg = "MEAN"
    if not agg:
        return None
    group = groups[0] if groups else None
    qident = lambda value: '"' + str(value).replace('"', '""') + '"'
    if agg in {"ROW_COUNT", "COUNT"} and not target:
        measure = "COUNT(*)"
    elif agg == "COUNT_DISTINCT" and target:
        measure = f"COUNT(DISTINCT {qident(target)})"
    elif agg == "RATE" and target:
        measure = f"AVG(CAST({qident(target)} AS DOUBLE)) * 100"
    elif agg in {"SUM", "MEAN", "MEDIAN", "MIN", "MAX", "COUNT", "VARIANCE", "STDDEV"} and target:
        sql_agg = {"MEAN": "AVG", "COUNT": "COUNT", "VARIANCE": "VAR_SAMP", "STDDEV": "STDDEV_SAMP"}.get(agg, agg)
        measure = f"{sql_agg}({qident(target)})"
    elif agg == "PERCENTILE" and target:
        percentile = (getattr(contract, "resolution_evidence", {}) or {}).get("percentile")
        if percentile is None or not isinstance(percentile, (int, float)) or not 0 <= float(percentile) <= 100:
            return None
        measure = f"QUANTILE_CONT({qident(target)}, {float(percentile) / 100.0})"
    elif agg == "ROW_COUNT":
        measure = "COUNT(*)"
    else:
        return None
    filters = list(getattr(contract, "filter_conditions", ()) or ())
    where_parts = []
    for condition in filters:
        col = qident(condition.get("column", ""))
        op = str(condition.get("operator", "")).upper()
        value = condition.get("value")
        if op == "IS NULL" or (op == "=" and value is None):
            where_parts.append(f"{col} IS NULL")
        elif op == "IS NOT NULL" or (op == "!=" and value is None):
            where_parts.append(f"{col} IS NOT NULL")
        elif op in {"=", "!=", ">", ">=", "<", "<="}:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                literal = str(value)
            elif isinstance(value, bool):
                literal = "TRUE" if value else "FALSE"
            else:
                literal = "'" + str(value).replace("'", "''") + "'"
            where_parts.append(f"{col} {op} {literal}")
        elif op in {"IN", "NOT IN"} and isinstance(value, (list, tuple)) and value:
            literals = []
            for item in value:
                if item is None:
                    return None  # SQL IN with NULL needs explicit three-valued-logic semantics.
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    literals.append(str(item))
                elif isinstance(item, bool):
                    literals.append("TRUE" if item else "FALSE")
                else:
                    literals.append("'" + str(item).replace("'", "''") + "'")
            where_parts.append(f"{col} {op} ({', '.join(literals)})")
        else:
            return None
    select_parts = [qident(item) for item in groups] + [f"{measure} AS metric_value"]
    sql = "SELECT " + ", ".join(select_parts) + " FROM data"
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    if groups:
        sql += " GROUP BY " + ", ".join(qident(item) for item in groups)
    if task == "RANKING" and groups:
        direction = str(getattr(contract, "ranking_direction", None) or "DESC").upper()
        sql += f" ORDER BY metric_value {'ASC' if direction == 'ASC' else 'DESC'}"
    return sql


def evaluate_guided_investigation(
    question: str,
    df: Optional[pd.DataFrame] = None,
    semantic: Any = None,
    available_columns: Optional[List[str]] = None,
    tables_info: Optional[Dict[str, List[str]]] = None,
    table_relationships: Optional[List[Dict[str, Any]]] = None,
    is_observational: bool = True,
    verification_passed: Optional[bool] = None,
) -> ProposedInterpretation:
    """Build the visible plan from the canonical contract and refuse silent clause loss.

    The legacy analyzer still detects specialized safety conditions (causal, predictive, joins),
    but it cannot make a plan executable if the canonical contract or material-clause ledger is
    incomplete. For supported simple aggregates, display text is regenerated from contract fields.
    """
    preview = _evaluate_guided_investigation_legacy(
        question=question,
        df=df,
        semantic=semantic,
        available_columns=available_columns,
        tables_info=tables_info,
        table_relationships=table_relationships,
        is_observational=is_observational,
        verification_passed=verification_passed,
    )
    contract = preview.canonical_contract
    if df is not None:
        from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
        from dataclasses import replace
        try:
            # Compile from source question once. Do not trust the legacy heuristic's fallback
            # fields as authority if it did not carry the compiled contract through an early return.
            compiled = compile_canonical_question_contract(question, df)
            contract = compiled
            contract = bind_default_aggregation(contract)
            # Build a persisted graph only for the one verified compound grammar supported by
            # the operation engine. Other compound questions remain unresolved by the ledger.
            if not getattr(contract, "operation_graph", None):
                try:
                    from packages.analytics_core.src.intelligence.analytical_operations import build_supported_aggregate_then_rank_graph
                    supported_graph = build_supported_aggregate_then_rank_graph(question, contract, list(map(str, df.columns)))
                    if supported_graph is not None:
                        contract = replace(contract, operation_graph=supported_graph.to_dict())
                except (ImportError, TypeError, ValueError):
                    pass
            # An explicit rank request must not turn identifier keys into additive measures.
            # If both requested roles are identifier-like, expose open slots for clarification
            # rather than inventing a numerical ranking over IDs.
            is_id_name = lambda value: bool(value and re.search(
                r"(?:^|[_\W])(id|key|uuid|guid|identifier)(?:$|[_\W])|(?:_id|_key|_uuid|_guid)$",
                str(value).casefold(),
            ))
            target_is_id = is_id_name(getattr(contract, "target_column", None))
            groups_before = tuple(getattr(contract, "grouping_columns", ()) or ())
            group_is_id = bool(groups_before) and all(is_id_name(value) for value in groups_before)
            if str(getattr(contract, "task_family", "")).upper() == "RANKING" and target_is_id and group_is_id:
                contract = replace(
                    contract,
                    target_column=None,
                    grouping_columns=(),
                    unresolved_roles=tuple(dict.fromkeys([
                        *(getattr(contract, "unresolved_roles", ()) or ()),
                        "target_column", "grouping_columns",
                    ])),
                    completeness="INCOMPLETE",
                    unresolved_semantics=tuple(dict.fromkeys([
                        *(getattr(contract, "unresolved_semantics", ()) or ()),
                        "identifier_fields_cannot_define_requested_numeric_ranking",
                    ])),
                )
            contract, ledger, requirements, option_patches = enrich_contract_intent_state(
                question=question, contract=contract, frame=df, guidance={}
            )
        except Exception as exc:
            requirements = [{"slot": "contract", "status": "UNRESOLVED", "reason": f"contract_validation_error:{type(exc).__name__}"}]
            ledger, option_patches = [], []
            contract = None
    else:
        ledger, requirements, option_patches = [], [], []

    if contract is not None:
        preview.canonical_contract = contract
        preview.clarification_option_patches = option_patches
        preview.pending_intent_slot = next(
            (str(item.get("slot")) for item in requirements if item.get("slot")), None
        )
        # Even an unresolved draft must display the contract's values, never stale values from
        # the legacy heuristic parser. This keeps the visible plan honest while asking clarification.
        preview.target_metric = getattr(contract, "target_column", None)
        _contract_groups = tuple(getattr(contract, "grouping_columns", ()) or ())
        preview.group_dimension = _contract_groups[0] if _contract_groups else None
        # A CLEAR result may not use a heuristic target/group that contradicts the contract.
        # Render supported operations from the contract so the display and approved hash agree.
        contract_target = getattr(contract, "target_column", None)
        contract_groups = tuple(getattr(contract, "grouping_columns", ()) or ())
        contract_group = contract_groups[0] if contract_groups else None
        _material_intent_open = bool(
            requirements
            or getattr(contract, "unresolved_roles", ())
            or (getattr(contract, "task_family", "") == "GENERAL_EXPLORATION" and not contract_target and preview.target_metric)
            or getattr(contract, "completeness", "NOT_ENFORCED") == "INCOMPLETE"
        )
        # A non-CLEAR legacy status does not make its fallback calculation trustworthy. Never
        # leave an executable-looking calculation visible behind a safety/ambiguity gate.
        if preview.ambiguity_status != AmbiguityStatus.CLEAR and not _material_intent_open:
            preview.can_auto_proceed = False
            preview.plan_hash = None
            preview.planned_calculation = "Not executable: the existing ambiguity or safety gate has not cleared."
            if not preview.clarification_prompt:
                preview.clarification_prompt = "Resolve the existing safety or ambiguity gate before execution."
        # Always replace a heuristic calculation when the canonical contract has open material
        # intent, including MISSING_DATA/AMBIGUOUS_SCOPE previews (not only originally-CLEAR ones).
        if preview.ambiguity_status == AmbiguityStatus.CLEAR or _material_intent_open:
            if _material_intent_open:
                _protected_statuses = {
                    AmbiguityStatus.JOIN_FANOUT_RISK,
                    AmbiguityStatus.CAUSAL_ON_OBSERVATIONAL,
                    AmbiguityStatus.VERIFICATION_FAILED,
                    AmbiguityStatus.UNEXPECTED_ASSUMPTION,
                }
                if preview.ambiguity_status not in _protected_statuses:
                    preview.ambiguity_status = (
                        AmbiguityStatus.AMBIGUOUS_SCOPE
                        if any(item.get("slot") in {"time_column", "time_range"} for item in requirements)
                        else AmbiguityStatus.MISSING_DATA
                    )
                preview.can_auto_proceed = False
                slots = list(dict.fromkeys(str(item.get("slot")) for item in requirements))
                if getattr(contract, "unresolved_roles", ()):
                    slots.extend(str(item) for item in contract.unresolved_roles if str(item) not in slots)
                if any(item.get("slot") == "operation_graph" for item in requirements):
                    preview.clarification_prompt = (
                        "This question contains multiple analytical operations, but the current approved intent "
                        "contract does not yet represent them as separate executable nodes. Split it into one "
                        "analytical question per operation; no partial calculation will be run."
                    )
                else:
                    preview.clarification_prompt = (
                        "MindEd found a plausible interpretation, but the canonical contract does not yet "
                        "account for every material part of the question. Resolve these fields before calculation: "
                        + (", ".join(slots) if slots else "target, grouping, or requested operation")
                        + "."
                    )
                preview.clarification_options = [item["label"] for item in option_patches]
                if not preview.clarification_options:
                    if requirements and any(item.get("slot") == "denominator_column" for item in requirements):
                        preview.clarification_options = [
                            "Specify the denominator/entity definition for the per-customer or per-unit metric",
                            "Provide a dataset column or documented metric definition for the denominator",
                        ]
                    elif requirements and any(item.get("slot") == "filter_conditions" for item in requirements):
                        preview.clarification_options = [
                            "Select the population filter to apply",
                            "Clarify the intended inclusion/exclusion condition",
                        ]
                    else:
                        preview.clarification_options = ["Clarify the unresolved analytical requirement"]
                if any(item.get("slot") == "operation_graph" for item in requirements):
                    preview.clarification_options = ["Split into separate single-operation questions"]
                    preview.clarification_option_patches = []
                else:
                    preview.clarification_option_patches = option_patches
                preview.planned_calculation = "Not executable yet: one or more material intent slots remain unresolved."
                preview.recommended_action = "Resolve the open intent slots; do not calculate from a fallback interpretation."
                preview.proposed_interpretation = (
                    "The current draft is not approved for execution because its meaning is incomplete. "
                    + preview.clarification_prompt
                )
                preview.plan_hash = None
            else:
                contract_sql = render_contract_preview_from_contract(contract)
                if contract_sql:
                    preview.target_metric = contract_target
                    contract_groups = tuple(getattr(contract, "grouping_columns", ()) or ())
                    preview.group_dimension = ", ".join(contract_groups) if contract_groups else None
                    preview.planned_calculation = contract_sql
                    agg = getattr(contract, "requested_aggregation", None) or "SUM"
                    if contract_groups:
                        preview.proposed_interpretation = (
                            f"Calculate {str(agg).lower()} of {contract_target or 'row count'} by {', '.join(contract_groups)}; "
                            "using only the filters recorded in the canonical contract."
                        )
                    elif contract_target:
                        preview.proposed_interpretation = f"Calculate {str(agg).lower()} of {contract_target} using the canonical contract."
                    preview.plan_hash = compute_resolved_plan_hash(contract=contract, dataset_fingerprint=None, schema_version=1)
        if preview.ambiguity_status != AmbiguityStatus.CLEAR:
            preview.can_auto_proceed = False
            preview.plan_hash = None
        # Preserve the exact ledger for audit and tests. It is regenerated from the same source
        # question and schema, not assembled from the Q&A transcript.
        prior_evidence = dict(getattr(contract, "resolution_evidence", {}) or {})
        preview.canonical_contract = replace(
            contract,
            resolution_evidence={
                **prior_evidence,
                "intent_ledger_summary": {
                    "entry_count": len(ledger),
                    "unresolved_count": len(requirements),
                },
            },
        )
    return preview


def _coverage_score(question: str, frame: Any) -> int:
    words = set(re.findall(r"[a-z0-9]+", str(question).lower()))
    stems = {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words} | words
    score = 0
    for column in frame.columns:
        parts = [p for p in re.split(r"[^a-z0-9]+", str(column).lower()) if p]
        if not parts:
            continue
        full = "_".join(parts)
        if full in stems or all(p in stems for p in parts):
            score += 2
        elif any(
            len(p) > 3 and (p in stems or any(len(w) >= 5 and p[:5] == w[:5] for w in words))
            for p in parts
        ):
            score += 1
    return score


def unique_best_dataset(question: str, frames: Mapping[str, Any]) -> Optional[str]:
    """Name of the one dataset that covers the question strictly better than every other, else None."""
    scored = sorted(((_coverage_score(question, f), n) for n, f in frames.items()), reverse=True)
    if len(scored) >= 2 and scored[0][0] > 0 and scored[0][0] > scored[1][0]:
        return scored[0][1]
    return None


def select_primary_dataset(question: str, datasets_map: Mapping[str, Any]) -> Tuple[Optional[str], Any]:
    """Pick the dataset whose columns the question refers to (first loaded when no signal)."""
    if not datasets_map:
        return None, None
    best = unique_best_dataset(question, datasets_map) or next(iter(datasets_map))
    return best, datasets_map[best]
