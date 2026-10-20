"use client";

import React, { useState, useMemo } from "react";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Cpu,
  Database,
  Filter,
  HelpCircle,
  Layers,
  Play,
  Plus,
  RotateCcw,
  ShieldAlert,
  Trash2,
  Info,
  Sparkles,
  TableProperties,
} from "lucide-react";
import { PreAnalysisPlan } from "../lib/api";

interface ProposedAnalysisPanelProps {
  plan: PreAnalysisPlan;
  availableDatasets?: Array<{ id: string; name: string }>;
  availableColumns?: string[];
  availableNumericCols?: string[];
  availableCategoricalCols?: string[];
  availableBooleanCols?: string[];
  /** Fingerprint of the guidance that has already been applied to the visible preview. */
  previewedGuidanceKey?: string;
  onConfirmAndRun: (customizedPlan: {
    selected_metric?: string;
    selected_aggregation?: string;
    selected_group_dimension?: string;
    selected_dataset?: string;
    confirmed_assumptions?: string[];
    scope_filter?: string;
    clarification_response?: string;
    custom_instruction?: string;
    intent_patch?: Record<string, unknown>;
    pending_intent_slot?: string;
    plan_hash?: string;
  }) => void;
  onCancel: () => void;
  isRunning?: boolean;
}

/**
 * Infer the aggregation the question asks for. Whole words only, and the aggregation WORD wins
 * over ranking words: "highest total revenue" is a SUM that is then ranked -- it is not MAX.
 * MIN/MAX are used only when the question explicitly asks for a minimum/maximum value.
 */
export function inferAggregation(question: string): string {
  const q = question.toLowerCase();
  const has = (re: RegExp) => re.test(q);
  if (has(/\b(distinct|unique)\b/)) return "COUNT_DISTINCT";
  if (has(/\b(how many|number of|count|different (tiers|types|categories))\b/)) return "COUNT";
  if (has(/\b(average|avg|mean)\b/)) return "AVG";
  if (has(/\bmedian\b/)) return "MEDIAN";
  if (has(/\b(total|sum|overall|combined|in total)\b/)) return "SUM";
  if (has(/\b(minimum|min)\b/)) return "MIN";
  if (has(/\b(maximum|max)\b/)) return "MAX";
  return "SUM";
}

// An analyst's edits must survive the panel being re-created (for example after answering a
// dataset clarification). They are kept per question for the browser session only.
function loadSaved(question: string): { metric?: string; aggregation?: string; group?: string } {
  try {
    const raw = window.sessionStorage.getItem(`minded.plan.${question}`);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}
function saveSaved(question: string, value: { metric?: string; aggregation?: string; group?: string }) {
  try {
    window.sessionStorage.setItem(`minded.plan.${question}`, JSON.stringify(value));
  } catch {
    /* storage unavailable: edits simply are not remembered */
  }
}

type CanonicalFilterCondition = {
  column?: string;
  operator?: string;
  value?: unknown;
};

function contractFilters(plan: PreAnalysisPlan): CanonicalFilterCondition[] {
  const contract = plan.canonical_contract as { filter_conditions?: unknown } | null | undefined;
  if (!Array.isArray(contract?.filter_conditions)) return [];
  return contract.filter_conditions.filter(
    (item): item is CanonicalFilterCondition =>
      Boolean(item && typeof item === "object" &&
        typeof (item as CanonicalFilterCondition).column === "string" &&
        typeof (item as CanonicalFilterCondition).operator === "string")
  );
}

function displayFilter(filter: CanonicalFilterCondition): string {
  const column = filter.column || "<unknown column>";
  const operator = (filter.operator || "").toUpperCase();
  if (operator === "IS NULL" || operator === "IS NOT NULL") return `${column} ${operator}`;
  const quote = (item: unknown): string => typeof item === "string" ? `'${item.replace(/'/g, "''")}'` : item === null || item === undefined ? "NULL" : String(item);
  const value = operator === "BETWEEN" && Array.isArray(filter.value) && filter.value.length === 2
    ? `${quote(filter.value[0])} AND ${quote(filter.value[1])}`
    : Array.isArray(filter.value)
      ? `(${filter.value.map(quote).join(", ")})`
      : quote(filter.value);
  return `${column} ${operator} ${value}`;
}

function sqlIdentifier(value: string): string {
  return `"${value.replace(/"/g, '""')}"`;
}

function sqlFilter(filter: CanonicalFilterCondition): string {
  const column = sqlIdentifier(filter.column || "unknown");
  const operator = (filter.operator || "").toUpperCase();
  if (operator === "IS NULL" || operator === "IS NOT NULL") return `${column} ${operator}`;
  const literal = (value: unknown): string => {
    if (value === null || value === undefined) return "NULL";
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    return `'${String(value).replace(/'/g, "''")}'`;
  };
  if ((operator === "IN" || operator === "NOT IN") && Array.isArray(filter.value)) {
    return `${column} ${operator} (${filter.value.map(literal).join(", ")})`;
  }
  if (operator === "BETWEEN" && Array.isArray(filter.value) && filter.value.length === 2) {
    return `${column} BETWEEN ${literal(filter.value[0])} AND ${literal(filter.value[1])}`;
  }
  return `${column} ${operator} ${literal(filter.value)}`;
}

type FilterOperator = "=" | "!=" | ">" | ">=" | "<" | "<=" | "IN" | "NOT IN" | "BETWEEN" | "IS NULL" | "IS NOT NULL";
type EditableFilter = { id: string; column: string; operator: FilterOperator; value: string };
const FILTER_OPERATORS: Array<{ value: FilterOperator; label: string }> = [
  { value: "=", label: "is" },
  { value: "!=", label: "is not" },
  { value: ">", label: "greater than" },
  { value: ">=", label: "at least" },
  { value: "<", label: "less than" },
  { value: "<=", label: "at most" },
  { value: "IN", label: "is one of" },
  { value: "NOT IN", label: "is not one of" },
  { value: "BETWEEN", label: "is between" },
  { value: "IS NULL", label: "is empty / NULL" },
  { value: "IS NOT NULL", label: "is not empty / NULL" },
];

function quoteSqlLiteral(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

function compileEditableFilters(
  rows: EditableFilter[],
  availableColumns: string[],
  numericColumns: string[],
  booleanColumns: string[],
): { sql: string | null; error: string | null } {
  const active = rows.filter((row) => Boolean(row.column.trim() || row.value.trim() || row.operator !== "="));
  if (!active.length) return { sql: null, error: null };
  const allowed = new Set(availableColumns.map((value) => value.trim().toLowerCase()));
  const numeric = new Set(numericColumns.map((value) => value.toLowerCase()));
  const booleans = new Set(booleanColumns.map((value) => value.toLowerCase()));
  const compiled: string[] = [];
  for (let index = 0; index < active.length; index += 1) {
    const row = active[index];
    const column = row.column.trim();
    const op = row.operator;
    if (!column) return { sql: null, error: `Choose a column for filter ${index + 1}.` };
    if (!allowed.has(column.toLowerCase())) {
      return { sql: null, error: `Column '${column}' is not available in the selected dataset. Refresh the dataset preview and choose a listed column.` };
    }
    if (op === "IS NULL" || op === "IS NOT NULL") {
      compiled.push(`${sqlIdentifier(column)} ${op}`);
      continue;
    }
    if (!row.value.trim()) return { sql: null, error: `Enter a value for filter ${index + 1}.` };
    const parts = op === "IN" || op === "NOT IN" || op === "BETWEEN"
      ? row.value.split(",").map((value) => value.trim())
      : [row.value.trim()];
    if (parts.some((value) => !value)) return { sql: null, error: `Filter ${index + 1} contains an empty value.` };
    if (op === "BETWEEN" && parts.length !== 2) return { sql: null, error: `Filter ${index + 1}: enter exactly two values separated by a comma for BETWEEN.` };
    if ((op === "IN" || op === "NOT IN") && parts.length < 1) return { sql: null, error: `Filter ${index + 1} requires at least one list value.` };
    const isNumeric = numeric.has(column.toLowerCase());
    const isBoolean = booleans.has(column.toLowerCase());
    if (isBoolean) {
      if ((op !== "=" && op !== "!=" && op !== "IN" && op !== "NOT IN") || parts.some((value) => !/^(true|false)$/i.test(value))) {
        return { sql: null, error: `Boolean column '${column}' accepts true/false with is, is not, is one of, or is not one of.` };
      }
    }
    if (isNumeric && parts.some((value) => !/^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$/.test(value))) {
      return { sql: null, error: `Numeric column '${column}' requires a numeric value.` };
    }
    if (op === "BETWEEN" && isNumeric && Number(parts[0]) > Number(parts[1])) {
      return { sql: null, error: `Filter ${index + 1}: the lower BETWEEN bound must not exceed the upper bound.` };
    }
    const literal = (value: string) => {
      if (isNumeric) return value;
      if (isBoolean) return value.toLowerCase();
      return quoteSqlLiteral(value);
    };
    if (op === "IN" || op === "NOT IN") {
      compiled.push(`${sqlIdentifier(column)} ${op} (${parts.map(literal).join(", ")})`);
    } else if (op === "BETWEEN") {
      compiled.push(`${sqlIdentifier(column)} BETWEEN ${literal(parts[0])} AND ${literal(parts[1])}`);
    } else {
      compiled.push(`${sqlIdentifier(column)} ${op} ${literal(parts[0])}`);
    }
  }
  return { sql: compiled.join(" AND "), error: null };
}

export default function ProposedAnalysisPanel({
  plan,
  availableDatasets = [],
  availableColumns = [],
  availableNumericCols = [],
  availableCategoricalCols = [],
  availableBooleanCols = [],
  previewedGuidanceKey = "",
  onConfirmAndRun,
  onCancel,
  isRunning = false,
}: ProposedAnalysisPanelProps) {
  const saved = useMemo(() => (typeof window === "undefined" ? {} : loadSaved(plan.question || "")), [plan.question]);
  const contract = (plan.canonical_contract || {}) as Record<string, any>;
  const retainedGuidance = plan.guidance_context || {};
  const contractMetric = typeof contract.target_column === "string" ? contract.target_column : undefined;
  const contractGroup = Array.isArray(contract.grouping_columns) ? contract.grouping_columns[0] : undefined;
  const contractAggRaw = String(contract.requested_aggregation || "").toUpperCase();
  const contractAgg = ({ MEAN: "AVG", AVERAGE: "AVG", TOTAL: "SUM", ROW_COUNT: "COUNT" } as Record<string, string>)[contractAggRaw] || contractAggRaw;
  // Session values are only safe to restore from an earlier run when this question's current
  // canonical preview is executable. Otherwise stale selections can masquerade as parsed intent.
  const canRestoreSavedParameters = plan.ambiguity_status === "CLEAR";
  const savedMetric = canRestoreSavedParameters ? saved.metric : undefined;
  const savedAggregation = canRestoreSavedParameters ? saved.aggregation : undefined;
  const savedGroup = canRestoreSavedParameters ? saved.group : undefined;
  const [selectedMetric, setSelectedMetric] = useState<string>(
    retainedGuidance.selected_metric ?? contractMetric ?? plan.target_metric ?? savedMetric ?? ""
  );
  const initialAgg = useMemo(() => inferAggregation(plan.question || ""), [plan.question]);
  const hasExplicitAggregation = /\b(total|sum|average|avg|mean|median|minimum|min|maximum|max|count|distinct|unique)\b/i.test(plan.question || "");

  const [selectedAggregation, setSelectedAggregation] = useState<string>(
    retainedGuidance.selected_aggregation ?? (contractAgg || (!hasExplicitAggregation ? savedAggregation : undefined) || initialAgg)
  );
  const [selectedGroup, setSelectedGroup] = useState<string>(
    retainedGuidance.selected_group_dimension ?? contractGroup ?? plan.group_dimension ?? savedGroup ?? ""
  );
  // A prior edit may be restored only when the current canonical preview has no resolved
  // value for that slot. Resolved question intent must win over sessionStorage from an older run.
  const [metricTouched, setMetricTouched] = useState(
    Boolean(retainedGuidance.selected_metric) || (!contractMetric && !plan.target_metric && canRestoreSavedParameters && saved.metric !== undefined)
  );
  const [aggregationTouched, setAggregationTouched] = useState(
    Boolean(retainedGuidance.selected_aggregation) || (!contractAgg && !hasExplicitAggregation && canRestoreSavedParameters && saved.aggregation !== undefined)
  );
  const [groupTouched, setGroupTouched] = useState(
    Boolean(retainedGuidance.selected_group_dimension) || (!contractGroup && !plan.group_dimension && canRestoreSavedParameters && saved.group !== undefined)
  );
  React.useEffect(() => {
    const v: { metric?: string; aggregation?: string; group?: string } = {};
    if (metricTouched) v.metric = selectedMetric;
    if (aggregationTouched) v.aggregation = selectedAggregation;
    if (groupTouched) v.group = selectedGroup;
    if (Object.keys(v).length) saveSaved(plan.question || "", v);
  }, [metricTouched, aggregationTouched, groupTouched, selectedMetric, selectedAggregation, selectedGroup, plan.question]);
  const [selectedDataset, setSelectedDataset] = useState<string>(retainedGuidance.selected_dataset || "");
  const [selectedScope, setSelectedScope] = useState<string>(retainedGuidance.scope_filter ? "CUSTOM" : "ALL");
  const [scopeTouched, setScopeTouched] = useState<boolean>(Boolean(retainedGuidance.scope_filter));
  const [customFilter, setCustomFilter] = useState<string>(retainedGuidance.scope_filter || "");
  const [filterMode, setFilterMode] = useState<"builder" | "sql">(retainedGuidance.scope_filter ? "sql" : "builder");
  const availableFilterColumns = useMemo(
    () => Array.from(new Set([...(availableColumns || []), ...availableCategoricalCols, ...availableNumericCols, ...availableBooleanCols])),
    [availableColumns, availableCategoricalCols, availableNumericCols, availableBooleanCols],
  );
  const [filterRows, setFilterRows] = useState<EditableFilter[]>([{
    id: "filter-1",
    column: "",
    operator: "=",
    value: "",
  }]);
  // The user picks the column; defaulting to the first one (usually an ID) invites a wrong filter.
  // Errors for an untouched, empty builder are not shown: only after the user starts editing.
  const [filterTouched, setFilterTouched] = useState(false);
  const filterRowsInitialRef = React.useRef(true);
  React.useEffect(() => {
    if (filterRowsInitialRef.current) { filterRowsInitialRef.current = false; return; }
    setFilterTouched(true);
  }, [filterRows, customFilter]);
  const filterBuilderResult = useMemo(
    () => compileEditableFilters(filterRows, availableFilterColumns, availableNumericCols, availableBooleanCols),
    [filterRows, availableFilterColumns, availableNumericCols, availableBooleanCols],
  );
  const activeCustomFilter = filterMode === "builder" ? filterBuilderResult.sql : customFilter.trim();
  const customFilterError = filterMode === "builder"
    ? (filterBuilderResult.error || (!filterBuilderResult.sql ? "Add at least one complete filter condition, or return to the full question scope." : null))
    : (!customFilter.trim() ? "Enter a SQL filter expression before running this subset." : null);
  // Once a decision has been applied to a refreshed preview, keep the decision itself.
  // Otherwise the panel remount clears the radio choice and every subsequent click re-previews
  // an uncorrected plan instead of ever reaching execution.
  const [selectedClarificationOption, setSelectedClarificationOption] = useState<string>(
    retainedGuidance.clarification_response || ""
  );
  const [selectedClarificationPatch, setSelectedClarificationPatch] = useState<Record<string, unknown> | undefined>(
    retainedGuidance.intent_patch
  );
  const [customInstruction, setCustomInstruction] = useState<string>(retainedGuidance.custom_instruction || "");
  const questionFilters = useMemo(() => contractFilters(plan), [plan]);
  const [showSql, setShowSql] = useState<boolean>(true);
  const [showAssumptions, setShowAssumptions] = useState<boolean>(true);

  const isAmbiguous = plan.ambiguity_status !== "CLEAR";
  const hasFreeformResolution = Boolean(customInstruction.trim());
  const hasStructuredResolution =
    (metricTouched && Boolean(selectedMetric.trim())) ||
    (groupTouched && Boolean(selectedGroup.trim())) ||
    Boolean(selectedDataset);
  const hasScopeResolution =
    scopeTouched ||
    (selectedScope === "CUSTOM" && Boolean(customFilter.trim()));
  const isDecisionRequiredAndMissing =
    isAmbiguous &&
    plan.clarification_options.length > 0 &&
    !selectedClarificationOption &&
    !hasFreeformResolution &&
    !hasStructuredResolution &&
    !hasScopeResolution;

  // Detect contradictory / invalid selections: SUM/AVG on categorical field
  const isMetricCategorical = useMemo(() => {
    const m = selectedMetric.trim().toLowerCase();
    if (!m) return false;
    return availableCategoricalCols.map((c) => c.toLowerCase()).includes(m);
  }, [selectedMetric, availableCategoricalCols]);

  const isInvalidCombination = useMemo(() => {
    return isMetricCategorical && ["SUM", "AVG", "MEDIAN", "MIN", "MAX"].includes(selectedAggregation);
  }, [isMetricCategorical, selectedAggregation]);
  const duplicateScopeInstruction =
    selectedScope === "CUSTOM" &&
    activeCustomFilter != null &&
    activeCustomFilter.trim().replace(/\s+/g, " ").toLowerCase() ===
      customInstruction.trim().replace(/\s+/g, " ").toLowerCase();

  // Show the backend's canonical plan verbatim until the analyst changes a parameter.
  // If the analyst edits the plan, keep every question-derived predicate in the displayed
  // calculation and add custom scope as an additional AND condition (the resolver does the same).
  const derivedCalculation = useMemo(() => {
    const hasParameterEdits = metricTouched || aggregationTouched || groupTouched;
    if (!hasParameterEdits && selectedScope !== "CUSTOM" && plan.planned_calculation) {
      return plan.planned_calculation;
    }
    const metric = selectedMetric.trim();
    const group = selectedGroup.trim();
    const agg = selectedAggregation.toUpperCase();
    const filters = questionFilters.map(sqlFilter);
    if (selectedScope === "CUSTOM" && activeCustomFilter) {
      const rawFilter = activeCustomFilter.trim().replace(/^where\s+/i, "");
      filters.push(`(${rawFilter})`);
    }
    const filterClause = filters.length ? `WHERE ${filters.join(" AND ")}` : "";

    let selectExpr = "";
    if (agg === "COUNT") {
      selectExpr = "COUNT(*) AS observation_count";
    } else if (agg === "COUNT_DISTINCT") {
      selectExpr = `COUNT(DISTINCT ${metric ? sqlIdentifier(metric) : "*"}) AS distinct_count`;
    } else {
      selectExpr = `${agg}(${metric ? sqlIdentifier(metric) : "*"}) AS metric_val`;
    }

    if (group) {
      const groupSql = sqlIdentifier(group);
      const orderCol = agg === "COUNT" ? "observation_count" : agg === "COUNT_DISTINCT" ? "distinct_count" : "metric_val";
      return `SELECT ${groupSql}, ${selectExpr}\nFROM data\n${filterClause ? filterClause + "\n" : ""}GROUP BY ${groupSql}\nORDER BY ${orderCol} DESC;`;
    }
    return `SELECT ${selectExpr}\nFROM data${filterClause ? "\n" + filterClause : ""};`;
  }, [selectedMetric, selectedAggregation, selectedGroup, selectedScope, activeCustomFilter, metricTouched, aggregationTouched, groupTouched, questionFilters, plan.planned_calculation]);

  // Dynamically derive the active assumptions
  const derivedAssumptions = useMemo(() => {
    const list: string[] = [];
    const metric = selectedMetric.trim();
    const group = selectedGroup.trim();
    const agg = selectedAggregation.toUpperCase();

    if (agg === "COUNT") {
      list.push("Metric: Total observation row count (COUNT(*))");
    } else if (agg === "COUNT_DISTINCT") {
      list.push(`Metric: Count of unique entities (COUNT(DISTINCT '${metric || "target"}'))`);
    } else {
      list.push(`Metric: Quantitative ${agg} of '${metric || "target"}'`);
    }

    if (group) {
      list.push(`Grouping dimension: '${group}'`);
    } else {
      list.push(`Grouping: overall (no group dimension)`);
    }

    for (const filter of questionFilters) {
      list.push(`Question filter (preserved): ${displayFilter(filter)}`);
    }
    if (selectedScope === "CUSTOM" && activeCustomFilter) {
      list.push(`Additional scope filter: ${activeCustomFilter}`);
    } else if (questionFilters.length) {
      list.push("Scope: Keep the population filters stated in your question; no additional UI filter.");
    } else {
      list.push("Scope: All available records; no additional filter selected.");
    }

    // Preserve any existing boundary assumptions from the backend
    for (const asm of plan.assumptions) {
      if (!asm.toLowerCase().startsWith("metric:") && !asm.toLowerCase().startsWith("grouping") && !asm.toLowerCase().startsWith("scope:")) {
        list.push(asm);
      }
    }
    return list;
  }, [selectedMetric, selectedAggregation, selectedGroup, selectedScope, activeCustomFilter, plan.assumptions, questionFilters]);

  const retainedConfirmedAssumptions = Array.isArray(retainedGuidance.confirmed_assumptions)
    ? retainedGuidance.confirmed_assumptions
    : [];
  // Do not carry an acknowledgement forward if a corrected interpretation changed its
  // assumptions. That requires the analyst to review the new parameter list again.
  const [assumptionsAcknowledged, setAssumptionsAcknowledged] = useState<boolean>(
    retainedConfirmedAssumptions.length > 0 &&
    JSON.stringify(retainedConfirmedAssumptions) === JSON.stringify(derivedAssumptions)
  );

  const getStatusBadge = () => {
    switch (plan.ambiguity_status) {
      case "CLEAR":
        return {
          label: "Ready to Execute",
          tone: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
          icon: <CheckCircle2 className="w-3.5 h-3.5" />,
        };
      case "AMBIGUOUS_METRIC":
        return {
          label: "Awaiting Metric Selection",
          tone: "border-amber-500/30 bg-amber-500/10 text-amber-200",
          icon: <HelpCircle className="w-3.5 h-3.5" />,
        };
      case "MISSING_DATA":
        return {
          label: "Missing Measure Clarification",
          tone: "border-rose-500/30 bg-rose-500/10 text-rose-300",
          icon: <AlertTriangle className="w-3.5 h-3.5" />,
        };
      case "CAUSAL_ON_OBSERVATIONAL":
        return {
          label: "Observational Boundary",
          tone: "border-amber-500/30 bg-amber-500/10 text-amber-200",
          icon: <ShieldAlert className="w-3.5 h-3.5" />,
        };
      case "JOIN_FANOUT_RISK":
        return {
          label: "Join Fanout Risk",
          tone: "border-rose-500/30 bg-rose-500/10 text-rose-300",
          icon: <AlertTriangle className="w-3.5 h-3.5" />,
        };
      default:
        return {
          label: "Awaiting Review",
          tone: "border-cyan-500/30 bg-cyan-500/10 text-cyan-200",
          icon: <Cpu className="w-3.5 h-3.5" />,
        };
    }
  };

  const badge = getStatusBadge();

  const buildGuidancePayload = () => {
    const freeformSubmitted = Boolean(customInstruction.trim());
    const selectedOptionIndex = plan.clarification_options.indexOf(selectedClarificationOption);
    const selectedOptionPatch = freeformSubmitted
      ? undefined
      : selectedOptionIndex >= 0
        ? plan.clarification_option_patches?.[selectedOptionIndex]?.patch
        : selectedClarificationPatch;
    const ledger = (plan.canonical_contract as { intent_ledger?: Array<{ slot?: string; status?: string }> } | null | undefined)?.intent_ledger || [];
    const pendingIntentSlot = freeformSubmitted
      ? retainedGuidance.pending_intent_slot || plan.pending_intent_slot || ledger.find((item) => item.status === "UNRESOLVED" && item.slot)?.slot
      : undefined;
    return {
      // Defaults shown in controls are not user intent. Only send a field after the analyst
      // actually edits it; unresolved defaults must not become accidental semantic patches.
      selected_metric: metricTouched ? selectedMetric.trim() || undefined : undefined,
      selected_aggregation: aggregationTouched ? selectedAggregation || undefined : undefined,
      selected_group_dimension: groupTouched ? selectedGroup.trim() || undefined : undefined,
      selected_dataset: selectedDataset || undefined,
      confirmed_assumptions: assumptionsAcknowledged ? derivedAssumptions : [],
      scope_filter: selectedScope === "CUSTOM" ? activeCustomFilter || undefined : undefined,
      clarification_response: selectedClarificationOption || undefined,
      custom_instruction: customInstruction.trim() || undefined,
      intent_patch: selectedOptionPatch,
      pending_intent_slot: pendingIntentSlot,
    };
  };

  const fingerprintGuidance = (fields: Record<string, unknown>) => {
    const entries = Object.entries(fields).filter(([, value]) =>
      value !== undefined && value !== null && value !== "" && !(Array.isArray(value) && value.length === 0)
    ).sort(([left], [right]) => left.localeCompare(right));
    return entries.length ? JSON.stringify(Object.fromEntries(entries)) : "";
  };
  const guidanceNeedsPreview = fingerprintGuidance(buildGuidancePayload()) !== previewedGuidanceKey;
  const canRunCurrentPlan = plan.can_auto_proceed && plan.ambiguity_status === "CLEAR";

  const handleRun = () => {
    if (isDecisionRequiredAndMissing || isInvalidCombination || duplicateScopeInstruction) return;
    if (!guidanceNeedsPreview && !canRunCurrentPlan) return;
    if (selectedScope === "CUSTOM" && (!activeCustomFilter || customFilterError)) return;
    onConfirmAndRun({
      ...buildGuidancePayload(),
      plan_hash: plan.plan_hash || undefined,
    });
  };

  return (
    <div className="rounded-2xl border border-cyan-500/30 bg-slate-900/95 p-6 shadow-2xl shadow-cyan-950/40 space-y-5 animate-in fade-in-50 duration-200">
      {/* Header Banner */}
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center space-x-2 text-[11px] font-bold uppercase tracking-wider text-cyan-400 mb-1">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Proposed Analysis Plan</span>
          </div>
          <h2 className="text-lg font-bold text-white tracking-tight">
            Review and Guide Before Execution
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            MindEd structured this interpretation from your question. Confirm or correct parameters before DuckDB execution.
          </p>
        </div>

        <div className={`inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${badge.tone}`}>
          {badge.icon}
          <span>{badge.label}</span>
        </div>
      </div>

      {/* Question & Interpretation Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-1.5">
          <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500">Your Question</div>
          <div className="text-sm font-medium text-white">{plan.question}</div>
        </div>

        <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-1.5">
          <div className="text-[10px] uppercase font-bold tracking-wider text-cyan-400">Proposed Interpretation</div>
          <div className="text-xs text-slate-200 leading-relaxed">{plan.proposed_interpretation}</div>
        </div>
      </div>

      {availableDatasets.length > 1 && (
        <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4 space-y-2 text-xs">
          <label htmlFor="guidance-dataset" className="text-[10px] font-bold uppercase text-slate-400">
            Dataset for this analysis
          </label>
          <select
            id="guidance-dataset"
            value={selectedDataset}
            onChange={(event) => setSelectedDataset(event.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-cyan-500"
          >
            <option value="">Let the contract resolver decide</option>
            {availableDatasets.map((dataset) => (
              <option key={dataset.id} value={dataset.name}>{dataset.name}</option>
            ))}
          </select>
        </div>
      )}

      {/* Consequential Ambiguity or Clarification Banner */}
      {isAmbiguous && plan.clarification_prompt && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-950/30 p-4 space-y-3 text-xs">
          <div className="flex items-center space-x-2 text-amber-300 font-bold">
            <HelpCircle className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <span>Human Decision Needed</span>
          </div>
          <p className="text-amber-100/90 leading-relaxed">
            {plan.clarification_prompt}
          </p>
          {plan.conflict && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-950/30 p-3 text-rose-100">
              <div className="font-semibold">
                Structured and freeform guidance conflict on {String(plan.conflict.field || "an analytical field")}.
              </div>
              <div className="mt-1 text-rose-200/80">
                Structured: {String(plan.conflict.structured_value ?? "unspecified")} · Freeform: {String(plan.conflict.freeform_value ?? "unspecified")}
              </div>
            </div>
          )}

          {plan.clarification_options.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <div className="text-[10px] uppercase font-bold text-amber-400">Choose the intended direction:</div>
              <div className="flex flex-col gap-2">
                {plan.clarification_options.map((opt, idx) => (
                  <label
                    key={idx}
                    className={`flex items-center space-x-2 p-2.5 rounded-lg border cursor-pointer transition ${
                      selectedClarificationOption === opt
                        ? "border-amber-400 bg-amber-500/20 text-white font-medium"
                        : "border-slate-800 bg-slate-950 text-slate-300 hover:border-slate-700"
                    }`}
                  >
                    <input
                      type="radio"
                      name="clarification_choice"
                      checked={selectedClarificationOption === opt}
                      onChange={() => {
                        setSelectedClarificationOption(opt);
                        setSelectedClarificationPatch(plan.clarification_option_patches?.[idx]?.patch);
                      }}
                      className="text-amber-500 focus:ring-amber-400"
                    />
                    <span>{opt}</span>
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Planned Calculation SQL Preview */}
      <div className="rounded-xl border border-slate-800 bg-slate-950/80 overflow-hidden">
        <button
          type="button"
          onClick={() => setShowSql(!showSql)}
          className="w-full px-4 py-2.5 flex items-center justify-between text-left hover:bg-slate-900/50 text-xs font-semibold text-slate-300"
        >
          <div className="flex items-center space-x-2">
            <TableProperties className="w-3.5 h-3.5 text-cyan-400" />
            <span>Planned Calculation Logic</span>
          </div>
          {showSql ? <ChevronUp className="w-3.5 h-3.5 text-slate-500" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-500" />}
        </button>
        {showSql && (
          <div className="p-3 border-t border-slate-800/80 bg-slate-950 space-y-2">
            <pre className="text-[11px] font-mono text-emerald-300 whitespace-pre-wrap break-words">
              {derivedCalculation}
            </pre>
            <p className="text-[10px] leading-relaxed text-slate-500">
              This is the proposed calculation preview, not a claim that execution has succeeded. The saved execution record and verification result are authoritative after you run it.
            </p>
          </div>
        )}
      </div>

      {/* Interactive Parameter Controls */}
      <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4 space-y-4">
        <div className="text-xs font-bold text-white flex items-center space-x-2">
          <Layers className="w-3.5 h-3.5 text-cyan-400" />
          <span>Confirm or Adjust Operational Parameters</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          {/* Metric Selector */}
          <div className="space-y-1">
            <label className="text-[10px] font-bold uppercase text-slate-400">Target Metric</label>
            <input
              type="text"
              value={selectedMetric}
              onChange={(e) => {
                setSelectedMetric(e.target.value);
                setMetricTouched(true);
              }}
              placeholder="e.g. net_revenue"
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-cyan-500"
            />
          </div>

          {/* Aggregation Selector */}
          <div className="space-y-1">
            <label className="text-[10px] font-bold uppercase text-slate-400">Aggregation</label>
            <select
              value={selectedAggregation}
              onChange={(e) => {
                setSelectedAggregation(e.target.value);
                setAggregationTouched(true);
              }}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-cyan-500"
            >
              <option value="COUNT">COUNT (Rows / Observations)</option>
              <option value="COUNT_DISTINCT">COUNT_DISTINCT (Unique Entities)</option>
              <option value="SUM">SUM (Total)</option>
              <option value="AVG">AVG (Mean Average)</option>
              <option value="MEDIAN">MEDIAN (50th Percentile)</option>
              <option value="MIN">MIN (Minimum Value)</option>
              <option value="MAX">MAX (Maximum Value)</option>
            </select>
          </div>

          {/* Grouping Dimension */}
          <div className="space-y-1">
            <label className="text-[10px] font-bold uppercase text-slate-400">Grouping Dimension</label>
            <input
              type="text"
              value={selectedGroup}
              onChange={(e) => {
                setSelectedGroup(e.target.value);
                setGroupTouched(true);
              }}
              placeholder="e.g. region, plan_tier"
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-cyan-500"
            />
          </div>
        </div>

        {/* Population & Scope Selection */}
        <div className="pt-2 border-t border-slate-800/80 space-y-2">
          <label className="text-[10px] font-bold uppercase text-slate-400 flex items-center space-x-1.5">
            <Filter className="w-3 h-3 text-cyan-400" />
            <span>Analysis Scope / Population</span>
          </label>
          <div className="flex flex-wrap gap-2 text-xs">
            <button
              type="button"
              onClick={() => {
                setSelectedScope("ALL");
                setScopeTouched(true);
              }}
              className={`px-3 py-1.5 rounded-lg border text-xs transition ${
                selectedScope === "ALL"
                  ? "border-cyan-500/50 bg-cyan-500/10 text-cyan-200 font-medium"
                  : "border-slate-800 bg-slate-900 text-slate-400 hover:text-white"
              }`}
            >
              {questionFilters.length ? "Keep question-defined scope" : "All available records"}
            </button>
            <button
              type="button"
              onClick={() => {
                setSelectedScope("CUSTOM");
                setScopeTouched(true);
              }}
              className={`px-3 py-1.5 rounded-lg border text-xs transition ${
                selectedScope === "CUSTOM"
                  ? "border-cyan-500/50 bg-cyan-500/10 text-cyan-200 font-medium"
                  : "border-slate-800 bg-slate-900 text-slate-400 hover:text-white"
              }`}
            >
              {questionFilters.length ? "Add a filter" : "Specific filter subset"}
            </button>
          </div>
          {selectedScope === "CUSTOM" && (
            <div className="mt-3 rounded-xl border border-slate-800 bg-slate-950/70 p-3.5 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="text-xs font-semibold text-white">Additional filter conditions</div>
                  <p className="mt-1 text-[11px] leading-relaxed text-slate-400">
                    Each condition is checked against the loaded dataset columns. Conditions are combined with AND; your question's filters remain in place.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setFilterMode(filterMode === "builder" ? "sql" : "builder")}
                  className="rounded-lg border border-slate-700 px-2.5 py-1.5 text-[11px] font-semibold text-cyan-200 hover:border-cyan-500/50"
                >
                  {filterMode === "builder" ? "Advanced SQL expression" : "Use filter builder"}
                </button>
              </div>

              {filterMode === "builder" ? (
                <div className="space-y-2.5">
                  {availableFilterColumns.length === 0 ? (
                    <div role="alert" className="rounded-lg border border-amber-500/30 bg-amber-950/20 p-3 text-xs text-amber-200">
                      Dataset columns are not available for a safe filter builder. Reload the dataset preview or use the advanced expression after confirming column names.
                    </div>
                  ) : filterRows.map((row, index) => (
                    <div key={row.id} className="grid grid-cols-1 sm:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,1.2fr)_auto] gap-2 items-center">
                      <label className="sr-only" htmlFor={`filter-column-${row.id}`}>Filter column {index + 1}</label>
                      <select
                        id={`filter-column-${row.id}`}
                        value={row.column}
                        onChange={(event) => {
                          const column = event.target.value;
                          setFilterRows((rows) => rows.map((item) => item.id === row.id ? { ...item, column } : item));
                          setScopeTouched(true);
                        }}
                        className="min-w-0 w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs text-white focus:outline-none focus:border-cyan-500"
                      >
                        <option value="">Choose column…</option>
                        {availableFilterColumns.map((column) => <option key={column} value={column}>{column}</option>)}
                      </select>
                      <label className="sr-only" htmlFor={`filter-operator-${row.id}`}>Filter operator {index + 1}</label>
                      <select
                        id={`filter-operator-${row.id}`}
                        value={row.operator}
                        onChange={(event) => {
                          const operator = event.target.value as FilterOperator;
                          // Keep the entered value when switching to a null operator so changing
                          // back does not silently erase the analyst's previous filter value.
                          setFilterRows((rows) => rows.map((item) => item.id === row.id ? { ...item, operator } : item));
                          setScopeTouched(true);
                        }}
                        className="min-w-0 w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs text-white focus:outline-none focus:border-cyan-500"
                      >
                        {FILTER_OPERATORS.map((operator) => <option key={operator.value} value={operator.value}>{operator.label}</option>)}
                      </select>
                      {row.operator === "IS NULL" || row.operator === "IS NOT NULL" ? (
                        <div className="rounded-lg border border-dashed border-slate-800 px-3 py-2 text-xs text-slate-500">No value needed</div>
                      ) : (
                        <>
                          <label className="sr-only" htmlFor={`filter-value-${row.id}`}>Filter value {index + 1}</label>
                          <input
                            id={`filter-value-${row.id}`}
                            value={row.value}
                            onChange={(event) => {
                              const value = event.target.value;
                              setFilterRows((rows) => rows.map((item) => item.id === row.id ? { ...item, value } : item));
                              setScopeTouched(true);
                            }}
                            placeholder={row.operator === "IN" || row.operator === "NOT IN" ? "Values separated by commas" : availableNumericCols.includes(row.column) ? "Enter a number" : "Enter a value"}
                            inputMode={availableNumericCols.includes(row.column) && !["IN", "NOT IN"].includes(row.operator) ? "decimal" : "text"}
                            className="min-w-0 w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                          />
                        </>
                      )}
                      <button
                        type="button"
                        disabled={filterRows.length === 1}
                        onClick={() => setFilterRows((rows) => rows.filter((item) => item.id !== row.id))}
                        title="Remove filter condition"
                        aria-label={`Remove filter condition ${index + 1}`}
                        className="inline-flex items-center justify-center rounded-lg border border-slate-800 p-2 text-slate-400 hover:border-rose-500/50 hover:text-rose-300 disabled:opacity-30"
                      ><Trash2 className="h-3.5 w-3.5" /></button>
                    </div>
                  ))}
                  {availableFilterColumns.length > 0 && (
                    <button
                      type="button"
                      onClick={() => setFilterRows((rows) => [...rows, { id: `filter-${Date.now()}-${rows.length}`, column: availableFilterColumns[0], operator: "=", value: "" }])}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-2 text-xs font-semibold text-cyan-200 hover:border-cyan-500/50"
                    ><Plus className="h-3.5 w-3.5" /> Add condition</button>
                  )}
                  {customFilterError && filterTouched && (
                    <p role="alert" className="text-xs text-amber-300">{customFilterError}</p>
                  )}
                  {filterBuilderResult.sql && !customFilterError && (
                    <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3 space-y-1">
                      <div className="text-[10px] font-semibold uppercase tracking-wider text-emerald-300">Filter preview · AND logic</div>
                      <code className="block whitespace-pre-wrap break-words text-[11px] text-emerald-100">{filterBuilderResult.sql}</code>
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-2">
                  <label htmlFor="advanced-scope-filter" className="block text-[11px] font-semibold text-slate-300">SQL WHERE expression (not a complete SELECT statement)</label>
                  <textarea
                    id="advanced-scope-filter"
                    value={customFilter}
                    onChange={(event) => {
                      setCustomFilter(event.target.value);
                      setScopeTouched(true);
                    }}
                    rows={3}
                    placeholder={questionFilters.length ? "channel = 'Web' AND revenue > 100" : "region = 'West' AND status IN ('Delivered', 'Shipped')"}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono"
                  />
                  <div className="flex items-start gap-2 rounded-lg border border-sky-500/20 bg-sky-500/5 p-3 text-[11px] leading-relaxed text-sky-100/90">
                    <Info className="h-4 w-4 shrink-0 text-sky-300 mt-0.5" />
                    <p>Use column names from the selected dataset. Supported forms include <code>status = 'Delivered'</code>, <code>revenue &gt;= 100</code>, <code>region IN ('East', 'West')</code>, <code>revenue BETWEEN 100 AND 500</code>, and <code>status IS NOT NULL</code>. Combine conditions with AND. OR, functions, comments, statement separators, qualified columns and unknown columns are rejected rather than guessed.</p>
                  </div>
                  {customFilterError && filterTouched && <p role="alert" className="text-xs text-amber-300">{customFilterError}</p>}
                </div>
              )}
              <p className="text-[10px] leading-relaxed text-slate-500">
                SQL behavior: conditions become a WHERE predicate. AND narrows the population. The builder escapes text literals and quotes column identifiers; the backend validates the expression against the selected dataset before execution. If no rows match, analysis stops rather than dropping the filter.
              </p>
            </div>
          )}
        </div>

        {/* Free-text instruction / guidance */}
        <div className="pt-2 border-t border-slate-800/80 space-y-1">
          <label className="text-[10px] font-bold uppercase text-slate-400">
            Optional specific instruction / question adjustment
          </label>
          <input
            type="text"
            value={customInstruction}
            onChange={(e) => setCustomInstruction(e.target.value)}
            placeholder="e.g. 'Exclude returns and zero-dollar orders' or 'Also report row counts per segment'"
            className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>
      </div>

      {/* Assumptions to Review */}
      {derivedAssumptions.length > 0 && (
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 space-y-2 text-xs">
          <button
            type="button"
            onClick={() => setShowAssumptions(!showAssumptions)}
            className="w-full flex items-center justify-between text-left text-[11px] font-bold uppercase tracking-wider text-slate-400"
          >
            <span>Assumptions to Review ({derivedAssumptions.length})</span>
            {showAssumptions ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
          {showAssumptions && (
            <div className="space-y-3 pt-1">
              <ul className="list-disc list-inside space-y-1 text-slate-300">
                {derivedAssumptions.map((asm, idx) => (
                  <li key={idx} className="leading-relaxed">{asm}</li>
                ))}
              </ul>
              <label className="flex items-center space-x-2 pt-2 border-t border-slate-800/80 cursor-pointer text-slate-300 hover:text-white">
                <input
                  type="checkbox"
                  checked={assumptionsAcknowledged}
                  onChange={(e) => setAssumptionsAcknowledged(e.target.checked)}
                  className="rounded text-cyan-500 focus:ring-cyan-400 bg-slate-900 border-slate-700"
                />
                <span className="text-[11px] font-medium">I have reviewed and acknowledge these analytical assumptions</span>
              </label>
            </div>
          )}
        </div>
      )}

      {duplicateScopeInstruction && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-950/30 p-3.5 text-xs text-amber-200">
          This expression is already in the scope-filter field. Clear the optional instruction field; it is for additional analytical guidance, not a duplicate filter.
        </div>
      )}

      {/* Invalid Combination Conflict Warning */}
      {isInvalidCombination && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/30 p-3.5 flex items-start space-x-3 text-xs text-rose-300">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-semibold text-rose-200">
              Contradictory plan: '{selectedMetric}' is a categorical field and cannot be combined with {selectedAggregation}.
            </p>
            <p className="text-[11px] text-rose-300/90 leading-relaxed">
              To analyze categorical values, select <strong>COUNT (Rows / Observations)</strong> to count records, or choose a numeric metric.
            </p>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800">
        <button
          type="button"
          onClick={onCancel}
          disabled={isRunning}
          className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-white border border-slate-800 hover:border-slate-700 bg-slate-950 transition disabled:opacity-50"
        >
          Cancel / Edit Question
        </button>

        <div className="flex items-center gap-3">
          {isDecisionRequiredAndMissing && (
            <span className="text-[11px] text-amber-300 font-semibold flex items-center gap-1.5 animate-pulse">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Select an option above to resolve ambiguity</span>
            </span>
          )}
          {!guidanceNeedsPreview && !canRunCurrentPlan && !isDecisionRequiredAndMissing && (
            <span role="status" className="max-w-md text-[11px] text-amber-300 font-medium">
              This preview is still unresolved. Change a parameter or choose a clarification before running.
            </span>
          )}
          <button
            type="button"
            onClick={handleRun}
            disabled={isRunning || isDecisionRequiredAndMissing || isInvalidCombination || duplicateScopeInstruction || (!guidanceNeedsPreview && !canRunCurrentPlan) || (selectedScope === "CUSTOM" && (!activeCustomFilter || Boolean(customFilterError)))}
            className="px-6 py-2.5 rounded-xl text-xs font-bold bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-600/30 flex items-center space-x-2 transition disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isRunning ? (
              <>
                <Cpu className="w-4 h-4 animate-spin" />
                <span>Running Investigation…</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>{guidanceNeedsPreview ? "Apply Decisions & Refresh Preview" : "Confirm & Run Analysis"}</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
