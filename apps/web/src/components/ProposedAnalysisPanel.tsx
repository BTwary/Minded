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
  RotateCcw,
  ShieldAlert,
  Sparkles,
  TableProperties,
} from "lucide-react";
import { PreAnalysisPlan } from "../lib/api";

interface ProposedAnalysisPanelProps {
  plan: PreAnalysisPlan;
  availableDatasets?: Array<{ id: string; name: string }>;
  availableNumericCols?: string[];
  availableCategoricalCols?: string[];
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

export default function ProposedAnalysisPanel({
  plan,
  availableDatasets = [],
  availableNumericCols = [],
  availableCategoricalCols = [],
  onConfirmAndRun,
  onCancel,
  isRunning = false,
}: ProposedAnalysisPanelProps) {
  const saved = useMemo(() => (typeof window === "undefined" ? {} : loadSaved(plan.question || "")), [plan.question]);
  const [selectedMetric, setSelectedMetric] = useState<string>(
    saved.metric ?? (plan.target_metric || (availableNumericCols[0] ?? ""))
  );
  const initialAgg = useMemo(() => inferAggregation(plan.question || ""), [plan.question]);

  const [selectedAggregation, setSelectedAggregation] = useState<string>(saved.aggregation ?? initialAgg);
  const [selectedGroup, setSelectedGroup] = useState<string>(
    saved.group ?? (plan.group_dimension || (availableCategoricalCols[0] ?? ""))
  );
  const [metricTouched, setMetricTouched] = useState(saved.metric !== undefined);
  const [aggregationTouched, setAggregationTouched] = useState(saved.aggregation !== undefined);
  const [groupTouched, setGroupTouched] = useState(saved.group !== undefined);
  React.useEffect(() => {
    const v: { metric?: string; aggregation?: string; group?: string } = {};
    if (metricTouched) v.metric = selectedMetric;
    if (aggregationTouched) v.aggregation = selectedAggregation;
    if (groupTouched) v.group = selectedGroup;
    if (Object.keys(v).length) saveSaved(plan.question || "", v);
  }, [metricTouched, aggregationTouched, groupTouched, selectedMetric, selectedAggregation, selectedGroup, plan.question]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedScope, setSelectedScope] = useState<string>("ALL");
  const [customFilter, setCustomFilter] = useState<string>("");
  // Invariant: For ambiguous plans, do not silently pre-select an option; require explicit human decision.
  const [selectedClarificationOption, setSelectedClarificationOption] = useState<string>("");
  const [selectedClarificationPatch, setSelectedClarificationPatch] = useState<Record<string, unknown> | undefined>(undefined);
  const [customInstruction, setCustomInstruction] = useState<string>("");
  const [showSql, setShowSql] = useState<boolean>(true);
  const [showAssumptions, setShowAssumptions] = useState<boolean>(true);
  const [assumptionsAcknowledged, setAssumptionsAcknowledged] = useState<boolean>(false);

  const isAmbiguous = plan.ambiguity_status !== "CLEAR";
  const hasFreeformResolution = Boolean(customInstruction.trim());
  const hasStructuredResolution =
    (metricTouched && Boolean(selectedMetric.trim())) ||
    (groupTouched && Boolean(selectedGroup.trim())) ||
    Boolean(selectedDataset);
  const isDecisionRequiredAndMissing =
    isAmbiguous &&
    plan.clarification_options.length > 0 &&
    !selectedClarificationOption &&
    !hasFreeformResolution &&
    !hasStructuredResolution;

  // Detect contradictory / invalid selections: SUM/AVG on categorical field
  const isMetricCategorical = useMemo(() => {
    const m = selectedMetric.trim().toLowerCase();
    if (!m) return false;
    return availableCategoricalCols.map((c) => c.toLowerCase()).includes(m);
  }, [selectedMetric, availableCategoricalCols]);

  const isInvalidCombination = useMemo(() => {
    return isMetricCategorical && ["SUM", "AVG", "MEDIAN", "MIN", "MAX"].includes(selectedAggregation);
  }, [isMetricCategorical, selectedAggregation]);

  // Dynamically derive the planned calculation SQL preview
  const derivedCalculation = useMemo(() => {
    const metric = selectedMetric.trim();
    const group = selectedGroup.trim();
    const agg = selectedAggregation.toUpperCase();
    const filterClause = selectedScope === "CUSTOM" && customFilter.trim() ? `WHERE ${customFilter.trim()}` : "";

    let selectExpr = "";
    if (agg === "COUNT") {
      // Row count: COUNT(*) preserves all rows including records where a dimension or metric is null
      selectExpr = "COUNT(*) AS observation_count";
    } else if (agg === "COUNT_DISTINCT") {
      selectExpr = `COUNT(DISTINCT ${metric || "*"}) AS distinct_count`;
    } else {
      selectExpr = `${agg}(${metric || "*"}) AS metric_val`;
    }

    if (group) {
      const orderCol = agg === "COUNT" ? "observation_count" : agg === "COUNT_DISTINCT" ? "distinct_count" : "metric_val";
      return `SELECT ${group}, ${selectExpr}\nFROM data\n${filterClause ? filterClause + "\n" : ""}GROUP BY ${group}\nORDER BY ${orderCol} DESC;`;
    } else {
      return `SELECT ${selectExpr}\nFROM data${filterClause ? "\n" + filterClause : ""};`;
    }
  }, [selectedMetric, selectedAggregation, selectedGroup, selectedScope, customFilter]);

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

    if (selectedScope === "CUSTOM" && customFilter.trim()) {
      list.push(`Scope filter: ${customFilter.trim()}`);
    } else {
      list.push(`Scope: All available records without filtering`);
    }

    // Preserve any existing boundary assumptions from the backend
    for (const asm of plan.assumptions) {
      if (!asm.toLowerCase().startsWith("metric:") && !asm.toLowerCase().startsWith("grouping") && !asm.toLowerCase().startsWith("scope:")) {
        list.push(asm);
      }
    }
    return list;
  }, [selectedMetric, selectedAggregation, selectedGroup, selectedScope, customFilter, plan.assumptions]);

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

  const handleRun = () => {
    if (isDecisionRequiredAndMissing) return;
    const freeformSubmitted = Boolean(customInstruction.trim());
    const selectedOptionIndex = plan.clarification_options.indexOf(selectedClarificationOption);
    const selectedOptionPatch = freeformSubmitted
      ? undefined
      : selectedOptionIndex >= 0
        ? plan.clarification_option_patches?.[selectedOptionIndex]?.patch
        : selectedClarificationPatch;
    const ledger = (plan.canonical_contract as { intent_ledger?: Array<{ slot?: string; status?: string }> } | null | undefined)?.intent_ledger || [];
    const pendingIntentSlot = freeformSubmitted
      ? plan.pending_intent_slot || ledger.find((item) => item.status === "UNRESOLVED" && item.slot)?.slot
      : undefined;
    onConfirmAndRun({
      // Defaults shown in controls are not user intent. Only send a field after the analyst
      // actually edits it; unresolved defaults must not become accidental semantic patches.
      selected_metric: metricTouched ? selectedMetric.trim() || undefined : undefined,
      selected_aggregation: aggregationTouched ? selectedAggregation || undefined : undefined,
      selected_group_dimension: groupTouched ? selectedGroup.trim() || undefined : undefined,
      selected_dataset: selectedDataset || undefined,
      confirmed_assumptions: assumptionsAcknowledged ? derivedAssumptions : [],
      scope_filter: selectedScope === "CUSTOM" && customFilter.trim() ? customFilter.trim() : undefined,
      clarification_response: selectedClarificationOption || undefined,
      custom_instruction: customInstruction.trim() || undefined,
      intent_patch: selectedOptionPatch,
      pending_intent_slot: pendingIntentSlot,
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
          <div className="p-3 border-t border-slate-800/80 bg-slate-950">
            <pre className="text-[11px] font-mono text-emerald-300 whitespace-pre-wrap break-all">
              {derivedCalculation}
            </pre>
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
              onClick={() => setSelectedScope("ALL")}
              className={`px-3 py-1.5 rounded-lg border text-xs transition ${
                selectedScope === "ALL"
                  ? "border-cyan-500/50 bg-cyan-500/10 text-cyan-200 font-medium"
                  : "border-slate-800 bg-slate-900 text-slate-400 hover:text-white"
              }`}
            >
              All available records
            </button>
            <button
              type="button"
              onClick={() => setSelectedScope("CUSTOM")}
              className={`px-3 py-1.5 rounded-lg border text-xs transition ${
                selectedScope === "CUSTOM"
                  ? "border-cyan-500/50 bg-cyan-500/10 text-cyan-200 font-medium"
                  : "border-slate-800 bg-slate-900 text-slate-400 hover:text-white"
              }`}
            >
              Specific filter subset
            </button>
          </div>
          {selectedScope === "CUSTOM" && (
            <input
              type="text"
              value={customFilter}
              onChange={(e) => setCustomFilter(e.target.value)}
              placeholder="e.g. region = 'West'"
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 mt-2 font-mono"
            />
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
          <button
            type="button"
            onClick={handleRun}
            disabled={isRunning || isDecisionRequiredAndMissing || isInvalidCombination}
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
                <span>Run Analysis</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
