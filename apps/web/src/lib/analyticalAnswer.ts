import type { AnalysisResponse } from "../types";

type RecordValue = Record<string, unknown>;
export type AnalyticalChartSpec = {
  chart_id: string;
  chart_type: string;
  title: string;
  explanation?: string;
  x_axis: { field: string; label?: string; type?: string; format?: string };
  y_axis: { field: string; label?: string; type?: string; format?: string };
  series_field?: string;
  data: Array<Record<string, unknown>>;
  metadata?: Record<string, unknown>;
};

export type AnswerNarrativeClaim = {
  text: string;
  evidenceRefs: string[];
  factIds: string[];
};

export type AnswerPresentation = {
  answer: string;
  findings: string[];
  interpreted: string[];
  calculation: string[];
  limitations: string[];
  charts: AnalyticalChartSpec[];
  supported: boolean;
  narrative?: {
    family: string;
    title: string;
    summary: string;
    paragraphs: string[];
    claims: AnswerNarrativeClaim[];
    supportedClaimCount: number;
  };
  table?: { headers: string[]; rows: string[][] };
};

const record = (v: unknown): RecordValue => v !== null && typeof v === "object" && !Array.isArray(v) ? v as RecordValue : {};
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const text = (v: unknown): string => typeof v === "string" ? v : "";
const strings = (v: unknown): string[] => Array.isArray(v) ? v.filter((s): s is string => typeof s === "string" && !!s.trim()) : [];
const number = (v: number): string => {
  // Significant-digit limits silently change large additive totals (e.g. 1,375,613.66
  // became 1,375,610) and ranking gaps. Use fractional precision for ordinary values;
  // retain significant precision only for very small scientific coefficients.
  const magnitude = Math.abs(v);
  const options: Intl.NumberFormatOptions = magnitude > 0 && magnitude < 0.0001
    ? { maximumSignificantDigits: 6 }
    : { maximumFractionDigits: 6 };
  return v.toLocaleString(undefined, options);
};
const percent = (v: number): string => `${(v * 100).toFixed(1)}%`;
const pValue = (v: unknown): string | null => finite(v) && v >= 0 && v <= 1 ? (v < 0.001 ? "p < 0.001" : `p = ${number(v)}`) : null;

const isRecord = (v: unknown): v is RecordValue => v !== null && typeof v === "object" && !Array.isArray(v);

function chartSpec(
  id: string,
  type: string,
  title: string,
  xField: string,
  yField: string,
  data: Array<Record<string, unknown>>,
  options: Partial<AnalyticalChartSpec> = {},
): AnalyticalChartSpec {
  return {
    chart_id: id,
    chart_type: type,
    title,
    explanation: options.explanation,
    x_axis: options.x_axis || { field: xField, label: xField },
    y_axis: options.y_axis || { field: yField, label: yField },
    series_field: options.series_field,
    data,
    metadata: options.metadata || { source_method: "presentation_from_persisted_result" },
  };
}

function usableExplicitChart(value: unknown): AnalyticalChartSpec | null {
  if (!isRecord(value)) return null;
  const type = text(value.chart_type || value.type).toLowerCase();
  const supportedTypes = new Set(["line", "multi_line", "bar", "grouped_bar", "stacked_bar", "scatter", "histogram", "donut", "boxplot", "waterfall"]);
  const data = Array.isArray(value.data) ? value.data.filter(isRecord) : [];
  if (!supportedTypes.has(type) || data.length === 0) return null;
  const x = record(value.x_axis);
  const y = record(value.y_axis);
  const xField = text(x.field || value.x_field || value.xField) || Object.keys(data[0])[0] || "category";
  const yField = text(y.field || value.y_field || value.yField) || Object.keys(data[0]).find((k) => data.some((row) => finite(row[k]))) || "value";
  return {
    chart_id: text(value.chart_id) || `chart_${type}_${xField}_${yField}`.replace(/[^a-z0-9_]+/gi, "_").toLowerCase(),
    chart_type: type,
    title: text(value.title || value.name) || `${yField} by ${xField}`,
    explanation: text(value.explanation) || undefined,
    x_axis: { field: xField, label: text(x.label || x.title) || xField, type: text(x.type) || undefined, format: text(x.format) || undefined },
    y_axis: { field: yField, label: text(y.label || y.title) || yField, type: text(y.type) || undefined, format: text(y.format) || undefined },
    series_field: text(value.series_field || value.seriesField) || undefined,
    data,
    metadata: isRecord(value.metadata) ? value.metadata : { source_method: "persisted_chart_payload" },
  };
}

/**
 * Select visualizations from verified, persisted result values only. This is a renderer-side
 * projection of the result contract: it does not reopen source rows, refit models, interpolate
 * missing values, or create synthetic observations.
 */
export function selectAnalyticalCharts(analysis: AnalysisResponse): AnalyticalChartSpec[] {
  const result = record(analysis.analyst_result);
  const n = record(result.numbers);
  const kind = text(result.kind).toUpperCase();
  const operation = text(n.operation || n.aggregation || record(n.canonical_contract).requested_aggregation).toUpperCase();
  const metric = text(n.target_column || record(n.canonical_contract).target_column) || "metric";
  const groups = strings(n.grouping_columns || record(n.canonical_contract).grouping_columns);
  const explicitCandidates: unknown[] = [
    ...(Array.isArray(n.trend_charts) ? n.trend_charts : []),
    n.trend_chart,
    n.trend_slope_chart,
    n.forecast_chart,
    n.comparison_chart,
    n.distribution_chart,
    n.relationship_chart,
    n.scatter_chart,
    ...(Array.isArray(n.chart_specs) ? n.chart_specs : []),
    ...(Array.isArray(n.visualizations) ? n.visualizations : []),
    ...(Array.isArray(analysis.findings) ? analysis.findings.map((f) => f.chart_data) : []),
  ];
  const explicit: AnalyticalChartSpec[] = [];
  const explicitKeys = new Set<string>();
  for (const candidate of explicitCandidates) {
    const spec = usableExplicitChart(candidate);
    if (!spec) continue;
    const identity = `${spec.chart_id}|${spec.chart_type}|${spec.title}`;
    if (!explicitKeys.has(identity)) {
      explicitKeys.add(identity);
      explicit.push(spec);
    }
  }
  if (explicit.length) return explicit;

  const charts: AnalyticalChartSpec[] = [];
  const question = analysis.question || "";
  const compositionSpec = record(n.composition_over_time);
  const compositionRows = Array.isArray(compositionSpec.data) ? compositionSpec.data.filter(isRecord) : [];
  const compositionTime = text(compositionSpec.time_field);
  const compositionSeries = text(compositionSpec.series_field);
  const compositionValue = text(compositionSpec.value_field);
  if (
    compositionRows.length >= 2 && compositionTime && compositionSeries && compositionValue &&
    compositionRows.every((row) => row[compositionTime] !== null && row[compositionTime] !== undefined && typeof row[compositionSeries] === "string" && finite(row[compositionValue]))
  ) {
    return [chartSpec(
      `composition_over_time_${compositionValue}`,
      "stacked_bar",
      `Composition of ${compositionValue} over time by ${compositionSeries}`,
      compositionTime,
      compositionValue,
      compositionRows,
      {
        explanation: "Stacked segments use saved period-by-group aggregates. The visualization does not invent missing periods or interpolate values.",
        x_axis: { field: compositionTime, label: compositionTime, type: "nominal" },
        y_axis: { field: compositionValue, label: compositionValue, type: "quantitative" },
        series_field: compositionSeries,
        metadata: { source_method: "persisted_composition_over_time", all_saved_points_included: true },
      },
    )];
  }
  const asksDistribution = /distribution|variability|spread|outlier|dispersion|volatility|more consistent|less consistent|range of values/i.test(question);
  // Use saved paired observations for a true relationship scatter plot. Never reconstruct
  // raw observations from a correlation coefficient or from marginal summaries.
  const scatterPayload = record(n.scatter_data);
  const scatterRowsRaw = Array.isArray(n.scatter_data) ? n.scatter_data : Array.isArray(scatterPayload.data) ? scatterPayload.data : Array.isArray(scatterPayload.points) ? scatterPayload.points : [];
  const scatterRows = scatterRowsRaw.filter(isRecord);
  const scatterX = text(scatterPayload.x_field || scatterPayload.x_column || n.scatter_x_field || n.explanatory_column || n.secondary_target_column);
  const scatterY = text(scatterPayload.y_field || scatterPayload.y_column || n.scatter_y_field || n.target_column);
  if (scatterRows.length >= 2 && scatterX && scatterY && scatterRows.every((row) => finite(row[scatterX]) && finite(row[scatterY]))) {
    charts.push(chartSpec(`scatter_${scatterX}_${scatterY}`, "scatter", `Relationship between ${scatterX} and ${scatterY}`, scatterX, scatterY,
      scatterRows.map((row) => ({ [scatterX]: row[scatterX], [scatterY]: row[scatterY] })), {
        explanation: "Each point is a saved paired observation. The scatter plot shows association and does not establish causation.",
        x_axis: { field: scatterX, label: scatterX, type: "quantitative" },
        y_axis: { field: scatterY, label: scatterY, type: "quantitative" },
        metadata: { source_method: "persisted_paired_observations", all_saved_points_included: true },
      }));
  }

  const histogramPayload = record(n.histogram);
  const histogramRaw = Array.isArray(n.histogram_bins) ? n.histogram_bins : Array.isArray(histogramPayload.bins) ? histogramPayload.bins : [];
  const histogramRows = histogramRaw.filter(isRecord).flatMap((row) => {
    const start = row.bin_start ?? row.start ?? row.left;
    const end = row.bin_end ?? row.end ?? row.right;
    const count = row.count ?? row.frequency ?? row.n;
    if (!finite(start) || !finite(end) || !finite(count) || count < 0 || end < start) return [];
    return [{ bin_label: text(row.bin_label || row.label) || `${number(start)}–${number(end)}`, count }];
  });
  if (asksDistribution && histogramRows.length >= 1) {
    charts.push(chartSpec(`histogram_${metric}`, "histogram", `Distribution of ${metric}`, "bin_label", "count", histogramRows, {
      explanation: "Histogram bars use the saved bin boundaries and frequencies; raw observations have not been reconstructed.",
      x_axis: { field: "bin_label", label: `Bins of ${metric}`, type: "nominal" },
      y_axis: { field: "count", label: "Frequency", type: "quantitative" },
      metadata: { source_method: "persisted_histogram_bins", bin_count: histogramRows.length },
    }));
  }

  const table = record(n.table);
  const tableEntries = Object.entries(table)
    .map(([category, raw]) => ({ category, value: record(raw).value, n: record(raw).n, raw: record(raw) }))
    .filter((item): item is { category: string; value: number; n: unknown; raw: RecordValue } => finite(item.value));

  if (tableEntries.length >= 2) {
    const ascending = /\b(lowest|least|smallest|ascending)\b/i.test(question) && !/\b(highest|largest|top|descending)\b/i.test(question);
    const ordered = [...tableEntries].sort((a, b) => ascending ? a.value - b.value : b.value - a.value);
    const xField = groups[0] || "category";
    const yField = metric || "value";
    const chartEntries = ordered.slice(0, 50);
    const rows = chartEntries.map((item) => ({ [xField]: item.category, [yField]: item.value, ...(finite(item.n) ? { n: item.n } : {}) }));
    const chartExplanation = ordered.length > rows.length
      ? `Bars use the recorded group-level results. Showing ${rows.length} of ${ordered.length} groups in the chart; the result table remains the saved analytical output. No additional model was fitted.`
      : "Bars use the recorded group-level results; no additional statistical model was fitted for this visualization.";
    charts.push(chartSpec(`comparison_${xField}_${yField}`, "bar", `${operation || "Observed value"} of ${yField} by ${xField}`, xField, yField, rows, {
      explanation: chartExplanation,
      x_axis: { field: xField, label: xField, type: "nominal" },
      y_axis: { field: yField, label: yField, type: "quantitative", format: operation === "RATE" || kind === "RATE_COMPARISON" ? "percent" : undefined },
      metadata: { source_method: "persisted_aggregate_table", observation_count: rows.length, available_group_count: ordered.length, ranking_order: ascending ? "ascending" : "descending" },
    }));

    const additive = ["SUM", "COUNT", "ROW_COUNT"].includes(operation);
    const total = ordered.reduce((sum, item) => sum + item.value, 0);
    if (additive && groups.length === 1 && ordered.length <= 8 && ordered.every((item) => item.value >= 0) && total > 0) {
      charts.push(chartSpec(`composition_${xField}_${yField}`, "donut", `Share of returned total ${yField} by ${xField}`, xField, yField,
        ordered.map((item) => ({ [xField]: item.category, [yField]: item.value })), {
          explanation: "Composition is calculated from the returned non-negative additive group totals; categories are displayed as parts of this returned total.",
          x_axis: { field: xField, label: xField, type: "nominal" },
          y_axis: { field: yField, label: yField, type: "quantitative" },
          metadata: { source_method: "persisted_aggregate_table", composition_total: total },
        }));
    }
  }

  const means = record(n.means);
  const rates = record(n.rates);
  const groupValues = Object.keys(means).length ? means : rates;
  const groupEntries = Object.entries(groupValues).filter((entry): entry is [string, number] => finite(entry[1]));
  if (!charts.length && groupEntries.length >= 2) {
    const isRate = Object.keys(rates).length > 0;
    const sortedGroups = [...groupEntries].sort((a, b) => b[1] - a[1]);
    const rows = sortedGroups.slice(0, 50).map(([category, value]) => ({ category, value }));
    charts.push(chartSpec(`groups_${metric}`, "bar", `${isRate ? "Rate" : "Recorded value"} of ${metric} by group`, "category", "value", rows, {
      explanation: sortedGroups.length > rows.length ? `Showing 50 of ${sortedGroups.length} saved group estimates.` : undefined,
      x_axis: { field: "category", label: groups[0] || "Group", type: "nominal" },
      y_axis: { field: "value", label: isRate ? `Rate of ${metric}` : metric, type: "quantitative", format: isRate ? "percent" : undefined },
      metadata: { source_method: "persisted_group_estimates", available_group_count: sortedGroups.length, groups_shown: rows.length },
    }));
  }

  const correlations = record(n.correlations);
  const correlationEntries = Object.entries(correlations).filter((entry): entry is [string, number] => finite(entry[1]) && Math.abs(entry[1]) <= 1);
  if (!charts.length && correlationEntries.length >= 2) {
    const sortedCorrelations = correlationEntries.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const shownCorrelations = sortedCorrelations.slice(0, 50);
    charts.push(chartSpec(`correlations_${metric}`, "bar", `Recorded correlations with ${metric}`, "variable", "correlation",
      shownCorrelations.map(([variable, value]) => ({ variable, correlation: value })), {
        x_axis: { field: "variable", label: "Variable", type: "nominal" },
        y_axis: { field: "correlation", label: "Pearson r", type: "quantitative" },
        explanation: `Correlation coefficients describe linear association; they do not establish causation or significance for each predictor.${sortedCorrelations.length > shownCorrelations.length ? ` Showing 50 of ${sortedCorrelations.length} recorded coefficients.` : ""}`,
        metadata: { source_method: "persisted_correlation_coefficients", reference_value: 0, coefficients_available: sortedCorrelations.length, coefficients_shown: shownCorrelations.length },
      }));
  }

  const distribution = record(n.distribution_profile);
  if (asksDistribution && Object.keys(distribution).length) {
    const summaries: Array<Record<string, unknown>> = [];
    const whiskerDefinitions = new Set<string>();
    const toBox = (label: string, profile: RecordValue) => {
      const hasPercentiles = finite(profile.p10) && finite(profile.p90);
      const low = profile.p10 ?? profile.min;
      const q1 = profile.p25 ?? profile.q1;
      const median = profile.p50 ?? profile.median;
      const q3 = profile.p75 ?? profile.q3;
      const high = profile.p90 ?? profile.max;
      if ([low, q1, median, q3, high].every(finite) && (low as number) <= (q1 as number) && (q1 as number) <= (median as number) && (median as number) <= (q3 as number) && (q3 as number) <= (high as number)) {
        summaries.push({ label, p10: low, p25: q1, p50: median, p75: q3, p90: high });
        whiskerDefinitions.add(hasPercentiles ? "p10-p90" : "min-max");
      }
    };
    if (isRecord(distribution.current_period) || isRecord(distribution.prior_period)) {
      if (isRecord(distribution.prior_period)) toBox("Prior period", record(distribution.prior_period));
      if (isRecord(distribution.current_period)) toBox("Current period", record(distribution.current_period));
    } else toBox("Observed distribution", distribution);
    if (summaries.length) charts.push(chartSpec(`distribution_${metric}`, "boxplot", `Distribution of ${metric}`, "label", metric, summaries, {
      explanation: `Box-and-whisker summary uses the recorded ${Array.from(whiskerDefinitions).join(" and ")} whiskers and quartiles; it does not reconstruct unavailable raw observations.`,
      x_axis: { field: "label", label: "Period / distribution", type: "nominal" },
      y_axis: { field: "p50", label: metric, type: "quantitative" },
      metadata: { source_method: "persisted_distribution_profile", whisker_definition: Array.from(whiskerDefinitions) },
    }));
  }

  const bridge = record(n.waterfall_bridge);
  const steps = Array.isArray(bridge.steps) ? bridge.steps.filter(isRecord) : [];
  if (/change|drop|declin|increase|decreas|shift|impact|period/i.test(kind + " " + question) && steps.length >= 2 && steps.every((step) => finite(step.delta) && finite(step.subtotal))) {
    let previous = 0;
    const rows = steps.map((step, index) => {
      const end = step.subtotal as number;
      const start = index === 0 ? 0 : previous;
      previous = end;
      return { step: text(step.step) || `Step ${index + 1}`, delta: step.delta as number, subtotal: end, start, end };
    });
    charts.push(chartSpec(`waterfall_${metric}`, "waterfall", `What contributed to the change in ${metric}`, "step", "delta", rows, {
      explanation: "Each contribution uses the saved reconciliation bridge. Intermediate bars span from the prior subtotal to the next subtotal.",
      x_axis: { field: "step", label: "Component", type: "nominal" },
      y_axis: { field: "delta", label: `Change in ${metric}`, type: "quantitative" },
      metadata: { source_method: "persisted_waterfall_bridge" },
    }));
  }

  return charts;
}

/** Presentation only: consume persisted aggregates/statistics, never rows or a new analysis.
 * Differences and shares below are arithmetic over returned values, not new inferential tests.
 * Verification/consistency remain owned by StructuredAnalysisAnswer.
 */
export function presentAnalyticalAnswer(
  analysis: AnalysisResponse,
  verification: string,
  consistencyFailure: string | null,
): AnswerPresentation {
  const result = record(analysis.analyst_result);
  const n = record(result.numbers);
  const contract = record(n.canonical_contract);
  const plan = record(analysis.analysis_plan);
  const estimand = record(plan.estimand);
  const semantics = record(plan.semantics);
  const kind = text(result.kind).toUpperCase();
  const operation = text(n.operation || n.aggregation || contract.requested_aggregation || estimand.requested_aggregation).toUpperCase();
  const metric = text(n.target_column || contract.target_column || estimand.target) || "the metric";
  const groups = strings(n.grouping_columns || contract.grouping_columns || semantics.grouping_columns);
  const diagnostic = /\b(why|caus\w*|explain\w*|driv(?:es?|ing|ers?))\b/i.test(analysis.question) || ["ROOT_CAUSE", "PERIOD_CHANGE"].includes(kind);
  const forecast = /\b(forecast\w*|predict|prediction|projected|projection)\b|\bnext\s+(period|month|week|quarter|year|day)\b/i.test(analysis.question) || plan.task === "FORECAST";
  const out: AnswerPresentation = { answer: "", findings: [], interpreted: [], calculation: [], limitations: strings(result.caveats), charts: [], supported: false };
  const execution = record(analysis.execution_record) || record(n.execution_record);
  const comparisonValues = strings(n.comparison_values || contract.comparison_values || semantics.comparison_values);
  const interpretedWindow = n.time_window ?? n.period ?? contract.time_range ?? semantics.time_range ?? plan.time_window;
  const reportContract = record(analysis.report_contract);
  if (Array.isArray(reportContract.interpreted)) {
    out.interpreted.push(...strings(reportContract.interpreted));
  } else {
    if (kind) out.interpreted.push(`Analysis type: ${kind.replaceAll("_", " ").toLowerCase()}.`);
    if (metric !== "the metric") out.interpreted.push(`Target metric / outcome: ${metric}.`);
    if (operation) out.interpreted.push(`Requested aggregation or method: ${operation.replaceAll("_", " ")}.`);
    if (groups.length) out.interpreted.push(`Grouping dimension(s): ${groups.join(", ")}.`);
    if (comparisonValues.length) out.interpreted.push(`Requested comparison values: ${comparisonValues.join(", ")}.`);
    if (interpretedWindow !== undefined && interpretedWindow !== null && interpretedWindow !== "") {
      out.interpreted.push(`Requested time scope: ${typeof interpretedWindow === "string" ? interpretedWindow : JSON.stringify(interpretedWindow)}.`);
    }
    const interpretedFilters = n.filters ?? contract.filter_conditions ?? semantics.filters;
    if (Array.isArray(interpretedFilters)) {
      out.interpreted.push(`Filters in the recorded interpretation: ${interpretedFilters.length ? JSON.stringify(interpretedFilters) : "none recorded"}.`);
    }
  }

  if (Array.isArray(reportContract.calculation)) {
    out.calculation.push(...strings(reportContract.calculation));
  } else if (Object.keys(n).length) {
    // Compatibility path for saved analyses made before the shared report contract.
    // A proposed plan is never described as proof of execution.
    const executedMetric = text(execution.target_column) || metric;
    const executedOperation = text(execution.aggregation) || operation;
    const executedGroups = strings(execution.grouping_columns).length ? strings(execution.grouping_columns) : groups;
    if (executedMetric !== "the metric") out.calculation.push(`Metric: ${executedMetric}.`);
    if (executedOperation) out.calculation.push(`Calculation: ${executedOperation.replaceAll("_", " ")}${executedGroups.length ? `, grouped by ${executedGroups.join(", ")}` : ""}.`);
    const filters = execution.filters ?? n.filters ?? contract.filter_conditions;
    if (Array.isArray(filters)) {
      const sourceLabel = execution.filters !== undefined ? "Executed filters" : "Filters in the recorded result contract";
      out.calculation.push(filters.length ? `${sourceLabel}: ${JSON.stringify(filters)}.` : `${sourceLabel}: none recorded.`);
    }
    const populationCount = execution.population_row_count ?? n.population_row_count;
    if (finite(populationCount)) out.calculation.push(`Records in scope after filtering: ${number(populationCount)}.`);
    const resultCount = execution.result_row_count ?? n.result_row_count;
    if (finite(resultCount)) out.calculation.push(`Rows / groups returned: ${number(resultCount)}.`);
    if (["SUM", "TOTAL"].includes(executedOperation)) out.calculation.push(`Aggregation logic: sum the valid ${executedMetric} values for each requested group after applying the executed filters.`);
    else if (["MEAN", "AVG", "AVERAGE"].includes(executedOperation)) out.calculation.push(`Aggregation logic: average the valid ${executedMetric} values within each requested group after applying the executed filters.`);
    else if (["COUNT", "ROW_COUNT"].includes(executedOperation)) out.calculation.push(`Aggregation logic: count eligible records within each requested group; null-handling follows the recorded execution contract.`);
    else if (["RATE", "PROPORTION"].includes(executedOperation)) out.calculation.push("Aggregation logic: use the recorded numerator and denominator definition; see denominator details in the execution record when supplied.");
    const table = record(n.table);
    const groupCounts = Object.entries(table).filter(([, row]) => finite(record(row).n)).map(([group, row]) => `${group}: n=${number(record(row).n as number)}`);
    if (groupCounts.length) out.calculation.push(`Recorded group sample sizes: ${groupCounts.join("; ")}.`);
    const exclusions = n.excluded_row_count ?? n.dropped_row_count ?? n.rows_excluded;
    if (finite(exclusions) && exclusions > 0) out.calculation.push(`Rows excluded during analysis: ${number(exclusions)}.`);
    const distribution = record(n.distribution_profile);
    if (finite(distribution.n)) out.calculation.push(`Valid observations in the distribution summary: ${number(distribution.n)}.`);
    const inference = record(n.trend_inference);
    if (text(inference.method)) out.calculation.push(`Trend method recorded: ${text(inference.method).replaceAll("_", " ")}.`);
    if (text(n.statistical_test || n.method)) out.calculation.push(`Statistical method recorded: ${text(n.statistical_test || n.method)}.`);
  }
  const finish = (): AnswerPresentation => {
    if (out.supported && analysis.status === "COMPLETED" && verification === "VERIFIED" && !consistencyFailure) {
      // Prefer the API's shared report contract so browser and PDF/DOCX use the same
      // interpretation and chart choices. Fall back only for older saved investigations.
      if (Array.isArray(reportContract.visualizations)) {
        out.charts = reportContract.visualizations
          .map((chart) => usableExplicitChart(chart))
          .filter((chart): chart is AnalyticalChartSpec => chart !== null);
      } else {
        out.charts = selectAnalyticalCharts(analysis);
      }
      const savedNarrative = record(reportContract.narrative);
      out.limitations.push(...strings(savedNarrative.limitations));
      const narrativeStatus = text(savedNarrative.status).toUpperCase();
      const savedSummary = text(savedNarrative.summary);
      if (["READY", "PARTIAL", "LIMITED"].includes(narrativeStatus) && savedSummary) {
        const supportedClaims = (Array.isArray(savedNarrative.claims) ? savedNarrative.claims : [])
          .map(record)
          .filter((claim) => text(claim.validation_status).toUpperCase() === "SUPPORTED" && !!text(claim.text))
          .map((claim) => ({
            text: text(claim.text),
            evidenceRefs: strings(claim.evidence_refs),
            factIds: strings(claim.fact_ids),
          }));
        const safeSummary = supportedClaims.length > 0
          ? supportedClaims[0].text
          : narrativeStatus === "LIMITED" ? savedSummary : "A narrative summary was withheld because no fully supported claim was available.";
        out.narrative = {
          family: text(savedNarrative.family) || "descriptive",
          title: text(savedNarrative.title) || "Analysis explanation",
          summary: safeSummary,
          paragraphs: supportedClaims.slice(1, 8).map((claim) => claim.text),
          claims: supportedClaims,
          supportedClaimCount: supportedClaims.length,
        };
      }
    }
    out.findings = out.findings.slice(0, 5);
    out.interpreted = Array.from(new Set(out.interpreted));
    out.calculation = Array.from(new Set(out.calculation));
    out.limitations = Array.from(new Set(out.limitations));
    return out;
  };

  if (consistencyFailure || analysis.status !== "COMPLETED" || verification !== "VERIFIED") {
    out.answer = consistencyFailure
      ? "This result cannot be treated as an answer because its saved evidence is inconsistent."
      : analysis.status === "FAILED"
        ? "The analysis failed, so there is no verified answer to this question."
        : `A verified answer is not available (${verification.toLowerCase().replaceAll("_", " ")}). Review the recorded evidence and limitations before drawing a conclusion.`;
    if (diagnostic && !consistencyFailure && analysis.status === "COMPLETED") {
      out.answer = `The available evidence does not establish a cause for the outcome in this question. Verification is ${verification.toLowerCase()}; no causal explanation can be concluded.`;
    }
    if (analysis.direct_answer && (analysis.status === "FAILED" || ["INCONCLUSIVE", "INSUFFICIENT_DATA", "CONFLICTING_EVIDENCE"].includes(String(analysis.verdict)))) {
      out.limitations.unshift(`Recorded outcome: ${analysis.direct_answer}`);
    }
    return out;
  }
  if (!analysis.analyst_result || ["INCONCLUSIVE", "INSUFFICIENT_DATA", "CONFLICTING_EVIDENCE"].includes(String(analysis.verdict))) {
    out.answer = "The available evidence does not support a conclusive answer to this question.";
    return out;
  }
  const unavailable = (reason: string) => {
    out.answer = diagnostic ? `The available analysis cannot establish why this happened. ${reason}` : reason;
    return out;
  };
  const interval = (scale = 1, unit = "") => finite(n.ci_low) && finite(n.ci_high) && n.ci_low <= n.ci_high
    ? `95% confidence interval: ${number(n.ci_low * scale)} to ${number(n.ci_high * scale)}${unit}.` : null;
  const addInference = (p: unknown, ci: string | null) => {
    if (ci) out.findings.push(ci);
    const pText = pValue(p);
    if (pText) out.findings.push(`Reported test: ${pText}.`);
    else out.limitations.push("A valid p-value was not recorded; statistical significance cannot be concluded.");
  };

  // A forecast is only reported when an actual projection exists. A historical total
  // or fitted trend must never stand in for a requested future quantity.
  if (forecast) {
    const projection = record(n.run_rate_projection);
    if (!finite(projection.projected_value)) return unavailable("No forecast was recorded. Historical results do not answer the requested future outcome.");
    out.answer = `The recorded next-${text(n.grain) || "period"} run-rate projection for ${metric} is ${number(projection.projected_value)}.`;
    if (finite(projection.pi_low) && finite(projection.pi_high) && projection.pi_low <= projection.projected_value && projection.projected_value <= projection.pi_high) {
      out.findings.push(`95% prediction interval: ${number(projection.pi_low)} to ${number(projection.pi_high)}.`);
    } else out.limitations.push("A valid prediction interval was not supplied; forecast uncertainty is unavailable.");
    out.limitations.push("This is a trend extrapolation for one period, not a guaranteed outcome or a forecast for a different horizon.");
  } else if (kind === "ROOT_CAUSE") {
    const candidate = text(n.top_candidate);
    const classification = text(n.confounding_classification).toUpperCase();
    const strength = finite(n.top_candidate_r) ? n.top_candidate_r : null;
    const attenuation = finite(n.attenuation_pct) ? n.attenuation_pct : null;
    const unadjusted = finite(n.unadjusted_diff) ? n.unadjusted_diff : null;
    const adjusted = finite(n.adjusted_diff) ? n.adjusted_diff : null;
    const screeningDetails = strings(result.details);
    if (candidate) {
      const strengthText = strength !== null ? ` (recorded association strength ${number(strength)})` : "";
      const adjustedText = unadjusted !== null && adjusted !== null
        ? ` Adjusting for this factor changed the group difference from ${number(unadjusted)} to ${number(adjusted)}${attenuation !== null ? ` (${number(attenuation)}% attenuation)` : ""}.`
        : "";
      if (["CONFOUNDER_EFFECT_REVERSAL", "STRATUM_EFFECT_REVERSAL"].includes(classification)) {
        out.answer = `${candidate}${strengthText} is the leading measured factor associated with the observed group difference. The saved adjustment/stratum analysis reports an effect-direction reversal${adjustedText} This indicates that the pooled comparison depends on group composition or strata; it does not prove a causal mechanism.`;
      } else if (classification === "CANDIDATE_CONFOUNDER_SUBSTANTIAL_ATTENUATION" || classification === "CANDIDATE_CONFOUNDER_PARTIAL_ATTENUATION") {
        out.answer = `${candidate}${strengthText} is the leading measured candidate confounder identified by screening.${adjustedText} The observed pattern is consistent with confounding, but observational adjustment alone cannot establish that ${candidate} caused the outcome.`;
      } else if (classification === "PERSISTENT_GROUP_DIFFERENCE") {
        out.answer = `${candidate}${strengthText} was identified as a candidate factor, but adjustment did not materially explain the observed group difference.${adjustedText} The difference persists in this model; that does not prove a direct causal effect.`;
      } else {
        out.answer = `The recorded analysis identifies ${candidate}${strengthText} as the strongest qualifying measured candidate factor.${adjustedText} Treat this as a diagnostic lead, not a confirmed cause.`;
      }
      if (attenuation !== null) out.findings.push(`Recorded adjustment attenuation: ${number(attenuation)}%.`);
      if (unadjusted !== null && adjusted !== null) out.findings.push(`Unadjusted difference: ${number(unadjusted)}; adjusted difference: ${number(adjusted)}.`);
      for (const detail of screeningDetails) {
        if (out.findings.length >= 5) break;
        if (!out.findings.includes(detail)) out.findings.push(detail);
      }
    } else {
      const noCandidate = screeningDetails.find((detail) => /none of the scanned factors met|no candidate factor|no qualifying candidate/i.test(detail));
      out.answer = noCandidate
        ? `${noCandidate} The available data did not identify a measured candidate factor meeting the screening criteria, so a specific explanation remains unsupported.`
        : "The recorded diagnostic did not identify a qualifying candidate factor. The available evidence is insufficient to name a specific explanation.";
      out.findings.push(...screeningDetails.slice(0, 4));
    }
    out.limitations.push("This is an observational diagnostic using measured candidate factors. Unmeasured confounding, model assumptions, data quality, and alternative explanations may remain; statistical adjustment does not establish causality.");
    out.supported = true;
  } else if (kind === "PERIOD_CHANGE") {
    // A period-change question is a diagnostic decomposition request, not a generic
    // causal question. Explain the observed movement from saved values, then surface
    // the saved bridge components as arithmetic contributions without calling them causes.
    if (!finite(n.previous) || !finite(n.current) || !finite(n.delta) ||
        Math.abs((n.current - n.previous) - n.delta) > 1e-8 * Math.max(1, Math.abs(n.previous), Math.abs(n.current))) {
      return unavailable("The recorded period comparison does not contain a consistent prior value, current value, and net change. No explanation can be safely inferred.");
    }
    const changeWord = n.delta > 0 ? "increased" : n.delta < 0 ? "decreased" : "was unchanged";
    const periodLabel = text(n.period) || "the selected period";
    const previousLabel = text(n.previous_start) || "the preceding period";
    const pct = finite(n.pct_change) ? n.pct_change : (n.previous !== 0 ? n.delta / n.previous : null);
    out.answer = `Recorded ${operation.toLowerCase() || "value"} of ${metric} ${changeWord} from ${number(n.previous)} in ${previousLabel} to ${number(n.current)} in ${periodLabel} (net change ${n.delta > 0 ? "+" : ""}${number(n.delta)}${pct !== null && Number.isFinite(pct) ? `; ${pct > 0 ? "+" : ""}${number(pct * 100)}%` : ""}).`;

    const bridge = record(n.waterfall_bridge);
    const steps = Array.isArray(bridge.steps) ? bridge.steps.filter(isRecord) : [];
    const contributions = steps.filter((step) => {
      const label = text(step.step).toLowerCase();
      return label && label !== "prior base" && label !== "net change" && finite(step.delta) && step.delta !== 0;
    });
    if (contributions.length) {
      const largest = [...contributions].sort((a, b) => Math.abs(b.delta as number) - Math.abs(a.delta as number))[0];
      out.findings.push(`Largest recorded bridge component: ${text(largest.step)} (${(largest.delta as number) > 0 ? "+" : ""}${number(largest.delta as number)}). This is an arithmetic decomposition of the observed change, not proof of a causal mechanism.`);
      out.findings.push(...contributions.slice(0, 4).map((step) => `${text(step.step)}: ${(step.delta as number) > 0 ? "+" : ""}${number(step.delta as number)}.`));
    }
    const pvm = record(n.pvm_decomposition);
    // Values read from a generic record are `unknown`; narrow each tuple before
    // assigning it to the typed effects collection so the production TS build
    // does not infer an `unknown` label from array destructuring.
    const rawPvmEffects: Array<[string, unknown]> = [
      ["Volume effect", pvm.volume_effect ?? n.volume_effect],
      ["Rate effect", pvm.rate_effect ?? n.rate_effect],
      ["Cross interaction", pvm.cross_effect ?? n.cross_effect],
    ];
    const pvmEffects: Array<[string, number]> = rawPvmEffects.filter(
      (entry): entry is [string, number] => finite(entry[1]),
    );
    if (pvmEffects.length && !contributions.length) {
      out.findings.push(...pvmEffects.map(([label, amount]) => `${label}: ${amount > 0 ? "+" : ""}${number(amount)} (recorded decomposition, not an independently established cause).`));
    }
    if (text(n.top_contributor) && finite(n.top_contribution_share)) {
      out.findings.unshift(`${text(n.top_contributor)} in ${text(n.top_dimension) || "the recorded breakdown"} accounted for ${percent(n.top_contribution_share)} of the recorded shift, using the report's saved contribution calculation.`);
    }
    const details = strings(result.details);
    for (const detail of details) {
      if (out.findings.length >= 5) break;
      if (!out.findings.includes(detail)) out.findings.push(detail);
    }
    out.limitations.push("The period comparison and bridge describe observed arithmetic contributions. They do not, by themselves, establish that a component caused the change or rule out confounding factors.");
    out.supported = true;
  } else if (["GROUP_COMPARISON", "RATE_COMPARISON", "ROOT_CAUSE"].includes(kind)) {
    const rates = kind === "RATE_COMPARISON";
    const values = record(rates ? n.rates : n.means);
    const subject = text(n.subject), reference = text(n.reference);
    const a = values[subject], b = values[reference];
    const diff = rates ? n.rate_diff : n.diff;
    const scale = rates ? 100 : 1;
    const unit = rates ? " percentage points" : "";
    const value = (v: number) => rates ? percent(v) : number(v);
    if (!subject && !reference && Object.keys(values).length >= 2 && Object.values(values).every(finite)) {
      const ordered = Object.entries(values).map(([name, v]) => ({name, value: v as number})).sort((a, b) => b.value - a.value);
      out.answer = `${rates ? "Observed rates" : "Observed averages"} of ${metric} range from ${value(ordered.at(-1)!.value)} (${ordered.at(-1)!.name}) to ${value(ordered[0].value)} (${ordered[0].name}).`;
      out.table = {headers: [groups.join(", ") || "Group", `${rates ? "Rate" : "Mean"} (${metric})`], rows: ordered.map(v => [v.name, value(v.value)])};
      addInference(n.anova_p ?? n.p_value, null);
      out.limitations.push("An overall group test does not identify which individual pairs differ; that requires pairwise comparisons.");
      if (diagnostic) {
        out.answer = `The available evidence does not establish a cause. ${out.answer}`;
        out.limitations.push("Observed group differences do not establish a causal effect.");
      }
      out.supported = true;
      return finish();
    }
    if (!subject || !reference || !finite(a) || !finite(b) || (rates && (a < 0 || a > 1 || b < 0 || b > 1)) || !finite(diff) || Math.abs((a - b) - diff) > 1e-8 * Math.max(1, Math.abs(a), Math.abs(b))) {
      return unavailable("The recorded analysis does not supply a consistent named pair and difference. It cannot answer a specific group comparison.");
    }
    out.answer = `Observed ${rates ? "rate of" : "average"} ${metric} is ${value(a)} for ${subject} versus ${value(b)} for ${reference}: ${number(Math.abs(diff) * scale)}${unit} ${diff > 0 ? "higher" : diff < 0 ? "lower" : "different"} (${subject} minus ${reference}).`;
    if (diff === 0) out.answer = `The observed ${rates ? "rates" : "averages"} for ${subject} and ${reference} are equal at ${value(a)}. This does not establish population equivalence.`;
    out.table = { headers: [groups.join(", ") || "Group", `${rates ? "Rate" : "Mean"} (${metric})`], rows: [[subject, value(a)], [reference, value(b)]] };
    addInference(n.p_value, interval(scale, unit));
    out.limitations.push("The observed group difference does not establish a causal effect.");
    if (diagnostic) {
      out.answer = `The available evidence does not establish a cause. ${out.answer}`;
      if (text(n.top_candidate) && finite(n.adjusted_diff)) {
        out.findings.unshift(`After adjustment for candidate factor ${text(n.top_candidate)}, the recorded difference is ${number(n.adjusted_diff)}. Adjustment alone does not establish causation.`);
      }
    }
  } else if (["ASSOCIATION", "CORRELATION", "REGRESSION"].includes(kind)) {
    const correlations = record(n.correlations);
    const entries = Object.entries(correlations);
    const validR = (v: unknown): v is number => finite(v) && Math.abs(v) <= 1;
    if (entries.length) {
      if (!entries.every(([, r]) => validR(r))) return unavailable("The recorded correlations contain missing or invalid values; predictor relationships cannot be compared.");
      const sorted = entries.map(([name, r]) => ({name, r: r as number})).sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
      out.answer = `${sorted[0].name} has the largest returned absolute correlation with ${metric} (r = ${number(sorted[0].r)}). This describes the sample, not a proven difference in predictive strength.`;
      out.findings = sorted.slice(0, 3).map(({name, r}) => `${name}: r = ${number(r)} (${r > 0 ? "positive" : r < 0 ? "negative" : "zero"} linear association).`);
      out.limitations.push("Per-predictor significance cannot be established from correlation coefficients alone.");
    } else {
      if (!validR(n.pearson_r)) return unavailable("A valid correlation coefficient was not recorded; the relationship cannot be characterized.");
      const predictor = strings(semantics.explanatory_columns)[0];
      out.answer = `${predictor ? `${predictor} and ${metric}` : "The recorded variables"} have ${n.pearson_r === 0 ? "zero" : n.pearson_r > 0 ? "positive" : "negative"} sample linear correlation (r = ${number(n.pearson_r)}).`;
      addInference(n.pearson_p, interval());
    }
    out.limitations.push("Correlation does not establish causation. Statistical power and robustness require separate assessments.");
  } else if (kind === "TREND") {
    if (!finite(n.slope_per_period)) return unavailable("A valid time trend was not recorded.");
    out.answer = `The fitted change in ${metric} is ${number(n.slope_per_period)} per ${text(n.grain) || "period"}${finite(n.periods) ? ` across ${number(n.periods)} periods` : ""}.`;
    if (finite(n.start_level) && finite(n.end_level)) out.findings.push(`The reported starting and ending levels are ${number(n.start_level)} and ${number(n.end_level)}${finite(n.pct_change) ? ` (${percent(n.pct_change)} change)` : ""}.`);
    addInference(n.slope_p, null);
    out.limitations.push("A fitted historical trend does not establish a cause or guarantee future change.");
  } else if (diagnostic) {
    out.answer = "The available analysis does not establish why this happened. Descriptive breakdowns and observed changes can identify where to investigate, but do not prove a cause.";
    if (finite(n.previous) && finite(n.current)) {
      out.answer = `Recorded ${metric} ${n.current > n.previous ? "increased" : n.current < n.previous ? "decreased" : "was unchanged"} from ${number(n.previous)} to ${number(n.current)}${text(n.period) ? ` in ${text(n.period)}` : ""}. The breakdown does not establish a cause for this change.`;
    }
    if (text(n.top_contributor) && finite(n.top_contribution_share)) out.findings.push(`${text(n.top_contributor)} accounts for ${percent(n.top_contribution_share)} of the recorded change; this is an arithmetic contribution, not a causal effect.`);
    out.limitations.push("Establishing a cause requires evidence that rules out competing explanations.");
  } else if (["DESCRIPTIVE", "RANKING"].includes(kind)) {
    const table = record(n.table);
    const entries = Object.entries(table);
    if (entries.length) {
      if (!entries.every(([, v]) => finite(record(v).value))) return unavailable("The returned breakdown contains missing or invalid values, so a complete summary is unavailable.");
      const sorted = entries.map(([name, v]) => ({name, value: record(v).value as number})).sort((a, b) => b.value - a.value);
      const high = sorted[0], low = sorted[sorted.length - 1];
      const highs = sorted.filter(v => v.value === high.value).map(v => v.name).join(", ");
      const lows = sorted.filter(v => v.value === low.value).map(v => v.name).join(", ");
      const additive = ["SUM", "COUNT", "ROW_COUNT"].includes(operation);
      const value = (v: number) => operation === "RATE" ? percent(v) : number(v);
      const metricLabel = `${operation === "SUM" ? "total" : operation.toLowerCase() || "reported"} ${metric}`;
      if (sorted.length === 1) out.answer = `${high.name} has ${metricLabel} of ${value(high.value)}. Only one group was returned.`;
      else if (high.value === low.value) out.answer = `All ${sorted.length} returned groups have the same ${metricLabel}: ${value(high.value)}.`;
      else {
        const lowestFirst = text(contract.ranking_direction || semantics.ranking_direction).toUpperCase() === "ASC" || text(contract.ranking_direction || semantics.ranking_direction).toUpperCase() === "LOWEST";
        out.answer = lowestFirst
          ? `${lows} has the lowest ${metricLabel} (${value(low.value)}); ${highs} has the highest (${value(high.value)}).`
          : `${highs} has the highest ${metricLabel} (${value(high.value)}); ${lows} has the lowest (${value(low.value)}).`;
        if (finite(high.value - low.value)) out.findings.push(`The highest-to-lowest gap is ${operation === "RATE" ? `${number((high.value - low.value) * 100)} percentage points` : value(high.value - low.value)}.`);
      }
      if (additive) {
        const total = sorted.reduce((sum, row) => sum + row.value, 0);
        if (finite(total)) {
          out.answer = `Total ${metric} across the ${sorted.length} returned groups is ${number(total)}. ${out.answer}`;
          if (total > 0 && sorted.every(v => v.value >= 0)) out.findings.unshift(`${highs} contributes ${percent(high.value / total)} of the returned total${highs.includes(", ") ? " per tied group" : ""}.`);
        }
      }
      out.limitations.push("These are descriptive results for the selected records; they do not establish statistical significance or explain causes.");
    } else if (finite(n.value) || typeof n.value === "string") {
      out.answer = `${operation.replaceAll("_", " ") || "Recorded value"} of ${metric}: ${finite(n.value) ? number(n.value) : n.value}.`;
    } else return unavailable("The recorded output does not contain the values needed to answer this question.");
  } else {
    // Specialist paths retain their backend-written answer rather than inventing a new one.
    out.answer = text(result.executive_summary) || text(result.headline);
    if (!out.answer) return unavailable("No structured analytical answer was recorded for this question.");
  }
  out.supported = true;
  return finish();
}
