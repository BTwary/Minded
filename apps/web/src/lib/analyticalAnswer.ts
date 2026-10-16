import type { AnalysisResponse } from "../types";

type RecordValue = Record<string, unknown>;
export type AnswerPresentation = {
  answer: string;
  findings: string[];
  calculation: string[];
  limitations: string[];
  supported: boolean;
  table?: { headers: string[]; rows: string[][] };
};

const record = (v: unknown): RecordValue => v !== null && typeof v === "object" && !Array.isArray(v) ? v as RecordValue : {};
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const text = (v: unknown): string => typeof v === "string" ? v : "";
const strings = (v: unknown): string[] => Array.isArray(v) ? v.filter((s): s is string => typeof s === "string" && !!s.trim()) : [];
const number = (v: number): string => v.toLocaleString(undefined, { maximumSignificantDigits: 6 });
const percent = (v: number): string => `${(v * 100).toFixed(1)}%`;
const pValue = (v: unknown): string | null => finite(v) && v >= 0 && v <= 1 ? (v < 0.001 ? "p < 0.001" : `p = ${number(v)}`) : null;

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
  const out: AnswerPresentation = { answer: "", findings: [], calculation: [], limitations: strings(result.caveats), supported: false };
  // A proposed plan is not proof that a calculation ran.
  if (Object.keys(n).length) {
    if (metric !== "the metric") out.calculation.push(`Metric: ${metric}.`);
    if (operation) out.calculation.push(`Calculation: ${operation.replaceAll("_", " ")}${groups.length ? `, grouped by ${groups.join(", ")}` : ""}.`);
    const filters = n.filters ?? contract.filter_conditions;
    if (Array.isArray(filters)) out.calculation.push(filters.length ? `Applied filters: ${JSON.stringify(filters)}.` : "Filters: none applied in the recorded contract.");
    if (finite(n.population_row_count)) out.calculation.push(`Records in scope after filtering: ${number(n.population_row_count)}.`);
  }

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
      return out;
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
  out.findings = out.findings.slice(0, 3);
  out.limitations = Array.from(new Set(out.limitations));
  return out;
}
