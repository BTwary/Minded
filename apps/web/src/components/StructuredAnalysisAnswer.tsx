"use client";

import React, { useMemo, useState } from "react";
import { AnalysisResponse } from "../types";
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  FileDown,
  HelpCircle,
  Loader2,
  RotateCcw,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import { downloadInvestigationExport, submitHumanReview } from "../lib/api";
import { presentAnalyticalAnswer } from "../lib/analyticalAnswer";

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : null;
}

function label(value: string): string {
  return value
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function display(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Not recorded";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "Not recorded";
    if (Math.abs(value) >= 1_000_000) {
      return `${(value / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 2 })}M`;
    }
    if (Math.abs(value) >= 1_000) {
      return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
    }
    return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }
  if (typeof value === "string" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return "Not available";
  }
}

function formatCompactNum(val: number): string {
  if (Math.abs(val) >= 1_000_000) {
    return `${(val / 1_000_000).toFixed(2)}M`;
  }
  if (Math.abs(val) >= 1_000) {
    return val.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return val.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function verificationState(analysis: AnalysisResponse): string {
  // 1. If claim_gate refused or flagged blocking conditions, do not report VERIFIED
  const claimGate = analysis.claim_gate;
  if (claimGate?.outcome === "REFUSE") return "REFUSED";
  if (claimGate?.outcome === "QUALIFIED_ANSWER" || (claimGate?.blocking_conditions && claimGate.blocking_conditions.length > 0)) {
    return "QUALIFIED";
  }

  const result = record(analysis.analyst_result);
  const numbers = record(result?.numbers);
  const execution = record(analysis.execution_record) || record(numbers?.execution_record);
  if (execution) {
    if (execution.contract_status === "MISMATCH" || execution.verification_status === "FAILED") return "FAILED";
    if (execution.contract_status !== "MATCHED" || execution.verification_status !== "VERIFIED") return "UNVERIFIED";
  }

  const evidence = analysis.evidence || [];
  if (evidence.some((item) => String(item.validation_status).toUpperCase() === "FAILED")) return "FAILED";
  if (!evidence.length) return "UNVERIFIED";

  const checks = analysis.evidence_verifications || [];
  if (!checks.length) return "UNVERIFIED";

  // Require that evidence verifications actually passed
  const hasPassedChecks = checks.some((check) =>
    ["PASSED", "VERIFIED"].includes(String(check.status || "").toUpperCase())
  );
  if (!hasPassedChecks) return "UNVERIFIED";

  const allIndependentlyVerified = evidence.every((item) => {
    const itemStatus = String(item.validation_status).toUpperCase();
    if (!["VERIFIED", "PASSED"].includes(itemStatus)) return false;
    return checks.some((check) =>
      check.evidence_id === item.id &&
      ["PASSED", "VERIFIED"].includes(String(check.status || "").toUpperCase())
    );
  });
  return allIndependentlyVerified ? "VERIFIED" : "UNVERIFIED";
}

function consistencyFailure(analysis: AnalysisResponse, res: JsonRecord): string | null {
  const execution = record(analysis.execution_record) || record(record(res.numbers)?.execution_record);
  if (execution && execution.contract_status !== "MATCHED") {
    return "The execution has not been verified against the approved plan.";
  }
  if (execution && execution.verification_status !== "VERIFIED") {
    return "Independent verification of the executed calculation did not pass.";
  }
  const recordedVerdict = res.final_verdict;
  if (
    typeof recordedVerdict === "string" &&
    analysis.verdict &&
    recordedVerdict.toUpperCase() !== String(analysis.verdict).toUpperCase()
  ) {
    return "The saved answer verdict does not match the investigation verdict.";
  }

  const requested = Array.isArray(analysis.dataset_scope) ? analysis.dataset_scope : null;
  const provenance = record(analysis.provenance);
  const recordedScope = Array.isArray(provenance?.requested_dataset_ids)
    ? provenance.requested_dataset_ids
    : null;
  if (requested && recordedScope) {
    const requestedIds = requested.map(String).sort();
    const recordedIds = recordedScope.map(String).sort();
    if (JSON.stringify(requestedIds) !== JSON.stringify(recordedIds)) {
      return "The saved answer scope does not match its recorded dataset scope.";
    }
  }
  return null;
}

export default function StructuredAnalysisAnswer({
  analysis,
  onRedo,
  onRequestCorrection,
}: {
  analysis: AnalysisResponse;
  onRedo?: () => void;
  onRequestCorrection?: (comment: string) => void;
}) {
  const result = record(analysis.analyst_result);
  const numbers = record(result?.numbers) || {};
  const execution = record(analysis.execution_record) || record(numbers.execution_record);
  const executedDataset = record(execution?.dataset);
  const approvedContract = record(execution?.approved_contract);
  const kind = String(result?.kind || "").toUpperCase();
  const integrityFailure = result ? consistencyFailure(analysis, result) : null;
  const verification = verificationState(analysis);
  const answer = presentAnalyticalAnswer(analysis, verification, integrityFailure);
  const provenance = record(analysis.provenance);
  const manifestHash = provenance?.manifest_hash || analysis.reproducible_manifest_hash;
  const requestedIds = Array.isArray(provenance?.requested_dataset_ids)
    ? provenance.requested_dataset_ids
    : [];
  const versionIds = Array.isArray(provenance?.dataset_version_ids)
    ? provenance.dataset_version_ids
    : [];

  const [exportingFormat, setExportingFormat] = useState<"pdf" | "docx" | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [reviewStatus, setReviewStatus] = useState<string | null>(null);
  const [reviewNotice, setReviewNotice] = useState<string | null>(null);
  const [reviewComment, setReviewComment] = useState<string>("");
  const [isSubmittingReview, setIsSubmittingReview] = useState<boolean>(false);
  const [reviewActionForm, setReviewActionForm] = useState<"CORRECTION" | "REJECT" | null>(null);

  const handleReviewAction = async (action: "ACCEPT" | "REQUEST_CORRECTION" | "REJECT" | "REDO", comments?: string) => {
    if (!analysis.id || isSubmittingReview) return;
    setIsSubmittingReview(true);
    try {
      const res = await submitHumanReview(analysis.id, {
        action,
        comments: comments || reviewComment || undefined,
      });
      const rev = (res.review as any) || {};
      setReviewStatus(rev.review_status || action);
      setReviewNotice(rev.notice || "Review recorded successfully.");
      setReviewActionForm(null);
      if (action === "REDO" && onRedo) {
        onRedo();
      } else if (action === "REQUEST_CORRECTION" && onRequestCorrection) {
        onRequestCorrection(comments || reviewComment);
      }
    } catch (err) {
      console.error("Failed to submit review:", err);
      alert(err instanceof Error ? err.message : "Review submission failed.");
    } finally {
      setIsSubmittingReview(false);
    }
  };

  const handleExport = async (fmt: "pdf" | "docx") => {
    if (!analysis.id || exportingFormat) return;
    setExportingFormat(fmt);
    setExportError(null);
    try {
      const blob = await downloadInvestigationExport(analysis.id, fmt);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `MindEd_Investigation_${analysis.id.slice(0, 8)}.${fmt}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error(`Failed to export ${fmt.toUpperCase()}:`, err);
      setExportError(err instanceof Error ? err.message : `Failed to export as ${fmt.toUpperCase()}`);
    } finally {
      setExportingFormat(null);
    }
  };



  // 1. Headline
  const headline = typeof result?.headline === "string" ? result.headline : "";

  // 2. Key Result / Principal Value extraction
  const keyResultInfo = useMemo(() => {
    const totalObservations = typeof numbers.n === "number"
      ? numbers.n
      : analysis.evidence?.[0]?.row_count_analyzed;

    // Check if Ranking
    if (kind === "RANKING" && numbers.table) {
      const topName = String(numbers.top || "");
      const topVal = typeof numbers.top_value === "number" ? numbers.top_value : null;
      const tableData = record(numbers.table) || {};
      const topRecord = record(tableData[topName]);
      const topRows = typeof topRecord?.n === "number" ? topRecord.n : null;

      // calculate grand total if possible
      let grandTotal = 0;
      let totalRows = 0;
      for (const k of Object.keys(tableData)) {
        const item = record(tableData[k]);
        if (item) {
          if (typeof item.value === "number") grandTotal += item.value;
          if (typeof item.n === "number") totalRows += item.n;
        }
      }

      const additive = ["sum", "count", "row_count"].includes(String(numbers.aggregation || numbers.operation || "").toLowerCase());
      const shareStr = additive && topVal && grandTotal > 0
        ? `${((topVal / grandTotal) * 100).toFixed(1)}% of overall`
        : "";

      return {
        category: "OBSERVED",
        categorySub: `Descriptive analysis${totalRows > 0 ? ` · ${totalRows.toLocaleString()} rows` : ""}`,
        mainValue: topVal !== null ? formatCompactNum(topVal) : null,
        mainValueSub: [
          topRows ? `${topRows.toLocaleString()} rows` : null,
          shareStr || null,
        ].filter(Boolean).join(" · "),
      };
    }

    // Check if Association / Statistical
    if (["ASSOCIATION", "CORRELATION", "REGRESSION"].includes(kind)) {
      const r = typeof numbers.pearson_r === "number" ? numbers.pearson_r : null;
      const slope = typeof numbers.slope === "number" ? numbers.slope : null;
      const nObs = typeof numbers.n === "number" ? numbers.n : totalObservations;

      return {
        category: "ASSOCIATION",
        categorySub: "Not causal evidence",
        mainValue: r !== null ? `r = ${r.toFixed(3)}` : slope !== null ? `slope = ${slope.toFixed(4)}` : null,
        mainValueSub: nObs ? `${nObs.toLocaleString()} complete observations analyzed` : null,
      };
    }

    // General default
    const execSummary = typeof result?.executive_summary === "string" ? result.executive_summary : "";
    return {
      category: kind || "ANALYSIS",
      categorySub: totalObservations ? `${totalObservations.toLocaleString()} rows analyzed` : "Empirical analysis",
      mainValue: null,
      mainValueSub: execSummary && execSummary !== headline ? execSummary : null,
    };
  }, [kind, numbers, analysis.evidence, result?.executive_summary, headline]);

  // 3. Structured Results Table Construction
  const structuredTable = useMemo(() => {
    // A. Ranking Table
    if (kind === "RANKING" && numbers.table) {
      const rawTable = record(numbers.table) || {};
      const entries = Object.entries(rawTable).map(([name, data]) => {
        const d = record(data) || {};
        return {
          name,
          value: typeof d.value === "number" && Number.isFinite(d.value) ? d.value : null,
          rows: typeof d.n === "number" && Number.isFinite(d.n) ? d.n : null,
        };
      });

      // Sort descending by value
      entries.sort((a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity));
      const additive = ["sum", "count", "row_count"].includes(String(numbers.aggregation || numbers.operation || "").toLowerCase());
      const grandTotal = entries.every(e => e.value !== null && e.value >= 0)
        ? entries.reduce((acc, curr) => acc + (curr.value ?? 0), 0) : 0;

      const rows = entries.map((e, index) => {
        const sharePct = additive && e.value !== null && grandTotal > 0 ? ((e.value / grandTotal) * 100).toFixed(1) + "%" : "Not applicable";
        return {
          rank: index + 1,
          group: e.name,
          total: display(e.value),
          rows: display(e.rows),
          share: sharePct,
        };
      });

      return {
        title: "Ranking by Metric",
        type: "ranking",
        headers: ["Rank", "Group / Entity", `${label(String(numbers.aggregation || numbers.operation || "Recorded"))} value`, "Rows", "Share"],
        rows,
      };
    }

    // B. Statistical Evidence Table (Correlation, Regression, Association)
    if (["ASSOCIATION", "CORRELATION", "REGRESSION"].includes(kind) && (numbers.pearson_r !== undefined || numbers.slope !== undefined)) {
      const metrics: Array<{ metric: string; result: string }> = [];

      if (typeof numbers.pearson_r === "number") {
        metrics.push({ metric: "Pearson correlation (r)", result: numbers.pearson_r.toFixed(3) });
      }
      if (typeof numbers.ci_low === "number" && typeof numbers.ci_high === "number") {
        metrics.push({ metric: "95% confidence interval", result: `${numbers.ci_low.toFixed(3)} to ${numbers.ci_high.toFixed(3)}` });
      }
      if (typeof numbers.pearson_p === "number" && Number.isFinite(numbers.pearson_p) && numbers.pearson_p >= 0 && numbers.pearson_p <= 1) {
        metrics.push({ metric: "P-value", result: numbers.pearson_p < 0.001 ? "p < 0.001" : numbers.pearson_p.toFixed(4) });
      }
      if (typeof numbers.slope === "number") {
        metrics.push({ metric: "Simple regression slope", result: numbers.slope >= 0 ? `+${numbers.slope.toFixed(4)}` : numbers.slope.toFixed(4) });
      }
      if (typeof numbers.r_squared === "number") {
        metrics.push({ metric: "Variation explained (R²)", result: `${(numbers.r_squared * 100).toFixed(2)}%` });
      }

      const multiv = record(numbers.multivariate_context);
      if (multiv) {
        if (typeof multiv.controlled_slope === "number") {
          metrics.push({ metric: "Adjusted slope (controlled)", result: multiv.controlled_slope >= 0 ? `+${multiv.controlled_slope.toFixed(4)}` : multiv.controlled_slope.toFixed(4) });
        }
        if (typeof multiv.adjusted_r_squared === "number") {
          metrics.push({ metric: "Adjusted model R²", result: `${(multiv.adjusted_r_squared * 100).toFixed(2)}%` });
        }
      }

      if (typeof numbers.spearman_rho === "number") {
        metrics.push({ metric: "Spearman rank correlation (rho)", result: numbers.spearman_rho.toFixed(3) });
      }

      if (metrics.length > 0) {
        return {
          title: "Statistical evidence",
          type: "statistical",
          headers: ["Metric", "Result"],
          rows: metrics,
        };
      }
    }

    // Named comparisons use the returned means/rates, never totals substituted for means.
    if (answer.table) {
      return {
        title: "Group comparison",
        type: "generic",
        headers: answer.table.headers,
        rows: answer.table.rows.map(row => Object.fromEntries(row.map((value, index) => [String(index), value]))),
      };
    }

    // C. Generic Fallback Table
    const tableObj = record(numbers.table);
    if (tableObj) {
      const entries = Object.entries(tableObj).slice(0, 50);
      const subKeys = Array.from(new Set(entries.flatMap(([, val]) => {
        const rec = record(val);
        return rec ? Object.keys(rec) : ["value"];
      }))).filter((k) => k !== "n" && k !== "value");

      const grouping = Array.isArray(numbers.grouping_columns) ? numbers.grouping_columns.join(" / ") : "Item";
      const metric = [numbers.operation, numbers.target_column].filter(v => typeof v === "string").join(" ");
      const headers = [grouping, metric || "Value", ...subKeys.map(label)];
      const rows = entries.map(([k, val]) => {
        const rec = record(val) || { value: val };
        return {
          item: k,
          value: display(rec.value),
          ...Object.fromEntries(subKeys.map((sk) => [sk, display(rec[sk])])),
        };
      });

      return {
        title: "Results table",
        type: "generic",
        headers,
        rows,
      };
    }

    return null;
  }, [kind, numbers, answer.table]);

  // 4. Key Insights (max 3 short, non-redundant one-line insights)
  const keyInsights = answer.findings;

  // 5. Limitations & Caveats
  const limitations = useMemo(() => {
    const claimGate = analysis.claim_gate;
    const list: string[] = [...answer.limitations];
    if (claimGate?.outcome === "QUALIFIED_ANSWER") {
      if (claimGate.allowed_claim) list.push(`Allowed claim: ${claimGate.allowed_claim}`);
      if (claimGate.blocked_claim) list.push(`Not supported: ${claimGate.blocked_claim}`);
      if (claimGate.reason) list.push(`Claim limitation: ${claimGate.reason}`);
    }

    return list;
  }, [answer.limitations, analysis.claim_gate]);

  // 6. Recommended Actions
  const actions = useMemo(() => {
    return Array.isArray(result?.next_steps)
      ? result.next_steps.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
      : [];
  }, [result?.next_steps]);

  return (
    <section aria-labelledby="structured-answer-title" className="space-y-6 rounded-2xl border border-[var(--surface-border)] bg-[var(--surface-1)] p-6 shadow-sm">
      {/* Top Banner: Question & Integrity Badges */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--surface-border)] pb-4">
        <div>
          <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-[var(--accent)]">
            <span className="rounded bg-[var(--accent-soft)] px-2 py-0.5 border border-[var(--accent)]/30 font-mono">
              {keyResultInfo.category}
            </span>
            <span className="text-[var(--muted)]">{keyResultInfo.categorySub}</span>
          </div>
          <h2 id="structured-answer-title" className="mt-1.5 text-sm font-semibold text-[var(--foreground)]">
            {analysis.question}
          </h2>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-[10px] font-mono">
          <span className="rounded-full border border-[var(--surface-border)] bg-[var(--surface-2)] px-2.5 py-1 text-[var(--muted)]">
            Status: {analysis.status}
          </span>
          <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 ${
            integrityFailure || verification === "FAILED" || verification === "REFUSED"
              ? "border-rose-500/40 bg-rose-500/10 text-rose-300 font-semibold"
              : verification === "VERIFIED"
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300 font-semibold"
                : "border-amber-500/40 bg-amber-500/10 text-amber-300 font-semibold"
          }`}>
            <ShieldCheck className="h-3 w-3" />
            {integrityFailure ? "Consistency: FAILED" : `Verification: ${verification}`}
          </span>

          {analysis.id && (
            <div className="flex items-center gap-1.5 ml-1">
              <button
                type="button"
                onClick={() => void handleExport("pdf")}
                disabled={exportingFormat !== null}
                className="inline-flex items-center gap-1 rounded-lg border border-[var(--surface-border)] bg-[var(--surface-2)] px-2.5 py-1 text-[11px] font-sans font-semibold text-[var(--foreground)] hover:border-[var(--accent)] hover:text-[var(--accent)] disabled:opacity-50 transition-colors shadow-sm"
                title="Export this analysis as a publication-ready PDF in standard structured format"
              >
                {exportingFormat === "pdf" ? (
                  <Loader2 className="h-3 w-3 animate-spin text-[var(--accent)]" />
                ) : (
                  <FileDown className="h-3 w-3 text-[var(--accent)]" />
                )}
                <span>PDF</span>
              </button>

              <button
                type="button"
                onClick={() => void handleExport("docx")}
                disabled={exportingFormat !== null}
                className="inline-flex items-center gap-1 rounded-lg border border-[var(--surface-border)] bg-[var(--surface-2)] px-2.5 py-1 text-[11px] font-sans font-semibold text-[var(--foreground)] hover:border-sky-400 hover:text-sky-300 disabled:opacity-50 transition-colors shadow-sm"
                title="Export this analysis as an editable Microsoft Word document (.docx) in standard structured format"
              >
                {exportingFormat === "docx" ? (
                  <Loader2 className="h-3 w-3 animate-spin text-sky-400" />
                ) : (
                  <FileDown className="h-3 w-3 text-sky-400" />
                )}
                <span>DOCX</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {exportError && (
        <div role="alert" className="rounded-xl border border-rose-500/40 bg-rose-950/30 p-2.5 text-xs text-rose-200">
          {exportError}
        </div>
      )}

      {/* Consistency Failure Warning */}
      {integrityFailure && (
        <div role="alert" className="rounded-xl border border-rose-500/40 bg-rose-950/30 p-3.5 text-xs text-rose-200 flex items-start gap-2">
          <AlertCircle className="h-4 w-4 shrink-0 text-rose-400 mt-0.5" />
          <div>
            <strong>Verification failure:</strong> {integrityFailure} Do not treat this result as verified.
          </div>
        </div>
      )}

      {/* Answer first; the table and evidence below support this explanation. */}
      <section aria-label="Answer" className="rounded-xl border border-[var(--accent)]/30 bg-[var(--accent-soft)]/40 p-5 space-y-2">
        <h3 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)]">Answer</h3>
        <p className="text-lg font-semibold tracking-tight text-[var(--foreground)] leading-relaxed">
          {answer.answer}
        </p>
      </section>

      {/* 2. Key Result Metric Banner */}
      {answer.supported && keyResultInfo.mainValue && (
        <section className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)]/60 p-4">
          <div className="text-2xl font-extrabold tracking-tight text-[var(--accent)] font-mono">
            {keyResultInfo.mainValue}
          </div>
          {keyResultInfo.mainValueSub && (
            <div className="mt-1 text-xs text-[var(--muted)] font-medium">
              {keyResultInfo.mainValueSub}
            </div>
          )}
        </section>
      )}

      {keyInsights.length > 0 && (
        <section aria-label="Key findings" className="space-y-2">
          <h3 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">Key findings</h3>
          <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-[var(--foreground)]">
            {keyInsights.map((insight, idx) => <li key={idx}>{insight}</li>)}
          </ul>
        </section>
      )}

      {/* 3. Structured Results Table */}
      {structuredTable && (
        <section className="space-y-2.5">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">
            {structuredTable.title}
          </div>
          <div className="overflow-x-auto rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)]/40 shadow-sm">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-[var(--surface-border)] bg-[var(--surface-2)] text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
                <tr>
                  {structuredTable.headers.map((h, i) => (
                    <th key={i} scope="col" className="px-3.5 py-2.5 font-semibold">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--surface-border)] font-normal text-[var(--foreground)]">
                {structuredTable.type === "ranking" && (
                  (structuredTable.rows as Array<{ rank: number; group: string; total: string; rows: string; share: string }>).map((r, idx) => (
                    <tr key={idx} className="hover:bg-[var(--surface-2)]/60 transition-colors">
                      <td className="px-3.5 py-2.5 font-mono text-xs text-[var(--muted)] font-semibold">{r.rank}</td>
                      <td className="px-3.5 py-2.5 font-semibold text-[var(--foreground)]">{r.group}</td>
                      <td className="px-3.5 py-2.5 font-mono font-medium text-[var(--accent)]">{r.total}</td>
                      <td className="px-3.5 py-2.5 font-mono text-[var(--muted)]">{r.rows}</td>
                      <td className="px-3.5 py-2.5 font-mono font-medium">{r.share}</td>
                    </tr>
                  ))
                )}
                {structuredTable.type === "statistical" && (
                  (structuredTable.rows as Array<{ metric: string; result: string }>).map((r, idx) => (
                    <tr key={idx} className="hover:bg-[var(--surface-2)]/60 transition-colors">
                      <td className="px-3.5 py-2.5 font-medium text-[var(--muted)]">{r.metric}</td>
                      <td className="px-3.5 py-2.5 font-mono font-semibold text-[var(--foreground)]">{r.result}</td>
                    </tr>
                  ))
                )}
                {structuredTable.type === "generic" && (
                  (structuredTable.rows as Array<Record<string, unknown>>).map((r, idx) => (
                    <tr key={idx} className="hover:bg-[var(--surface-2)]/60 transition-colors">
                      {Object.values(r).map((val, cellIdx) => (
                        <td key={cellIdx} className="px-3.5 py-2.5 font-mono text-[var(--foreground)]">
                          {String(val)}
                        </td>
                      ))}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-[var(--muted-2)]">
            Recorded analytical output. Units follow the source metric; a currency is not assumed.
          </p>
        </section>
      )}

      <section aria-label="What MindEd did" className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)]/40 p-4 space-y-3">
        <h3 className="text-sm font-semibold text-[var(--foreground)]">What MindEd did</h3>
        {execution ? <>
          <p className="text-xs text-[var(--muted)]">
            Execution: <strong>{display(execution.execution_status)}</strong> · Plan match: <strong>{display(execution.contract_status)}</strong> · Independent verification: <strong>{display(execution.verification_status)}</strong>
          </p>
          {execution.contract_status !== "MATCHED" && <p role="alert" className="text-xs text-amber-600">The execution does not establish an answer to the approved question. Actual SQL is shown below where recorded.</p>}
          <dl className="grid gap-2 text-xs sm:grid-cols-2 text-[var(--muted)]">
            <div><dt className="font-semibold">Dataset used</dt><dd>{display(executedDataset?.name)} · ID {display(executedDataset?.dataset_id)}</dd></div>
            <div><dt className="font-semibold">Version used</dt><dd>{display(executedDataset?.dataset_version)} · {display(executedDataset?.dataset_version_id)}</dd></div>
            <div><dt className="font-semibold">Metric and aggregation executed</dt><dd>{display(execution.aggregation)}({display(execution.target_column)})</dd></div>
            <div><dt className="font-semibold">Grouping executed</dt><dd>{display(execution.grouping_columns)}</dd></div>
            <div><dt className="font-semibold">Filters executed</dt><dd>{display(execution.filters)}</dd></div>
            <div><dt className="font-semibold">Date range executed</dt><dd>{display(execution.time_range)}</dd></div>
            <div><dt className="font-semibold">Rows in executed scope</dt><dd>{display(execution.population_row_count)}</dd></div>
            <div><dt className="font-semibold">Result rows / groups</dt><dd>{display(execution.result_row_count)}</dd></div>
          </dl>
          {typeof execution.sql === "string" && <details><summary className="cursor-pointer text-xs font-semibold">SQL actually executed</summary><pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-xs text-[var(--muted)]">{execution.sql}</pre></details>}
          {record(execution.execution_specification) && <details><summary className="cursor-pointer text-xs font-semibold">Executed specification</summary><pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-xs text-[var(--muted)]">{JSON.stringify(execution.execution_specification, null, 2)}</pre></details>}
          {approvedContract && <details><summary className="cursor-pointer text-xs font-semibold">Approved plan (separate from execution)</summary><pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-xs text-[var(--muted)]">{JSON.stringify(approvedContract, null, 2)}</pre></details>}
        </> : <p className="text-xs text-[var(--muted)]">No reconciled execution record was saved for this analysis. Recorded evidence below may include executed queries; a proposed plan is not proof of execution.</p>}
      </section>

      {answer.calculation.length > 0 && (
        <section aria-label="How the result was calculated" className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)]/40 p-4 space-y-2">
          <h3 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">How the result was calculated</h3>
          {answer.calculation.map((detail, idx) => <p key={idx} className="text-xs text-[var(--muted)]">{detail}</p>)}
          <p className="text-xs text-[var(--muted)]">Recorded queries and source identifiers are available in Evidence and provenance below.</p>
        </section>
      )}

      {/* 5. Limitations & Claim Restrictions */}
      {limitations.length > 0 && (
        <section className="rounded-xl border border-amber-500/30 bg-amber-950/15 p-4 space-y-2">
          <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-[var(--foreground)]">
            <AlertTriangle className="h-3.5 w-3.5" />
            <span>Limitations &amp; Assumptions</span>
          </div>
          <div className="space-y-1.5 text-xs leading-relaxed text-[var(--muted)]">
            {limitations.map((item, idx) => (
              <p key={idx}>• {item}</p>
            ))}
          </div>
        </section>
      )}

      {/* 6. Recommended Actions */}
      {answer.supported && actions.length > 0 && (
        <section className="space-y-2">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">Recommended actions</div>
          <ol className="list-decimal space-y-1.5 pl-5 text-xs leading-relaxed text-[var(--foreground)]">
            {actions.map((action, idx) => (
              <li key={idx} className="pl-1">
                {action}
              </li>
            ))}
          </ol>
        </section>
      )}

      {/* 7. Expandable Evidence & Provenance */}
      <details className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)]/40 p-3.5 transition-all">
        <summary className="cursor-pointer text-xs font-semibold text-[var(--foreground)] flex items-center justify-between">
          <span>Evidence and provenance</span>
          <span className="text-[10px] font-mono text-[var(--muted)]">Expand to audit calculations</span>
        </summary>
        <div className="mt-3.5 space-y-3 text-xs text-[var(--muted)] border-t border-[var(--surface-border)] pt-3">
          <div>Independent verification: <span className="font-mono text-[var(--foreground)] font-semibold">{verification}</span></div>
          <div>Claim status: <span className="font-mono text-[var(--foreground)]">{analysis.claim_gate?.outcome || "NOT_RECORDED"}</span></div>
          {manifestHash && (
            <div className="break-all">Manifest SHA-256: <span className="font-mono text-[var(--foreground)]">{display(manifestHash)}</span></div>
          )}
          <div>Requested dataset IDs: <span className="font-mono text-[var(--foreground)]">{requestedIds.length ? requestedIds.map(String).join(", ") : "Not recorded"}</span></div>
          <div>Dataset version IDs: <span className="font-mono text-[var(--foreground)]">{versionIds.length ? versionIds.map(String).join(", ") : "Not recorded"}</span></div>

          {analysis.evidence && analysis.evidence.length > 0 ? (
            <div className="space-y-2 mt-2">
              {analysis.evidence.map((item) => (
                <details key={item.id} className="rounded-lg border border-[var(--surface-border)] bg-[var(--surface-1)] p-2.5">
                  <summary className="cursor-pointer text-[var(--foreground)] font-medium">
                    {item.statement} · <span className="text-[var(--accent)] font-mono text-[10px]">{item.validation_status}</span>
                  </summary>
                  <div className="mt-2 space-y-1 text-[11px]">
                    <p>{item.calculation_summary || "No calculation summary recorded."}</p>
                    <p>Rows analyzed: {item.row_count_analyzed != null ? item.row_count_analyzed.toLocaleString() : "Not recorded"}</p>
                    {item.p_value !== undefined && <p>p-value: {display(item.p_value)}</p>}
                    {item.effect_size !== undefined && <p>Effect size: {display(item.effect_size)}</p>}
                    {item.sql_executed && (
                      <pre className="overflow-x-auto whitespace-pre-wrap rounded bg-black/40 p-2 font-mono text-[10px] text-[var(--muted)]">
                        {item.sql_executed}
                      </pre>
                    )}
                  </div>
                </details>
              ))}
            </div>
          ) : (
            <p className="text-xs">No evidence records attached to this result.</p>
          )}
          <p className="text-[10px] text-[var(--muted-2)]">
            This view displays canonical persisted output and provenance. It guarantees zero retroactive modification.
          </p>
        </div>
      </details>

      {/* 8. Human Review & Decision Signoff */}
      <div className="rounded-xl border border-cyan-500/20 bg-slate-950/70 p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="text-[10px] font-bold uppercase tracking-wider text-cyan-400">Collaborative Review & Decision</div>
            <div className="text-xs font-semibold text-white">Direct the analysis or record your review outcome</div>
          </div>
          {reviewStatus && (
            <div className={`px-2.5 py-0.5 rounded-full text-[10px] font-semibold border ${
              reviewStatus.includes("ACCEPT")
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                : reviewStatus === "REJECTED"
                  ? "border-rose-500/30 bg-rose-500/10 text-rose-300"
                  : "border-amber-500/30 bg-amber-500/10 text-amber-200"
            }`}>
              Outcome: {reviewStatus}
            </div>
          )}
        </div>

        {reviewNotice && (
          <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-2.5 text-xs text-slate-300 leading-relaxed">
            {reviewNotice}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-2 pt-1">
          <button
            type="button"
            disabled={isSubmittingReview}
            onClick={() => handleReviewAction("ACCEPT")}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition disabled:opacity-50 flex items-center gap-1.5 shadow-sm"
            title="Accept this result for your workflow. Does not override machine verification."
          >
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Accept for Workflow</span>
          </button>

          <button
            type="button"
            disabled={isSubmittingReview}
            onClick={() => setReviewActionForm(reviewActionForm === "CORRECTION" ? null : "CORRECTION")}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-amber-500/40 text-amber-300 hover:bg-amber-500/10 transition disabled:opacity-50 flex items-center gap-1.5"
            title="Request a correction or explain a missing dimension"
          >
            <HelpCircle className="w-3.5 h-3.5" />
            <span>Request Correction</span>
          </button>

          <button
            type="button"
            disabled={isSubmittingReview}
            onClick={() => setReviewActionForm(reviewActionForm === "REJECT" ? null : "REJECT")}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-rose-500/40 text-rose-300 hover:bg-rose-500/10 transition disabled:opacity-50 flex items-center gap-1.5"
            title="Mark result as unusable or unsupported with documented reason"
          >
            <XCircle className="w-3.5 h-3.5" />
            <span>Reject Result</span>
          </button>

          {onRedo && (
            <button
              type="button"
              disabled={isSubmittingReview}
              onClick={() => handleReviewAction("REDO")}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-slate-700 text-slate-300 hover:text-white hover:bg-slate-800 transition disabled:opacity-50 flex items-center gap-1.5 ml-auto"
              title="Redo analysis with revised metric, filters, or method"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Redo Analysis</span>
            </button>
          )}
        </div>

        {/* Input prompt for Correction or Rejection */}
        {reviewActionForm && (
          <div className="space-y-2 pt-2 border-t border-slate-800/80">
            <label className="text-[10px] font-bold uppercase text-slate-400">
              {reviewActionForm === "CORRECTION"
                ? "Describe what needs correction or what factor was missing:"
                : "Reason for rejecting this result (required):"}
            </label>
            <textarea
              value={reviewComment}
              onChange={(e) => setReviewComment(e.target.value)}
              placeholder={reviewActionForm === "CORRECTION"
                ? "e.g. 'Use gross margin instead of net revenue' or 'Filter out wholesale accounts'"
                : "e.g. 'Data quality issues in region column make conclusion unsupported'"}
              rows={2}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
            />
            <div className="flex gap-2">
              <button
                type="button"
                disabled={isSubmittingReview || (reviewActionForm === "REJECT" && !reviewComment.trim())}
                onClick={() => handleReviewAction(reviewActionForm === "CORRECTION" ? "REQUEST_CORRECTION" : "REJECT", reviewComment)}
                className={`px-3 py-1 rounded-lg text-xs font-semibold transition disabled:opacity-50 ${
                  reviewActionForm === "CORRECTION"
                    ? "bg-amber-600 hover:bg-amber-500 text-white"
                    : "bg-rose-600 hover:bg-rose-500 text-white"
                }`}
              >
                Submit {reviewActionForm === "CORRECTION" ? "Correction Request" : "Rejection"}
              </button>
              <button
                type="button"
                onClick={() => setReviewActionForm(null)}
                className="px-3 py-1 rounded-lg text-xs text-slate-400 hover:text-white border border-slate-800"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
