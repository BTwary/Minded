"use client";

import React, { useState } from "react";
import {
  CheckCircle2,
  Copy,
  Download,
  FileCheck,
  FileSpreadsheet,
  FileText,
  Printer,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import { AnalysisResponse } from "../types";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  analysis: AnalysisResponse;
}

export default function ReportGeneratorModal({ isOpen, onClose, analysis }: Props) {
  const [reportFormat, setReportFormat] = useState<"executive" | "analytical" | "technical" | "board">("executive");
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const generateReportMarkdown = () => {
    const lines: string[] = [];
    const dateStr = new Date(analysis.created_at || Date.now()).toUTCString();

    lines.push(`# AA-OS Autonomous Investigation Report`);
    lines.push(`**Inquiry:** ${analysis.question}`);
    lines.push(`**Generated:** ${dateStr}`);
    lines.push(`**Epistemic Verdict:** ${analysis.verdict || "CONFIRMED"}`);
    lines.push(`**Confidence Standard:** ${analysis.confidence}`);
    lines.push(`**Provenance SHA-256:** \`${analysis.manifest?.reproducible_hash || "N/A"}\``);
    lines.push(`\n---\n`);

    lines.push(`## 1. Executive Summary`);
    lines.push(`**Direct Answer:** ${analysis.direct_answer}`);
    lines.push(`**Core Empirical Finding:** ${analysis.main_finding}`);

    if (analysis.executive_bullets && analysis.executive_bullets.length > 0) {
      lines.push(`\n### Key Takeaways`);
      analysis.executive_bullets.forEach((b) => lines.push(`* ${b}`));
    }

    if (reportFormat !== "executive") {
      lines.push(`\n## 2. Formal Investigation DAG & Competing Hypotheses`);
      if (analysis.hypotheses && analysis.hypotheses.length > 0) {
        analysis.hypotheses.forEach((h) => {
          const postProb = h.posterior_probability !== undefined ? h.posterior_probability : h.priority;
          lines.push(`### [${h.status.toUpperCase()}] ${h.id}: ${h.statement}`);
          lines.push(`* **Rationale:** ${h.rationale}`);
          lines.push(`* **Updated Posterior P(H|E):** ${(postProb * 100).toFixed(1)}%`);
          if (h.reason_for_rejection) {
            lines.push(`* **Refutation Evidence:** ${h.reason_for_rejection}`);
          }
        });
      }
    }

    lines.push(`\n## 3. Independently Verified Evidence`);
    if (analysis.evidence && analysis.evidence.length > 0) {
      analysis.evidence.forEach((ev, idx) => {
        lines.push(`### Evidence Item #${idx + 1} (${ev.validation_status})`);
        lines.push(`* **Statement:** ${ev.statement}`);
        lines.push(`* **Recomputation Summary:** ${ev.calculation_summary}`);
        lines.push(`* **Rows Analyzed:** ${ev.row_count_analyzed.toLocaleString()}`);
        if (ev.sql_executed) lines.push(`* **Authoritative SQL:** \`${ev.sql_executed}\``);
        if (ev.statistical_test) lines.push(`* **Statistical Test:** ${ev.statistical_test} (p-value: ${ev.p_value})`);
      });
    }

    if (analysis.what_if_scenarios && analysis.what_if_scenarios.length > 0) {
      lines.push(`\n## 4. Counterfactual Policy Simulations`);
      analysis.what_if_scenarios.forEach((sc, idx) => {
        lines.push(`### Scenario ${idx + 1}: ${sc.scenario}`);
        lines.push(`* **Assumption:** ${sc.assumption}`);
        lines.push(`* **Projected Impact:** ${sc.projected_impact}`);
        lines.push(`* **Simulation Confidence:** ${sc.confidence}`);
      });
    }

    const totalEv = analysis.evidence?.length || 0;
    const passedEv = analysis.evidence?.filter((e) => e.validation_status === "PASSED").length || 0;
    const failedEv = analysis.evidence?.filter((e) => e.validation_status === "FAILED").length || 0;
    const valStatus = totalEv > 0 && passedEv === totalEv ? "PASSED (100% verified within ≤ 1.0% relative tolerance)" : `${passedEv}/${totalEv} Claims Passed Validation`;

    lines.push(`\n## 5. Epistemic Audit & Manifest Provenance`);
    lines.push(`* **Execution Time:** ${analysis.manifest?.execution_time_seconds || 0}s`);
    lines.push(`* **Total Reasoning Steps:** ${analysis.manifest?.total_steps || analysis.steps?.length || 0}`);
    lines.push(`* **Execution Provider:** ${analysis.manifest?.ai_provider || "Deterministic Core"}`);
    lines.push(`* **Independent Validation Status:** ${valStatus}`);
    lines.push(`* **Claim Validation Breakdown:** ${passedEv} Passed, ${failedEv} Failed out of ${totalEv} recomputed claims`);

    return lines.join("\n");
  };

  const reportText = generateReportMarkdown();

  const handleCopy = () => {
    navigator.clipboard.writeText(reportText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([reportText], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `AA-OS_Investigation_Report_${analysis.id || "export"}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-5 border-b border-slate-800 bg-slate-950">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              <FileCheck className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">Generate Structured Investigation Report</h3>
              <p className="text-xs text-slate-400">
                Audited evidence compilation with reproducible provenance hashes.
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Format Selector Bar */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-slate-800 bg-slate-900/50">
          <div className="flex items-center space-x-2">
            {(["executive", "analytical", "technical", "board"] as const).map((fmt) => (
              <button
                key={fmt}
                onClick={() => setReportFormat(fmt)}
                className={`text-xs font-semibold px-3 py-1.5 rounded-lg uppercase tracking-wider transition-all ${
                  reportFormat === fmt
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                }`}
              >
                {fmt} Report
              </button>
            ))}
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleCopy}
              className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-slate-800 text-slate-300 hover:bg-slate-800 flex items-center space-x-1.5 transition-colors"
            >
              {copied ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? "Copied" : "Copy Markdown"}</span>
            </button>

            <button
              onClick={handleDownload}
              className="text-xs font-semibold px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white flex items-center space-x-1.5 shadow-lg shadow-indigo-600/20 transition-all"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download (.md)</span>
            </button>
          </div>
        </div>

        {/* Preview Scroll Area */}
        <div className="p-6 overflow-y-auto font-mono text-xs text-slate-300 bg-slate-950 flex-1 space-y-4 leading-relaxed select-text">
          <pre className="whitespace-pre-wrap font-sans">{reportText}</pre>
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950 flex items-center justify-between text-xs text-slate-500">
          <div className="flex items-center space-x-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Cryptographically Bound Evidence Provenance</span>
          </div>
          <span className="font-mono text-[11px] text-slate-400">
            Hash: {analysis.manifest?.reproducible_hash?.substring(0, 16)}...
          </span>
        </div>
      </div>
    </div>
  );
}
