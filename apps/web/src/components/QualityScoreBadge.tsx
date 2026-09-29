"use client";

import React from "react";
import { CheckCircle2, AlertTriangle, XCircle, ShieldCheck } from "lucide-react";
import { DataQualityBreakdown } from "../types";

interface Props {
  score: number | null;
  breakdown?: DataQualityBreakdown;
  size?: "sm" | "md" | "lg";
}

export default function QualityScoreBadge({ score, breakdown, size = "md" }: Props) {
  const getColors = (val: number) => {
    if (val >= 90) return { bg: "bg-emerald-500/10", border: "border-emerald-500/30", text: "text-emerald-400", bar: "bg-emerald-500" };
    if (val >= 70) return { bg: "bg-amber-500/10", border: "border-amber-500/30", text: "text-amber-400", bar: "bg-amber-500" };
    return { bg: "bg-rose-500/10", border: "border-rose-500/30", text: "text-rose-400", bar: "bg-rose-500" };
  };

  if (score == null || Number.isNaN(score)) {
    return (
      <div className={`p-4 rounded-xl border bg-slate-900/60 border-slate-800`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <ShieldCheck className="w-5 h-5 text-slate-500" />
            <span className="text-sm font-semibold text-white">Data Quality Score</span>
          </div>
          <span className="text-sm font-semibold font-mono text-slate-400">Not measured</span>
        </div>
      </div>
    );
  }

  const colors = getColors(score);

  if (size === "sm") {
    return (
      <span className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-xs font-semibold border ${colors.bg} ${colors.border} ${colors.text}`}>
        <ShieldCheck className="w-3 h-3" />
        <span>{score.toFixed(0)}/100</span>
      </span>
    );
  }

  return (
    <div className={`p-4 rounded-xl border ${colors.bg} ${colors.border}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <ShieldCheck className={`w-5 h-5 ${colors.text}`} />
          <span className="text-sm font-semibold text-white">Data Quality Score</span>
        </div>
        <span className={`text-xl font-bold font-mono ${colors.text}`}>{score.toFixed(1)}/100</span>
      </div>

      {/* Progress bar */}
      <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden mb-4">
        <div className={`h-full ${colors.bar} transition-all duration-500`} style={{ width: `${Math.min(100, Math.max(0, score))}%` }} />
      </div>

      {/* Breakdown metrics */}
      {breakdown && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex justify-between bg-slate-900/60 p-2 rounded border border-slate-800">
            <span className="text-slate-400">Missing Values</span>
            <span className="font-mono text-slate-200">{breakdown.missing_values_score.toFixed(0)}%</span>
          </div>
          <div className="flex justify-between bg-slate-900/60 p-2 rounded border border-slate-800">
            <span className="text-slate-400">Duplicates</span>
            <span className="font-mono text-slate-200">{breakdown.duplicate_rows_score.toFixed(0)}%</span>
          </div>
          <div className="flex justify-between bg-slate-900/60 p-2 rounded border border-slate-800">
            <span className="text-slate-400">Type Consistency</span>
            <span className="font-mono text-slate-200">{breakdown.type_consistency_score.toFixed(0)}%</span>
          </div>
          <div className="flex justify-between bg-slate-900/60 p-2 rounded border border-slate-800">
            <span className="text-slate-400">Outlier Risk</span>
            <span className="font-mono text-slate-200">{breakdown.outlier_risk_score.toFixed(0)}%</span>
          </div>
        </div>
      )}
    </div>
  );
}
