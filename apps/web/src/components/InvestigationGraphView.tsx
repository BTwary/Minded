"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  Clock,
  Compass,
  Cpu,
  Database,
  GitBranch,
  Info,
  Layers,
  Network,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { Hypothesis } from "../types";

interface Props {
  hypotheses: Hypothesis[];
  entropy?: number;
  objective?: string;
  targetMetric?: string;
}

export default function InvestigationGraphView({
  hypotheses,
  entropy,
  objective = "Diagnostic",
  targetMetric = "Primary Metric",
}: Props) {
  const [selectedHypothesis, setSelectedHypothesis] = useState<Hypothesis | null>(
    hypotheses.length > 0 ? hypotheses[0] : null
  );

  if (!hypotheses || hypotheses.length === 0) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 text-center text-xs text-slate-400">
        <Network className="w-8 h-8 text-slate-600 mx-auto mb-2 animate-pulse" />
        No active Investigation Graph nodes initialized.
      </div>
    );
  }

  const getBeliefBadge = (status: string) => {
    switch (status.toLowerCase()) {
      case "supported":
      case "highly_likely":
        return {
          label: "SUPPORTED",
          bg: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
          icon: <CheckCircle2 className="w-3 h-3 text-emerald-400" />,
        };
      case "rejected":
      case "refuted":
        return {
          label: "REFUTED",
          bg: "bg-rose-500/10 text-rose-400 border-rose-500/30",
          icon: <AlertTriangle className="w-3 h-3 text-rose-400" />,
        };
      case "weakened":
        return {
          label: "WEAKENED",
          bg: "bg-amber-500/10 text-amber-400 border-amber-500/30",
          icon: <Clock className="w-3 h-3 text-amber-400" />,
        };
      default:
        return {
          label: "UNTESTED",
          bg: "bg-slate-500/10 text-slate-400 border-slate-500/30",
          icon: <Cpu className="w-3 h-3 text-slate-400" />,
        };
    }
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-6">
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold text-indigo-400 uppercase tracking-wider">
            <Network className="w-4 h-4" />
            <span>Formal Epistemic Investigation DAG</span>
          </div>
          <h3 className="text-base font-bold text-white mt-0.5">
            Competing Hypotheses & Bayesian Belief Distribution
          </h3>
        </div>

        {/* Global Entropy Metric */}
        {entropy !== undefined && (
          <div className="flex items-center space-x-3 bg-slate-950 px-3.5 py-1.5 rounded-xl border border-slate-800">
            <Activity className="w-3.5 h-3.5 text-cyan-400" />
            <div className="text-xs">
              <span className="text-slate-400 mr-1.5">Shannon Entropy:</span>
              <span className="font-mono font-bold text-cyan-300">
                {entropy.toFixed(3)} bits
              </span>
            </div>
          </div>
        )}
      </div>

      {/* DAG Visualization Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Interactive Node Tree */}
        <div className="lg:col-span-7 space-y-3">
          {/* Root Unknown Node */}
          <div className="bg-slate-950 border border-indigo-500/30 rounded-xl p-3.5 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-lg bg-indigo-500/20 border border-indigo-500/40 flex items-center justify-center">
                <Compass className="w-4 h-4 text-indigo-400" />
              </div>
              <div>
                <div className="text-[10px] font-mono text-indigo-400 uppercase font-semibold">
                  Root Objective ({objective})
                </div>
                <div className="text-xs font-bold text-white">
                  Target Variable: {targetMetric}
                </div>
              </div>
            </div>
            <span className="text-[10px] font-mono bg-indigo-500/10 text-indigo-300 px-2 py-0.5 rounded border border-indigo-500/20">
              {hypotheses.length} Competing Nodes
            </span>
          </div>

          <div className="flex justify-center -my-1">
            <ArrowDown className="w-4 h-4 text-slate-600" />
          </div>

          {/* Hypothesis Children Cards */}
          <div className="space-y-2.5">
            {hypotheses.map((h, idx) => {
              const badge = getBeliefBadge(h.status);
              const isSelected = selectedHypothesis?.id === h.id;
              const probPct = Math.round(h.priority * 100);

              return (
                <div
                  key={h.id || idx}
                  onClick={() => setSelectedHypothesis(h)}
                  className={`cursor-pointer rounded-xl p-4 transition-all border ${
                    isSelected
                      ? "bg-slate-800/90 border-cyan-500/50 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-500/30"
                      : "bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-800/40"
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start space-x-3">
                      <span className="text-[11px] font-mono font-bold text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
                        {h.id || `H-${idx + 1}`}
                      </span>
                      <div>
                        <div className="text-xs font-semibold text-white line-clamp-1">
                          {h.statement}
                        </div>
                        <div className="text-[11px] text-slate-400 mt-1 line-clamp-1">
                          {h.rationale}
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-col items-end space-y-1.5 flex-shrink-0">
                      <span
                        className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full border flex items-center space-x-1 ${badge.bg}`}
                      >
                        {badge.icon}
                        <span>{badge.label}</span>
                      </span>

                      {/* Posterior Probability Meter */}
                      <div className="flex items-center space-x-1.5 text-[11px] font-mono">
                        <span className="text-slate-400">P(H|E):</span>
                        <span className="font-bold text-cyan-300">{probPct}%</span>
                      </div>
                    </div>
                  </div>

                  {/* Visual posterior bar */}
                  <div className="w-full bg-slate-900 rounded-full h-1.5 mt-3 overflow-hidden border border-slate-800">
                    <div
                      className={`h-full transition-all ${
                        h.status.toLowerCase().includes("supported") ||
                        h.status.toLowerCase().includes("highly")
                          ? "bg-emerald-400"
                          : h.status.toLowerCase().includes("rejected") ||
                            h.status.toLowerCase().includes("refuted")
                          ? "bg-rose-400"
                          : "bg-cyan-400"
                      }`}
                      style={{ width: `${Math.max(5, probPct)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Detailed Node Inspector */}
        <div className="lg:col-span-5 bg-slate-950 border border-slate-800 rounded-xl p-5 space-y-4 flex flex-col justify-between">
          {selectedHypothesis ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-mono font-bold text-cyan-400">
                    {selectedHypothesis.id}
                  </span>
                  <span className="text-xs font-semibold text-white">
                    Node Inspector
                  </span>
                </div>
                {(() => {
                  const b = getBeliefBadge(selectedHypothesis.status);
                  return (
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded-full border flex items-center space-x-1 ${b.bg}`}
                    >
                      {b.icon}
                      <span>{b.label}</span>
                    </span>
                  );
                })()}
              </div>

              <div>
                <label className="text-[10px] font-mono uppercase text-slate-400 font-semibold block mb-1">
                  Hypothesis Formulation
                </label>
                <p className="text-xs font-medium text-slate-200 leading-relaxed bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  {selectedHypothesis.statement}
                </p>
              </div>

              <div>
                <label className="text-[10px] font-mono uppercase text-slate-400 font-semibold block mb-1">
                  Epistemic Rationale
                </label>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {selectedHypothesis.rationale}
                </p>
              </div>

              {selectedHypothesis.reason_for_rejection && (
                <div className="bg-rose-500/10 border border-rose-500/20 rounded-lg p-3">
                  <div className="flex items-center space-x-1.5 text-xs font-semibold text-rose-400 mb-1">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    <span>Falsification Evidence</span>
                  </div>
                  <p className="text-xs text-rose-300">
                    {selectedHypothesis.reason_for_rejection}
                  </p>
                </div>
              )}

              {selectedHypothesis.investigation_steps &&
                selectedHypothesis.investigation_steps.length > 0 && (
                  <div>
                    <label className="text-[10px] font-mono uppercase text-slate-400 font-semibold block mb-1.5">
                      Completed Analytical Tests
                    </label>
                    <div className="space-y-1.5">
                      {selectedHypothesis.investigation_steps.map((step, sIdx) => (
                        <div
                          key={sIdx}
                          className="flex items-center space-x-2 text-xs text-slate-300 bg-slate-900/60 px-2.5 py-1.5 rounded border border-slate-800 font-mono"
                        >
                          <Zap className="w-3 h-3 text-cyan-400" />
                          <span>{step}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
            </div>
          ) : (
            <div className="text-center py-12 text-xs text-slate-500">
              Select a hypothesis node to inspect its Bayesian likelihood updates and falsification tests.
            </div>
          )}

          <div className="pt-3 border-t border-slate-800 text-[11px] text-slate-500 flex items-center justify-between">
            <span>Bayesian Normalizer: Active</span>
            <span className="text-cyan-400 font-mono">P(H|E) ∝ P(E|H) · P(H)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
