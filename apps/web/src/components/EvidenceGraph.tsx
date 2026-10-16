"use client";

import React, { useState } from "react";
import { ArrowRight, CheckCircle2, ChevronDown, ChevronUp, Code2, Database, GitBranch, Layers, ShieldCheck } from "lucide-react";
import { EvidenceGraphNode } from "../types";

interface EvidenceGraphProps {
  nodes?: EvidenceGraphNode[];
}

export default function EvidenceGraph({ nodes = [] }: EvidenceGraphProps) {
  const [isOpen, setIsOpen] = useState(true);

  if (!nodes || nodes.length === 0) return null;

  const validationLabel = (node: EvidenceGraphNode) => String(node.validation_status || "UNVERIFIED").toUpperCase();

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between cursor-pointer select-none"
      >
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <GitBranch className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <span>Auditable Evidence Graph &amp; Provenance Lineage</span>
            </h3>
            <p className="text-[11px] text-slate-400">
              Recorded evidence and provenance details. Validation status is shown for each evidence item.
            </p>
          </div>
        </div>

        <button className="text-slate-400 hover:text-white p-1 rounded-lg">
          {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>

      {isOpen && (
        <div className="space-y-3 pt-2">
          {nodes.map((node, idx) => (
            <div
              key={idx}
              className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-3 font-mono text-xs"
            >
              {/* Chain Steps */}
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="bg-slate-900 text-cyan-300 px-2.5 py-1 rounded-lg border border-slate-800 font-bold">
                  {node.finding_id}
                </span>
                <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
                <span className="bg-indigo-950/40 text-indigo-300 px-2.5 py-1 rounded-lg border border-indigo-500/30">
                  Evidence: {node.evidence_id}
                </span>
                <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
                <span className="bg-slate-900 text-amber-300 px-2.5 py-1 rounded-lg border border-slate-800">
                  Tool: {node.tool_used}
                </span>
                <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
                <span className="bg-slate-900 text-slate-300 px-2.5 py-1 rounded-lg border border-slate-800 flex items-center space-x-1">
                  <Database className="w-3 h-3 text-cyan-400" />
                  <span>{node.table_name}.parquet (v{node.dataset_version})</span>
                </span>
                <span className={`ml-auto inline-flex items-center space-x-1 text-[11px] font-semibold ${
                  node.tolerance_verified && ["PASSED", "VERIFIED"].includes(validationLabel(node))
                    ? "text-emerald-400"
                    : validationLabel(node) === "FAILED"
                      ? "text-rose-300"
                      : "text-amber-200"
                }`}>
                  {node.tolerance_verified && ["PASSED", "VERIFIED"].includes(validationLabel(node))
                    ? <CheckCircle2 className="w-3.5 h-3.5" />
                    : <ShieldCheck className="w-3.5 h-3.5" />}
                  <span>{validationLabel(node)}{node.tolerance_verified ? " · tolerance verified" : ""}</span>
                </span>
              </div>

              {/* Executed Query Code */}
              {node.query_executed && (
                <div className="space-y-1 pt-1">
                  <span className="text-[10px] text-slate-500 uppercase tracking-wider block">
                    Executed Deterministic SQL Query:
                  </span>
                  <pre className="p-2.5 rounded-lg bg-slate-900 border border-slate-800/80 text-[11px] text-cyan-300 overflow-x-auto">
                    {node.query_executed.trim()}
                  </pre>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
