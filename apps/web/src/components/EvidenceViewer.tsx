"use client";

import React, { useState } from "react";
import { CheckCircle2, ChevronDown, ChevronRight, Database, FileCode, Layers, ShieldCheck, GitBranch } from "lucide-react";
import { EvidenceItem } from "../types";

interface Props {
  evidence: EvidenceItem[];
}

export default function EvidenceViewer({ evidence }: Props) {
  const [openItem, setOpenItem] = useState<string | null>(evidence[0]?.id || null);

  const toggle = (id: string) => {
    setOpenItem(openItem === id ? null : id);
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-semibold text-white">Mathematical & Auditable Evidence</h3>
        </div>
        <span className="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded-full font-mono">
          100% Deterministic Verification
        </span>
      </div>

      <div className="space-y-3">
        {evidence.map((ev) => {
          const isOpen = openItem === ev.id;
          return (
            <div
              key={ev.id}
              className="border border-slate-800 rounded-lg bg-slate-950/50 overflow-hidden"
            >
              <button
                onClick={() => toggle(ev.id)}
                className="w-full p-3.5 flex items-start justify-between text-left hover:bg-slate-900/40 transition-colors"
              >
                <div className="space-y-1 pr-4">
                  <div className="flex items-center space-x-2">
                    <span className="text-[11px] font-mono bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded border border-slate-700">
                      {ev.id}
                    </span>
                    <span className="text-xs font-semibold text-white">{ev.statement}</span>
                  </div>
                  <p className="text-xs text-slate-400">{ev.calculation_summary}</p>
                </div>

                <div className="flex items-center space-x-2 flex-shrink-0 mt-0.5">
                  <span className="inline-flex items-center space-x-1 text-[11px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded-full">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>{ev.validation_status}</span>
                  </span>
                  {isOpen ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
                </div>
              </button>

              {isOpen && (
                <div className="p-4 bg-slate-900/80 border-t border-slate-800 text-xs space-y-3">
                  {/* Meta pills */}
                  <div className="flex flex-wrap gap-2 text-[11px] font-mono">
                    <div className="bg-slate-950 border border-slate-800 px-2.5 py-1 rounded text-slate-300 flex items-center space-x-1.5">
                      <Layers className="w-3 h-3 text-indigo-400" />
                      <span>Dataset: {ev.dataset_version}</span>
                    </div>
                    <div className="bg-slate-950 border border-slate-800 px-2.5 py-1 rounded text-slate-300 flex items-center space-x-1.5">
                      <Database className="w-3 h-3 text-sky-400" />
                      <span>Rows Analyzed: {ev.row_count_analyzed?.toLocaleString()}</span>
                    </div>
                    {ev.time_window && (
                      <div className="bg-slate-950 border border-slate-800 px-2.5 py-1 rounded text-slate-300">
                        Window: {ev.time_window}
                      </div>
                    )}
                  </div>

                  {/* SQL Statement if executed */}
                  {ev.sql_executed && (
                    <div>
                      <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block mb-1 flex items-center space-x-1">
                        <FileCode className="w-3 h-3 text-sky-400" />
                        <span>Exact DuckDB SQL Query</span>
                      </span>
                      <pre className="p-3 rounded bg-slate-950 border border-slate-800 text-sky-300 font-mono text-[11px] overflow-x-auto whitespace-pre-wrap">
                        {ev.sql_executed}
                      </pre>
                    </div>
                  )}

                  {/* Calculation Lineage */}
                  {ev.calculation_trace && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 flex items-center space-x-1">
                          <GitBranch className="w-3 h-3 text-emerald-400" />
                          <span>Calculation Trace</span>
                        </span>
                        <span className="text-[10px] font-mono text-slate-500">{ev.calculation_trace.steps.length} lineage steps</span>
                      </div>
                      <div className="rounded bg-slate-950 border border-slate-800 p-3 space-y-2">
                        {ev.calculation_trace.steps.map((step, idx) => (
                          <div key={step.step_id} className="border-l-2 border-slate-700 pl-3 py-1.5">
                            <div className="flex items-center justify-between gap-3">
                              <div className="text-[11px] text-white font-semibold">{idx + 1}. {step.title}</div>
                              <span className="text-[9px] font-mono uppercase text-slate-500">{step.kind}</span>
                            </div>
                            {step.formula && (
                              <div className="mt-1.5">
                                <div className="text-[9px] uppercase tracking-wider text-slate-500 mb-0.5">Formula</div>
                                <pre className="text-[10px] text-amber-300 font-mono whitespace-pre-wrap break-words">{step.formula}</pre>
                              </div>
                            )}
                            {step.executable_expression && step.executable_expression !== step.formula && (
                              <div className="mt-1.5">
                                <div className="text-[9px] uppercase tracking-wider text-slate-500 mb-0.5">Executable expression</div>
                                <pre className="text-[10px] text-sky-300 font-mono whitespace-pre-wrap break-words">{step.executable_expression}</pre>
                              </div>
                            )}
                            {(Object.keys(step.inputs).length > 0 || Object.keys(step.parameters).length > 0 || Object.keys(step.output).length > 0) && (
                              <details className="mt-1.5">
                                <summary className="cursor-pointer text-[9px] uppercase tracking-wider text-slate-500 hover:text-slate-300">Inputs · parameters · outputs</summary>
                                <pre className="mt-1.5 p-2 rounded bg-black/30 text-[10px] text-slate-300 font-mono overflow-x-auto">{JSON.stringify({ inputs: step.inputs, parameters: step.parameters, output: step.output }, null, 2)}</pre>
                              </details>
                            )}
                            {step.execution_engine && (
                              <div className="mt-1 text-[9px] font-mono text-slate-500">Engine: {step.execution_engine}{step.verification_status ? ` · ${step.verification_status}` : ""}</div>
                            )}
                            {step.notes && <div className="mt-1 text-[10px] text-slate-500">{step.notes}</div>}
                          </div>
                        ))}
                        <div className="pt-2 mt-2 border-t border-slate-800">
                          <div className="text-[9px] uppercase tracking-wider text-slate-500 mb-1">Trace hash</div>
                          <div className="font-mono text-[10px] text-slate-300 break-all">{ev.calculation_trace.canonical_hash}</div>
                          <div className="mt-1 text-[9px] text-slate-500">{ev.calculation_trace.reproducibility_statement}</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Raw Metrics JSON */}
                  {ev.raw_metrics && Object.keys(ev.raw_metrics).length > 0 && (
                    <div>
                      <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block mb-1">
                        Recomputed Raw Metrics
                      </span>
                      <pre className="p-2.5 rounded bg-slate-950 border border-slate-800 text-slate-300 font-mono text-[11px] overflow-x-auto">
                        {JSON.stringify(ev.raw_metrics, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
