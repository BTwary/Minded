"use client";

import React, { useState } from "react";
import { CheckCircle2, ChevronDown, ChevronRight, Clock, Code2, Database, ShieldCheck, Sparkles } from "lucide-react";
import { AnalysisStep } from "../types";

interface Props {
  steps: AnalysisStep[];
}

export default function StepTracker({ steps }: Props) {
  const [expandedStep, setExpandedStep] = useState<number | null>(null);

  const toggleStep = (stepNum: number) => {
    setExpandedStep(expandedStep === stepNum ? null : stepNum);
  };

  const getActionIcon = (actionType: string) => {
    switch (actionType) {
      case "sql_query":
        return <Database className="w-3.5 h-3.5 text-sky-400" />;
      case "validation":
        return <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />;
      case "forecast":
      case "stats_test":
        return <Sparkles className="w-3.5 h-3.5 text-purple-400" />;
      default:
        return <Code2 className="w-3.5 h-3.5 text-indigo-400" />;
    }
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>Autonomous Analytical Steps</span>
        </h3>
        <span className="text-xs text-slate-400 font-mono">{steps.length} steps executed</span>
      </div>

      <div className="space-y-2">
        {steps.map((step) => {
          const isExpanded = expandedStep === step.step_number;
          return (
            <div
              key={step.step_number}
              className="border border-slate-800/80 rounded-lg bg-slate-950/40 overflow-hidden transition-colors hover:border-slate-700"
            >
              <button
                onClick={() => toggleStep(step.step_number)}
                className="w-full px-3.5 py-2.5 flex items-center justify-between text-left text-xs text-slate-200"
              >
                <div className="flex items-center space-x-3 flex-1 min-w-0 pr-2">
                  <div className="w-5 h-5 rounded-full bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center flex-shrink-0">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                  </div>
                  <div className="flex items-center space-x-2 truncate">
                    {getActionIcon(step.action_type)}
                    <span className="font-medium text-slate-200 truncate">{step.title}</span>
                  </div>
                </div>

                <div className="flex items-center space-x-3 text-slate-400 flex-shrink-0">
                  <span className="text-[11px] font-mono flex items-center space-x-1">
                    <Clock className="w-3 h-3" />
                    <span>{step.duration_ms}ms</span>
                  </span>
                  {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                </div>
              </button>

              {isExpanded && (
                <div className="px-4 py-3 bg-slate-900/90 border-t border-slate-800 text-xs space-y-2.5">
                  <p className="text-slate-300">{step.description}</p>
                  
                  {step.tool_input && (
                    <div>
                      <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block mb-1">
                        Input Parameters
                      </span>
                      <pre className="p-2.5 rounded bg-slate-950 border border-slate-800 text-slate-300 font-mono text-[11px] overflow-x-auto whitespace-pre-wrap">
                        {typeof step.tool_input === "string" ? step.tool_input : JSON.stringify(step.tool_input, null, 2)}
                      </pre>
                    </div>
                  )}

                  {step.tool_output && (
                    <div>
                      <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block mb-1">
                        Deterministic Output
                      </span>
                      <pre className="p-2.5 rounded bg-slate-950 border border-slate-800 text-emerald-300 font-mono text-[11px] overflow-x-auto whitespace-pre-wrap">
                        {typeof step.tool_output === "string" ? step.tool_output : JSON.stringify(step.tool_output, null, 2)}
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
