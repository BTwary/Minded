"use client";

import React from "react";
import {
  Activity,
  AlertOctagon,
  ArrowRight,
  BarChart3,
  Bot,
  Compass,
  Cpu,
  LineChart,
  PieChart,
  Search,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  Users,
  Zap,
} from "lucide-react";

interface Props {
  onSelectQuestion: (question: string) => void;
  isLoading: boolean;
}

const QUICK_ACTIONS = [
  {
    id: "root-cause",
    title: "Root Cause Diagnostic",
    question: "Why did revenue fall in March?",
    description: "Decomposes period-over-period variance across dimensions, pricing shifts, and customer segments.",
    icon: <TrendingDown className="w-5 h-5 text-rose-400" />,
    badge: "Diagnostic DAG",
    color: "from-rose-500/10 to-rose-600/5 hover:border-rose-500/40",
  },
  {
    id: "correlation",
    title: "Correlation Driver Analysis",
    question: "What are the correlation drivers of revenue?",
    description: "Calculates Pearson correlation matrices and statistical significance (p < 0.05) across numerical measures.",
    icon: <Activity className="w-5 h-5 text-cyan-400" />,
    badge: "Statistical",
    color: "from-cyan-500/10 to-cyan-600/5 hover:border-cyan-500/40",
  },
  {
    id: "forecast",
    title: "Trajectory Forecasting",
    question: "Forecast next month's revenue trajectory",
    description: "Fits exponential smoothing and ARIMA models with backtest residual validation.",
    icon: <LineChart className="w-5 h-5 text-emerald-400" />,
    badge: "Predictive",
    color: "from-emerald-500/10 to-emerald-600/5 hover:border-emerald-500/40",
  },
  {
    id: "rfm-churn",
    title: "Customer RFM & Churn Risk",
    question: "Which customers are likely to churn?",
    description: "Segments customer frequency distributions and isolates long-tail at-risk accounts.",
    icon: <Users className="w-5 h-5 text-indigo-400" />,
    badge: "Cohort",
    color: "from-indigo-500/10 to-indigo-600/5 hover:border-indigo-500/40",
  },
  {
    id: "catalog",
    title: "Product Catalog Performance",
    question: "Which products are performing badly?",
    description: "Aggregates revenue volume contribution and isolates bottom quartile underperformers.",
    icon: <BarChart3 className="w-5 h-5 text-amber-400" />,
    badge: "Comparative",
    color: "from-amber-500/10 to-amber-600/5 hover:border-amber-500/40",
  },
  {
    id: "discovery",
    title: "Autonomous Discovery & Audit",
    question: "Explore this dataset and tell me what deserves investigation",
    description: "Discovers primary entities, time grains, additivity classes, and anomalies without predefined templates.",
    icon: <Sparkles className="w-5 h-5 text-purple-400" />,
    badge: "Discovery",
    color: "from-purple-500/10 to-purple-600/5 hover:border-purple-500/40",
  },
];

export default function QuickAnalysisCenter({ onSelectQuestion, isLoading }: Props) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
          <Zap className="w-4 h-4 text-cyan-400" />
          <span>Quick Autonomous Investigations</span>
        </div>
        <span className="text-[11px] text-slate-500">
          Launches active Investigation DAGs with deterministic verification
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
        {QUICK_ACTIONS.map((action) => (
          <button
            key={action.id}
            disabled={isLoading}
            onClick={() => onSelectQuestion(action.question)}
            className={`text-left p-4 rounded-xl border border-slate-800 bg-gradient-to-br ${action.color} bg-slate-900/60 transition-all duration-200 hover:shadow-lg hover:shadow-black/40 hover:-translate-y-0.5 group disabled:opacity-50 disabled:cursor-not-allowed flex flex-col justify-between`}
          >
            <div>
              <div className="flex items-center justify-between mb-2.5">
                <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                  {action.icon}
                </div>
                <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-slate-950 text-slate-400 border border-slate-800">
                  {action.badge}
                </span>
              </div>

              <h4 className="text-xs font-bold text-white group-hover:text-cyan-300 transition-colors">
                {action.title}
              </h4>

              <p className="text-[11px] text-slate-400 mt-1.5 line-clamp-2 leading-relaxed">
                {action.description}
              </p>
            </div>

            <div className="mt-3.5 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400 group-hover:text-cyan-400 font-medium">
              <span className="truncate mr-2 font-mono text-[10px] text-slate-400">"{action.question}"</span>
              <ArrowRight className="w-3.5 h-3.5 flex-shrink-0 transition-transform group-hover:translate-x-1" />
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
