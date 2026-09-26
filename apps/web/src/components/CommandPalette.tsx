"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  AlertOctagon,
  BarChart3,
  Bot,
  Compass,
  Database,
  FileSpreadsheet,
  FileText,
  LineChart,
  Network,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Users,
  X,
  Zap,
} from "lucide-react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onRunQuickAnalysis?: (question: string) => void;
}

export default function CommandPalette({ isOpen, onClose, onRunQuickAnalysis }: Props) {
  const [query, setQuery] = useState("");
  const router = useRouter();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (isOpen) onClose();
      }
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const COMMANDS = [
    {
      category: "Autonomous Investigations",
      items: [
        {
          title: "Why did revenue fall in March?",
          subtitle: "Root Cause Variance Decomposition & Price Sensitivity",
          icon: <Zap className="w-4 h-4 text-rose-400" />,
          action: () => {
            if (onRunQuickAnalysis) onRunQuickAnalysis("Why did revenue fall in March?");
            else router.push("/analyst?q=Why+did+revenue+fall+in+March%3F");
            onClose();
          },
        },
        {
          title: "Forecast next month's revenue trajectory",
          subtitle: "Holt Exponential Smoothing & Time Series Trend",
          icon: <LineChart className="w-4 h-4 text-emerald-400" />,
          action: () => {
            if (onRunQuickAnalysis) onRunQuickAnalysis("Forecast next month's revenue trajectory");
            else router.push("/analyst?q=Forecast+next+month%27s+revenue+trajectory");
            onClose();
          },
        },
        {
          title: "What are the correlation drivers of revenue?",
          subtitle: "Pearson Correlation Matrix & Significance Testing",
          icon: <Activity className="w-4 h-4 text-cyan-400" />,
          action: () => {
            if (onRunQuickAnalysis) onRunQuickAnalysis("What are the correlation drivers of revenue?");
            else router.push("/analyst?q=What+are+the+correlation+drivers+of+revenue%3F");
            onClose();
          },
        },
        {
          title: "Which customers are likely to churn?",
          subtitle: "Customer RFM Cohort Segmentation",
          icon: <Users className="w-4 h-4 text-indigo-400" />,
          action: () => {
            if (onRunQuickAnalysis) onRunQuickAnalysis("Which customers are likely to churn?");
            else router.push("/analyst?q=Which+customers+are+likely+to+churn%3F");
            onClose();
          },
        },
        {
          title: "Explore dataset and discover what deserves investigation",
          subtitle: "Unsupervised Semantic Model Profiling",
          icon: <Sparkles className="w-4 h-4 text-purple-400" />,
          action: () => {
            if (onRunQuickAnalysis) onRunQuickAnalysis("Explore this dataset and tell me what deserves investigation");
            else router.push("/analyst?q=Explore+this+dataset+and+tell+me+what+deserves+investigation");
            onClose();
          },
        },
      ],
    },
    {
      category: "Data & Navigation",
      items: [
        {
          title: "Data Lake & Column Profiler",
          subtitle: "Explore raw tables, null distributions, and quality scores",
          icon: <Database className="w-4 h-4 text-indigo-400" />,
          action: () => {
            router.push("/datasets");
            onClose();
          },
        },
        {
          title: "AI Analyst Studio",
          subtitle: "Interactive Investigation Graph and Evidence Explorer",
          icon: <Bot className="w-4 h-4 text-cyan-400" />,
          action: () => {
            router.push("/analyst");
            onClose();
          },
        },
        {
          title: "Real-time Dashboards",
          subtitle: "Verified metrics with click-to-verify evidence links",
          icon: <BarChart3 className="w-4 h-4 text-emerald-400" />,
          action: () => {
            router.push("/dashboards");
            onClose();
          },
        },
        {
          title: "Reports & Executive Briefings",
          subtitle: "Audited multi-format investigation exports",
          icon: <FileText className="w-4 h-4 text-amber-400" />,
          action: () => {
            router.push("/reports");
            onClose();
          },
        },
        {
          title: "Autonomous Sentinel & Alerts",
          subtitle: "Background metric anomaly detection rules",
          icon: <AlertOctagon className="w-4 h-4 text-rose-400" />,
          action: () => {
            router.push("/alerts");
            onClose();
          },
        },
      ],
    },
  ];

  const filteredCategories = COMMANDS.map((cat) => ({
    ...cat,
    items: cat.items.filter(
      (item) =>
        item.title.toLowerCase().includes(query.toLowerCase()) ||
        item.subtitle.toLowerCase().includes(query.toLowerCase())
    ),
  })).filter((cat) => cat.items.length > 0);

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-24 p-4 bg-slate-950/80 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Search Input Bar */}
        <div className="flex items-center px-4 py-3.5 border-b border-slate-800 bg-slate-950">
          <Search className="w-5 h-5 text-slate-400 mr-3" />
          <input
            type="text"
            autoFocus
            placeholder="Type an analytical question or navigate (e.g. 'revenue', 'forecast', 'datasets')..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full bg-transparent text-sm text-white placeholder-slate-500 focus:outline-none"
          />
          <kbd className="hidden sm:inline-block text-[10px] font-mono bg-slate-800 text-slate-400 px-2 py-0.5 rounded border border-slate-700">
            ESC
          </kbd>
        </div>

        {/* Command Results */}
        <div className="max-h-[60vh] overflow-y-auto p-3 space-y-4">
          {filteredCategories.length > 0 ? (
            filteredCategories.map((cat) => (
              <div key={cat.category} className="space-y-1.5">
                <div className="text-[10px] font-mono uppercase text-slate-500 font-semibold px-2">
                  {cat.category}
                </div>
                {cat.items.map((item, idx) => (
                  <button
                    key={idx}
                    onClick={item.action}
                    className="w-full flex items-center justify-between p-2.5 rounded-xl hover:bg-slate-800 text-left transition-colors group"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="p-2 rounded-lg bg-slate-950 border border-slate-800 group-hover:border-slate-700">
                        {item.icon}
                      </div>
                      <div>
                        <div className="text-xs font-semibold text-slate-200 group-hover:text-white">
                          {item.title}
                        </div>
                        <div className="text-[11px] text-slate-400">{item.subtitle}</div>
                      </div>
                    </div>
                    <span className="text-[10px] font-mono text-slate-600 group-hover:text-slate-400">
                      ↵ Select
                    </span>
                  </button>
                ))}
              </div>
            ))
          ) : (
            <div className="py-8 text-center text-xs text-slate-500">
              No matching commands or investigations found for "{query}".
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-slate-800 bg-slate-950 flex items-center justify-between text-[11px] text-slate-500 font-mono">
          <span>AA-OS Command Engine</span>
          <div className="flex items-center space-x-3">
            <span>↑↓ Navigate</span>
            <span>↵ Execute</span>
          </div>
        </div>
      </div>
    </div>
  );
}
