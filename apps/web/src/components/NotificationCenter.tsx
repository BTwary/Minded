"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  ArrowRight,
  Bell,
  CheckCircle2,
  Clock,
  ExternalLink,
  ShieldAlert,
  Sparkles,
  TrendingDown,
  X,
  Zap,
} from "lucide-react";

interface SentinelAlert {
  id: string;
  title: string;
  severity: "critical" | "warning" | "info";
  description: string;
  metric_name: string;
  deviation_pct: number;
  autonomous_investigation_available: boolean;
  question: string;
  timestamp: string;
}

const SAMPLE_SENTINEL_ALERTS: SentinelAlert[] = [
  {
    id: "ALT-801",
    title: "Metric Anomaly: Net Revenue Contraction Detected",
    severity: "critical",
    description: "Period-over-period revenue dropped by -23.4% in March across sales transaction stream.",
    metric_name: "revenue",
    deviation_pct: -23.4,
    autonomous_investigation_available: true,
    question: "Why did revenue fall in March?",
    timestamp: "2 mins ago",
  },
  {
    id: "ALT-802",
    title: "Distribution Drift: Order Frequency Long-Tail Widening",
    severity: "warning",
    description: "46.3% of active accounts dropped below repeat purchase threshold in customer ledger.",
    metric_name: "order_frequency",
    deviation_pct: -12.1,
    autonomous_investigation_available: true,
    question: "Which customers are likely to churn?",
    timestamp: "18 mins ago",
  },
  {
    id: "ALT-803",
    title: "Predictive Sentinel: Trajectory Model Backtest Validated",
    severity: "info",
    description: "Next month revenue projected at $351,296.45 with 8.32% MAPE error tolerance.",
    metric_name: "revenue_forecast",
    deviation_pct: 0.0,
    autonomous_investigation_available: true,
    question: "Forecast next month's revenue trajectory",
    timestamp: "1 hour ago",
  },
];

export default function NotificationCenter() {
  const [isOpen, setIsOpen] = useState(false);
  const [alerts, setAlerts] = useState<SentinelAlert[]>(SAMPLE_SENTINEL_ALERTS);
  const router = useRouter();

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case "critical":
        return {
          bg: "bg-rose-500/10 text-rose-400 border-rose-500/30",
          icon: <AlertOctagon className="w-3.5 h-3.5 text-rose-400" />,
        };
      case "warning":
        return {
          bg: "bg-amber-500/10 text-amber-400 border-amber-500/30",
          icon: <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />,
        };
      default:
        return {
          bg: "bg-cyan-500/10 text-cyan-400 border-cyan-500/30",
          icon: <Activity className="w-3.5 h-3.5 text-cyan-400" />,
        };
    }
  };

  const handleLaunchInvestigation = (q: string) => {
    setIsOpen(false);
    router.push(`/analyst?q=${encodeURIComponent(q)}`);
  };

  return (
    <div className="relative">
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative p-2 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-white transition-colors"
        title="Autonomous Sentinel Alerts"
      >
        <Bell className="w-4 h-4" />
        {alerts.length > 0 && (
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
        )}
      </button>

      {/* Notification Dropdown Panel */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-96 bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl z-50 overflow-hidden flex flex-col">
          {/* Panel Header */}
          <div className="p-4 border-b border-slate-800 bg-slate-950 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
              <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                Autonomous Sentinel Stream
              </h3>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="text-slate-500 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Alert List */}
          <div className="max-h-96 overflow-y-auto divide-y divide-slate-800/80 p-2 space-y-2">
            {alerts.map((alt) => {
              const badge = getSeverityBadge(alt.severity);
              return (
                <div
                  key={alt.id}
                  className="p-3 rounded-xl bg-slate-950/60 hover:bg-slate-800/50 border border-slate-800/80 transition-colors space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full border flex items-center space-x-1 ${badge.bg}`}
                    >
                      {badge.icon}
                      <span className="uppercase">{alt.severity}</span>
                    </span>
                    <span className="text-[10px] text-slate-500 font-mono">
                      {alt.timestamp}
                    </span>
                  </div>

                  <div>
                    <h4 className="text-xs font-semibold text-white leading-tight">
                      {alt.title}
                    </h4>
                    <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                      {alt.description}
                    </p>
                  </div>

                  {alt.autonomous_investigation_available && (
                    <button
                      onClick={() => handleLaunchInvestigation(alt.question)}
                      className="w-full mt-1 bg-cyan-600/10 hover:bg-cyan-600/20 text-cyan-300 border border-cyan-500/30 rounded-lg p-2 text-[11px] font-semibold flex items-center justify-between transition-colors group"
                    >
                      <div className="flex items-center space-x-1.5">
                        <Zap className="w-3.5 h-3.5 text-cyan-400" />
                        <span>Launch Investigation DAG</span>
                      </div>
                      <ArrowRight className="w-3 h-3 transition-transform group-hover:translate-x-0.5" />
                    </button>
                  )}
                </div>
              );
            })}
          </div>

          {/* Footer */}
          <div className="p-3 border-t border-slate-800 bg-slate-950 flex items-center justify-between text-[11px] text-slate-500">
            <span>Continuous Background Auditing</span>
            <button
              onClick={() => {
                setIsOpen(false);
                router.push("/alerts");
              }}
              className="text-cyan-400 hover:underline font-medium"
            >
              Alert Rules →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
