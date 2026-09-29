"use client";

import React, { useEffect, useState } from "react";
import { AlertCircle, Bell, CheckCircle2, Clock, Plus, ShieldCheck } from "lucide-react";
import { fetchAlertEvents, fetchAlertRules } from "../../lib/api";
import { AlertEvent, AlertRule } from "../../types";

export default function AlertsPage() {
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [events, setEvents] = useState<AlertEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [r, e] = await Promise.allSettled([fetchAlertRules(), fetchAlertEvents()]);
        if (r.status === "fulfilled") setRules(r.value);
        if (e.status === "fulfilled") setEvents(e.value);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const defaultRules = [
    {
      name: "Daily Revenue Deviation Monitor",
      metric_name: "Net Revenue",
      threshold_pct_change: 15.0,
      frequency: "daily",
      severity: "critical",
      is_active: true,
    },
    {
      name: "Product Margin Anomaly Watcher",
      metric_name: "Profit Margin",
      threshold_pct_change: 10.0,
      frequency: "hourly",
      severity: "high",
      is_active: true,
    },
  ];

  const displayRules = rules.length > 0 ? rules : defaultRules;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold text-cyan-400 mb-1">
            <Bell className="w-4 h-4" />
            <span>AUTONOMOUS ANOMALY MONITORING</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Alerts &amp; Monitoring</h1>
          <p className="text-xs text-slate-400 mt-1">
            Configure automated anomaly detectors that continually scan metrics and alert on deviations.
          </p>
        </div>
      </div>

      {/* Rules List */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-4">
        <h2 className="text-sm font-semibold text-white">Active Monitoring Rules</h2>
        <div className="space-y-3">
          {displayRules.map((rule: any, idx) => (
            <div
              key={idx}
              className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 flex items-center justify-between"
            >
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-sm text-white">{rule.name}</span>
                  <span className="text-[10px] font-mono uppercase bg-rose-500/10 text-rose-400 border border-rose-500/30 px-2 py-0.5 rounded">
                    {rule.severity}
                  </span>
                </div>
                <div className="flex items-center space-x-3 text-xs text-slate-400 font-mono">
                  <span>Metric: {rule.metric_name}</span>
                  <span>&bull;</span>
                  <span>Threshold: &plusmn;{rule.threshold_pct_change}%</span>
                  <span>&bull;</span>
                  <span>Frequency: {rule.frequency}</span>
                </div>
              </div>

              <span className="inline-flex items-center space-x-1.5 text-xs text-emerald-400 font-medium">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span>Monitoring Active</span>
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
