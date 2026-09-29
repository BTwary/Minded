"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Bell,
  Bot,
  CheckCircle2,
  Database,
  FileText,
  HardDrive,
  MessageSquare,
  Plus,
  ShieldCheck,
  Sparkles,
  Upload,
} from "lucide-react";
import { fetchProjects, fetchProjectSummary } from "../lib/api";

const starterQuestions = [
  "Why did sales change this quarter?",
  "Will sales likely increase next quarter?",
  "Which customers or segments are driving the change?",
  "Is this difference statistically meaningful?",
];

export default function HomePage() {
  const [summary, setSummary] = useState<any>(null);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const projects = await fetchProjects();
        const projectId = projects[0]?.id || "";
        setSummary(await fetchProjectSummary(projectId));
      } catch (error) {
        console.error("Failed to load workspace summary:", error);
        setLoadError(true);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const totalDatasets = Number(summary?.total_datasets || 0);
  const totalRows = Number(summary?.total_rows || 0);
  const avgQuality = Number(summary?.avg_data_quality || 0);
  const summaryUnavailable = loadError || !summary;
  const recentAnalyses = summary?.recent_analyses || [];
  const datasets = summary?.datasets || [];
  const recentAlerts = summary?.recent_alerts || [];

  const launchQuestion = (value: string) => {
    const trimmed = value.trim();
    window.location.href = trimmed
      ? `/analyst?question=${encodeURIComponent(trimmed)}`
      : "/analyst";
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] space-y-6">
      <section className="rounded-2xl border border-slate-800 bg-slate-900/70 shadow-2xl overflow-hidden">
        <div className="px-6 py-5 border-b border-slate-800 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
              <Bot className="h-5 w-5 text-cyan-400" />
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-[0.16em] text-cyan-400 font-semibold">MindEd AA-OS</div>
              <h1 className="text-lg font-semibold text-white">Analytical Workspace</h1>
            </div>
          </div>
          <div className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-wide">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/25 bg-emerald-500/10 px-2.5 py-1.5 text-emerald-300">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> Local
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-950/70 px-2.5 py-1.5 text-slate-300">
              <HardDrive className="h-3 w-3" /> User-owned data
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-950/70 px-2.5 py-1.5 text-slate-300">
              <ShieldCheck className="h-3 w-3" /> Traceable
            </span>
          </div>
        </div>

        <div className="p-6 md:p-8">
          <div className="max-w-3xl">
            <div className="text-xs text-slate-400 mb-2">Ask a question about your data</div>
            <form
              onSubmit={(event) => {
                event.preventDefault();
                launchQuestion(question);
              }}
              className="flex flex-col md:flex-row gap-3"
            >
              <div className="relative flex-1">
                <MessageSquare className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 text-cyan-400" />
                <input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="e.g. Will sales likely increase next quarter?"
                  className="w-full rounded-xl border border-slate-700 bg-slate-950/90 pl-11 pr-4 py-3.5 text-sm text-white placeholder:text-slate-600 outline-none focus:border-cyan-500/60 focus:ring-1 focus:ring-cyan-500/20"
                />
              </div>
              <button
                type="submit"
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 px-5 py-3.5 text-xs font-semibold text-white transition-colors"
              >
                Investigate <ArrowRight className="h-4 w-4" />
              </button>
            </form>

            <div className="mt-4 flex flex-wrap gap-2">
              {starterQuestions.map((item) => (
                <button
                  key={item}
                  onClick={() => launchQuestion(item)}
                  className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-[11px] text-slate-400 hover:border-cyan-500/30 hover:text-cyan-300 transition-colors text-left"
                >
                  {item}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-2 xl:grid-cols-4 gap-3">
        {[
          { label: "Datasets", value: summaryUnavailable ? "Unavailable" : totalDatasets.toLocaleString(), sub: "in this workspace", icon: Database, tone: "text-cyan-400" },
          { label: "Rows available", value: summaryUnavailable ? "Unavailable" : totalRows.toLocaleString(), sub: "source rows indexed", icon: Activity, tone: "text-indigo-400" },
          { label: "Data quality", value: summaryUnavailable ? "Unavailable" : totalDatasets ? `${avgQuality}/100` : "—", sub: "automated profiling", icon: ShieldCheck, tone: "text-emerald-400" },
          { label: "Alerts", value: summaryUnavailable ? "Unavailable" : recentAlerts.length.toLocaleString(), sub: recentAlerts.length ? "need attention" : "no active alerts", icon: Bell, tone: recentAlerts.length ? "text-amber-400" : "text-slate-500" },
        ].map(({ label, value, sub, icon: Icon, tone }) => (
          <div key={label} className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span>{label}</span>
              <Icon className={`h-4 w-4 ${tone}`} />
            </div>
            <div className="mt-2 text-xl font-semibold font-mono text-white">{loading ? "…" : value}</div>
            <div className="mt-1 text-[10px] text-slate-500">{sub}</div>
          </div>
        ))}
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2 rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-[0.16em] text-slate-500 font-semibold">Workspace</div>
              <h2 className="text-sm font-semibold text-white mt-1">Recent investigations</h2>
            </div>
            <Link href="/analyst" className="inline-flex items-center gap-1.5 text-[11px] text-cyan-400 hover:text-cyan-300">
              Open analyst <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
          <div className="divide-y divide-slate-800/80">
            {summaryUnavailable ? (
              <div className="p-8 text-center">
                <div className="mx-auto h-10 w-10 rounded-xl border border-dashed border-amber-500/30 flex items-center justify-center text-amber-400">
                  <ShieldCheck className="h-4 w-4" />
                </div>
                <p className="mt-3 text-xs text-slate-400">Workspace summary is unavailable. No zero values are being substituted.</p>
              </div>
            ) : recentAnalyses.length === 0 ? (
              <div className="p-8 text-center">
                <div className="mx-auto h-10 w-10 rounded-xl border border-dashed border-slate-700 flex items-center justify-center text-slate-500">
                  <Sparkles className="h-4 w-4" />
                </div>
                <p className="mt-3 text-xs text-slate-400">No investigations yet.</p>
                <Link href="/analyst" className="inline-flex items-center gap-2 mt-4 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-[11px] text-cyan-300">
                  Start an investigation <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            ) : (
              recentAnalyses.slice(0, 6).map((item: any) => (
                <div key={item.id} className="px-5 py-4 flex items-center justify-between gap-4 hover:bg-slate-950/30 transition-colors">
                  <div className="min-w-0">
                    <div className="text-xs text-white font-medium truncate">{item.question}</div>
                    <div className="mt-1 text-[10px] text-slate-500 font-mono truncate">{item.id}</div>
                  </div>
                  <div className="shrink-0 flex items-center gap-2">
                    <span className="rounded-full border border-slate-700 bg-slate-950 px-2 py-1 text-[9px] font-mono text-slate-300">
                      {item.verdict || item.status || "ANALYSIS"}
                    </span>
                    {item.status === "COMPLETED" && <CheckCircle2 className="h-4 w-4 text-emerald-400" />}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-[0.16em] text-slate-500 font-semibold">Data</div>
              <h2 className="text-sm font-semibold text-white mt-1">Active datasets</h2>
            </div>
            <Link href="/datasets" className="text-[11px] text-cyan-400 hover:text-cyan-300">Manage</Link>
          </div>
          <div className="p-4 space-y-2">
            {summaryUnavailable ? (
              <div className="rounded-lg border border-dashed border-amber-500/30 p-5 text-center">
                <ShieldCheck className="h-5 w-5 mx-auto text-amber-400" />
                <p className="mt-2 text-[11px] text-slate-500">Dataset information is unavailable because the workspace API did not return a verified response.</p>
              </div>
            ) : datasets.length === 0 ? (
              <div className="rounded-lg border border-dashed border-slate-700 p-5 text-center">
                <Database className="h-5 w-5 mx-auto text-slate-600" />
                <p className="mt-2 text-[11px] text-slate-500">Bring your first dataset into the workspace.</p>
                <Link href="/datasets" className="inline-flex items-center gap-2 mt-3 rounded-lg bg-slate-800 px-3 py-2 text-[11px] text-slate-200">
                  <Upload className="h-3.5 w-3.5" /> Add data
                </Link>
              </div>
            ) : (
              datasets.slice(0, 5).map((d: any) => (
                <Link href="/datasets" key={d.id} className="block rounded-lg border border-slate-800 bg-slate-950/40 p-3 hover:border-slate-700 transition-colors">
                  <div className="flex items-center gap-2">
                    <Database className="h-3.5 w-3.5 text-cyan-400" />
                    <span className="text-xs text-white font-medium truncate">{d.name}</span>
                  </div>
                  <div className="mt-1 text-[10px] font-mono text-slate-500">{Number(d.row_count || 0).toLocaleString()} rows · v{d.current_version}</div>
                </Link>
              ))
            )}
          </div>
        </div>
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-900/40 px-5 py-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <FileText className="h-4 w-4 text-slate-500" />
          <div>
            <div className="text-xs text-white">Every important result can be inspected</div>
            <div className="text-[10px] text-slate-500">Evidence, formulas, execution, verification and provenance remain part of the investigation.</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/reports" className="text-[11px] text-slate-400 hover:text-white">Reports</Link>
          <Link href="/settings" className="text-[11px] text-slate-400 hover:text-white">Settings</Link>
        </div>
      </section>
    </div>
  );
}
