"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  BarChart2,
  CheckCircle2,
  Database,
  FileText,
  FolderOpen,
  HardDrive,
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
  const [showWelcome, setShowWelcome] = useState(false);

  useEffect(() => {
    // Check first-run experience status from local storage
    try {
      const seen = localStorage.getItem("minded_has_seen_welcome");
      if (!seen) {
        setShowWelcome(true);
      }
    } catch {
      // In private mode or restricted context
    }

    (async () => {
      try {
        const projects = await fetchProjects();
        const projectId = projects[0]?.id || "";
        if (projectId) {
          setSummary(await fetchProjectSummary(projectId));
        }
      } catch (error) {
        console.error("Failed to load workspace summary:", error);
        setLoadError(true);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const handleDismissWelcome = () => {
    setShowWelcome(false);
    try {
      localStorage.setItem("minded_has_seen_welcome", "true");
    } catch {
      // Ignore storage errors
    }
  };

  const totalDatasets = Number(summary?.total_datasets || 0);
  const totalRows = Number(summary?.total_rows || 0);
  const avgQuality = Number(summary?.avg_data_quality || 0);
  const summaryUnavailable = loadError || !summary;
  const recentAnalyses = summary?.recent_analyses || [];
  const datasets = summary?.datasets || [];

  const launchQuestion = (value: string) => {
    const trimmed = value.trim();
    window.location.href = trimmed
      ? `/analyst?question=${encodeURIComponent(trimmed)}`
      : "/analyst";
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] space-y-6">
      {/* First-Run Welcome Modal / Overlay */}
      {showWelcome && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg rounded-2xl border border-emerald-500/30 bg-slate-900 p-8 shadow-2xl space-y-6">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
                <ShieldCheck className="h-6 w-6 text-emerald-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white tracking-tight">Welcome to MindEd AA-OS</h2>
                <p className="text-xs text-slate-400">Your autonomous data analyst companion</p>
              </div>
            </div>

            <p className="text-sm text-slate-300 leading-relaxed">
              A local-first autonomous data analyst companion designed for real, trustworthy data investigation.
            </p>

            <div className="space-y-3 rounded-xl border border-slate-800 bg-slate-950/70 p-4 text-xs">
              <div className="flex items-start gap-2.5 text-slate-300">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <span><strong>Your data remains on this computer</strong> by default.</span>
              </div>
              <div className="flex items-start gap-2.5 text-slate-300">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <span><strong>Deterministic analysis</strong> does not require an AI API key.</span>
              </div>
              <div className="flex items-start gap-2.5 text-slate-300">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <span><strong>Offline analysis is fully supported</strong> with local storage and calculation.</span>
              </div>
              <div className="flex items-start gap-2.5 text-slate-300">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <span><strong>Results include complete evidence</strong>, calculations, and falsifiable verification.</span>
              </div>
            </div>

            <button
              onClick={handleDismissWelcome}
              className="w-full py-3 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-sm transition-colors shadow-lg shadow-emerald-950/40"
            >
              Continue
            </button>
          </div>
        </div>
      )}

      {/* Main Desktop Landing Header */}
      <section className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900/90 via-slate-900/60 to-slate-950 p-8 shadow-2xl">
        <div className="max-w-4xl space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="text-xs uppercase tracking-[0.2em] text-emerald-400 font-semibold mb-1">
                MindEd AA-OS
              </div>
              <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
                Your autonomous data analyst companion
              </h1>
            </div>
            <div className="flex items-center gap-2 text-[11px] font-mono">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-emerald-300 font-medium">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                Local engine
              </span>
              <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-300">
                <HardDrive className="h-3 w-3 text-slate-400" />
                Local storage
              </span>
              <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-300">
                <ShieldCheck className="h-3 w-3 text-slate-400" />
                Offline capable
              </span>
            </div>
          </div>

          <p className="text-sm text-slate-300 leading-relaxed max-w-2xl">
            Analyze real data locally. Deterministic calculations. Evidence-backed results. Human-reviewable analysis.
          </p>

          {/* Primary Quick Actions */}
          <div className="flex flex-wrap items-center gap-4 pt-2">
            <Link
              href="/analyst"
              className="inline-flex items-center gap-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 px-6 py-3.5 text-sm font-semibold text-white transition-colors shadow-lg shadow-emerald-950/40"
            >
              <Plus className="h-4 w-4" />
              New Analysis
            </Link>
            <Link
              href="/datasets"
              className="inline-flex items-center gap-2.5 rounded-xl border border-slate-700 bg-slate-800/80 hover:bg-slate-800 px-6 py-3.5 text-sm font-semibold text-slate-200 transition-colors"
            >
              <FolderOpen className="h-4 w-4 text-cyan-400" />
              Open Dataset
            </Link>
          </div>

          {/* Integrated Question Launcher */}
          <div className="pt-4 border-t border-slate-800/80">
            <div className="text-xs text-slate-400 mb-2 font-medium">Ask a question about your data</div>
            <form
              onSubmit={(event) => {
                event.preventDefault();
                launchQuestion(question);
              }}
              className="flex flex-col sm:flex-row gap-3"
            >
              <div className="relative flex-1">
                <input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="e.g. Why did revenue change? or Will sales likely increase next quarter?"
                  className="w-full rounded-xl border border-slate-700 bg-slate-950/90 px-4 py-3.5 text-sm text-white placeholder:text-slate-500 outline-none focus:border-emerald-500/60 focus:ring-1 focus:ring-emerald-500/20"
                />
              </div>
              <button
                type="submit"
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-800 hover:bg-slate-700 px-5 py-3.5 text-xs font-semibold text-slate-200 transition-colors border border-slate-700"
              >
                Investigate <ArrowRight className="h-4 w-4" />
              </button>
            </form>

            <div className="mt-3 flex flex-wrap gap-2">
              {starterQuestions.map((item) => (
                <button
                  key={item}
                  onClick={() => launchQuestion(item)}
                  className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-1.5 text-[11px] text-slate-400 hover:border-emerald-500/30 hover:text-emerald-300 transition-colors text-left"
                >
                  {item}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Metrics & Work Overview */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { label: "Datasets", value: summaryUnavailable ? "—" : totalDatasets.toLocaleString(), sub: "local datasets", icon: Database, tone: "text-emerald-400" },
          { label: "Rows Available", value: summaryUnavailable ? "—" : totalRows.toLocaleString(), sub: "records indexed", icon: Activity, tone: "text-cyan-400" },
          { label: "Data Quality", value: summaryUnavailable ? "—" : totalDatasets ? `${avgQuality}/100` : "—", sub: "integrity profile", icon: ShieldCheck, tone: "text-indigo-400" },
          { label: "Analyses", value: summaryUnavailable ? "—" : recentAnalyses.length.toLocaleString(), sub: "completed records", icon: BarChart2, tone: "text-emerald-400" },
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

      {/* Recent Work: Investigations and Datasets */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Recent Investigations */}
        <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-[0.16em] text-slate-400 font-semibold">Recent Work</div>
              <h2 className="text-sm font-semibold text-white mt-1">Investigations</h2>
            </div>
            <Link href="/analyst" className="inline-flex items-center gap-1.5 text-xs text-emerald-400 hover:text-emerald-300">
              Open Workspace <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
          <div className="divide-y divide-slate-800/80">
            {summaryUnavailable || recentAnalyses.length === 0 ? (
              <div className="p-8 text-center">
                <div className="mx-auto h-10 w-10 rounded-xl border border-dashed border-slate-700 flex items-center justify-center text-slate-500">
                  <Sparkles className="h-4 w-4" />
                </div>
                <p className="mt-3 text-xs text-slate-400">No previous investigations yet.</p>
                <Link href="/analyst" className="inline-flex items-center gap-2 mt-4 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-300">
                  Start an investigation <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            ) : (
              recentAnalyses.slice(0, 5).map((item: any) => (
                <div key={item.id} className="px-5 py-3.5 flex items-center justify-between gap-4 hover:bg-slate-950/30 transition-colors">
                  <div className="min-w-0">
                    <div className="text-xs text-white font-medium truncate">{item.question}</div>
                    <div className="mt-0.5 text-[10px] text-slate-500 font-mono truncate">{item.id}</div>
                  </div>
                  <div className="shrink-0 flex items-center gap-2">
                    <span className="rounded-full border border-slate-700 bg-slate-950 px-2 py-0.5 text-[9px] font-mono text-slate-300">
                      {item.verdict || item.status || "ANALYZED"}
                    </span>
                    {item.status === "COMPLETED" && <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Datasets */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-[0.16em] text-slate-400 font-semibold">Local Storage</div>
              <h2 className="text-sm font-semibold text-white mt-1">Datasets</h2>
            </div>
            <Link href="/datasets" className="text-xs text-emerald-400 hover:text-emerald-300">
              Manage
            </Link>
          </div>
          <div className="p-4 space-y-2">
            {summaryUnavailable || datasets.length === 0 ? (
              <div className="rounded-lg border border-dashed border-slate-700 p-5 text-center">
                <Database className="h-5 w-5 mx-auto text-slate-600" />
                <p className="mt-2 text-xs text-slate-400">No datasets imported yet.</p>
                <Link href="/datasets" className="inline-flex items-center gap-2 mt-3 rounded-lg bg-slate-800 px-3 py-2 text-xs text-slate-200">
                  <Upload className="h-3.5 w-3.5" /> Import CSV or Excel
                </Link>
              </div>
            ) : (
              datasets.slice(0, 5).map((d: any) => (
                <Link href="/datasets" key={d.id} className="block rounded-lg border border-slate-800 bg-slate-950/40 p-3 hover:border-slate-700 transition-colors">
                  <div className="flex items-center gap-2">
                    <Database className="h-3.5 w-3.5 text-emerald-400" />
                    <span className="text-xs text-white font-medium truncate">{d.name}</span>
                  </div>
                  <div className="mt-1 text-[10px] font-mono text-slate-500">{Number(d.row_count || 0).toLocaleString()} rows</div>
                </Link>
              ))
            )}
          </div>
        </div>
      </section>

      {/* Footer Navigation Strip */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 px-5 py-3.5 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-emerald-400" />
          <span>Every analysis includes traceable evidence, formulas, and verification.</span>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/reports" className="hover:text-white transition-colors">Reports</Link>
          <Link href="/settings" className="hover:text-white transition-colors">Settings</Link>
        </div>
      </section>
    </div>
  );
}
