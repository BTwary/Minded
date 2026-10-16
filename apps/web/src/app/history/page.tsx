"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowUpRight, BookOpen, Clock3, Database, FileDown, Loader2, RefreshCw, ShieldCheck, Trash2 } from "lucide-react";
import {
  AnalysisHistoryRecord,
  AnalysisHistoryResponse,
  clearAnalysisHistory,
  downloadInvestigationExport,
  fetchAnalysisHistory,
} from "../../lib/api";

function formatTimestamp(value: string | null): string {
  if (!value) return "Time not recorded";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function formatJson(value: unknown): string {
  if (value === null || value === undefined) return "No saved result data was recorded.";
  try {
    return JSON.stringify(value, null, 2);
  } catch (error) {
    console.error("Unable to render saved history detail.", error);
    return "Saved detail is unavailable because it could not be rendered.";
  }
}

function statusTone(status: string): string {
  const normalized = status.toUpperCase();
  if (normalized === "COMPLETED" || normalized === "VERIFIED" || normalized === "OBSERVED") {
    return "border-emerald-500/30 bg-emerald-500/10 text-emerald-300";
  }
  if (normalized === "FAILED" || normalized === "REFUSE" || normalized === "CANCELLED") {
    return "border-rose-500/30 bg-rose-500/10 text-rose-300";
  }
  return "border-amber-500/30 bg-amber-500/10 text-amber-200";
}

export default function HistoryPage() {
  const [history, setHistory] = useState<AnalysisHistoryResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clearMessage, setClearMessage] = useState<string | null>(null);
  const [exportingId, setExportingId] = useState<string | null>(null);

  const handleExport = async (id: string, fmt: "pdf" | "docx") => {
    if (!id || exportingId) return;
    setExportingId(id);
    try {
      const blob = await downloadInvestigationExport(id, fmt);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `MindEd_Investigation_${id.slice(0, 8)}.${fmt}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error(`Failed to export ${fmt}:`, err);
      alert(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setExportingId(null);
    }
  };

  const loadHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchAnalysisHistory(100);
      setHistory(result);
      const requestedId = typeof window === "undefined"
        ? ""
        : new URLSearchParams(window.location.search).get("investigation_id") || "";
      if (requestedId) {
        if (result.records.some((record) => record.id === requestedId)) {
          setSelectedId(requestedId);
        } else {
          setError("The requested historical investigation is not available in this local history.");
        }
      }
    } catch (cause) {
      console.error("Unable to load analysis history.", cause);
      setHistory(null);
      setError(cause instanceof Error ? cause.message : "Analysis history could not be loaded.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  const selectedRecord = useMemo(
    () => history?.records.find((record) => record.id === selectedId) || null,
    [history, selectedId]
  );

  const handleClearHistory = async () => {
    if (clearing) return;
    if (!window.confirm("Clear all saved analysis history? This does not remove projects or datasets and cannot be undone.")) {
      return;
    }
    setClearing(true);
    setError(null);
    setClearMessage(null);
    try {
      const result = await clearAnalysisHistory();
      if (result.status !== "success") {
        throw new Error("The server did not confirm that history was cleared.");
      }
      setHistory((current) => current ? { ...current, records: [] } : current);
      setSelectedId(null);
      setClearMessage("Analysis history was cleared. Projects and datasets were preserved.");
    } catch (cause) {
      console.error("Unable to clear analysis history.", cause);
      setError(cause instanceof Error ? cause.message : "Analysis history could not be cleared.");
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-cyan-300">
            <Clock3 className="h-4 w-4" />
            <span>Saved locally</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Analysis History</h1>
          <p className="mt-1 max-w-2xl text-xs text-slate-400">
            Reopen recorded questions, results, source identities, and verification evidence. History is independent of Reset Data.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => void loadHistory()}
            disabled={loading || clearing}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-200 hover:bg-slate-800 disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
          <button
            type="button"
            onClick={() => void handleClearHistory()}
            disabled={clearing || loading || !history?.records.length}
            className="inline-flex items-center gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs font-semibold text-rose-200 hover:bg-rose-500/20 disabled:opacity-50"
          >
            <Trash2 className="h-3.5 w-3.5" />
            {clearing ? "Clearing…" : "Clear History"}
          </button>
        </div>
      </header>

      {error && (
        <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-rose-500/30 bg-rose-950/30 p-4 text-xs text-rose-200">
          <span>History is unavailable: {error}</span>
          <button type="button" onClick={() => void loadHistory()} className="underline underline-offset-2">Retry</button>
        </div>
      )}
      {clearMessage && (
        <div role="status" className="rounded-xl border border-emerald-500/30 bg-emerald-950/30 p-3 text-xs text-emerald-200">{clearMessage}</div>
      )}

      {loading ? (
        <div role="status" className="rounded-2xl border border-slate-800 bg-slate-900/60 p-10 text-center text-sm text-slate-400">
          Loading saved analyses…
        </div>
      ) : history && (
        <>
          {!!history.projects.length && (
            <section className="space-y-3">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Projects and workspaces</h2>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                {history.projects.map((project) => (
                  <div key={project.id} className="flex min-w-0 items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold text-white">{project.name}</div>
                      <div className="mt-1 truncate font-mono text-[10px] text-slate-500">{project.id}</div>
                    </div>
                    <Link
                      href={`/datasets?project_id=${encodeURIComponent(project.id)}`}
                      className="inline-flex shrink-0 items-center gap-1 text-xs text-cyan-300 hover:text-cyan-200"
                    >
                      Open <ArrowUpRight className="h-3.5 w-3.5" />
                    </Link>
                  </div>
                ))}
              </div>
            </section>
          )}

          {!history.records.length ? (
            <div className="rounded-2xl border border-dashed border-slate-700 bg-slate-900/40 p-10 text-center">
              <BookOpen className="mx-auto h-8 w-8 text-slate-500" />
              <h2 className="mt-3 text-sm font-semibold text-white">No saved analyses yet</h2>
              <p className="mt-1 text-xs text-slate-400">Completed, inconclusive, and failed investigation records will appear here.</p>
              <Link href="/analyst" className="mt-4 inline-flex rounded-lg bg-cyan-600 px-3 py-2 text-xs font-semibold text-white hover:bg-cyan-500">Start an analysis</Link>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1.15fr)_minmax(360px,0.85fr)]">
              <section className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/50">
                <div className="border-b border-slate-800 px-4 py-3 text-xs text-slate-400">
                  Showing the latest {history.records.length} saved record{history.records.length === 1 ? "" : "s"}
                </div>
                <div className="divide-y divide-slate-800/80">
                  {history.records.map((record: AnalysisHistoryRecord) => (
                    <article key={`${record.record_type}:${record.id}`} className={`p-4 ${selectedId === record.id ? "bg-cyan-950/30" : "hover:bg-slate-950/40"}`}>
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="min-w-0 flex-1">
                          <div className="text-[10px] text-slate-500">{formatTimestamp(record.created_at)} · {record.project_name}</div>
                          <h3 className="mt-1 break-words text-sm font-medium text-white">{record.question || record.project_name || "Untitled analysis"}</h3>
                          <div className="mt-2 flex flex-wrap items-center gap-2">
                            <span className={`rounded-full border px-2 py-0.5 text-[9px] font-mono ${statusTone(record.status)}`}>{record.status}</span>
                            {!!record.verdict_type && (
                              <span className={`rounded-full border px-2 py-0.5 text-[9px] font-mono ${statusTone(String(record.verdict_type))}`}>
                                Result: {String(record.verdict_type)}
                              </span>
                            )}
                            {record.claim_status === "QUALIFIED_ANSWER" && (
                              <span className={`rounded-full border px-2 py-0.5 text-[9px] font-mono ${statusTone("QUALIFIED")}`}>Qualified claim</span>
                            )}
                            <span className={`rounded-full border px-2 py-0.5 text-[9px] font-mono ${statusTone(record.verification_status)}`}>Verification: {record.verification_status}</span>
                          </div>
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            {record.source_datasets.length ? record.source_datasets.map((source) => (
                              <span key={`${source.dataset_id || source.name}:${source.dataset_version}`} className="inline-flex items-center gap-1 rounded-md border border-slate-800 bg-slate-950/70 px-2 py-1 text-[10px] text-slate-400">
                                <Database className="h-3 w-3" />
                                {source.name}{source.dataset_version ? ` v${source.dataset_version}` : ""}
                              </span>
                            )) : <span className="text-[10px] text-slate-500">Source datasets not recorded</span>}
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => setSelectedId(record.id)}
                          className="shrink-0 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-xs font-semibold text-cyan-200 hover:bg-cyan-500/20"
                        >
                          View result
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              </section>

              <section className="min-w-0 rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
                {selectedRecord ? (
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-cyan-300">Original saved result</div>
                        <h2 className="mt-1 break-words text-base font-bold text-white">{selectedRecord.question || "Untitled analysis"}</h2>
                        <p className="mt-1 text-[10px] text-slate-500">{formatTimestamp(selectedRecord.created_at)} · {selectedRecord.project_name}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => void handleExport(selectedRecord.id, "pdf")}
                          disabled={exportingId === selectedRecord.id}
                          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:border-amber-400 hover:text-amber-300 disabled:opacity-50 transition-colors shadow-sm"
                          title="Export saved analysis as publication-ready PDF"
                        >
                          {exportingId === selectedRecord.id ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-400" />
                          ) : (
                            <FileDown className="h-3.5 w-3.5 text-amber-400" />
                          )}
                          <span>PDF</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => void handleExport(selectedRecord.id, "docx")}
                          disabled={exportingId === selectedRecord.id}
                          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:border-sky-400 hover:text-sky-300 disabled:opacity-50 transition-colors shadow-sm"
                          title="Export saved analysis as editable Microsoft Word document"
                        >
                          {exportingId === selectedRecord.id ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin text-sky-400" />
                          ) : (
                            <FileDown className="h-3.5 w-3.5 text-sky-400" />
                          )}
                          <span>DOCX</span>
                        </button>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <span className={`rounded-full border px-2 py-1 text-[10px] font-mono ${statusTone(selectedRecord.status)}`}>Status: {selectedRecord.status}</span>
                      <span className={`rounded-full border px-2 py-1 text-[10px] font-mono ${statusTone(selectedRecord.verification_status)}`}>
                        Verification: {selectedRecord.verification_status}
                      </span>
                      <span className={`rounded-full border px-2 py-1 text-[10px] font-mono ${statusTone(selectedRecord.claim_status)}`}>
                        Claim: {selectedRecord.claim_status}
                      </span>
                    </div>
                    {selectedRecord.headline && (
                      <div className="rounded-xl border border-cyan-500/20 bg-cyan-950/20 p-4">
                        <div className="text-[10px] uppercase tracking-wider text-cyan-300">Saved headline</div>
                        <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-slate-100">{selectedRecord.headline}</p>
                      </div>
                    )}
                    {selectedRecord.source_datasets.map((source) => (
                      <div key={`source-${source.dataset_id || source.name}`} className={`rounded-xl border p-3 text-xs ${source.available ? "border-slate-800 bg-slate-950/50 text-slate-300" : "border-amber-500/30 bg-amber-950/20 text-amber-100"}`}>
                        {source.available
                          ? `Source available: ${source.name}, version ${source.dataset_version ?? "not recorded"}.`
                          : `Source unavailable or changed: ${source.name}, recorded version ${source.dataset_version ?? "not recorded"}. This saved result is view-only and cannot be rerun against another dataset.`}
                        {source.fingerprint && <div className="mt-1 break-all font-mono text-[10px] opacity-75">Fingerprint: {source.fingerprint}</div>}
                      </div>
                    ))}
                    {!selectedRecord.source_datasets.length && (
                      <div className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-3 text-xs text-amber-100">
                        No source identity was recorded for this result. It remains view-only and will not be rerun against current data.
                      </div>
                    )}

                    {Array.isArray(selectedRecord.evidence) && selectedRecord.evidence.length > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-xs font-semibold text-slate-200"><ShieldCheck className="h-3.5 w-3.5 text-emerald-300" />Recorded evidence</div>
                        {selectedRecord.evidence.map((evidence: any, index: number) => (
                          <details key={evidence.id || index} className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                            <summary className="cursor-pointer text-xs text-slate-200">{evidence.statement || evidence.id || `Evidence ${index + 1}`} · {evidence.validation_status || "status not recorded"}</summary>
                            <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap break-words text-[10px] leading-relaxed text-slate-400">{formatJson(evidence)}</pre>
                          </details>
                        ))}
                      </div>
                    )}
                    <details className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                      <summary className="cursor-pointer text-xs font-semibold text-slate-300">Recorded result data and evidence references</summary>
                      <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap break-words text-[10px] leading-relaxed text-slate-400">
                        {formatJson(selectedRecord.analyst_result || selectedRecord.saved_result || {
                          direct_answer: selectedRecord.direct_answer,
                          evidence_ids: selectedRecord.evidence_ids,
                        })}
                      </pre>
                    </details>
                    <p className="text-[10px] leading-relaxed text-slate-500">
                      This is the original persisted output and verification status. Opening it does not rerun analysis or update it using current data.
                    </p>
                  </div>
                ) : (
                  <div className="flex min-h-56 flex-col items-center justify-center text-center">
                    <BookOpen className="h-7 w-7 text-slate-600" />
                    <p className="mt-3 text-sm font-semibold text-white">Select a saved analysis</p>
                    <p className="mt-1 text-xs text-slate-400">Its recorded result, source identity, and evidence will appear here.</p>
                  </div>
                )}
              </section>
            </div>
          )}
        </>
      )}
    </div>
  );
}
