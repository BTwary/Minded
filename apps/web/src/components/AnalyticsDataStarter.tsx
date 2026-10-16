"use client";

import React, { useState } from "react";
import { Database, UploadCloud } from "lucide-react";
import { uploadDatasetFile } from "../lib/api";
import { beginWorkspaceUpload, finishWorkspaceUpload } from "../lib/workspaceState";

export default function AnalyticsDataStarter({ projectId, disabled, onUploaded, onBusyChange }: {
  projectId: string;
  disabled: boolean;
  onBusyChange: (busy: boolean) => void;
  onUploaded: (datasetId: string, sample: boolean) => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function upload(file?: File) {
    if (busy || disabled || !projectId) return;
    setBusy(true);
    onBusyChange(true);
    setError(null);
    beginWorkspaceUpload();
    const sample = !file;
    try {
      if (!file) {
        const response = await fetch("/samples/sample-sales.csv");
        if (!response.ok) throw new Error("The sample file could not be loaded. Try uploading your own CSV.");
        file = new File([await response.blob()], "sample-sales.csv", { type: "text/csv" });
      }
      const form = new FormData();
      form.append("file", file);
      form.append("project_id", projectId);
      form.append("business_timezone", "UTC");
      form.append("description", sample ? "Synthetic sales data for learning MindEd; not real business records." : `Uploaded ${file.name}`);
      const result = await uploadDatasetFile(form);
      await onUploaded(result.dataset_id, sample);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Upload could not finish. Please try again.");
    } finally {
      setBusy(false);
      onBusyChange(false);
      finishWorkspaceUpload();
    }
  }

  return (
    <section aria-label="Bring your data" className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-white">1. Bring your data</h2>
          <p className="text-xs text-slate-400 mt-1">Upload a CSV, then choose it below. Files use your configured local workspace storage; dates are interpreted in UTC.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <label className={`inline-flex items-center gap-2 rounded-xl bg-cyan-600 px-4 py-2.5 text-xs font-semibold text-white ${busy || disabled || !projectId ? "opacity-50" : "cursor-pointer hover:bg-cyan-500"}`}>
            <UploadCloud className="h-4 w-4" /> Upload CSV
            <input aria-label="Upload CSV" type="file" accept=".csv" className="sr-only" disabled={busy || disabled || !projectId}
              onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void upload(file); }} />
          </label>
          <button type="button" disabled={busy || disabled || !projectId} onClick={() => void upload()}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-600 bg-[var(--surface-2)] px-4 py-2.5 text-xs text-[var(--foreground)] hover:bg-[var(--surface-3)] disabled:opacity-50">
            <Database className="h-4 w-4" /> Try sample sales data
          </button>
        </div>
      </div>
      <p className="text-xs text-slate-500">The sample contains 24 synthetic orders across four regions. You’ll review the question interpretation before any analysis runs.</p>
      {busy && <p role="status" className="text-sm text-cyan-300">Uploading and profiling your dataset…</p>}
      {error && <p role="alert" className="text-sm text-rose-300">{error}</p>}
    </section>
  );
}
