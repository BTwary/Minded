"use client";

import React, { useState } from "react";
import { Database, UploadCloud } from "lucide-react";
import { uploadDatasetFile } from "../lib/api";
import { beginWorkspaceUpload, finishWorkspaceUpload } from "../lib/workspaceState";

export default function AnalyticsDataStarter({ projectId, disabled, onUploaded, onBusyChange }: {
  projectId: string;
  disabled: boolean;
  onBusyChange: (busy: boolean) => void;
  onUploaded: (datasetId: string | string[], sample: boolean) => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const [progressMsg, setProgressMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function upload(selectedFiles?: FileList | File[] | File | null, sample = false) {
    if (busy || disabled || !projectId) return;
    setBusy(true);
    onBusyChange(true);
    setError(null);
    beginWorkspaceUpload();
    try {
      if (sample || !selectedFiles) {
        setProgressMsg("Loading sample sales data...");
        const response = await fetch("/samples/sample-sales.csv");
        if (!response.ok) throw new Error("The sample file could not be loaded. Try uploading your own CSV.");
        const file = new File([await response.blob()], "sample-sales.csv", { type: "text/csv" });
        const form = new FormData();
        form.append("file", file);
        form.append("project_id", projectId);
        form.append("business_timezone", "UTC");
        form.append("description", "Synthetic sales data for learning MindEd; not real business records.");
        const result = await uploadDatasetFile(form);
        await onUploaded(result.dataset_id, true);
      } else {
        const files = selectedFiles instanceof File ? [selectedFiles] : Array.from(selectedFiles);
        if (!files.length) return;
        const uploadedIds: string[] = [];
        const errors: string[] = [];
        for (let i = 0; i < files.length; i++) {
          const file = files[i];
          setProgressMsg(`Uploading ${i + 1}/${files.length}: ${file.name}...`);
          const form = new FormData();
          form.append("file", file);
          form.append("project_id", projectId);
          form.append("business_timezone", "UTC");
          form.append("description", `Uploaded ${file.name}`);
          try {
            const result = await uploadDatasetFile(form);
            uploadedIds.push(result.dataset_id);
          } catch (err: any) {
            errors.push(`${file.name}: ${err?.message || "Failed"}`);
          }
        }
        if (errors.length > 0) {
          setError(`Some uploads failed: ${errors.join(", ")}`);
        }
        if (uploadedIds.length > 0) {
          await onUploaded(uploadedIds, false);
        }
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Upload could not finish. Please try again.");
    } finally {
      setBusy(false);
      setProgressMsg(null);
      onBusyChange(false);
      finishWorkspaceUpload();
    }
  }

  return (
    <section aria-label="Bring your data" className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-white">1. Bring your data</h2>
          <p className="text-xs text-slate-400 mt-1">Upload CSV files, then choose them below. Files use your configured local workspace storage; dates are interpreted in UTC.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <label className={`inline-flex items-center gap-2 rounded-xl bg-cyan-600 px-4 py-2.5 text-xs font-semibold text-white ${busy || disabled || !projectId ? "opacity-50" : "cursor-pointer hover:bg-cyan-500"}`}>
            <UploadCloud className="h-4 w-4" /> Upload CSV(s)
            <input aria-label="Upload CSV" type="file" accept=".csv" multiple className="sr-only" disabled={busy || disabled || !projectId}
              onChange={(event) => { const files = event.target.files; event.target.value = ""; if (files && files.length > 0) void upload(files, false); }} />
          </label>
          <button type="button" disabled={busy || disabled || !projectId} onClick={() => void upload(null, true)}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-600 bg-[var(--surface-2)] px-4 py-2.5 text-xs text-[var(--foreground)] hover:bg-[var(--surface-3)] disabled:opacity-50">
            <Database className="h-4 w-4" /> Try sample sales data
          </button>
        </div>
      </div>
      <p className="text-xs text-slate-500">You can select multiple CSV files at once. Each will be profiled and added to your workspace catalog.</p>
      {busy && <p role="status" className="text-sm text-cyan-300">{progressMsg || "Uploading and profiling your dataset…"}</p>}
      {error && <p role="alert" className="text-sm text-rose-300">{error}</p>}
    </section>
  );
}
