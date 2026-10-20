"use client";

import React, { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Bot,
  CheckCircle2,
  Database,
  Download,
  Eye,
  FileSpreadsheet,
  Filter,
  Layers,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  Table,
  UploadCloud,
  Zap,
} from "lucide-react";
import { fetchDatasets, fetchDatasetDetails, fetchDatasetData, uploadDatasetFile, fetchProjects } from "../../lib/api";
import { Dataset, ColumnProfile, Project } from "../../types";
import QualityScoreBadge from "../../components/QualityScoreBadge";
import {
  beginWorkspaceUpload,
  finishWorkspaceUpload,
  getWorkspaceRequestGeneration,
  workspaceResetEvents,
} from "../../lib/workspaceState";

export default function DatasetsPage() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [tableData, setTableData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [workspaceResetting, setWorkspaceResetting] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"profile" | "data" | "semantic">("profile");
  const [searchQuery, setSearchQuery] = useState("");
  const [loadError, setLoadError] = useState(false);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const requestGenerationRef = useRef(0);
  const selectionRequestRef = useRef(0);
  const selectedProjectIdRef = useRef(selectedProjectId);

  const loadAll = async (requestedProjectId?: string) => {
    const requestId = ++requestGenerationRef.current;
    const workspaceGeneration = getWorkspaceRequestGeneration();
    setLoading(true);
    setLoadError(false);
    try {
      const projList = await fetchProjects();
      if (requestId !== requestGenerationRef.current || workspaceGeneration !== getWorkspaceRequestGeneration()) return;
      const requestedFromUrl = typeof window === "undefined"
        ? ""
        : new URLSearchParams(window.location.search).get("project_id") || "";
      const routeParams = typeof window === "undefined" ? new URLSearchParams() : new URLSearchParams(window.location.search);
      const requestedDatasetId = routeParams.get("dataset_id");
      const requestedVersionValue = routeParams.get("version");
      const requestedVersion = requestedVersionValue ? Number(requestedVersionValue) : undefined;
      const requestedFingerprint = routeParams.get("fingerprint") || undefined;
      const projectId = requestedProjectId || selectedProjectIdRef.current || requestedFromUrl || projList[0]?.id || "";
      const data = await fetchDatasets(projectId || undefined);
      if (requestId !== requestGenerationRef.current || workspaceGeneration !== getWorkspaceRequestGeneration()) return;
      setDatasets(data);
      setProjects(projList);
      if (projectId) {
        selectedProjectIdRef.current = projectId;
        setSelectedProjectId(projectId);
        try {
          window.localStorage.setItem("minded-active-project-id", projectId);
        } catch (error) {
          console.error("Unable to persist the active project selection.", error);
        }
      }
      if (data.length > 0) {
        if (requestedDatasetId) {
          const requestedDataset = data.find((dataset) => dataset.id === requestedDatasetId);
          if (!requestedDataset) {
            selectionRequestRef.current += 1;
            setDatasetLoading(false);
            setSelectedDataset(null);
            setTableData(null);
            setSourceError("The original alert source dataset is unavailable in this workspace. No replacement dataset was selected.");
            return;
          }
          if (!Number.isInteger(requestedVersion) || (requestedVersion ?? 0) < 1) {
            selectionRequestRef.current += 1;
            setDatasetLoading(false);
            setSelectedDataset(null);
            setTableData(null);
            setSourceError("The alert did not include a valid source version. Its historical profile cannot be opened as an exact snapshot.");
            return;
          }
          setSourceError(null);
          await selectDataset(
            requestedDataset.id,
            requestedVersion,
            workspaceGeneration,
            true,
            requestedFingerprint
          );
          return;
        }
        setSourceError(null);
        const current = data.find((dataset) => dataset.id === selectedDataset?.id);
        const next = current || data[0];
        await selectDataset(next.id, next.current_version, workspaceGeneration);
      } else {
        selectionRequestRef.current += 1;
        setDatasetLoading(false);
        setSelectedDataset(null);
        setTableData(null);
        setPreviewError(null);
        if (requestedDatasetId) {
          setSourceError("The original alert source dataset is unavailable. Restore that dataset before opening its recorded profile.");
        }
      }
    } catch (e) {
      console.error(e);
      if (requestId === requestGenerationRef.current && workspaceGeneration === getWorkspaceRequestGeneration()) {
        setDatasets([]);
        setDatasetLoading(false);
        setSelectedDataset(null);
        setTableData(null);
        setPreviewError(null);
        setLoadError(true);
      }
    } finally {
      if (requestId === requestGenerationRef.current && workspaceGeneration === getWorkspaceRequestGeneration()) {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    void loadAll();
    const onResetStarted = () => {
      requestGenerationRef.current += 1;
      selectionRequestRef.current += 1;
      setWorkspaceResetting(true);
      setLoading(true);
      setDatasetLoading(false);
      setLoadError(false);
      setDatasets([]);
      setSelectedDataset(null);
      setTableData(null);
      setPreviewError(null);
    };
    const onResetFinished = () => {
      setWorkspaceResetting(false);
      void loadAll();
    };
    window.addEventListener(workspaceResetEvents.started, onResetStarted);
    window.addEventListener(workspaceResetEvents.completed, onResetFinished);
    window.addEventListener(workspaceResetEvents.failed, onResetFinished);
    return () => {
      window.removeEventListener(workspaceResetEvents.started, onResetStarted);
      window.removeEventListener(workspaceResetEvents.completed, onResetFinished);
      window.removeEventListener(workspaceResetEvents.failed, onResetFinished);
    };
  }, []);

  const selectDataset = async (
    datasetId: string,
    expectedVersion?: number,
    generation = getWorkspaceRequestGeneration(),
    exactVersion = false,
    expectedFingerprint?: string
  ) => {
    const requestId = ++selectionRequestRef.current;
    setSelectedDataset(null);
    setTableData(null);
    setPreviewError(null);
    setDatasetLoading(true);
    setLoadError(false);
    try {
      const details = await fetchDatasetDetails(datasetId, exactVersion ? expectedVersion : undefined);
      if (generation !== getWorkspaceRequestGeneration() || requestId !== selectionRequestRef.current) return;
      if (!exactVersion && expectedVersion !== undefined && details.current_version !== expectedVersion) {
        throw new Error("The dataset changed while its profile was loading. Refresh the catalog to verify the current version.");
      }
      const loadedVersion = details.profile_version ?? details.current_version;
      if (exactVersion && loadedVersion !== expectedVersion) {
        throw new Error("The exact historical dataset version is unavailable.");
      }
      if (expectedFingerprint && details.dataset_fingerprint !== expectedFingerprint) {
        throw new Error("The source fingerprint does not match the historical alert. The profile is hidden to prevent snapshot substitution.");
      }
      setSelectedDataset(details);
      try {
        const rows = await fetchDatasetData(datasetId, loadedVersion, 50);
        if (generation !== getWorkspaceRequestGeneration() || requestId !== selectionRequestRef.current) return;
        setTableData(rows);
      } catch (e) {
        console.error(e);
        if (generation === getWorkspaceRequestGeneration() && requestId === selectionRequestRef.current) {
          setTableData(null);
          setPreviewError(e instanceof Error ? e.message : "Data preview is unavailable.");
        }
      }
    } catch (e) {
      console.error(e);
      if (generation === getWorkspaceRequestGeneration() && requestId === selectionRequestRef.current) {
        setSelectedDataset(null);
        setTableData(null);
        setPreviewError(null);
        setLoadError(true);
        if (exactVersion) {
          setSourceError(e instanceof Error ? e.message : "The exact historical source version could not be verified.");
        }
      }
    } finally {
      if (generation === getWorkspaceRequestGeneration() && requestId === selectionRequestRef.current) {
        setDatasetLoading(false);
      }
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;
    if (workspaceResetting) return;

    const files = Array.from(fileList);
    const targetProject = selectedProjectId || (projects.length > 0 ? projects[0].id : "");
    if (!targetProject) {
      alert("No local project is available yet. Refresh the workspace after local initialization.");
      return;
    }

    setIsUploading(true);
    beginWorkspaceUpload();

    const errors: string[] = [];
    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        setUploadProgress(`Profiling ${i + 1}/${files.length}: ${file.name}...`);
        const formData = new FormData();
        formData.append("file", file);
        formData.append("project_id", targetProject);
        formData.append("description", `Uploaded ${file.name}`);
        formData.append("business_timezone", "UTC");
        try {
          await uploadDatasetFile(formData);
        } catch (err: any) {
          errors.push(`${file.name}: ${err?.message || "Failed"}`);
        }
      }
      if (errors.length > 0) {
        alert(`Some uploads failed:\n${errors.join("\n")}`);
      }
      await loadAll(targetProject);
    } finally {
      setIsUploading(false);
      setUploadProgress(null);
      finishWorkspaceUpload();
      e.target.value = "";
    }
  };

  const filteredRows = tableData?.rows?.filter((row: any) => {
    if (!searchQuery.trim()) return true;
    return Object.values(row).some((val) =>
      String(val).toLowerCase().includes(searchQuery.toLowerCase())
    );
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold text-cyan-400 mb-1">
            <Database className="w-4 h-4" />
            <span>DATA LAKE & PROFILING CATALOG</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Data Explorer & Schema Intelligence</h1>
          <p className="text-xs text-slate-400 mt-1">
            Immutable Parquet versioning, automated column profiling, and composite data quality scoring.
          </p>
        </div>

        {/* Upload Button */}
        {projects.length > 1 && (
          <select
            value={selectedProjectId}
            onChange={(event) => {
              const projectId = event.target.value;
              selectedProjectIdRef.current = projectId;
              setSelectedProjectId(projectId);
              void loadAll(projectId);
            }}
            aria-label="Select project workspace"
            className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-xl px-3 py-2.5"
          >
            {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
          </select>
        )}
        <label className={`cursor-pointer bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold px-4 py-2.5 rounded-xl shadow-lg shadow-cyan-600/30 flex items-center space-x-2 transition-all ${workspaceResetting ? "opacity-50 pointer-events-none" : ""}`}>
          <UploadCloud className="w-4 h-4" />
          <span>{uploadProgress || (isUploading ? "Profiling..." : "Upload New Dataset")}</span>
          <input
            type="file"
            multiple
            accept=".csv,.xlsx,.json,.parquet"
            onChange={handleFileUpload}
            disabled={isUploading || workspaceResetting}
            className="hidden"
          />
        </label>
      </div>

      {/* Dataset Selection Bar or Empty State */}
      {loadError ? (
        <div className="p-10 rounded-2xl bg-slate-900/60 border border-dashed border-amber-500/30 text-center space-y-2">
          <ShieldCheck className="w-7 h-7 mx-auto text-amber-400" />
          <h3 className="text-sm font-bold text-white">Dataset state unavailable</h3>
          <p className="text-xs text-slate-400 max-w-lg mx-auto">{sourceError || "The local API did not return a verified dataset catalog, so MindEd will not display zero datasets as though that were observed."}</p>
        </div>
      ) : datasets.length === 0 ? (
        <div className="p-10 rounded-2xl bg-slate-900/60 border border-dashed border-slate-800 text-center space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mx-auto text-cyan-400">
            <UploadCloud className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-white">No Datasets Ingested</h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              Upload your own CSV, Excel, Parquet, or JSON files. They are stored as local versioned Parquet, profiled deterministically, and made available to DuckDB-backed analytical workflows.
            </p>
          </div>
          <label className="cursor-pointer inline-flex items-center space-x-2 bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white text-xs font-semibold px-5 py-2.5 rounded-xl shadow-lg shadow-cyan-500/20 transition-all">
            <UploadCloud className="w-4 h-4" />
            <span>{uploadProgress || "Select File(s) to Ingest"}</span>
            <input
              type="file"
              multiple
              accept=".csv,.xlsx,.json,.parquet"
              onChange={handleFileUpload}
              disabled={isUploading}
              className="hidden"
            />
          </label>
        </div>
      ) : (
        <div className="flex items-center space-x-2 overflow-x-auto pb-2 border-b border-slate-800">
          {datasets.map((d) => (
            <button
              key={d.id}
              onClick={() => selectDataset(d.id, d.current_version)}
              className={`px-4 py-2 rounded-xl text-xs font-medium flex items-center space-x-2 transition-all flex-shrink-0 ${
                selectedDataset?.id === d.id
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                  : "bg-slate-900/60 text-slate-400 hover:text-white border border-slate-800"
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              <span className="font-semibold">{d.name}</span>
              <span className="text-[10px] font-mono opacity-70">v{d.current_version}</span>
            </button>
          ))}
        </div>
      )}
      {sourceError && !loadError && (
        <div role="alert" className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-4 text-xs text-amber-100">
          {sourceError}
        </div>
      )}

      {datasetLoading && (
        <div role="status" className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-xs text-slate-400">
          Loading and verifying the selected dataset version…
        </div>
      )}

      {selectedDataset && (
        <div className="space-y-6">
          {/* First Look: Deterministic Analyst Orientation and Suggested Questions */}
          {selectedDataset.profile?.first_look && (
            <section className="bg-slate-900/80 border border-cyan-500/30 rounded-2xl p-6 shadow-xl space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
                <div className="flex items-center space-x-2 text-cyan-400">
                  <Sparkles className="w-4 h-4" />
                  <span className="text-xs font-bold uppercase tracking-wider">Dataset First Look — Analyst Orientation</span>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-slate-400">Quality Score:</span>
                  <span className="font-bold text-emerald-400 font-mono">{selectedDataset.profile.first_look.quality_score}/100</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                {/* Row meaning & shape */}
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-2">
                  <div className="font-semibold text-white flex items-center gap-2">
                    <Database className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Unit of Analysis & Scope</span>
                  </div>
                  <p className="text-slate-300 leading-relaxed">
                    {selectedDataset.profile.first_look.row_meaning}
                  </p>
                  <div className="text-[11px] text-slate-500 pt-1 font-mono">
                    Total Observations: <strong className="text-white">{selectedDataset.row_count?.toLocaleString()}</strong> · Total Attributes: <strong className="text-white">{selectedDataset.column_count}</strong>
                  </div>
                </div>

                {/* Quality Findings */}
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-2">
                  <div className="font-semibold text-white flex items-center gap-2">
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Data Quality & Integrity Findings</span>
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-slate-300">
                    {selectedDataset.profile.first_look.quality_findings.map((f: string, i: number) => (
                      <li key={i}>{f}</li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Clickable Suggested Questions */}
              {selectedDataset.profile.first_look.suggested_questions?.length > 0 && (
                <div className="pt-2 border-t border-slate-800/80 space-y-2.5">
                  <div className="text-[11px] font-bold uppercase tracking-wide text-slate-400 flex items-center gap-2">
                    <Bot className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Recommended Analyst Investigations (Click to Launch in Studio)</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2.5">
                    {selectedDataset.profile.first_look.suggested_questions.map((q: any, i: number) => (
                      <button
                        key={i}
                        type="button"
                        onClick={() => router.push(`/analyst?project_id=${encodeURIComponent(selectedProjectId)}&dataset=${encodeURIComponent(selectedDataset.id)}&q=${encodeURIComponent(q.question)}`)}
                        className="text-left bg-slate-950/90 hover:bg-cyan-950/30 border border-slate-800 hover:border-cyan-500/40 rounded-xl p-3 transition group flex flex-col justify-between"
                      >
                        <div>
                          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 uppercase">
                            {q.category}
                          </span>
                          <p className="text-xs font-semibold text-white mt-1.5 group-hover:text-cyan-300 transition">
                            {q.question}
                          </p>
                        </div>
                        <p className="text-[10px] text-slate-500 mt-2">
                          {q.purpose}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}

          {/* Immediate deterministic analytics: computed from the same authoritative ingestion profile. */}
          <section className="space-y-3">
            <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 text-cyan-300 text-xs font-semibold uppercase tracking-wider">
                  <Zap className="w-4 h-4" />
                  <span>Instant Analytics</span>
                </div>
                <h2 className="text-lg font-bold text-white mt-1">First-pass numeric findings</h2>
                <p className="text-[11px] text-slate-400 mt-1 max-w-2xl">
                  These values come directly from the authoritative ingestion profile for this dataset version. No demo values, sampling, or AI-generated numbers are used here.
                </p>
              </div>
              <div className="text-right text-[10px] font-mono text-slate-500 space-y-1">
                <div>Population variance · missing values excluded</div>
                <div>Source v{selectedDataset.profile_version ?? selectedDataset.current_version} · {selectedDataset.row_count.toLocaleString()} observed rows</div>
              </div>
            </div>

            {(() => {
              if (!selectedDataset.profile) {
                return (
                  <div className="bg-slate-900/60 border border-dashed border-amber-500/30 rounded-2xl p-5 text-xs text-amber-200">
                    The current dataset version is available, but its ingestion profile could not be verified. Profile statistics are hidden.
                  </div>
                );
              }

              const numericColumns = (selectedDataset.profile.columns || []).filter((col: any) => {
                const type = String(col.data_type || "").toUpperCase();
                return ["INTEGER", "FLOAT", "DECIMAL", "NUMERIC"].includes(type);
              });

              if (numericColumns.length === 0) {
                return (
                  <div className="bg-slate-900/60 border border-dashed border-slate-800 rounded-2xl p-5 text-xs text-slate-400">
                    No numeric columns were detected in this dataset. Semantic and categorical profiling continues below.
                  </div>
                );
              }

              return (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  {numericColumns.map((col: any) => (
                    <div
                      key={col.name}
                      className="bg-slate-900/70 border border-cyan-500/20 rounded-2xl p-4 shadow-lg shadow-cyan-950/10"
                    >
                      <div className="flex items-start justify-between gap-3 mb-3">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-emerald-400 shrink-0" />
                            <h3 className="text-sm font-bold text-white truncate">{col.name}</h3>
                          </div>
                          <p className="text-[10px] text-slate-500 mt-1 font-mono">
                            {col.semantic_type || "NUMERIC"} · {col.unique_count?.toLocaleString()} unique · {col.null_percentage ?? 0}% null
                          </p>
                        </div>
                        <span className="text-[10px] font-mono px-2 py-1 rounded-lg bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                          {col.data_type}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                        {[
                          ["Mean", col.mean_value],
                          ["Mode", col.mode_value],
                          ["Mode frequency", col.mode_frequency],
                          ["Variance", col.variance],
                          ["Lowest", col.min_value],
                          ["Highest", col.max_value],
                        ].map(([label, value]) => (
                          <div key={label} className="bg-slate-950/80 border border-slate-800 rounded-xl p-3">
                            <span className="block text-[10px] text-slate-500 uppercase tracking-wide">{label}</span>
                            <span className="block mt-1 text-sm font-semibold text-white font-mono truncate" title={value === null || value === undefined ? "—" : String(value)}>
                              {value === null || value === undefined ? (label === "Mode" ? "No mode" : "—") : typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : String(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              );
            })()}

            <div className="rounded-xl border border-slate-800 bg-slate-950/40 px-4 py-3 text-[10px] text-slate-400 font-mono">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
                <span>Profiled at: {selectedDataset.profile?.profiled_at ? new Date(selectedDataset.profile.profiled_at).toLocaleString() : "recorded ingestion time unavailable"}</span>
                <span>Source SHA-256: {selectedDataset.source_sha256 || "not recorded"}</span>
              </div>
              <div className="mt-1 text-slate-500">Every figure above is persisted with dataset version {selectedDataset.current_version} and reconciles to the source file hash.</div>
            </div>
          </section>

          {/* Metadata & Quality Banner */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center space-x-2">
                    <h2 className="text-xl font-bold text-white">{selectedDataset.name}</h2>
                    <span className="text-[10px] font-mono uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded-full flex items-center space-x-1">
                      <CheckCircle2 className="w-3 h-3" />
                      <span>RAW DATA AVAILABLE</span>
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">{selectedDataset.description || "Ingested relational table with Snappy Parquet backing."}</p>
                </div>

                {(selectedDataset.profile_version ?? selectedDataset.current_version) === selectedDataset.current_version ? (
                  <button
                    onClick={() => router.push(`/analyst?project_id=${encodeURIComponent(selectedProjectId)}&dataset=${encodeURIComponent(selectedDataset.id)}`)}
                    className="bg-cyan-600/20 hover:bg-cyan-600/30 text-cyan-300 border border-cyan-500/40 px-3.5 py-1.5 rounded-xl text-xs font-semibold flex items-center space-x-1.5 transition-all shadow-md"
                  >
                    <Bot className="w-3.5 h-3.5" />
                    <span>Investigate in Studio</span>
                  </button>
                ) : (
                  <span className="rounded-xl border border-amber-500/30 bg-amber-950/20 px-3.5 py-1.5 text-[10px] text-amber-100">
                    Historical snapshot · read only
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[11px] text-slate-400 block">Row Count</span>
                  <span className="text-lg font-bold text-white font-mono">{selectedDataset.row_count?.toLocaleString()}</span>
                  <span className="text-[9px] text-slate-500 block mt-1">Observed in ingested v{selectedDataset.profile_version ?? selectedDataset.current_version}</span>
                </div>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[11px] text-slate-400 block">Column Count</span>
                  <span className="text-lg font-bold text-white font-mono">{selectedDataset.column_count}</span>
                  <span className="text-[9px] text-slate-500 block mt-1">Observed in ingested v{selectedDataset.profile_version ?? selectedDataset.current_version}</span>
                </div>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[11px] text-slate-400 block">Storage Engine</span>
                  <span className="text-lg font-bold text-amber-300 font-mono">DuckDB + Parquet</span>
                  <span className="text-[9px] text-slate-500 block mt-1">Local analytical backing</span>
                </div>
              </div>
              <div className="bg-slate-950/70 p-3 rounded-xl border border-cyan-500/10 text-[10px] text-slate-400 font-mono">
                <div className="flex items-center gap-2 text-cyan-300 font-sans font-semibold mb-1.5">
                  <ShieldCheck className="w-3.5 h-3.5" /> Source provenance
                </div>
                <div>Source file: <span className="text-slate-200">{selectedDataset.source_filename || "not recorded"}</span></div>
                <div>Dataset version: <span className="text-slate-200">v{selectedDataset.profile_version ?? selectedDataset.current_version}</span></div>
                <div className="break-all">SHA-256: <span className="text-cyan-300">{selectedDataset.source_sha256 || "not recorded"}</span></div>
              </div>
            </div>

            {/* Quality Score Breakdown Card */}
            <QualityScoreBadge
              score={selectedDataset.profile ? (selectedDataset.data_quality_score ?? null) : null}
              breakdown={selectedDataset.profile?.data_quality}
            />
          </div>

          {/* Navigation Tabs (Profile vs Data Preview vs Semantic) */}
          <div className="flex items-center space-x-2 border-b border-slate-800">
            <button
              onClick={() => setActiveTab("profile")}
              className={`px-4 py-2.5 text-xs font-semibold border-b-2 transition-all flex items-center space-x-2 ${
                activeTab === "profile"
                  ? "border-cyan-500 text-cyan-400"
                  : "border-transparent text-slate-400 hover:text-white"
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span>Column Profiles & Distributions</span>
            </button>
            <button
              onClick={() => setActiveTab("data")}
              className={`px-4 py-2.5 text-xs font-semibold border-b-2 transition-all flex items-center space-x-2 ${
                activeTab === "data"
                  ? "border-cyan-500 text-cyan-400"
                  : "border-transparent text-slate-400 hover:text-white"
              }`}
            >
              <Table className="w-4 h-4" />
              <span>Data Explorer Grid</span>
            </button>
          </div>

          {/* Tab 1: Column Profiles */}
          {activeTab === "profile" && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {selectedDataset.profile?.columns ? (
                selectedDataset.profile.columns.map((col) => (
                  <div
                    key={col.name}
                    className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-3 hover:border-slate-700 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-sm text-white truncate pr-2">{col.name}</span>
                      <span className="text-[10px] font-mono uppercase bg-slate-800 text-cyan-300 px-2 py-0.5 rounded border border-slate-700">
                        {col.data_type}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                      <div className="bg-slate-950 p-2 rounded border border-slate-800/80">
                        <span className="text-[10px] text-slate-400 block font-sans">Unique Values</span>
                        <span className="text-slate-200">{col.unique_count?.toLocaleString()}</span>
                      </div>
                      <div className="bg-slate-950 p-2 rounded border border-slate-800/80">
                        <span className="text-[10px] text-slate-400 block font-sans">Nulls</span>
                        <span className="text-slate-200">{col.null_percentage}%</span>
                      </div>
                    </div>

                    {col.mean_value !== null && col.mean_value !== undefined && (
                      <div className="text-xs text-slate-300 space-y-1 pt-2 border-t border-slate-800">
                        <div className="flex justify-between font-mono text-[11px]">
                          <span className="text-slate-400">Mean:</span>
                          <span>{col.mean_value.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between font-mono text-[11px]">
                          <span className="text-slate-400">Median:</span>
                          <span>{col.median_value?.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between font-mono text-[11px]">
                          <span className="text-slate-400">Min / Max:</span>
                          <span>{col.min_value} / {col.max_value}</span>
                        </div>
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="col-span-3 bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center text-xs text-slate-400">
                  Profile metadata being computed.
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Data Explorer Grid */}
          {activeTab === "data" && !tableData && !datasetLoading && previewError && (
            <div role="alert" className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-4 text-xs text-amber-100">
              This dataset version and its saved profile are available, but the row preview could not be loaded: {previewError}
            </div>
          )}
          {activeTab === "data" && tableData && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden space-y-3 p-4">
              <div className="flex items-center justify-between gap-4">
                <div className="relative flex-1 max-w-sm">
                  <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
                  <input
                    type="text"
                    placeholder="Search in table rows..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <span className="text-xs text-slate-400 font-mono">
                  Showing {filteredRows?.length || 0} of {tableData.total_rows?.toLocaleString()} rows
                </span>
              </div>

              <div className="overflow-x-auto rounded-lg border border-slate-800">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider font-mono text-[10px]">
                    <tr>
                      {tableData.columns.map((c: string) => (
                        <th key={c} className="p-3 border-b border-slate-800">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 text-slate-300">
                    {filteredRows?.map((r: any, idx: number) => (
                      <tr key={idx} className="hover:bg-slate-800/30">
                        {tableData.columns.map((c: string) => (
                          <td key={c} className="p-3 font-mono text-[11px]">
                            {String(r[c] !== null && r[c] !== undefined ? r[c] : "NULL")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
