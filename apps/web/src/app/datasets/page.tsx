"use client";

import React, { useEffect, useState } from "react";
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

export default function DatasetsPage() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [tableData, setTableData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [activeTab, setActiveTab] = useState<"profile" | "data" | "semantic">("profile");
  const [searchQuery, setSearchQuery] = useState("");
  const [loadError, setLoadError] = useState(false);

  const loadAll = async () => {
    setLoading(true);
    setLoadError(false);
    try {
      const [data, projList] = await Promise.all([
        fetchDatasets(),
        fetchProjects().catch(() => []),
      ]);
      setDatasets(data);
      setProjects(projList);
      if (projList.length > 0 && !selectedProjectId) {
        setSelectedProjectId(projList[0].id);
      }
      if (data.length > 0) {
        selectDataset(data[0].id);
      }
    } catch (e) {
      console.error(e);
      setLoadError(true);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAll();
  }, []);

  const selectDataset = async (datasetId: string) => {
    try {
      const details = await fetchDatasetDetails(datasetId);
      setSelectedDataset(details);
      const rows = await fetchDatasetData(datasetId, details.current_version, 50);
      setTableData(rows);
    } catch (e) {
      console.error(e);
      setLoadError(true);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    const targetProject = selectedProjectId || (projects.length > 0 ? projects[0].id : "");
    if (!targetProject) {
      alert("No local project is available yet. Refresh the workspace after local initialization.");
      setIsUploading(false);
      return;
    }
    formData.append("project_id", targetProject);
    formData.append("description", `Uploaded ${file.name}`);

    try {
      await uploadDatasetFile(formData);
      await loadAll();
    } catch (err: any) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
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
        <label className="cursor-pointer bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold px-4 py-2.5 rounded-xl shadow-lg shadow-cyan-600/30 flex items-center space-x-2 transition-all">
          <UploadCloud className="w-4 h-4" />
          <span>{isUploading ? "Profiling..." : "Upload New Dataset"}</span>
          <input
            type="file"
            accept=".csv,.xlsx,.json,.parquet"
            onChange={handleFileUpload}
            disabled={isUploading}
            className="hidden"
          />
        </label>
      </div>

      {/* Dataset Selection Bar or Empty State */}
      {loadError ? (
        <div className="p-10 rounded-2xl bg-slate-900/60 border border-dashed border-amber-500/30 text-center space-y-2">
          <ShieldCheck className="w-7 h-7 mx-auto text-amber-400" />
          <h3 className="text-sm font-bold text-white">Dataset state unavailable</h3>
          <p className="text-xs text-slate-400 max-w-lg mx-auto">The local API did not return a verified dataset catalog, so MindEd will not display zero datasets as though that were observed.</p>
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
            <span>Select File to Ingest</span>
            <input
              type="file"
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
              onClick={() => selectDataset(d.id)}
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

      {selectedDataset && (
        <div className="space-y-6">
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
                <div>Source v{selectedDataset.current_version} · {selectedDataset.row_count.toLocaleString()} observed rows</div>
              </div>
            </div>

            {(() => {
              const numericColumns = (selectedDataset.profile?.columns || []).filter((col: any) => {
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

                <button
                  onClick={() => router.push(`/analyst?dataset=${selectedDataset.id}`)}
                  className="bg-cyan-600/20 hover:bg-cyan-600/30 text-cyan-300 border border-cyan-500/40 px-3.5 py-1.5 rounded-xl text-xs font-semibold flex items-center space-x-1.5 transition-all shadow-md"
                >
                  <Bot className="w-3.5 h-3.5" />
                  <span>Investigate in Studio</span>
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[11px] text-slate-400 block">Row Count</span>
                  <span className="text-lg font-bold text-white font-mono">{selectedDataset.row_count?.toLocaleString()}</span>
                  <span className="text-[9px] text-slate-500 block mt-1">Observed in ingested v{selectedDataset.current_version}</span>
                </div>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[11px] text-slate-400 block">Column Count</span>
                  <span className="text-lg font-bold text-white font-mono">{selectedDataset.column_count}</span>
                  <span className="text-[9px] text-slate-500 block mt-1">Observed in ingested v{selectedDataset.current_version}</span>
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
                <div>Dataset version: <span className="text-slate-200">v{selectedDataset.current_version}</span></div>
                <div className="break-all">SHA-256: <span className="text-cyan-300">{selectedDataset.source_sha256 || "not recorded"}</span></div>
              </div>
            </div>

            {/* Quality Score Breakdown Card */}
            <QualityScoreBadge
              score={selectedDataset.data_quality_score ?? null}
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
