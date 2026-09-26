"use client";

import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  Bot,
  Cloud,
  CheckCircle2,
  Cpu,
  Database,
  Globe,
  HardDrive,
  Key,
  Layers,
  Lock,
  RefreshCw,
  Save,
  Server,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import {
  fetchAISettings,
  fetchInfrastructureMatrix,
  testAIConnection,
  updateAISettings,
  testCloudStorage,
  createCloudBackup,
  listCloudBackups,
  restoreCloudBackup,
  recoverUploadedBackup,
  migrateLocalStorageToCloud,
} from "../../lib/api";

export default function SettingsPage() {
  const [settings, setSettings] = useState<any>(null);
  const [infraMatrix, setInfraMatrix] = useState<any>(null);
  const [aiEnabled, setAiEnabled] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState("none");
  const [selectedModel, setSelectedModel] = useState("auto");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");

  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<{
    status: "success" | "error";
    message: string;
    latency_ms?: number;
    sample_response?: string;
  } | null>(null);

  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  const [cloudProvider, setCloudProvider] = useState<"s3" | "gcs">("s3");
  const [cloudBucket, setCloudBucket] = useState("");
  const [cloudRegion, setCloudRegion] = useState("us-east-1");
  const [cloudEndpoint, setCloudEndpoint] = useState("");
  const [cloudAccessKey, setCloudAccessKey] = useState("");
  const [cloudSecretKey, setCloudSecretKey] = useState("");
  const [cloudProjectId, setCloudProjectId] = useState("");
  const [backupPassphrase, setBackupPassphrase] = useState("");
  const [includeSourceFiles, setIncludeSourceFiles] = useState(true);
  const [cloudBusy, setCloudBusy] = useState(false);
  const [cloudMessage, setCloudMessage] = useState<string | null>(null);
  const [cloudBackups, setCloudBackups] = useState<any[]>([]);
  const [recoveryFile, setRecoveryFile] = useState<File | null>(null);

  const loadAll = async () => {
    try {
      const [aiData, matrixData] = await Promise.all([
        fetchAISettings().catch(() => null),
        fetchInfrastructureMatrix().catch(() => null),
      ]);
      if (aiData) {
        setSettings(aiData);
        setAiEnabled(Boolean((aiData as any).enabled));
        setSelectedProvider((aiData as any).provider || "none");
        setSelectedModel((aiData as any).model || "auto");
        setBaseUrl((aiData as any).base_url || "");
      }
      if (matrixData) {
        setInfraMatrix(matrixData);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    loadAll();
  }, []);

  const cloudConfig = () => ({
    provider: cloudProvider,
    bucket: cloudBucket,
    region: cloudRegion,
    endpoint_url: cloudEndpoint || undefined,
    access_key_id: cloudAccessKey || undefined,
    secret_access_key: cloudSecretKey || undefined,
    project_id: cloudProjectId || undefined,
  });

  const handleCloudTest = async () => {
    setCloudBusy(true);
    setCloudMessage(null);
    try {
      const res = await testCloudStorage(cloudConfig()) as any;
      setCloudMessage(res?.message || "Cloud storage connection verified.");
    } catch (e: any) {
      setCloudMessage(`Cloud test failed: ${e.message}`);
    } finally {
      setCloudBusy(false);
    }
  };

  const handleLocalStorageMigration = async () => {
    setCloudBusy(true);
    setCloudMessage(null);
    try {
      const res = await migrateLocalStorageToCloud(cloudConfig()) as any;
      setCloudMessage(`Local storage migration completed: ${res?.uploaded_files ?? 0} files copied; ${res?.database_paths_updated ?? 0} dataset paths updated. Local copies were retained.`);
    } catch (e: any) {
      setCloudMessage(`Migration failed: ${e.message}`);
    } finally {
      setCloudBusy(false);
    }
  };

  const handleCloudBackup = async () => {
    if (backupPassphrase.length < 8) {
      setCloudMessage("Use a backup passphrase of at least 8 characters. AA-OS does not store it.");
      return;
    }
    setCloudBusy(true);
    setCloudMessage(null);
    try {
      const res = await createCloudBackup({ ...cloudConfig(), passphrase: backupPassphrase, include_source_files: includeSourceFiles }) as any;
      setCloudMessage(`Encrypted backup created: ${res?.backup_id} (${res?.size_bytes ?? 0} bytes).`);
      const listed = await listCloudBackups({ provider: cloudProvider, bucket: cloudBucket, region: cloudRegion, endpoint_url: cloudEndpoint || undefined }) as any;
      setCloudBackups(listed?.backups || []);
    } catch (e: any) {
      setCloudMessage(`Backup failed: ${e.message}`);
    } finally {
      setCloudBusy(false);
    }
  };

  const handleCloudRestore = async (objectKey: string) => {
    if (backupPassphrase.length < 8) {
      setCloudMessage("Enter the passphrase used when the backup was created.");
      return;
    }
    setCloudBusy(true);
    setCloudMessage(null);
    try {
      const res = await restoreCloudBackup({ ...cloudConfig(), object_key: objectKey, passphrase: backupPassphrase, replace_existing: false }) as any;
      setCloudMessage(`Restore completed: ${res?.restored_storage_files ?? 0} source files and database state restored.`);
    } catch (e: any) {
      setCloudMessage(`Restore failed: ${e.message}`);
    } finally {
      setCloudBusy(false);
    }
  };

  const handleLocalRecovery = async () => {
    if (!recoveryFile || backupPassphrase.length < 8) {
      setCloudMessage("Select an .aaosbackup file and enter its passphrase.");
      return;
    }
    setCloudBusy(true);
    setCloudMessage(null);
    try {
      const res = await recoverUploadedBackup(recoveryFile, backupPassphrase, false) as any;
      setCloudMessage(`Local migration completed: ${res?.status || "RESTORED"}.`);
    } catch (e: any) {
      setCloudMessage(`Local recovery failed: ${e.message}`);
    } finally {
      setCloudBusy(false);
    }
  };

  const handleModeToggle = (enabled: boolean) => {
    setAiEnabled(enabled);
    setTestResult(null);
    setSaveMessage(null);
    if (!enabled) {
      setSelectedProvider("none");
    } else if (selectedProvider === "none") {
      setSelectedProvider("gemini");
      const provObj = settings?.supported_providers?.find((p: any) => p.id === "gemini");
      if (provObj && provObj.models?.length > 0) {
        setSelectedModel(provObj.models[0]);
      }
    }
  };

  const handleProviderSelect = (provId: string) => {
    setSelectedProvider(provId);
    setTestResult(null);
    setSaveMessage(null);

    if (provId === "none") {
      setAiEnabled(false);
    } else {
      setAiEnabled(true);
    }

    const provObj = settings?.supported_providers?.find((p: any) => p.id === provId);
    if (provObj && provObj.models && provObj.models.length > 0) {
      setSelectedModel(provObj.models[0]);
      if (provObj.default_base_url && provObj.default_base_url.startsWith("http")) {
        setBaseUrl(provObj.default_base_url);
      } else {
        setBaseUrl("");
      }
    }
  };

  const handleTestConnection = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      const res = await testAIConnection({
        provider: selectedProvider,
        model: selectedModel,
        api_key: apiKey,
        base_url: baseUrl,
      });
      setTestResult(res as any);
    } catch (e: any) {
      setTestResult({
        status: "error",
        message: e.message || "Failed to reach AI endpoint.",
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    setSaveMessage(null);
    try {
      const res = (await updateAISettings({
        enabled: aiEnabled,
        provider: aiEnabled ? selectedProvider : "none",
        model: selectedModel,
        api_key: apiKey,
        base_url: baseUrl,
      })) as any;
      setSaveMessage(res?.message || "Settings updated successfully!");
      setSettings(res?.config);
      await loadAll();
    } catch (e: any) {
      setSaveMessage("Error updating settings: " + e.message);
    } finally {
      setIsSaving(false);
    }
  };

  const currentProviderInfo = settings?.supported_providers?.find(
    (p: any) => p.id === selectedProvider
  );

  return (
    <div className="space-y-10 max-w-5xl pb-12">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-semibold text-cyan-400 mb-1">
          <Globe className="w-4 h-4" />
          <span>DUAL-OPERATING ARCHITECTURE: MODE 1 (FREE) &amp; MODE 2 (AI AUGMENTED)</span>
        </div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Operating Mode &amp; AI Provider Settings</h1>
        <p className="text-xs text-slate-400 mt-1">
          AA-OS performs real analytical work deterministically without requiring any paid AI API. AI is an optional augmentation layer, never a hard dependency.
        </p>
      </div>

      {/* Dual Mode Selector Banner */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Mode 1 Card */}
        <div
          onClick={() => handleModeToggle(false)}
          className={`p-5 rounded-2xl border cursor-pointer transition-all ${
            !aiEnabled
              ? "bg-emerald-950/20 border-emerald-500/80 shadow-lg shadow-emerald-500/10 ring-1 ring-emerald-500/30"
              : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
          }`}
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
              <h3 className="text-sm font-bold text-white">Mode 1: Free / Zero AI API</h3>
            </div>
            {!aiEnabled && (
              <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 uppercase">
                ACTIVE
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 leading-relaxed mb-3">
            100% deterministic local analytics execution. Runs dataset profiling, hypothesis synthesis, EIG ranking, DuckDB SQL queries, statistical tests, Simpson&apos;s paradox detection, and Bayesian updates with zero external API calls.
          </p>
          <div className="flex items-center space-x-2 text-[11px] font-mono text-emerald-400/90">
            <span>$0 Cost</span>
            <span>•</span>
            <span>100% Private</span>
            <span>•</span>
            <span>No API Keys Required</span>
          </div>
        </div>

        {/* Mode 2 Card */}
        <div
          onClick={() => handleModeToggle(true)}
          className={`p-5 rounded-2xl border cursor-pointer transition-all ${
            aiEnabled
              ? "bg-cyan-950/20 border-cyan-500/80 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-500/30"
              : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
          }`}
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
              <h3 className="text-sm font-bold text-white">Mode 2: Optional AI Augmentation</h3>
            </div>
            {aiEnabled && (
              <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 uppercase">
                ACTIVE
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 leading-relaxed mb-3">
            Augments the core deterministic engine with your own LLM (Gemini, Claude, OpenAI, Ollama, Groq). AI assists with natural-language interpretation, ambiguous columns, and narrative synthesis without altering mathematical ground truth.
          </p>
          <div className="flex items-center space-x-2 text-[11px] font-mono text-cyan-400/90">
            <span>Bring-Your-Own-Key</span>
            <span>•</span>
            <span>Fail-Safe Fallbacks</span>
            <span>•</span>
            <span>Zero SaaS Lock-in</span>
          </div>
        </div>
      </div>

      {/* BYOI Real-Time Status & Cost Transparency Matrix */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
            <Server className="w-4 h-4 text-cyan-400" />
            <span>Connected Infrastructure Plane</span>
          </h2>
          <span className="text-xs px-2.5 py-0.5 rounded-full font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            {infraMatrix?.mode === "LOCAL_FREE" ? "🟢 Local / Free Community Mode" : "🔵 Connected BYOI Mode"}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {infraMatrix?.layers?.map((layer: any, idx: number) => (
            <div
              key={idx}
              className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-3 relative overflow-hidden"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white flex items-center space-x-1.5">
                  {layer.category === "database" && <Database className="w-3.5 h-3.5 text-cyan-400" />}
                  {layer.category === "storage" && <HardDrive className="w-3.5 h-3.5 text-indigo-400" />}
                  {layer.category === "ai" && <Bot className="w-3.5 h-3.5 text-sky-400" />}
                  {layer.category === "analytics" && <Cpu className="w-3.5 h-3.5 text-amber-400" />}
                  {layer.category === "sandbox" && <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />}
                  <span>{layer.name}</span>
                </span>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 uppercase">
                  {layer.status}
                </span>
              </div>

              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between text-slate-400">
                  <span>Engine:</span>
                  <span className="font-mono text-slate-200">{layer.provider}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Ownership:</span>
                  <span className="font-mono text-slate-300">{layer.owner}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Est. Billing:</span>
                  <span className="font-mono text-emerald-400">{layer.estimated_cost}</span>
                </div>
              </div>

              <p className="text-[11px] text-slate-400 border-t border-slate-800/80 pt-2 leading-relaxed">
                {layer.message}
              </p>
            </div>
          ))}
        </div>

        {/* Cost & Privacy Ownership Guarantee */}
        <div className="p-4 bg-slate-950/80 border border-slate-800 rounded-xl flex items-start space-x-3 text-xs text-slate-400">
          <ShieldCheck className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-semibold text-slate-200">Zero SaaS Markup &amp; Data Sovereignty Guarantee</h4>
            <p className="text-[11px] leading-relaxed">
              AA-OS does not rent compute, storage, or models. All credentials stay inside your secure environment. Third-party provider usage is billed directly to your own accounts without middleware surcharge.
            </p>
          </div>
        </div>
      </div>

      {/* Bring Your Own Model (BYOM) Selection */}
      <div className="space-y-4 pt-4 border-t border-slate-800">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
          <Zap className="w-4 h-4 text-cyan-400" />
          <span>Bring Your Own Model (BYOM)</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {settings?.supported_providers?.map((p: any) => {
            const isSelected = selectedProvider === p.id;
            return (
              <div
                key={p.id}
                onClick={() => handleProviderSelect(p.id)}
                className={`p-4 rounded-xl border cursor-pointer transition-all ${
                  isSelected
                    ? "bg-cyan-950/20 border-cyan-500 shadow-lg shadow-cyan-500/10"
                    : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-white">{p.name}</span>
                  {isSelected && <CheckCircle2 className="w-4 h-4 text-cyan-400" />}
                </div>
                <div className="space-y-1">
                  <span className="text-[10px] text-slate-400 block font-mono">MODELS:</span>
                  <div className="flex flex-wrap gap-1">
                    {p.models.slice(0, 2).map((m: string) => (
                      <span key={m} className="text-[10px] font-mono bg-slate-950 text-slate-300 px-1.5 py-0.5 rounded border border-slate-800">
                        {m}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Model Configuration Form */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-6">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2 border-b border-slate-800 pb-3">
          <Key className="w-4 h-4 text-cyan-400" />
          <span>Configure &amp; Test {currentProviderInfo?.name}</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Model selection */}
          <div>
            <label className="text-xs font-semibold text-slate-300 block mb-2">
              Model Selection
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
            >
              <option value="auto">Auto (Default Provider Model)</option>
              {currentProviderInfo?.models?.map((m: string) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          {/* Base URL */}
          <div>
            <label className="text-xs font-semibold text-slate-300 block mb-2">
              API Base URL / Endpoint
            </label>
            <input
              type="text"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="e.g. http://localhost:11434 or https://api.openai.com/v1"
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
            />
          </div>

          {/* API Key */}
          {currentProviderInfo?.requires_api_key && (
            <div className="md:col-span-2">
              <label className="text-xs font-semibold text-slate-300 block mb-2 flex items-center justify-between">
                <span>API Secret Key</span>
                <span className="text-[11px] text-slate-400 font-normal">
                  Masked key: <span className="font-mono">{settings?.api_key_masked}</span>
                </span>
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter your private API key..."
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
              />
              <p className="text-[11px] text-slate-400 mt-1.5">
                Your key is stored locally in environment variables and never logged or sent to external servers.
              </p>
            </div>
          )}
        </div>

        {/* Test Connection Output */}
        {testResult && (
          <div
            className={`p-4 rounded-xl border text-xs ${
              testResult.status === "success"
                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                : "bg-rose-500/10 border-rose-500/30 text-rose-300"
            }`}
          >
            <div className="flex items-center space-x-2 font-semibold">
              {testResult.status === "success" ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : (
                <AlertCircle className="w-4 h-4 text-rose-400" />
              )}
              <span>{testResult.message}</span>
            </div>
            {testResult.latency_ms && (
              <p className="font-mono text-[11px] text-slate-400 mt-1">
                Roundtrip Latency: {testResult.latency_ms}ms
              </p>
            )}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center justify-between pt-4 border-t border-slate-800">
          <button
            onClick={handleTestConnection}
            disabled={isTesting}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-semibold flex items-center space-x-2 transition-all disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isTesting ? "animate-spin" : ""}`} />
            <span>{isTesting ? "Testing Connection..." : "Test Model Connection"}</span>
          </button>

          <div className="flex items-center space-x-3">
            {saveMessage && (
              <span className="text-xs text-emerald-400 font-mono">{saveMessage}</span>
            )}
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="px-5 py-2 bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white rounded-xl text-xs font-semibold flex items-center space-x-2 shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-50"
            >
              <Save className="w-3.5 h-3.5" />
              <span>{isSaving ? "Saving..." : "Save Configuration"}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Optional user-owned cloud continuity */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-sm font-bold text-white">
              <Cloud className="w-4 h-4 text-cyan-400" />
              Optional Cloud Backup &amp; Migration
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl">
              Local storage remains the default. You can connect your own S3-compatible bucket or Google Cloud Storage account so AA-OS can keep an encrypted continuity backup. Cloud storage, transfer, and provider charges are yours; AA-OS does not require or subsidize them.
            </p>
          </div>
          <span className="text-[10px] font-semibold px-2 py-1 rounded-full border border-cyan-500/30 text-cyan-300 bg-cyan-500/10">BYOI • OPTIONAL</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <select value={cloudProvider} onChange={(e) => setCloudProvider(e.target.value as "s3" | "gcs")} className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-white">
            <option value="s3">S3 / S3-compatible (AWS, R2, MinIO)</option>
            <option value="gcs">Google Cloud Storage</option>
          </select>
          <input value={cloudBucket} onChange={(e) => setCloudBucket(e.target.value)} placeholder="Bucket name" className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-white" />
          <input value={cloudRegion} onChange={(e) => setCloudRegion(e.target.value)} placeholder="Region" className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-white" />
          <input value={cloudEndpoint} onChange={(e) => setCloudEndpoint(e.target.value)} placeholder="Endpoint URL (optional for R2/MinIO)" className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-white" />
          {cloudProvider === "gcs" ? (
            <input value={cloudProjectId} onChange={(e) => setCloudProjectId(e.target.value)} placeholder="GCP project ID (optional with ADC)" className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-white" />
          ) : (
            <>
              <input value={cloudAccessKey} onChange={(e) => setCloudAccessKey(e.target.value)} placeholder="Access key (not stored by AA-OS backup)" className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-white font-mono" />
              <input type="password" value={cloudSecretKey} onChange={(e) => setCloudSecretKey(e.target.value)} placeholder="Secret key (used only for this operation)" className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-white font-mono" />
            </>
          )}
          <input type="password" value={backupPassphrase} onChange={(e) => setBackupPassphrase(e.target.value)} placeholder="Backup encryption passphrase (8+ chars)" className="bg-slate-950 border border-emerald-500/30 rounded-xl px-3 py-2.5 text-xs text-white font-mono" />
        </div>

        <label className="flex items-center gap-2 text-xs text-slate-300">
          <input type="checkbox" checked={includeSourceFiles} onChange={(e) => setIncludeSourceFiles(e.target.checked)} />
          Include locally retained source files in the encrypted backup
        </label>

        <div className="flex flex-wrap gap-2">
          <button onClick={handleCloudTest} disabled={cloudBusy || !cloudBucket} className="px-3 py-2 rounded-xl bg-slate-800 text-xs text-slate-200 disabled:opacity-50">Test cloud connection</button>
          <button onClick={handleCloudBackup} disabled={cloudBusy || !cloudBucket || backupPassphrase.length < 8} className="px-3 py-2 rounded-xl bg-cyan-600/80 text-xs font-semibold text-white disabled:opacity-50">Create encrypted backup</button>
          <button onClick={handleLocalStorageMigration} disabled={cloudBusy || !cloudBucket} className="px-3 py-2 rounded-xl bg-indigo-600/80 text-xs font-semibold text-white disabled:opacity-50">Migrate local storage</button>
          <button onClick={async () => {
            try {
              const res = await listCloudBackups({ provider: cloudProvider, bucket: cloudBucket, region: cloudRegion, endpoint_url: cloudEndpoint || undefined }) as any;
              setCloudBackups(res?.backups || []);
              setCloudMessage(`${res?.backups?.length || 0} backup(s) found.`);
            } catch (e: any) { setCloudMessage(`Could not list backups: ${e.message}`); }
          }} disabled={cloudBusy || !cloudBucket} className="px-3 py-2 rounded-xl bg-slate-800 text-xs text-slate-200 disabled:opacity-50">Refresh backups</button>
        </div>

        {cloudMessage && <div className="p-3 rounded-xl border border-cyan-500/20 bg-cyan-500/5 text-xs text-cyan-200">{cloudMessage}</div>}

        {cloudBackups.length > 0 && (
          <div className="space-y-2">
            <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Available encrypted backups</div>
            {cloudBackups.map((backup) => (
              <div key={backup.object_key} className="flex items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <span className="font-mono text-[10px] text-slate-300 truncate">{backup.backup_id}</span>
                <button onClick={() => handleCloudRestore(backup.object_key)} disabled={cloudBusy || backupPassphrase.length < 8} className="px-3 py-1.5 rounded-lg bg-emerald-600/80 text-[10px] font-semibold text-white disabled:opacity-50">Restore / migrate</button>
              </div>
            ))}
          </div>
        )}

        <div className="border-t border-slate-800 pt-4 space-y-2">
          <div className="text-xs font-semibold text-white">Reinstall / offline recovery</div>
          <p className="text-[11px] text-slate-400">You can also restore a downloaded <code>.aaosbackup</code> file locally. This loop is restricted to the local AA-OS API and is intended for first-run recovery after software deletion/reinstallation.</p>
          <div className="flex flex-wrap items-center gap-2">
            <input type="file" accept=".aaosbackup,application/octet-stream" onChange={(e) => setRecoveryFile(e.target.files?.[0] || null)} className="text-[11px] text-slate-400" />
            <button onClick={handleLocalRecovery} disabled={cloudBusy || !recoveryFile || backupPassphrase.length < 8} className="px-3 py-2 rounded-xl bg-indigo-600/80 text-xs font-semibold text-white disabled:opacity-50">Recover from backup file</button>
          </div>
        </div>
      </div>

    </div>
  );
}
