"use client";

import React, { useState } from "react";
import {
  Activity,
  Cpu,
  Database,
  FolderKanban,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  Zap,
  Sun,
  Moon,
} from "lucide-react";
import { fetchAISettings, fetchProjects, resetWorkspace } from "../lib/api";
import CommandPalette from "./CommandPalette";
import NotificationCenter from "./NotificationCenter";
import { useTheme } from "./ThemeProvider";
import {
  beginWorkspaceReset,
  completeWorkspaceReset,
  failWorkspaceReset,
  workspaceResetEvents,
} from "../lib/workspaceState";

export default function Header() {
  const [isResetting, setIsResetting] = useState(false);
  const [isUploadingDataset, setIsUploadingDataset] = useState(false);
  const [resetMessage, setResetMessage] = useState<string | null>(null);
  const [resetError, setResetError] = useState<string | null>(null);
  const [isCommandOpen, setIsCommandOpen] = useState(false);
  const [aiMode, setAiMode] = useState<{ active_mode?: string; provider?: string } | null>(null);
  const { theme, toggleTheme } = useTheme();

  React.useEffect(() => {
    fetchAISettings()
      .then((res: any) => {
        if (res) setAiMode(res);
      })
      .catch(() => {});
  }, []);

  React.useEffect(() => {
    const onUploadStarted = () => setIsUploadingDataset(true);
    const onUploadFinished = () => setIsUploadingDataset(false);
    window.addEventListener(workspaceResetEvents.uploadStarted, onUploadStarted);
    window.addEventListener(workspaceResetEvents.uploadFinished, onUploadFinished);
    return () => {
      window.removeEventListener(workspaceResetEvents.uploadStarted, onUploadStarted);
      window.removeEventListener(workspaceResetEvents.uploadFinished, onUploadFinished);
    };
  }, []);

  const handleReset = async () => {
    if (isResetting || isUploadingDataset) return;
    if (!window.confirm("Reset active datasets, profiles, alerts, and workspace artifacts? Saved analysis history, projects, theme, and application preferences will be preserved.")) {
      return;
    }
    setIsResetting(true);
    setResetMessage(null);
    setResetError(null);
    beginWorkspaceReset();
    try {
      const projects = await fetchProjects();
      let savedProjectId: string | null = null;
      try {
        savedProjectId = window.localStorage.getItem("minded-active-project-id");
      } catch (error) {
        console.error("Unable to read the active project selection.", error);
      }
      const activeProjId = projects.find((project) => project.id === savedProjectId)?.id || projects[0]?.id;
      if (!activeProjId) {
        throw new Error("No project is available to reset.");
      }
      const result = await resetWorkspace(activeProjId);
      if (result.status !== "success" || result.storage_cleanup_pending) {
        throw new Error("The backend did not confirm a complete workspace reset.");
      }
      completeWorkspaceReset();
      setResetMessage("Reset complete. Active dataset data was removed; saved analysis history was preserved.");
    } catch (e: any) {
      failWorkspaceReset();
      setResetError(`Reset failed: ${e?.message || "The server did not confirm completion."}`);
    } finally {
      setIsResetting(false);
    }
  };

  return (
    <>
      <header className="h-16 bg-slate-950/90 backdrop-blur border-b border-slate-800 flex items-center justify-between px-6 sticky top-0 z-20">
        {/* Left: Workspace & Search launcher */}
        <div className="flex items-center space-x-3 shrink-0">
          <div className="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800 text-xs">
            <FolderKanban className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400 hidden sm:inline">Workspace:</span>
            <span className="font-semibold text-white">Default Workspace</span>
          </div>

          {/* Quick Command Launcher button */}
          <button
            onClick={() => setIsCommandOpen(true)}
            className="flex items-center space-x-2 bg-slate-900/90 hover:bg-slate-800 text-slate-400 hover:text-slate-200 px-3 py-1.5 rounded-lg border border-slate-800 text-xs transition-colors"
          >
            <Search className="w-3.5 h-3.5 text-slate-500" />
            <span className="hidden md:inline">Search AA-OS...</span>
            <kbd className="text-[10px] font-mono bg-slate-950 text-slate-400 px-1.5 py-0.5 rounded border border-slate-800">
              Ctrl+K
            </kbd>
          </button>
        </div>

        {/* Engine Status & Workspace Control Actions */}
        <div className="flex items-center space-x-2.5 shrink-0">
          {/* Theme control: animated transition with fade, rotation, and color morph */}
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            className="theme-toggle group inline-flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/90 hover:border-amber-500/40 px-3 py-1.5 text-xs text-slate-300 hover:text-white transition-all duration-200 shadow-sm"
          >
            <span className="relative flex items-center justify-center w-4 h-4">
              <Sun
                className={`absolute inset-0 h-4 w-4 text-amber-400 transition-all duration-300 transform ${
                  theme === "dark"
                    ? "opacity-100 rotate-0 scale-100 drop-shadow-[0_0_6px_rgba(255,204,112,0.6)]"
                    : "opacity-0 -rotate-90 scale-50"
                }`}
              />
              <Moon
                className={`absolute inset-0 h-4 w-4 text-emerald-400 transition-all duration-300 transform ${
                  theme === "light"
                    ? "opacity-100 rotate-0 scale-100 drop-shadow-[0_0_6px_rgba(0,168,107,0.5)]"
                    : "opacity-0 rotate-90 scale-50"
                }`}
              />
            </span>
            <span className="hidden sm:inline font-medium transition-colors duration-200">
              {theme === "dark" ? "Light Mode" : "Dark Mode"}
            </span>
          </button>

          {/* Notification Center */}
          <NotificationCenter />

          {/* Dual-Mode Indicator Badge (Mode 1: Deterministic vs Mode 2: AI Augmented) */}
          <a
            href="/settings"
            title={
              aiMode?.active_mode === "AI_AUGMENTED"
                ? `Mode 2: AI Augmented (${aiMode?.provider || "Configured provider"})`
                : aiMode ? "Mode 1: Deterministic Free Mode ($0 / Zero AI API)" : "Analytical mode status unavailable"
            }
            className={`flex items-center space-x-1.5 text-xs px-2.5 py-1.5 rounded-full border transition-all ${
              aiMode?.active_mode === "AI_AUGMENTED"
                ? "bg-cyan-950/40 border-cyan-500/40 text-cyan-300 hover:border-cyan-400"
                : aiMode
                ? "bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700"
                : "bg-slate-900 border-amber-500/30 text-amber-300 hover:border-amber-400"
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                aiMode?.active_mode === "AI_AUGMENTED"
                  ? "bg-cyan-400 animate-pulse"
                  : aiMode ? "bg-emerald-400" : "bg-amber-400"
              }`}
            />
            <span className="font-medium hidden sm:inline">
              {aiMode?.active_mode === "AI_AUGMENTED" ? "AI Augmented" : aiMode ? "Deterministic" : "Status unavailable"}
            </span>
            <span className="text-[10px] font-mono px-1 py-0.5 rounded bg-slate-950/80 border border-slate-800 text-slate-400">
              {aiMode?.active_mode === "AI_AUGMENTED" ? (aiMode?.provider || "AI").toUpperCase() : aiMode ? "FREE" : "N/A"}
            </span>
          </a>
          {/* Quick Action: Reset Workspace to Clean Slate */}
          <button
            onClick={handleReset}
            disabled={isResetting || isUploadingDataset}
            title="Remove active dataset and workspace data while preserving saved analysis history"
            className="text-xs bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 px-2.5 py-1.5 rounded-lg font-medium flex items-center space-x-1.5 transition-all disabled:opacity-50"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">{isResetting ? "Resetting..." : isUploadingDataset ? "Importing..." : "Reset Data"}</span>
          </button>
        </div>
      </header>
      {(resetMessage || resetError) && (
        <div
          role={resetError ? "alert" : "status"}
          className={`fixed right-4 top-20 z-50 max-w-md rounded-xl border px-4 py-3 text-xs shadow-xl ${
            resetError
              ? "border-rose-500/40 bg-slate-950 text-rose-200"
              : "border-emerald-500/40 bg-slate-950 text-emerald-200"
          }`}
        >
          {resetError || resetMessage}
          <button
            type="button"
            onClick={() => { setResetError(null); setResetMessage(null); }}
            className="ml-3 underline underline-offset-2"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Global Command Palette */}
      <CommandPalette
        isOpen={isCommandOpen}
        onClose={() => setIsCommandOpen(false)}
      />
    </>
  );
}
