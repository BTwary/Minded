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

export default function Header() {
  const [isResetting, setIsResetting] = useState(false);
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

  const handleReset = async () => {
    if (!confirm("Are you sure you want to reset all workspace data to a clean slate (0 datasets, 0 analyses)?")) {
      return;
    }
    setIsResetting(true);
    try {
      const projects = await fetchProjects().catch(() => []);
      const activeProjId = projects[0]?.id || "";
      await resetWorkspace(activeProjId);
      window.location.reload();
    } catch (e: any) {
      alert(`Reset failed: ${e.message}`);
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
          {/* Theme control: persistent local light/dark presentation */}
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            className="theme-toggle inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1.5 text-xs text-slate-300 transition-colors"
          >
            {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
            <span className="hidden sm:inline">{theme === "dark" ? "Light" : "Dark"}</span>
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
            disabled={isResetting}
            title="Reset all tables, analyses, and files to 0"
            className="text-xs bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 px-2.5 py-1.5 rounded-lg font-medium flex items-center space-x-1.5 transition-all disabled:opacity-50"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">{isResetting ? "Resetting..." : "Reset Data"}</span>
          </button>
        </div>
      </header>

      {/* Global Command Palette */}
      <CommandPalette
        isOpen={isCommandOpen}
        onClose={() => setIsCommandOpen(false)}
      />
    </>
  );
}
