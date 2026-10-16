"use client";

import React, { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  Bell,
  Check,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Database,
  ExternalLink,
  Flame,
  HelpCircle,
  Info,
  LoaderCircle,
  MessageSquareText,
  RefreshCw,
  Search,
  ShieldCheck,
  Sliders,
  Sparkles,
  X,
} from "lucide-react";
import {
  createInvestigation,
  evaluateAlertRules,
  fetchAlertEvents,
  fetchAlertRules,
  fetchDatasets,
  fetchInvestigation,
  fetchMonitoringStatus,
  fetchProjects,
  transitionAlertEvent,
  Investigation,
} from "../../lib/api";
import { AlertEvent, AlertRule, Dataset, MonitoringStatus, Project } from "../../types";

type ConversationMessage = {
  eventId: string;
  role: "user" | "assistant";
  text: string;
  status?: string;
  investigationId?: string;
};

function eventStatus(event: AlertEvent): string {
  if (event.lifecycle_status) return event.lifecycle_status.toLowerCase();
  return event.is_resolved ? "resolved" : "active";
}

function statusClasses(status: string): string {
  if (status === "resolved") {
    return "border-emerald-500/30 bg-emerald-500/10 text-emerald-400";
  }
  if (status === "acknowledged") {
    return "border-amber-500/30 bg-amber-500/10 text-amber-300";
  }
  if (status === "snoozed") {
    return "border-violet-500/30 bg-violet-500/10 text-violet-300";
  }
  if (status === "failed") {
    return "border-rose-500/30 bg-rose-500/10 text-rose-300";
  }
  return "border-rose-500/40 bg-rose-500/15 text-rose-200 font-semibold";
}

function severityClasses(severity: string | undefined): { badge: string; border: string; accent: string } {
  const sev = (severity || "").toLowerCase();
  if (sev === "critical" || sev === "high") {
    return {
      badge: "border-rose-500/40 bg-rose-500/15 text-rose-200",
      border: "border-rose-500/30",
      accent: "text-rose-400",
    };
  }
  if (sev === "medium") {
    return {
      badge: "border-amber-500/40 bg-amber-500/15 text-amber-200",
      border: "border-amber-500/30",
      accent: "text-amber-400",
    };
  }
  return {
    badge: "border-sky-500/40 bg-sky-500/15 text-sky-200",
    border: "border-sky-500/30",
    accent: "text-sky-400",
  };
}

function fmt(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? "Not recorded"
    : value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function formatRelativeTime(dateString: string | null | undefined): string {
  if (!dateString) return "Never";
  try {
    const timestamp = new Date(dateString).getTime();
    if (Number.isNaN(timestamp)) return "Invalid date";
    const deltaSeconds = Math.floor((Date.now() - timestamp) / 1000);
    if (deltaSeconds < 45) return "Just now";
    if (deltaSeconds < 3600) return `${Math.floor(deltaSeconds / 60)}m ago`;
    if (deltaSeconds < 86400) return `${Math.floor(deltaSeconds / 3600)}h ago`;
    return new Date(dateString).toLocaleDateString();
  } catch {
    return "Not recorded";
  }
}

export default function AlertsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState("");
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [events, setEvents] = useState<AlertEvent[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [monitoring, setMonitoring] = useState<MonitoringStatus | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [inspectModalEvent, setInspectModalEvent] = useState<AlertEvent | null>(null);
  const [loading, setLoading] = useState(true);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [sending, setSending] = useState(false);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);

  const selectedEvent = useMemo(
    () => events.find((event) => event.id === selectedEventId) || null,
    [events, selectedEventId]
  );

  const activeAlertsCount = useMemo(
    () => events.filter((e) => eventStatus(e) === "active").length,
    [events]
  );

  const refresh = useCallback(async (nextProjectId?: string) => {
    setLoading(true);
    setError(null);
    try {
      const loadedProjects = projects.length ? projects : await fetchProjects();
      const requestedFromUrl = typeof window === "undefined"
        ? ""
        : new URLSearchParams(window.location.search).get("project_id") || "";
      const savedProjectId = typeof window === "undefined"
        ? ""
        : window.localStorage.getItem("minded-active-project-id") || "";
      const requestedProject = nextProjectId
        || requestedFromUrl
        || (loadedProjects.some((project) => project.id === savedProjectId) ? savedProjectId : "")
        || (loadedProjects.some((project) => project.id === projectId) ? projectId : "")
        || loadedProjects[0]?.id
        || "";

      if (!requestedProject) {
        setProjects(loadedProjects);
        setProjectId("");
        setRules([]);
        setEvents([]);
        setDatasets([]);
        setMonitoring(null);
        setSelectedEventId(null);
        return;
      }

      const [loadedRules, loadedEvents, loadedMonitoring, loadedDatasets] = await Promise.all([
        fetchAlertRules(requestedProject),
        fetchAlertEvents(undefined, requestedProject),
        fetchMonitoringStatus(requestedProject),
        fetchDatasets(requestedProject).catch(() => []),
      ]);

      setProjects(loadedProjects);
      setProjectId(requestedProject);
      setRules(loadedRules);
      setEvents(loadedEvents);
      setMonitoring(loadedMonitoring);
      setDatasets(loadedDatasets);

      const eventIdFromUrl = typeof window === "undefined"
        ? ""
        : new URLSearchParams(window.location.search).get("event_id") || "";
      const preferredEventId = loadedEvents.some((event) => event.id === eventIdFromUrl)
        ? eventIdFromUrl
        : null;

      setSelectedEventId((current) =>
        preferredEventId
        || (current && loadedEvents.some((event) => event.id === current) ? current : null)
        || loadedEvents[0]?.id
        || null
      );

      try {
        window.localStorage.setItem("minded-active-project-id", requestedProject);
      } catch (storageError) {
        console.warn("Could not persist the active monitoring workspace.", storageError);
      }
    } catch (loadError) {
      console.error("Failed to load persisted monitoring data.", loadError);
      setRules([]);
      setEvents([]);
      setDatasets([]);
      setMonitoring(null);
      setError(loadError instanceof Error ? loadError.message : "Monitoring data is unavailable.");
    } finally {
      setLoading(false);
    }
  }, [projectId, projects.length]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const runEvaluation = async () => {
    if (!projectId || evaluating) return;
    setEvaluating(true);
    setActionError(null);
    try {
      const result = await evaluateAlertRules(projectId);
      await refresh(projectId);
      if (result.status === "failed") {
        setActionError("Evaluation completed with one or more rule failures. See each rule's last evaluation status.");
      } else if (result.status === "qualified") {
        setActionError("Evaluation was recorded, but one or more rules could not be compared because a baseline, profile, schema, or required metric was unavailable.");
      }
    } catch (evaluationError) {
      setActionError(evaluationError instanceof Error ? evaluationError.message : "Alert evaluation failed.");
    } finally {
      setEvaluating(false);
    }
  };

  const transition = async (event: AlertEvent, action: "acknowledge" | "resolve" | "snooze") => {
    setActionError(null);
    try {
      await transitionAlertEvent(event.id, action, action === "snooze" ? 60 : undefined);
      await refresh(projectId);
      if (inspectModalEvent?.id === event.id) {
        setInspectModalEvent((prev) => prev ? {
          ...prev,
          lifecycle_status: action === "resolve" ? "resolved" : action === "acknowledge" ? "acknowledged" : "snoozed",
          is_resolved: action === "resolve" ? true : prev.is_resolved,
        } : null);
      }
    } catch (transitionError) {
      setActionError(transitionError instanceof Error ? transitionError.message : "Could not update alert status.");
    }
  };

  const startInvestigationWithPrompt = async (targetEvent: AlertEvent, promptText: string) => {
    if (!promptText.trim() || !projectId || sending) return;
    if (
      !targetEvent.source_available ||
      !targetEvent.dataset_id ||
      !targetEvent.dataset_version_id
    ) {
      setActionError("The exact source snapshot for this historical alert is unavailable. Restore it before starting an investigation.");
      return;
    }

    setSelectedEventId(targetEvent.id);
    setSending(true);
    setActionError(null);
    setQuestion("");
    setMessages((previous) => [...previous, {
      eventId: targetEvent.id,
      role: "user",
      text: promptText,
    }]);

    try {
      const created = await createInvestigation({
        question: promptText,
        project_id: projectId,
        dataset_ids: [targetEvent.dataset_id],
        monitoring_event_id: targetEvent.id,
      });

      if (created.status === "NEEDS_CLARIFICATION") {
        setMessages((previous) => [...previous, {
          eventId: targetEvent.id,
          role: "assistant",
          text: created.resolution?.reason
            || "This alert follow-up needs clarification before it can run. Open the investigation in Analyst to resolve the requested measure or scope.",
          status: created.status,
          investigationId: created.investigation_id,
        }]);
        return;
      }
      if (created.status === "READY") {
        setMessages((previous) => [...previous, {
          eventId: targetEvent.id,
          role: "assistant",
          text: created.dispatch_error
            || "The alert follow-up is ready, but its worker dispatch failed. Retry from the investigation details.",
          status: created.status,
          investigationId: created.investigation_id,
        }]);
        return;
      }

      setMessages((previous) => [...previous, {
        eventId: targetEvent.id,
        role: "assistant",
        text: "Investigation started with the selected alert evidence and exact dataset version scope...",
        status: created.status,
        investigationId: created.investigation_id,
      }]);

      let latest: Investigation | null = null;
      for (let attempt = 0; attempt < 60; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 1500));
        latest = await fetchInvestigation(created.investigation_id);
        if (["COMPLETED", "PARTIAL", "FAILED", "CANCELLED"].includes(String(latest.status).toUpperCase())) break;
      }

      if (!latest) throw new Error("The investigation did not return a status.");
      const directAnswer = typeof latest.direct_answer === "string" ? latest.direct_answer : "";
      const errorMessage = typeof latest.error_message === "string" ? latest.error_message : "";
      const resultText = directAnswer
        || errorMessage
        || `Investigation concluded with status ${latest.status}; open History to inspect the full trace and proof.`;

      setMessages((previous) => previous.map((message) =>
        message.investigationId === created.investigation_id
          ? { ...message, text: resultText, status: String(latest?.status || message.status) }
          : message
      ));
    } catch (investigationError) {
      const message = investigationError instanceof Error ? investigationError.message : "Could not investigate this alert.";
      setActionError(message);
      setMessages((previous) => [...previous, {
        eventId: targetEvent.id,
        role: "assistant",
        text: message,
        status: "FAILED",
      }]);
    } finally {
      setSending(false);
    }
  };

  const handleAskSubmit = (formEvent: FormEvent) => {
    formEvent.preventDefault();
    if (!selectedEvent) return;
    void startInvestigationWithPrompt(selectedEvent, question);
  };

  const handleQuickPromptClick = (starterPrompt: string) => {
    if (!selectedEvent) return;
    setQuestion(starterPrompt);
  };

  const visibleMessages = selectedEvent
    ? messages.filter((message) => message.eventId === selectedEvent.id)
    : [];

  const starterPrompts = [
    "Why did this metric deviate?",
    "Which segments contributed most to this change?",
    "Is this caused by data quality or a business change?",
    "Compare with previous period baseline distribution",
  ];

  return (
    <div className="space-y-6 pb-12">
      {/* Header & Workspace Switcher */}
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-[var(--surface-border)] pb-5">
        <div>
          <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-[var(--accent)]">
            <Sparkles className="h-4 w-4" />
            <span>Monitoring Assistant</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--foreground)]">
            Alerts &amp; Continuous Monitoring
          </h1>
          <p className="mt-1 text-xs text-[var(--muted)]">
            Local data · Automatic profiling · Evidence-backed alerts
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {projects.length > 1 && (
            <select
              aria-label="Monitoring workspace"
              value={projectId}
              onChange={(event) => void refresh(event.target.value)}
              className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)] px-3 py-2 text-xs text-[var(--foreground)] focus:border-[var(--accent)] focus:outline-none"
            >
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          )}

          <button
            type="button"
            onClick={() => void refresh(projectId)}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)] px-3.5 py-2 text-xs font-medium text-[var(--foreground)] hover:border-[var(--accent)] disabled:opacity-50 transition-colors"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>

          <button
            type="button"
            onClick={() => void runEvaluation()}
            disabled={!projectId || evaluating || loading}
            className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-[var(--accent)] to-[var(--accent-strong)] px-4 py-2 text-xs font-semibold text-black shadow-sm hover:opacity-95 disabled:opacity-50 transition-opacity"
          >
            {evaluating ? (
              <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <ShieldCheck className="h-3.5 w-3.5" />
            )}
            {evaluating ? "Evaluating..." : "Evaluate rules now"}
          </button>
        </div>
      </header>

      {/* KPI Summary Tiles: Local | Active alerts | Data sources | Last evaluation */}
      <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {/* KPI 1: Local / Engine Status */}
        <div className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface-1)] p-4 shadow-sm backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">
              Status
            </span>
            <span className="flex h-2 w-2 rounded-full bg-emerald-400 ring-4 ring-emerald-400/20" />
          </div>
          <div className="mt-2 text-xl font-bold tracking-tight text-[var(--foreground)]">
            Local
          </div>
          <div className="mt-1 text-[11px] text-[var(--muted)]">
            {monitoring?.scheduler_active ? "Continuous scheduler" : "Import & manual triggers"}
          </div>
        </div>

        {/* KPI 2: Active Alerts */}
        <div className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface-1)] p-4 shadow-sm backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">
              Active alerts
            </span>
            <Flame className="h-3.5 w-3.5 text-[var(--accent)]" />
          </div>
          <div className="mt-2 text-xl font-bold tracking-tight text-[var(--foreground)]">
            {loading ? "..." : activeAlertsCount}
          </div>
          <div className="mt-1 text-[11px] text-[var(--muted)]">
            {events.length} total persisted events
          </div>
        </div>

        {/* KPI 3: Monitored Data sources */}
        <div className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface-1)] p-4 shadow-sm backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">
              Data sources
            </span>
            <Database className="h-3.5 w-3.5 text-[var(--muted)]" />
          </div>
          <div className="mt-2 text-xl font-bold tracking-tight text-[var(--foreground)]">
            {loading ? "..." : datasets.length || (rules.length > 0 ? new Set(rules.map((r) => r.dataset_id).filter(Boolean)).size : 0)}
          </div>
          <div className="mt-1 text-[11px] text-[var(--muted)]">
            {rules.length} configured alert rule{rules.length === 1 ? "" : "s"}
          </div>
        </div>

        {/* KPI 4: Last evaluation */}
        <div className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface-1)] p-4 shadow-sm backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">
              Last evaluation
            </span>
            <Clock3 className="h-3.5 w-3.5 text-[var(--muted)]" />
          </div>
          <div className="mt-2 text-xl font-bold tracking-tight text-[var(--foreground)]">
            {loading ? "..." : formatRelativeTime(monitoring?.last_evaluation_at)}
          </div>
          <div className="mt-1 truncate text-[11px] text-[var(--muted)]" title={monitoring?.last_evaluation_status || "No evaluation yet"}>
            {monitoring?.evaluation_mode ? `${monitoring.evaluation_mode} · ` : ""}
            {monitoring?.last_evaluation_status || "No runs recorded"}
          </div>
        </div>
      </section>

      {/* Error Banners */}
      {error && (
        <div role="alert" className="rounded-xl border border-rose-500/40 bg-rose-950/30 p-4 text-xs text-rose-200">
          <div className="flex items-center gap-2 font-semibold">
            <AlertCircle className="h-4 w-4" />
            <span>Monitoring data unavailable</span>
          </div>
          <p className="mt-1 text-slate-300">{error}</p>
        </div>
      )}

      {actionError && (
        <div role="alert" className="rounded-xl border border-amber-500/40 bg-amber-950/30 p-3 text-xs text-amber-200 flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
          <div className="flex-1">{actionError}</div>
          <button
            type="button"
            onClick={() => setActionError(null)}
            className="text-amber-400 hover:text-amber-200 text-xs"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* Two Column Layout: Left = Alert Timeline & Configured Rules; Right = Conversational Assistant */}
      <section className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.25fr)_minmax(380px,0.75fr)]">
        {/* Left Column: Live Persisted Alert Cards */}
        <div className="space-y-6">
          <div className="rounded-2xl border border-[var(--surface-border)] bg-[var(--surface-1)] p-5 shadow-sm">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--surface-border)] pb-3">
              <div>
                <h2 className="text-sm font-semibold text-[var(--foreground)]">
                  Persisted Monitoring Events
                </h2>
                <p className="text-[11px] text-[var(--muted)]">
                  Genuine anomalies detected by version-to-version profiling comparisons.
                </p>
              </div>
              <span className="rounded-full border border-[var(--surface-border)] bg-[var(--surface-2)] px-2.5 py-0.5 text-[11px] text-[var(--muted)] font-mono">
                {loading ? "loading..." : `${events.length} event${events.length === 1 ? "" : "s"}`}
              </span>
            </div>

            {loading ? (
              <div role="status" className="flex items-center justify-center p-8 text-xs text-[var(--muted)]">
                <LoaderCircle className="h-4 w-4 animate-spin mr-2" />
                Loading persisted alerts from local store...
              </div>
            ) : error ? null : events.length === 0 ? (
              <div className="rounded-xl border border-dashed border-[var(--surface-border)] p-8 text-center bg-[var(--surface-2)]/30">
                <Bell className="mx-auto h-7 w-7 text-[var(--muted-2)]" />
                <p className="mt-3 text-sm font-semibold text-[var(--foreground)]">
                  No alerts detected
                </p>
                <p className="mt-1 text-xs text-[var(--muted)] max-w-md mx-auto">
                  All metrics in your local datasets are currently within configured thresholds, or baseline versions are still establishing.
                </p>
              </div>
            ) : (
              <div className="space-y-3.5">
                {events.map((event) => {
                  const status = eventStatus(event);
                  const isSelected = event.id === selectedEventId;
                  const sevStyle = severityClasses(event.severity);

                  return (
                    <article
                      key={event.id}
                      onClick={() => setSelectedEventId(event.id)}
                      className={`cursor-pointer rounded-xl border p-4.5 transition-all duration-150 ${
                        isSelected
                          ? "border-[var(--accent)] bg-[var(--surface-2)] shadow-[var(--panel-glow)]"
                          : "border-[var(--surface-border)] bg-[var(--surface-1)] hover:border-[var(--muted-2)] hover:bg-[var(--surface-2)]/50"
                      }`}
                    >
                      {/* Top Header of Card */}
                      <div className="flex flex-wrap items-start justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className={`rounded-md border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${sevStyle.badge}`}>
                            {event.severity || "Standard"}
                          </span>
                          <span className={`rounded-full border px-2.5 py-0.5 text-[10px] capitalize ${statusClasses(status)}`}>
                            {status}
                          </span>
                        </div>
                        <time className="inline-flex items-center gap-1 text-[11px] text-[var(--muted-2)]">
                          <Clock3 className="h-3 w-3" />
                          {event.triggered_at ? new Date(event.triggered_at).toLocaleString() : "Unrecorded"}
                        </time>
                      </div>

                      {/* Title & Description */}
                      <h3 className="mt-2 text-sm font-semibold text-[var(--foreground)] flex items-center gap-1.5">
                        {event.title}
                      </h3>
                      <p className="mt-1 text-xs leading-relaxed text-[var(--muted)]">
                        {event.description}
                      </p>

                      {/* Metric Comparison Strip */}
                      <div className="mt-3 grid grid-cols-2 gap-2 rounded-lg border border-[var(--surface-border)] bg-[var(--surface-2)]/60 p-2.5 text-[11px] sm:grid-cols-4">
                        <div>
                          <div className="text-[10px] text-[var(--muted-2)]">Metric</div>
                          <div className="font-semibold text-[var(--foreground)] truncate">
                            {event.metric_name || "Unspecified"}
                          </div>
                        </div>
                        <div>
                          <div className="text-[10px] text-[var(--muted-2)]">Observed</div>
                          <div className="font-semibold text-[var(--foreground)]">
                            {fmt(event.current_value)}
                          </div>
                        </div>
                        <div>
                          <div className="text-[10px] text-[var(--muted-2)]">Baseline</div>
                          <div className="font-semibold text-[var(--muted)]">
                            {fmt(event.expected_value)}
                          </div>
                        </div>
                        <div>
                          <div className="text-[10px] text-[var(--muted-2)]">Deviation</div>
                          <div className={`font-semibold ${event.deviation_percentage && event.deviation_percentage > 0 ? "text-rose-400" : "text-sky-400"}`}>
                            {event.deviation_percentage == null ? "N/A" : `${fmt(event.deviation_percentage)}%`}
                          </div>
                        </div>
                      </div>

                      {/* Source Metadata & Snapshot */}
                      <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 text-[10px] text-[var(--muted-2)]">
                        <div>
                          <span className="font-medium text-[var(--muted)]">{event.dataset_name || "Dataset"}</span>
                          {" · "}
                          <span>{event.dataset_version == null ? "v?" : `v${event.dataset_version}`}</span>
                          {event.dataset_fingerprint && (
                            <span className="font-mono text-[9px] ml-1">
                              ({event.dataset_fingerprint.slice(0, 8)})
                            </span>
                          )}
                        </div>
                        {event.baseline_period && event.comparison_period && (
                          <div>
                            Window: {event.baseline_period} → {event.comparison_period}
                          </div>
                        )}
                      </div>

                      {/* Explicit Action Buttons */}
                      <div className="mt-3.5 flex flex-wrap items-center gap-2 border-t border-[var(--surface-border)] pt-3">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedEventId(event.id);
                            setInspectModalEvent(event);
                          }}
                          className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--surface-border)] bg-[var(--surface-2)] px-2.5 py-1 text-xs font-medium text-[var(--foreground)] hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors"
                        >
                          <Search className="h-3 w-3" />
                          Inspect alert
                        </button>

                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedEventId(event.id);
                            const defaultInvestigationPrompt = `Why did ${event.metric_name || "this metric"} change from ${fmt(event.expected_value)} to ${fmt(event.current_value)}?`;
                            void startInvestigationWithPrompt(event, defaultInvestigationPrompt);
                          }}
                          disabled={!event.source_available || sending}
                          className="inline-flex items-center gap-1.5 rounded-lg bg-[var(--accent-soft)] border border-[var(--accent)]/30 px-2.5 py-1 text-xs font-semibold text-[var(--accent)] hover:bg-[var(--accent)] hover:text-black disabled:opacity-50 transition-all"
                        >
                          <MessageSquareText className="h-3 w-3" />
                          Investigate
                        </button>

                        {/* Lifecycle quick actions */}
                        <div className="ml-auto flex items-center gap-1.5">
                          {status !== "resolved" && (
                            <>
                              {status !== "acknowledged" && (
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    void transition(event, "acknowledge");
                                  }}
                                  className="rounded-lg border border-[var(--surface-border)] px-2 py-1 text-[11px] text-[var(--muted)] hover:text-[var(--foreground)] hover:border-[var(--muted-2)]"
                                  title="Acknowledge this alert"
                                >
                                  Ack
                                </button>
                              )}
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  void transition(event, "resolve");
                                }}
                                className="inline-flex items-center gap-1 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-[11px] text-emerald-300 hover:bg-emerald-500/20"
                                title="Resolve this alert"
                              >
                                <Check className="h-3 w-3" /> Resolve
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </div>

          {/* Configured Alert Rules section */}
          <div className="rounded-2xl border border-[var(--surface-border)] bg-[var(--surface-1)] p-5 shadow-sm">
            <div className="mb-3 flex items-center justify-between border-b border-[var(--surface-border)] pb-2.5">
              <h2 className="text-sm font-semibold text-[var(--foreground)]">
                Configured Alert Rules
              </h2>
              <span className="text-[10px] text-[var(--muted)]">Local Rule Store</span>
            </div>
            {rules.length === 0 ? (
              <p className="rounded-xl border border-dashed border-[var(--surface-border)] p-4 text-xs text-[var(--muted)]">
                No alert rules configured in this workspace yet. Ingestion will create standard quality baseline checks.
              </p>
            ) : (
              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                {rules.map((rule) => (
                  <article key={rule.id} className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)]/60 p-3">
                    <div className="flex items-start justify-between gap-1.5">
                      <h3 className="text-xs font-semibold text-[var(--foreground)] truncate">{rule.name}</h3>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded border ${rule.is_active ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400" : "border-slate-700 bg-slate-800 text-slate-400"}`}>
                        {rule.is_active ? "active" : "disabled"}
                      </span>
                    </div>
                    <p className="mt-1 text-[10px] text-[var(--muted)]">
                      Metric: <span className="text-[var(--foreground)] font-medium">{rule.metric_name}</span> · Threshold: {rule.threshold_pct_change == null ? "None" : `${rule.threshold_pct_change}%`}
                    </p>
                    <p className="mt-1 text-[9px] text-[var(--muted-2)]">
                      Last evaluated: {rule.last_evaluated_at ? formatRelativeTime(rule.last_evaluated_at) : "Never"}
                      {rule.last_evaluation_status ? ` (${rule.last_evaluation_status})` : ""}
                    </p>
                  </article>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Conversational Assistant & Investigation Panel */}
        <aside className="space-y-4">
          <div className="rounded-2xl border border-[var(--surface-border)] bg-[var(--surface-1)] p-5 shadow-sm sticky top-4">
            <div className="mb-3.5 flex items-center justify-between border-b border-[var(--surface-border)] pb-3">
              <div className="flex items-center gap-2">
                <MessageSquareText className="h-4 w-4 text-[var(--accent)]" />
                <h2 className="text-sm font-semibold text-[var(--foreground)]">
                  Monitoring Assistant
                </h2>
              </div>
              {selectedEvent && (
                <span className="text-[10px] text-[var(--muted)] font-mono">
                  Event: {selectedEvent.id.slice(0, 8)}...
                </span>
              )}
            </div>

            {!selectedEvent ? (
              <div className="rounded-xl border border-dashed border-[var(--surface-border)] p-6 text-center bg-[var(--surface-2)]/30">
                <HelpCircle className="mx-auto h-6 w-6 text-[var(--muted-2)]" />
                <p className="mt-2 text-xs font-semibold text-[var(--foreground)]">
                  No alert selected
                </p>
                <p className="mt-1 text-[11px] text-[var(--muted)]">
                  Click on an alert card or its Investigate button to ask follow-up questions grounded in that event's evidence.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Active Selected Alert Context Pill */}
                <div className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)] p-3">
                  <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-[var(--muted-2)]">
                    <span>Investigating Scope</span>
                    <span className="font-semibold text-[var(--accent)]">{selectedEvent.metric_name}</span>
                  </div>
                  <h3 className="mt-1 text-xs font-semibold text-[var(--foreground)]">
                    {selectedEvent.title}
                  </h3>
                  <div className="mt-1 text-[10px] text-[var(--muted)]">
                    Baseline: {fmt(selectedEvent.expected_value)} → Observed: {fmt(selectedEvent.current_value)} ({selectedEvent.deviation_percentage ? `${fmt(selectedEvent.deviation_percentage)}%` : "N/A"})
                  </div>

                  {selectedEvent.source_available && selectedEvent.dataset_id && (
                    <div className="mt-2.5 pt-2 border-t border-[var(--surface-border)]/60 flex items-center justify-between text-[10px]">
                      <span className="text-[var(--muted)]">
                        {selectedEvent.dataset_name || "Dataset"} (v{selectedEvent.dataset_version})
                      </span>
                      <Link
                        href={`/datasets/?project_id=${encodeURIComponent(projectId)}&dataset_id=${encodeURIComponent(selectedEvent.dataset_id)}&version=${encodeURIComponent(String(selectedEvent.dataset_version || ""))}&event_id=${encodeURIComponent(selectedEvent.id)}`}
                        className="inline-flex items-center gap-1 font-semibold text-[var(--accent)] hover:underline"
                      >
                        <Database className="h-2.5 w-2.5" />
                        Open dataset profile <ExternalLink className="h-2.5 w-2.5" />
                      </Link>
                    </div>
                  )}

                  {!selectedEvent.source_available && (
                    <div className="mt-2 rounded bg-amber-500/10 p-2 text-[10px] text-amber-200">
                      Original source version snapshot is unavailable. Historical record is preserved.
                    </div>
                  )}
                </div>

                {/* Conversation Timeline */}
                <div aria-live="polite" className="max-h-64 space-y-2.5 overflow-y-auto pr-1">
                  {visibleMessages.length === 0 ? (
                    <div className="rounded-xl border border-dashed border-[var(--surface-border)] p-4 text-center text-xs text-[var(--muted)]">
                      Ask a question below or choose a starter topic to investigate this alert.
                    </div>
                  ) : (
                    visibleMessages.map((message, idx) => (
                      <div
                        key={`${message.eventId}:${idx}`}
                        className={`rounded-xl p-3 text-xs leading-relaxed ${
                          message.role === "user"
                            ? "ml-6 bg-[var(--surface-2)] text-[var(--foreground)] border border-[var(--surface-border)]"
                            : "mr-3 bg-[var(--surface-3)] text-[var(--foreground)] border border-[var(--accent)]/30 shadow-sm"
                        }`}
                      >
                        <div className="mb-1 flex items-center justify-between text-[9px] uppercase tracking-wider text-[var(--muted-2)]">
                          <span>{message.role === "user" ? "You" : "Local Analyst Engine"}</span>
                          {message.status && <span>{message.status}</span>}
                        </div>
                        <p className="whitespace-pre-wrap">{message.text}</p>
                        {message.investigationId && (
                          <div className="mt-2 pt-1 border-t border-[var(--surface-border)]/50 flex justify-end">
                            <Link
                              href={`/history/?investigation_id=${encodeURIComponent(message.investigationId)}`}
                              className="inline-flex items-center gap-1 text-[10px] font-semibold text-[var(--accent)] hover:underline"
                            >
                              Open full analysis log <ArrowRight className="h-2.5 w-2.5" />
                            </Link>
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>

                {/* Quick Prompts / Starter Chips */}
                <div className="space-y-1.5">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">
                    Suggested Questions
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {starterPrompts.map((promptText, i) => (
                      <button
                        key={i}
                        type="button"
                        onClick={() => handleQuickPromptClick(promptText)}
                        disabled={sending || !selectedEvent.source_available}
                        className="rounded-lg border border-[var(--surface-border)] bg-[var(--surface-2)] px-2.5 py-1 text-[11px] text-[var(--muted)] hover:border-[var(--accent)] hover:text-[var(--foreground)] transition-colors text-left disabled:opacity-50"
                      >
                        {promptText}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Composer Form */}
                <form onSubmit={handleAskSubmit} className="space-y-2 pt-2">
                  <textarea
                    id="alert-follow-up"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    disabled={sending || !selectedEvent.source_available}
                    placeholder="Ask about this alert (e.g., Why did revenue drop? Which products contributed most?)"
                    rows={3}
                    className="w-full resize-none rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)] p-3 text-xs text-[var(--foreground)] placeholder:text-[var(--muted-2)] focus:border-[var(--accent)] focus:outline-none disabled:opacity-50"
                  />
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[10px] text-[var(--muted-2)]">
                      Scoped to {selectedEvent.dataset_name || "dataset"}
                    </span>
                    <button
                      type="submit"
                      disabled={sending || !question.trim() || !selectedEvent.source_available}
                      className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-[var(--accent)] to-[var(--accent-strong)] px-4 py-2 text-xs font-semibold text-black shadow-sm hover:opacity-95 disabled:cursor-not-allowed disabled:opacity-50 transition-opacity"
                    >
                      {sending ? (
                        <>
                          <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
                          Investigating...
                        </>
                      ) : (
                        <>
                          <MessageSquareText className="h-3.5 w-3.5" />
                          Ask
                        </>
                      )}
                    </button>
                  </div>
                </form>
              </div>
            )}
          </div>

          {/* Engine guarantee note */}
          <div className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface-1)] p-4 text-[11px] leading-relaxed text-[var(--muted)]">
            <div className="font-semibold text-[var(--foreground)] flex items-center gap-1.5 mb-1">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
              Local-first Architecture
            </div>
            <p>
              Alerts evaluate persisted SQLite records against local DuckDB tables. Zero external cloud AI or telemetry required.
            </p>
          </div>
        </aside>
      </section>

      {/* Inspect Alert Modal / Drawer */}
      {inspectModalEvent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm">
          <div className="relative w-full max-w-2xl rounded-2xl border border-[var(--surface-border)] bg-[var(--surface-1)] p-6 shadow-2xl">
            <div className="flex items-start justify-between border-b border-[var(--surface-border)] pb-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className={`rounded px-2 py-0.5 text-[10px] font-semibold uppercase ${severityClasses(inspectModalEvent.severity).badge}`}>
                    {inspectModalEvent.severity || "Standard"}
                  </span>
                  <span className={`rounded-full px-2 py-0.5 text-[10px] capitalize ${statusClasses(eventStatus(inspectModalEvent))}`}>
                    {eventStatus(inspectModalEvent)}
                  </span>
                </div>
                <h3 className="mt-2 text-base font-bold text-[var(--foreground)]">
                  {inspectModalEvent.title}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setInspectModalEvent(null)}
                className="rounded-lg p-1.5 text-[var(--muted)] hover:bg-[var(--surface-2)] hover:text-[var(--foreground)]"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="mt-4 max-h-[70vh] space-y-4 overflow-y-auto pr-1 text-xs">
              <div>
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">
                  Description &amp; Summary
                </h4>
                <p className="mt-1 leading-relaxed text-[var(--foreground)]">
                  {inspectModalEvent.description}
                </p>
                {inspectModalEvent.evidence_summary && (
                  <p className="mt-2 rounded-lg bg-[var(--surface-2)] p-2.5 text-[11px] text-[var(--muted)]">
                    <strong className="text-[var(--foreground)]">Evidence:</strong> {inspectModalEvent.evidence_summary}
                  </p>
                )}
              </div>

              {/* Metric Breakdown Table */}
              <div>
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">
                  Metric Measurement
                </h4>
                <div className="mt-1.5 grid grid-cols-2 gap-3 sm:grid-cols-4 rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)] p-3">
                  <div>
                    <span className="text-[10px] text-[var(--muted-2)]">Metric</span>
                    <p className="font-semibold text-[var(--foreground)]">{inspectModalEvent.metric_name || "N/A"}</p>
                  </div>
                  <div>
                    <span className="text-[10px] text-[var(--muted-2)]">Observed Value</span>
                    <p className="font-semibold text-[var(--foreground)]">{fmt(inspectModalEvent.current_value)}</p>
                  </div>
                  <div>
                    <span className="text-[10px] text-[var(--muted-2)]">Expected Baseline</span>
                    <p className="font-semibold text-[var(--muted)]">{fmt(inspectModalEvent.expected_value)}</p>
                  </div>
                  <div>
                    <span className="text-[10px] text-[var(--muted-2)]">Deviation</span>
                    <p className="font-semibold text-rose-400">
                      {inspectModalEvent.deviation_percentage == null ? "N/A" : `${fmt(inspectModalEvent.deviation_percentage)}%`}
                    </p>
                  </div>
                </div>
              </div>

              {/* Source & Version Provenance */}
              <div>
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">
                  Dataset Provenance
                </h4>
                <div className="mt-1.5 space-y-1.5 rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)] p-3 text-[11px]">
                  <div className="flex justify-between">
                    <span className="text-[var(--muted-2)]">Dataset:</span>
                    <span className="font-medium text-[var(--foreground)]">{inspectModalEvent.dataset_name || inspectModalEvent.dataset_id || "Unspecified"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[var(--muted-2)]">Version Number:</span>
                    <span className="font-mono text-[var(--foreground)]">{inspectModalEvent.dataset_version == null ? "Not recorded" : `v${inspectModalEvent.dataset_version}`}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[var(--muted-2)]">Content Fingerprint:</span>
                    <span className="font-mono text-[var(--muted)]">{inspectModalEvent.dataset_fingerprint || "None recorded"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[var(--muted-2)]">Comparison Window:</span>
                    <span className="text-[var(--muted)]">{inspectModalEvent.baseline_period || "Baseline"} → {inspectModalEvent.comparison_period || "Current"}</span>
                  </div>
                </div>
              </div>

              {/* Affected Dimensions JSON */}
              {inspectModalEvent.affected_dimensions && Object.keys(inspectModalEvent.affected_dimensions).length > 0 && (
                <div>
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-2)]">
                    Recorded Dimensions &amp; Context
                  </h4>
                  <pre className="mt-1.5 max-h-40 overflow-auto rounded-xl border border-[var(--surface-border)] bg-[var(--surface-2)] p-3 font-mono text-[10px] text-[var(--muted)]">
                    {JSON.stringify(inspectModalEvent.affected_dimensions, null, 2)}
                  </pre>
                </div>
              )}
            </div>

            {/* Modal Actions */}
            <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-[var(--surface-border)] pt-4">
              <div className="flex items-center gap-2">
                {eventStatus(inspectModalEvent) !== "resolved" && (
                  <>
                    <button
                      type="button"
                      onClick={() => void transition(inspectModalEvent, "acknowledge")}
                      className="rounded-lg border border-[var(--surface-border)] bg-[var(--surface-2)] px-3 py-1.5 text-xs text-[var(--foreground)] hover:border-[var(--muted-2)]"
                    >
                      Acknowledge
                    </button>
                    <button
                      type="button"
                      onClick={() => void transition(inspectModalEvent, "resolve")}
                      className="rounded-lg bg-emerald-500/20 border border-emerald-500/40 px-3 py-1.5 text-xs font-medium text-emerald-300 hover:bg-emerald-500/30"
                    >
                      Resolve alert
                    </button>
                  </>
                )}
              </div>

              <div className="flex items-center gap-2">
                {inspectModalEvent.source_available && inspectModalEvent.dataset_id && (
                  <Link
                    href={`/datasets/?project_id=${encodeURIComponent(projectId)}&dataset_id=${encodeURIComponent(inspectModalEvent.dataset_id)}&version=${encodeURIComponent(String(inspectModalEvent.dataset_version || ""))}&event_id=${encodeURIComponent(inspectModalEvent.id)}`}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--surface-border)] bg-[var(--surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--foreground)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
                  >
                    <Database className="h-3.5 w-3.5" />
                    Open dataset profile
                  </Link>
                )}
                <button
                  type="button"
                  onClick={() => {
                    const evt = inspectModalEvent;
                    setInspectModalEvent(null);
                    setSelectedEventId(evt.id);
                    const defaultPrompt = `Why did ${evt.metric_name || "this metric"} deviate?`;
                    void startInvestigationWithPrompt(evt, defaultPrompt);
                  }}
                  disabled={!inspectModalEvent.source_available || sending}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-[var(--accent)] to-[var(--accent-strong)] px-3.5 py-1.5 text-xs font-semibold text-black shadow-sm hover:opacity-95 disabled:opacity-50"
                >
                  <MessageSquareText className="h-3.5 w-3.5" />
                  Start conversation
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
