"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { AlertCircle, Bell, Clock3, RefreshCw, X } from "lucide-react";
import { fetchAlertEvents, fetchProjects } from "../lib/api";
import { AlertEvent } from "../types";

export default function NotificationCenter() {
  const [isOpen, setIsOpen] = useState(false);
  const [events, setEvents] = useState<AlertEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const loadEvents = async () => {
    setLoading(true);
    setError(null);
    try {
      const projects = await fetchProjects();
      const projectId = typeof window !== "undefined"
        ? window.localStorage.getItem("minded-active-project-id")
        : null;
      const project = projects.find((item) => item.id === projectId) || projects[0];
      setEvents(project ? await fetchAlertEvents(undefined, project.id) : []);
    } catch (cause) {
      console.error("Unable to load persisted monitoring notifications.", cause);
      setEvents([]);
      setError(cause instanceof Error ? cause.message : "Monitoring events are unavailable.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) void loadEvents();
  }, [isOpen]);

  const openEvent = (event: AlertEvent) => {
    setIsOpen(false);
    router.push(`/alerts?project_id=${encodeURIComponent(event.project_id || "")}&event_id=${encodeURIComponent(event.id)}`);
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setIsOpen((open) => !open)}
        className="relative rounded-xl border border-slate-800 bg-slate-900/90 p-2 text-slate-400 transition-colors hover:bg-slate-800 hover:text-white"
        title="Persisted monitoring events"
        aria-label="Open persisted monitoring events"
      >
        <Bell className="h-4 w-4" />
        {events.length > 0 && <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-cyan-400" />}
      </button>

      {isOpen && (
        <div className="absolute right-0 z-50 mt-2 flex max-h-[min(80vh,36rem)] w-[min(24rem,calc(100vw-2rem))] flex-col overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 shadow-2xl">
          <div className="flex items-center justify-between border-b border-slate-800 bg-slate-950 p-4">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-white">Monitoring events</h3>
              <p className="mt-1 text-[10px] text-slate-500">Persisted local records only</p>
            </div>
            <div className="flex items-center gap-2">
              <button type="button" onClick={() => void loadEvents()} disabled={loading} aria-label="Refresh events" className="text-slate-500 hover:text-cyan-300 disabled:opacity-40">
                <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
              </button>
              <button type="button" onClick={() => setIsOpen(false)} aria-label="Close events" className="text-slate-500 hover:text-white">
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          <div className="max-h-96 space-y-2 overflow-y-auto p-3">
            {loading ? (
              <div role="status" className="p-5 text-center text-xs text-slate-400">Loading persisted events…</div>
            ) : error ? (
              <div role="alert" className="rounded-xl border border-rose-500/30 bg-rose-950/30 p-3 text-xs text-rose-200">
                <div className="flex items-center gap-2 font-semibold"><AlertCircle className="h-3.5 w-3.5" /> Events unavailable</div>
                <p className="mt-1">{error}</p>
              </div>
            ) : events.length === 0 ? (
              <div className="p-5 text-center text-xs text-slate-400">
                No persisted monitoring events.
              </div>
            ) : events.map((event) => (
              <button
                type="button"
                key={event.id}
                onClick={() => openEvent(event)}
                className="w-full rounded-xl border border-slate-800 bg-slate-950/60 p-3 text-left transition-colors hover:border-cyan-500/40"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="rounded-full border border-slate-700 px-2 py-0.5 text-[9px] uppercase text-slate-300">
                    {event.severity || "severity not recorded"} · {event.lifecycle_status || "status not recorded"}
                  </span>
                  <time className="inline-flex shrink-0 items-center gap-1 text-[9px] text-slate-500">
                    <Clock3 className="h-3 w-3" />
                    {event.triggered_at ? new Date(event.triggered_at).toLocaleString() : "Time not recorded"}
                  </time>
                </div>
                <div className="mt-2 text-xs font-semibold text-white">{event.title}</div>
                <p className="mt-1 line-clamp-2 text-[10px] leading-relaxed text-slate-400">{event.description}</p>
                <p className="mt-2 text-[9px] text-cyan-300">Open recorded evidence →</p>
              </button>
            ))}
          </div>

          <div className="flex items-center justify-between border-t border-slate-800 bg-slate-950 p-3 text-[10px] text-slate-500">
            <span>Local evaluation · no scheduler active</span>
            <button
              type="button"
              onClick={() => {
                setIsOpen(false);
                router.push("/alerts");
              }}
              className="font-medium text-cyan-400 hover:underline"
            >
              Monitoring workspace →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
