"use client";

import React, { useCallback, useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, Copy, HelpCircle, ShieldCheck, XCircle } from "lucide-react";
import { fetchVerification, recordVerificationDecision, signOffVerification } from "../lib/api";
import {
  VerificationDecision,
  VerificationItem,
  VerificationSignoffOutcome,
  VerificationState,
} from "../types";

const PRIORITY_STYLE: Record<string, string> = {
  BLOCKER: "bg-rose-500/10 text-rose-300 border-rose-500/30",
  REVIEW: "bg-amber-500/10 text-amber-300 border-amber-500/30",
  SPOT_CHECK: "bg-slate-800 text-slate-300 border-slate-700",
  INFO: "bg-slate-900 text-slate-400 border-slate-800",
};

const MACHINE_LABEL: Record<string, { text: string; style: string }> = {
  PASSED: { text: "Machine-checked: passed", style: "text-emerald-400" },
  FAILED: { text: "Machine-checked: failed", style: "text-rose-400" },
  WARNING: { text: "Machine-checked: warning", style: "text-amber-400" },
  NOT_RUN: { text: "Not tested (not a pass)", style: "text-amber-400" },
  NEEDS_HUMAN: { text: "Needs a human", style: "text-cyan-300" },
};

const STATE_LABEL: Record<string, string> = {
  NOT_STARTED: "Not started",
  IN_PROGRESS: "In progress",
  READY_TO_SIGN: "Ready to sign off",
  OBJECTIONS_RAISED: "You raised objections",
};

function ItemCard({
  item,
  state,
  busy,
  onDecide,
}: {
  item: VerificationItem;
  state: VerificationState;
  busy: boolean;
  onDecide: (item: VerificationItem, decision: VerificationDecision, comment: string) => void;
}) {
  const current = state.review.per_item[item.id];
  const [open, setOpen] = useState(item.priority === "BLOCKER" || item.priority === "REVIEW");
  const [comment, setComment] = useState("");
  const machine = MACHINE_LABEL[item.machine_status] || MACHINE_LABEL.NEEDS_HUMAN;
  const decided = current && !current.stale;
  const commentRequired = item.priority === "BLOCKER";

  return (
    <div className={`rounded-xl border p-4 space-y-3 bg-slate-950 ${decided ? "border-slate-800" : item.requires_decision ? "border-amber-500/30" : "border-slate-800"}`}>
      <button type="button" onClick={() => setOpen(!open)} className="w-full flex items-start justify-between gap-3 text-left">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${PRIORITY_STYLE[item.priority]}`}>{item.priority.replace("_", " ")}</span>
            <span className={`text-[10px] font-semibold ${machine.style}`}>{machine.text}</span>
            {decided && (
              <span className="text-[10px] font-semibold text-slate-300">
                • You: {current.decision.replace("_", " ")}
              </span>
            )}
            {current?.stale && <span className="text-[10px] font-semibold text-amber-400">• Your earlier decision is out of date</span>}
          </div>
          <div className="text-sm font-semibold text-white">{item.title}</div>
        </div>
        {open ? <ChevronUp className="w-4 h-4 text-slate-500 shrink-0 mt-1" /> : <ChevronDown className="w-4 h-4 text-slate-500 shrink-0 mt-1" />}
      </button>

      {open && (
        <div className="space-y-3 text-xs">
          <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-3 text-cyan-100">
            <div className="text-[10px] uppercase tracking-wide text-cyan-400 mb-1">You confirm</div>
            {item.you_confirm}
          </div>
          <div className="text-slate-400"><span className="text-slate-500">What the machine did: </span>{item.machine_did}</div>
          <div className="text-slate-400"><span className="text-slate-500">Why it matters: </span>{item.why_it_matters}</div>

          {item.reproduce?.sql && (
            <div className="rounded-lg border border-slate-800 bg-slate-900 p-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] uppercase tracking-wide text-slate-500">
                  Reproduce it yourself — table <code>{item.reproduce.table_alias || "data_table"}</code> = dataset {item.reproduce.dataset ? `'${item.reproduce.dataset}'` : "(see scope)"}
                </span>
                <button type="button" className="text-slate-400 hover:text-white" onClick={() => navigator.clipboard?.writeText(item.reproduce?.sql || "")} title="Copy SQL">
                  <Copy className="w-3.5 h-3.5" />
                </button>
              </div>
              <pre className="text-[11px] text-emerald-300 whitespace-pre-wrap break-all">{item.reproduce.sql}</pre>
            </div>
          )}

          {Object.keys(item.evidence || {}).length > 0 && (
            <details className="text-slate-400">
              <summary className="cursor-pointer text-slate-500 hover:text-slate-300">Evidence detail</summary>
              <pre className="mt-2 text-[11px] bg-slate-900 border border-slate-800 rounded-lg p-3 overflow-x-auto">{JSON.stringify(item.evidence, null, 2)}</pre>
            </details>
          )}

          {current?.comment && <div className="text-slate-400"><span className="text-slate-500">Your comment: </span>{current.comment}</div>}

          <div className="space-y-2 pt-1">
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder={commentRequired ? "Comment required for a blocker (min 10 characters)…" : "Comment (required if you reject or ask for rework)…"}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 placeholder-slate-600"
              rows={2}
            />
            <div className="flex flex-wrap gap-2">
              <button disabled={busy} onClick={() => { onDecide(item, "CONFIRMED", comment); setComment(""); }}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-50 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> {item.requires_decision ? "Confirm" : "Looks fine"}
              </button>
              <button disabled={busy} onClick={() => { onDecide(item, "REJECTED", comment); setComment(""); }}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-rose-500/40 text-rose-300 hover:bg-rose-500/10 disabled:opacity-50 flex items-center gap-1">
                <XCircle className="w-3.5 h-3.5" /> Reject
              </button>
              <button disabled={busy} onClick={() => { onDecide(item, "NEEDS_REWORK", comment); setComment(""); }}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-slate-600 text-slate-300 hover:bg-slate-800 disabled:opacity-50 flex items-center gap-1">
                <HelpCircle className="w-3.5 h-3.5" /> Needs rework
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function VerificationPanel({ investigationId }: { investigationId: string }) {
  const [state, setState] = useState<VerificationState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [signComment, setSignComment] = useState("");

  const load = useCallback(async () => {
    try {
      setError(null);
      setState(await fetchVerification(investigationId));
    } catch (e: any) {
      setError(e?.message || "Could not load the verification worklist.");
    }
  }, [investigationId]);

  useEffect(() => { load(); }, [load]);

  const onDecide = async (item: VerificationItem, decision: VerificationDecision, comment: string) => {
    setBusy(true); setError(null);
    try {
      setState(await recordVerificationDecision(investigationId, {
        item_id: item.id, decision, comment, item_fingerprint: item.fingerprint,
      }));
    } catch (e: any) {
      setError(e?.message || "Could not record your decision.");
    } finally { setBusy(false); }
  };

  const onSign = async (outcome: VerificationSignoffOutcome) => {
    if (!state) return;
    setBusy(true); setError(null);
    try {
      setState(await signOffVerification(investigationId, {
        outcome, comment: signComment, packet_fingerprint: state.packet.packet_fingerprint,
      }));
      setSignComment("");
    } catch (e: any) {
      setError(e?.message || "Could not record the sign-off.");
    } finally { setBusy(false); }
  };

  if (!state) {
    return <div className="text-xs text-slate-400">{error ? <span className="text-rose-400">{error}</span> : "Preparing your verification worklist…"}</div>;
  }

  const { packet, review, signoff } = state;
  const required = packet.items.filter((i) => i.requires_decision);
  const optional = packet.items.filter((i) => !i.requires_decision);
  const done = review.required_item_ids.length - review.pending.length;

  return (
    <div className="space-y-5">
      <div className="bg-slate-900 border border-cyan-500/20 rounded-2xl p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-sm font-bold text-white">
            <ShieldCheck className="w-4 h-4 text-cyan-400" /> Verify this result
          </div>
          <span className="text-[10px] font-mono px-2 py-1 rounded border border-slate-700 bg-slate-950 text-cyan-300">
            {STATE_LABEL[review.state] || review.state} • {done}/{review.required_item_ids.length} decided
          </span>
        </div>
        <div className="text-xs text-slate-300">
          <span className="text-slate-500">AA-OS concluded ({packet.machine_claim.verdict || "no verdict"}): </span>
          {packet.machine_claim.headline || "—"}
        </div>
        <div className="text-xs text-slate-400">
          AA-OS did the analysis and checked {packet.summary.machine_checked_and_passed} thing(s) itself. What is left for you:{" "}
          <span className="text-rose-300 font-semibold">{packet.summary.blockers} blocker(s)</span>,{" "}
          <span className="text-amber-300 font-semibold">{packet.summary.needs_your_judgment} to judge</span>, and{" "}
          {packet.summary.optional_spot_checks} optional spot check(s).
        </div>
        {signoff && (
          <div className={`text-xs rounded-lg border p-3 ${signoff.stale ? "border-amber-500/30 text-amber-300" : signoff.outcome === "VERIFIED" ? "border-emerald-500/30 text-emerald-300" : "border-rose-500/30 text-rose-300"}`}>
            {signoff.stale
              ? `A previous sign-off (${signoff.outcome}) no longer applies: the result changed since.`
              : `Signed off as ${signoff.outcome.replace("_", " ")} by ${signoff.reviewer?.email || "you"}${signoff.comment ? ` — ${signoff.comment}` : ""}`}
          </div>
        )}
        {error && <div className="text-xs text-rose-400 flex items-center gap-1"><AlertTriangle className="w-3.5 h-3.5" /> {error}</div>}
      </div>

      <div className="space-y-3">
        <div className="text-[11px] uppercase tracking-wide text-slate-500">Needs your decision ({required.length})</div>
        {required.length === 0 && <div className="text-xs text-slate-400">Nothing requires a decision.</div>}
        {required.map((it) => <ItemCard key={`${it.id}:${it.fingerprint}`} item={it} state={state} busy={busy} onDecide={onDecide} />)}
      </div>

      {optional.length > 0 && (
        <div className="space-y-3">
          <div className="text-[11px] uppercase tracking-wide text-slate-500">Machine-checked — optional spot checks and context ({optional.length})</div>
          {optional.map((it) => <ItemCard key={`${it.id}:${it.fingerprint}`} item={it} state={state} busy={busy} onDecide={onDecide} />)}
        </div>
      )}

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
        <div className="text-sm font-bold text-white">Sign off</div>
        <p className="text-[11px] text-slate-500">{packet.signoff_policy}</p>
        {!review.can_verify && review.blocking_reasons.length > 0 && (
          <div className="text-xs text-amber-300">{review.blocking_reasons.join(" ")}</div>
        )}
        <textarea value={signComment} onChange={(e) => setSignComment(e.target.value)} rows={2}
          placeholder="Comment (required unless you are marking it verified)…"
          className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 placeholder-slate-600" />
        <div className="flex flex-wrap gap-2">
          <button disabled={busy || !review.can_verify} onClick={() => onSign("VERIFIED")}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed">
            Mark as verified
          </button>
          <button disabled={busy} onClick={() => onSign("NEEDS_REWORK")}
            className="px-4 py-2 rounded-lg text-xs font-semibold border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-50">
            Send back for rework
          </button>
          <button disabled={busy} onClick={() => onSign("REJECTED")}
            className="px-4 py-2 rounded-lg text-xs font-semibold border border-rose-500/40 text-rose-300 hover:bg-rose-500/10 disabled:opacity-50">
            Reject result
          </button>
        </div>
      </div>
    </div>
  );
}
