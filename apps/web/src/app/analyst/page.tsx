"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Code2,
  Compass,
  Cpu,
  Database,
  Download,
  FileCheck,
  FileText,
  Filter,
  GitBranch,
  Layers,
  Lightbulb,
  Network,
  RotateCcw,
  Send,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  UploadCloud,
  XCircle,
  Zap,
} from "lucide-react";
import { createInvestigation, fetchDatasets, fetchInvestigation, streamInvestigationEvents, fetchProjects } from "../../lib/api";
import { AnalysisResponse, Dataset } from "../../types";
import StepTracker from "../../components/StepTracker";
import EvidenceViewer from "../../components/EvidenceViewer";
import EvidenceGraph from "../../components/EvidenceGraph";
import PlotlyChart from "../../components/PlotlyChart";
import InvestigationGraphView from "../../components/InvestigationGraphView";
import QuickAnalysisCenter from "../../components/QuickAnalysisCenter";
import ReportGeneratorModal from "../../components/ReportGeneratorModal";
import VerificationPanel from "../../components/VerificationPanel";

export default function AnalystPage() {
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [chatMessages, setChatMessages] = useState<Array<{ role: "user" | "assistant"; text: string }>>([]);
  const [interpretation, setInterpretation] = useState<any>(null);
  const [followUps, setFollowUps] = useState<string[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progressText, setProgressText] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [projectId, setProjectId] = useState<string>("");
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<string[]>([]);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"findings" | "graph" | "evidence" | "steps" | "scenarios" | "review">("findings");

  useEffect(() => {
    async function loadProjects() {
      try {
        const projects = await fetchProjects();
        setProjectId(projects[0]?.id || "");
      } catch (e) {
        console.error("Failed to load projects:", e);
      }
    }
    loadProjects();
  }, []);

  useEffect(() => {
    if (!projectId) {
      setDatasets([]);
      setSelectedDatasetIds([]);
      return;
    }
    async function loadProjectDatasets() {
      try {
        const ds = await fetchDatasets(projectId);
        setDatasets(ds);
        setSelectedDatasetIds((current) => current.filter((id) => ds.some((dataset) => dataset.id === id)));
      } catch (e) {
        console.error("Failed to load project datasets:", e);
        setDatasets([]);
        setSelectedDatasetIds([]);
      }
    }
    loadProjectDatasets();
  }, [projectId]);

  const handleAsk = async (q: string) => {
    if (!q.trim()) return;
    setIsLoading(true);
    setError(null);
    setProgressText("Queueing autonomous investigation…");
    setQuestion(q);

    try {
      if (!projectId) {
        throw new Error("No local analytical workspace is available yet.");
      }
      // Use the durable investigation plane rather than the synchronous chat
      // endpoint. The worker executes independently, while SSE events provide
      // live progress and the UI remains responsive for long-running analyses.
      const investigation = await createInvestigation({
        question: q,
        project_id: projectId,
        dataset_ids: selectedDatasetIds.length > 0 ? selectedDatasetIds : undefined,
      });

      setConversationId(undefined);
      setChatMessages((prev) => [
        ...prev,
        { role: "user", text: q },
        { role: "assistant", text: "Investigation queued. AA-OS is running the autonomous evidence loop locally." },
      ]);

      const toAnalysisResponse = (inv: any): AnalysisResponse => {
        const score = typeof inv.confidence_score === "number" ? inv.confidence_score : undefined;
        const verdictType = inv.verdict_type || "INCONCLUSIVE";
        // `status` is the investigation's *lifecycle* state (it reaches
        // "COMPLETED" whenever the evidence loop finishes running, whether
        // or not it found anything) -- it is not the epistemic outcome.
        // `verdict_type` is the actual conclusion, and INCONCLUSIVE /
        // INSUFFICIENT_DATA / CONFLICTING_EVIDENCE all mean "no real finding
        // to be confident about." Checking `inv.status === "INCONCLUSIVE"`
        // here almost never matched (status is normally "COMPLETED" even
        // for an inconclusive verdict), so a genuinely inconclusive
        // investigation fell through to the numeric-score buckets below and
        // rendered as "Low confidence" -- which reads as a real, if weak,
        // finding instead of "we didn't find anything conclusive."
        const isInsufficientVerdict =
          verdictType === "INCONCLUSIVE" || verdictType === "INSUFFICIENT_DATA" || verdictType === "CONFLICTING_EVIDENCE";
        const confidence = isInsufficientVerdict || inv.status === "FAILED" || score == null
          ? "Insufficient evidence"
          : score >= 0.75 ? "High confidence" : score >= 0.5 ? "Medium confidence" : "Low confidence";
        return {
          id: inv.id,
          project_id: inv.project_id,
          question: inv.question,
          status: inv.status,
          verdict: verdictType,
          direct_answer: inv.direct_answer || "",
          claim_gate: inv.claim_gate,
          main_finding: inv.main_finding || inv.direct_answer || "",
          confidence,
          hypotheses: (inv.hypotheses || []).map((h: any, i: number) => ({
            id: h.id || h.hypothesis_code || `H-${i + 1}`,
            statement: h.statement || "",
            rationale: h.rationale || "",
            priority: Number(h.posterior_probability ?? h.prior_probability ?? 0),
            prior_probability: h.prior_probability,
            posterior_probability: h.posterior_probability,
            status: h.belief_state || "active",
            investigation_steps: [],
            confirmed_findings: [],
          })),
          steps: (inv.experiments || []).map((e: any, i: number) => ({
            step_number: i + 1,
            title: `${e.tool_name || "experiment"} (${e.id || i + 1})`,
            action_type: e.tool_name || "experiment",
            description: e.status || "",
            tool_name: e.tool_name,
            duration_ms: 0,
            status: String(e.status || "planned").toLowerCase(),
          })),
          findings: [],
          evidence: (inv.evidence || []).map((e: any) => ({
            id: e.id,
            statement: e.statement || "",
            calculation_summary: "",
            dataset_version: "1",
            row_count_analyzed: 0,
            sql_executed: undefined,
            p_value: e.p_value,
            validation_status: e.validation_status || "SKIPPED",
            raw_metrics: {},
            verified_at: new Date().toISOString(),
          })),
          suggested_followups: [],
          executive_bullets: [],
          analysis_plan: inv.analysis_plan || undefined,
          decision_impact: inv.decision_impact || undefined,
          dataset_scope: inv.requested_dataset_ids || [],
          cross_dataset_review: inv.cross_dataset_review || undefined,
          created_at: inv.created_at || new Date().toISOString(),
        };
      };

      let stopped = false;
      const cleanup = streamInvestigationEvents(
        investigation.investigation_id,
        (event: any) => {
          const payload = event?.payload || {};
          const phase = payload.phase || event?.event_type || "WORKING";
          const experiment = payload.experiment_code ? ` • ${payload.experiment_code}` : "";
          setProgressText(`${String(phase).replaceAll("_", " ")}${experiment}`);
        },
        {
          onError: () => {
            // SSE is supplemental; polling below remains authoritative.
          },
        }
      );

      try {
        while (!stopped) {
          const current = await fetchInvestigation(investigation.investigation_id);
          const status = String(current.status || "").toUpperCase();
          if (["COMPLETED", "FAILED", "CANCELLED", "PARTIAL"].includes(status)) {
            const mapped = toAnalysisResponse(current);
            setAnalysis(mapped);
            setActiveTab("findings");
            if (status === "FAILED") {
              setError(current.error_message || "AA-OS investigation failed. Review the investigation events for the recorded failure.");
            }
            break;
          }
          setProgressText(`AA-OS is investigating • ${status || "RUNNING"}`);
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      } finally {
        stopped = true;
        cleanup();
      }
    } catch (err: any) {
      setError(err.message || "Failed to start autonomous investigation.");
    } finally {
      setProgressText(null);
      setIsLoading(false);
    }
  };

  const getVerdictBadge = (verdict?: string) => {
    switch (verdict) {
      case "DIAGNOSED":
        return {
          label: "DIAGNOSED (Variance Driver Proven)",
          color: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
          icon: <ShieldCheck className="w-4 h-4 text-emerald-400" />,
        };
      case "STATISTICALLY_SIGNIFICANT":
        return {
          label: "STATISTICALLY SIGNIFICANT (p < 0.05)",
          color: "bg-cyan-500/10 text-cyan-400 border-cyan-500/30",
          icon: <Activity className="w-4 h-4 text-cyan-400" />,
        };
      case "REFUTED":
        return {
          label: "REFUTED (Comparison Reverses Within Groups)",
          color: "bg-rose-500/10 text-rose-400 border-rose-500/30",
          icon: <ShieldAlert className="w-4 h-4 text-rose-400" />,
        };
      case "PREDICTED":
        return {
          label: "PREDICTED (Fitted Model with Backtest)",
          color: "bg-indigo-500/10 text-indigo-400 border-indigo-500/30",
          icon: <Sparkles className="w-4 h-4 text-indigo-400" />,
        };
      case "OBSERVED":
        return {
          label: "OBSERVED (Descriptive / Recomputed)",
          color: "bg-amber-500/10 text-amber-400 border-amber-500/30",
          icon: <Database className="w-4 h-4 text-amber-400" />,
        };
      case "NO_DETECTABLE_EFFECT":
        return {
          label: "NO DETECTABLE EFFECT (Null Supported with CI)",
          color: "bg-blue-500/10 text-blue-400 border-blue-500/30",
          icon: <FileCheck className="w-4 h-4 text-blue-400" />,
        };
      default:
        return {
          label: verdict || "CONFIRMED",
          color: "bg-cyan-500/10 text-cyan-400 border-cyan-500/30",
          icon: <CheckCircle2 className="w-4 h-4 text-cyan-400" />,
        };
    }
  };

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold text-cyan-400 mb-1">
            <Bot className="w-4 h-4" />
            <span>AUTONOMOUS ANALYTICAL INTELLIGENCE OPERATING SYSTEM (AA-OS)</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">AI Analyst & Investigation Studio</h1>
          <p className="text-xs text-slate-400 mt-1">
            Zero hallucinated arithmetic: DuckDB OLAP execution, Bayesian Investigation DAGs, and NumPy recomputation.
          </p>
        </div>

        {/* Dataset Workspace / Scope */}
        <div className="w-full xl:w-auto bg-slate-900 px-3 py-3 rounded-xl border border-slate-800 text-xs">
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <Filter className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">Dataset workspace:</span>
            <span className="text-cyan-300 font-medium">
              {selectedDatasetIds.length === 0 ? `All ${datasets.length} datasets` : `${selectedDatasetIds.length} selected`}
            </span>
            {selectedDatasetIds.length > 0 && (
              <button
                type="button"
                onClick={() => setSelectedDatasetIds([])}
                className="ml-2 text-[10px] text-slate-500 hover:text-white"
              >
                Clear scope
              </button>
            )}
          </div>
          <div className="flex flex-wrap gap-1.5 max-w-3xl">
            {datasets.map((d) => {
              const selected = selectedDatasetIds.includes(d.id);
              return (
                <button
                  key={d.id}
                  type="button"
                  onClick={() =>
                    setSelectedDatasetIds((current) =>
                      selected ? current.filter((id) => id !== d.id) : [...current, d.id]
                    )
                  }
                  className={`px-2.5 py-1.5 rounded-lg border text-[11px] transition ${
                    selected
                      ? "border-cyan-500/50 bg-cyan-500/10 text-cyan-200"
                      : "border-slate-700 bg-slate-950 text-slate-400 hover:text-white hover:border-slate-600"
                  }`}
                  title={`${d.name} • ${d.row_count?.toLocaleString()} rows • v${d.current_version}`}
                >
                  {d.name}
                  <span className="ml-1 text-[9px] opacity-60">{d.row_count?.toLocaleString()}r</span>
                </button>
              );
            })}
          </div>
          {selectedDatasetIds.length > 0 && (
            <div className="mt-2 text-[10px] text-slate-500">
              All selected datasets are requested together for this investigation. AA-OS must prove every dataset used in the analytical evidence path; selected data that are not needed are recorded as unused rather than silently treated as analyzed.
            </div>
          )}
        </div>
     </div>

      {/* Query Input Bar */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl focus-within:border-cyan-500/50 transition-all">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleAsk(question);
          }}
          className="flex items-center gap-3"
        >
          <div className="relative flex-1">
            <input
              type="text"
              placeholder="Ask an analytical question (e.g. 'Why did revenue fall in March?', 'Forecast trajectory', 'Find churn risk')..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={isLoading}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/80 font-sans"
            />
          </div>

          <button
            type="submit"
            disabled={isLoading || !question.trim()}
            className="bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-semibold px-6 py-3 rounded-xl flex items-center space-x-2 shadow-lg shadow-cyan-600/30 transition-all flex-shrink-0"
          >
            {isLoading ? (
              <>
                <Cpu className="w-4 h-4 animate-spin" />
                <span>Investigating...</span>
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                <span>Run Investigation</span>
              </>
            )}
          </button>
        </form>
      </div>

      {isLoading && progressText && (
        <div className="flex items-center gap-3 rounded-xl border border-cyan-500/20 bg-cyan-500/5 px-4 py-3 text-xs text-cyan-200">
          <Cpu className="w-4 h-4 animate-spin" />
          <span>{progressText}</span>
          <span className="ml-auto text-[10px] uppercase tracking-wider text-slate-500">Durable local worker</span>
        </div>
      )}

      {/* Local conversational analyst transcript */}
      {chatMessages.length > 0 && (
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-[11px] font-bold uppercase tracking-wide text-cyan-400">Local Analyst Conversation</div>
              <div className="text-[11px] text-slate-500 mt-1">Natural-language routing is local; calculations and evidence come from the canonical analytical engine.</div>
            </div>
            {interpretation && (
              <span className="text-[10px] px-2 py-1 rounded-lg border border-slate-700 bg-slate-950 text-slate-300 font-mono">
                {interpretation.problem_class}
              </span>
            )}
          </div>
          <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
            {chatMessages.map((m, idx) => (
              <div key={idx} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-3xl rounded-xl px-3 py-2 text-xs leading-relaxed whitespace-pre-wrap border ${m.role === "user" ? "bg-cyan-600/10 border-cyan-500/20 text-cyan-100" : "bg-slate-950 border-slate-800 text-slate-300"}`}>
                  {m.text}
                </div>
              </div>
            ))}
          </div>
          {followUps.length > 0 && !isLoading && (
            <div className="flex flex-wrap gap-2 pt-1">
              {followUps.slice(0, 3).map((item) => (
                <button
                  key={item}
                  onClick={() => handleAsk(item)}
                  className="text-[11px] px-2.5 py-1.5 rounded-lg border border-slate-700 bg-slate-950 hover:bg-slate-800 text-slate-300 hover:text-white"
                >
                  {item}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Quick Analysis Center */}
      {!analysis && !isLoading && (
        <QuickAnalysisCenter
          isLoading={isLoading}
          onSelectQuestion={(q) => handleAsk(q)}
        />
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-2xl p-5 flex items-start space-x-3 text-rose-300 text-xs">
          <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-400 mt-0.5" />
          <div>
            <span className="font-bold block mb-1">Investigation Execution Error</span>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Loading Progress State */}
      {isLoading && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-10 text-center space-y-4">
          <div className="relative w-12 h-12 mx-auto">
            <Cpu className="w-12 h-12 text-cyan-400 animate-spin" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white">Autonomous Investigation in Progress</h3>
            <p className="text-xs text-slate-400 mt-1 max-w-md mx-auto">
              Reconstructing semantic graph, formulating competing hypotheses, evaluating Expected Information Gain, and recomputing evidence via secondary validator...
            </p>
          </div>
        </div>
      )}

      {/* Full Analysis Results */}
      {analysis && !isLoading && (
        <div className="space-y-6">
          {/* Executive Direct Answer & Verdict Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
            <div className="flex flex-wrap items-start justify-between gap-4 mb-4">
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  {(() => {
                    const vb = getVerdictBadge(analysis.verdict);
                    return (
                      <span className={`text-xs font-mono font-bold px-3 py-1 rounded-full border flex items-center space-x-1.5 ${vb.color}`}>
                        {vb.icon}
                        <span>{vb.label}</span>
                      </span>
                    );
                  })()}
                  <span className="text-xs font-mono text-slate-500 bg-slate-950 px-2.5 py-1 rounded border border-slate-800">
                    Confidence: {analysis.confidence}
                  </span>
                </div>
                <h2 className="text-xl font-bold text-white tracking-tight mt-2">
                  {analysis.direct_answer}
                </h2>
                {Array.isArray((analysis as any).dataset_scope) && (analysis as any).dataset_scope.length > 0 && (
                  <div className="mt-2 text-[10px] text-slate-500">
                    Investigation scope: {(analysis as any).dataset_scope.length} selected dataset(s)
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setIsReportModalOpen(true)}
                  className="text-xs font-semibold bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/40 px-3.5 py-2 rounded-xl flex items-center space-x-2 transition-all shadow-lg shadow-indigo-600/10"
                >
                  <FileCheck className="w-4 h-4" />
                  <span>Generate Report</span>
                </button>
              </div>
            </div>

            {analysis.claim_gate && (
              <div className={`mt-4 rounded-xl border p-4 ${
                analysis.claim_gate.outcome === "ANSWER"
                  ? "border-emerald-500/30 bg-emerald-500/5"
                  : analysis.claim_gate.outcome === "QUALIFIED_ANSWER"
                    ? "border-amber-500/30 bg-amber-500/5"
                    : "border-red-500/30 bg-red-500/5"
              }`}>
                <div className="flex items-center justify-between gap-3 mb-3">
                  <span className="text-xs font-mono font-bold tracking-wider text-white">
                    CLAIM GATE: {analysis.claim_gate.outcome}
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">
                    Evidence L{analysis.claim_gate.evidence_level} · Design {analysis.claim_gate.design_status}
                  </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-emerald-300 mb-1">Claim supported</div>
                    <p className="text-xs text-slate-200 leading-relaxed">{analysis.claim_gate.allowed_claim}</p>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-red-300 mb-1">Claim not supported</div>
                    <p className="text-xs text-slate-200 leading-relaxed">{analysis.claim_gate.blocked_claim || "None at the current evidence level."}</p>
                  </div>
                </div>
                <div className="mt-3 text-xs text-slate-300">
                  <span className="font-semibold text-white">Why:</span> {analysis.claim_gate.reason}
                </div>
                {analysis.claim_gate.recovery_actions?.length > 0 && (
                  <div className="mt-3">
                    <div className="text-[10px] uppercase tracking-wider text-cyan-300 mb-2">To strengthen the claim</div>
                    <div className="space-y-2">
                      {analysis.claim_gate.recovery_actions.map((action) => (
                        <button
                          key={action}
                          type="button"
                          onClick={() => handleAsk(`For the current finding, pursue this recovery action and return only evidence supported by the available data: ${action}`)}
                          disabled={isLoading}
                          className="w-full text-left rounded-lg border border-cyan-500/20 bg-cyan-500/5 hover:bg-cyan-500/10 disabled:opacity-50 px-3 py-2 text-xs text-slate-200 transition-colors"
                        >
                          <span className="text-cyan-300 mr-2">↗</span>{action}
                        </button>
                      ))}
                    </div>
                    <p className="text-[10px] text-slate-500 mt-2">Running a recovery action starts a fresh deterministic investigation. It does not raise the evidence level by assertion.</p>
                  </div>
                )}
                {analysis.claim_gate.computed_evidence && (
                  <div className="mt-3 flex flex-wrap gap-2 text-[10px] font-mono text-slate-400">
                    {typeof analysis.claim_gate.computed_evidence.verified_evidence_count === "number" && (
                      <span className="rounded-md border border-slate-800 bg-slate-950 px-2 py-1">Verified evidence: {analysis.claim_gate.computed_evidence.verified_evidence_count}</span>
                    )}
                    {Array.isArray(analysis.claim_gate.computed_evidence.evidence_ids) && (
                      <span className="rounded-md border border-slate-800 bg-slate-950 px-2 py-1">Evidence records: {analysis.claim_gate.computed_evidence.evidence_ids.length}</span>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Core Finding Callout */}
            <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 text-xs text-slate-300 leading-relaxed font-sans">
              <span className="font-semibold text-cyan-300 block mb-1">Key Empirical Finding:</span>
              {analysis.main_finding}
            </div>

            {/* Executive Bullets */}
            {analysis.executive_bullets && analysis.executive_bullets.length > 0 && (
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-2.5">
                {analysis.executive_bullets.map((bullet, bIdx) => (
                  <div key={bIdx} className="flex items-start space-x-2 text-xs text-slate-300 bg-slate-950/50 p-2.5 rounded-lg border border-slate-800/80">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <span>{bullet}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {analysis.analysis_plan && (
            <div className="bg-slate-900 border border-cyan-500/20 rounded-2xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2 text-sm font-bold text-white">
                  <Compass className="w-4 h-4 text-cyan-400" />
                  <span>Analysis Contract — why AA-OS is doing these tests</span>
                </div>
                <span className="text-[10px] font-mono px-2 py-1 rounded border border-slate-700 bg-slate-950 text-cyan-300">
                  {String(analysis.analysis_plan.task || "ANALYSIS")}
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                <div className="bg-slate-950 rounded-xl p-3 border border-slate-800">
                  <div className="text-[10px] uppercase tracking-wide text-slate-500">Claim ceiling</div>
                  <div className="text-slate-200 mt-1 font-semibold">{String(analysis.analysis_plan.claim_type || "Not specified")}</div>
                </div>
                <div className="bg-slate-950 rounded-xl p-3 border border-slate-800">
                  <div className="text-[10px] uppercase tracking-wide text-slate-500">Estimand</div>
                  <div className="text-slate-200 mt-1">{String(analysis.analysis_plan.estimand?.estimand || "Question-specific")}</div>
                </div>
                <div className="bg-slate-950 rounded-xl p-3 border border-slate-800">
                  <div className="text-[10px] uppercase tracking-wide text-slate-500">Decision status</div>
                  <div className="text-slate-200 mt-1">{String(analysis.analysis_plan.decision_status || "Unknown")}</div>
                </div>
              </div>
              {Array.isArray(analysis.analysis_plan.experiments) && analysis.analysis_plan.experiments.length > 0 && (
                <div>
                  <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-2">Planned minimum-sufficient experiments</div>
                  <div className="space-y-2">
                    {analysis.analysis_plan.experiments.map((exp: any) => (
                      <div key={String(exp.code)} className="flex items-start gap-3 bg-slate-950/70 rounded-xl border border-slate-800 p-3">
                        <span className="text-[10px] font-mono text-cyan-300 bg-cyan-500/10 border border-cyan-500/20 rounded px-2 py-1">{String(exp.code)}</span>
                        <div className="text-xs text-slate-300">{String(exp.purpose)}<div className="text-[10px] text-slate-500 mt-1">Method: {String(exp.method)}</div></div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div className="text-[11px] text-slate-400">
                <span className="text-slate-500">Stopping rule: </span>{String(analysis.analysis_plan.stopping_rule || "Not specified")}
              </div>
              {Array.isArray(analysis.analysis_plan.unresolved_questions) && analysis.analysis_plan.unresolved_questions.length > 0 && (
                <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-[11px] text-amber-200">
                  Unresolved context: {analysis.analysis_plan.unresolved_questions.join("; ")}
                </div>
              )}
            </div>
          )}

          {analysis.decision_impact && (
            <div className="bg-slate-900 border border-emerald-500/20 rounded-2xl p-5 space-y-3">
              <div className="flex items-center gap-2 text-sm font-bold text-white">
                <ArrowRight className="w-4 h-4 text-emerald-400" />
                <span>Decision Guidance</span>
                <span className="text-[10px] font-mono ml-auto text-slate-500">CONDITIONAL</span>
              </div>
              <div className="text-[11px] text-slate-400">AA-OS does not invent financial impact or convert association into causation.</div>
              {Array.isArray(analysis.decision_impact.actions) && analysis.decision_impact.actions.map((action: any, idx: number) => (
                <div key={idx} className="bg-slate-950 rounded-xl border border-slate-800 p-3 text-xs text-slate-300">
                  <div className="text-[10px] uppercase text-emerald-400 mb-1">{String(action.type || "OPTION")}</div>
                  {String(action.action || "")}
                </div>
              ))}
              {Array.isArray(analysis.decision_impact.disclosed_unknowns) && analysis.decision_impact.disclosed_unknowns.length > 0 && (
                <div className="text-[10px] text-slate-500">Unknowns: {analysis.decision_impact.disclosed_unknowns.join(", ")}</div>
              )}
            </div>
          )}

          {/* Navigation Tabs Bar */}
          <div className="flex items-center space-x-2 border-b border-slate-800 pb-2">
            {[
              { id: "findings", label: "Findings & Actions", icon: <Lightbulb className="w-4 h-4" /> },
              { id: "graph", label: "Investigation DAG", icon: <Network className="w-4 h-4" /> },
              { id: "evidence", label: `Verified Evidence (${analysis.evidence?.length || 0})`, icon: <ShieldCheck className="w-4 h-4" /> },
              { id: "steps", label: `Reasoning Steps (${analysis.steps?.length || 0})`, icon: <Clock className="w-4 h-4" /> },
              { id: "scenarios", label: "Policy Scenarios", icon: <Compass className="w-4 h-4" /> },
              { id: "review", label: "Human Review", icon: <FileCheck className="w-4 h-4" /> },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`text-xs font-semibold px-4 py-2 rounded-xl flex items-center space-x-2 transition-all ${
                  activeTab === tab.id
                    ? "bg-slate-800 text-cyan-300 border border-slate-700 shadow-md"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            ))}
          </div>

          {/* TAB 1: Findings & Recommended Actions */}
          {activeTab === "findings" && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-8 space-y-4">
                {analysis.findings && analysis.findings.length > 0 ? (
                  analysis.findings.map((f, idx) => (
                    <div key={f.id || idx} className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <span className="text-xs font-mono font-bold text-cyan-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                            Finding #{idx + 1}
                          </span>
                          <h3 className="text-sm font-bold text-white">{f.title}</h3>
                        </div>
                        <span className="text-xs text-slate-400">
                          Priority: {Math.round(f.importance_score * 100)}%
                        </span>
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed">{f.summary}</p>

                      {/* Chart preview if available */}
                      {f.chart_data && (
                        <div className="pt-2">
                          <PlotlyChart spec={f.chart_data} />
                        </div>
                      )}

                      {/* Recommended Actions */}
                      {f.recommended_actions && f.recommended_actions.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-slate-800/80">
                          <span className="text-[10px] font-mono uppercase text-cyan-400 font-semibold block mb-1.5">
                            Prescriptive Action Items
                          </span>
                          <div className="space-y-1.5">
                            {f.recommended_actions.map((act, aIdx) => (
                              <div key={aIdx} className="flex items-center space-x-2 text-xs text-slate-300 bg-slate-950/60 px-3 py-1.5 rounded-lg border border-slate-800">
                                <ArrowRight className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                                <span>{act}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center text-xs text-slate-400">
                    Zero raw findings extracted.
                  </div>
                )}
              </div>

              {/* Right Sidebar: Multi-Vector Confidence & Manifest */}
              <div className="lg:col-span-4 space-y-4">
                {analysis.confidence_breakdown && (
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
                    <div className="flex items-center space-x-2 text-xs font-bold text-white border-b border-slate-800 pb-3">
                      <ShieldCheck className="w-4 h-4 text-cyan-400" />
                      <span>Multi-Vector Confidence Breakdown</span>
                    </div>

                    <div className="space-y-3 text-xs">
                      <div>
                        <div className="flex justify-between text-slate-400 mb-1">
                          <span>Evidence Quality</span>
                          <span className="font-mono text-cyan-300 font-bold">{analysis.confidence_breakdown.evidence_quality}%</span>
                        </div>
                        <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
                          <div className="bg-cyan-400 h-full" style={{ width: `${analysis.confidence_breakdown.evidence_quality}%` }} />
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-slate-400 mb-1">
                          <span>Data Hygiene</span>
                          <span className="font-mono text-cyan-300 font-bold">{analysis.confidence_breakdown.data_quality}%</span>
                        </div>
                        <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
                          <div className="bg-emerald-400 h-full" style={{ width: `${analysis.confidence_breakdown.data_quality}%` }} />
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-slate-400 mb-1">
                          <span>Statistical Power</span>
                          <span className="font-mono text-cyan-300 font-bold">{analysis.confidence_breakdown.statistical_strength}%</span>
                        </div>
                        <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
                          <div className="bg-indigo-400 h-full" style={{ width: `${analysis.confidence_breakdown.statistical_strength}%` }} />
                        </div>
                      </div>

                      <div className="pt-2 border-t border-slate-800 flex justify-between text-slate-400">
                        <span>Causal Standard</span>
                        <span className="font-mono text-cyan-300 font-semibold">{analysis.confidence_breakdown.causal_evidence}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Provenance Manifest Card */}
                {analysis.manifest && (
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3 font-mono text-xs">
                    <div className="flex items-center space-x-2 text-white font-sans font-bold border-b border-slate-800 pb-2.5">
                      <Code2 className="w-4 h-4 text-emerald-400" />
                      <span>Canonical Manifest Lineage</span>
                    </div>
                    <div className="space-y-1.5 text-[11px] text-slate-400">
                      <div>Execution Time: <span className="text-white">{analysis.manifest.execution_time_seconds}s</span></div>
                      <div>Total Steps: <span className="text-white">{analysis.manifest.total_steps}</span></div>
                      <div>Engine: <span className="text-amber-300">DuckDB (In-Process)</span></div>
                      <div>AI Provider: <span className="text-cyan-300">{analysis.manifest.ai_provider}</span></div>
                      <div className="pt-2 border-t border-slate-800">
                        <span className="text-[10px] text-slate-500 uppercase block mb-0.5">Content-Addressed SHA-256</span>
                        <span className="text-cyan-400 text-[10px] break-all">{analysis.manifest.reproducible_hash}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 2: Investigation DAG */}
          {activeTab === "graph" && (
            <InvestigationGraphView
              hypotheses={analysis.hypotheses}
              objective={analysis.question.toLowerCase().includes("why") ? "Diagnostic" : "Analytical"}
              targetMetric="Revenue"
            />
          )}

          {/* TAB 3: Independently Verified Evidence */}
          {activeTab === "evidence" && (
            <div className="space-y-6">
              <EvidenceViewer evidence={analysis.evidence || []} />
              {analysis.evidence_graph && (
                <EvidenceGraph nodes={analysis.evidence_graph} />
              )}
            </div>
          )}

          {/* TAB 4: Reasoning Steps */}
          {activeTab === "steps" && (
            <StepTracker steps={analysis.steps} />
          )}

          {/* TAB 5: Human review of cross-dataset execution */}
          {activeTab === "review" && (
            <div className="space-y-5">
              <VerificationPanel investigationId={analysis.id} />
              <div className="bg-slate-900 border border-cyan-500/20 rounded-2xl p-5 space-y-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-sm font-bold text-white">
                    <FileCheck className="w-4 h-4 text-cyan-400" />
                    <span>Cross-Dataset Execution Ledger</span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-1 rounded border border-slate-700 bg-slate-950 text-cyan-300">
                    {analysis.cross_dataset_review?.status || "NO_CROSS_DATASET_REVIEW"}
                  </span>
                </div>
                <p className="text-xs text-slate-300">
                  {analysis.cross_dataset_review?.review_message || "No cross-dataset execution artifact was returned for this investigation."}
                </p>
                {analysis.cross_dataset_review && (
                  <>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
                      <div className="bg-slate-950 rounded-xl p-3 border border-slate-800"><div className="text-[10px] uppercase text-slate-500">Selected datasets</div><div className="text-white font-bold mt-1">{analysis.cross_dataset_review.selected_dataset_count}</div></div>
                      <div className="bg-slate-950 rounded-xl p-3 border border-slate-800"><div className="text-[10px] uppercase text-slate-500">Relational experiments</div><div className="text-white font-bold mt-1">{analysis.cross_dataset_review.relational_experiments.length}</div></div>
                      <div className="bg-slate-950 rounded-xl p-3 border border-slate-800"><div className="text-[10px] uppercase text-slate-500">Verified relations</div><div className="text-white font-bold mt-1">{analysis.cross_dataset_review.relational_experiments.filter((r) => r.independently_verified).length}</div></div>
                      <div className="bg-slate-950 rounded-xl p-3 border border-slate-800"><div className="text-[10px] uppercase text-slate-500">Join blocks</div><div className="text-white font-bold mt-1">{analysis.cross_dataset_review.join_safety_block_count}</div></div>
                    </div>

                    <div className="space-y-2">
                      {analysis.cross_dataset_review.datasets.map((dataset) => (
                        <div key={dataset.dataset_id} className="flex items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-950/70 p-3 text-xs">
                          <div className="min-w-0">
                            <div className="font-mono text-slate-200 truncate">{dataset.dataset_name || dataset.dataset_id}</div>
                            <div className="text-[10px] text-slate-500 mt-1">ID: {dataset.dataset_id} · Used by: {dataset.used_by_experiments.length ? dataset.used_by_experiments.join(", ") : "No recorded cross-dataset claim"}</div>
                          </div>
                          <span className={dataset.used_in_verified_relational_claim ? "text-emerald-300" : "text-amber-300"}>
                            {dataset.used_in_verified_relational_claim ? "VERIFIED CONTRIBUTOR" : "NOT VERIFIED AS CONTRIBUTOR"}
                          </span>
                        </div>
                      ))}
                    </div>

                    <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4 space-y-2">
                      <div className="text-[10px] uppercase tracking-wide text-slate-500">Reviewer checklist</div>
                      {Object.entries(analysis.cross_dataset_review.review_checks).map(([key, value]) => (
                        <div key={key} className="flex items-center justify-between gap-3 text-[11px]">
                          <span className="text-slate-300">{key.replaceAll("_", " ")}</span>
                          <span className={value === true ? "text-emerald-300" : value === false ? "text-red-300" : "text-amber-300"}>{value === true ? "PASS" : value === false ? "FAIL" : "NOT APPLICABLE"}</span>
                        </div>
                      ))}
                    </div>

                    {analysis.cross_dataset_review.relational_experiments.map((rel) => (
                      <div key={rel.experiment_id} className="rounded-xl border border-slate-800 bg-slate-950/70 p-4 space-y-3">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-[10px] font-mono px-2 py-1 rounded border border-cyan-500/20 bg-cyan-500/10 text-cyan-300">{rel.experiment_code}</span>
                          <span className="text-[10px] text-slate-400">{rel.source_tables.join(" → ")}</span>
                          <span className={rel.independently_verified ? "text-[10px] text-emerald-300" : "text-[10px] text-amber-300"}>{rel.independently_verified ? "INDEPENDENTLY VERIFIED" : "NOT INDEPENDENTLY VERIFIED"}</span>
                        </div>
                        <div className="space-y-1">
                          {rel.join_hops.map((hop, idx) => (
                            <div key={idx} className="text-[11px] text-slate-300 font-mono">{String(hop.left_table)}.{String(hop.left_key)} → {String(hop.right_table)}.{String(hop.right_key)} ({String(hop.join_type || "INNER")})</div>
                          ))}
                        </div>
                        {rel.verification_statuses.length > 0 && <div className="text-[10px] text-slate-500">Verification records: {rel.verification_statuses.join(", ")}</div>}
                      </div>
                    ))}
                  </>
                )}
              </div>
            </div>
          )}

          {/* TAB 6: Policy Scenarios */}
          {activeTab === "scenarios" && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {analysis.what_if_scenarios && analysis.what_if_scenarios.length > 0 ? (
                analysis.what_if_scenarios.map((sc, scIdx) => (
                  <div key={scIdx} className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
                    <div className="flex items-center space-x-2 text-xs font-bold text-cyan-400">
                      <Compass className="w-4 h-4" />
                      <span>Policy Intervention #{scIdx + 1}</span>
                    </div>
                    <h4 className="text-sm font-bold text-white">{sc.scenario}</h4>
                    <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1.5 text-xs">
                      <div>
                        <span className="text-slate-500">Assumption:</span>{" "}
                        <span className="text-slate-300">{sc.assumption}</span>
                      </div>
                      <div>
                        <span className="text-slate-500">Projected Outcome:</span>{" "}
                        <span className="text-emerald-400 font-bold">{sc.projected_impact}</span>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center text-xs text-slate-400">
                  No policy simulation scenarios generated for this inquiry.
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Structured Report Modal */}
      {analysis && (
        <ReportGeneratorModal
          isOpen={isReportModalOpen}
          onClose={() => setIsReportModalOpen(false)}
          analysis={analysis}
        />
      )}
    </div>
  );
}
