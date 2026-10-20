"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  Bot,
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
import {
  createInvestigation,
  createAnalysisSession,
  fetchAnalysisSession,
  updateAnalysisSession,
  closeAnalysisSession,
  downloadAnalysisSessionExport,
  saveExportBlob,
  AnalysisSession,
  fetchDatasets,
  fetchDatasetData,
  fetchInvestigation,
  streamInvestigationEvents,
  fetchProjects,
  interpretQuestion,
  PreAnalysisPlan,
  InterpretGuidance,
  reviseInvestigationGuidance,
  retryInvestigationDispatch,
} from "../../lib/api";
import { AnalysisResponse, Dataset, Project } from "../../types";
import AnalyticsDataStarter from "../../components/AnalyticsDataStarter";
import StepTracker from "../../components/StepTracker";
import EvidenceViewer from "../../components/EvidenceViewer";
import EvidenceGraph from "../../components/EvidenceGraph";
import PlotlyChart from "../../components/PlotlyChart";
import InvestigationGraphView from "../../components/InvestigationGraphView";
import QuickAnalysisCenter from "../../components/QuickAnalysisCenter";
import ReportGeneratorModal from "../../components/ReportGeneratorModal";
import VerificationPanel from "../../components/VerificationPanel";
import StructuredAnalysisAnswer from "../../components/StructuredAnalysisAnswer";
import ProposedAnalysisPanel from "../../components/ProposedAnalysisPanel";
import RecordLevelInspector from "../../components/RecordLevelInspector";

export default function AnalystPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const initialDataset = useRef<string | null>(null);
  const initialAnalysisSession = useRef<string | null>(null);
  const sessionResumeAttempted = useRef(false);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isPlanning, setIsPlanning] = useState(false);
  const [planNotice, setPlanNotice] = useState<string | null>(null);
  const [proposedPlan, setProposedPlan] = useState<PreAnalysisPlan | null>(null);
  // Fingerprint of the guidance the current proposedPlan.plan_hash was computed for. The hash the
  // analyst approves must belong to the plan they were shown, so edited guidance is re-previewed
  // (via /interpret) before it can be executed.
  const [previewedGuidanceKey, setPreviewedGuidanceKey] = useState<string>("");
  const [clarificationInvestigationId, setClarificationInvestigationId] = useState<string | null>(null);
  const [dispatchRetryId, setDispatchRetryId] = useState<string | null>(null);
  const [recordSampleRows, setRecordSampleRows] = useState<any[]>([]);
  const [recordSampleCols, setRecordSampleCols] = useState<string[]>([]);
  const [totalRecordCount, setTotalRecordCount] = useState<number>(0);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [chatMessages, setChatMessages] = useState<Array<{ role: "user" | "assistant"; text: string }>>([]);
  const [interpretation, setInterpretation] = useState<any>(null);
  const [followUps, setFollowUps] = useState<string[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [analysisSession, setAnalysisSession] = useState<AnalysisSession | null>(null);
  const analysisSessionRef = useRef<AnalysisSession | null>(null);
  const [sessionTitleDraft, setSessionTitleDraft] = useState("New analysis");
  const [sessionScopeDraft, setSessionScopeDraft] = useState("");
  const [isSessionBusy, setIsSessionBusy] = useState(false);
  const [sessionNotice, setSessionNotice] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [sessionExporting, setSessionExporting] = useState<"pdf" | "docx" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progressText, setProgressText] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [projectId, setProjectId] = useState<string>("");
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<string[]>([]);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"findings" | "records" | "graph" | "evidence" | "steps" | "scenarios" | "review">("findings");

  const commitAnalysisSession = useCallback((session: AnalysisSession | null) => {
    analysisSessionRef.current = session;
    setAnalysisSession(session);
    if (session) {
      setSessionTitleDraft(session.title || "New analysis");
      setSessionScopeDraft(session.scope_filter || "");
    }
  }, []);

  useEffect(() => {
    async function loadProjects() {
      try {
        const projects = await fetchProjects();
        setProjects(projects);
        if (!projects.length) throw new Error("No workspace is available for your account.");
        const params = new URLSearchParams(window.location.search);
        const requestedProject = params.get("project_id");
        if (requestedProject && !projects.some((project) => project.id === requestedProject)) {
          throw new Error("This workspace is unavailable. Open an accessible workspace from Datasets.");
        }
        initialDataset.current = params.get("dataset");
        initialAnalysisSession.current = params.get("analysis_session_id");
        setProjectId(requestedProject || projects[0]?.id || "");
        if (typeof window !== "undefined") {
          const params = new URLSearchParams(window.location.search);
          const initialQ = params.get("q") || params.get("question");
          if (initialQ) {
            setQuestion(initialQ);
          }
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unable to load your workspace. Refresh to try again.");
        setLoadingDatasets(false);
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
    let cancelled = false;
    setLoadingDatasets(true);
    async function loadProjectDatasets() {
      try {
        const ds = await fetchDatasets(projectId);
        if (cancelled) return;
        setDatasets(ds);
        const requested = initialDataset.current;
        initialDataset.current = null;
        if (requested && !ds.some((dataset) => dataset.id === requested)) {
          setSelectedDatasetIds([]);
          setError("The requested dataset is unavailable in this workspace. Choose a dataset below.");
        } else {
          setSelectedDatasetIds(requested ? [requested] : ds.length === 1 ? [ds[0].id] : []);
        }
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Unable to load datasets. Refresh to try again.");
        setDatasets([]);
        setSelectedDatasetIds([]);
      } finally {
        if (!cancelled) setLoadingDatasets(false);
      }
    }
    void loadProjectDatasets();
    return () => { cancelled = true; };
  }, [projectId]);

  useEffect(() => {
    if (!projectId || loadingDatasets || !initialAnalysisSession.current || sessionResumeAttempted.current) return;
    const sessionId = initialAnalysisSession.current;
    sessionResumeAttempted.current = true;
    let cancelled = false;
    fetchAnalysisSession(sessionId).then((session) => {
      if (cancelled) return;
      if (session.project_id !== projectId) throw new Error("The requested analysis session belongs to a different workspace.");
      const missing = (session.dataset_ids || []).filter((id) => !datasets.some((dataset) => dataset.id === id));
      if (missing.length) throw new Error("This analysis references a dataset that is no longer available in the selected workspace.");
      setSelectedDatasetIds(session.dataset_ids || []);
      commitAnalysisSession(session);
      setSessionNotice(session.status === "OPEN" ? "Resumed the existing analysis session. Follow-up questions will continue its report." : "This analysis is closed and read-only. Start a new analysis to ask more questions.");
    }).catch((cause) => {
      if (!cancelled) setSessionError(cause instanceof Error ? cause.message : "Unable to resume the requested analysis session.");
    });
    return () => { cancelled = true; };
  }, [projectId, loadingDatasets, datasets, commitAnalysisSession]);

  useEffect(() => {
    let cancelled = false;
    setRecordSampleRows([]);
    setRecordSampleCols([]);
    setTotalRecordCount(0);
    setPreviewError(null);
    const targetId = selectedDatasetIds[0];
    if (!targetId) return;
    fetchDatasetData(targetId).then((data: any) => {
      if (cancelled) return;
      setRecordSampleRows(data.rows || []);
      setRecordSampleCols(data.columns?.length ? data.columns : Object.keys(data.rows?.[0] || {}));
      setTotalRecordCount(data.total_rows || data.row_count || data.rows?.length || 0);
    }).catch((cause) => {
      if (!cancelled) setPreviewError(cause instanceof Error ? cause.message : "Dataset preview unavailable.");
    });
    return () => { cancelled = true; };
  }, [selectedDatasetIds]);

  const ensureOpenAnalysisSession = async (): Promise<AnalysisSession> => {
    const current = analysisSessionRef.current;
    if (current && current.project_id === projectId) {
      if (current.status !== "OPEN") {
        throw new Error("This analysis is closed. Click New Analysis before asking another question so it is not appended to a finalized report.");
      }
      const selected = [...selectedDatasetIds].sort();
      const pinned = [...(current.dataset_ids || [])].sort();
      if (JSON.stringify(selected) === JSON.stringify(pinned)) return current;
      throw new Error("The selected datasets differ from this analysis session's pinned scope. Start a new analysis to change datasets.");
    }
    if (!projectId || !selectedDatasetIds.length) {
      throw new Error("Select at least one dataset before starting an analysis.");
    }
    const created = await createAnalysisSession({
      project_id: projectId,
      title: sessionTitleDraft.trim() || undefined,
      dataset_ids: selectedDatasetIds,
      scope_filter: sessionScopeDraft.trim() || undefined,
    });
    commitAnalysisSession(created);
    setSessionNotice("Analysis session created. Accepted question results will accumulate in one report.");
    return created;
  };

  const handleNewAnalysis = async () => {
    if (isSessionBusy || busy || !projectId || !selectedDatasetIds.length) return;
    setIsSessionBusy(true);
    setSessionError(null);
    setSessionNotice(null);
    try {
      const previous = analysisSessionRef.current;
      if (previous && previous.status === "OPEN") {
        const proceed = window.confirm("Close the current analysis and start a new one? The previous session and its report history will be preserved.");
        if (!proceed) return;
        const closedPrevious = await closeAnalysisSession(previous.id);
        // Reflect the server-confirmed closure before creating the next session. If the
        // new-session request fails, do not leave the UI believing the old session is open.
        commitAnalysisSession(closedPrevious.session);
      }
      const created = await createAnalysisSession({
        project_id: projectId,
        title: sessionTitleDraft.trim() || undefined,
        dataset_ids: selectedDatasetIds,
        scope_filter: sessionScopeDraft.trim() || undefined,
      });
      commitAnalysisSession(created);
      setAnalysis(null);
      setProposedPlan(null);
      setPlanNotice(null);
      setQuestion("");
      setChatMessages([]);
      setFollowUps([]);
      setInterpretation(null);
      setError(null);
      setClarificationInvestigationId(null);
      setDispatchRetryId(null);
      setIsReportModalOpen(false);
      setSessionNotice("New analysis started. Ask questions and accept verified results to build its cumulative report.");
    } catch (cause) {
      setSessionError(cause instanceof Error ? cause.message : "Unable to start a new analysis session.");
    } finally {
      setIsSessionBusy(false);
    }
  };

  const handleSaveSessionContext = async () => {
    const current = analysisSessionRef.current;
    if (!current || current.status !== "OPEN" || isSessionBusy) return;
    setIsSessionBusy(true);
    setSessionError(null);
    setSessionNotice(null);
    try {
      const updated = await updateAnalysisSession(current.id, {
        title: sessionTitleDraft.trim() || current.title,
        scope_filter: sessionScopeDraft,
      });
      commitAnalysisSession(updated);
      setSessionNotice("Shared population context saved. It will be included in each newly approved question contract.");
    } catch (cause) {
      setSessionError(cause instanceof Error ? cause.message : "Unable to save the shared analysis context.");
    } finally {
      setIsSessionBusy(false);
    }
  };

  const handleCloseAnalysis = async () => {
    const current = analysisSessionRef.current;
    if (!current || current.status !== "OPEN" || isSessionBusy) return;
    if (!window.confirm("Close this analysis? New questions will be blocked, and the accepted report will be finalized for export.")) return;
    setIsSessionBusy(true);
    setSessionError(null);
    setSessionNotice(null);
    try {
      const response = await closeAnalysisSession(current.id);
      commitAnalysisSession(response.session);
      setSessionTitleDraft("New analysis");
      setSessionNotice("Analysis closed. The final cumulative PDF and DOCX are available below. Edit the title or default filters above before starting the next analysis.");
    } catch (cause) {
      setSessionError(cause instanceof Error ? cause.message : "Unable to close the analysis.");
    } finally {
      setIsSessionBusy(false);
    }
  };

  const handleExportSession = async (format: "pdf" | "docx") => {
    const current = analysisSessionRef.current;
    if (!current || sessionExporting) return;
    setSessionExporting(format);
    setSessionError(null);
    setSessionNotice(null);
    try {
      const blob = await downloadAnalysisSessionExport(current.id, format);
      const delivery = await saveExportBlob(blob, `MindEd_Analysis_${current.id.slice(0, 8)}_R${current.report_revision}.${format}`);
      setSessionNotice(delivery.message);
    } catch (cause) {
      setSessionError(cause instanceof Error ? cause.message : `Unable to export the cumulative ${format.toUpperCase()} report.`);
    } finally {
      setSessionExporting(null);
    }
  };

  const refreshAnalysisSession = async () => {
    const current = analysisSessionRef.current;
    if (!current) return;
    try {
      const refreshed = await fetchAnalysisSession(current.id);
      commitAnalysisSession(refreshed);
    } catch (cause) {
      console.warn("Analysis session refresh failed; the saved session can be refreshed from history.", cause);
    }
  };

  const changeDatasetScope = async (ids: string[]): Promise<boolean> => {
    const currentSession = analysisSessionRef.current;
    const nextIds = [...ids].sort();
    const pinnedIds = [...(currentSession?.dataset_ids || [])].sort();
    const scopeChanged = JSON.stringify(nextIds) !== JSON.stringify(pinnedIds);
    if (currentSession?.status === "OPEN" && scopeChanged) {
      const proceed = window.confirm("Changing datasets will close the current analysis. Its saved questions and report history will be preserved. Continue?");
      if (!proceed) return false;
      setIsSessionBusy(true);
      setSessionError(null);
      try {
        await closeAnalysisSession(currentSession.id);
        commitAnalysisSession(null);
        setSessionTitleDraft("New analysis");
      } catch (cause) {
        setSessionError(cause instanceof Error ? cause.message : "Unable to close the current analysis before changing dataset scope.");
        return false;
      } finally {
        setIsSessionBusy(false);
      }
    } else if (currentSession?.status === "CLOSED" && scopeChanged) {
      // A closed session is immutable. Changing its dataset selection starts a new
      // session context instead of leaving a closed session attached to another dataset.
      commitAnalysisSession(null);
      setSessionTitleDraft("New analysis");
    }
    setSelectedDatasetIds(ids);
    setProposedPlan(null);
    setPlanNotice(null);
    setPreviewedGuidanceKey("");
    setClarificationInvestigationId(null);
    setDispatchRetryId(null);
    setAnalysis(null);
    setChatMessages([]);
    setInterpretation(null);
    setFollowUps([]);
    setIsReportModalOpen(false);
    setError(null);
    return true;
  };

  const busy = isLoading || isPlanning || isUploading || loadingDatasets || isSessionBusy;
  const journeyStep = isLoading ? 3 : analysis ? 4 : proposedPlan ? 2 : selectedDatasetIds.length ? 1 : 0;

  const guidanceFingerprint = (fields: InterpretGuidance) => {
    const entries = Object.entries(fields).filter(([, value]) =>
      value !== undefined && value !== null && value !== "" &&
      !(Array.isArray(value) && value.length === 0)
    ).sort(([a], [b]) => a.localeCompare(b));
    return entries.length ? JSON.stringify(Object.fromEntries(entries)) : "";
  };

  const handlePlanQuestion = async (q: string, guidance: InterpretGuidance = {}) => {
    if (!q.trim() || busy || !selectedDatasetIds.length) return;
    setClarificationInvestigationId(null);
    setDispatchRetryId(null);
    setIsPlanning(true);
    setError(null);
    setQuestion(q);
    setAnalysis(null);
    setProposedPlan(null);
    setPlanNotice(null);
    try {
      if (!projectId) {
        throw new Error("No local analytical workspace is available yet.");
      }
      const activeSession = await ensureOpenAnalysisSession();
      const plan = await interpretQuestion({
        question: q,
        project_id: projectId,
        analysis_session_id: activeSession.id,
        dataset_ids: selectedDatasetIds.length > 0 ? selectedDatasetIds : undefined,
        ...guidance,
      });
      // Keep the structured correction in the plan shown to the user. The panel will
      // initialize from it, so the next Run click approves the exact same guidance.
      setProposedPlan({ ...plan, guidance_context: guidance });
      setPreviewedGuidanceKey(guidanceFingerprint(guidance));
    } catch (err: any) {
      console.error("Pre-analysis plan interpretation failed:", err);
      // Invariant: An interpretation service failure must NOT silently bypass human-guidance review
      // or execute the unreviewed request. Expose the error transparently for human recovery.
      setError(
        `Pre-analysis interpretation could not compile a safe plan for this question: ${err?.message || err}. Please inspect your dataset columns or rephrase the question.`
      );
    } finally {
      setIsPlanning(false);
    }
  };

  const handleExecuteInvestigation = async (
    q: string,
    guidance: {
      selected_metric?: string;
      selected_aggregation?: string;
      selected_group_dimension?: string;
      selected_dataset?: string;
      confirmed_assumptions?: string[] | { add?: string[]; remove?: string[]; replace?: string[] };
      scope_filter?: string;
      clarification_response?: string;
      custom_instruction?: string;
      intent_patch?: Record<string, unknown>;
      pending_intent_slot?: string;
      plan_hash?: string;
    } = {},
    retryId?: string
  ) => {
    if (!q.trim() || busy) return;
    setIsLoading(true);
    setError(null);
    setPlanNotice(null);
    setProgressText("Queueing autonomous investigation…");
    setQuestion(q);

    try {
      if (!projectId) {
        throw new Error("No local analytical workspace is available yet.");
      }
      // Use the durable investigation plane rather than the synchronous chat
      // endpoint. The worker executes independently, while SSE events provide
      // live progress and the UI remains responsive for long-running analyses.
      const guidanceFields = {
        selected_metric: guidance.selected_metric,
        selected_aggregation: guidance.selected_aggregation,
        selected_group_dimension: guidance.selected_group_dimension,
        selected_dataset: guidance.selected_dataset,
        confirmed_assumptions: guidance.confirmed_assumptions,
        scope_filter: guidance.scope_filter,
        clarification_response: guidance.clarification_response,
        custom_instruction: guidance.custom_instruction,
        intent_patch: guidance.intent_patch,
        pending_intent_slot: guidance.pending_intent_slot,
      };
      const hasGuidance = Object.values(guidanceFields).some(
        (v) => v !== undefined && v !== null && v !== "" && !(Array.isArray(v) && v.length === 0)
      );
      const guidanceKey = hasGuidance ? guidanceFingerprint(guidanceFields) : "";
      if (!retryId && !clarificationInvestigationId && guidanceKey !== previewedGuidanceKey) {
        // The preview on screen was computed without this guidance. Re-preview it so the analyst
        // approves the plan that will actually run, then stop; they confirm again.
        const activeSession = await ensureOpenAnalysisSession();
        const refreshed = await interpretQuestion({
          question: q,
          project_id: projectId,
          analysis_session_id: activeSession.id,
          dataset_ids: selectedDatasetIds.length > 0 ? selectedDatasetIds : undefined,
          ...guidanceFields,
        });
        // Keep the exact guidance alongside the new preview. The panel is keyed by the plan
        // hash and remounts when the contract changes; without this context, radio choices,
        // custom filters, and the assumptions acknowledgement vanish, causing repeated re-previews.
        setProposedPlan({ ...refreshed, guidance_context: guidanceFields });
        setPreviewedGuidanceKey(guidanceKey);
        setProgressText(null);
        setPlanNotice(refreshed.can_auto_proceed && refreshed.ambiguity_status === "CLEAR"
          ? "Your decisions are now included in the preview. Review the updated calculation, then select Confirm & Run Analysis."
          : "Your decisions were applied, but the preview is still unresolved. Review the updated clarification and change the plan before execution.");
        return;
      }
      const requestGuidance = {
        selected_metric: guidance.selected_metric,
        selected_aggregation: guidance.selected_aggregation,
        selected_group_dimension: guidance.selected_group_dimension,
        selected_dataset: guidance.selected_dataset,
        confirmed_assumptions: guidance.confirmed_assumptions,
        scope_filter: guidance.scope_filter,
        clarification_response: guidance.clarification_response,
        custom_instruction: guidance.custom_instruction,
        intent_patch: guidance.intent_patch,
        pending_intent_slot: guidance.pending_intent_slot,
        plan_hash: guidance.plan_hash || proposedPlan?.plan_hash,
      };
      const investigation = retryId
        ? await retryInvestigationDispatch(retryId)
        : clarificationInvestigationId
          ? await reviseInvestigationGuidance(clarificationInvestigationId, requestGuidance)
          : await createInvestigation({
              question: q,
              project_id: projectId,
              analysis_session_id: (await ensureOpenAnalysisSession()).id,
              dataset_ids: selectedDatasetIds.length > 0 ? selectedDatasetIds : undefined,
              ...requestGuidance,
            });

      if (investigation.status === "NEEDS_CLARIFICATION") {
        setClarificationInvestigationId(investigation.investigation_id);
        setDispatchRetryId(null);
        setError(null);
        setProposedPlan((current) => current ? {
          ...current,
          ambiguity_status: "AMBIGUOUS_SCOPE",
          clarification_prompt:
            investigation.resolution?.reason ||
            "More detail is needed before this analysis can be executed.",
          clarification_options: investigation.resolution?.suggested_resolutions || [],
          resolution_status: investigation.resolution?.status,
          resolution_reason: investigation.resolution?.reason,
          unresolved_instruction: investigation.resolution?.unresolved_instruction,
          suggested_resolutions: investigation.resolution?.suggested_resolutions || [],
          conflict: investigation.resolution?.conflict || null,
          can_auto_proceed: false,
        } : current);
        setChatMessages((prev) => [
          ...prev,
          { role: "user", text: q },
          {
            role: "assistant",
            text: investigation.resolution?.reason || "Please clarify the request before execution.",
          },
        ]);
        return;
      }

      if (investigation.status === "READY") {
        setDispatchRetryId(investigation.investigation_id);
        setError(
          investigation.dispatch_error ||
          "The canonical contract is locked, but dispatch failed. Retry dispatch without resubmitting the analysis."
        );
        return;
      }

      setClarificationInvestigationId(null);
      setDispatchRetryId(null);
      setProposedPlan(null);
      setPlanNotice(null);
      setPreviewedGuidanceKey("");

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
          analyst_result: inv.analyst_result || null,
          execution_record: inv.execution_record || null,
          report_contract: inv.report_contract || null,
          provenance: inv.provenance || undefined,
          evidence_verifications: inv.evidence_verifications || [],
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
            calculation_summary: e.calculation_summary || "",
            dataset_version: e.source_dataset
              ? JSON.stringify(e.source_dataset)
              : undefined,
            row_count_analyzed: Number.isFinite(e.row_count_analyzed)
              ? Number(e.row_count_analyzed)
              : undefined,
            sql_executed: e.sql_executed || undefined,
            p_value: e.p_value,
            validation_status: e.validation_status || "UNVERIFIED",
            raw_metrics: {},
            verified_at: e.verified_at,
            effect_size: e.effect_size,
            calculation_trace: e.calculation_trace,
          })),
          suggested_followups: [],
          executive_bullets: [],
          analysis_plan: inv.analysis_plan || undefined,
          decision_impact: inv.decision_impact || undefined,
          // The detail API currently records scope under provenance, not at the
          // top level. Retain the submitted scope for an independent comparison.
          dataset_scope: inv.requested_dataset_ids ?? selectedDatasetIds,
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
          onError: (streamError) => {
            console.warn("Live investigation progress is unavailable; status polling continues.", streamError);
            setProgressText("Live progress is unavailable; checking the investigation status directly…");
          },
        }
      );

      try {
        const queuedAt = Date.now();
        let queueNoticeShown = false;
        while (!stopped) {
          const current = await fetchInvestigation(investigation.investigation_id);
          const status = String(current.status || "").toUpperCase();
          if (["COMPLETED", "FAILED", "CANCELLED", "PARTIAL"].includes(status)) {
            const mapped = toAnalysisResponse(current);
            setAnalysis(mapped);
            void refreshAnalysisSession();
            setActiveTab("findings");
            if (status === "FAILED") {
              setError(current.error_message || "This run did not produce a result. Review the interpretation and try a more specific question. The failed run is saved in History.");
            }
            break;
          }
          const queuedForSeconds = Math.floor((Date.now() - queuedAt) / 1000);
          if (status === "QUEUED" && queuedForSeconds >= 30) {
            queueNoticeShown = true;
            setProgressText(
              `Still queued after ${queuedForSeconds}s. The local worker has not claimed this request yet; keep this page open and do not submit it again.`
            );
          } else {
            setProgressText(
              queueNoticeShown && status !== "QUEUED"
                ? `Worker claimed the request • ${status || "RUNNING"}`
                : `AA-OS is investigating • ${status || "RUNNING"}`
            );
          }
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

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold text-cyan-400 mb-1">
            <Bot className="w-4 h-4" />
            <span>AUTONOMOUS ANALYTICAL INTELLIGENCE OPERATING SYSTEM (AA-OS)</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">From data to a clear answer</h1>
          <p className="text-xs text-slate-400 mt-1">
            Choose data, review your question, then explore the calculated result and its evidence.
          </p>
        </div>

        <label className="text-xs text-slate-400">Workspace
          <select aria-label="Workspace" value={projectId} disabled={busy}
            onChange={(event) => {
              const nextProjectId = event.target.value;
              void (async () => {
                const changed = await changeDatasetScope([]);
                if (!changed) return;
                setDatasets([]);
                setProjectId(nextProjectId);
              })();
            }}
            className="ml-2 rounded-lg border border-slate-700 bg-slate-900 p-2 text-slate-200">
            {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
          </select>
        </label>
      </div>

      <ol aria-label="Analysis journey" className="grid grid-cols-2 gap-2 sm:grid-cols-5">
        {["Choose data", "Ask a question", "Approve interpretation", "Run analysis", "Explore & export"].map((label, index) => (
          <li key={label} aria-current={journeyStep === index ? "step" : undefined}
            className={`rounded-xl border px-3 py-3 text-xs ${journeyStep === index ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--foreground)]" : "border-slate-800 text-slate-400"}`}>
            <span className="mr-2 font-mono">{index + 1}.</span>{label}
          </li>
        ))}
      </ol>

      <AnalyticsDataStarter projectId={projectId} disabled={isLoading || isPlanning || loadingDatasets}
        onBusyChange={setIsUploading}
        onUploaded={async (id, sample) => {
          const refreshed = await fetchDatasets(projectId);
          setDatasets(refreshed);
          const ids = Array.isArray(id) ? id : [id];
          const found = ids.filter((target) => refreshed.some((dataset) => dataset.id === target));
          if (found.length === 0) throw new Error("Upload finished, but the dataset list could not be refreshed. Reload the page to select it.");
          const changed = await changeDatasetScope(found);
          if (!changed) return;
          if (sample) setQuestion("What is the total revenue by region?");
        }} />

      <section aria-label="Dataset selection">
        {/* Dataset Workspace / Scope */}
        <div className="w-full xl:w-auto bg-slate-900 px-3 py-3 rounded-xl border border-slate-800 text-xs">
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <Filter className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">Dataset workspace:</span>
            <span className="text-cyan-300 font-medium">
              {selectedDatasetIds.length === 0 ? `Choose a dataset` : `${selectedDatasetIds.length} selected`}
            </span>
            {selectedDatasetIds.length > 0 && (
              <button
                type="button"
                disabled={busy}
                onClick={() => { void changeDatasetScope([]); }}
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
                  disabled={busy}
                  aria-pressed={selected}
                  onClick={() => { void changeDatasetScope(selected ? selectedDatasetIds.filter((id) => id !== d.id) : [...selectedDatasetIds, d.id]); }}
                  className={`px-2.5 py-1.5 rounded-lg border text-[11px] transition ${
                    selected
                      ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--foreground)]"
                      : "border-slate-700 bg-slate-950 text-slate-400 hover:text-white hover:border-slate-600"
                  }`}
                  title={`${d.name} • ${d.row_count?.toLocaleString()} rows • v${d.current_version}`}
                >
                  {d.name}
                  <span className="ml-1 text-[9px] opacity-60">{d.row_count?.toLocaleString()} rows</span>
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
        {loadingDatasets && <p role="status" className="mt-3 text-xs text-slate-400">Loading your workspace…</p>}
        {!loadingDatasets && !datasets.length && <p className="mt-3 text-sm text-slate-400">No datasets yet. Upload your CSV or try the sample above.</p>}
        {previewError && <p role="alert" className="mt-3 text-xs text-amber-300">Preview unavailable: {previewError}</p>}
        {selectedDatasetIds.length > 0 && recordSampleCols.length > 0 && <p className="mt-3 text-xs text-slate-400">Available columns: {recordSampleCols.join(", ")}</p>}
      </section>

      {/* One analysis session owns many question runs and one cumulative report. */}
      <section aria-label="Analysis session and cumulative report" className="rounded-2xl border border-slate-700 bg-slate-900/80 p-4 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-amber-300">
              <Layers className="h-4 w-4" /> Analysis session
            </div>
            <h2 className="mt-1 text-base font-semibold text-white">
              {analysisSession?.title || "Start a new analysis"}
            </h2>
            <p className="mt-1 max-w-3xl text-xs text-slate-400">
              Ask multiple questions in this session. Accepted, verified results are added to one cumulative PDF/DOCX. History is stored once per analysis, with individual questions inside it.
            </p>
          </div>
          {analysisSession && (
            <span className={`rounded-full border px-2.5 py-1 text-[10px] font-mono ${analysisSession.status === "OPEN" ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300" : "border-slate-600 bg-slate-800 text-slate-300"}`}>
              {analysisSession.status} · {analysisSession.question_count} question{analysisSession.question_count === 1 ? "" : "s"} · {analysisSession.accepted_question_count} accepted
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <label className="block text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Analysis title
            <input value={sessionTitleDraft} onChange={(event) => setSessionTitleDraft(event.target.value)} disabled={busy}
              placeholder={analysisSession?.status === "CLOSED" ? "Title for the next analysis" : "e.g. Q3 Delivery Performance"}
              className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-normal normal-case text-white placeholder:text-slate-600 disabled:opacity-60" />
          </label>
          <label className="block text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            {analysisSession?.status === "CLOSED" ? "Default filters for next analysis (optional)" : "Shared population filters (optional)"}
            <textarea value={sessionScopeDraft} onChange={(event) => setSessionScopeDraft(event.target.value)} disabled={busy} rows={2}
              placeholder="e.g. customer_segment = 'Small Business' AND quarter = 'Q3' AND delivery_completed_flag = 1"
              className="mt-1.5 w-full resize-y rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs font-mono normal-case text-white placeholder:text-slate-600 disabled:opacity-60" />
            <span className="mt-1 block font-normal normal-case text-slate-500">These filters define the default population for new questions. Each saved result retains its exact executed filters.</span>
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {!analysisSession || analysisSession.status === "CLOSED" ? (
            <button type="button" onClick={() => void handleNewAnalysis()} disabled={busy || !projectId || !selectedDatasetIds.length}
              className="rounded-lg bg-amber-400 px-3 py-2 text-xs font-semibold text-slate-950 hover:bg-amber-300 disabled:opacity-50">
              {analysisSession?.status === "CLOSED" ? "New Analysis" : "Start Analysis"}
            </button>
          ) : (
            <>
              <button type="button" onClick={() => void handleSaveSessionContext()} disabled={busy}
                className="rounded-lg border border-slate-600 px-3 py-2 text-xs font-semibold text-slate-200 hover:bg-slate-800 disabled:opacity-50">Save session context</button>
              <button type="button" onClick={() => void handleCloseAnalysis()} disabled={busy}
                className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs font-semibold text-amber-200 hover:bg-amber-500/20 disabled:opacity-50">Close Analysis</button>
            </>
          )}
          {analysisSession && analysisSession.accepted_question_count > 0 && (
            <>
              <button type="button" onClick={() => void handleExportSession("pdf")} disabled={!!sessionExporting}
                className="rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-xs font-semibold text-cyan-200 hover:bg-cyan-500/20 disabled:opacity-50">{sessionExporting === "pdf" ? "Preparing PDF…" : "Export cumulative PDF"}</button>
              <button type="button" onClick={() => void handleExportSession("docx")} disabled={!!sessionExporting}
                className="rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-xs font-semibold text-cyan-200 hover:bg-cyan-500/20 disabled:opacity-50">{sessionExporting === "docx" ? "Preparing DOCX…" : "Export cumulative DOCX"}</button>
            </>
          )}
          {analysisSession && <Link href="/history" className="px-2 py-2 text-xs text-slate-400 hover:text-white">View analysis history</Link>}
        </div>
        {sessionNotice && <p role="status" className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-2.5 text-xs text-emerald-200">{sessionNotice}</p>}
        {sessionError && <p role="alert" className="rounded-lg border border-rose-500/30 bg-rose-500/5 p-2.5 text-xs text-rose-200">{sessionError}</p>}

        {!!analysisSession?.questions?.length && (
          <div className="border-t border-slate-800 pt-3">
            <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">Question trail</h3>
            <ol className="space-y-2">
              {analysisSession.questions.map((item, index) => (
                <li key={item.id} className="flex items-start gap-2 rounded-lg border border-slate-800 bg-slate-950/60 p-2.5">
                  <span className="mt-0.5 min-w-5 font-mono text-[10px] text-slate-500">{String(index + 1).padStart(2, "0")}</span>
                  <div className="min-w-0 flex-1">
                    <p className="break-words text-xs text-slate-200">{item.question}</p>
                    <p className="mt-1 text-[10px] text-slate-500">{item.status} · {item.review_status || "Not reviewed"}{item.accepted_into_report ? " · Included in report" : " · Not in report"}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        )}
      </section>

      {/* Query Input Bar */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl focus-within:border-cyan-500/50 transition-all">
        <label htmlFor="business-question" className="block text-base font-semibold text-white mb-2">2. Ask a business question</label>
        <p className="text-xs text-slate-400 mb-3">Use the column names above. Next, you’ll review what will be calculated before approving the run.</p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handlePlanQuestion(question);
          }}
          className="flex items-center gap-3"
        >
          <div className="relative flex-1">
            <input
              id="business-question"
              type="text"
              placeholder="Ask an analytical question (e.g. 'Which region has the highest revenue?', 'Does discount_pct affect quantity?', 'Why did revenue change?')..."
              value={question}
              onChange={(e) => { setQuestion(e.target.value); setProposedPlan(null); setPlanNotice(null); setPreviewedGuidanceKey(""); }}
              disabled={busy || analysisSession?.status === "CLOSED"}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/80 font-sans"
            />
          </div>

          <button
            type="submit"
            disabled={busy || analysisSession?.status === "CLOSED" || !question.trim() || !selectedDatasetIds.length}
            className="bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-semibold px-6 py-3 rounded-xl flex items-center space-x-2 shadow-lg shadow-cyan-600/30 transition-all flex-shrink-0"
          >
            {isPlanning ? (
              <>
                <Cpu className="w-4 h-4 animate-spin" />
                <span>Interpreting...</span>
              </>
            ) : isLoading ? (
              <>
                <Cpu className="w-4 h-4 animate-spin" />
                <span>Investigating...</span>
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                <span>Review interpretation</span>
              </>
            )}
          </button>
        </form>
      </div>

      {/* Pre-Analysis Planning Progress */}
      {isPlanning && (
        <div className="flex items-center gap-3 rounded-xl border border-cyan-500/20 bg-cyan-500/5 px-4 py-3 text-xs text-cyan-200 animate-pulse">
          <Cpu className="w-4 h-4 animate-spin text-cyan-400" />
          <span>Analyzing question structure, resolving physical columns, and compiling analytical contract…</span>
        </div>
      )}

      {/* Proposed Analysis Panel (Pre-execution confirmation) */}
      {proposedPlan && planNotice && <p role="status" className="rounded-xl border border-slate-700 bg-[var(--accent-soft)] p-4 text-sm text-[var(--foreground)]">{planNotice}</p>}
      {proposedPlan && (
        <ProposedAnalysisPanel
          key={`${clarificationInvestigationId || "new"}-${proposedPlan.question}-${proposedPlan.plan_hash || "no-hash"}`}
          plan={proposedPlan}
          availableDatasets={datasets.map((dataset) => ({ id: dataset.id, name: dataset.name }))}
          availableColumns={recordSampleCols}
          availableNumericCols={recordSampleCols.filter((c) => {
            const sample = recordSampleRows.find((row) => row?.[c] !== null && row?.[c] !== undefined)?.[c];
            return typeof sample === "number";
          })}
          availableCategoricalCols={recordSampleCols.filter((c) => {
            const sample = recordSampleRows.find((row) => row?.[c] !== null && row?.[c] !== undefined)?.[c];
            return sample !== undefined && typeof sample !== "number" && typeof sample !== "boolean";
          })}
          availableBooleanCols={recordSampleCols.filter((c) => {
            const sample = recordSampleRows.find((row) => row?.[c] !== null && row?.[c] !== undefined)?.[c];
            return typeof sample === "boolean";
          })}
          previewedGuidanceKey={previewedGuidanceKey}
          onConfirmAndRun={(customized) => handleExecuteInvestigation(proposedPlan.question, customized)}
          onCancel={() => {
            setProposedPlan(null);
            setPlanNotice(null);
            setClarificationInvestigationId(null);
          }}
          isRunning={busy}
        />
      )}

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
                  onClick={() => handlePlanQuestion(item)}
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
      {!analysis && !busy && !proposedPlan && selectedDatasetIds.length > 0 && (
        <details className="rounded-xl border border-slate-800 p-4">
          <summary className="cursor-pointer text-sm text-slate-300">More question ideas</summary>
          <p className="my-3 text-xs text-slate-400">Choose an idea only if your selected data contains the relevant columns.</p>
          <QuickAnalysisCenter isLoading={busy} onSelectQuestion={(q) => handlePlanQuestion(q)} />
        </details>
      )}

      {/* Error Display */}
      {error && (
        <div role="alert" className="bg-rose-500/10 border border-rose-500/30 rounded-2xl p-5 flex items-start space-x-3 text-rose-300 text-xs">
          <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-400 mt-0.5" />
          <div>
            <span className="font-bold block mb-1">We couldn’t continue</span>
            <p>{error}</p>
            {dispatchRetryId && (
              <button
                type="button"
                onClick={() => handleExecuteInvestigation(question, {}, dispatchRetryId)}
                disabled={isLoading}
                className="mt-3 rounded-lg border border-rose-400/40 bg-rose-500/10 px-3 py-2 font-semibold text-rose-100 hover:bg-rose-500/20 disabled:opacity-50"
              >
                Retry dispatch
              </button>
            )}
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
            <h3 className="text-base font-bold text-white">Your analysis is running</h3>
            <p className="text-xs text-slate-400 mt-1 max-w-md mx-auto">
              The local worker is calculating results and checking supporting evidence. Keep this page open; completed analyses are also available in History.
            </p>
          </div>
        </div>
      )}

      {/* Full Analysis Results */}
      {analysis && !isLoading && (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-900 p-4">
            <div role="status" className="text-sm text-slate-200">{analysis.status === "COMPLETED" ? "Analysis complete — review the findings and limitations below." : `Analysis ${String(analysis.status).toLowerCase()} — review the recorded outcome below.`}</div>
            <Link href="/history" className="text-xs text-cyan-300 hover:underline">Open saved analyses</Link>
            {analysisSession && <span className="text-xs text-slate-500">Results are added to the session report when accepted and verified.</span>}
          </div>
          <StructuredAnalysisAnswer
            analysis={analysis}
            cumulativeReportMode={!!analysisSession}
            onReviewChanged={() => { void refreshAnalysisSession(); }}
            onRedo={() => {
              if (analysis.question) {
                handlePlanQuestion(analysis.question);
              }
            }}
            onRequestCorrection={(correction) => {
              // Never splice instructions into the original question and re-parse them as
              // if they were one sentence. Preserve the original question and carry the
              // correction through the API's explicit guidance fields.
              const execution = (analysis.execution_record || {}) as Record<string, any>;
              const contract = (execution.approved_contract ||
                (analysis as any).locked_canonical_contract?.contract ||
                (analysis as any).canonical_contract || {}) as Record<string, any>;
              const approvedAggRaw = String(contract.requested_aggregation || execution.aggregation || "").toUpperCase();
              const approvedAgg = ({ MEAN: "AVG", AVERAGE: "AVG", TOTAL: "SUM", ROW_COUNT: "COUNT" } as Record<string, string>)[approvedAggRaw] || approvedAggRaw;
              const text = correction.toLowerCase();
              const correctionAgg = /\b(average|avg|mean)\b/.test(text) ? "AVG"
                : /\b(total|sum|summed)\b/.test(text) ? "SUM"
                : /\bmedian\b/.test(text) ? "MEDIAN"
                : /\b(count|number of rows|number of records)\b/.test(text) ? "COUNT"
                : /\bminimum|min value\b/.test(text) ? "MIN"
                : /\bmaximum|max value\b/.test(text) ? "MAX" : undefined;
              const target = String(contract.target_column || execution.target_column || "");
              const groups = Array.isArray(contract.grouping_columns) ? contract.grouping_columns : [];
              const existingGroup = String(groups[0] || execution.group_dimension || "");
              const aggregation = correctionAgg || (
                ["SUM", "AVG", "MEDIAN", "COUNT", "COUNT_DISTINCT", "MIN", "MAX"].includes(approvedAgg)
                  ? approvedAgg : undefined
              );
              const guidance: InterpretGuidance = {
                ...(target ? { selected_metric: target } : {}),
                ...(existingGroup ? { selected_group_dimension: existingGroup } : {}),
                ...(aggregation ? { selected_aggregation: aggregation } : {}),
                custom_instruction: correction.trim(),
              };
              handlePlanQuestion(analysis.question || question, guidance);
            }}
          />

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
              {Array.isArray(analysis.analysis_plan.executed_experiments) && analysis.analysis_plan.executed_experiments.length > 0 && (
                <div className="text-[11px] text-slate-400">
                  <span className="text-slate-500">Executed: </span>
                  {analysis.analysis_plan.executed_experiments.map((e: any) => `${String(e.code)} (${Number(e.rows_analyzed || 0).toLocaleString()} rows, ${String(e.status)})`).join("; ")}
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
          <div className="flex items-center space-x-2 overflow-x-auto border-b border-slate-800 pb-2">
            {[
              { id: "findings", label: "Findings & Actions", icon: <Lightbulb className="w-4 h-4" /> },
              { id: "records", label: "Record Inspector", icon: <Database className="w-4 h-4" /> },
              { id: "graph", label: "Investigation DAG", icon: <Network className="w-4 h-4" /> },
              { id: "evidence", label: `Evidence (${analysis.evidence?.length || 0})`, icon: <ShieldCheck className="w-4 h-4" /> },
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

          {/* TAB: Record Inspector */}
          {activeTab === "records" && (
            <RecordLevelInspector
              rows={
                recordSampleRows.length > 0
                  ? recordSampleRows
                  : Array.isArray(analysis.analyst_result?.records)
                  ? (analysis.analyst_result.records as Record<string, any>[])
                  : (analysis.evidence?.map(e => ({ id: e.id, statement: e.statement, rows: e.row_count_analyzed, status: e.validation_status })) || [])
              }
              columns={
                recordSampleCols.length > 0
                  ? recordSampleCols
                  : Array.isArray(analysis.analyst_result?.records) && analysis.analyst_result.records.length > 0
                  ? Object.keys(analysis.analyst_result.records[0] as object)
                  : ["id", "statement", "rows", "status"]
              }
              totalRowCount={
                totalRecordCount ||
                (Array.isArray(analysis.analyst_result?.records) ? analysis.analyst_result.records.length : 0) ||
                analysis.evidence?.length ||
                0
              }
              onScopeChange={(selectedRows, scopeDescription) => {
                console.log("Inspector focused subset:", selectedRows.length, scopeDescription);
              }}
            />
          )}

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
                  No separate investigation findings were recorded. See the canonical answer above.
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
              targetMetric={String(analysis.hypotheses?.[0]?.target_metric || analysis.analysis_plan?.estimand?.target_metric || "Target metric")}
              descriptiveAnswer={String(analysis.verdict || "").toUpperCase().includes("OBSERVED")}
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
