import {
  AlertEvent,
  AlertRule,
  AnalysisResponse,
  BusinessMetric,
  Dataset,
  GlossaryTerm,
  MonitoringStatus,
  Project,
  VerificationDecision,
  VerificationSignoffOutcome,
  VerificationState,
} from "../types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "/api/v1";

/**
 * Durable AA-OS investigation lifecycle.
 * Keep these values aligned with the backend state machine.
 */
export type InvestigationStatus =
  | "PLANNED"
  | "QUEUED"
  | "CLAIMED"
  | "RUNNING"
  | "WAITING_FOR_USER"
  | "WAITING_FOR_RESOURCE"
  | "VERIFYING"
  | "COMPLETED"
  | "PARTIAL"
  | "FAILED"
  | "CANCELLED";

export type InvestigationEventType =
  | "investigation.created"
  | "investigation.queued"
  | "investigation.started"
  | "step.started"
  | "step.completed"
  | "experiment.started"
  | "experiment.completed"
  | "evidence.created"
  | "verification.completed"
  | "belief.updated"
  | "waiting_for_user"
  | "waiting_for_resource"
  | "investigation.completed"
  | "investigation.failed"
  | "investigation.cancelled"
  | string;

export interface Investigation {
  analysis_mode?: "DETERMINISTIC" | "AI_AUGMENTED";
  ai_status_message?: string;
  explanation?: {
    mode: string;
    summary: string;
    what_was_analyzed: string;
    evidence: string[];
    conclusion: string;
    limitations: string[];
    verification: string;
    ai_rewrite?: string;
  };
  id: string;
  project_id: string;
  question: string;
  status: InvestigationStatus | string;
  objective?: string | null;
  current_step_id?: string | null;
  created_at?: string;
  started_at?: string | null;
  completed_at?: string | null;
  error_code?: string | null;
  error_message?: string | null;
  requested_dataset_ids?: string[];
  [key: string]: unknown;
}

export interface AnalysisHistoryRecord {
  id: string;
  record_type: "investigation" | "legacy_analysis";
  project_id: string;
  project_name: string;
  question: string;
  created_at: string | null;
  updated_at: string | null;
  status: string;
  headline: string | null;
  direct_answer: string | null;
  analyst_result: Record<string, unknown> | null;
  saved_result?: Record<string, unknown> | null;
  verification_status: string;
  claim_status: string;
  source_datasets: Array<{
    dataset_id: string | null;
    name: string;
    dataset_version: number | string | null;
    dataset_version_id: string | null;
    fingerprint: string | null;
    available: boolean;
  }>;
  evidence_ids: string[];
  [key: string]: unknown;
}

export interface AnalysisHistoryResponse {
  records: AnalysisHistoryRecord[];
  projects: Array<{ id: string; name: string; description?: string | null }>;
}

export interface InvestigationEvent {
  id?: string;
  sequence: number;
  investigation_id: string;
  event_type: InvestigationEventType;
  created_at?: string;
  payload?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface InvestigationDecision {
  decision: string;
  value?: unknown;
}

export interface ApiErrorBody {
  code?: string;
  message?: string;
  detail?: string;
  details?: unknown;
  request_id?: string;
}

export class AAOSApiError extends Error {
  code?: string;
  status: number;
  details?: unknown;
  requestId?: string;

  constructor(
    message: string,
    status: number,
    code?: string,
    details?: unknown,
    requestId?: string
  ) {
    super(message);
    this.name = "AAOSApiError";
    this.status = status;
    this.code = code;
    this.details = details;
    this.requestId = requestId;
  }
}

let cachedToken: string | null = null;
let tokenBootstrapPromise: Promise<string> | null = null;

/**
 * Demo authentication is intentionally not an automatic production fallback.
 *
 * In hosted/production mode, the application should obtain authentication
 * through its real login/session mechanism and call setAuthToken() only when
 * a bearer-token flow is explicitly being used.
 *
 * Local/demo mode may optionally expose a demo token endpoint, but this
 * helper never assumes that endpoint exists.
 */
export async function getValidToken(): Promise<string> {
  if (cachedToken) return cachedToken;

  if (typeof window !== "undefined") {
    const stored = sessionStorage.getItem("aa_os_token");
    if (stored) {
      cachedToken = stored;
      return stored;
    }

    // Local-first bootstrap only. Never auto-provision credentials against a
    // non-local origin. Production authentication remains explicit.
    const localMode = process.env.NEXT_PUBLIC_LOCAL_MODE === "true" || /localhost|127\.0\.0\.1/.test(window.location.hostname);
    if (localMode) {
      if (!tokenBootstrapPromise) {
        tokenBootstrapPromise = (async () => {
          try {
            const res = await fetch(`${API_BASE}/auth/demo-token`, { method: "POST" });
            if (res.ok) {
              const body = await res.json();
              if (body?.access_token) {
                setAuthToken(body.access_token);
                return body.access_token;
              }
            }
          } catch {
            // Let the original caller surface the real connectivity error.
          } finally {
            tokenBootstrapPromise = null;
          }
          return "";
        })();
      }
      return tokenBootstrapPromise;
    }
  }

  return "";
}

export function setAuthToken(token: string) {
  cachedToken = token;

  if (typeof window !== "undefined") {
    sessionStorage.setItem("aa_os_token", token);
  }
}

export function clearAuthToken() {
  cachedToken = null;
  tokenBootstrapPromise = null;

  if (typeof window !== "undefined") {
    sessionStorage.removeItem("aa_os_token");
  }
}

async function parseApiError(res: Response): Promise<AAOSApiError> {
  let body: ApiErrorBody | null = null;

  try {
    body = (await res.json()) as ApiErrorBody;
  } catch {
    // Response may not contain JSON.
  }

  const message =
    body?.message ||
    body?.detail ||
    res.statusText ||
    `Request failed with HTTP ${res.status}`;

  return new AAOSApiError(
    message,
    res.status,
    body?.code,
    body?.details,
    body?.request_id
  );
}

async function authFetch(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const token = await getValidToken();

  const headers = new Headers(options.headers || {});

  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  return fetch(url, {
    ...options,
    headers,
  });
}

async function authFetchJson<T>(
  url: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await authFetch(url, options);

  if (!res.ok) {
    throw await parseApiError(res);
  }

  return (await res.json()) as T;
}

/* -------------------------------------------------------------------------- */
/* Projects                                                                   */
/* -------------------------------------------------------------------------- */

export async function fetchProjects(): Promise<Project[]> {
  const projects = await authFetchJson<Project[]>(`${API_BASE}/projects`);
  const localMode = typeof window !== "undefined" && (process.env.NEXT_PUBLIC_LOCAL_MODE === "true" || /localhost|127\.0\.0\.1/.test(window.location.hostname));
  if (localMode && projects.length === 0) {
    const created = await authFetchJson<Project>(`${API_BASE}/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "Local Analytical Workspace", description: "Automatically initialized local AA-OS workspace." }),
    });
    return [created];
  }
  return projects;
}

/* -------------------------------------------------------------------------- */
/* Datasets                                                                   */
/* -------------------------------------------------------------------------- */

export async function fetchDatasets(projectId?: string): Promise<Dataset[]> {
  const url = projectId
    ? `${API_BASE}/datasets?project_id=${encodeURIComponent(projectId)}`
    : `${API_BASE}/datasets`;

  return authFetchJson<Dataset[]>(url);
}

export async function fetchDatasetDetails(
  datasetId: string,
  version?: number
): Promise<Dataset> {
  const params = new URLSearchParams();
  if (version !== undefined) params.set("version", String(version));
  const suffix = params.toString();
  return authFetchJson<Dataset>(
    `${API_BASE}/datasets/${encodeURIComponent(datasetId)}${suffix ? `?${suffix}` : ""}`,
    { cache: "no-store" }
  );
}

export async function fetchDatasetData(
  datasetId: string,
  version?: number,
  limit: number = 50
) {
  const params = new URLSearchParams();
  params.set("limit", String(limit));

  if (version !== undefined) {
    params.set("version", String(version));
  }

  return authFetchJson(
    `${API_BASE}/datasets/${encodeURIComponent(datasetId)}/data?${params.toString()}`
  );
}

export async function uploadDatasetFile(formData: FormData) {
  const res = await authFetch(`${API_BASE}/datasets/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw await parseApiError(res);
  }

  return res.json();
}

/* -------------------------------------------------------------------------- */
/* Investigations — canonical AA-OS analytical API                           */
/* -------------------------------------------------------------------------- */

export interface CreateInvestigationInput {
  question: string;
  project_id: string;
  dataset_ids?: string[];
  monitoring_event_id?: string;
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
}

export interface PreAnalysisPlan {
  question: string;
  task: string;
  proposed_interpretation: string;
  planned_calculation: string;
  target_metric: string | null;
  group_dimension: string | null;
  time_period: string | null;
  scope: string;
  assumptions: string[];
  ambiguity_status: "CLEAR" | "AMBIGUOUS_METRIC" | "MISSING_DATA" | "AMBIGUOUS_SCOPE" | "JOIN_FANOUT_RISK" | "CAUSAL_ON_OBSERVATIONAL" | "VERIFICATION_FAILED" | string;
  clarification_prompt?: string | null;
  clarification_options: string[];
  can_auto_proceed: boolean;
  recommended_action: string;
  resolution_status?: string;
  resolution_reason?: string | null;
  unresolved_instruction?: string | null;
  suggested_resolutions?: string[];
  conflict?: Record<string, unknown> | null;
  plan_hash?: string;
  canonical_contract?: Record<string, unknown> | null;
  clarification_option_patches?: Array<{ label: string; patch: Record<string, unknown> }>;
  pending_intent_slot?: string | null;
}

export interface CreateInvestigationResponse {
  investigation_id: string;
  job_id: string | null;
  status: string;
  question: string;
  project_id: string;
  created_at: string | null;
  resolution?: {
    status: string;
    reason?: string | null;
    unresolved_instruction?: string | null;
    suggested_resolutions?: string[];
    conflict?: Record<string, unknown> | null;
  };
  dispatch_error?: string;
  retry_endpoint?: string;
}

export interface InterpretGuidance {
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
}

export async function interpretQuestion(input: {
  question: string;
  project_id: string;
  dataset_ids?: string[];
} & InterpretGuidance): Promise<PreAnalysisPlan> {
  return authFetchJson<PreAnalysisPlan>(`${API_BASE}/investigations/interpret`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question: input.question.trim(),
      project_id: input.project_id,
      dataset_ids: input.dataset_ids,
      selected_metric: input.selected_metric,
      selected_aggregation: input.selected_aggregation,
      selected_group_dimension: input.selected_group_dimension,
      selected_dataset: input.selected_dataset,
      confirmed_assumptions: input.confirmed_assumptions,
      scope_filter: input.scope_filter,
      clarification_response: input.clarification_response,
      custom_instruction: input.custom_instruction,
      intent_patch: input.intent_patch,
      pending_intent_slot: input.pending_intent_slot,
    }),
  });
}

export async function submitHumanReview(
  investigationId: string,
  payload: {
    action: "ACCEPT" | "REQUEST_CORRECTION" | "REJECT" | "REDO";
    comments?: string;
    modified_parameters?: Record<string, unknown>;
  }
): Promise<{ status: string; review: Record<string, unknown>; investigation_id: string }> {
  return authFetchJson(`${API_BASE}/investigations/${encodeURIComponent(investigationId)}/review`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function createInvestigation(
  input: CreateInvestigationInput
): Promise<CreateInvestigationResponse> {
  if (!input.question.trim()) {
    throw new AAOSApiError(
      "Investigation question is required.",
      400,
      "QUESTION_REQUIRED"
    );
  }

  if (!input.project_id) {
    throw new AAOSApiError(
      "Project ID is required.",
      400,
      "PROJECT_REQUIRED"
    );
  }

  return authFetchJson<CreateInvestigationResponse>(`${API_BASE}/investigations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question: input.question.trim(),
      project_id: input.project_id,
      dataset_ids: input.dataset_ids,
      monitoring_event_id: input.monitoring_event_id,
      selected_metric: input.selected_metric,
      selected_aggregation: input.selected_aggregation,
      selected_group_dimension: input.selected_group_dimension,
      selected_dataset: input.selected_dataset,
      confirmed_assumptions: input.confirmed_assumptions,
      scope_filter: input.scope_filter,
      clarification_response: input.clarification_response,
      custom_instruction: input.custom_instruction,
      intent_patch: input.intent_patch,
      pending_intent_slot: input.pending_intent_slot,
      plan_hash: input.plan_hash,
    }),
  });
}

/** Append a correction to a queued investigation without replacing prior guidance. */
export async function reviseInvestigationGuidance(
  investigationId: string,
  guidance: Omit<CreateInvestigationInput, "question" | "project_id" | "dataset_ids" | "monitoring_event_id">
): Promise<CreateInvestigationResponse> {
  return authFetchJson(`${API_BASE}/investigations/${encodeURIComponent(investigationId)}/guidance`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(guidance),
  });
}

export async function retryInvestigationDispatch(
  investigationId: string
): Promise<CreateInvestigationResponse> {
  return authFetchJson(`${API_BASE}/investigations/${encodeURIComponent(investigationId)}/dispatch`, {
    method: "POST",
  });
}

export async function listInvestigations(
  projectId?: string
): Promise<Investigation[]> {
  const url = projectId
    ? `${API_BASE}/investigations?project_id=${encodeURIComponent(projectId)}`
    : `${API_BASE}/investigations`;

  return authFetchJson<Investigation[]>(url);
}

export async function fetchAnalysisHistory(limit = 100): Promise<AnalysisHistoryResponse> {
  return authFetchJson<AnalysisHistoryResponse>(
    `${API_BASE}/investigations/history?limit=${encodeURIComponent(String(limit))}`,
    { cache: "no-store" }
  );
}

export async function clearAnalysisHistory(): Promise<{
  status: string;
  removed_investigations: number;
  removed_legacy_analyses: number;
  projects_and_datasets_preserved: boolean;
}> {
  return authFetchJson(`${API_BASE}/investigations/history`, {
    method: "DELETE",
    cache: "no-store",
  });
}

export async function fetchInvestigation(
  investigationId: string
): Promise<Investigation> {
  return authFetchJson<Investigation>(
    `${API_BASE}/investigations/${encodeURIComponent(investigationId)}`
  );
}

export async function fetchVerification(investigationId: string): Promise<VerificationState> {
  return authFetchJson<VerificationState>(
    `${API_BASE}/investigations/${encodeURIComponent(investigationId)}/verification`
  );
}

export async function recordVerificationDecision(
  investigationId: string,
  body: { item_id: string; decision: VerificationDecision; comment?: string; item_fingerprint?: string }
): Promise<VerificationState> {
  return authFetchJson<VerificationState>(
    `${API_BASE}/investigations/${encodeURIComponent(investigationId)}/verification/decisions`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
  );
}

export async function signOffVerification(
  investigationId: string,
  body: { outcome: VerificationSignoffOutcome; comment?: string; packet_fingerprint?: string }
): Promise<VerificationState> {
  return authFetchJson<VerificationState>(
    `${API_BASE}/investigations/${encodeURIComponent(investigationId)}/verification/signoff`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
  );
}

export async function fetchInvestigationExplanation(
  investigationId: string,
  mode: "DETERMINISTIC" | "AI_AUGMENTED" = "DETERMINISTIC"
): Promise<Investigation["explanation"] & { analysis_id: string; analysis_mode: string }> {
  return authFetchJson(`${API_BASE}/investigations/${encodeURIComponent(investigationId)}/explanation?mode=${mode}`);
}

export async function downloadInvestigationExport(
  investigationId: string,
  format: "pdf" | "docx"
): Promise<Blob> {
  const token = await getValidToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(
    `${API_BASE}/investigations/${encodeURIComponent(investigationId)}/export?format=${format}`,
    { headers }
  );
  if (!res.ok) {
    throw new Error(`Export failed with HTTP ${res.status}: ${res.statusText}`);
  }
  return res.blob();
}

export async function cancelInvestigation(
  investigationId: string
): Promise<Investigation> {
  return authFetchJson<Investigation>(
    `${API_BASE}/investigations/${encodeURIComponent(
      investigationId
    )}/cancel`,
    {
      method: "POST",
    }
  );
}

export async function submitInvestigationDecision(
  investigationId: string,
  decision: InvestigationDecision
): Promise<Investigation> {
  return authFetchJson<Investigation>(
    `${API_BASE}/investigations/${encodeURIComponent(
      investigationId
    )}/decision`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(decision),
    }
  );
}

/**
 * Deprecated compatibility wrapper.
 *
 * New UI code should use createInvestigation().
 *
 * This wrapper remains temporarily so existing pages can migrate one file
 * at a time without breaking the application.
 */
export async function runAnalysis(
  question: string,
  projectId: string,
  datasetIds?: string[]
): Promise<AnalysisResponse> {
  const investigation = await createInvestigation({
    question,
    project_id: projectId,
    dataset_ids: datasetIds,
  });

  return investigation as unknown as AnalysisResponse;
}

/* -------------------------------------------------------------------------- */
/* Investigation event streaming                                              */
/* -------------------------------------------------------------------------- */

export function streamInvestigationEvents(
  investigationId: string,
  onEvent: (event: InvestigationEvent) => void,
  options?: {
    lastEventId?: number;
    onError?: (error: Error) => void;
  }
): () => void {
  if (typeof window === "undefined") {
    return () => undefined;
  }

  const controller = new AbortController();
  let cancelled = false;

  const start = async () => {
    try {
    const token = await getValidToken();
    if (cancelled) return;
    if (!token) {
      throw new Error("Authentication is unavailable for investigation progress updates.");
    }
    const params = new URLSearchParams();
    if (options?.lastEventId !== undefined) {
      params.set("last_event_id", String(options.lastEventId));
    }
    const suffix = params.toString();
    const url =
      `${API_BASE}/investigations/${encodeURIComponent(
        investigationId
      )}/events` + (suffix ? `?${suffix}` : "");
    const response = await fetch(url, {
      headers: {
        Accept: "text/event-stream",
        Authorization: `Bearer ${token}`,
      },
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`Investigation progress stream returned HTTP ${response.status}.`);
    }
    if (!response.body) {
      throw new Error("Investigation progress stream returned no response body.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (!cancelled) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        const data = block
          .split(/\r?\n/)
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (!data) continue;
        try {
          const parsed = JSON.parse(data) as InvestigationEvent;
          onEvent(parsed);
        } catch (error) {
          console.warn("Invalid investigation SSE event:", error);
        }
      }
      if (done) break;
    }
    reader.releaseLock();
    } catch (error) {
    if (!cancelled && !(error instanceof DOMException && error.name === "AbortError")) {
      options?.onError?.(error instanceof Error ? error : new Error("Investigation progress stream failed."));
    }
    }
  };

  void start();

  return () => {
    cancelled = true;
    controller.abort();
  };
}

/* -------------------------------------------------------------------------- */
/* Metrics / semantic context                                                 */
/* -------------------------------------------------------------------------- */

export async function fetchMetrics(
  projectId?: string
): Promise<BusinessMetric[]> {
  const url = projectId
    ? `${API_BASE}/metrics?project_id=${encodeURIComponent(projectId)}`
    : `${API_BASE}/metrics`;

  return authFetchJson<BusinessMetric[]>(url);
}

export async function fetchGlossary(
  projectId?: string
): Promise<GlossaryTerm[]> {
  const url = projectId
    ? `${API_BASE}/glossary?project_id=${encodeURIComponent(projectId)}`
    : `${API_BASE}/glossary`;

  return authFetchJson<GlossaryTerm[]>(url);
}

/* -------------------------------------------------------------------------- */
/* Alerts                                                                      */
/* -------------------------------------------------------------------------- */

export async function fetchAlertRules(
  projectId?: string
): Promise<AlertRule[]> {
  const url = projectId
    ? `${API_BASE}/alerts/rules?project_id=${encodeURIComponent(projectId)}`
    : `${API_BASE}/alerts/rules`;

  return authFetchJson<AlertRule[]>(url);
}

export async function fetchAlertEvents(
  ruleId?: string,
  projectId?: string
): Promise<AlertEvent[]> {
  const params = new URLSearchParams();
  if (ruleId) params.set("rule_id", ruleId);
  if (projectId) params.set("project_id", projectId);
  const suffix = params.toString();
  const url = `${API_BASE}/alerts/events${suffix ? `?${suffix}` : ""}`;

  return authFetchJson<AlertEvent[]>(url);
}

export async function fetchMonitoringStatus(projectId: string): Promise<MonitoringStatus> {
  return authFetchJson<MonitoringStatus>(
    `${API_BASE}/alerts/status?project_id=${encodeURIComponent(projectId)}`,
    { cache: "no-store" }
  );
}

export async function evaluateAlertRules(projectId: string) {
  return authFetchJson<{
    status: string;
    project_id: string;
    trigger: string;
    evaluated_rule_count: number;
    triggered_count: number;
    outcomes: Array<Record<string, unknown>>;
    scheduler_active: boolean;
  }>(`${API_BASE}/alerts/evaluate?project_id=${encodeURIComponent(projectId)}`, {
    method: "POST",
    cache: "no-store",
  });
}

export async function transitionAlertEvent(
  eventId: string,
  action: "acknowledge" | "resolve" | "snooze",
  minutes?: number
): Promise<AlertEvent> {
  return authFetchJson<AlertEvent>(
    `${API_BASE}/alerts/events/${encodeURIComponent(eventId)}/${action}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: action === "snooze" ? JSON.stringify({ minutes: minutes ?? 60 }) : undefined,
      cache: "no-store",
    }
  );
}

/* -------------------------------------------------------------------------- */
/* Project-scoped operations                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Project ID is intentionally required.
 *
 * There is no production default project.
 */
export async function fetchProjectSummary(projectId: string) {
  if (!projectId) {
    throw new AAOSApiError(
      "Project ID is required.",
      400,
      "PROJECT_REQUIRED"
    );
  }

  return authFetchJson(
    `${API_BASE}/projects/summary?project_id=${encodeURIComponent(projectId)}`,
    { cache: "no-store" }
  );
}

export async function resetWorkspace(projectId: string): Promise<{
  status: string;
  message: string;
  removed_datasets: number;
  removed_dataset_versions: number;
  history_preserved: boolean;
  storage_cleanup_pending: boolean;
}> {
  if (!projectId) {
    throw new AAOSApiError(
      "Project ID is required.",
      400,
      "PROJECT_REQUIRED"
    );
  }

  return authFetchJson(
    `${API_BASE}/projects/reset?project_id=${encodeURIComponent(projectId)}`,
    {
      method: "POST",
    }
  );
}


/* -------------------------------------------------------------------------- */
/* Chat                                                                       */
/* -------------------------------------------------------------------------- */

export async function sendChatMessage(
  message: string,
  projectId: string,
  conversationId?: string,
  datasetIds?: string[],
  analysisMode: "DETERMINISTIC" | "AI_AUGMENTED" = "DETERMINISTIC"
) {
  if (!message.trim()) {
    throw new AAOSApiError(
      "Message is required.",
      400,
      "MESSAGE_REQUIRED"
    );
  }

  if (!projectId) {
    throw new AAOSApiError(
      "Project ID is required.",
      400,
      "PROJECT_REQUIRED"
    );
  }

  return authFetchJson(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: message.trim(),
      project_id: projectId,
      conversation_id: conversationId,
      dataset_ids: datasetIds,
      analysis_mode: analysisMode,
    }),
  });
}

/* -------------------------------------------------------------------------- */
/* AI settings                                                                */
/* -------------------------------------------------------------------------- */

export async function fetchAISettings() {
  return authFetchJson(`${API_BASE}/settings/ai`, {
    cache: "no-store",
  });
}

export async function updateAISettings(data: {
  enabled?: boolean;
  provider?: string;
  model?: string;
  api_key?: string;
  base_url?: string;
}) {
  return authFetchJson(`${API_BASE}/settings/ai`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
}

export async function testAIConnection(data: {
  provider: string;
  model?: string;
  api_key?: string;
  base_url?: string;
}) {
  return authFetchJson(`${API_BASE}/settings/ai/test`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
}

export async function fetchInfrastructureMatrix() {
  return authFetchJson(`${API_BASE}/settings/infrastructure`, {
    cache: "no-store",
  });
}

/* -------------------------------------------------------------------------- */
/* Optional cloud backup / continuity                                        */
/* -------------------------------------------------------------------------- */

export interface CloudBackupConfig {
  provider: "s3" | "gcs";
  bucket: string;
  region?: string;
  endpoint_url?: string;
  access_key_id?: string;
  secret_access_key?: string;
  project_id?: string;
}

export async function testCloudStorage(data: CloudBackupConfig) {
  return authFetchJson(`${API_BASE}/backups/cloud/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function createCloudBackup(data: CloudBackupConfig & { passphrase: string; include_source_files?: boolean }) {
  return authFetchJson(`${API_BASE}/backups/cloud/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function listCloudBackups(data: { provider: string; bucket: string; region?: string; endpoint_url?: string }) {
  const params = new URLSearchParams({ provider: data.provider, bucket: data.bucket });
  if (data.region) params.set("region", data.region);
  if (data.endpoint_url) params.set("endpoint_url", data.endpoint_url);
  return authFetchJson(`${API_BASE}/backups/cloud?${params.toString()}`, { cache: "no-store" });
}

export async function restoreCloudBackup(data: CloudBackupConfig & { object_key: string; passphrase: string; replace_existing?: boolean }) {
  return authFetchJson(`${API_BASE}/backups/cloud/restore`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function recoverUploadedBackup(file: File, passphrase: string, replaceExisting = false) {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams({ passphrase });
  params.set("replace_existing", String(replaceExisting));
  const res = await authFetch(`${API_BASE}/backups/recover-upload?${params.toString()}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function migrateLocalStorageToCloud(data: CloudBackupConfig) {
  return authFetchJson(`${API_BASE}/backups/cloud/migrate-local`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export interface QueryExecutionResult {
  columns?: string[];
  rows?: any[];
  execution_time_ms?: number;
  row_count?: number;
  detail?: string;
}

export async function executeQuery(sql: string, projectId: string): Promise<QueryExecutionResult> {
  return authFetchJson(`${API_BASE}/query/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sql, project_id: projectId }),
  });
}
