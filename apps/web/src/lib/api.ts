import {
  AlertEvent,
  AlertRule,
  AnalysisResponse,
  BusinessMetric,
  Dataset,
  GlossaryTerm,
  Project,
  VerificationDecision,
  VerificationSignoffOutcome,
  VerificationState,
} from "../types";

const API_BASE =
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
      }
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
  datasetId: string
): Promise<Dataset> {
  return authFetchJson<Dataset>(
    `${API_BASE}/datasets/${encodeURIComponent(datasetId)}`
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
}

export interface CreateInvestigationResponse {
  investigation_id: string;
  job_id: string;
  status: string;
  question: string;
  project_id: string;
  created_at: string;
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
    }),
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
    onError?: (error: Event) => void;
  }
): () => void {
  if (typeof window === "undefined") {
    return () => undefined;
  }

  const tokenPromise = getValidToken();
  let source: EventSource | null = null;
  let cancelled = false;

  const start = async () => {
    const token = await tokenPromise;

    if (cancelled) return;

    const params = new URLSearchParams();

    if (token) {
      params.set("token", token);
    }

    if (options?.lastEventId !== undefined) {
      params.set("last_event_id", String(options.lastEventId));
    }

    const suffix = params.toString();
    const url =
      `${API_BASE}/investigations/${encodeURIComponent(
        investigationId
      )}/events` + (suffix ? `?${suffix}` : "");

    source = new EventSource(url);

    source.onmessage = (message) => {
      try {
        const parsed = JSON.parse(message.data) as InvestigationEvent;
        onEvent(parsed);
      } catch (error) {
        console.warn("Invalid investigation SSE event:", error);
      }
    };

    source.onerror = (error) => {
      options?.onError?.(error);
    };
  };

  void start();

  return () => {
    cancelled = true;
    source?.close();
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
  ruleId?: string
): Promise<AlertEvent[]> {
  const url = ruleId
    ? `${API_BASE}/alerts/events?rule_id=${encodeURIComponent(ruleId)}`
    : `${API_BASE}/alerts/events`;

  return authFetchJson<AlertEvent[]>(url);
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

export async function resetWorkspace(projectId: string) {
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
