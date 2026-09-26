export type ConfidenceLevel =
  | "High confidence"
  | "Medium confidence"
  | "Low confidence"
  | "Insufficient evidence";

export type ValidationStatus = "PASSED" | "FAILED" | "WARNING" | "SKIPPED";

export type ClaimGateOutcome = "ANSWER" | "QUALIFIED_ANSWER" | "REFUSE";

export interface ClaimGateResult {
  outcome: ClaimGateOutcome;
  requested_claim: string;
  evidence_level: number;
  design_status: "KNOWN" | "ASSUMED" | "UNKNOWN";
  assumptions: string[];
  allowed_claim: string;
  blocked_claim?: string | null;
  reason: string;
  recovery_actions: string[];
  computed_evidence: Record<string, any>;
  refusal_id?: string | null;
  requested_level?: number;
  max_supported_level?: number;
  identification_strategy?: string | null;
  blocking_conditions: string[];
  missing_evidence: string[];
}

export type AnalyticalVerdict =
  | "OBSERVED"
  | "DIAGNOSED"
  | "STATISTICALLY_SIGNIFICANT"
  | "NO_DETECTABLE_EFFECT"
  | "PREDICTED"
  | "SIMULATED"
  | "CONFIRMED"
  | "REJECTED"
  | "REFUTED"
  | "INCONCLUSIVE"
  | "INSUFFICIENT_DATA"
  | "CONFLICTING_EVIDENCE";

export interface DataQualityBreakdown {
  missing_values_score: number;
  duplicate_rows_score: number;
  type_consistency_score: number;
  outlier_risk_score: number;
  completeness?: number;
  validity?: number;
  uniqueness?: number;
  consistency?: number;
  temporal_stability?: number;
}

export interface MultiVectorConfidence {
  evidence_quality: number;
  data_quality: number;
  statistical_strength: number;
  causal_evidence: string;
  model_reliability: number;
  overall_verdict: string;
}

export interface EvidenceGraphNode {
  finding_id: string;
  evidence_id: string;
  tool_used: string;
  query_executed?: string;
  table_name: string;
  dataset_version: number;
  validation_status: string;
  tolerance_verified: boolean;
}

export interface DiscoverySummary {
  grain: string;
  candidate_keys: string[];
  primary_metrics: string[];
  primary_dimensions: string[];
  time_span?: string;
  worthy_inquiries: string[];
}

export interface AuditSummary {
  data_quality_score: number;
  total_rows: number;
  null_columns_count: number;
  duplicate_rows_count: number;
  anomalous_outliers_count: number;
  temporal_drift_detected: boolean;
}

export interface CalculationTraceStep {
  step_id: string;
  kind: string;
  title: string;
  formula?: string;
  executable_expression?: string;
  inputs: Record<string, any>;
  parameters: Record<string, any>;
  output: Record<string, any>;
  source_columns: string[];
  source_row_scope?: string;
  execution_engine?: string;
  parent_step_ids: string[];
  verification_status?: string;
  notes?: string;
}

export interface CalculationTrace {
  trace_id: string;
  trace_version: string;
  investigation_id: string;
  experiment_id: string;
  dataset_fingerprints: Record<string, string>;
  input_row_count: number;
  output_row_count: number;
  steps: CalculationTraceStep[];
  final_output: Record<string, any>;
  reproducibility_statement: string;
  canonical_hash: string;
}

export interface EvidenceItem {
  id: string;
  finding_id?: string;
  statement: string;
  calculation_summary: string;
  dataset_version: string;
  row_count_analyzed: number;
  time_window?: string;
  sql_executed?: string;
  python_executed?: string;
  statistical_test?: string;
  p_value?: number;
  effect_size?: number;
  validation_status: ValidationStatus;
  raw_metrics: Record<string, any>;
  verified_at: string;
  calculation_trace?: CalculationTrace;
}

export interface Finding {
  id: string;
  title: string;
  summary: string;
  importance_score: number;
  impact_magnitude?: string;
  confidence: ConfidenceLevel;
  verdict?: AnalyticalVerdict;
  evidence_items: EvidenceItem[];
  suggested_chart_type?: string;
  chart_data?: any;
  recommended_actions: string[];
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  org_id: string;
  owner_id: string;
}

export interface Hypothesis {
  id: string;
  statement: string;
  rationale: string;
  priority: number;
  prior_probability?: number;
  posterior_probability?: number;
  status: string;
  reason_for_rejection?: string;
  investigation_steps: string[];
  confirmed_findings: string[];
}

export interface AnalysisStep {
  step_number: number;
  title: string;
  action_type: string;
  description: string;
  tool_name?: string;
  tool_input?: any;
  tool_output?: any;
  duration_ms: number;
  status: string;
  error_message?: string;
}

export interface AnalysisManifest {
  analysis_id: string;
  user_question: string;
  project_id: string;
  dataset_versions_used: Array<{ table: string; row_count: number; version: number }>;
  ai_provider: string;
  ai_model: string;
  prompt_versions: Record<string, string>;
  created_at: string;
  execution_time_seconds: number;
  total_steps: number;
  replan_count?: number;
  validation_summary: Record<string, any>;
  reproducible_hash: string;
}

export interface AnalysisResponse {
  id: string;
  project_id: string;
  question: string;
  dataset_scope?: string[];
  status: string;
  verdict?: AnalyticalVerdict;
  direct_answer: string;
  main_finding: string;
  confidence: ConfidenceLevel;
  confidence_breakdown?: MultiVectorConfidence;
  discovery?: DiscoverySummary;
  audit?: AuditSummary;
  hypotheses: Hypothesis[];
  steps: AnalysisStep[];
  findings: Finding[];
  evidence: EvidenceItem[];
  evidence_graph?: EvidenceGraphNode[];
  suggested_followups?: string[];
  executive_bullets?: string[];
  what_if_scenarios?: Array<{
    scenario: string;
    assumption: string;
    projected_impact: string;
    confidence: string;
    uncertainty_note?: string;
  }>;
  analysis_plan?: Record<string, any>;
  cross_dataset_review?: {
    status: string;
    selected_dataset_count: number;
    requested_dataset_ids: string[];
    datasets: Array<{
      dataset_id: string;
      dataset_name?: string | null;
      selected: boolean;
      used_by_experiments: string[];
      used_in_verified_relational_claim: boolean;
    }>;
    relational_experiments: Array<{
      experiment_id: string;
      experiment_code: string;
      status?: string;
      source_tables: string[];
      join_hops: Array<Record<string, any>>;
      sql?: string | null;
      evidence_ids: string[];
      verification_statuses: string[];
      independently_verified: boolean;
    }>;
    join_safety_block_count: number;
    review_checks: Record<string, boolean | null>;
    review_message: string;
  };
  claim_gate?: ClaimGateResult;
  decision_impact?: Record<string, any>;
  manifest?: AnalysisManifest;
  created_at: string;
}

export interface ColumnProfile {
  name: string;
  data_type: string;
  semantic_type: string;
  null_count: number;
  null_percentage: number;
  unique_count: number;
  cardinality_ratio: number;
  min_value?: any;
  max_value?: any;
  mean?: number;
  median?: number;
  std_dev?: number;
  variance?: number;
  mode_value?: any;
  mode_frequency?: number;
  sample_values?: any[];
}

export interface DatasetVersion {
  id: string;
  dataset_id: string;
  version_number: number;
  row_count: number;
  column_count: number;
  size_bytes: number;
  parquet_path: string;
  schema_hash: string;
  data_quality_score?: number;
  column_profiles_json?: Record<string, ColumnProfile>;
  created_at: string;
}

export interface Dataset {
  id: string;
  name: string;
  description?: string;
  format: string;
  current_version: number;
  row_count: number;
  column_count: number;
  data_quality_score?: number;
  source_filename?: string;
  source_sha256?: string;
  raw_file_path?: string;
  profile?: {
    data_quality?: any;
    profiled_at?: string;
    columns?: Array<{
      name: string;
      data_type: string;
      null_percentage?: number;
      unique_count?: number;
      mean_value?: number;
      median_value?: number;
      std_dev?: number;
      variance?: number;
      mode_value?: any;
      mode_frequency?: number;
      min_value?: any;
      max_value?: any;
    }>;
  };
  updated_at: string;
}

export interface BusinessMetric {
  id: string;
  name: string;
  display_name: string;
  description: string;
  sql_formula: string;
  unit?: string;
  category?: string;
}

export interface GlossaryTerm {
  id: string;
  term: string;
  definition: string;
  category?: string;
}

export interface AlertRule {
  id: string;
  name: string;
  metric_name: string;
  threshold_pct_change: number;
  frequency: string;
  severity: string;
  is_active: boolean;
}

export interface AlertEvent {
  id: string;
  rule_id: string;
  title: string;
  severity: string;
  description: string;
  current_value: number;
  expected_value: number;
  deviation_percentage: number;
  triggered_at: string;
}

// --- Human verification workflow (AA-OS works, the analyst verifies) ---------
export type VerificationPriority = "BLOCKER" | "REVIEW" | "SPOT_CHECK" | "INFO";
export type VerificationMachineStatus = "PASSED" | "FAILED" | "WARNING" | "NOT_RUN" | "NEEDS_HUMAN";
export type VerificationDecision = "CONFIRMED" | "REJECTED" | "NEEDS_REWORK";
export type VerificationSignoffOutcome = "VERIFIED" | "REJECTED" | "NEEDS_REWORK";

export interface VerificationItem {
  id: string;
  category: string;
  priority: VerificationPriority;
  machine_status: VerificationMachineStatus;
  title: string;
  machine_did: string;
  you_confirm: string;
  why_it_matters: string;
  evidence: Record<string, any>;
  reproduce?: { dataset?: string | null; table_alias?: string; sql?: string | null } | null;
  requires_decision: boolean;
  fingerprint: string;
}

export interface VerificationPacket {
  schema_version: string;
  investigation_id: string;
  manifest_hash?: string | null;
  packet_fingerprint: string;
  question: string;
  machine_claim: { verdict?: string | null; headline?: string | null; confidence?: number | null };
  summary: {
    total: number;
    blockers: number;
    needs_your_judgment: number;
    optional_spot_checks: number;
    context_only: number;
    machine_checked_and_passed: number;
    decisions_required: number;
  };
  items: VerificationItem[];
  signoff_policy: string;
}

export interface VerificationReview {
  state: "NOT_STARTED" | "IN_PROGRESS" | "READY_TO_SIGN" | "OBJECTIONS_RAISED";
  per_item: Record<string, { decision: VerificationDecision; comment?: string; reviewer?: { email?: string }; at?: string; stale: boolean }>;
  required_item_ids: string[];
  pending: string[];
  confirmed: string[];
  rejected: string[];
  needs_rework: string[];
  stale: string[];
  can_verify: boolean;
  blocking_reasons: string[];
}

export interface VerificationSignoff {
  outcome: VerificationSignoffOutcome;
  comment?: string;
  reviewer?: { email?: string };
  at?: string;
  stale: boolean;
}

export interface VerificationState {
  packet: VerificationPacket;
  review: VerificationReview;
  signoff: VerificationSignoff | null;
}
