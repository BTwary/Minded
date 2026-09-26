import os
"""Production Investigation Runtime.

[THIN DELEGATING ADAPTER] This module previously contained a second,
fully independent ~900-line investigation loop (its own semantic-model
construction, hypothesis synthesis, experiment planning, and verdict
formulation) that ran in parallel to `packages.analytics_core.src.runtime
.controller.InvestigationController` -- the canonical, scientifically
rigorous investigation engine (dual-engine verification, EIG-optimized
experiment selection, Bayesian belief updates, adversarial challenge,
evidence-ledger-backed stopping). It accepted a `controller` constructor
argument but never called it (`self.controller` was written once and
never read again), so every investigation run through this class silently
bypassed the canonical engine entirely -- including the benchmark/
regression scripts that exist specifically to validate that engine
(`scripts/run_benchmarks.py`, `scripts/adversarial_audit_suite.py`,
`scripts/test_aa_os_reproducibility.py`,
`scripts/test_property_based_generalization.py`), which were unknowingly
validating dead legacy code instead.

`AnalysisService` (`apps/api/src/services/analysis_service.py`) already
replaced this class for the live HTTP request path. This class now exists
only for callers that need an in-memory-datasets -> `AnalysisResponse`
call shape (the AI-fallback and zero-AI-mode acceptance suites, and the
`AutonomousOrchestrator`/`AutonomousOrchestrator`-based benchmark
scripts) -- it drives the same real `InvestigationController` against an
ephemeral, self-contained SQLite database (the same pattern used in
`scripts/test_decision_recommendation_api_e2e.py`) and translates the
resulting persisted DB state into an `AnalysisResponse`, instead of
running a second, uncoordinated investigation engine.
"""
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.src.core.config import settings
from apps.api.src.ai.providers.base import BaseAIProvider
from apps.api.src.models.entities import (
    Base,
    DecisionRecommendationRecord,
    Evidence,
    Hypothesis,
    Investigation,
    InvestigationEvent,
    InvestigationVerdict,
    Observation,
    Project,
    User,
    gen_uuid,
)
from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider
from packages.analytics_core.src.intelligence.dataset_question_discovery import DatasetQuestionDiscovery
from packages.analytics_core.src.execution.state_machine import (
    InvestigationState as _ExecInvestigationState,
    to_public_validation_status,
)
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.schemas.src.analysis import (
    AnalysisManifestSchema,
    AnalysisResponse,
    AnalysisStepSchema,
    DecisionRecommendation,
    EvidenceSchema,
    ExpectedUtilityCalculation,
    FindingSchema,
    HypothesisSchema,
)
from packages.shared.src.enums import AnalysisStatus, AnalyticalVerdict, ConfidenceLevel

_INSUFFICIENT_VERDICTS = {"INSUFFICIENT_DATA", "INCONCLUSIVE", "CONFLICTING_EVIDENCE"}


def _confidence_level(verdict_type: str, score: Optional[float]) -> ConfidenceLevel:
    if verdict_type in _INSUFFICIENT_VERDICTS or score is None:
        return ConfidenceLevel.INSUFFICIENT_EVIDENCE
    if score >= 0.75:
        return ConfidenceLevel.HIGH
    if score >= 0.5:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


class InvestigationRuntime:
    """[THIN DELEGATING ADAPTER] Drives the real InvestigationController
    against an ephemeral, self-contained SQLite database and translates
    the persisted result into an AnalysisResponse. See module docstring.
    """

    def __init__(self, ai_provider: BaseAIProvider, controller: Optional[Any] = None):
        self.provider = ai_provider
        # Kept only for constructor-signature compatibility with existing
        # callers; each call below builds its own isolated
        # InvestigationController bound to an ephemeral in-memory database,
        # since callers of this adapter (benchmark/acceptance scripts) pass
        # raw in-memory datasets rather than a pre-populated project DB.
        self._controller_hint = controller

    def execute_investigation(
        self,
        question: str,
        project_id: str,
        datasets: Dict[str, pd.DataFrame],
        business_metrics: Optional[List[Dict[str, Any]]] = None,
        max_steps: int = 0,
    ) -> AnalysisResponse:
        analysis_id = f"ANA-{gen_uuid()[:8]}"

        if not datasets:
            return self._build_empty_response(analysis_id, project_id, question)

        start_time = time.time()

        engine = create_engine("sqlite:///:memory:")
        if settings.ENVIRONMENT.lower() != "production" and os.getenv("AAOS_AUTO_CREATE_DB_SCHEMA", "true").lower() in {"1", "true", "yes"}:
            Base.metadata.create_all(engine)
        SessionFactory = sessionmaker(bind=engine)

        investigation_id = f"INV-{gen_uuid()[:12]}"
        with SessionFactory() as session:
            user = User(
                id="runtime-adapter-user", email="runtime-adapter@local", hashed_password="x",
                full_name="Runtime Adapter", is_active=True, role="admin",
            )
            proj = Project(id=project_id, name=project_id, description="", owner_id=user.id)
            session.add_all([user, proj])
            session.commit()
            inv = Investigation(
                id=investigation_id, project_id=proj.id, user_id=user.id,
                question=question, status=_ExecInvestigationState.PLANNED,
            )
            session.add(inv)
            session.commit()

        ds_provider = InMemoryDatasetProvider(dict(datasets))
        controller = self._controller_hint or InvestigationController(
            session_factory=SessionFactory,
            dataset_provider=ds_provider,
            ai_provider=self.provider,
        )
        try:
            controller.execute_investigation(investigation_id=investigation_id, worker_id="runtime-adapter")
        except Exception as e:
            with SessionFactory() as session:
                inv = session.query(Investigation).filter(Investigation.id == investigation_id).first()
                if inv is not None:
                    inv.status = "FAILED"
                    inv.verdict_type = "INCONCLUSIVE"
                    inv.direct_answer = f"Investigation failed: {e}"
                    session.commit()

        execution_time = time.time() - start_time

        with SessionFactory() as session:
            inv = session.query(Investigation).filter(Investigation.id == investigation_id).first()
            verdict_row = (
                session.query(InvestigationVerdict)
                .filter(InvestigationVerdict.investigation_id == investigation_id)
                .first()
            )
            hyp_rows = session.query(Hypothesis).filter(Hypothesis.investigation_id == investigation_id).all()
            evidence_rows = session.query(Evidence).filter(Evidence.investigation_id == investigation_id).all()
            experiment_ids = [e.experiment_id for e in evidence_rows if e.experiment_id]
            obs_by_experiment_id = {
                o.experiment_id: o
                for o in session.query(Observation).filter(Observation.experiment_id.in_(experiment_ids)).all()
            } if experiment_ids else {}
            event_rows = (
                session.query(InvestigationEvent)
                .filter(InvestigationEvent.investigation_id == investigation_id)
                .order_by(InvestigationEvent.sequence)
                .all()
            )
            decision_recs = (
                session.query(DecisionRecommendationRecord)
                .filter(DecisionRecommendationRecord.investigation_id == investigation_id)
                .all()
            )

            verdict_type = (inv.verdict_type if inv else None) or "INCONCLUSIVE"
            if verdict_type not in [v.value for v in AnalyticalVerdict]:
                verdict_type = "INCONCLUSIVE"
            confidence_score = verdict_row.confidence_score if verdict_row else None
            status_str = (
                "COMPLETED" if inv and inv.status == "COMPLETED"
                else "FAILED" if inv and inv.status == "FAILED"
                else "INCONCLUSIVE"
            )

            hypotheses = [
                HypothesisSchema(
                    id=h.id, statement=h.statement, rationale=h.rationale or "",
                    priority=float(h.posterior_probability or 0.5), status=h.status or "proposed",
                    reason_for_rejection=h.reason_for_rejection,
                    prior_probability=h.prior_probability, posterior_probability=h.posterior_probability,
                    target_metric=h.target_metric, target_dimension=h.target_dimension,
                    target_value=h.target_value, mechanism=h.mechanism_detail,
                    generated_reason=h.generated_reason,
                )
                for h in hyp_rows
            ]
            evidence = [
                EvidenceSchema(
                    id=e.id, statement=e.statement, calculation_summary=e.calculation_summary or "",
                    dataset_version="1",
                    row_count_analyzed=(obs_by_experiment_id.get(e.experiment_id).row_count_analyzed if e.experiment_id in obs_by_experiment_id else 0),
                    sql_executed=(obs_by_experiment_id.get(e.experiment_id).sql_executed if e.experiment_id in obs_by_experiment_id else None),
                    raw_metrics=((obs_by_experiment_id.get(e.experiment_id).result_json or {}).get("structured_result", {}) if e.experiment_id in obs_by_experiment_id else {}),
                    p_value=e.p_value, effect_size=e.effect_size,
                    # P0 epistemic-integrity fix: this was the third
                    # independent instance of the same case-sensitive
                    # comparison bug fixed in AnalysisService and
                    # state_reconstruction.py (comparing lowercase
                    # "verified"/"failed" literals against the uppercase
                    # VerificationStatus values the controller actually
                    # persists), so this path -- used by AutonomousOrchestrator
                    # / InvestigationRuntime -- silently reported every
                    # evidence row as SKIPPED too. Routed through the single
                    # authoritative mapping.
                    validation_status=to_public_validation_status(e.validation_status),
                    calculation_trace=(
                        obs_by_experiment_id.get(e.experiment_id).result_json.get("calculation_trace")
                        if e.experiment_id in obs_by_experiment_id and (obs_by_experiment_id.get(e.experiment_id).result_json or {}).get("calculation_trace")
                        else None
                    ),
                )
                for e in evidence_rows
            ]
            steps = [
                AnalysisStepSchema(
                    step_number=idx + 1,
                    title=ev.event_type,
                    action_type=ev.event_type,
                    description=str(ev.event_payload_json or {}),
                    status="completed",
                )
                for idx, ev in enumerate(event_rows)
            ] or [
                AnalysisStepSchema(
                    step_number=1, title="investigation.executed", action_type="execute",
                    description="Controller executed the investigation.", status="completed",
                )
            ]
            findings = [
                FindingSchema(
                    id=h.id, title=h.statement[:120] if h.statement else h.hypothesis_code,
                    summary=h.rationale or h.statement or "",
                    importance_score=max(0.0, min(1.0, float(h.posterior_probability or 0.0))),
                    confidence=_confidence_level(verdict_type, h.posterior_probability),
                    verdict=AnalyticalVerdict(verdict_type),
                    evidence_items=evidence,
                )
                for h in hyp_rows
            ]
            decision_recommendations = [
                DecisionRecommendation(
                    recommendation_id=r.recommendation_id, action_title=r.action_title,
                    action_description=r.action_description, grounded_hypothesis_id=r.grounded_hypothesis_id,
                    target_metric=r.target_metric,
                    expected_utility=ExpectedUtilityCalculation(
                        action_name=r.action_title,
                        expected_gain_metric=r.expected_gain_metric,
                        downside_risk_metric=r.downside_risk_metric,
                        probability_of_success=r.probability_of_success,
                        net_expected_utility=r.net_expected_utility,
                        utility_function_description=r.utility_function_description or "",
                    ),
                    policy_compliance_passed=r.policy_compliance_passed,
                    required_preconditions=r.required_preconditions_json or [],
                )
                for r in decision_recs
            ]

            dataset_versions_used = [
                {"name": name, "fingerprint": fp}
                for name, fp in ds_provider.acquire_context(project_id).dataset_fingerprints.items()
            ]
            manifest = AnalysisManifestSchema(
                analysis_id=investigation_id,
                user_question=question,
                project_id=project_id,
                dataset_versions_used=dataset_versions_used,
                ai_provider=type(self.provider).__name__ if self.provider else "none",
                ai_model=getattr(self.provider, "model", "none") if self.provider else "none",
                ai_mode="DETERMINISTIC",
                ai_fallback_triggered=False,
                ai_status_message=(
                    "This runtime adapter routes exclusively through the deterministic "
                    "InvestigationController; no AI provider is invoked by the scientific loop."
                ),
                prompt_versions={},
                created_at=inv.created_at if inv and inv.created_at else datetime.now(timezone.utc),
                execution_time_seconds=execution_time,
                total_steps=len(steps),
                replan_count=0,
                validation_summary={
                    "verdict_type": verdict_type,
                    "hypotheses_count": len(hyp_rows),
                    "evidence_count": len(evidence_rows),
                },
                reproducible_hash=(inv.reproducible_manifest_hash if inv and inv.reproducible_manifest_hash else "n/a"),
            )

            discovered_questions = [q.__dict__ for q in DatasetQuestionDiscovery.discover(datasets)]

            return AnalysisResponse(
                id=investigation_id,
                project_id=project_id,
                question=question,
                status=AnalysisStatus(status_str) if status_str in [s.value for s in AnalysisStatus] else AnalysisStatus.INCONCLUSIVE,
                verdict=AnalyticalVerdict(verdict_type),
                direct_answer=inv.direct_answer if inv else None,
                main_finding=verdict_row.justification if verdict_row else (inv.direct_answer if inv else None),
                confidence=_confidence_level(verdict_type, confidence_score),
                hypotheses=hypotheses,
                steps=steps,
                findings=findings,
                evidence=evidence,
                suggested_questions=discovered_questions,
                decision_recommendations=decision_recommendations,
                ai_mode="DETERMINISTIC",
                ai_fallback_triggered=False,
                manifest=manifest,
                created_at=inv.created_at if inv and inv.created_at else datetime.now(timezone.utc),
                updated_at=inv.updated_at if inv and inv.updated_at else datetime.now(timezone.utc),
            )

    def _build_empty_response(self, analysis_id: str, project_id: str, question: str) -> AnalysisResponse:
        """Return clean zero-state response when no data is uploaded."""
        return AnalysisResponse(
            id=analysis_id,
            project_id=project_id,
            question=question,
            status=AnalysisStatus.INSUFFICIENT_DATA,
            verdict=AnalyticalVerdict.INSUFFICIENT_DATA,
            direct_answer="No active datasets found in workspace to analyze.",
            main_finding="Please upload a CSV, Excel, Parquet file or load demo benchmark data to begin.",
            confidence=ConfidenceLevel.INSUFFICIENT_EVIDENCE,
            hypotheses=[],
            steps=[],
            findings=[],
            evidence=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
