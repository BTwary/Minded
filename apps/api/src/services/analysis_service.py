"""Analysis Orchestration and Persistence Service."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy.orm import Session

from apps.api.src.models.entities import (
    AnalysisRun,
    Dataset,
    Evidence,
    Experiment,
    Hypothesis,
    Investigation,
    InvestigationContract,
    InvestigationVerdict,
    InvestigationEvent,
    ClaimGateDecision,
    Observation,
    gen_uuid,
)
from apps.api.src.services.dataset_service import DatasetService
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.engines.local_explanation import LocalExplanationEngine
from packages.analytics_core.src.intelligence.dataset_question_discovery import DatasetQuestionDiscovery
from packages.analytics_core.src.engines.exploratory_analysis import ExploratoryAnalysisEngine
from packages.analytics_core.src.governance.claim_gate import (
    ClaimType as GateClaimType,
    DesignStatus as GateDesignStatus,
    EvidenceLevel as GateEvidenceLevel,
    evaluate_gate,
)
from packages.analytics_core.src.execution.state_machine import to_public_validation_status
from packages.schemas.src.analysis import (
    AnalysisCreate,
    AnalysisResponse,
    AnalysisStatus,
    AnalyticalVerdict,
    ConfidenceLevel,
    AnalysisStepSchema,
    EvidenceSchema,
    HypothesisSchema,
    ClaimGateResultSchema,
)

# Confidence score -> ConfidenceLevel bucketing. The controller persists a
# numeric confidence_score on InvestigationVerdict; the API schema exposes a
# coarse ConfidenceLevel. An INSUFFICIENT_DATA/INCONCLUSIVE verdict is always
# reported as insufficient evidence regardless of whatever numeric score the
# controller happened to compute, since a low-information verdict should
# never read as "Low confidence" (which implies a real, if weak, finding).
_INSUFFICIENT_VERDICTS = {"INSUFFICIENT_DATA", "INCONCLUSIVE", "CONFLICTING_EVIDENCE"}


def _confidence_level(verdict_type: str, score: Optional[float]) -> ConfidenceLevel:
    if verdict_type in _INSUFFICIENT_VERDICTS or score is None:
        return ConfidenceLevel.INSUFFICIENT_EVIDENCE
    if score >= 0.75:
        return ConfidenceLevel.HIGH
    if score >= 0.5:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


class AnalysisService:
    """Service coordinating end-to-end autonomous analysis execution and persistence.

    This drives the real InvestigationController -- the canonical AA-OS
    autonomous investigation loop (hypothesis generation, experiment
    selection, dual-engine verification, adversarial challenge, evidence-
    driven stopping) -- synchronously, in-process, for the request/response
    HTTP endpoint. It replaces a prior keyword-bucket "legacy executive
    runtime adapter" that never actually invoked the controller it was
    constructed with, and had no path to report insufficient evidence.
    """

    def __init__(self, db: Session, dataset_service: Optional[DatasetService] = None):
        self.db = db
        self.dataset_service = dataset_service or DatasetService(db)

    def execute_analysis(self, payload: AnalysisCreate, user_id: Optional[str] = None) -> AnalysisResponse:
        """Run the real autonomous InvestigationController and record all audit artifacts."""
        investigation_id = f"INV-{gen_uuid()[:12]}"
        inv = Investigation(
            id=investigation_id,
            project_id=payload.project_id,
            user_id=user_id,
            question=payload.question,
            status="PLANNED",
            # DEFECT-003 fix: this field existed on the request schema but
            # was never read here, so a request-scoped dataset selection was
            # silently dropped and the controller always acquired every
            # dataset in the project regardless of what was selected.
            requested_dataset_ids_json=list(payload.dataset_ids) if payload.dataset_ids else None,
        )
        self.db.add(inv)
        self.db.commit()

        # Explicit dependency injection: pass an AI provider to the controller
        # only when AI is intentionally enabled for the user/environment.
        # Otherwise, the controller defaults to purely deterministic analysis.
        ai_provider = None
        if user_id:
            try:
                from apps.api.src.ai.providers.factory import is_ai_enabled, get_ai_provider
                if is_ai_enabled(user_id=user_id):
                    ai_provider = get_ai_provider(user_id=user_id)
            except Exception:
                ai_provider = None
        else:
            try:
                from packages.analytics_core.src.platform_local_first import resolve_ai_enabled
                if resolve_ai_enabled():
                    from apps.api.src.ai.providers.factory import get_ai_provider
                    ai_provider = get_ai_provider()
            except Exception:
                ai_provider = None

        controller = InvestigationController(
            session_factory=lambda: _NoCloseSessionWrapper(self.db),
            ai_provider=ai_provider,
        )

        try:
            controller.execute_investigation(investigation_id=investigation_id, worker_id="sync-http-api")
        except Exception as e:
            self.db.rollback()
            inv = self.db.query(Investigation).filter(Investigation.id == investigation_id).first()
            if inv is None:
                raise
            inv.status = "FAILED"
            inv.verdict_type = "INCONCLUSIVE"
            inv.direct_answer = f"Investigation failed: {e}"
            self.db.commit()

        self.db.expire_all()
        inv = self.db.query(Investigation).filter(Investigation.id == investigation_id).first()
        verdict_row = (
            self.db.query(InvestigationVerdict)
            .filter(InvestigationVerdict.investigation_id == investigation_id)
            .first()
        )
        hyp_rows = self.db.query(Hypothesis).filter(Hypothesis.investigation_id == investigation_id).all()
        evidence_rows = self.db.query(Evidence).filter(Evidence.investigation_id == investigation_id).all()
        # P0 projection-completeness fix: the controller genuinely executes
        # experiments (persisted as Experiment/Observation rows), but this
        # method never queried them, so AnalysisResponse.steps was always
        # `[]` and any consumer computing num_steps = len(res.steps) always
        # got 0 regardless of how much work the controller actually did.
        # AnalysisStepSchema is an existing, already-declared field on
        # AnalysisResponse -- this populates it from real persisted data,
        # it does not invent step counts.
        exp_rows = (
            self.db.query(Experiment)
            .filter(Experiment.investigation_id == investigation_id)
            .order_by(Experiment.created_at.asc())
            .all()
        )
        obs_by_experiment_id = {
            o.experiment_id: o
            for o in self.db.query(Observation).filter(Observation.experiment_id.in_([e.id for e in exp_rows])).all()
        } if exp_rows else {}

        plan_event = (
            self.db.query(InvestigationEvent)
            .filter(InvestigationEvent.investigation_id == investigation_id, InvestigationEvent.event_type == "investigation.universal_analysis_plan")
            .order_by(InvestigationEvent.sequence.desc())
            .first()
        )
        analysis_plan = (plan_event.event_payload_json if plan_event is not None else None)

        verdict_type = inv.verdict_type or "INCONCLUSIVE"
        if verdict_type not in AnalyticalVerdict.__members__.values() and verdict_type not in [v.value for v in AnalyticalVerdict]:
            verdict_type = "INCONCLUSIVE"
        confidence_score = verdict_row.confidence_score if verdict_row else None
        status_str = inv.status if inv.status in [s.value for s in AnalysisStatus] else (
            "COMPLETED" if inv.status == "COMPLETED" else "FAILED" if inv.status == "FAILED" else "INCONCLUSIVE"
        )

        hypotheses = [
            HypothesisSchema(
                id=h.id,
                statement=h.statement,
                rationale=h.rationale or "",
                priority=float(h.posterior_probability or 0.5),
                status=h.status or "proposed",
                reason_for_rejection=h.reason_for_rejection,
                prior_probability=h.prior_probability,
                posterior_probability=h.posterior_probability,
                target_metric=h.target_metric,
                target_dimension=h.target_dimension,
                target_value=h.target_value,
                mechanism=h.mechanism_detail,
                generated_reason=h.generated_reason,
            )
            for h in hyp_rows
        ]
        evidence = [
            EvidenceSchema(
                id=e.id,
                statement=e.statement,
                calculation_summary=e.calculation_summary or "",
                dataset_version="1",
                row_count_analyzed=(obs_by_experiment_id.get(e.experiment_id).row_count_analyzed if e.experiment_id in obs_by_experiment_id else 0),
                sql_executed=(obs_by_experiment_id.get(e.experiment_id).sql_executed if e.experiment_id in obs_by_experiment_id else None),
                raw_metrics=((obs_by_experiment_id.get(e.experiment_id).result_json or {}).get("structured_result", {}) if e.experiment_id in obs_by_experiment_id else {}),
                p_value=e.p_value,
                effect_size=e.effect_size,
                # P0 epistemic-integrity fix: was a case-sensitive inline
                # comparison against lowercase strings that never matched
                # the uppercase VerificationStatus values the controller
                # actually persists, so every evidence row silently became
                # SKIPPED regardless of its real verification outcome. Now
                # routed through the single authoritative mapping so the
                # public API faithfully reflects internal state.
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
                title=f"{exp.tool_name} ({exp.test_code})",
                action_type=exp.tool_name,
                description=exp.rationale or exp.selection_rationale or "",
                tool_name=exp.tool_name,
                tool_input=exp.arguments_json or {},
                duration_ms=int((obs_by_experiment_id.get(exp.id).execution_time_ms or 0)
                                if obs_by_experiment_id.get(exp.id) else 0),
                status="completed" if exp.status == "EXECUTED" else (exp.status or "planned").lower(),
            )
            for idx, exp in enumerate(exp_rows)
        ]

        # Deterministic exploratory reconnaissance runs locally from the selected data.
        # It is a screening layer for structure discovery, not an inferential/causal verdict.
        exploratory_analysis: Optional[Dict[str, Any]] = None
        try:
            _explore_tables = {}
            selected_ids = list(payload.dataset_ids) if payload.dataset_ids else None
            selected_datasets = self.db.query(Dataset).filter(Dataset.project_id == payload.project_id).all()
            for ds in selected_datasets:
                if selected_ids and ds.id not in selected_ids:
                    continue
                try:
                    _explore_tables[ds.name] = self.dataset_service.get_dataset_dataframe(ds.id)
                except Exception:
                    continue
            if _explore_tables:
                # Preserve deterministic behavior for multi-table projects by analyzing the
                # primary selected table; relational semantics remain handled by the world model.
                primary_name = next(iter(_explore_tables))
                exploratory_analysis = ExploratoryAnalysisEngine.analyze(_explore_tables[primary_name])
                exploratory_analysis["dataset"] = primary_name
        except Exception as _explore_exc:
            exploratory_analysis = {
                "engine_version": ExploratoryAnalysisEngine.VERSION,
                "status": "UNAVAILABLE",
                "reason": type(_explore_exc).__name__,
            }

        # Release Claim Gate: deterministic three-state admissibility decision.
        # It consumes existing contract/evidence/causal artifacts and never calls an LLM.
        _claim_raw = str((analysis_plan or {}).get("claim_type", "OBSERVATION")).upper()
        _claim_map = {
            "OBSERVATION": GateClaimType.OBSERVATION,
            "ASSOCIATION": GateClaimType.ASSOCIATION,
            "PREDICTION": GateClaimType.PREDICTION,
            "CAUSAL": GateClaimType.CAUSAL,
            "CAUSAL_INFERENCE": GateClaimType.CAUSAL,
            "RECOMMENDATION": GateClaimType.RECOMMENDATION,
            "SIMULATION": GateClaimType.RECOMMENDATION,
        }
        requested_claim = _claim_map.get(_claim_raw, GateClaimType.OBSERVATION)
        verified_statuses = {"PASSED", "VERIFIED", "PARTIALLY_VERIFIED"}
        has_verified = any(str(getattr(e, "validation_status", "")).upper() in verified_statuses for e in evidence)
        raw_verified = [getattr(obs_by_experiment_id.get(e.experiment_id), "result_json", {}) or {} for e in evidence_rows]
        text_blob = " ".join(
            str(x)
            for x in [
                *(exp.tool_name for exp in exp_rows),
                *(exp.test_code for exp in exp_rows),
                *(exp.rationale for exp in exp_rows),
                *(exp.selection_rationale for exp in exp_rows),
                *(str(r) for r in raw_verified),
            ]
            if x
        ).lower()
        prediction_oos = requested_claim != GateClaimType.PREDICTION or any(
            token in text_blob for token in ("out-of-sample", "out_of_sample", "holdout", "rolling-origin", "backtest")
        )
        if not evidence:
            achieved_level = GateEvidenceLevel.RAW_OBSERVATION
        elif requested_claim == GateClaimType.OBSERVATION:
            achieved_level = GateEvidenceLevel.VALIDATED_COMPUTATION if has_verified else GateEvidenceLevel.RAW_OBSERVATION
        elif requested_claim == GateClaimType.ASSOCIATION:
            achieved_level = GateEvidenceLevel.ASSOCIATION if has_verified else GateEvidenceLevel.VALIDATED_COMPUTATION
        elif requested_claim == GateClaimType.PREDICTION:
            achieved_level = GateEvidenceLevel.PREDICTION if (has_verified and prediction_oos) else GateEvidenceLevel.VALIDATED_COMPUTATION
        elif requested_claim == GateClaimType.CAUSAL:
            causal_gate_event = (
                self.db.query(InvestigationEvent)
                .filter(InvestigationEvent.investigation_id == investigation_id, InvestigationEvent.event_type == "investigation.causal_gate.evaluated")
                .order_by(InvestigationEvent.sequence.desc())
                .first()
            )
            causal_payload = causal_gate_event.event_payload_json if causal_gate_event else {}
            causal_effect_event = (
                self.db.query(InvestigationEvent)
                .filter(InvestigationEvent.investigation_id == investigation_id, InvestigationEvent.event_type == "investigation.causal_effect.estimated")
                .order_by(InvestigationEvent.sequence.desc())
                .first()
            )
            if causal_payload.get("is_identifiable") and causal_effect_event and has_verified:
                achieved_level = GateEvidenceLevel.CAUSAL_ESTIMATION
            elif causal_payload.get("is_identifiable"):
                achieved_level = GateEvidenceLevel.CAUSAL_IDENTIFICATION
            else:
                achieved_level = GateEvidenceLevel.ASSOCIATION if has_verified else GateEvidenceLevel.VALIDATED_COMPUTATION
        else:
            achieved_level = GateEvidenceLevel.ASSOCIATION if has_verified else GateEvidenceLevel.VALIDATED_COMPUTATION

        # `inv.active_contract_id` is the authoritative pointer to the current
        # contract revision (same pattern used in controller.py); fall back to
        # the highest-version contract row for this investigation if the
        # pointer isn't set. Both `contract_assumptions` below and the
        # selection-bias check further down depend on this being the real
        # contract row, not an undefined name.
        latest_contract = None
        if getattr(inv, "active_contract_id", None):
            latest_contract = self.db.query(InvestigationContract).filter(
                InvestigationContract.id == inv.active_contract_id
            ).first()
        if latest_contract is None:
            latest_contract = (
                self.db.query(InvestigationContract)
                .filter(InvestigationContract.investigation_id == investigation_id)
                .order_by(InvestigationContract.version.desc())
                .first()
            )
        contract_assumptions = list(getattr(latest_contract, "assumptions_json", None) or [])
        unresolved = list((analysis_plan or {}).get("unresolved_questions", []) or [])
        high_risk = bool(unresolved)
        design_status = (
            GateDesignStatus.KNOWN
            if (analysis_plan and (analysis_plan.get("decision_status") == "RESOLVED") and not unresolved)
            else GateDesignStatus.ASSUMED
            if analysis_plan
            else GateDesignStatus.UNKNOWN
        )
        causal_event = (
            self.db.query(InvestigationEvent)
            .filter(InvestigationEvent.investigation_id == investigation_id, InvestigationEvent.event_type == "investigation.causal_gate.evaluated")
            .order_by(InvestigationEvent.sequence.desc())
            .first()
        )
        causal_payload = causal_event.event_payload_json if causal_event else {}
        identification_strategy = None
        if causal_payload.get("is_identifiable"):
            status_name = str(causal_payload.get("status") or "IDENTIFIED").upper()
            identification_strategy = status_name
        selection_bias = bool(getattr(inv, "selection_bias_indicators", None))
        if latest_contract is not None:
            scope = latest_contract.scope_json or {}
            selection_bias = selection_bias or bool(scope.get("epistemic_manifest", {}).get("selection_bias_indicators"))
        gate = evaluate_gate(
            claim_type=requested_claim,
            evidence_level=achieved_level,
            design_status=design_status,
            assumptions=contract_assumptions,
            identification_strategy=identification_strategy,
            computed_evidence={
                "verified_evidence_count": sum(1 for e in evidence if str(getattr(e, "validation_status", "")).upper() in verified_statuses),
                "estimate": next((e.effect_size for e in evidence if e.effect_size is not None), None),
                "evidence_ids": [e.id for e in evidence],
            },
            assumption_risk=high_risk,
            prediction_out_of_sample=prediction_oos,
            selection_bias=selection_bias,
            verification_passed=(has_verified or not evidence),
        )
        existing_gate = (
            self.db.query(ClaimGateDecision)
            .filter(ClaimGateDecision.investigation_id == investigation_id)
            .order_by(ClaimGateDecision.created_at.desc())
            .first()
        )
        if existing_gate is not None:
            gate_id = existing_gate.id
            gate_schema = ClaimGateResultSchema.model_validate({
                "outcome": existing_gate.outcome,
                "requested_claim": existing_gate.requested_claim,
                "evidence_level": existing_gate.evidence_level,
                "design_status": existing_gate.design_status,
                "assumptions": existing_gate.assumptions_json or [],
                "allowed_claim": existing_gate.allowed_claim,
                "blocked_claim": existing_gate.blocked_claim,
                "reason": existing_gate.reason,
                "recovery_actions": existing_gate.recovery_actions_json or [],
                "computed_evidence": existing_gate.computed_evidence_json or {},
                "refusal_id": existing_gate.id if existing_gate.outcome == "REFUSE" else None,
                "requested_level": existing_gate.requested_level,
                "max_supported_level": existing_gate.max_supported_level,
                "identification_strategy": existing_gate.identification_strategy,
                "blocking_conditions": existing_gate.blocking_conditions_json or [],
                "missing_evidence": existing_gate.missing_evidence_json or [],
            })
        else:
            gate_id = f"RG-{gen_uuid()[:12]}"
            gate_payload = gate.to_dict()
            # gate.to_dict() renders evidence_level as its enum *name* (e.g. "RAW_OBSERVATION")
            # for human-readable audit logs. ClaimGateResultSchema.evidence_level is a
            # Pydantic IntEnum, which only accepts the numeric value, not the name string.
            # Passing the name straight through raised a validation error (HTTP 500) on
            # every path that reaches this branch, e.g. forecast questions.
            gate_payload["evidence_level"] = gate.evidence_level.value
            gate_payload["refusal_id"] = gate_id if gate.outcome == "REFUSE" else None
            gate_row = ClaimGateDecision(
                id=gate_id,
                investigation_id=investigation_id,
                question_id=investigation_id,
                requested_claim=gate.requested_claim.name,
                requested_level=gate.requested_level if gate.requested_level is not None else gate.requested_claim.value,
                evidence_level=gate.evidence_level.value,
                max_supported_level=gate.max_supported_level if gate.max_supported_level is not None else gate.evidence_level.value,
                outcome=gate.outcome,
                design_status=gate.design_status.value,
                identification_strategy=gate.identification_strategy,
                assumptions_json=gate.assumptions,
                blocking_conditions_json=gate.blocking_conditions,
                missing_evidence_json=gate.missing_evidence,
                recovery_actions_json=gate.recovery_actions,
                allowed_claim=gate.allowed_claim,
                blocked_claim=gate.blocked_claim,
                reason=gate.reason,
                computed_evidence_json=gate.computed_evidence,
            )
            self.db.add(gate_row)
            self.db.flush()
            gate_schema = ClaimGateResultSchema.model_validate({**gate_payload, "refusal_id": gate_id if gate.outcome == "REFUSE" else None})

        # Local deterministic explanation is always generated from canonical persisted
        # investigation facts. It does not call an LLM and remains available when the
        # machine has no network connection or no AI provider configured.
        response_question = payload.question
        local_explanation = LocalExplanationEngine.build(
            question=response_question,
            verdict_type=verdict_type,
            direct_answer=inv.direct_answer,
            justification=verdict_row.justification if verdict_row else inv.direct_answer,
            hypotheses=hyp_rows,
            experiments=exp_rows,
            observations=list(obs_by_experiment_id.values()),
            evidence_rows=evidence,
        )
        # Dataset-driven question discovery: independently inspect the actual
        # selected dataset(s) and expose questions MindEd can answer without
        # requiring an LLM or paid service. Cross-table suggestions are based
        # only on discovered relationships and remain subject to join-safety
        # preflight when executed.
        try:
            _question_datasets = {}
            selected_ids = list(payload.dataset_ids) if payload.dataset_ids else None
            dataset_rows = self.db.query(Dataset).filter(Dataset.project_id == payload.project_id).all()
            for ds in dataset_rows:
                if selected_ids and ds.id not in selected_ids:
                    continue
                try:
                    _question_datasets[ds.name] = self.dataset_service.get_dataset_dataframe(ds.id)
                except Exception:
                    continue
            discovered_questions = [q.__dict__ for q in DatasetQuestionDiscovery.discover(_question_datasets)]
        except Exception:
            # Question discovery is an enhancement to the response surface;
            # failure here must never alter the completed scientific result.
            discovered_questions = []


        response = AnalysisResponse(
            id=investigation_id,
            project_id=payload.project_id,
            question=response_question,
            status=AnalysisStatus(status_str) if status_str in [s.value for s in AnalysisStatus] else AnalysisStatus.INCONCLUSIVE,
            verdict=AnalyticalVerdict(verdict_type),
            direct_answer=inv.direct_answer,
            main_finding=verdict_row.justification if verdict_row else inv.direct_answer,
            confidence=_confidence_level(verdict_type, confidence_score),
            hypotheses=hypotheses,
            steps=steps,
            evidence=evidence,
            suggested_questions=discovered_questions,
            exploratory_analysis=exploratory_analysis,
            explanation=local_explanation.to_dict(),
            analysis_plan=analysis_plan,
            claim_gate=gate_schema,
            ai_mode="DETERMINISTIC",
            ai_fallback_triggered=False,
            ai_status_message="Local deterministic analysis and explanation are active; no AI provider required.",
            created_at=inv.created_at or datetime.now(timezone.utc),
            updated_at=inv.updated_at or datetime.now(timezone.utc),
        )

        # Optional AI augmentation happens only AFTER the deterministic analysis is
        # complete. The AI receives a compact verified-facts packet and can only add
        # presentation text; it cannot alter the canonical verdict/evidence. Any AI
        # failure falls back to the local explanation without failing the analysis.
        if payload.analysis_mode.upper() == "AI_AUGMENTED":
            # Use the per-user provider already resolved above — never re-discover
            # via the global InfrastructureManager cache, which would break per-user
            # isolation (User A's cached provider could serve User B's augmentation).
            if ai_provider is not None:
                provider_ok, provider_message = ai_provider.test_connection()
            else:
                provider_ok, provider_message = False, "AI is disabled for this user/environment."
            if provider_ok and ai_provider.provider_type != "mock":
                fact_packet = {
                    "question": payload.question,
                    "verdict": verdict_type,
                    "direct_answer": inv.direct_answer,
                    "main_finding": verdict_row.justification if verdict_row else inv.direct_answer,
                    "local_explanation": local_explanation.to_dict(),
                }
                ai_prompt = (
                    "Rewrite the following verified analytical explanation for a business user. "
                    "Do not introduce new facts, numbers, causal claims, or recommendations. "
                    "Preserve every limitation and uncertainty. Return plain text only.\n\n"
                    + str(fact_packet)
                )
                ai_text = ai_provider.generate_response(
                    ai_prompt,
                    system_prompt=(
                        "You are an explanation layer only. The supplied verdict and evidence are authoritative. "
                        "You may improve clarity but may not change analytical facts or epistemic strength."
                    ),
                )
                if ai_text and not ai_text.lower().startswith(("error:", "ollama execution error", "openai api error", "gemini api error")):
                    explanation_payload = local_explanation.to_dict()
                    explanation_payload["ai_rewrite"] = ai_text.strip()
                    explanation_payload["mode"] = "AI_AUGMENTED"
                    response.explanation = explanation_payload
                    response.ai_mode = "AI_AUGMENTED"
                    response.ai_status_message = f"AI explanation augmentation active via {ai_provider.provider_type}/{ai_provider.model_name}. Canonical analysis remains deterministic."
                else:
                    response.ai_fallback_triggered = True
                    response.ai_status_message = f"AI augmentation unavailable; local explanation retained. {provider_message}"
            else:
                response.ai_fallback_triggered = True
                response.ai_status_message = f"AI augmentation unavailable; local explanation retained. {provider_message}"

        # Persist AnalysisRun for the list/detail-by-id endpoints.
        run_record = AnalysisRun(
            id=response.id,
            project_id=payload.project_id,
            question=response.question,
            status=response.status.value,
            direct_answer=response.direct_answer,
            main_finding=response.main_finding,
            confidence=response.confidence.value,
            hypotheses_json=[h.model_dump(mode="json") for h in response.hypotheses],
            evidence_json=[e.model_dump(mode="json") for e in response.evidence],
            manifest_json={
                "analysis_mode": response.ai_mode,
                "ai_fallback_triggered": response.ai_fallback_triggered,
                "ai_status_message": response.ai_status_message,
                "explanation": response.explanation,
                "analysis_plan": response.analysis_plan,
                "claim_gate": response.claim_gate.model_dump(mode="json") if response.claim_gate else None,
                "evidence_integrity": {
                    "evidence_count": len(response.evidence),
                    "verified_evidence_count": sum(1 for e in response.evidence if e.validation_status == "PASSED"),
                    "has_evidence": bool(response.evidence),
                },
                "exploratory_analysis": response.exploratory_analysis,
                "source": "canonical_investigation_state",
            },
        )
        self.db.add(run_record)
        self.db.commit()

        return response


class _NoCloseSessionWrapper:
    """Delegates every attribute to a shared SQLAlchemy Session but no-ops
    .close(), so the controller (which manages sessions with `with
    self.session_factory() as session:` and expects each acquisition to be
    independently closeable) can safely share this request's single
    connection/session instead of opening a second one mid-request."""

    def __init__(self, session: Session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def close(self):
        # No-op: several call sites (e.g. DatabaseDatasetProvider) call
        # `session_factory().close()` directly instead of via `with`, and
        # closing the shared request-scoped session mid-request would break
        # every subsequent query in this HTTP request.
        pass

    def __getattr__(self, name):
        return getattr(self._session, name)

