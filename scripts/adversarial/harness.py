"""
Phase 10 shared test harness.

Every scenario in scripts/test_phase10_adversarial_real_world.py goes through
this single entry point, `run_investigation(...)`, which:

  1. spins up a fresh in-memory SQLite database,
  2. registers the given dataset(s) via InMemoryDatasetProvider,
  3. creates a real Investigation row,
  4. calls the REAL, unmodified InvestigationController.execute_investigation(...),
  5. reads back everything persisted (hypotheses, experiments, evidence,
     verifications, belief updates, step trace, verdict, provenance hash),
  6. returns it as a plain ResultBundle for the test to grade against
     independently-computed ground truth.

No scenario test is allowed to call IntentEngine / HypothesisSynthesizer /
EIGOptimizer / VerdictEngine / VerificationEngine directly to manufacture the
investigation path. Only this harness's call into the controller does that,
and the controller decides everything itself.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from apps.api.src.models.entities import (
    Base,
    BeliefUpdate,
    Evidence,
    EvidenceVerification,
    Experiment,
    Hypothesis,
    Investigation,
    InvestigationEvent,
    InvestigationStepExecution,
    InvestigationVerdict,
    Observation,
    Project,
    User,
    gen_uuid,
)
from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider
from packages.analytics_core.src.execution.state_machine import InvestigationState
from packages.analytics_core.src.runtime.controller import InvestigationController


@dataclass
class ResultBundle:
    ok: bool
    investigation: Optional[Investigation]
    verdict: Optional[InvestigationVerdict]
    hypotheses: List[Hypothesis] = field(default_factory=list)
    experiments: List[Experiment] = field(default_factory=list)
    evidences: List[Evidence] = field(default_factory=list)
    verifications: List[EvidenceVerification] = field(default_factory=list)
    belief_updates: List[BeliefUpdate] = field(default_factory=list)
    steps: List[InvestigationStepExecution] = field(default_factory=list)
    error_message: Optional[str] = None
    failure_taxonomy: Optional[str] = None
    # Phase 12 addition (additive/backward-compatible): raw provenance
    # events, so tests can inspect structured, reproducible records (e.g.
    # "investigation.missingness_sensitivity") without touching engine
    # internals directly.
    events: List[InvestigationEvent] = field(default_factory=list)

    def event_payloads(self, event_type: str) -> List[Dict[str, Any]]:
        return [e.event_payload_json for e in self.events if e.event_type == event_type]

    @property
    def status(self) -> Optional[str]:
        return self.investigation.status if self.investigation else None

    @property
    def verdict_type(self) -> Optional[str]:
        return self.investigation.verdict_type if self.investigation else None

    @property
    def direct_answer(self) -> str:
        return (self.investigation.direct_answer or "") if self.investigation else ""

    @property
    def confidence(self) -> Optional[float]:
        return self.verdict.confidence_score if self.verdict else None

    @property
    def provenance_hash(self) -> Optional[str]:
        return self.investigation.reproducible_manifest_hash if self.investigation else None

    @property
    def step_types(self) -> List[str]:
        return [s.step_type for s in self.steps]

    def print_trace(self, scenario_name: str) -> None:
        print("=" * 80)
        print(f"SCENARIO: {scenario_name}")
        print("=" * 80)
        print(f" -> ok={self.ok}  status={self.status}  verdict_type={self.verdict_type}")
        if not self.ok:
            print(f" -> failure_taxonomy={self.failure_taxonomy}")
            print(f" -> error_message={self.error_message}")
        print(f" -> hypotheses={len(self.hypotheses)} "
              f"posteriors={[round(h.posterior_probability, 4) for h in self.hypotheses]}")
        print(f" -> experiments={[e.test_code for e in self.experiments]}")
        print(f" -> verifications={[v.status for v in self.verifications]}")
        print(f" -> step_trace={self.step_types}")
        print(f" -> confidence={self.confidence}")
        print(f" -> direct_answer={self.direct_answer!r}")
        if self.verdict:
            print(f" -> justification={self.verdict.justification!r}")
        print(f" -> provenance_hash={self.provenance_hash}")
        print("=" * 80)


def run_investigation(
    datasets_map: Dict[str, pd.DataFrame],
    question: str,
    scenario_tag: str = "scn",
) -> ResultBundle:
    """Runs the real, unmodified InvestigationController against the given
    in-memory dataset(s) and question. This is the ONLY function in the
    Phase 10 suite that touches InvestigationController; every scenario test
    calls this and then only inspects the persisted result."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        user = User(id=f"usr-{scenario_tag}", email=f"{scenario_tag}@aaos.ai",
                    hashed_password="pw", full_name="Phase10 Tester", is_active=True)
        proj = Project(id=f"prj-{scenario_tag}", name=f"Phase10 {scenario_tag}", description="", owner_id=user.id)
        session.add_all([user, proj])
        session.commit()

        inv_id = f"INV-{scenario_tag.upper()}-{gen_uuid()[:8]}"
        inv = Investigation(
            id=inv_id,
            project_id=proj.id,
            user_id=user.id,
            question=question,
            status=InvestigationState.PLANNED,
        )
        session.add(inv)
        session.commit()

    ds_provider = InMemoryDatasetProvider(dict(datasets_map))
    controller = InvestigationController(session_factory=session_factory, dataset_provider=ds_provider)

    ok = controller.execute_investigation(investigation_id=inv_id, worker_id=f"w-{scenario_tag}")

    with session_factory() as session:
        inv = session.query(Investigation).filter(Investigation.id == inv_id).first()
        verdict = session.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == inv_id).first()
        hyps = session.query(Hypothesis).filter(Hypothesis.investigation_id == inv_id).all()
        exps = session.query(Experiment).filter(Experiment.investigation_id == inv_id).all()
        evs = session.query(Evidence).filter(Evidence.investigation_id == inv_id).all()
        verifs = (
            session.query(EvidenceVerification)
            .join(Evidence, EvidenceVerification.evidence_id == Evidence.id)
            .filter(Evidence.investigation_id == inv_id)
            .all()
        )
        beliefs = session.query(BeliefUpdate).filter(BeliefUpdate.investigation_id == inv_id).all()
        steps = (
            session.query(InvestigationStepExecution)
            .filter(InvestigationStepExecution.investigation_id == inv_id)
            .order_by(InvestigationStepExecution.step_index)
            .all()
        )
        events = (
            session.query(InvestigationEvent)
            .filter(InvestigationEvent.investigation_id == inv_id)
            .order_by(InvestigationEvent.sequence)
            .all()
        )

        error_message = None
        failure_taxonomy = None
        if not ok:
            from apps.api.src.models.entities import InvestigationExecution
            exec_rec = (
                session.query(InvestigationExecution)
                .filter(InvestigationExecution.investigation_id == inv_id)
                .order_by(InvestigationExecution.id.desc())
                .first()
            )
            if exec_rec is not None:
                error_message = exec_rec.error_message
                failure_taxonomy = exec_rec.error_code

        # Detach objects for use outside the session by expunging.
        for obj in [inv, verdict, *hyps, *exps, *evs, *verifs, *beliefs, *steps, *events]:
            if obj is not None:
                session.expunge(obj)

        return ResultBundle(
            ok=ok,
            investigation=inv,
            verdict=verdict,
            hypotheses=hyps,
            experiments=exps,
            evidences=evs,
            verifications=verifs,
            belief_updates=beliefs,
            steps=steps,
            error_message=error_message,
            failure_taxonomy=failure_taxonomy,
            events=events,
        )
