from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from apps.api.src.models.entities import Base, Investigation, InvestigationEvent
from packages.analytics_core.src.runtime.scientific_state_snapshot import build_scientific_state_snapshot
from packages.analytics_core.src.runtime.state_reconstruction import reconstruct_investigation_state
from packages.analytics_core.src.runtime.state import InvestigationStateManager
from packages.shared.src.enums import AnalyticalVerdict


def test_canonical_runtime_snapshot_round_trips_full_state():
    engine = create_engine('sqlite:///:memory:', future=True)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, expire_on_commit=False)()
    db.add(Investigation(id='I-SNAP', project_id='P1', question='q', status='COMPLETED'))
    db.commit()
    mgr = InvestigationStateManager('I-SNAP', 'q')
    mgr.state.known_facts = ['fact-a']
    mgr.state.unknowns = ['unknown-b']
    mgr.state.analytical_assumptions = ['assumption-c']
    mgr.set_stopping_state({'reason': 'ANSWER_ESTABLISHED', 'round': 3})
    mgr.finalize_verdict(AnalyticalVerdict.CONFIRMED, 'm'*64)
    snap = build_scientific_state_snapshot(db, 'I-SNAP', runtime_state=mgr.get_state())
    db.add(InvestigationEvent(sequence=1, investigation_id='I-SNAP', event_type='investigation.scientific_state_snapshot', event_payload_json={'snapshot': snap}))
    db.commit()
    restored = reconstruct_investigation_state('I-SNAP', db, question='q')
    assert restored.state.known_facts == ['fact-a']
    assert restored.state.unknowns == ['unknown-b']
    assert restored.state.analytical_assumptions == ['assumption-c']
    assert restored.state.stopping_state['reason'] == 'ANSWER_ESTABLISHED'
    assert restored.state.final_verdict == AnalyticalVerdict.CONFIRMED
    assert restored.state.provenance_hash == 'm'*64
