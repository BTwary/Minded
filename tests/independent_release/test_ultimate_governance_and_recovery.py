from __future__ import annotations
import pandas as pd
from packages.analytics_core.src.governance.claim_gate import admit_positive_claim
from packages.analytics_core.src.runtime.scientific_state_snapshot import build_scientific_state_snapshot
from packages.analytics_core.src.statistics.universal_preflight import inspect_design


def test_positive_claim_gate_blocks_unverified_result():
    result = admit_positive_claim(verdict_type="DIAGNOSED", directly_tested=True, verified_evidence=False)
    assert not result.allowed
    assert "verified_evidence_missing" in result.reasons


def test_universal_preflight_blocks_degenerate_design():
    df = pd.DataFrame({"y": [1, 1, 1], "g": ["A", "A", "A"]})
    result = inspect_design(df, target="y", grouping="g")
    assert not result.allowed
    assert "target_constant_or_near_constant" in result.reasons
    assert "grouping_has_fewer_than_two_levels" in result.reasons


def test_scientific_snapshot_is_deterministic():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from apps.api.src.models.entities import Base, Investigation
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Investigation(id="INV-SNAP", project_id="P", question="q", status="RUNNING"))
    session.commit()
    a = build_scientific_state_snapshot(session, "INV-SNAP")
    b = build_scientific_state_snapshot(session, "INV-SNAP")
    assert a["snapshot_hash"] == b["snapshot_hash"]
    assert a["hypotheses"] == []
