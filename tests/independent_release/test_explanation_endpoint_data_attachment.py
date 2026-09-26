"""Regression test: GET /investigations/{id}/explanation must attach the
predictions, belief updates, evidence verifications, and event-derived
payloads it fetches, instead of silently dropping them.

This mirrors an earlier bug fixed in the sibling GET /investigations/{id}
endpoint (see BUGFIXES_2026-09-09_v4.md); the /explanation endpoint had the
same fetch-but-never-attach pattern, undetected because the local variables
it built were never wired into the returned dict. Verified directly against
the real endpoint function with an in-memory SQLite-backed session and a
minimal seeded investigation -- no mocking of the function under test.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.src.models.entities import (
    Base,
    User,
    Project,
    Investigation,
    Hypothesis,
    Prediction,
    Evidence,
    EvidenceVerification,
    BeliefUpdate,
    gen_uuid,
)
from apps.api.src.api.v1.investigations import get_investigation_explanation


def _seeded_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    user = User(id=gen_uuid(), email="t@example.com", hashed_password="x", full_name="Tester", role="analyst")
    db.add(user)
    project = Project(id=gen_uuid(), owner_id=user.id, name="Test Project")
    db.add(project)
    db.flush()

    inv = Investigation(
        id="INV-TEST-EXPLANATION-001",
        project_id=project.id,
        user_id=user.id,
        question="Did churn increase?",
        status="COMPLETED",
        verdict_type="STATISTICALLY_SIGNIFICANT",
        direct_answer="Yes, churn increased.",
        main_finding="Churn rose 4pp month over month.",
    )
    db.add(inv)
    db.flush()

    hyp = Hypothesis(
        id=gen_uuid(),
        investigation_id=inv.id,
        canonical_identity="hyp-1",
        statement="Pricing change drove churn.",
        posterior_probability=0.82,
    )
    db.add(hyp)
    db.flush()

    pred = Prediction(
        id=gen_uuid(),
        investigation_id=inv.id,
        hypothesis_id=hyp.id,
        statement="Churn rate rises after the pricing change.",
        status="SUPPORTED",
    )
    db.add(pred)

    evid = Evidence(
        id=gen_uuid(),
        investigation_id=inv.id,
        hypothesis_id=hyp.id,
        statement="Observed churn delta of 4pp.",
        confidence_score=0.9,
        validation_status="verified",
    )
    db.add(evid)
    db.flush()

    verification = EvidenceVerification(
        id=gen_uuid(),
        evidence_id=evid.id,
        primary_tool="duckdb",
        secondary_tool="polars",
        status="PASSED",
    )
    db.add(verification)

    belief = BeliefUpdate(
        id=gen_uuid(),
        investigation_id=inv.id,
        hypothesis_id=hyp.id,
        evidence_id=evid.id,
        prior_probability=0.5,
        likelihood_p=0.9,
        posterior_probability=0.82,
    )
    db.add(belief)
    db.commit()

    return db, user, inv


def test_explanation_endpoint_attaches_fetched_data():
    db, user, inv = _seeded_session()
    try:
        result = get_investigation_explanation(
            investigation_id=inv.id,
            mode="DETERMINISTIC",
            current_user=user,
            db=db,
        )
    finally:
        db.close()

    # These keys were being computed (queried from the DB) but never
    # written into `result` -- this is the actual bug being tested.
    for key in (
        "predictions",
        "belief_updates",
        "evidence_verifications",
        "semantic_world_model",
        "adversarial_findings",
        "multiverse",
        "missingness",
        "causal_status",
        "epistemic_assessment",
        "provenance_manifest_event",
        "analysis_plan",
    ):
        assert key in result, f"explanation response is missing '{key}'"

    assert len(result["predictions"]) == 1
    assert result["predictions"][0]["statement"] == "Churn rate rises after the pricing change."
    assert len(result["belief_updates"]) == 1
    assert result["belief_updates"][0]["posterior_probability"] == 0.82
    assert len(result["evidence_verifications"]) == 1
    assert result["evidence_verifications"][0]["status"] == "PASSED"


if __name__ == "__main__":
    test_explanation_endpoint_attaches_fetched_data()
    print("PASS: explanation endpoint attaches fetched predictions/belief_updates/evidence_verifications.")
