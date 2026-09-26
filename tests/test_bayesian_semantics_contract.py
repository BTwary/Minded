"""P0 contract tests: Bayes factors must never be stored as probabilities."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.src.models.entities import Base, BeliefUpdate, AnalysisRun, gen_uuid


def test_belief_update_exposes_nullable_bayes_factor_and_legacy_probability():
    assert BeliefUpdate.bayes_factor.property.columns[0].nullable is True
    assert BeliefUpdate.likelihood_p.property.columns[0].nullable is True


def test_new_belief_update_can_store_bf_without_probability():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        # Foreign-key checks are not needed for this schema contract test.
        row = BeliefUpdate(
            id=gen_uuid(), investigation_id="inv", hypothesis_id="hyp",
            prior_probability=0.5, likelihood_p=None, bayes_factor=12.0,
            posterior_probability=12.0 / 13.0,
        )
        db.add(row)
        db.commit()
        saved = db.query(BeliefUpdate).one()
        assert saved.bayes_factor == 12.0
        assert saved.likelihood_p is None
    finally:
        db.close()
        engine.dispose()


def test_analysis_run_has_no_optimistic_confidence_default():
    column = AnalysisRun.confidence.property.columns[0]
    assert column.default is None
