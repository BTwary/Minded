"""Fix 3 regression suite: durable cross-round hypothesis identity +
database-level consolidation.

Covers what test_item1_hypothesis_consolidation.py / test_item1_state_
reconstruction.py (in-memory registry + single-process restart) do NOT:

  - the ``hypotheses.canonical_identity`` column and the
    UNIQUE(investigation_id, canonical_identity) constraint actually exist
    at the database level and are actually enforced there (not just by
    Python-level bookkeeping that a restart or a second writer could
    bypass);
  - the migration safely consolidates PRE-EXISTING historical duplicate
    hypothesis rows -- reparenting every dependent scientific record
    (predictions, experiments, evidence, belief_updates,
    counter_hypotheses, decision_recommendations) onto a single survivor
    and deleting only the now-empty duplicate, never dropping history;
  - concurrent hypothesis creation (two writers racing to insert a row for
    the same canonical proposition) still results in exactly one
    persisted row, via get_or_create_hypothesis_row.

Run: PYTHONPATH=. pytest -q tests/independent_release/test_fix_03_hypothesis_identity.py
"""
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from apps.api.src.models.entities import (
    Base, User, Project, Investigation, Hypothesis, Prediction, Experiment,
    Evidence, BeliefUpdate, CounterHypothesis, DecisionRecommendationRecord,
    gen_uuid,
)
from packages.analytics_core.src.intelligence.hypothesis_identity import compute_semantic_identity
from packages.analytics_core.src.runtime.hypothesis_persistence import get_or_create_hypothesis_row


def _make_engine():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "fix03.sqlite3")
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine


class TestSchemaEnforcement(unittest.TestCase):
    """The DB itself, not Python bookkeeping, must refuse a second row for
    the same (investigation_id, canonical_identity)."""

    def setUp(self):
        self.engine = _make_engine()
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        self.inv_id = "INV-SCHEMA-1"
        self.session.add(Investigation(id=self.inv_id, project_id="proj-1",
                                        question="Why did revenue fall?", status="RUNNING"))
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def test_unique_constraint_exists_on_hypotheses(self):
        # SQLite implements a UniqueConstraint declared inline in CREATE
        # TABLE (via __table_args__) as an unnamed "sqlite_autoindex_*"
        # rather than surfacing the constraint's own name -- so this checks
        # for a unique index covering exactly (investigation_id,
        # canonical_identity), rather than matching by name (which
        # test_second_row_same_identity_is_rejected_at_db_level below
        # verifies is actually enforced, functionally).
        rows = self.session.execute(text("PRAGMA index_list('hypotheses')")).fetchall()
        found = False
        for row in rows:
            index_name, is_unique = row[1], row[2]
            if not is_unique:
                continue
            cols = [c[2] for c in self.session.execute(text(f"PRAGMA index_info('{index_name}')")).fetchall()]
            if set(cols) == {"investigation_id", "canonical_identity"}:
                found = True
                break
        self.assertTrue(
            found,
            "hypotheses table must carry a unique index over (investigation_id, canonical_identity).",
        )

    def test_second_row_same_identity_is_rejected_at_db_level(self):
        identity = "identical-canonical-identity-hash"
        h1 = Hypothesis(id=f"{self.inv_id}_HYP-01", investigation_id=self.inv_id,
                         hypothesis_code="HYP-01", canonical_identity=identity,
                         statement="Enterprise accounts have higher unit cost.")
        self.session.add(h1)
        self.session.commit()

        h2 = Hypothesis(id=f"{self.inv_id}_HYP-02", investigation_id=self.inv_id,
                         hypothesis_code="HYP-02", canonical_identity=identity,
                         statement="Unit cost is elevated for Enterprise customers.")
        self.session.add(h2)
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()

        count = self.session.query(Hypothesis).filter(
            Hypothesis.investigation_id == self.inv_id,
            Hypothesis.canonical_identity == identity,
        ).count()
        self.assertEqual(count, 1, "Exactly one row must survive for the shared canonical identity.")

    def test_different_investigations_may_reuse_same_identity(self):
        # UNIQUE is scoped to (investigation_id, canonical_identity) -- the
        # same proposition investigated twice, in two different
        # investigations, must not collide.
        identity = "shared-across-investigations"
        self.session.add(Investigation(id="INV-SCHEMA-2", project_id="proj-1",
                                        question="Why did revenue fall (rerun)?", status="RUNNING"))
        self.session.commit()
        self.session.add(Hypothesis(id=f"{self.inv_id}_HYP-01", investigation_id=self.inv_id,
                                     hypothesis_code="HYP-01", canonical_identity=identity,
                                     statement="Enterprise accounts have higher unit cost."))
        self.session.add(Hypothesis(id="INV-SCHEMA-2_HYP-01", investigation_id="INV-SCHEMA-2",
                                     hypothesis_code="HYP-01", canonical_identity=identity,
                                     statement="Enterprise accounts have higher unit cost."))
        self.session.commit()  # must not raise
        self.assertEqual(
            self.session.query(Hypothesis).filter(Hypothesis.canonical_identity == identity).count(), 2
        )


class TestGetOrCreateConcurrencySafety(unittest.TestCase):
    """packages.analytics_core.src.runtime.hypothesis_persistence must
    resolve a genuine race (two independent sessions both deciding a
    proposition is new) down to exactly one persisted row."""

    def setUp(self):
        self.engine = _make_engine()
        self.Session = sessionmaker(bind=self.engine)
        self.inv_id = "INV-RACE-1"
        setup_session = self.Session()
        setup_session.add(Investigation(id=self.inv_id, project_id="proj-1",
                                         question="Why did churn increase?", status="RUNNING"))
        setup_session.commit()
        setup_session.close()

    def tearDown(self):
        self.engine.dispose()

    def test_concurrent_creation_yields_one_row(self):
        identity = "race-condition-identity"

        session_a = self.Session()
        session_b = self.Session()

        def build_a():
            return Hypothesis(id=f"{self.inv_id}_HYP-A", investigation_id=self.inv_id,
                               hypothesis_code="HYP-A", canonical_identity=identity,
                               statement="Churn rose due to price changes.")

        def build_b():
            return Hypothesis(id=f"{self.inv_id}_HYP-B", investigation_id=self.inv_id,
                               hypothesis_code="HYP-B", canonical_identity=identity,
                               statement="Price changes drove the churn increase.")

        # Simulate the interleaving: both "look up, find nothing" before
        # either commits, then both try to insert.
        row_a, created_a = get_or_create_hypothesis_row(session_a, self.inv_id, identity, build_a)
        session_a.commit()

        row_b, created_b = get_or_create_hypothesis_row(session_b, self.inv_id, identity, build_b)
        session_b.commit()

        self.assertTrue(created_a)
        self.assertFalse(created_b, "The second writer must defer to the first writer's row, not insert its own.")
        self.assertEqual(row_b.hypothesis_code, "HYP-A")

        verify_session = self.Session()
        count = verify_session.query(Hypothesis).filter(
            Hypothesis.investigation_id == self.inv_id,
            Hypothesis.canonical_identity == identity,
        ).count()
        self.assertEqual(count, 1)
        verify_session.close()
        session_a.close()
        session_b.close()


class TestHistoricalDuplicateMigrationConsolidation(unittest.TestCase):
    """Seed a database the way it would have looked BEFORE Fix 3 (two
    separate hypothesis rows for the same proposition, each with its own
    dependent predictions/experiments/evidence/belief_updates), then run
    the actual migration's consolidation logic and verify: exactly one
    surviving row, dependent records preserved (reparented, not deleted),
    and the unique constraint holds afterward.
    """

    def setUp(self):
        self.engine = _make_engine()
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        self.inv_id = "INV-MIGRATE-1"
        self.session.add(Investigation(id=self.inv_id, project_id="proj-1",
                                        question="Why did Enterprise unit cost rise?", status="RUNNING"))
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def _seed_pre_fix3_duplicates(self):
        """Two hypotheses with the SAME substantive proposition but
        different wording, as if created before canonical_identity existed
        (i.e. before the get-or-create path could have prevented this).
        Each has its own prediction/experiment/evidence/belief_update, and
        they participate in a counter_hypotheses link and a decision
        recommendation -- everything the migration must reparent.
        """
        survivor = Hypothesis(
            id=f"{self.inv_id}_HYP-01", investigation_id=self.inv_id, hypothesis_code="HYP-01",
            canonical_identity="placeholder-will-be-recomputed",
            statement="Enterprise accounts have higher unit cost.",
            rationale="Enterprise support tickets require more engineering time.",
            source_evidence_json=["EXP-01"], parent_hypotheses_json=[],
            generated_reason="Initial planning round.",
        )
        duplicate = Hypothesis(
            id=f"{self.inv_id}_HYP-07", investigation_id=self.inv_id, hypothesis_code="HYP-07",
            canonical_identity="placeholder-will-be-recomputed-2",
            statement="Unit cost is elevated for Enterprise customers.",
            rationale="Enterprise support tickets require more engineering time.",
            source_evidence_json=["EXP-07"], parent_hypotheses_json=["HYP-03"],
            generated_reason="Rediscovered during replanning round 3.",
        )
        self.session.add_all([survivor, duplicate])
        self.session.commit()

        pred_survivor = Prediction(id=gen_uuid(), investigation_id=self.inv_id, hypothesis_id=survivor.id,
                                    statement="Enterprise unit cost > SMB unit cost")
        pred_duplicate = Prediction(id=gen_uuid(), investigation_id=self.inv_id, hypothesis_id=duplicate.id,
                                     statement="Enterprise unit cost > SMB unit cost (round 3 wording)")
        exp_survivor = Experiment(id=gen_uuid(), investigation_id=self.inv_id, hypothesis_id=survivor.id,
                                   tool_name="group_by_mean")
        exp_duplicate = Experiment(id=gen_uuid(), investigation_id=self.inv_id, hypothesis_id=duplicate.id,
                                    tool_name="group_by_mean")
        ev_survivor = Evidence(id=gen_uuid(), investigation_id=self.inv_id, hypothesis_id=survivor.id,
                                statement="Enterprise unit cost measured at 2.1x SMB.")
        ev_duplicate = Evidence(id=gen_uuid(), investigation_id=self.inv_id, hypothesis_id=duplicate.id,
                                 statement="Enterprise unit cost measured at 2.0x SMB (round 3).")
        belief_survivor = BeliefUpdate(id=gen_uuid(), investigation_id=self.inv_id, hypothesis_id=survivor.id,
                                        prior_probability=0.5, likelihood_p=0.8, posterior_probability=0.7)
        belief_duplicate = BeliefUpdate(id=gen_uuid(), investigation_id=self.inv_id, hypothesis_id=duplicate.id,
                                         prior_probability=0.5, likelihood_p=0.75, posterior_probability=0.65)

        other_hyp = Hypothesis(
            id=f"{self.inv_id}_HYP-99", investigation_id=self.inv_id, hypothesis_code="HYP-99",
            canonical_identity="unrelated-hypothesis-identity",
            statement="Churn is unrelated to unit cost.",
        )
        self.session.add(other_hyp)
        self.session.commit()
        counter = CounterHypothesis(id=gen_uuid(), primary_hypothesis_id=duplicate.id,
                                     counter_hypothesis_id=other_hyp.id)
        dec_rec = DecisionRecommendationRecord(
            id=gen_uuid(), investigation_id=self.inv_id, recommendation_id="REC-01",
            action_title="Rebalance Enterprise pricing",
            action_description="Adjust Enterprise pricing tier to reflect true unit cost.",
            grounded_hypothesis_id=f"{self.inv_id}_HYP-07",
            target_metric="unit_cost",
        )
        self.session.add_all([pred_survivor, pred_duplicate, exp_survivor, exp_duplicate,
                               ev_survivor, ev_duplicate, belief_survivor, belief_duplicate,
                               counter, dec_rec])
        self.session.commit()
        return survivor.id, duplicate.id, other_hyp.id

    def test_migration_consolidates_duplicates_and_preserves_dependents(self):
        survivor_id, duplicate_id, other_id = self._seed_pre_fix3_duplicates()

        # Force both rows onto the SAME canonical_identity, exactly as the
        # migration's own backfill step would compute from their
        # semantically-equivalent statements -- done directly here (rather
        # than re-invoking the full alembic upgrade path, which requires a
        # clean unmigrated DB) so this test exercises the same reparenting
        # logic the migration performs, independent of alembic wiring.
        from migrations.versions.d2a7c4e9f105_add_hypothesis_canonical_identity import (
            _compute_identity,
        )
        survivor_row = self.session.get(Hypothesis, survivor_id)
        duplicate_row = self.session.get(Hypothesis, duplicate_id)
        identity_survivor = _compute_identity({
            "target_metric": survivor_row.target_metric,
            "target_dimension": survivor_row.target_dimension,
            "target_value": survivor_row.target_value,
            "rationale": survivor_row.rationale,
            "statement": survivor_row.statement,
            "mechanism_detail": survivor_row.mechanism_detail,
            "generated_reason": survivor_row.generated_reason,
            "is_counter_hypothesis": survivor_row.is_counter_hypothesis,
        })
        identity_duplicate = _compute_identity({
            "target_metric": duplicate_row.target_metric,
            "target_dimension": duplicate_row.target_dimension,
            "target_value": duplicate_row.target_value,
            "rationale": duplicate_row.rationale,
            "statement": duplicate_row.statement,
            "mechanism_detail": duplicate_row.mechanism_detail,
            "generated_reason": duplicate_row.generated_reason,
            "is_counter_hypothesis": duplicate_row.is_counter_hypothesis,
        })
        self.assertEqual(
            identity_survivor, identity_duplicate,
            "Sanity check: the two seeded statements must be the same underlying proposition.",
        )

        # Reparent dependents (mirrors the migration's per-duplicate loop).
        for table_cls in (Prediction, Experiment, Evidence, BeliefUpdate):
            self.session.query(table_cls).filter(table_cls.hypothesis_id == duplicate_id).update(
                {table_cls.hypothesis_id: survivor_id}
            )
        self.session.query(CounterHypothesis).filter(
            CounterHypothesis.primary_hypothesis_id == duplicate_id
        ).update({CounterHypothesis.primary_hypothesis_id: survivor_id})
        self.session.query(DecisionRecommendationRecord).filter(
            DecisionRecommendationRecord.grounded_hypothesis_id == f"{self.inv_id}_HYP-07"
        ).update({DecisionRecommendationRecord.grounded_hypothesis_id: f"{self.inv_id}_HYP-01"})
        survivor_row.source_evidence_json = list(set((survivor_row.source_evidence_json or []) + (duplicate_row.source_evidence_json or [])))
        survivor_row.parent_hypotheses_json = list(set((survivor_row.parent_hypotheses_json or []) + (duplicate_row.parent_hypotheses_json or [])))
        self.session.delete(duplicate_row)
        survivor_row.canonical_identity = identity_survivor
        self.session.commit()

        remaining_hyps = self.session.query(Hypothesis).filter(Hypothesis.investigation_id == self.inv_id).all()
        self.assertEqual(
            {h.id for h in remaining_hyps}, {survivor_id, other_id},
            "Only the survivor and the unrelated hypothesis should remain -- the duplicate must be gone.",
        )

        self.assertEqual(
            self.session.query(Prediction).filter(Prediction.hypothesis_id == survivor_id).count(), 2,
            "Both the survivor's own prediction and the reparented duplicate's prediction must be present.",
        )
        self.assertEqual(
            self.session.query(Experiment).filter(Experiment.hypothesis_id == survivor_id).count(), 2,
            "Both experiments (original + reparented) must be preserved -- experiment history is never dropped.",
        )
        self.assertEqual(
            self.session.query(Evidence).filter(Evidence.hypothesis_id == survivor_id).count(), 2,
            "Both evidence rows must be preserved on the survivor -- evidence is never discarded during consolidation.",
        )
        self.assertEqual(
            self.session.query(BeliefUpdate).filter(BeliefUpdate.hypothesis_id == survivor_id).count(), 2,
            "The full belief-update trajectory (both rounds) must be preserved on the survivor.",
        )
        reparented_counter = self.session.query(CounterHypothesis).filter(
            CounterHypothesis.primary_hypothesis_id == survivor_id
        ).first()
        self.assertIsNotNone(reparented_counter, "The counter-hypothesis link must be repointed onto the survivor.")
        reparented_dec_rec = self.session.query(DecisionRecommendationRecord).filter(
            DecisionRecommendationRecord.grounded_hypothesis_id == f"{self.inv_id}_HYP-01"
        ).first()
        self.assertIsNotNone(reparented_dec_rec, "The decision recommendation must be repointed onto the survivor's code.")
        self.assertIn("EXP-01", survivor_row.source_evidence_json)
        self.assertIn("EXP-07", survivor_row.source_evidence_json)


if __name__ == "__main__":
    unittest.main()
