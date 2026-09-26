"""Regression tests for the three root causes behind the DEFECT-007 controller failures.

1. Synthesized multi-row cross-tab queries (EXP-CONFOUND / EXP-COND-*) must be
   deterministically ordered, or the execution provider (correctly) rejects them.
2. An ambiguous two-dimension result must be recorded as AMBIGUOUS_RESULT_COLUMNS
   and must not crash with UnboundLocalError.
3. A later, distinct test carrying the same analytical identity as an existing
   PRIMARY must run as an audited SUPPORTING follow-up -- never a second PRIMARY
   (the DB unique index stays authoritative) and never silently dropped.
"""
import re

import pandas as pd

from apps.api.src.models.entities import Experiment, Investigation, InvestigationEvent, Observation, gen_uuid
from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider
from packages.analytics_core.src.runtime.controller import InvestigationController
from tests.independent_release.test_defect_007_controller_closure import BaseIsolatedControllerTest


def _two_dim_df(surge_mod=0):
    return pd.DataFrame({
        "datacenter_region": ["us-east", "us-west", "eu-central", "ap-south"] * 100,
        "tier": (["Enterprise", "SMB"] * 50) + (["SMB", "SMB"] * 150),
        "cost_metric": [500.0 if (i % 4 == surge_mod and i < 200) else 100.0 for i in range(400)],
    })


class TestSession5Regressions(BaseIsolatedControllerTest):
    def _run(self, df):
        inv = Investigation(
            id=f"INV-S5-{gen_uuid()[:8]}", project_id=self.proj.id, user_id=self.user.id,
            question="Why did cost_metric surge across datacenter_region?", status="QUEUED",
        )
        self.db.add(inv)
        self.db.commit()
        controller = InvestigationController(
            session_factory=self.SessionFactory,
            dataset_provider=InMemoryDatasetProvider({"cloud_costs": df}),
        )
        self.assertTrue(controller.execute_investigation(investigation_id=inv.id, worker_id="w-s5"))
        self.db.expire_all()
        return inv

    def test_confound_crosstab_executes_and_is_ordered(self):
        inv = self._run(_two_dim_df())
        exps = {e.test_code: e for e in self.db.query(Experiment).filter(Experiment.investigation_id == inv.id)}
        self.assertIn("EXP-CONFOUND", exps)
        conf = exps["EXP-CONFOUND"]
        self.assertEqual(conf.status, "EXECUTED")
        self.assertRegex(conf.arguments_json.get("sql", ""), re.compile(r"\bORDER\s+BY\b", re.I))
        obs = self.db.query(Observation).filter(Observation.experiment_id == conf.id).first()
        self.assertIsNotNone(obs)
        self.assertIn("AMBIGUOUS_RESULT_COLUMNS", str(obs.result_json),
                      "A two-dimension cross-tab must be flagged ambiguous, not reduced to an arbitrary pair.")

    def test_same_identity_followup_is_supporting_and_single_primary(self):
        inv = self._run(_two_dim_df())
        exps = self.db.query(Experiment).filter(Experiment.investigation_id == inv.id).all()
        isolate = [e for e in exps if e.test_code.startswith("EXP-ISOLATE")]
        self.assertTrue(isolate, "Adaptive isolation round must run, not be blocked as a duplicate PRIMARY.")
        self.assertEqual(isolate[0].status, "EXECUTED")
        self.assertEqual(isolate[0].experiment_role, "SUPPORTING")
        primaries = [e for e in exps if e.experiment_role == "PRIMARY" and e.analytical_identity]
        idents = [e.analytical_identity for e in primaries]
        self.assertEqual(len(idents), len(set(idents)), "At most one PRIMARY per analytical identity.")
        events = [
            ev.event_type for ev in
            self.db.query(InvestigationEvent).filter(InvestigationEvent.investigation_id == inv.id)
        ]
        self.assertIn("investigation.primary_identity.demoted_to_supporting", events)
        self.assertNotIn("investigation.primary_identity.blocked", events)
