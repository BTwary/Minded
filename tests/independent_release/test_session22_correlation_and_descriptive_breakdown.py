"""Session 22 regression tests (DEFECT-039, DEFECT-040), driven through the REAL
InvestigationController path against the session-21 stress-harness datasets.

DEFECT-039a: symmetric bivariate role swap between compiler plan and final contract
             must not fail-close a clean correlation question.
DEFECT-039b: correlation Bayes factor must use the experiment's own pairwise-complete
             result frame, not a wider primary_df (nulls made evidence silently neutral).
DEFECT-040a: DESCRIPTIVE small-n singleton group must not block a descriptive breakdown.
DEFECT-040b: a pure "<rate> by <group>" churn question is answered descriptively.
"""
import os, sys, tempfile, unittest, warnings

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
warnings.filterwarnings("ignore")

from apps.api.src.models.entities import Base, User, Project, Investigation
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider
from packages.analytics_core.src.engines.analyst_answer import is_pure_descriptive_breakdown
from packages.analytics_core.src.intelligence.experiment_contract_validation import (
    detect_plan_final_contract_conflicts,
)
from packages.analytics_core.src.engines.belief import BeliefEngine


def _run(name, df, question):
    d = tempfile.mkdtemp()
    eng = create_engine(f"sqlite:///{d}/t.db", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=eng)
    SF = sessionmaker(bind=eng)
    db = SF()
    db.add_all([User(id="u1", email="a@b.c", hashed_password="x", full_name="U", is_active=True),
                Project(id="p1", name="P", owner_id="u1")])
    db.commit()
    db.add(Investigation(id="INV-1", project_id="p1", user_id="u1", question=question, status="QUEUED"))
    db.commit()
    InvestigationController(session_factory=SF, dataset_provider=InMemoryDatasetProvider({name: df})).execute_investigation(
        investigation_id="INV-1", worker_id="w")
    db.expire_all()
    return db.query(Investigation).filter(Investigation.id == "INV-1").first()


def _saas(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.exponential(12, n).clip(0, 60)
    tickets = rng.poisson(1.5, n).astype(float)
    plan = rng.choice(["Free", "Starter", "Pro", "Enterprise"], n, p=[.4, .3, .2, .1])
    p = 1 / (1 + np.exp(-(0.3 * tickets - 0.05 * tenure - 1.0)))
    df = pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(n)], "tenure_months": tenure.round(1),
                       "support_tickets": tickets, "plan_type": plan, "churned": (rng.random(n) < p).astype(int)})
    df.loc[df.sample(30, random_state=1).index, "support_tickets"] = np.nan
    return df


class TestCorrelationDefect039(unittest.TestCase):
    def test_clean_correlation_is_not_refused_and_finds_the_planted_signal(self):
        r = _run("saas", _saas(), "Is churn correlated with support tickets?")
        self.assertNotIn("authority conflict", (r.direct_answer or "").lower())
        self.assertEqual(r.verdict_type, "STATISTICALLY_SIGNIFICANT")

    def test_reversed_phrasing_gives_same_verdict(self):
        r = _run("saas", _saas(), "Is there a relationship between support tickets and churn?")
        self.assertEqual(r.verdict_type, "STATISTICALLY_SIGNIFICANT")

    def test_bayes_factor_uses_pairwise_complete_result_frame_despite_nulls(self):
        df = _saas()
        pair = df[["churned", "support_tickets"]].dropna()
        hyps = [type("H", (), {"hypothesis_code": "HYP-01", "is_counter_hypothesis": False, "target_dimension": None})(),
                type("H", (), {"hypothesis_code": "HYP-02", "is_counter_hypothesis": True, "target_dimension": None})()]
        f, diag = BeliefEngine.compute_model_based_bayes_factors(
            hypotheses=hyps, primary_df=df, result_df=pair, target_metric_col="churned",
            group_dimension_col="plan_type", aggregation_type="CORRELATION",
            tested_hypothesis_codes=["HYP-01", "HYP-02"])
        self.assertGreater(f[0], 100.0)
        self.assertEqual(diag[0]["method"], "BIC_GAUSSIAN_LINEAR_ASSOCIATION")

    def _final(self, **kw):
        base = dict(canonical_task="ASSOCIATION", selected_method_code="association_numeric",
                    target_column="churned", predictor_columns=("support_tickets",),
                    time_column=None, comparison_dimension=None)
        base.update(kw)
        return type("F", (), base)()

    def _plan(self, target, preds):
        sem = type("S", (), dict(target_column=target, explanatory_columns=preds, time_column=None, grouping_columns=[]))()
        return type("P", (), dict(task="ASSOCIATION", semantics=sem))()

    def test_pure_role_swap_is_overridden_not_blocking(self):
        out = detect_plan_final_contract_conflicts(self._plan("support_tickets", ["churned"]), self._final())
        self.assertTrue(out)
        self.assertFalse(any(c["severity"] == "BLOCKING" for c in out))

    def test_different_columns_still_block(self):
        out = detect_plan_final_contract_conflicts(self._plan("revenue", ["churned"]), self._final())
        self.assertTrue(any(c["severity"] == "BLOCKING" for c in out))

    def test_directional_method_role_swap_still_blocks(self):
        out = detect_plan_final_contract_conflicts(
            self._plan("support_tickets", ["churned"]), self._final(selected_method_code="association_categorical_binary"))
        self.assertTrue(any(c["severity"] == "BLOCKING" for c in out))


class TestDescriptiveBreakdownDefect040(unittest.TestCase):
    def test_singleton_group_average_is_reported_not_refused(self):
        df = pd.DataFrame({"department": ["Eng", "Eng", "Sales", "Sales", "Sales", "HR"],
                           "attrition": [1, 0, 1, 1, 0, 0], "tenure_years": [2.1, 5.3, .8, 1.2, 3.0, 4.4]})
        r = _run("hr", df, "What is the average tenure by department?")
        self.assertNotIn("admissibility gate blocked", (r.direct_answer or "").lower())
        for token in ("HR", "Eng", "Sales", "4.40", "3.70", "1.67"):
            self.assertIn(token, r.direct_answer)
        self.assertIn("not reliable", r.direct_answer)

    def test_churn_rate_by_group_reports_rates(self):
        r = _run("saas", _saas(), "What is the churn rate by plan type?")
        self.assertEqual(r.verdict_type, "OBSERVED")
        for plan in ("Free", "Starter", "Pro", "Enterprise"):
            self.assertIn(plan, r.direct_answer)

    def test_helper_is_conservative(self):
        yes = ["What is the churn rate by plan type?", "What is the average tenure by department?",
               "What is the total revenue by region?"]
        no = ["Which plan has the highest churn rate?", "Why is churn higher for Free plan?",
              "Is churn correlated with plan type?", "How does churn rate differ by plan?",
              "What is churn rate by plan since January?", ""]
        for q in yes: self.assertTrue(is_pure_descriptive_breakdown(q), q)
        for q in no: self.assertFalse(is_pure_descriptive_breakdown(q), q)


if __name__ == "__main__":
    unittest.main()
