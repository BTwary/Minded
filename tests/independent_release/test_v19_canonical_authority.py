"""v19 canonical analytical authority regression suite.

Covers AAOS_V19_CANONICAL_AUTHORITY.md section 13's authority-conflict
scenarios that are reachable without a seeded DB / full InvestigationController
run (A, B/C combined via the registry-status path, E, F, G). Scenarios that
require a live InvestigationController + persisted contract (D full semantic-
binding rejection, H replanning, I provenance-in-persisted-events) are NOT
covered here -- see AAOS_V19_RELEASE_STATUS.md for what remains.
"""
import hashlib
import os
import random
import sys
import unittest
import uuid

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Investigation, InvestigationContract
from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition
from packages.schemas.src.analysis import AggregationType
from packages.analytics_core.src.engines.method_selection import (
    MethodSelectionEngine, MethodFamily, ProblemClass,
    CANONICAL_TASK_AUTHORITY, TaskAuthorityStatus, VERDICT_TIER_STATISTICALLY_SIGNIFICANT,
)


def _semantic(metric="revenue", group="region", secondary=None, churn_event=None,
              table="orders", md=None):
    return SemanticResolution(
        primary_dataset_name=table, target_metric_col=metric, group_dimension_col=group,
        time_col=None, table_grain="record_level", available_numeric_cols=[metric],
        available_categorical_cols=[group] if group else [], world_model=None,
        metric_definition=md or MetricDefinition(
            name=metric, table_name=table, source_columns=[metric], semantic_type="sum_measure",
            aggregation_type=AggregationType.SUM, is_additive=True,
            valid_aggregations=[AggregationType.SUM], semantic_resolution_status="RESOLVED"),
        direction_hint="unspecified", secondary_metric_col=secondary,
        churn_event_col=churn_event, churn_exposure_col=None, churn_censored_col=None,
        churn_confounder_cols=[], churn_outcome_available=True,
    )


class InMemoryDatasetProvider(BaseDatasetProvider):
    def __init__(self, datasets_map):
        self.datasets_map = datasets_map

    def acquire_context(self, project_id, dataset_ids=None):
        fingerprints = {
            name: hashlib.sha256(df.to_json().encode()).hexdigest()
            for name, df in self.datasets_map.items()
        }
        return InvestigationDataContext(
            project_id=project_id, datasets_map=self.datasets_map,
            dataset_fingerprints=fingerprints, requested_dataset_ids=dataset_ids,
        )


def _run_scenario_investigation(df: pd.DataFrame, question: str, table_name: str):
    from packages.analytics_core.src.runtime.controller import InvestigationController

    db = SessionLocal()
    inv_id = f"INV-V19-{uuid.uuid4().hex[:10]}"
    proj_id = f"proj-v19-{uuid.uuid4().hex[:8]}"
    inv = Investigation(id=inv_id, project_id=proj_id, question=question, status="PLANNED")
    db.add(inv)
    db.commit()
    db.close()

    provider = InMemoryDatasetProvider({table_name: df})
    ctrl = InvestigationController(session_factory=SessionLocal, dataset_provider=provider)
    ctrl.execute_investigation(investigation_id=inv_id, worker_id="test-v19-canonical-authority")

    fresh_db = SessionLocal()
    inv_result = fresh_db.query(Investigation).filter(Investigation.id == inv_id).first()
    contract = None
    if inv_result is not None and inv_result.active_contract_id:
        contract = fresh_db.query(InvestigationContract).filter(
            InvestigationContract.id == inv_result.active_contract_id
        ).first()
    fresh_db.close()
    return inv_result, contract


class TestA_LegacyCanonicalConflict(unittest.TestCase):
    """A. Different IntentEngine and UniversalQuestionCompiler classifications
    -> the canonical (UniversalQuestionCompiler-derived) result wins."""

    def test_general_intent_vs_comparison_task(self):
        sem = _semantic(secondary="cost")
        df = pd.DataFrame({"revenue": [1, 2, 3, 4, 5] * 5, "cost": [2, 3, 4, 5, 6] * 5,
                            "region": (["a", "b"] * 12) + ["a"]})
        q = "What drives higher revenue between regions?"
        plan = UniversalQuestionCompiler.compile(q, semantic=sem, df=df)
        intent = IntentEngine.parse_intent(q)

        # Confirm the two classifiers actually disagree before the fix would
        # have mattered -- this is the precondition for the whole test.
        self.assertEqual(plan.task, "COMPARISON")
        self.assertEqual(intent.intent_type, "GENERAL")

        decision = MethodSelectionEngine.decide(intent, sem, q, primary_df=df)
        self.assertEqual(decision.problem_class, ProblemClass.DESCRIPTIVE,
                          "precondition: legacy intent-derived decision must be non-canonical")

        decision = MethodSelectionEngine.select_for_plan(decision, plan, sem, None, df)
        self.assertEqual(decision.problem_class, ProblemClass.COMPARATIVE)
        self.assertEqual(decision.canonical_task, "COMPARISON")
        self.assertEqual(decision.analytical_task_status, "IMPLEMENTED")


class TestB_ContractMethodDivergence(unittest.TestCase):
    """B (adapted): the *persisted* InvestigationContract's selected_method_json
    is populated at contract-creation time from the compiler's proposed
    experiment codes (a different vocabulary, e.g. "correlation_or_regression"),
    while the actual method MethodSelectionEngine selects and executes
    ("association_numeric") is computed afterward and was never reconciled
    back into the contract at all before this phase. Proves the divergence
    is real via a live InvestigationController run, and that the
    execution_state_json.reconciled_method_decision fix added this phase
    makes the reconciled decision recoverable from the persisted contract.
    """

    def test_persisted_contract_carries_reconciled_method_decision(self):
        rng = random.Random(11)
        n = 200
        revenue = [100.0 + i * 0.7 + (rng.random() - 0.5) * 5 for i in range(n)]
        cost = [50.0 + r * 0.4 + (rng.random() - 0.5) * 3 for r in revenue]
        df = pd.DataFrame({"revenue": revenue, "cost": cost})

        inv, contract = _run_scenario_investigation(
            df, "Is there a correlation between revenue and cost?", table_name="v19_assoc"
        )
        self.assertIsNotNone(contract, "contract must be persisted")

        # v20-C4.2.3 authority model: the compiler's experiment is a PROPOSAL
        # (proposed_method_json); the reconciled decision is the FINAL contract
        # (final_contract_json, projected into selected_method_json).  The
        # v19 execution_state_json["reconciled_method_decision"] scratch copy was a
        # competing second truth and no longer exists.
        #
        # Precondition (unchanged intent): the divergence is real -- the
        # compiler-proposed method label and the reconciled method are genuinely
        # different vocabularies.
        compiler_proposed = (contract.proposed_method_json or {}).get("method")
        self.assertIsNotNone(contract.final_contract_json,
                             "the reconciled decision must be persisted as the final contract")
        final = contract.final_contract_json
        self.assertIn(final["selected_method_code"],
                      {"association_numeric", "association_categorical_binary"})
        self.assertNotEqual(compiler_proposed, final["selected_method_code"],
                             "precondition: contract's compiler-proposed method label and the "
                             "reconciled runtime decision are genuinely different vocabularies")
        self.assertEqual(final["canonical_task"], contract.problem_class)
        # The selected method slot is a projection of the final contract, never the proposal.
        self.assertEqual(contract.selected_method_json["method_code"], final["selected_method_code"])
        self.assertEqual(contract.analytical_identity, final["analytical_identity"])
        # Stricter than v19: no competing copy of the decision may remain.
        self.assertNotIn("reconciled_method_decision", contract.execution_state_json or {})



    """C. Candidate says FORECAST, contract says ASSOCIATION -> candidate rejected."""

    def test_forecast_intent_but_association_task(self):
        sem = _semantic(secondary="cost")
        df = pd.DataFrame({"revenue": [1, 2, 3, 4, 5] * 5, "cost": [2, 3, 4, 5, 6] * 5})
        q = "Is there a correlation between revenue and cost?"
        plan = UniversalQuestionCompiler.compile(q, semantic=sem, df=df)
        self.assertEqual(plan.task, "ASSOCIATION")

        intent = IntentEngine.parse_intent(q)
        decision = MethodSelectionEngine.decide(intent, sem, q, primary_df=df)
        decision = MethodSelectionEngine.select_for_plan(decision, plan, sem, None, df)

        self.assertEqual(decision.method_family, MethodFamily.CORRELATION)
        self.assertNotEqual(decision.method_family, MethodFamily.FORECAST)
        self.assertIn(decision.selected_method_code,
                      {"association_numeric", "association_categorical_binary"})


class TestE_UnsupportedTask(unittest.TestCase):
    """E. Canonical task GOVERNANCE has no registered analytical method
    -> UNSUPPORTED_ANALYTICAL_TASK, not a legacy method fallback."""

    def test_governance_task_is_unsupported(self):
        sem = _semantic()
        df = pd.DataFrame({"revenue": [1, 2, 3], "region": ["a", "b", "c"]})
        q = "Does this dataset raise any GDPR privacy or compliance concerns?"
        plan = UniversalQuestionCompiler.compile(q, semantic=sem, df=df)
        self.assertEqual(plan.task, "GOVERNANCE")
        self.assertEqual(CANONICAL_TASK_AUTHORITY["GOVERNANCE"].status, TaskAuthorityStatus.UNSUPPORTED)

        intent = IntentEngine.parse_intent(q)
        decision = MethodSelectionEngine.decide(intent, sem, q, primary_df=df)
        decision = MethodSelectionEngine.select_for_plan(decision, plan, sem, None, df)

        self.assertEqual(decision.analytical_task_status, "UNSUPPORTED_ANALYTICAL_TASK")
        self.assertEqual(decision.admissible_families, set())
        self.assertIsNone(decision.selected_method_code)
        self.assertTrue(decision.fallback_used)
        self.assertIn("UNSUPPORTED_ANALYTICAL_TASK", decision.rationale)

    def test_prescriptive_task_is_unsupported(self):
        sem = _semantic()
        df = pd.DataFrame({"revenue": [1, 2, 3], "region": ["a", "b", "c"]})
        q = "What should we do to increase revenue?"
        plan = UniversalQuestionCompiler.compile(q, semantic=sem, df=df)
        self.assertEqual(plan.task, "PRESCRIPTIVE")
        intent = IntentEngine.parse_intent(q)
        decision = MethodSelectionEngine.decide(intent, sem, q, primary_df=df)
        decision = MethodSelectionEngine.select_for_plan(decision, plan, sem, None, df)
        self.assertEqual(decision.analytical_task_status, "UNSUPPORTED_ANALYTICAL_TASK")


class TestF_SupportingExperimentPreserved(unittest.TestCase):
    """F. A legitimate churn specialization is not clobbered by the canonical
    task overwrite (decision.problem_class stays SURVIVAL_CHURN even though
    no compiler task named 'CHURN' exists)."""

    def test_churn_specialization_survives_canonical_reconciliation(self):
        sem = _semantic(metric="churned", churn_event="churned",
                         md=MetricDefinition(
                             name="churned", table_name="customers", source_columns=["churned"],
                             semantic_type="binary", aggregation_type=AggregationType.SUM,
                             is_additive=True, valid_aggregations=[AggregationType.SUM],
                             semantic_resolution_status="RESOLVED"))
        df = pd.DataFrame({"churned": [0, 1] * 20, "region": ["a", "b"] * 20})
        q = "Why are customers churning?"
        plan = UniversalQuestionCompiler.compile(q, semantic=sem, df=df)
        intent = IntentEngine.parse_intent(q)
        decision = MethodSelectionEngine.decide(intent, sem, q, primary_df=df)
        self.assertEqual(decision.problem_class, ProblemClass.SURVIVAL_CHURN,
                          "precondition: decide() must have resolved the churn specialization")

        decision = MethodSelectionEngine.select_for_plan(decision, plan, sem, None, df)
        self.assertEqual(decision.problem_class, ProblemClass.SURVIVAL_CHURN,
                          "canonical-task overwrite must not clobber a resolved churn specialization")
        self.assertEqual(decision.canonical_task, plan.task)
        self.assertEqual(decision.analytical_task_status, "SPECIALIST_BATTERY")


class TestG_ClaimCeiling(unittest.TestCase):
    """G. Association evidence cannot produce a causal verdict: max_verdict_tier
    caps at STATISTICALLY_SIGNIFICANT for CORRELATIONAL, never uncapped-causal."""

    def test_association_claim_ceiling_capped(self):
        sem = _semantic(secondary="cost")
        df = pd.DataFrame({"revenue": [1, 2, 3, 4, 5] * 5, "cost": [2, 3, 4, 5, 6] * 5})
        q = "Is there a correlation between revenue and cost?"
        plan = UniversalQuestionCompiler.compile(q, semantic=sem, df=df)
        intent = IntentEngine.parse_intent(q)
        decision = MethodSelectionEngine.decide(intent, sem, q, primary_df=df)
        decision = MethodSelectionEngine.select_for_plan(decision, plan, sem, None, df)
        self.assertEqual(decision.max_verdict_tier, VERDICT_TIER_STATISTICALLY_SIGNIFICANT)

    def test_causal_wording_does_not_bypass_observational_gate(self):
        sem = _semantic(group="region")
        df = pd.DataFrame({"revenue": [1, 2, 3, 4, 5] * 5, "region": (["a", "b"] * 12) + ["a"]})
        q = "What is the causal effect of region on revenue?"
        plan = UniversalQuestionCompiler.compile(q, semantic=sem, df=df)
        intent = IntentEngine.parse_intent(q)
        decision = MethodSelectionEngine.decide(intent, sem, q, primary_df=df)
        self.assertEqual(decision.problem_class, ProblemClass.CAUSAL)
        decision = MethodSelectionEngine.select_for_plan(decision, plan, sem, None, df)
        # Canonical-task overwrite must not silently drop the causal cap either.
        self.assertEqual(decision.problem_class, ProblemClass.CAUSAL)
        self.assertEqual(decision.max_verdict_tier, VERDICT_TIER_STATISTICALLY_SIGNIFICANT)
        self.assertEqual(decision.causal_intent.assumed_dag_id, "NO_DAG_ASSUMED_OBSERVATIONAL_ONLY")

    def test_bare_root_cause_wording_not_promoted_to_causal_by_compiler_task(self):
        """Regression guard for the CAUSAL specialization carve-out itself:
        the compiler's cruder 'caus(e|ed|es|al)' stem match would classify
        plain root-cause wording as task=CAUSAL, but decide()'s narrower,
        intentional causal-language gate correctly does NOT. The canonical
        overwrite must not force problem_class to CAUSAL in that case,
        since doing so would remove the max_verdict_tier cap decide() chose
        not to apply, i.e. it would make claims look causal-eligible when
        the dedicated causal-effect-language check found none.
        """
        sem = _semantic(group="region")
        df = pd.DataFrame({"revenue": [1, 2, 3, 4, 5] * 5, "region": (["a", "b"] * 12) + ["a"]})
        q = "What is the root cause of the revenue drop?"
        plan = UniversalQuestionCompiler.compile(q, semantic=sem, df=df)
        intent = IntentEngine.parse_intent(q)
        decision = MethodSelectionEngine.decide(intent, sem, q, primary_df=df)
        self.assertEqual(decision.problem_class, ProblemClass.DIAGNOSTIC)
        self.assertIsNone(decision.causal_intent)

        decision = MethodSelectionEngine.select_for_plan(decision, plan, sem, None, df)
        if plan.task == "CAUSAL":
            # The compiler's broader regex did fire here; prove the carve-out held.
            self.assertEqual(decision.problem_class, ProblemClass.DIAGNOSTIC)
            self.assertIsNone(decision.causal_intent)
            self.assertIsNone(decision.max_verdict_tier)


class TestRegistryAuthorityDriftFailsClosed(unittest.TestCase):
    """Every canonical task must resolve to a non-UNKNOWN authority status;
    the registry itself must cover the full closed vocabulary."""

    def test_all_thirteen_tasks_classified(self):
        expected = {
            "ASSOCIATION", "FORECAST", "PREDICTION", "SEGMENTATION", "RECONCILIATION",
            "COMPARISON", "DIAGNOSTIC", "DESCRIPTIVE", "CAUSAL", "DATA_QUALITY",
            "GOVERNANCE", "PRESCRIPTIVE", "GENERAL_EXPLORATION",
        }
        self.assertEqual(set(CANONICAL_TASK_AUTHORITY.keys()), expected)
        for task, entry in CANONICAL_TASK_AUTHORITY.items():
            self.assertIn(entry.status, {
                TaskAuthorityStatus.IMPLEMENTED,
                TaskAuthorityStatus.SPECIALIST_BATTERY,
                TaskAuthorityStatus.UNSUPPORTED,
            })


if __name__ == "__main__":
    unittest.main()
