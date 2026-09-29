"""v20-A regression suite: semantic-role binding + experiment contract
validation (MindEd_AAOS_v20-A).

Two layers are tested:

1. Fast, deterministic unit-style coverage of
   ``SemanticBindingSet.validate()`` and
   ``validate_experiment_against_contract()`` against hand-built contracts
   and candidates (section 16's A-I cases, plus section 20's 10 named
   scenarios).
2. A small number of true end-to-end runs through the real
   ``InvestigationController``, inspecting the actually-persisted
   ``InvestigationContract.semantic_bindings_json`` (section 15, 20).

This suite does not re-verify v19's own scenarios (see
test_v19_canonical_authority.py) -- section 21's "run the full v19
regression suite" is satisfied by running the whole tests/independent_release
directory together with this file, not by duplicating those cases here.
"""
import os
import sys
import unittest
import uuid

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Investigation, InvestigationContract
from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
from packages.analytics_core.src.engines.method_selection import MethodFamily
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.intelligence.experiment_synthesizer import CandidateExperiment
from packages.analytics_core.src.intelligence.experiment_contract_validation import (
    AnalyticalContractSnapshot,
    validate_experiment_against_contract,
)
from packages.analytics_core.src.intelligence.semantic_binding_builder import build_binding_set
from packages.schemas.src.semantic_binding import SemanticBinding, SemanticBindingSet
from packages.schemas.src.semantic_role import ExperimentRole, ResolutionStatus, SemanticRole
from packages.schemas.src.analysis import AggregationType
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _semantic(metric="churned", group=None, secondary=None, churn_event=None,
              table="customers", exposure=None, censored=None, confounders=None):
    return SemanticResolution(
        primary_dataset_name=table, target_metric_col=metric, group_dimension_col=group,
        time_col=None, table_grain="record_level", available_numeric_cols=[metric],
        available_categorical_cols=[group] if group else [], world_model=None,
        metric_definition=MetricDefinition(
            name=metric, table_name=table, source_columns=[metric], semantic_type="sum_measure",
            aggregation_type=AggregationType.SUM, is_additive=True,
            valid_aggregations=[AggregationType.SUM], semantic_resolution_status="RESOLVED"),
        direction_hint="unspecified", secondary_metric_col=secondary,
        churn_event_col=churn_event, churn_exposure_col=exposure, churn_censored_col=censored,
        churn_confounder_cols=confounders or [], churn_outcome_available=churn_event is not None,
    )


def _contract(bindings, problem_class="ASSOCIATION", method_family=MethodFamily.CORRELATION,
              selected_method="association_categorical_binary", grain=None, population="primary_dataset",
              time_scope=None):
    return AnalyticalContractSnapshot(
        contract_id="CONTRACT-TEST", contract_version=1,
        problem_class=problem_class, method_family=method_family,
        selected_method=selected_method, estimand="association",
        semantic_bindings=SemanticBindingSet(bindings=bindings),
        population_scope=population, grain=grain, time_scope=time_scope,
    )


def _experiment(**kwargs):
    defaults = dict(
        code="EXP-TEST", target_hypothesis_code="HYP-01", tool_name="duckdb_sql", query_sql="SELECT 1",
        description="test", aggregation_type="CORRELATION",
        discriminating_power=0.8, discrimination_value=0.8, estimated_cost=1.0,
        reliability_weight=0.9, decision_relevance=0.9,
        method_code="association_categorical_binary",
        experiment_role=ExperimentRole.PRIMARY.value,
    )
    defaults.update(kwargs)
    return CandidateExperiment(**defaults)


class InMemoryDatasetProvider(BaseDatasetProvider):
    def __init__(self, datasets_map):
        self.datasets_map = datasets_map

    def acquire_context(self, project_id, dataset_ids=None):
        import hashlib
        fingerprints = {n: hashlib.sha256(df.to_json().encode()).hexdigest() for n, df in self.datasets_map.items()}
        return InvestigationDataContext(
            project_id=project_id, datasets_map=self.datasets_map,
            dataset_fingerprints=fingerprints, requested_dataset_ids=dataset_ids,
        )


def _run_investigation(df: pd.DataFrame, question: str):
    from packages.analytics_core.src.runtime.controller import InvestigationController

    db = SessionLocal()
    inv_id = f"INV-V20A-{uuid.uuid4().hex[:10]}"
    tbl = f"tbl_{uuid.uuid4().hex[:8]}"
    proj_id = f"proj-v20a-{uuid.uuid4().hex[:8]}"
    inv = Investigation(id=inv_id, project_id=proj_id, question=question, status="PLANNED")
    db.add(inv)
    db.commit()
    db.close()

    provider = InMemoryDatasetProvider({tbl: df})
    ctrl = InvestigationController(session_factory=SessionLocal, dataset_provider=provider)
    ctrl.execute_investigation(investigation_id=inv_id, worker_id="test-v20a")

    fresh_db = SessionLocal()
    result = fresh_db.query(Investigation).filter(Investigation.id == inv_id).first()
    contract = (fresh_db.query(InvestigationContract)
                .filter(InvestigationContract.investigation_id == inv_id)
                .order_by(InvestigationContract.version.desc()).first())
    fresh_db.close()
    return result, contract


# ---------------------------------------------------------------------------
# Section 20, case 1: Association churned <-> support_tickets (valid)
# ---------------------------------------------------------------------------

class TestValidAssociation(unittest.TestCase):
    def test_association_valid_binding_and_experiment(self):
        semantic = _semantic(metric="churned", secondary="support_tickets")
        bindings = build_binding_set(semantic, "ASSOCIATION")
        self.assertEqual([], bindings.validate())
        contract = _contract(bindings.bindings)
        exp = _experiment(metrics=["churned", "support_tickets"], target_metric="churned")
        result = validate_experiment_against_contract(exp, contract)
        self.assertTrue(result.compatible, result.errors)


# ---------------------------------------------------------------------------
# Section 15/20 case 2: False self-correlation churned <-> churned (unit)
# ---------------------------------------------------------------------------

class TestSelfCorrelationCollision(unittest.TestCase):
    def test_self_correlation_rejected(self):
        semantic = _semantic(metric="churned", secondary="support_tickets")
        bindings = build_binding_set(semantic, "ASSOCIATION")
        contract = _contract(bindings.bindings)
        # The candidate itself carries the bug: both sides of the pairing
        # are the same column, regardless of what the contract says.
        exp = _experiment(metrics=["churned", "churned"], target_metric="churned")
        result = validate_experiment_against_contract(exp, contract)
        self.assertFalse(result.compatible)
        self.assertTrue(any("OUTCOME_PREDICTOR_COLLISION" in e for e in result.errors), result.errors)

    def test_self_correlation_rejected_end_to_end_via_controller(self):
        """Section 15: must be tested through the actual InvestigationController,
        not only the validator unit function. We cannot force AA-OS's own
        (already-fixed) synthesis pipeline to reproduce the historical bug, so
        this drives the real controller/contract/binding pipeline for a
        genuine association question and independently confirms (a) the
        persisted contract carries non-conflicting bindings and (b) directly
        feeding a hand-crafted colliding candidate against that SAME
        persisted contract is rejected -- proving the gate would reject the
        bug in production if synthesis ever regressed to reproduce it.
        """
        import random
        from scripts.generate_churn_seed_data import scenario_genuine_effect
        rng = random.Random(42)
        df, _ = scenario_genuine_effect(rng)
        inv, contract = _run_investigation(df, "Which customer segments have higher churn rates?")
        self.assertIsNotNone(contract)
        bindings = SemanticBindingSet.from_dict_list(contract.semantic_bindings_json or [])
        self.assertEqual([], bindings.validate(), "persisted contract bindings must be conflict-free")

        snapshot = AnalyticalContractSnapshot(
            contract_id=contract.id, contract_version=contract.version,
            problem_class=contract.problem_class, method_family=None,
            selected_method=(contract.selected_method_json or {}).get("method_code"),
            estimand="association", semantic_bindings=bindings,
        )
        outcome_col = bindings.projected_target_column()
        if outcome_col:
            bad_exp = _experiment(metrics=[outcome_col, outcome_col], target_metric=outcome_col)
            result = validate_experiment_against_contract(bad_exp, snapshot)
            self.assertFalse(result.compatible)
            self.assertTrue(any("OUTCOME_PREDICTOR_COLLISION" in e for e in result.errors), result.errors)


# ---------------------------------------------------------------------------
# Section 20 case 3/4: ambiguous vs resolved churn outcome
# ---------------------------------------------------------------------------

class TestResolutionStatusGating(unittest.TestCase):
    def test_ambiguous_outcome_blocks_execution(self):
        bindings = SemanticBindingSet(bindings=[
            SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.AMBIGUOUS, confidence=0.4),
            SemanticBinding(column="cancelled", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.AMBIGUOUS, confidence=0.4),
        ])
        contract = _contract(bindings.bindings, problem_class="SURVIVAL_CHURN")
        exp = _experiment(metrics=["churned"], target_metric="churned", outcome="churned")
        result = validate_experiment_against_contract(exp, contract)
        self.assertFalse(result.compatible)
        self.assertTrue(any("UNRESOLVED_BINDING" in e for e in result.errors), result.errors)

    def test_resolved_outcome_permits_execution(self):
        bindings = SemanticBindingSet(bindings=[
            SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED, confidence=1.0),
            SemanticBinding(column="support_tickets", table="t", role=SemanticRole.EXPLANATORY_VARIABLE,
                             resolution_status=ResolutionStatus.RESOLVED, confidence=1.0),
        ])
        contract = _contract(bindings.bindings, problem_class="ROOT_CAUSE")
        exp = _experiment(metrics=["churned", "support_tickets"], target_metric="churned",
                           problem_class="ROOT_CAUSE")
        result = validate_experiment_against_contract(exp, contract)
        self.assertTrue(result.compatible, result.errors)


# ---------------------------------------------------------------------------
# Section 20 case 5: forecast with wrong target rejected
# ---------------------------------------------------------------------------

class TestForecastWrongTarget(unittest.TestCase):
    def test_forecast_wrong_target_rejected(self):
        bindings = SemanticBindingSet(bindings=[
            SemanticBinding(column="revenue", table="t", role=SemanticRole.TARGET,
                             resolution_status=ResolutionStatus.RESOLVED),
        ])
        contract = _contract(bindings.bindings, problem_class="FORECAST",
                              method_family=MethodFamily.FORECAST, selected_method="forecast_rolling_origin")
        exp = _experiment(code="EXP-FORECAST-TREND", method_code="forecast_rolling_origin",
                           aggregation_type="FORECAST", metrics=["signups"], target_metric="signups",
                           outcome="signups")
        result = validate_experiment_against_contract(exp, contract)
        self.assertFalse(result.compatible)
        self.assertTrue(any("OUTCOME_MISMATCH" in e for e in result.errors), result.errors)


# ---------------------------------------------------------------------------
# Section 20 case 6/7/8: SUPPORTING / ADVERSARIAL / VERIFICATION acceptance
# ---------------------------------------------------------------------------

class TestExperimentRoleDivergence(unittest.TestCase):
    def _base_bindings(self):
        return [
            SemanticBinding(column="revenue_change", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED),
            SemanticBinding(column="region", table="t", role=SemanticRole.EXPLANATORY_VARIABLE,
                             resolution_status=ResolutionStatus.RESOLVED),
        ]

    def test_supporting_diagnostic_battery_accepted_alongside_correlation_contract(self):
        contract = _contract(self._base_bindings(), problem_class="DIAGNOSTIC",
                              method_family=MethodFamily.DIAGNOSTIC_BATTERY,
                              selected_method="binary_risk_prediction")
        exp = _experiment(code="EXP-CONFOUND", experiment_role=ExperimentRole.SUPPORTING.value,
                           method_code="generic_diagnostic_battery", aggregation_type="SUM",
                           metrics=["revenue_change"], dimensions=["region"], target_metric="revenue_change")
        result = validate_experiment_against_contract(exp, contract)
        self.assertTrue(result.compatible, result.errors)

    def test_adversarial_experiment_accepted(self):
        contract = _contract(self._base_bindings(), problem_class="ASSOCIATION",
                              method_family=MethodFamily.CORRELATION,
                              selected_method="association_categorical_binary")
        exp = _experiment(code="EXP-ADV", experiment_role=ExperimentRole.ADVERSARIAL.value,
                           method_code="generic_diagnostic_battery", aggregation_type="SUM",
                           metrics=["revenue_change"], dimensions=["region"], target_metric="revenue_change")
        result = validate_experiment_against_contract(exp, contract)
        self.assertTrue(result.compatible, result.errors)

    def test_verification_experiment_requires_same_family(self):
        contract = _contract(self._base_bindings(), problem_class="ASSOCIATION",
                              method_family=MethodFamily.CORRELATION,
                              selected_method="association_categorical_binary")
        good = _experiment(code="EXP-VERIFY", experiment_role=ExperimentRole.VERIFICATION.value,
                            method_code="association_numeric", aggregation_type="CORRELATION",
                            metrics=["revenue_change", "region"], target_metric="revenue_change")
        self.assertTrue(validate_experiment_against_contract(good, contract).compatible)

        bad = _experiment(code="EXP-VERIFY-BAD", experiment_role=ExperimentRole.VERIFICATION.value,
                           method_code="forecast_rolling_origin", aggregation_type="FORECAST",
                           metrics=["revenue_change"], target_metric="revenue_change")
        bad_result = validate_experiment_against_contract(bad, contract)
        self.assertFalse(bad_result.compatible)
        self.assertTrue(any("ESTIMAND_MISMATCH" in e for e in bad_result.errors), bad_result.errors)


# ---------------------------------------------------------------------------
# Section 20 case 9/10: wrong population / wrong grain rejected
# ---------------------------------------------------------------------------

class TestScopeMismatch(unittest.TestCase):
    def test_wrong_population_rejected(self):
        bindings = [SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                                     resolution_status=ResolutionStatus.RESOLVED)]
        contract = _contract(bindings, population="primary_dataset")
        exp = _experiment(metrics=["churned"], target_metric="churned", population="filtered_subset")
        result = validate_experiment_against_contract(exp, contract)
        self.assertFalse(result.compatible)
        self.assertTrue(any("POPULATION_MISMATCH" in e for e in result.errors), result.errors)

    def test_wrong_grain_rejected(self):
        bindings = [SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                                     resolution_status=ResolutionStatus.RESOLVED)]
        contract = _contract(bindings, grain="account_level")
        exp = _experiment(metrics=["churned"], target_metric="churned", grain="transaction_level")
        result = validate_experiment_against_contract(exp, contract)
        self.assertFalse(result.compatible)
        self.assertTrue(any("GRAIN_MISMATCH" in e for e in result.errors), result.errors)


# ---------------------------------------------------------------------------
# Section 16: role-conflict cases A-I
# ---------------------------------------------------------------------------

class TestRoleConflictMatrix(unittest.TestCase):
    def test_A_outcome_and_distinct_explanatory_valid(self):
        semantic = _semantic(metric="churned", secondary="support_tickets")
        bindings = build_binding_set(semantic, "ASSOCIATION")
        self.assertEqual([], bindings.validate())

    def test_B_same_column_outcome_and_explanatory_rejected_by_default(self):
        bindings = SemanticBindingSet(bindings=[
            SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED),
            SemanticBinding(column="churned", table="t", role=SemanticRole.EXPLANATORY_VARIABLE,
                             resolution_status=ResolutionStatus.RESOLVED),
        ])
        conflicts = bindings.validate()
        self.assertEqual(1, len(conflicts))
        self.assertEqual("CONFLICTING_ROLE", conflicts[0].conflict_type.value)

    def test_C_ambiguous_outcome_between_two_candidates(self):
        bindings = SemanticBindingSet(bindings=[
            SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.AMBIGUOUS),
            SemanticBinding(column="cancelled", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.AMBIGUOUS),
        ])
        self.assertEqual(0, len(bindings.executable_bindings()))

    def test_D_two_valid_predictor_candidates_each_own_binding(self):
        bindings = SemanticBindingSet(bindings=[
            SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED),
            SemanticBinding(column="support_tickets", table="t", role=SemanticRole.EXPLANATORY_VARIABLE,
                             resolution_status=ResolutionStatus.RESOLVED),
            SemanticBinding(column="complaints", table="t", role=SemanticRole.EXPLANATORY_VARIABLE,
                             resolution_status=ResolutionStatus.RESOLVED),
        ])
        self.assertEqual([], bindings.validate())
        self.assertEqual(2, len(bindings.get_by_role(SemanticRole.EXPLANATORY_VARIABLE)))

    def test_E_missing_outcome_rejected(self):
        contract = _contract([SemanticBinding(column="support_tickets", table="t",
                                               role=SemanticRole.EXPLANATORY_VARIABLE,
                                               resolution_status=ResolutionStatus.RESOLVED)])
        exp = _experiment(metrics=["support_tickets", "region"], target_metric=None, outcome=None)
        result = validate_experiment_against_contract(exp, contract)
        # No outcome bound at all -> nothing to collide with, but the
        # association-style predictor-missing check should still fire since
        # there's no distinguishable outcome/predictor pairing declared.
        self.assertFalse(result.compatible)

    def test_F_missing_predictor_for_association_rejected(self):
        contract = _contract([SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                                               resolution_status=ResolutionStatus.RESOLVED)])
        exp = _experiment(metrics=["churned"], target_metric="churned", outcome="churned")
        result = validate_experiment_against_contract(exp, contract)
        self.assertFalse(result.compatible)
        self.assertTrue(any("PREDICTOR_MISSING" in e for e in result.errors), result.errors)

    def test_G_wrong_grain_rejected(self):
        contract = _contract([SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                                               resolution_status=ResolutionStatus.RESOLVED)],
                              grain="account_level")
        exp = _experiment(metrics=["churned"], target_metric="churned", grain="session_level")
        self.assertFalse(validate_experiment_against_contract(exp, contract).compatible)

    def test_H_wrong_population_rejected(self):
        contract = _contract([SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                                               resolution_status=ResolutionStatus.RESOLVED)],
                              population="primary_dataset")
        exp = _experiment(metrics=["churned"], target_metric="churned", population="excluded_trial_users")
        self.assertFalse(validate_experiment_against_contract(exp, contract).compatible)

    def test_I_wrong_time_scope_rejected(self):
        contract = _contract([SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                                               resolution_status=ResolutionStatus.RESOLVED)],
                              time_scope={"time_column": "signup_date"})
        exp = _experiment(metrics=["churned"], target_metric="churned",
                           time_scope={"time_column": "last_login_date"})
        self.assertFalse(validate_experiment_against_contract(exp, contract).compatible)


# ---------------------------------------------------------------------------
# Unassigned experiment role must not silently execute as PRIMARY
# ---------------------------------------------------------------------------

class TestUnassignedRole(unittest.TestCase):
    def test_unassigned_role_blocked(self):
        contract = _contract([
            SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED),
            SemanticBinding(column="support_tickets", table="t", role=SemanticRole.EXPLANATORY_VARIABLE,
                             resolution_status=ResolutionStatus.RESOLVED),
        ])
        exp = _experiment(metrics=["churned", "support_tickets"], target_metric="churned",
                           experiment_role=ExperimentRole.UNASSIGNED.value)
        result = validate_experiment_against_contract(exp, contract)
        self.assertFalse(result.compatible)
        self.assertTrue(any("EXPERIMENT_ROLE_UNASSIGNED" in e for e in result.errors), result.errors)


# ---------------------------------------------------------------------------
# End-to-end: real controller run persists conflict-free bindings and the
# hard gate fires (approved) for a genuine, well-posed question.
# ---------------------------------------------------------------------------

class TestEndToEndGateWiring(unittest.TestCase):
    def test_real_investigation_persists_bindings_and_gate_approves(self):
        import random
        from scripts.generate_churn_seed_data import scenario_genuine_effect
        rng = random.Random(42)
        df, _ = scenario_genuine_effect(rng)
        inv, contract = _run_investigation(df, "Which customer segments have higher churn rates?")
        self.assertIsNotNone(contract, "contract must be persisted")
        self.assertTrue(len(contract.semantic_bindings_json or []) > 0,
                         "contract must carry a non-empty semantic binding set")
        bindings = SemanticBindingSet.from_dict_list(contract.semantic_bindings_json)
        self.assertEqual([], bindings.validate())

        db = SessionLocal()
        from apps.api.src.models.entities import InvestigationEvent
        events = (db.query(InvestigationEvent)
                  .filter(InvestigationEvent.investigation_id == inv.id,
                          InvestigationEvent.event_type == "investigation.contract_validation.approved")
                  .all())
        db.close()
        self.assertTrue(len(events) >= 1, "the hard gate must have approved at least one real experiment")


if __name__ == "__main__":
    unittest.main()
