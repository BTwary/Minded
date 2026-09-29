"""v20-B1 regression suite: fail-closed hardening of the semantic-binding /
contract-validation gate (see docs audit dated 2026-09-16).

Covers the five ordered fixes:
  1. An experiment column with NO semantic binding at all is now a HARD
     ERROR for PRIMARY/UNASSIGNED experiments (previously silently passed).
  2. A PRIMARY predictor not among the contract's EXPLANATORY_VARIABLE
     bindings is now a HARD ERROR (previously only a warning).
  3. A column bound to an incompatible SemanticRole for how the experiment
     uses it (e.g. a predictor bound only as OUTCOME/TARGET) is now a HARD
     ERROR for PRIMARY/UNASSIGNED experiments.
  4. The controller's Simpson's-paradox conditional candidate now declares
     experiment_role=ADVERSARIAL explicitly at construction, rather than
     relying on the gate-time UNASSIGNED backfill (which mis-tagged it as
     PRIMARY/SUPPORTING).
  5. SemanticBindingSet's compatibility projections (projected_target_column,
     projected_time_column, projected_group_dimension) now fail closed
     (return None) on >1 executable candidates instead of first-wins.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.engines.method_selection import MethodFamily
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.intelligence.experiment_synthesizer import CandidateExperiment
from packages.analytics_core.src.intelligence.experiment_contract_validation import (
    AnalyticalContractSnapshot,
    validate_experiment_against_contract,
)
from packages.analytics_core.src.runtime.controller import build_simpsons_conditional_candidate
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition
from packages.schemas.src.analysis import AggregationType
from packages.schemas.src.semantic_binding import SemanticBinding, SemanticBindingSet
from packages.schemas.src.semantic_role import ExperimentRole, ResolutionStatus, SemanticRole


def _contract(bindings, problem_class="ASSOCIATION", method_family=MethodFamily.CORRELATION,
              selected_method="association_categorical_binary"):
    return AnalyticalContractSnapshot(
        contract_id="CONTRACT-TEST-B1", contract_version=1,
        problem_class=problem_class, method_family=method_family,
        selected_method=selected_method, estimand="association",
        semantic_bindings=SemanticBindingSet(bindings=bindings),
    )


def _experiment(**kwargs):
    defaults = dict(
        code="EXP-TEST-B1", target_hypothesis_code="HYP-01", tool_name="duckdb_sql", query_sql="SELECT 1",
        description="test", aggregation_type="CORRELATION",
        discriminating_power=0.8, discrimination_value=0.8, estimated_cost=1.0,
        reliability_weight=0.9, decision_relevance=0.9,
        method_code="association_categorical_binary",
        experiment_role=ExperimentRole.PRIMARY.value,
    )
    defaults.update(kwargs)
    return CandidateExperiment(**defaults)


# ---------------------------------------------------------------------------
# Fix 1: unbound column -> hard error for PRIMARY/UNASSIGNED
# ---------------------------------------------------------------------------

class TestUnboundColumnHardError(unittest.TestCase):
    def test_primary_experiment_with_entirely_unbound_predictor_rejected(self):
        contract = _contract([
            SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED),
        ])
        # "region" has NO binding at all in the contract's SemanticBindingSet.
        exp = _experiment(metrics=["churned", "region"], target_metric="churned", outcome="churned")
        result = validate_experiment_against_contract(exp, contract)
        self.assertFalse(result.compatible)
        self.assertTrue(any("UNBOUND_COLUMN" in e for e in result.errors), result.errors)

    def test_adversarial_experiment_with_unbound_column_only_warns(self):
        """A SUPPORTING/ADVERSARIAL/VERIFICATION experiment may legitimately
        probe a column the canonical binding set has not registered (e.g. a
        confounding dimension surfaced by adversarial challenge) -- it is
        warned, not blocked, preserving today's adversarial-challenge
        pipeline."""
        contract = _contract([
            SemanticBinding(column="revenue_change", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED),
        ], problem_class="DIAGNOSTIC", method_family=MethodFamily.DIAGNOSTIC_BATTERY,
           selected_method="generic_diagnostic_battery")
        exp = _experiment(code="EXP-CONFOUND", experiment_role=ExperimentRole.ADVERSARIAL.value,
                           method_code="generic_diagnostic_battery", aggregation_type="SUM",
                           metrics=["revenue_change"], dimensions=["unregistered_dim"],
                           target_metric="revenue_change")
        result = validate_experiment_against_contract(exp, contract)
        self.assertTrue(result.compatible, result.errors)
        self.assertTrue(any("UNBOUND_COLUMN" in w for w in result.warnings), result.warnings)

    def test_fully_bound_predictor_still_passes(self):
        contract = _contract([
            SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED),
            SemanticBinding(column="support_tickets", table="t", role=SemanticRole.EXPLANATORY_VARIABLE,
                             resolution_status=ResolutionStatus.RESOLVED),
        ])
        exp = _experiment(metrics=["churned", "support_tickets"], target_metric="churned", outcome="churned")
        result = validate_experiment_against_contract(exp, contract)
        self.assertTrue(result.compatible, result.errors)


# ---------------------------------------------------------------------------
# Fix 2: PRIMARY predictor mismatch -> hard error (was a warning)
# ---------------------------------------------------------------------------

class TestPredictorMismatchHardError(unittest.TestCase):
    def test_primary_predictor_outside_declared_explanatory_set_rejected(self):
        contract = _contract([
            SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED),
            SemanticBinding(column="support_tickets", table="t", role=SemanticRole.EXPLANATORY_VARIABLE,
                             resolution_status=ResolutionStatus.RESOLVED),
            # "price" is bound, but as ENTITY_KEY -- not a predictor-
            # compatible role (EXPLANATORY_VARIABLE/GROUPING_DIMENSION/
            # TREATMENT/TIME_VARIABLE are; see predictor_compatible_columns()),
            # so it's outside the declared predictor set while still
            # avoiding the fix-1 unbound-column path entirely.
            SemanticBinding(column="price", table="t", role=SemanticRole.ENTITY_KEY,
                             resolution_status=ResolutionStatus.RESOLVED),
        ])
        exp = _experiment(metrics=["churned"], target_metric="churned", outcome="churned",
                           predictors=["price"], dimensions=["price"])
        result = validate_experiment_against_contract(exp, contract)
        self.assertFalse(result.compatible)
        self.assertTrue(any("PREDICTOR_NOT_IN_CONTRACT" in e for e in result.errors), result.errors)

    def test_supporting_predictor_mismatch_still_only_warns(self):
        contract = _contract([
            SemanticBinding(column="revenue_change", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED),
            SemanticBinding(column="region", table="t", role=SemanticRole.EXPLANATORY_VARIABLE,
                             resolution_status=ResolutionStatus.RESOLVED),
            SemanticBinding(column="channel", table="t", role=SemanticRole.GROUPING_DIMENSION,
                             resolution_status=ResolutionStatus.RESOLVED),
        ], problem_class="DIAGNOSTIC", method_family=MethodFamily.DIAGNOSTIC_BATTERY,
           selected_method="generic_diagnostic_battery")
        exp = _experiment(code="EXP-SUPPORT", experiment_role=ExperimentRole.SUPPORTING.value,
                           method_code="generic_diagnostic_battery", aggregation_type="SUM",
                           metrics=["revenue_change"], dimensions=["channel"],
                           predictors=["channel"], target_metric="revenue_change")
        result = validate_experiment_against_contract(exp, contract)
        # channel is bound (GROUPING_DIMENSION), just not in the
        # EXPLANATORY_VARIABLE set -- the PREDICTOR_NOT_IN_CONTRACT check
        # only applies to PRIMARY/UNASSIGNED experiments, so a SUPPORTING
        # experiment using it is unaffected (neither errors nor warns).
        self.assertTrue(result.compatible, result.errors)
        self.assertFalse(any("PREDICTOR_NOT_IN_CONTRACT" in e for e in result.errors), result.errors)


# ---------------------------------------------------------------------------
# Fix 3: role-incompatible column usage -> hard error for PRIMARY/UNASSIGNED
# ---------------------------------------------------------------------------

class TestRoleIncompatibleColumnUsage(unittest.TestCase):
    def test_predictor_bound_only_as_outcome_rejected(self):
        contract = _contract([
            SemanticBinding(column="churned", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED),
            # "cancelled" is bound, but ONLY as OUTCOME -- using it as a
            # predictor alongside "churned" is a role mismatch, not merely
            # "not declared as explanatory" (contract_explanatory is empty
            # here, so fix-2's PREDICTOR_NOT_IN_CONTRACT wouldn't fire).
            SemanticBinding(column="cancelled", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED),
        ])
        exp = _experiment(metrics=["churned", "cancelled"], target_metric="churned", outcome="churned",
                           predictors=["cancelled"])
        result = validate_experiment_against_contract(exp, contract)
        self.assertFalse(result.compatible)
        self.assertTrue(any("ROLE_INCOMPATIBLE" in e for e in result.errors), result.errors)


# ---------------------------------------------------------------------------
# Fix 4: controller's Simpson's-conditional candidate is explicitly
# ADVERSARIAL at construction time.
# ---------------------------------------------------------------------------

class TestConditionalCandidateExplicitRole(unittest.TestCase):
    def test_simpsons_conditional_candidate_is_explicitly_adversarial(self):
        semantic = SemanticResolution(
            primary_dataset_name="customers", target_metric_col="monthly_spend",
            group_dimension_col="region", time_col=None, table_grain="row",
            available_numeric_cols=["monthly_spend"], available_categorical_cols=["region"],
            metric_definition=MetricDefinition(
                name="monthly_spend", table_name="customers", source_columns=["monthly_spend"],
                semantic_type="sum_measure", aggregation_type=AggregationType.SUM,
            ),
        )
        candidate, diagnostic = build_simpsons_conditional_candidate(semantic, "plan", "HYP-01")
        self.assertIsNone(diagnostic)
        self.assertIsNotNone(candidate)
        self.assertEqual(ExperimentRole.ADVERSARIAL.value, candidate.experiment_role)


# ---------------------------------------------------------------------------
# Fix 5: projections fail closed (None) on >1 executable candidates.
# ---------------------------------------------------------------------------

class TestProjectionsFailClosedOnAmbiguity(unittest.TestCase):
    def test_projected_target_column_none_on_two_executable_targets(self):
        bindings = SemanticBindingSet(bindings=[
            SemanticBinding(column="revenue", table="t", role=SemanticRole.TARGET,
                             resolution_status=ResolutionStatus.RESOLVED),
            SemanticBinding(column="signups", table="t", role=SemanticRole.TARGET,
                             resolution_status=ResolutionStatus.RESOLVED),
        ])
        self.assertIsNone(bindings.projected_target_column())

    def test_projected_target_column_resolved_on_single_candidate(self):
        bindings = SemanticBindingSet(bindings=[
            SemanticBinding(column="revenue", table="t", role=SemanticRole.TARGET,
                             resolution_status=ResolutionStatus.RESOLVED),
        ])
        self.assertEqual("revenue", bindings.projected_target_column())

    def test_projected_target_column_none_on_zero_candidates(self):
        bindings = SemanticBindingSet(bindings=[])
        self.assertIsNone(bindings.projected_target_column())

    def test_projected_time_column_none_on_ambiguity(self):
        bindings = SemanticBindingSet(bindings=[
            SemanticBinding(column="signup_date", table="t", role=SemanticRole.TIME_VARIABLE,
                             resolution_status=ResolutionStatus.RESOLVED),
            SemanticBinding(column="last_login_date", table="t", role=SemanticRole.TIME_VARIABLE,
                             resolution_status=ResolutionStatus.RESOLVED),
        ])
        self.assertIsNone(bindings.projected_time_column())

    def test_projected_group_dimension_none_on_ambiguity(self):
        bindings = SemanticBindingSet(bindings=[
            SemanticBinding(column="region", table="t", role=SemanticRole.GROUPING_DIMENSION,
                             resolution_status=ResolutionStatus.RESOLVED),
            SemanticBinding(column="segment", table="t", role=SemanticRole.GROUPING_DIMENSION,
                             resolution_status=ResolutionStatus.RESOLVED),
        ])
        self.assertIsNone(bindings.projected_group_dimension())


if __name__ == "__main__":
    unittest.main()
