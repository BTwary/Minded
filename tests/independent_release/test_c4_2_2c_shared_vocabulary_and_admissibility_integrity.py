"""v20-C4.2.2c: regression tests for the two production fixes made in
response to AAOS_V20C4_2_2B_COMPILER_SEMANTIC_AUTHORITY_AUDIT.md's
Findings 2 and 3.

Fix 1 (Finding 2): UniversalQuestionCompiler._ASSOC is now a direct alias
of IntentEngine._CORRELATION_KEYWORD_RE, so the two classifiers cannot
disagree on association vocabulary again.

Fix 2 (Finding 3): gate_method_for_plan()'s numeric branch now checks
predictor dtype as well as target dtype (matching admissibility_for_plan()),
and admissibility_for_plan() is now method-code-aware, so it can never
again silently report a different method's admissibility under a
requested method's name.

These tests exercise the REAL pipeline end to end -- no dataclasses.replace()
arrangement -- reproducing the exact natural-language questions the audit
used to find the defects.
"""
import dataclasses
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.method_selection import MethodRegistry, MethodSelectionEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.schemas.src.semantic_resolution_contract import QuestionRoleProposal


def _df(n=150, seed=2026):
    rng = np.random.RandomState(seed)
    price = rng.uniform(5, 80, n)
    annual_sales = 300 - 3 * price + rng.normal(0, 10, n)
    customer_id = [f"CUST-{i:05d}" for i in range(n)]
    return pd.DataFrame({"customer_id": customer_id, "price": price, "annual_sales": annual_sales})


def _run(question, df):
    intent = IntentEngine.parse_intent(question, available_columns=list(df.columns))
    semantic = SemanticEngine.resolve_schema_static(intent, {"data_table": df})
    plan = UniversalQuestionCompiler.compile(question, semantic=semantic, df=df)
    roles = QuestionRoleProposal.from_plan_semantics(plan.semantics)
    decision = MethodSelectionEngine.decide(intent=intent, semantic=semantic, question=question, primary_df=df, question_roles=roles)
    result = MethodSelectionEngine.select_for_plan(dataclasses.replace(decision), plan, semantic, None, df)
    return intent, semantic, plan, decision, result


class TestFix1SharedAssociationVocabulary(unittest.TestCase):
    """Finding 2: the compiler's task classifier and IntentEngine must
    agree on association vocabulary."""

    def test_influence_now_classifies_as_association_task(self):
        df = _df()
        _, _, plan, decision, result = _run("Does price influence annual_sales?", df)
        self.assertEqual(plan.task, "ASSOCIATION")
        self.assertEqual(decision.problem_class.value, "CORRELATIONAL")
        self.assertEqual(result.analytical_task_status, "IMPLEMENTED")
        self.assertEqual(result.selected_method_code, "association_numeric")

    def test_affect_still_classifies_as_association_task(self):
        # Sanity: the already-working synonym must remain unaffected by
        # switching _ASSOC to the shared regex.
        df = _df()
        _, _, plan, decision, result = _run("Does price affect annual_sales?", df)
        self.assertEqual(plan.task, "ASSOCIATION")
        self.assertEqual(result.selected_method_code, "association_numeric")

    def test_impact_now_classifies_as_association_task(self):
        # Another vocabulary item IntentEngine had that the compiler's old
        # _ASSOC pattern lacked.
        df = _df()
        _, _, plan, decision, result = _run("What is the impact of price on annual_sales?", df)
        self.assertEqual(plan.task, "ASSOCIATION")

    def test_associated_wording_still_classifies_as_association_task(self):
        df = _df()
        _, _, plan, decision, result = _run("Is price associated with annual_sales?", df)
        self.assertEqual(plan.task, "ASSOCIATION")
        self.assertEqual(result.selected_method_code, "association_numeric")

    def test_shared_regex_object_identity(self):
        # The whole point of the fix is a single source of truth -- assert
        # it directly, not just its observable effects.
        from packages.analytics_core.src.engines.intent import _CORRELATION_KEYWORD_RE
        self.assertIs(UniversalQuestionCompiler._ASSOC, _CORRELATION_KEYWORD_RE)


class TestFix2MethodAdmissibilityIntegrity(unittest.TestCase):
    """Finding 3: an invalid (non-numeric) predictor must never cause
    association_numeric to be scored/selected/executor-bound."""

    def test_customer_id_predictor_never_selects_association_numeric(self):
        df = _df()
        _, _, plan, decision, result = _run("Is customer_id associated with annual_sales?", df)
        self.assertNotEqual(result.selected_method_code, "association_numeric")
        # No admissible candidate at all is the correct outcome here (both
        # association_categorical_binary and association_numeric are
        # genuinely inadmissible for this target/predictor combination).
        self.assertIsNone(result.selected_method_code)

    def test_customer_id_predictor_never_gets_association_numeric_executor(self):
        df = _df()
        _, _, plan, decision, result = _run("Is customer_id associated with annual_sales?", df)
        self.assertNotEqual(
            MethodRegistry.executor_id(result.selected_method_code) if result.selected_method_code else None,
            "scientific_loop:association_numeric",
        )

    def test_valid_numeric_predictor_still_selects_association_numeric(self):
        # Before/after clean contrast the review explicitly asked for.
        df = _df()
        _, _, plan, decision, result = _run("Is price associated with annual_sales?", df)
        self.assertEqual(result.selected_method_code, "association_numeric")
        self.assertEqual(MethodRegistry.executor_id(result.selected_method_code), "scientific_loop:association_numeric")

    def test_gate_method_for_plan_is_now_dtype_symmetric(self):
        from packages.analytics_core.src.engines.method_selection import gate_method_for_plan
        df = _df()
        _, semantic, plan, decision, result = _run("Is customer_id associated with annual_sales?", df)
        self.assertIsNone(gate_method_for_plan(plan, semantic, df))
        _, semantic2, plan2, decision2, result2 = _run("Is price associated with annual_sales?", df)
        self.assertEqual(gate_method_for_plan(plan2, semantic2, df), "association_numeric")

    def test_admissibility_for_plan_rejects_mismatched_method_code(self):
        df = _df()
        _, semantic, plan, decision, result = _run("Is customer_id associated with annual_sales?", df)
        # Directly request the method that would previously have been
        # scored admissible via the substitution bug.
        ok, gate = MethodRegistry.admissibility_for_plan(plan, semantic, None, df, method_code="association_numeric")
        self.assertFalse(ok)
        self.assertIn("requested_method_mismatch", "".join(gate["errors"]))

    def test_admissibility_for_plan_backward_compatible_without_method_code(self):
        # Every pre-existing caller omits method_code entirely; confirm
        # that path (derive-and-validate-own-best-fit) is untouched.
        df = _df()
        _, semantic, plan, decision, result = _run("Is price associated with annual_sales?", df)
        ok, gate = MethodRegistry.admissibility_for_plan(plan, semantic, None, df)
        self.assertTrue(ok)
        self.assertEqual(gate["method"], "association_numeric")


if __name__ == "__main__":
    unittest.main()
