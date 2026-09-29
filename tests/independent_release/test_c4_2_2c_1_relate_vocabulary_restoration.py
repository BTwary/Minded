"""v20-C4.2.2c.1: regression tests restoring the bare `relat\\w*` stem
coverage ("relate" / "related" / "relation", without requiring the
"-ship" suffix) into the single shared association-vocabulary regex,
IntentEngine._CORRELATION_KEYWORD_RE (of which
UniversalQuestionCompiler._ASSOC is a direct alias since C4.2.2c).

Context: C4.2.2c's alias swap deliberately, and explicitly, narrowed this
one edge -- the compiler's OLD, separately maintained `_ASSOC` pattern
matched bare `relat\\w*`, but the shared regex it was replaced with only
had `relationship\\w*`. That narrowing was flagged in
AAOS_V20C4_2_2C_SHARED_VOCABULARY_AND_ADMISSIBILITY_INTEGRITY.md as a
checked, deliberate trade-off (no test in the repo at the time exercised
bare "relate"/"related" wording against UniversalQuestionCompiler), with
restoring it named as the very next step -- this file is that step.

These tests exercise the REAL pipeline end to end (IntentEngine ->
SemanticEngine -> UniversalQuestionCompiler -> MethodSelectionEngine),
the same pattern C4.2.2c's own regression suite uses, so a disagreement
between IntentEngine and the compiler on this vocabulary would show up
here exactly as it would in production.
"""
import dataclasses
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.intent import IntentEngine, _CORRELATION_KEYWORD_RE
from packages.analytics_core.src.engines.method_selection import MethodSelectionEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.schemas.src.semantic_resolution_contract import QuestionRoleProposal


def _df(n=150, seed=2026):
    rng = np.random.RandomState(seed)
    price = rng.uniform(5, 80, n)
    sales = 300 - 3 * price + rng.normal(0, 10, n)
    return pd.DataFrame({"price": price, "sales": sales})


def _run(question, df):
    intent = IntentEngine.parse_intent(question, available_columns=list(df.columns))
    semantic = SemanticEngine.resolve_schema_static(intent, {"data_table": df})
    plan = UniversalQuestionCompiler.compile(question, semantic=semantic, df=df)
    roles = QuestionRoleProposal.from_plan_semantics(plan.semantics)
    decision = MethodSelectionEngine.decide(intent=intent, semantic=semantic, question=question, primary_df=df, question_roles=roles)
    result = MethodSelectionEngine.select_for_plan(dataclasses.replace(decision), plan, semantic, None, df)
    return intent, semantic, plan, decision, result


class TestBareRelatStemRestored(unittest.TestCase):
    """The four wordings named in the C4.2.2c.1 task: two bare-"relate"
    phrasings (previously broken by the C4.2.2c alias narrowing), plus
    "associated"/"influence" as unaffected controls confirming the
    restoration didn't regress the vocabulary C4.2.2c already fixed."""

    def test_are_x_and_y_related_question_wording(self):
        df = _df()
        intent, _, plan, decision, result = _run("Are price and sales related?", df)
        self.assertEqual(intent.intent_type, "CORRELATION")
        self.assertEqual(plan.task, "ASSOCIATION")
        self.assertEqual(result.analytical_task_status, "IMPLEMENTED")
        self.assertEqual(result.selected_method_code, "association_numeric")

    def test_is_x_related_to_y_question_wording(self):
        df = _df()
        intent, _, plan, decision, result = _run("Is price related to sales?", df)
        self.assertEqual(intent.intent_type, "CORRELATION")
        self.assertEqual(plan.task, "ASSOCIATION")
        self.assertEqual(result.analytical_task_status, "IMPLEMENTED")
        self.assertEqual(result.selected_method_code, "association_numeric")

    def test_associated_wording_unaffected_control(self):
        df = _df()
        _, _, plan, decision, result = _run("Are price and sales associated?", df)
        self.assertEqual(plan.task, "ASSOCIATION")
        self.assertEqual(result.selected_method_code, "association_numeric")

    def test_influence_wording_unaffected_control(self):
        df = _df()
        _, _, plan, decision, result = _run("Does price influence sales?", df)
        self.assertEqual(plan.task, "ASSOCIATION")
        self.assertEqual(result.selected_method_code, "association_numeric")

    def test_shared_regex_still_a_single_identity_after_widening(self):
        # C4.2.2c's core guarantee (one shared object, not two drifting
        # copies) must still hold after this widening edit.
        self.assertIs(UniversalQuestionCompiler._ASSOC, _CORRELATION_KEYWORD_RE)

    def test_bare_relat_stem_matches_directly_on_the_regex(self):
        for q in ("Are price and sales related?", "Is price related to sales?",
                  "How do price and sales relate?", "What is their relation?"):
            self.assertTrue(_CORRELATION_KEYWORD_RE.search(q.lower()), q)

    def test_unrelated_as_a_compound_word_is_not_a_false_positive(self):
        # \b enforces a boundary before "relat", so "unrelated" (no
        # boundary between "un" and "related") must not match -- this is
        # the same behavior the old, pre-C4.2.2c compiler-only _ASSOC had,
        # and the widening must not introduce a new false-positive class.
        self.assertFalse(_CORRELATION_KEYWORD_RE.search("this metric looks unrelated to churn"))

    def test_widening_is_a_pure_superset_of_relationship_stem(self):
        # Every previously-matching "relationship*" inflection must still
        # match -- relat\w* is a strict superset of relationship\w*.
        for q in ("What is the relationship between price and sales?",
                  "Is there a relationship here?"):
            self.assertTrue(_CORRELATION_KEYWORD_RE.search(q.lower()), q)


if __name__ == "__main__":
    unittest.main()
