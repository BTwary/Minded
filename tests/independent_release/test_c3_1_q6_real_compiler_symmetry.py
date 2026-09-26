"""v20-C3.1 regression suite: Q6 symmetric-association test through the
real question/intention compilation path.

C3's Q6 regression (test_c3_canonical_experiment_contract.py::
TestQ6SymmetricAssociation) proves the synthesizer follows whatever
(target, predictor) pair ``decision.estimand`` carries -- but it builds
that decision by hand (via ``MethodSelectionEngine._estimand`` plus a
manually-set ``predictor_columns``), which does not exercise the actual
NL -> intent -> semantic -> decision compilation pipeline a real question
goes through. C3.1 finding 3 requires exercising that real path with two
symmetric natural-language phrasings and either confirming they compile to
the same canonical bindings, or documenting the exact limitation if they
don't.

The real compilation path is:
    IntentEngine.parse_intent(question)
        -> InvestigationIntent
    SemanticEngine.resolve_schema_static(intent, {table: df})
        -> SemanticResolution
    MethodSelectionEngine.decide(intent=..., semantic=..., question=..., primary_df=df)
        -> MethodSelectionDecision (carries the canonical estimand)
    ExperimentSynthesizer.synthesize_candidate_experiments(..., decision=...)
        -> the actual CandidateExperiment the decision is consumed into

Update (Q6.1, v20-C3.1a follow-up): this suite originally documented a real
compiler limitation for "Is price associated with sales?" / "Are sales and
price associated?" (IntentEngine.parse_intent's CORRELATION keyword list
had "association" but not "associated"). That gap is now fixed --
TestQ6RealCompilerPathFixedByQ6_1 below asserts the corrected positive
behavior instead of the former limitation.

Run with: python -m unittest tests/independent_release/test_c3_1_q6_real_compiler_symmetry.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.method_selection import MethodFamily, MethodSelectionEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.intelligence.experiment_synthesizer import ExperimentSynthesizer
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis


def _dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "price": [10.0, 12.0, 9.0, 15.0, 11.0, 13.0, 14.0, 10.5, 9.5, 12.5] * 3,
        "sales": [100, 120, 90, 150, 110, 130, 140, 105, 95, 125] * 3,
    })


def _compile(question: str, df: pd.DataFrame):
    """Run the real question -> intent -> semantic -> decision pipeline."""
    intent = IntentEngine.parse_intent(question, available_columns=list(df.columns))
    semantic = SemanticEngine.resolve_schema_static(intent, {"data_table": df})
    decision = MethodSelectionEngine.decide(
        intent=intent, semantic=semantic, question=question, primary_df=df,
    )
    return intent, semantic, decision


def _hypothesis(secondary_metric: str, target_metric: str = "price") -> PredictiveHypothesis:
    return PredictiveHypothesis(
        id="HYP-01", hypothesis_code="HYP-01", claim="test claim", mechanism="test mechanism",
        predicted_observables_if_true=["obs_true"], predicted_observables_if_false=["obs_false"],
        falsification_criteria="none", required_assumptions=[], prior_probability=0.5,
        posterior_probability=0.5, target_metric=target_metric, secondary_metric=secondary_metric,
        claim_type="ASSOCIATION",
    )


class TestQ6RealCompilerPathFixedByQ6_1(unittest.TestCase):
    """C3.1 finding 3, literal phrasing from the audit: "Is price associated
    with sales?" / "Are sales and price associated?".

    FORMERLY A DOCUMENTED LIMITATION, FIXED BY Q6.1 (v20-C3.1a follow-up):
    as of C3.1, IntentEngine.parse_intent's CORRELATION keyword check was a
    literal-substring list containing "association" but not its "-ed"
    inflection "associated" (see packages/analytics_core/src/engines/
    intent.py), so both of these exact questions fell through every
    classifier branch to intent_type == "GENERAL" and never reached the
    CORRELATION method family or the (target, predictor) binding step at
    all.

    Q6.1 replaced that literal list with `_CORRELATION_KEYWORD_RE`, a
    regex that matches single-root words by stem (`associat\\w*`,
    `correlat\\w*`, etc. -- the same inflection-robust approach already
    used elsewhere in this codebase, e.g. `_ASSOC` in
    intelligence/universal_question_planner.py). This test now asserts
    the corrected, positive behavior: both phrasings reach CORRELATION,
    and compile to identical bindings regardless of word order -- mirroring
    TestQ6RealCompilerPathPositiveSymmetry below, which already covered
    "correlated" phrasing.
    """

    def test_both_phrasings_are_classified_as_correlation(self):
        df = _dataframe()
        intent_a, _, decision_a = _compile("Is price associated with sales?", df)
        intent_b, _, decision_b = _compile("Are sales and price associated?", df)

        self.assertEqual(intent_a.intent_type, "CORRELATION")
        self.assertEqual(intent_b.intent_type, "CORRELATION")
        self.assertEqual(decision_a.method_family, MethodFamily.CORRELATION)
        self.assertEqual(decision_b.method_family, MethodFamily.CORRELATION)

    def test_both_phrasings_compile_to_identical_bindings(self):
        # Word order must not, on its own, change which (target, predictor)
        # pair the real pipeline resolves.
        df = _dataframe()
        _, _, decision_a = _compile("Is price associated with sales?", df)
        _, _, decision_b = _compile("Are sales and price associated?", df)

        self.assertEqual(decision_a.method_family, decision_b.method_family)
        self.assertEqual(decision_a.estimand.target_column, decision_b.estimand.target_column)
        self.assertEqual(decision_a.estimand.predictor_columns, decision_b.estimand.predictor_columns)

    def test_downstream_experiment_matches_the_correlated_phrasing_case(self):
        # Full pipeline through to the actual generated CandidateExperiment,
        # same shape as TestQ6RealCompilerPathPositiveSymmetry below --
        # proves the fix reaches all the way through, not just intent
        # classification.
        df = _dataframe()
        _, semantic_a, decision_a = _compile("Is price associated with sales?", df)
        _, semantic_b, decision_b = _compile("Are sales and price associated?", df)

        h_a = _hypothesis(secondary_metric=semantic_a.secondary_metric_col or "")
        h_b = _hypothesis(secondary_metric=semantic_b.secondary_metric_col or "")
        candidates_a = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h_a], semantic_a, decision=decision_a,
        )
        candidates_b = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h_b], semantic_b, decision=decision_b,
        )
        self.assertEqual(len(candidates_a), 1)
        self.assertEqual(len(candidates_b), 1)
        self.assertEqual(candidates_a[0].code, "EXP-CORR")
        self.assertEqual(candidates_b[0].code, "EXP-CORR")
        self.assertEqual(candidates_a[0].target_metric, candidates_b[0].target_metric)
        self.assertEqual(sorted(candidates_a[0].metrics), sorted(candidates_b[0].metrics))


class TestQ6RealCompilerPathPositiveSymmetry(unittest.TestCase):
    """Positive-path coverage for C3.1 finding 3: a symmetric-association
    phrasing that the current IntentEngine keyword list DOES recognize
    ("correlated", which is in the CORRELATION list) exercises the full
    real compilation path end to end, including the downstream experiment,
    and proves word-order reversal does not change the canonical binding.
    """

    def test_price_correlated_sales_and_sales_correlated_price_bind_identically(self):
        df = _dataframe()
        q_a = "Is price correlated with sales?"
        q_b = "Are sales and price correlated?"

        intent_a, semantic_a, decision_a = _compile(q_a, df)
        intent_b, semantic_b, decision_b = _compile(q_b, df)

        # Both phrasings are actually recognized as CORRELATION intent by
        # the real compiler (unlike the "associated" wording above) --
        # this is what makes this pair a meaningful exercise of the
        # compilation path rather than a coincidental match on GENERAL.
        self.assertEqual(intent_a.intent_type, "CORRELATION")
        self.assertEqual(intent_b.intent_type, "CORRELATION")
        self.assertEqual(decision_a.method_family, MethodFamily.CORRELATION)
        self.assertEqual(decision_b.method_family, MethodFamily.CORRELATION)

        # The canonical (target, predictor) pair the real pipeline resolves
        # is identical regardless of which variable is named first.
        self.assertEqual(decision_a.estimand.target_column, decision_b.estimand.target_column)
        self.assertEqual(decision_a.estimand.predictor_columns, decision_b.estimand.predictor_columns)

        # And the downstream experiment the synthesizer actually produces
        # from each decision consumes that same pair -- not merely the
        # estimand object, but the real generated CandidateExperiment.
        h_a = _hypothesis(secondary_metric=semantic_a.secondary_metric_col or "")
        h_b = _hypothesis(secondary_metric=semantic_b.secondary_metric_col or "")
        candidates_a = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h_a], semantic_a, decision=decision_a,
        )
        candidates_b = ExperimentSynthesizer.synthesize_candidate_experiments(
            [h_b], semantic_b, decision=decision_b,
        )
        self.assertEqual(len(candidates_a), 1)
        self.assertEqual(len(candidates_b), 1)
        self.assertEqual(candidates_a[0].target_metric, candidates_b[0].target_metric)
        self.assertEqual(sorted(candidates_a[0].metrics), sorted(candidates_b[0].metrics))
        self.assertEqual(candidates_a[0].aggregation_type, "CORRELATION")
        self.assertEqual(candidates_b[0].aggregation_type, "CORRELATION")


if __name__ == "__main__":
    unittest.main()
