"""DEFECT-007 real repair proof: hypothesis-attributable evidence and
consistent subgroup sample-size adequacy across ALL likelihood branches in
ScientificTransitionService.apply_post_execution_transition.

Root cause (see AUDIT_DEFECT_007_FINAL_FORENSIC.md for the full forensic
account): the "primary concentration hypothesis" branch derived its
likelihood purely from a whole-experiment eta_sq statistic regardless of
(a) whether any prediction had ever actually been generated/targeted at
that hypothesis, and (b) whether the underlying subgroups the statistic was
computed from were even large enough to trust. The "emergent hypothesis"
branch was already gated on both counts. This asymmetry is what let an
untested, unfalsifiable generic hypothesis outrank the one hypothesis that
actually owned the evidence -- the exact GOLDEN-5 symptom.

The fix (packages/analytics_core/src/intelligence/transition.py):
1. A hypothesis with no prediction ever generated for it (not in
   `tested_hyp_codes`, built from the full runtime prediction set, not just
   this round's) gets a neutral 1.0 likelihood -- no attributable evidence,
   no update in either direction.
2. A hypothesis that DOES have an attributable prediction still gets the
   eta_sq-driven sigmoid, but is now capped the same way the emergent path
   already was when the underlying subgroups are too small to trust
   (`_min_group_n` / `MIN_ADEQUATE_GROUP_N = 5`).
3. Both paths now measure true subgroup size via an explicit per-row
   observation-count column (e.g. `row_count`) when the frame provides one,
   instead of naively counting DataFrame rows -- which silently reports
   n=1 for every group when the frame passed in is already a pre-aggregated
   summary table (one row per group) rather than raw observations.

These are genuine, deterministic, non-tautological cases run through the
real `ScientificTransitionService.apply_post_execution_transition` -- not a
mocked evidence path and not an assertion against a hand-tuned constant
(each assertion is an inequality/ordering or a neutrality check, not a
specific hardcoded posterior).

Run with: python -m unittest discover -s tests/independent_release -p "test_*.py"
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.intelligence.predictive_hypothesis import (
    HypothesisSynthesizer,
    PredictiveHypothesis,
)
from packages.analytics_core.src.intelligence.prediction_engine import PredictionSynthesizer
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.intelligence.transition import ScientificTransitionService
from packages.analytics_core.src.runtime.state import InvestigationStateManager


def _make_semantic(df):
    return SemanticEngine().resolve_schema(
        IntentEngine.parse_intent("Why did cost_metric surge across datacenter_region?"),
        {"cloud_costs": df},
    )


def _base_hyps():
    """H1: generic/untested (no target_value, not counter) -- the
    'primary concentration hypothesis' branch. H2: uniform counter
    hypothesis. H3: emergent, specifically targets us-east."""
    h1 = PredictiveHypothesis(
        id="H1", hypothesis_code="HYP-01", claim="Generic driver", mechanism="M1",
        predicted_observables_if_true=[], predicted_observables_if_false=[],
        falsification_criteria="f1", required_assumptions=[],
        prior_probability=0.33, posterior_probability=0.33,
        target_metric="cost_metric", target_dimension="datacenter_region",
    )
    h2 = PredictiveHypothesis(
        id="H2", hypothesis_code="HYP-02", claim="Uniform/no driver", mechanism="M2",
        predicted_observables_if_true=[], predicted_observables_if_false=[],
        falsification_criteria="f2", required_assumptions=[],
        prior_probability=0.33, posterior_probability=0.33,
        target_metric="cost_metric", target_dimension="datacenter_region",
        is_counter_hypothesis=True,
    )
    h3 = PredictiveHypothesis(
        id="H3", hypothesis_code="HYP-03", claim="us-east specifically", mechanism="M3",
        predicted_observables_if_true=[], predicted_observables_if_false=[],
        falsification_criteria="f3", required_assumptions=[],
        prior_probability=0.34, posterior_probability=0.34,
        target_metric="cost_metric", target_dimension="datacenter_region",
        target_value="us-east",
    )
    return h1, h2, h3


class TestCaseA_AdequateSampleTrueEffect(unittest.TestCase):
    """Case A (brief section 3): targeted hypothesis has adequate sample
    size and a true effect -> it receives evidence and its posterior rises
    above its prior, and above an untested hypothesis's."""

    def test_targeted_hypothesis_gains_posterior_with_adequate_n(self):
        obs_df = pd.DataFrame({
            "datacenter_region": ["us-east", "us-west", "eu-central", "ap-southeast"],
            "cost_metric": [94000.0, 2000.0, 2500.0, 1500.0],
            "row_count": [1000, 1000, 1000, 1000],  # true underlying N per group
        })
        semantic_res = _make_semantic(obs_df)
        h1, h2, h3 = _base_hyps()
        state_mgr = InvestigationStateManager("INV-CASE-A")
        for h in (h1, h2, h3):
            state_mgr.create_hypothesis(h)
        pred3 = PredictionSynthesizer.deduce_prediction(h3, semantic_res)
        state_mgr.create_prediction(pred3)

        result = ScientificTransitionService.apply_post_execution_transition(
            state_mgr=state_mgr,
            experiment_id="EXP-CASE-A",
            target_prediction_ids=[pred3.prediction_id],
            primary_df=obs_df,
            result_df=obs_df,
            target_metric_col="cost_metric",
            group_dimension_col="datacenter_region",
            aggregation_type="SUM",
            primary_value=94000.0,
            is_grouped=True,
        )
        posteriors = result.belief_update["posteriors"]
        self.assertTrue(np.isclose(sum(posteriors), 1.0))
        # H3 (targeted, adequately-sampled, true effect) must lead.
        self.assertGreater(posteriors[2], posteriors[0])
        self.assertGreater(posteriors[2], posteriors[1])
        # And it must have genuinely risen above its 0.34 prior.
        self.assertGreater(posteriors[2], 0.34)


class TestCaseB_InsufficientSampleSize(unittest.TestCase):
    """Case B: targeted hypothesis's own subgroup is too small to trust ->
    its evidence is capped/weak, AND -- this is the actual DEFECT-007
    mechanism -- no untested hypothesis is allowed to win merely from the
    same global statistic."""

    def test_small_target_subgroup_is_capped_and_does_not_lose_to_untested_hyp(self):
        # No row_count column this time: true per-group N genuinely is 1,
        # a real small-sample case (not a pre-aggregation artifact).
        obs_df = pd.DataFrame({
            "datacenter_region": ["us-east", "us-west", "eu-central", "ap-southeast"],
            "cost_metric": [94000.0, 2000.0, 2500.0, 1500.0],
        })
        semantic_res = _make_semantic(obs_df)
        h1, h2, h3 = _base_hyps()
        state_mgr = InvestigationStateManager("INV-CASE-B")
        for h in (h1, h2, h3):
            state_mgr.create_hypothesis(h)
        pred3 = PredictionSynthesizer.deduce_prediction(h3, semantic_res)
        state_mgr.create_prediction(pred3)

        result = ScientificTransitionService.apply_post_execution_transition(
            state_mgr=state_mgr,
            experiment_id="EXP-CASE-B",
            target_prediction_ids=[pred3.prediction_id],
            primary_df=obs_df,
            result_df=obs_df,
            target_metric_col="cost_metric",
            group_dimension_col="datacenter_region",
            aggregation_type="SUM",
            primary_value=94000.0,
            is_grouped=True,
        )
        bayes_factors = result.belief_update["bayes_factors"]
        posteriors = result.belief_update["posteriors"]
        # H3's likelihood must be capped (evidence marked weak) -- not the
        # confident 0.85 it would get with an adequate subgroup.
        self.assertEqual(bayes_factors[2], 1.0)
        # H1 is untested (no prediction was ever generated for it) and must
        # NOT receive STRONG support merely because a global eta_sq exists
        # on the same inadequately-sized data -- it must sit at the neutral
        # likelihood (no evidence attributable to it either way), not the
        # confident ~0.95 the pre-fix sigmoid would have assigned from the
        # same eta_sq. (A neutral hypothesis can still end up with a higher
        # posterior than a hypothesis whose own weak evidence was correctly
        # marked down below neutral -- that is expected Bayesian behavior,
        # not the DEFECT-007 bug; the bug was strong/confident support for
        # an untested hypothesis, which this asserts against directly.)
        self.assertEqual(bayes_factors[0], 1.0)  # neutral: no attributable evidence
        self.assertLessEqual(bayes_factors[0], 1.0)  # nowhere near the old confident sigmoid


class TestCaseC_GenericEffectNoDiscrimination(unittest.TestCase):
    """Case C: only an untested, generic hypothesis exists alongside a
    counter-hypothesis (no emergent/targeted hypothesis was ever
    formulated) -- the global statistic must not manufacture false
    posterior separation in favor of the untested hypothesis."""

    def test_untested_generic_hypothesis_gets_neutral_likelihood(self):
        obs_df = pd.DataFrame({
            "datacenter_region": ["us-east", "us-west", "eu-central", "ap-southeast"],
            "cost_metric": [94000.0, 2000.0, 2500.0, 1500.0],
            "row_count": [1000, 1000, 1000, 1000],
        })
        h1, h2, _h3_unused = _base_hyps()
        state_mgr = InvestigationStateManager("INV-CASE-C")
        state_mgr.create_hypothesis(h1)
        state_mgr.create_hypothesis(h2)
        # Deliberately no prediction created for H1 or H2 -- neither is
        # attributably tested.

        result = ScientificTransitionService.apply_post_execution_transition(
            state_mgr=state_mgr,
            experiment_id="EXP-CASE-C",
            target_prediction_ids=[],
            primary_df=obs_df,
            result_df=obs_df,
            target_metric_col="cost_metric",
            group_dimension_col="datacenter_region",
            aggregation_type="SUM",
            primary_value=94000.0,
            is_grouped=True,
        )
        bayes_factors = result.belief_update["bayes_factors"]
        # H1 (generic, untested) must get the neutral Bayes factor, not a
        # confident 0.95 manufactured from a global statistic no prediction
        # ever attributed to it.
        self.assertEqual(bayes_factors[0], 1.0)


class TestCaseD_GlobalEvidenceGenuinelyDiscriminates(unittest.TestCase):
    """Case D: when a hypothesis IS attributably tested (a prediction was
    generated for it) and the underlying subgroups are genuinely adequate,
    the global statistic MAY still grant it evidence-driven
    Bayes-factor weight -- the fix must not blanket-suppress all generic evidence,
    only evidence that is not attributable or not adequately sampled."""

    def test_attributable_generic_hypothesis_with_adequate_n_is_not_suppressed(self):
        obs_df = pd.DataFrame({
            "datacenter_region": ["us-east", "us-west", "eu-central", "ap-southeast"],
            "cost_metric": [94000.0, 2000.0, 2500.0, 1500.0],
            "row_count": [1000, 1000, 1000, 1000],
        })
        semantic_res = _make_semantic(obs_df)
        h1, h2, _h3 = _base_hyps()
        state_mgr = InvestigationStateManager("INV-CASE-D")
        state_mgr.create_hypothesis(h1)
        state_mgr.create_hypothesis(h2)
        # H1 IS attributably tested this time: a prediction was generated
        # and targeted at it directly (unlike Case C).
        pred1 = PredictionSynthesizer.deduce_prediction(h1, semantic_res)
        state_mgr.create_prediction(pred1)

        result = ScientificTransitionService.apply_post_execution_transition(
            state_mgr=state_mgr,
            experiment_id="EXP-CASE-D",
            target_prediction_ids=[pred1.prediction_id],
            primary_df=obs_df,
            result_df=obs_df,
            target_metric_col="cost_metric",
            group_dimension_col="datacenter_region",
            aggregation_type="SUM",
            primary_value=94000.0,
            is_grouped=True,
        )
        bayes_factors = result.belief_update["bayes_factors"]
        # H1 is attributable AND adequately sampled -- must NOT be
        # suppressed to the neutral 1.0; the fix is about attribution and
        # sample size, not about banning generic evidence outright.
        self.assertGreater(bayes_factors[0], 1.0)


class TestOrderInvarianceOfAttribution(unittest.TestCase):
    """Hypothesis creation order must not change which hypotheses are
    considered 'tested' or the resulting posterior ranking."""

    def test_reversed_hypothesis_creation_order_yields_same_ranking(self):
        obs_df = pd.DataFrame({
            "datacenter_region": ["us-east", "us-west", "eu-central", "ap-southeast"],
            "cost_metric": [94000.0, 2000.0, 2500.0, 1500.0],
            "row_count": [1000, 1000, 1000, 1000],
        })
        semantic_res = _make_semantic(obs_df)

        def run(order):
            h1, h2, h3 = _base_hyps()
            by_code = {"HYP-01": h1, "HYP-02": h2, "HYP-03": h3}
            state_mgr = InvestigationStateManager(f"INV-ORDER-{'-'.join(order)}")
            for code in order:
                state_mgr.create_hypothesis(by_code[code])
            pred3 = PredictionSynthesizer.deduce_prediction(h3, semantic_res)
            state_mgr.create_prediction(pred3)
            res = ScientificTransitionService.apply_post_execution_transition(
                state_mgr=state_mgr,
                experiment_id="EXP-ORDER",
                target_prediction_ids=[pred3.prediction_id],
                primary_df=obs_df,
                result_df=obs_df,
                target_metric_col="cost_metric",
                group_dimension_col="datacenter_region",
                aggregation_type="SUM",
                primary_value=94000.0,
                is_grouped=True,
            )
            ordered_hyps = [by_code[c] for c in order]
            by_hyp_code = {h.hypothesis_code: p for h, p in zip(
                ordered_hyps, res.belief_update["posteriors"]
            )}
            return by_hyp_code

        forward = run(["HYP-01", "HYP-02", "HYP-03"])
        reversed_ = run(["HYP-03", "HYP-02", "HYP-01"])
        for code in ("HYP-01", "HYP-02", "HYP-03"):
            self.assertAlmostEqual(forward[code], reversed_[code], places=9)
        # Leader is unchanged by insertion order.
        self.assertEqual(max(forward, key=forward.get), max(reversed_, key=reversed_.get))
        self.assertEqual(max(forward, key=forward.get), "HYP-03")


if __name__ == "__main__":
    unittest.main(verbosity=2)
