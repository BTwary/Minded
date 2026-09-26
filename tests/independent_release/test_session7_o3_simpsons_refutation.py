"""DEFECT-021 / O3: a rigorously-detected Simpson's reversal must falsify.

Before this fix `is_falsified` was False on every branch of the adversarial
attacker, so the controller's falsification handling could never fire.
"""
import numpy as np
import pandas as pd

from packages.analytics_core.src.intelligence.adversarial_attacker import AdversarialAttacker
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis


def _hyp(dim, metric, code="HYP-01"):
    return PredictiveHypothesis(
        id=code, hypothesis_code=code, claim="c", mechanism="m",
        predicted_observables_if_true=[], predicted_observables_if_false=[],
        falsification_criteria="f", required_assumptions=[],
        prior_probability=0.5, posterior_probability=0.9,
        target_metric=metric, target_dimension=dim, dimension_resolution_status="RESOLVED",
    )


def _simpsons_df():
    # Drug C looks best marginally only because it is given mostly to mild cases;
    # within every severity stratum C is no better than A/B (sign reverses).
    g = np.random.default_rng(3)
    spec = {("A", "mild"): (300, 90), ("A", "severe"): (300, 50), ("B", "mild"): (300, 88),
            ("B", "severe"): (300, 48), ("C", "mild"): (40, 92), ("C", "severe"): (560, 52)}
    rows = [(d, s, v) for (d, s), (n, mu) in spec.items() for v in g.normal(mu, 3, n)]
    return pd.DataFrame(rows, columns=["drug", "severity", "recovery"])


def test_genuine_simpsons_reversal_is_refuted_and_falsified():
    r = AdversarialAttacker.design_attack(_hyp("drug", "recovery"), [_hyp("drug", "recovery", "HYP-02")], _simpsons_df())
    assert r.simpsons_paradox_detected is True
    assert r.attack_status == "REFUTED"
    assert r.is_falsified is True
    assert r.details["simpsons_reversal"]["marginal_difference"] * r.details["simpsons_reversal"]["adjusted_difference"] < 0


def test_no_reversal_is_never_falsified():
    g = np.random.default_rng(11)
    df = pd.DataFrame({"drug": g.choice(["A", "B"], 400), "severity": g.choice(["mild", "severe"], 400),
                       "recovery": g.normal(100, 15, 400)})
    df.loc[df.drug == "B", "recovery"] += 40  # large, consistent effect in every stratum
    r = AdversarialAttacker.design_attack(_hyp("drug", "recovery"), [_hyp("drug", "recovery", "HYP-02")], df)
    assert r.simpsons_paradox_detected is False
    assert r.is_falsified is False
    assert r.attack_status != "REFUTED"


def test_softer_findings_still_only_weaken():
    # Low-dispersion dataset: weakened, but nothing demonstrates the claimed direction is wrong.
    g = np.random.default_rng(5)
    df = pd.DataFrame({"drug": g.choice(["A", "B"], 300), "recovery": g.normal(100, 0.5, 300)})
    r = AdversarialAttacker.design_attack(_hyp("drug", "recovery"), [_hyp("drug", "recovery", "HYP-02")], df)
    assert r.attack_status == "WEAKENED"
    assert r.is_falsified is False


# ---------------------------------------------------------------------------
# Analyst-facing behaviour: the refutation must reach the human as a finding.
# ---------------------------------------------------------------------------
import unittest

from packages.analytics_core.src.engines.provenance import ProvenanceEngine
from packages.analytics_core.src.engines.refutation_narrative import describe_simpsons_refutation
from packages.analytics_core.src.engines.verdict import VerdictEngine

_REFUTATION = {
    "groups": ["A", "C"], "marginal_difference": 15.36, "adjusted_difference": -1.966,
    "strata_used": 2, "secondary_dimension": "severity", "primary_dimension": "drug", "metric": "recovery",
}


class TestRefutationNarrative(unittest.TestCase):
    def test_headline_and_finding_are_specific_and_correctly_signed(self):
        headline, finding, steps = describe_simpsons_refutation(_REFUTATION)
        self.assertIn("recovery", headline); self.assertIn("drug", headline); self.assertIn("severity", headline)
        self.assertIn("15.36 higher", finding)      # pooled: A above C
        self.assertIn("1.966 lower", finding)       # within severity: A below C
        self.assertIn("does not by itself show", finding)   # no over-claiming of the reversed direction
        self.assertGreaterEqual(len(steps), 3)

    def test_direct_answer_is_not_the_generic_inconclusive_message(self):
        ans, finding = ProvenanceEngine.formulate_direct_answer(
            intent=None, semantic=None, leading_hypothesis_code="HYP-01", leading_claim="c", posterior=0.5,
            verdict_type="REFUTED", experiments_summary=[{"code": "E"}], adversarial_refutation=_REFUTATION)
        self.assertNotIn("unable to find sufficient evidence", ans.lower())
        self.assertIn("reverses", ans)
        self.assertIn("Suggested next steps", finding)


class TestVerdictEngineRefutation(unittest.TestCase):
    _base = dict(leading_hypothesis_code="HYP-01", leading_hypothesis_posterior=0.99, all_verifications_passed=True,
                 adversarial_attack_survived=False, causal_identifiable=True)

    def test_refutation_yields_refuted_with_zero_confidence_and_analyst_sections(self):
        v = VerdictEngine.evaluate_verdict(**self._base, adversarial_refutation=_REFUTATION)
        self.assertEqual(v.verdict_type, "REFUTED")
        self.assertEqual(v.confidence_score, 0.0)
        self.assertFalse(v.counter_hypothesis_refuted)
        self.assertIn("reverses", v.direct_answer)
        self.assertTrue(any("stratified" in s for s in v.what_to_test_next))
        self.assertTrue(any("Cannot claim" in s for s in v.what_we_cannot_claim))
        self.assertEqual(v.what_supports_it, [])

    def test_a_high_posterior_or_method_ceiling_cannot_mask_a_refutation(self):
        v = VerdictEngine.evaluate_verdict(**self._base, max_verdict_tier="STATISTICALLY_SIGNIFICANT",
                                           adversarial_refutation=_REFUTATION)
        self.assertEqual(v.verdict_type, "REFUTED")
        self.assertNotIn("restricted", v.direct_answer.lower())

    def test_no_refutation_leaves_existing_behaviour_untouched(self):
        v = VerdictEngine.evaluate_verdict(**{**self._base, "adversarial_attack_survived": True})
        self.assertEqual(v.verdict_type, "DIAGNOSED")
        self.assertGreater(v.confidence_score, 0.9)

    def test_a_leading_null_hypothesis_is_not_marked_refuted_by_a_reversal(self):
        v = VerdictEngine.evaluate_verdict(**self._base, leading_hypothesis_is_counter=True, adversarial_refutation=_REFUTATION)
        self.assertNotEqual(v.verdict_type, "REFUTED")


# ---------------------------------------------------------------------------
# End to end through the real controller.
# ---------------------------------------------------------------------------
from tests.independent_release.test_defect_007_controller_closure import (  # noqa: E402
    BaseIsolatedControllerTest as _ControllerBase, Investigation, Hypothesis, InvestigationController,
    InMemoryDatasetProvider, gen_uuid,
)


class TestEndToEndAnalystExperience(_ControllerBase):
    def _run(self, df, q="Why does recovery differ across drug?"):
        inv = Investigation(id=f"INV-O3-{gen_uuid()[:8]}", project_id=self.proj.id, user_id=self.user.id, question=q, status="QUEUED")
        self.db.add(inv); self.db.commit()
        ctl = InvestigationController(session_factory=self.SessionFactory, dataset_provider=InMemoryDatasetProvider({"trial": df}))
        self.assertTrue(ctl.execute_investigation(investigation_id=inv.id, worker_id="w-o3"))
        self.db.expire_all()
        return (self.db.query(Investigation).filter(Investigation.id == inv.id).one(),
                {h.hypothesis_code: h for h in self.db.query(Hypothesis).filter(Hypothesis.investigation_id == inv.id).all()})

    def test_simpsons_reversal_is_reported_to_the_analyst_as_a_refutation(self):
        inv, hyps = self._run(_simpsons_df())
        self.assertEqual(inv.verdict_type, "REFUTED")
        self.assertEqual(inv.confidence_score, 0.0)
        self.assertIn("reverses", inv.direct_answer)
        self.assertIn("severity", inv.direct_answer)
        self.assertIn("Simpson", inv.main_finding)
        self.assertIn("Suggested next steps", inv.main_finding)
        self.assertNotIn("unable to find sufficient evidence", (inv.direct_answer or "").lower())
        self.assertEqual(hyps["HYP-01"].belief_state, "REFUTED")   # persisted, not just in memory

    def test_a_genuinely_supported_finding_is_still_diagnosed_not_refuted(self):
        # Known-positive scenario (same data as the DEFECT-007 replanning trace):
        # every adversarial attack SURVIVES here, so the refutation path must not fire.
        df = pd.DataFrame({
            "datacenter_region": ["us-east", "us-west", "eu-central", "ap-south"] * 100,
            "tier": (["Enterprise", "SMB"] * 50) + (["SMB", "SMB"] * 150),
            "cost_metric": [500.0 if (i % 4 == 0 and i < 200) else 100.0 for i in range(400)],
        })
        inv, hyps = self._run(df, q="Why did cost_metric surge across datacenter_region?")
        self.assertEqual(inv.verdict_type, "DIAGNOSED")
        self.assertNotEqual(hyps["HYP-01"].belief_state, "REFUTED")
        self.assertNotIn("reverses", inv.direct_answer)
