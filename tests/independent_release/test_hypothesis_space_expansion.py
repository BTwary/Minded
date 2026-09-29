import unittest
import ast
from pathlib import Path
from types import SimpleNamespace

from packages.analytics_core.src.runtime.state import InvestigationStateManager

ROOT = Path(__file__).resolve().parents[2]


def _h(code, prior, posterior, identity):
    return SimpleNamespace(
        id=code,
        hypothesis_code=code,
        canonical_identity=identity,
        claim=code,
        mechanism=code,
        prior_probability=prior,
        posterior_probability=posterior,
        belief_state="ACTIVE",
        target_metric="metric",
        target_dimension="segment",
        target_value=None,
        is_counter_hypothesis=False,
        source_evidence=[],
        parent_hypotheses=[],
        generated_reason="",
        prediction_ids=[],
        supporting_prediction_count=0,
        refuted_prediction_count=0,
        unresolved_prediction_count=0,
        supporting_evidence_ids=[],
        contradicting_evidence_ids=[],
        posterior_history=[],
        provenance={},
        consolidation_log=[],
    )


class TestHypothesisSpaceExpansion(unittest.TestCase):
    def test_late_hypothesis_expansion_preserves_existing_posterior_odds(self):
        state = InvestigationStateManager("INV-EXPAND")
        h1 = _h("H1", 0.6, 0.75, "A")
        h2 = _h("H2", 0.4, 0.25, "B")
        state.create_hypothesis(h1)
        state.create_hypothesis(h2)
    
        before_ratio = h1.posterior_probability / h2.posterior_probability
        before_prior_ratio = h1.prior_probability / h2.prior_probability
    
        h3 = _h("H3", 0.4, 0.4, "C")
        action, code, resolved = state.expand_hypothesis_space(h3)
    
        assert action == "created"
        assert code == "H3"
        assert resolved.posterior_probability == 0.05
        assert abs(sum(h.posterior_probability for h in state.get_active_hypotheses()) - 1.0) < 1e-12
        assert abs(sum(h.prior_probability for h in state.get_active_hypotheses()) - 1.0) < 1e-12
        assert abs(h1.posterior_probability / h2.posterior_probability - before_ratio) < 1e-12
        assert abs(h1.prior_probability / h2.prior_probability - before_prior_ratio) < 1e-12
        assert h1.posterior_probability < 0.75
        assert h2.posterior_probability < 0.25
    

    def test_duplicate_expansion_does_not_allocate_new_mass(self):
        state = InvestigationStateManager("INV-DUP")
        h1 = _h("H1", 1.0, 1.0, "A")
        state.create_hypothesis(h1)
        duplicate = _h("H2", 0.4, 0.4, "A")
        action, code, resolved = state.expand_hypothesis_space(duplicate)
        assert action == "merged"
        assert code == "H1"
        assert len(state.get_active_hypotheses()) == 1
        assert state.get_hypothesis("H1").posterior_probability == 1.0



    def test_late_discovery_cannot_promote_untested_generic_hypothesis(self):
        state = InvestigationStateManager("INV-GUARD")
        generic = _h("H1", 0.7, 0.654, "GENERIC")
        specific = _h("H2", 0.3, 0.346, "SPECIFIC")
        state.create_hypothesis(generic)
        state.create_hypothesis(specific)
        late = _h("H3", 0.4, 0.4, "LATE")

        state.expand_hypothesis_space(late)

        # The old bug could leave the generic untested hypothesis above the
        # positive-verdict threshold merely because the model gained/changed
        # a third hypothesis. Expansion must reduce existing mass uniformly.
        assert state.get_hypothesis("H1").posterior_probability < 0.654
        assert state.get_hypothesis("H1").posterior_probability < 0.65
        assert state.get_hypothesis("H3").posterior_probability <= 0.05

    def test_controller_no_longer_rewrites_priors_from_posteriors_on_emergence(self):
        source = (ROOT / "packages/analytics_core/src/runtime/controller.py").read_text()
        assert "state_mgr.expand_hypothesis_space(new_h)" in source
        assert "h.prior_probability = h.posterior_probability" not in source
        assert "total_mass = sum(h.posterior_probability for h in current_hyps)" not in source
