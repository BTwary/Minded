"""Fix 4: controller-level order invariance regression tests.

The scientific result must not depend on valid arrival order.  These tests target
three concrete order-sensitive surfaces:
1. Bayesian/EIG hypothesis indexing;
2. experiment tie-breaking;
3. controller belief-update persistence targeting.
"""
from pathlib import Path
import ast
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.analytics_core.src.intelligence.eig_optimizer import EIGOptimizer
from packages.analytics_core.src.intelligence.experiment_synthesizer import CandidateExperiment
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis


def _hyp(code, identity, posterior):
    return PredictiveHypothesis(
        id=code,
        hypothesis_code=code,
        claim=f"Claim {code}",
        mechanism="controlled mechanism",
        predicted_observables_if_true=["x"],
        predicted_observables_if_false=["y"],
        falsification_criteria="not x",
        required_assumptions=[],
        prior_probability=0.5,
        posterior_probability=posterior,
        target_metric="revenue",
        target_dimension="region",
        canonical_identity=identity,
    )


def _exp(code, hyp):
    return CandidateExperiment(
        code=code,
        target_hypothesis_code=hyp,
        tool_name="duckdb_sql",
        query_sql="SELECT region, SUM(revenue) AS total FROM data_table GROUP BY region",
        description=code,
        aggregation_type="SUM",
        discriminating_power=0.75,
        estimated_cost=1.0,
        reliability_weight=1.0,
        decision_relevance=1.0,
        target_dimension="region",
        target_metric="revenue",
    )


def test_eig_is_invariant_to_hypothesis_arrival_order():
    a = _hyp("H-A", "aaa", 0.40)
    b = _hyp("H-B", "bbb", 0.35)
    c = _hyp("H-C", "ccc", 0.25)
    candidates = [_exp("EXP-A", "H-A"), _exp("EXP-B", "H-B"), _exp("EXP-C", "H-C")]

    rank1 = EIGOptimizer.score_candidate_experiments(candidates, [a, b, c])
    rank2 = EIGOptimizer.score_candidate_experiments(list(reversed(candidates)), [c, a, b])

    assert [x.experiment.code for x in rank1] == [x.experiment.code for x in rank2]
    scores1 = {x.experiment.code: x.utility_score for x in rank1}
    scores2 = {x.experiment.code: x.utility_score for x in rank2}
    assert scores1 == scores2


def test_equal_utility_uses_deterministic_scientific_tiebreaker():
    a = _hyp("H-A", "aaa", 0.5)
    b = _hyp("H-B", "bbb", 0.5)
    candidates1 = [_exp("EXP-Z", "H-B"), _exp("EXP-A", "H-A")]
    candidates2 = list(reversed(candidates1))

    selected1, _ = EIGOptimizer.select_next_experiment(candidates1, [a, b])
    selected2, _ = EIGOptimizer.select_next_experiment(candidates2, [b, a])
    assert selected1.code == selected2.code == "EXP-A"


def test_controller_does_not_attach_belief_update_to_arrival_order():
    controller = ROOT / "packages" / "analytics_core" / "src" / "runtime" / "controller.py"
    tree = ast.parse(controller.read_text())
    source = controller.read_text()

    assert "current_hyps[0].hypothesis_code" not in source
    assert "current_hyps[1]" not in source
    assert "target_idx = next(" in source

    # Verify the persistence logic explicitly looks up the hypothesis actually
    # targeted by the selected experiment rather than list position.
    assert "selected_exp.target_hypothesis_code" in source
    assert "hypothesis_id=f\"{investigation_id}_{current_hyps[target_idx].hypothesis_code}\"" in source
    assert tree is not None
