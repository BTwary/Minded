"""Regression tests for the repository-wide EIG closure invariant."""
from pathlib import Path

from packages.analytics_core.src.intelligence.eig_optimizer import EIGOptimizer

ROOT = Path(__file__).resolve().parents[1]


def test_no_numeric_eig_assignments_in_production_source():
    import re
    pattern = re.compile(r"expected_information_gain\s*[:=]\s*[0-9]")
    offenders = []
    for base in (ROOT / "packages", ROOT / "apps", ROOT / "scripts"):
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                offenders.append(str(path))
    assert not offenders, offenders


def test_missing_likelihoods_never_become_eig():
    ranked = EIGOptimizer.rank_candidates(
        [{"code": "A", "expected_information_gain": 0.99, "estimated_cost": 1.0}],
        priors=[0.5, 0.5],
    )
    assert ranked[0]["computed_expected_information_gain"] == 0.0


def test_explicit_predictive_likelihoods_produce_eig():
    ranked = EIGOptimizer.rank_candidates(
        [{
            "code": "A",
            "target_hypothesis_index": 0,
            "likelihood_if_true": 0.9,
            "likelihood_if_false": 0.1,
            "estimated_cost": 1.0,
        }],
        priors=[0.5, 0.5],
    )
    assert ranked[0]["computed_expected_information_gain"] > 0.0


def test_zero_entropy_has_zero_eig():
    assert EIGOptimizer.compute_expected_entropy_reduction(
        [1.0, 0.0], 0, 0.99, 0.9, 0.1
    ) == 0.0
