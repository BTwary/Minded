"""Regression tests: semantic roles must never depend on physical column order."""
import pandas as pd

from packages.analytics_core.src.semantic.world_model import SemanticWorldModelBuilder
from packages.analytics_core.src.semantic.metric_semantics import MetricSemanticsResolver
from packages.analytics_core.src.intelligence.adversarial_attacker import AdversarialAttacker


def test_world_model_does_not_choose_first_declared_key():
    df = pd.DataFrame({"first_key": [1, 2, 3], "second_key": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
    df.attrs["candidate_keys"] = ["first_key", "second_key"]
    wm = SemanticWorldModelBuilder().build_world_model({"t": df})
    assert wm.table_grains["t"] == "ambiguous_declared_key"


def test_ratio_pair_requires_unique_numerator_and_denominator():
    df = pd.DataFrame({
        "conversion_rate": [0.1, 0.2],
        "conversions": [1, 2],
        "orders": [1, 2],
        "sessions": [10, 10],
        "visits": [10, 20],
    })
    assert MetricSemanticsResolver._find_numerator_denominator_pair("conversion_rate", df) == (None, None)


def test_weight_column_requires_unique_candidate():
    df = pd.DataFrame({"rate": [0.1, 0.2], "sessions": [10, 20], "visits": [20, 30]})
    assert MetricSemanticsResolver._find_denominator_pair("rate", df) is None


def test_adversarial_secondary_dimension_tie_is_not_order_selected():
    df = pd.DataFrame({
        "treatment": ["A", "B"] * 10,
        "metric": list(range(20)),
        "region": ["x", "y"] * 10,
        "channel": ["x", "y"] * 10,
    })
    h = type("H", (), {"target_dimension": "treatment", "target_metric": "metric", "hypothesis_code": "H1"})()
    result = AdversarialAttacker.design_attack(h, [], df)
    assert result.details.get("secondary_dimension") is None or result.simpsons_paradox_detected is False
