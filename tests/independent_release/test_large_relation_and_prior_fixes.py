"""Regression tests for the real-dataset (120k-row) fixes of 2026-10-07."""
import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.execution_provider import (
    relation_digest, relation_digests_equivalent,
)
from packages.analytics_core.src.intelligence.multiverse_engine import MultiverseEngine
from packages.analytics_core.src.intelligence.predictive_hypothesis import HypothesisSynthesizer


class _H:
    def __init__(self):
        self.prior_probability = 1.0
        self.posterior_probability = 1.0


def test_uniform_priors_are_normalized():
    hs = [_H(), _H()]
    HypothesisSynthesizer._normalize_uniform_priors(hs)
    assert [h.prior_probability for h in hs] == [0.5, 0.5]
    assert [h.posterior_probability for h in hs] == [0.5, 0.5]


def _frame(seed=0, n=5000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "region": rng.choice(["W", "E", "S"], n),
        "revenue": rng.gamma(2.0, 100.0, n),
    })


def test_digest_is_order_independent_and_matches_across_copies():
    df = _frame()
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    ok, delta, reason = relation_digests_equivalent(relation_digest(df), relation_digest(shuffled))
    assert ok, reason


def test_digest_detects_row_value_and_category_changes():
    df = _frame()
    base = relation_digest(df)
    assert not relation_digests_equivalent(base, relation_digest(df.iloc[:-1]))[0]
    changed = df.copy()
    changed.loc[0, "revenue"] += 5.0
    assert not relation_digests_equivalent(base, relation_digest(changed))[0]
    moved = df.copy()
    moved.loc[0, "region"] = "ZZ"
    assert not relation_digests_equivalent(base, relation_digest(moved))[0]


def test_multiverse_total_question_not_flagged_fragile_by_trimming():
    # West leads on SUM under every trim; East leads per row. Pinning SUM must not
    # cross aggregations, and the same top group must stay concordant.
    rng = np.random.default_rng(3)
    west = pd.DataFrame({"region": "W", "revenue": rng.gamma(2.0, 100.0, 3000)})
    east = pd.DataFrame({"region": "E", "revenue": rng.gamma(2.0, 104.0, 2500)})
    low = pd.DataFrame({"region": "C", "revenue": rng.gamma(2.0, 100.0, 800)})
    df = pd.concat([west, east, low], ignore_index=True)
    rep = MultiverseEngine.evaluate_specification_curve(df, "region", "revenue", aggregations=["SUM"])
    assert rep.total_specifications == 3
    assert rep.robustness_pct == 100.0


def test_evidence_dataset_label_uses_fingerprint_not_literal_one():
    from apps.api.src.services.analysis_service import _evidence_dataset_label

    class _E:
        arguments_json = {"source_dataset_fingerprints": {"orders": "a04025a2a5005b44"}}

    assert _evidence_dataset_label(_E()) == "orders@a04025a2"
    assert _evidence_dataset_label(None) == "unknown"


def test_single_dataset_scope_is_recorded_without_explicit_selection():
    from packages.analytics_core.src.runtime.cross_dataset_review import build_cross_dataset_review

    review = build_cross_dataset_review(
        requested_dataset_ids=[],
        dataset_name_by_id={},
        experiments=[{"id": "e1", "test_code": "EXP-CONC", "status": "EXECUTED",
                      "arguments_json": {"source_datasets": ["orders"]}}],
        evidences=[], verifications=[], join_events=[],
    )
    assert review["review_checks"]["dataset_scope_recorded"] is True
    assert review["status"] == "NOT_CROSS_DATASET"


def test_multiverse_pin_maps_rate_and_enum_aggregations():
    from enum import Enum
    from types import SimpleNamespace
    from packages.analytics_core.src.runtime.controller import _multiverse_pinned_aggregations

    class Agg(Enum):
        SUM = "sum"
        RATE = "rate"
        COUNT = "count"

    def sem(a):
        return SimpleNamespace(metric_definition=SimpleNamespace(aggregation_type=a))

    assert _multiverse_pinned_aggregations(sem(Agg.SUM)) == ["SUM"]
    assert _multiverse_pinned_aggregations(sem(Agg.RATE)) == ["MEAN"]
    assert _multiverse_pinned_aggregations(sem("proportion")) == ["MEAN"]
    assert _multiverse_pinned_aggregations(sem(Agg.COUNT)) is None
