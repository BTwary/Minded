"""Regression tests for the 2026-09-24 vision audit (session 6).

Each test fails on the session-5 code and passes now.
"""
import re
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from packages.analytics_core.src.engines.stopping import StoppingEngine
from packages.analytics_core.src.intelligence.adversarial_attacker import (
    AdversarialAttacker,
    detect_simpsons_reversal,
)
from packages.analytics_core.src.platform_local_first import resolve_ai_enabled
from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- stopping
class TestRecursiveUncertaintyGate:
    KW = dict(max_iterations=12, current_posteriors=[0.55, 0.25, 0.20], counter_hypothesis_evaluated=True)

    def test_plateau_after_early_learning_is_recognised_as_stagnation(self):
        d = StoppingEngine.evaluate_stopping(
            current_entropy=0.90, initial_entropy=1.50, iteration_count=3,
            entropy_history=[1.50, 0.95, 0.90, 0.90], **self.KW,
        )
        assert d.should_stop and d.reason == "UNCERTAINTY_STAGNATED"

    def test_still_learning_does_not_stop(self):
        d = StoppingEngine.evaluate_stopping(
            current_entropy=0.60, initial_entropy=1.50, iteration_count=3,
            entropy_history=[1.50, 1.20, 0.90, 0.60], **self.KW,
        )
        assert not d.should_stop

    def test_legacy_behaviour_without_history_is_unchanged(self):
        d = StoppingEngine.evaluate_stopping(
            current_entropy=0.90, initial_entropy=1.50, iteration_count=5, **self.KW,
        )
        assert not d.should_stop

    def test_stagnation_never_overrides_unresolved_adversarial_issue(self):
        d = StoppingEngine.evaluate_stopping(
            current_entropy=0.90, initial_entropy=1.50, iteration_count=3,
            entropy_history=[1.50, 0.95, 0.90, 0.90],
            unresolved_adversarial_issues=[{"x": 1}], **self.KW,
        )
        assert not d.should_stop


# --------------------------------------------------------------------------- adversarial
def _hyp(dim="plan", metric="revenue"):
    return NS(hypothesis_code="H1", target_dimension=dim, target_metric=metric,
              dimension_resolution_status="RESOLVED")


class TestAdversarialHonesty:
    def test_unbuildable_attack_is_not_reported_as_survived(self):
        r = AdversarialAttacker.design_attack(_hyp(dim=""), [_hyp()], pd.DataFrame({"revenue": [1, 2, 3]}))
        assert r.attack_status == "NOT_APPLICABLE"
        r = AdversarialAttacker.design_attack(_hyp(), [_hyp()], pd.DataFrame({"revenue": [1, 2, 3]}))
        assert r.attack_status == "NOT_APPLICABLE"

    @staticmethod
    def _noise(seed, n=300, effect=0.0):
        g = np.random.default_rng(seed)
        d = pd.DataFrame({"plan": g.choice(["basic", "pro"], n), "region": g.choice(list("ABCDE"), n),
                          "revenue": g.normal(100, 15, n)})
        d.loc[d.plan == "pro", "revenue"] += effect
        return d

    def test_noise_and_consistent_effects_are_not_flagged_as_simpsons(self):
        flagged_noise = sum(detect_simpsons_reversal(self._noise(s), "plan", "region", "revenue") is not None for s in range(60))
        flagged_effect = sum(detect_simpsons_reversal(self._noise(s, effect=7.5), "plan", "region", "revenue") is not None for s in range(60))
        assert flagged_noise <= 1 and flagged_effect <= 1

    def test_reversal_outside_the_first_two_groups_is_detected(self):
        g = np.random.default_rng(3)
        spec = {("A", "mild"): (300, 90), ("A", "severe"): (300, 50), ("B", "mild"): (300, 88),
                ("B", "severe"): (300, 48), ("C", "mild"): (40, 92), ("C", "severe"): (560, 52)}
        rows = [(d, s, v) for (d, s), (n, mu) in spec.items() for v in g.normal(mu, 3, n)]
        df = pd.DataFrame(rows, columns=["drug", "severity", "recovery"])
        found = detect_simpsons_reversal(df, "drug", "severity", "recovery")
        assert found is not None and found["marginal_difference"] * found["adjusted_difference"] < 0

    def test_effect_modification_without_reversal_is_not_simpsons(self):
        g = np.random.default_rng(9)
        rows = []
        for (t, s), mu in {("A", "x"): 110, ("B", "x"): 100, ("A", "y"): 98, ("B", "y"): 100}.items():
            rows += [(t, s, v) for v in g.normal(mu, 3, 100)]
        df = pd.DataFrame(rows, columns=["t", "s", "m"])  # sign differs by stratum, marginal & adjusted agree
        assert detect_simpsons_reversal(df, "t", "s", "m") is None

    def test_tiny_cells_cannot_trigger_a_reversal(self):
        df = pd.DataFrame({"t": ["A", "A", "B", "B"] * 2, "s": ["x", "y", "x", "y"] * 2, "m": [1.0, 9.0, 5.0, 2.0] * 2})
        assert detect_simpsons_reversal(df, "t", "s", "m", min_cell_n=5) is None


# --------------------------------------------------------------------------- AI + secrets
class TestZeroAiAuthority:
    def test_explicit_disable_beats_a_leftover_provider(self):
        assert resolve_ai_enabled({"AI_ENABLED": "false", "AI_PROVIDER": "openai"}) is False

    def test_provider_only_opts_in_when_flag_unset(self):
        assert resolve_ai_enabled({"AI_PROVIDER": "openai"}) is True
        assert resolve_ai_enabled({"AI_PROVIDER": "none"}) is False
        assert resolve_ai_enabled({}) is False

    def test_explicit_enable_is_respected(self):
        assert resolve_ai_enabled({"AI_ENABLED": "true"}) is True

    def test_settings_and_factory_share_one_resolver(self):
        # Both the settings status API and the agent factory must derive "AI enabled"
        # from the same function, so they can never disagree.
        import inspect
        from apps.api.src.ai.providers import factory
        from apps.api.src.core import config
        assert factory.resolve_ai_enabled is resolve_ai_enabled
        assert config.resolve_ai_enabled is resolve_ai_enabled
        assert "resolve_ai_enabled()" in inspect.getsource(config.Settings)


class TestProductionSecret:
    @staticmethod
    def _validate(secret):
        from apps.api.src.core.config import Settings
        Settings.validate_production_security(NS(ENVIRONMENT="production", SECRET_KEY=secret,
                                                 CORS_ORIGINS=["https://x.example"], CODE_EXECUTION_ENABLED=False))

    @pytest.mark.parametrize("bad", ["replace-with-a-secure-random-secret-key-min-32-chars", "abc",
                                     "super-secret-dev-x" * 3, "changeme" + "x" * 30])
    def test_weak_or_placeholder_secrets_are_rejected(self, bad):
        with pytest.raises(ValueError):
            self._validate(bad)

    def test_strong_secret_is_accepted(self):
        self._validate("Nlqlc5MCCla798lSuMewX_qA9zP4rT1uVw2yZ3bC4dE5fG6h")


# --------------------------------------------------------------------------- data quality
class TestKeyDuplicationMateriality:
    @staticmethod
    def _df(dups):
        g = np.random.default_rng(0)
        n = 1000
        ids = [f"C{i}" for i in range(n)]
        for i in range(dups):
            ids[i] = "C_dup"
        return pd.DataFrame({"customer_id": ids, "region": g.choice(list("NSEW"), n), "revenue": g.normal(100, 20, n)})

    def test_immaterial_key_duplication_warns_but_does_not_block(self):
        a = DataQualityGate.evaluate_fitness(self._df(2))
        assert a.can_proceed and a.fitness_verdict == "CAUTION"
        assert a.duplicate_key_columns["customer_id"]["duplicated_row_pct"] < 1.0

    def test_material_key_duplication_still_fails_closed(self):
        a = DataQualityGate.evaluate_fitness(self._df(20))
        assert not a.can_proceed and a.fitness_verdict == "UNFIT"

    def test_clean_keys_are_fit(self):
        assert DataQualityGate.evaluate_fitness(self._df(0)).fitness_verdict == "FIT"


# --------------------------------------------------------------------------- controller
def test_multiverse_skip_emits_exactly_one_event_with_the_true_reason():
    src = (REPO / "packages/analytics_core/src/runtime/controller.py").read_text()
    assert len(re.findall(r'"investigation\.multiverse\.skipped"', src)) == 1
    assert "No unambiguous grouping dimension resolved." in src
