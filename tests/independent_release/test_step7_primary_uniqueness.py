"""Primary analytical-identity uniqueness proofs."""
from types import SimpleNamespace
import pytest
from packages.analytics_core.src.intelligence.experiment_contract_validation import assert_single_primary_per_identity


def _exp(code, role, ident):
    return SimpleNamespace(code=code, experiment_role=role, analytical_identity=ident)


def test_second_primary_same_identity_is_rejected():
    exps = [_exp("E1", "PRIMARY", "abc"), _exp("E2", "PRIMARY", "abc")]
    with pytest.raises(ValueError, match="duplicate PRIMARY"):
        assert_single_primary_per_identity(exps)


def test_supporting_and_adversarial_same_identity_do_not_count_as_primary():
    exps = [_exp("E1", "PRIMARY", "abc"), _exp("E2", "SUPPORTING", "abc"), _exp("E3", "ADVERSARIAL", "abc")]
    assert_single_primary_per_identity(exps)
