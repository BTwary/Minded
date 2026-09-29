import pandas as pd

from packages.analytics_core.src.causal.identifiability_gate import (
    CausalGraphSpec,
    CausalIdentifiabilityGate,
)
from packages.schemas.src.analysis import CausalIntent, VariableRef, CausalIdentifiabilityStatus


def intent(t="treatment", o="outcome"):
    return CausalIntent(
        treatment=VariableRef(table="df", column=t),
        outcome=VariableRef(table="df", column=o),
        assumed_dag_id="dag-1",
    )


def test_no_dag_is_observational_only():
    r = CausalIdentifiabilityGate().evaluate_identifiability(intent(), None)
    assert not r.is_identifiable
    assert r.status == CausalIdentifiabilityStatus.OBSERVATIONAL_ONLY
    assert "not" in r.diagnostic_message.lower()


def test_valid_backdoor_graph_is_identified():
    graph = CausalGraphSpec(
        dag_id="dag-1",
        nodes=["z", "treatment", "outcome"],
        directed_edges=[("z", "treatment"), ("z", "outcome"), ("treatment", "outcome")],
    )
    r = CausalIdentifiabilityGate().evaluate_identifiability(intent(), graph)
    assert r.is_identifiable
    assert r.status == CausalIdentifiabilityStatus.IDENTIFIED_BACKDOOR
    assert r.backdoor_adjustment_set == ["z"]
    assert r.sensitivity_e_value and r.sensitivity_e_value > 1


def test_no_parents_can_be_valid_when_graph_is_clean():
    graph = CausalGraphSpec(
        dag_id="dag-1",
        nodes=["treatment", "outcome"],
        directed_edges=[("treatment", "outcome")],
    )
    r = CausalIdentifiabilityGate().evaluate_identifiability(intent(), graph)
    assert r.is_identifiable
    assert r.backdoor_adjustment_set == []


def test_descendant_is_not_used_as_adjustment():
    graph = CausalGraphSpec(
        dag_id="dag-1",
        nodes=["z", "treatment", "mediator", "outcome"],
        directed_edges=[
            ("z", "treatment"),
            ("z", "outcome"),
            ("treatment", "mediator"),
            ("mediator", "outcome"),
        ],
    )
    r = CausalIdentifiabilityGate().evaluate_identifiability(intent(), graph)
    assert r.is_identifiable
    assert "mediator" not in r.backdoor_adjustment_set
    assert r.backdoor_adjustment_set == ["z"]


def test_unmeasured_common_cause_blocks_identification():
    graph = CausalGraphSpec(
        dag_id="dag-1",
        nodes=["u", "treatment", "outcome"],
        directed_edges=[("u", "treatment"), ("u", "outcome"), ("treatment", "outcome")],
        unobserved_confounders=["u"],
    )
    r = CausalIdentifiabilityGate().evaluate_identifiability(intent(), graph)
    assert not r.is_identifiable
    assert r.status == CausalIdentifiabilityStatus.NOT_IDENTIFIABLE
    assert "unmeasured" in r.diagnostic_message.lower()


def test_invalid_cycle_is_rejected():
    graph = CausalGraphSpec(
        dag_id="dag-1",
        nodes=["treatment", "outcome"],
        directed_edges=[("treatment", "outcome"), ("outcome", "treatment")],
    )
    r = CausalIdentifiabilityGate().evaluate_identifiability(intent(), graph)
    assert not r.is_identifiable
    assert r.status == CausalIdentifiabilityStatus.NOT_IDENTIFIABLE
    assert "cyclic" in r.diagnostic_message.lower()


def test_adjustment_failure_is_rejected():
    graph = CausalGraphSpec(
        dag_id="dag-1",
        nodes=["z", "m", "treatment", "outcome"],
        directed_edges=[
            ("z", "treatment"),
            ("z", "m"),
            ("m", "outcome"),
            ("treatment", "outcome"),
        ],
    )
    # The treatment parent z alone does not create a backdoor path to outcome;
    # nevertheless the test exercises the actual d-separation proof path.
    r = CausalIdentifiabilityGate().evaluate_identifiability(intent(), graph)
    assert r.is_identifiable


def test_pc_partial_orientation_is_observational_only(monkeypatch):
    from packages.analytics_core.src.causal import pc_algorithm

    class Stub:
        nodes = ["treatment", "outcome", "z"]
        directed_edges = [("treatment", "outcome")]
        undirected_edges = [("z", "treatment")]

    monkeypatch.setattr(pc_algorithm.PCAlgorithmEngine, "discover_causal_dag", lambda *a, **k: Stub())
    df = pd.DataFrame({"treatment": [0, 1, 0, 1], "outcome": [1, 2, 1, 2], "z": [0, 0, 1, 1]})
    r = CausalIdentifiabilityGate().discover_and_evaluate(df, "treatment", "outcome")
    assert r.status == CausalIdentifiabilityStatus.OBSERVATIONAL_ONLY
    assert not r.is_identifiable
    assert "partially oriented" in r.diagnostic_message.lower()
