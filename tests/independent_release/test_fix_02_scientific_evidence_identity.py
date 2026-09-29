from packages.analytics_core.src.engines.evidence_ledger import EvidenceLedger
from packages.analytics_core.src.graph.evidence_identity import (
    canonical_dataset_identity,
    compute_evidence_identity,
)
from packages.schemas.src.analysis import EpistemicClaimType


def _identity(query="SELECT region, SUM(revenue) AS revenue FROM sales GROUP BY region", *, dataset="v1", dimension="region"):
    ds = canonical_dataset_identity({"sales": dataset})
    scope = {"metric": "revenue", "dimension": dimension, "aggregation_type": "SUM", "grouped": True}
    return compute_evidence_identity(
        ds, query, "duckdb_sql", scope,
        estimator="comparison", formula="SUM(revenue)", parameters=scope,
        output_columns=["region", "revenue"], input_row_count=100, output_row_count=3,
    )


def test_identical_computation_has_one_identity_but_changed_scope_does_not():
    a = _identity()
    b = _identity()
    assert a == b
    assert a != _identity(dimension="segment")
    assert a != _identity(dataset="v2")
    assert a != _identity(query="SELECT region, SUM(revenue) AS revenue FROM sales WHERE year=2025 GROUP BY region")


def test_ledger_returns_canonical_record_for_duplicate_identity():
    ledger = EvidenceLedger("INV-1")
    identity = _identity()
    first = ledger.record_claim(
        claim_statement="Revenue concentration by region.",
        claim_type=EpistemicClaimType.OBSERVATION,
        source_experiment_id="EXP-1",
        computation_proof={"sql": "same", "value": 10},
        verification_status="VERIFIED",
        evidence_identity=identity,
    )
    second = ledger.record_claim(
        claim_statement="Same computation retried.",
        claim_type=EpistemicClaimType.OBSERVATION,
        source_experiment_id="EXP-2",
        computation_proof={"sql": "same", "value": 10},
        verification_status="VERIFIED",
        evidence_identity=identity,
    )
    assert second is first
    assert len(ledger.get_all_claims()) == 1
