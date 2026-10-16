"""Reconciliation must never manufacture MATCHED from missing execution evidence."""
from packages.analytics_core.src.governance.plan_reconciliation import reconcile_execution_with_contract
from packages.analytics_core.src.intelligence.canonical_question_contract import CanonicalQuestionContract


def _contract(target="revenue", groups=("region",)):
    return CanonicalQuestionContract(
        question_text="q", task_family="RANKING", claim_type="OBSERVATION", estimand="ranking",
        target_column=target, grouping_columns=tuple(groups),
    )


def test_matched_requires_positive_evidence_for_every_fixed_operation():
    assert reconcile_execution_with_contract(_contract(), "revenue", "region")["status"] == "MATCHED"


def test_group_reported_without_target_is_not_matched():
    r = reconcile_execution_with_contract(_contract(), None, "region")
    assert r["status"] == "NOT_VERIFIABLE"


def test_target_reported_without_group_is_not_matched():
    assert reconcile_execution_with_contract(_contract(), "revenue", None)["status"] == "NOT_VERIFIABLE"


def test_nothing_reported_is_not_verifiable():
    assert reconcile_execution_with_contract(_contract(), None, None)["status"] == "NOT_VERIFIABLE"


def test_contradiction_is_mismatch_and_beats_missing_evidence():
    assert reconcile_execution_with_contract(_contract(), "profit", None)["status"] == "MISMATCH"
    assert reconcile_execution_with_contract(_contract(), "revenue", "country")["status"] == "MISMATCH"


def test_one_reported_group_cannot_prove_two_required_groups():
    r = reconcile_execution_with_contract(_contract(groups=("region", "channel")), "revenue", "region")
    assert r["status"] == "NOT_VERIFIABLE"


def test_contract_fixing_nothing_cannot_match():
    assert reconcile_execution_with_contract(_contract(None, ()), "revenue", "region")["status"] == "NOT_VERIFIABLE"
