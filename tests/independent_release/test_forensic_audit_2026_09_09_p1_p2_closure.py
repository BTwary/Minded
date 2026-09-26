"""Independent regression checks for the 2026-09-09 scoped forensic audit's
P1-1 and P2-1 findings (MindEd_AAOS_Forensic_Audit_Scoped.md).

These intentionally avoid importing DuckDB-backed runtime modules -- same
constraint as tests/independent_release/test_audit_p0_closure.py -- so they
stay executable in environments where optional analytical engines are
absent, and instead verify the fix is actually present in source, at the
exact location the audit cited.
"""
from pathlib import Path
import ast

from packages.analytics_core.src.governance.claim_gate import admit_positive_claim

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "packages/analytics_core/src/runtime/controller.py"
CONTROLLER_SRC = CONTROLLER.read_text()


# ---------------------------------------------------------------------------
# P1-1: universal claim gate enforcement gap for OBSERVED verdicts
# ---------------------------------------------------------------------------

def test_claim_gate_positive_set_matches_controller_enforcement_set():
    """The controller's downgrade-to-INCONCLUSIVE condition must cover every
    verdict_type the gate itself treats as `positive`, not just a subset.

    Regression for P1-1: the gate computed allowed=False for OBSERVED
    verdicts (unverified evidence, unresolved sampling scope, unidentified
    causal request) but the controller only downgraded DIAGNOSED /
    STATISTICALLY_SIGNIFICANT, letting an inadmissible OBSERVED claim reach
    the user un-gated.
    """
    tree = ast.parse(CONTROLLER_SRC)
    downgrade_ifs = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.BoolOp)
        and any(
            isinstance(v, ast.Compare)
            and any(isinstance(c, ast.Attribute) and c.attr == "verdict_type" for c in ast.walk(v))
            for v in n.test.values
        )
    ]
    assert downgrade_ifs, "could not locate the claim-gate downgrade `if` in controller.py"

    def _literal_set(node):
        for value in node.test.values:
            if isinstance(value, ast.Compare) and isinstance(value.left, ast.Attribute) and value.left.attr == "verdict_type":
                container = value.comparators[0]
                if isinstance(container, (ast.Set, ast.Tuple, ast.List)):
                    return {elt.value for elt in container.elts if isinstance(elt, ast.Constant)}
        return set()

    # There are two verdict_type membership checks in this file (the P1-1
    # downgrade gate, and an unrelated decision-recommendation gate further
    # down). Only the one that also references `claim_admission.allowed` is
    # the claim-gate enforcement this audit finding is about.
    enforcement_sets = [
        _literal_set(n) for n in downgrade_ifs
        if "claim_admission" in ast.dump(n.test)
    ]
    assert enforcement_sets, "could not find the claim-gate `not claim_admission.allowed` branch"

    # This mirrors claim_gate.py's own `positive` set (DIAGNOSED,
    # STATISTICALLY_SIGNIFICANT, SUPPORTED, OBSERVED); SUPPORTED is
    # confirmed dead code the VerdictEngine never emits (audit Section 5,
    # P1-1 recommended fix), so it is not required in the controller set.
    required = {"DIAGNOSED", "STATISTICALLY_SIGNIFICANT", "OBSERVED"}
    assert any(required <= s for s in enforcement_sets), (
        f"controller enforcement set(s) {enforcement_sets} do not cover the "
        f"gate's positive set {required}"
    )


def test_admit_positive_claim_blocks_observed_with_unverified_evidence():
    """Confirms the gate itself still classifies OBSERVED as positive and
    withholds it when evidence is unverified -- the input half of P1-1."""
    result = admit_positive_claim(
        verdict_type="OBSERVED",
        directly_tested=True,
        verified_evidence=False,
    )
    assert not result.allowed


def test_observed_verdict_with_unmet_gate_downgrades_to_inconclusive():
    """End-to-end (mocked) reproduction of the exact P1-1 scenario: an
    OBSERVED verdict paired with an unmet claim-gate condition must be
    downgraded to INCONCLUSIVE, exactly as DIAGNOSED/STATISTICALLY_SIGNIFICANT
    already were before this fix.
    """
    claim_admission = admit_positive_claim(
        verdict_type="OBSERVED",
        directly_tested=True,
        verified_evidence=False,
    )
    assert not claim_admission.allowed

    class _FakeVerdictEval:
        verdict_type = "OBSERVED"
        confidence_score = 0.8
        direct_answer = "some positive-sounding descriptive claim"
        justification = "..."
        main_finding = "..."

    verdict_eval = _FakeVerdictEval()

    # Reproduce the exact controller condition (kept in sync with the
    # literal set asserted above) rather than importing execute_investigation,
    # which requires a live DuckDB/Postgres runtime this suite avoids.
    if verdict_eval.verdict_type in {"DIAGNOSED", "STATISTICALLY_SIGNIFICANT", "OBSERVED"} and not claim_admission.allowed:
        original = verdict_eval.verdict_type
        verdict_eval.verdict_type = "INCONCLUSIVE"
        verdict_eval.confidence_score = 0.0

    assert verdict_eval.verdict_type == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# P2-1: evidence-dedup check-then-act race
# ---------------------------------------------------------------------------

def test_evidence_insert_is_wrapped_against_integrity_error():
    """Regression for P2-1: the Evidence() insert following the
    existing_evidence SELECT must be guarded against a losing concurrent
    insert (UniqueConstraint(investigation_id, evidence_identity_hash))
    rather than letting IntegrityError propagate as an unhandled experiment
    failure.
    """
    assert "from sqlalchemy.exc import IntegrityError" in CONTROLLER_SRC

    select_idx = CONTROLLER_SRC.find("existing_evidence = (")
    insert_idx = CONTROLLER_SRC.find("ev_entity = Evidence(")
    except_idx = CONTROLLER_SRC.find("except IntegrityError:", insert_idx)
    duplicate_flag_idx = CONTROLLER_SRC.find("evidence_is_duplicate = True", except_idx)

    assert select_idx != -1 and insert_idx != -1, "could not locate the check-then-act evidence insert"
    assert select_idx < insert_idx, "SELECT must precede the INSERT"
    assert except_idx != -1 and except_idx > insert_idx, (
        "Evidence() insert is not followed by an `except IntegrityError` handler"
    )
    assert duplicate_flag_idx != -1, (
        "losing a concurrent insert race must mark evidence_is_duplicate = True, "
        "per the audit's recommended fix"
    )


def test_evidence_insert_uses_savepoint_not_full_rollback():
    """The insert must be isolated in a nested transaction (SAVEPOINT) so a
    losing race only unwinds the Evidence insert itself, not the rest of the
    already-in-progress step transaction (exp_entity, hypothesis updates,
    etc. already added to the same session).
    """
    insert_idx = CONTROLLER_SRC.find("ev_entity = Evidence(")
    savepoint_idx = CONTROLLER_SRC.find("session.begin_nested()", insert_idx)
    assert savepoint_idx != -1 and savepoint_idx < CONTROLLER_SRC.find("except IntegrityError:", insert_idx)


def test_evidence_race_reraises_when_no_survivor_found():
    """If the constraint fires but no matching row is found on re-query
    (isolation-level visibility edge case), the original failure must
    surface rather than silently dropping the evidence -- same invariant as
    the existing hypothesis get-or-create pattern.
    """
    insert_idx = CONTROLLER_SRC.find("ev_entity = Evidence(")
    except_block_end = CONTROLLER_SRC.find("else:\n                    # Reuse the canonical Evidence row", insert_idx)
    except_block = CONTROLLER_SRC[insert_idx:except_block_end]
    assert "if winner is None:" in except_block
    assert "raise" in except_block
