"""Session 8: AA-OS as the analyst's worker, the human as the verifier.

Covers (1) the verification packet as a pure function, (2) the review / sign-off
state machine, (3) the API surface against a real controller run, and (4) three
defects found while wiring the workflow to real data:

  DEFECT-022  GET /investigations/{id} raised NameError for every investigation.
  DEFECT-023  The controller's EXP-COND (Simpson's conditional) query had no
              ORDER BY, so it failed on every run, invisibly.
  (visibility) adversarial / multiverse / failed-step / stopping outcomes never
              reached the API, so a verifier could not see what was tried.
"""
import copy

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from packages.analytics_core.src.governance import verification_packet as vp
from packages.analytics_core.src.governance.verification_packet import (
    BLOCKER, REVIEW, SPOT_CHECK, INFO, PASSED, FAILED, WARNING, NOT_RUN, NEEDS_HUMAN,
    CONFIRMED, REJECTED, NEEDS_REWORK,
    build_verification_packet, evaluate_review, validate_decision, validate_signoff, current_signoff,
)


# ---------------------------------------------------------------------------
# A payload shaped like GET /investigations/{id}, for pure-function tests.
# ---------------------------------------------------------------------------
def _payload(**over):
    p = {
        "id": "INV-X", "status": "COMPLETED", "question": "Why does cost differ across region?",
        "verdict_type": "DIAGNOSED", "confidence_score": 0.9, "direct_answer": "Region A drives cost.",
        "reproducible_manifest_hash": "abc123", "stopping_criteria_met": True, "stopping_reason": "DECISIVE_SIGNAL_RESOLVED",
        "semantic_world_model": {
            "primary_dataset": "ds",
            "contract": {
                "claim_type": "ASSOCIATION", "problem_class": "DIAGNOSTIC",
                "target": {"target": "cost", "aggregation_type": "sum"},
                "grain": {"grain": "order", "grain_proven": True, "primary_key": "id"},
                "semantic_interpretations": ["x"], "unresolved_questions": [],
                "scope": {"dataset": "ds"},
            },
            "data_readiness": {
                "dataset_name": "ds", "row_count": 100, "column_count": 3, "fitness_verdict": "FIT",
                "source_fingerprints": {"ds": "fp"},
                "checks": {"missingness": {"cost": 0.0}, "duplicates": 0, "selection_bias_indicators": [],
                           "leakage_indicators": [], "unit_consistency_indicators": [], "temporal_issues": []},
            },
        },
        "evidence": [{"id": "E1", "statement": "s", "validation_status": "VERIFIED", "row_count_analyzed": 100,
                      "sql_executed": "SELECT 1"}],
        "evidence_verifications": [{"evidence_id": "E1", "status": "VERIFIED", "primary_tool": "duckdb_sql",
                                    "secondary_tool": "polars_vectorized", "observed_delta_pct": 0.0,
                                    "tolerance_threshold": 0.01}],
        "adversarial_findings": [{"leading_hypothesis": "HYP-01", "attack_status": "SURVIVED",
                                  "attack_mechanism": "m", "epistemic_impact": "none",
                                  "discriminating_test_sql": "SELECT 2", "details": {}}],
        "multiverse": {"applicable": True, "total_specifications": 9, "concordant_specifications": 9,
                       "robustness_pct": 100.0, "specification_curve": [], "dimension": "region", "metric": "cost"},
        "missingness": {"claim_relevant": True, "epistemic_classification": "ROBUST", "aggregation_type": "sum",
                        "semantic_resolution_status": "RESOLVED", "grouping_column": "region", "metric_name": "cost"},
        "assumption_ledger": {"items": []},
        "claim_gate": {"outcome": "ANSWER", "requested_claim": "ASSOCIATION", "allowed_claim": "Association supported."},
        "causal_status": {"gate": {"status": "OBSERVATIONAL_ONLY"}},
        "decision_recommendations": [], "failed_steps": [],
    }
    for k, v in over.items():
        p[k] = v
    return p


def _by_id(packet):
    return {i["id"]: i for i in packet["items"]}


def _one(packet, prefix):
    hits = [i for i in packet["items"] if i["id"].startswith(prefix)]
    assert len(hits) == 1, [i["id"] for i in packet["items"]]
    return hits[0]


# ---------------------------------------------------------------------------
# 1. The packet
# ---------------------------------------------------------------------------
class TestPacketIsHonest:
    def test_clean_run_asks_the_human_only_for_what_only_a_human_can_judge(self):
        pk = build_verification_packet(_payload())
        assert pk["summary"]["blockers"] == 0
        required = {i["id"] for i in pk["items"] if i["requires_decision"]}
        assert required == {"INTERP-QUESTION"}          # interpretation is the one thing the machine cannot check
        for i in pk["items"]:
            if i["priority"] == SPOT_CHECK:
                assert i["machine_status"] == PASSED     # optional spot checks are machine-verified

    def test_unknown_is_never_a_pass(self):
        # No readiness, no adversarial record, no multiverse, no claim gate, no evidence.
        p = _payload(adversarial_findings=[], multiverse=None, claim_gate=None, evidence=[], evidence_verifications=[])
        p["semantic_world_model"]["data_readiness"] = None
        pk = build_verification_packet(p)
        by = _by_id(pk)
        assert by["DATA-READINESS"]["machine_status"] == NOT_RUN and by["DATA-READINESS"]["requires_decision"]
        assert by["CHALLENGE-NONE"]["machine_status"] == NOT_RUN
        assert by["ROBUSTNESS"]["machine_status"] == NOT_RUN
        assert by["CLAIM-GATE"]["machine_status"] == NOT_RUN and by["CLAIM-GATE"]["requires_decision"]
        assert by["EVID-NONE"]["priority"] == BLOCKER
        assert pk["summary"]["machine_checked_and_passed"] == 0 or all(
            i["machine_status"] != PASSED for i in pk["items"] if i["id"] in
            {"DATA-READINESS", "CHALLENGE-NONE", "ROBUSTNESS", "CLAIM-GATE", "EVID-NONE"})

    def test_the_packet_is_a_pure_deterministic_function(self):
        p = _payload()
        assert build_verification_packet(copy.deepcopy(p)) == build_verification_packet(copy.deepcopy(p))

    def test_fingerprint_changes_when_the_substance_changes(self):
        a = build_verification_packet(_payload())
        p = _payload(); p["evidence"][0]["row_count_analyzed"] = 99
        b = build_verification_packet(p)
        assert _by_id(a)["EVID-VERIFIED"]["fingerprint"] != _by_id(b)["EVID-VERIFIED"]["fingerprint"]
        assert a["packet_fingerprint"] != b["packet_fingerprint"]
        assert _by_id(a)["INTERP-QUESTION"]["fingerprint"] == _by_id(b)["INTERP-QUESTION"]["fingerprint"]

    def test_items_are_ordered_attention_first(self):
        p = _payload(adversarial_findings=[{"leading_hypothesis": "H", "attack_status": "REFUTED", "epistemic_impact": "x", "details": {}}])
        order = [i["priority"] for i in build_verification_packet(p)["items"]]
        assert order == sorted(order, key=lambda x: {BLOCKER: 0, REVIEW: 1, SPOT_CHECK: 2, INFO: 3}[x])
        assert order[0] == BLOCKER


class TestEachRiskIsSurfaced:
    def test_a_defaulted_aggregation_is_a_blocker_and_notes_the_ledger_contradiction(self):
        p = _payload()
        p["missingness"]["semantic_resolution_status"] = "UNRESOLVED_DEFAULT_SUM"
        p["assumption_ledger"] = {"items": [{"statement": "[METRIC_SEMANTICS] Target metric 'cost' is interpreted using SUM aggregation.",
                                             "validated": True, "sensitivity_risk": "low"}]}
        it = _by_id(build_verification_packet(p))["METRIC-AGGREGATION"]
        assert it["priority"] == BLOCKER and it["machine_status"] == NEEDS_HUMAN
        assert it["evidence"]["assumption_ledger_marks_this_validated"] is True
        assert "SUM" in it["you_confirm"]

    def test_a_refutation_asks_whether_the_stratifier_is_a_confounder_not_a_mediator(self):
        rev = {"primary_dimension": "drug", "secondary_dimension": "severity", "groups": ["A", "C"],
               "marginal_difference": 15.36, "adjusted_difference": -1.97, "strata_used": 2}
        p = _payload(adversarial_findings=[{"leading_hypothesis": "HYP-01", "attack_status": "REFUTED", "is_falsified": True,
                                            "epistemic_impact": "reversal", "details": {"simpsons_reversal": rev}}])
        it = _one(build_verification_packet(p), "CHALLENGE-")
        assert it["priority"] == BLOCKER and it["machine_status"] == FAILED
        assert "mediator" in it["you_confirm"] and "severity" in it["you_confirm"] and "drug" in it["you_confirm"]
        assert it["evidence"]["simpsons_reversal"]["adjusted_difference"] == -1.97

    @pytest.mark.parametrize("status,priority,machine", [
        ("WEAKENED", REVIEW, WARNING), ("NOT_APPLICABLE", REVIEW, NOT_RUN),
        ("SURVIVED", SPOT_CHECK, PASSED), ("SOMETHING_NEW", REVIEW, NEEDS_HUMAN), ("CONTRADICTED", BLOCKER, FAILED),
    ])
    def test_every_attack_outcome_maps_to_the_right_attention_level(self, status, priority, machine):
        p = _payload(adversarial_findings=[{"leading_hypothesis": "H", "attack_status": status, "epistemic_impact": "e", "details": {}}])
        it = _one(build_verification_packet(p), "CHALLENGE-")
        assert (it["priority"], it["machine_status"]) == (priority, machine)

    def test_a_skipped_challenge_is_context_not_a_pass(self):
        p = _payload(adversarial_findings=[{"task": "DESCRIPTIVE", "reason": "Not required by the universal analysis evidence contract."}])
        it = _by_id(build_verification_packet(p))["CHALLENGE-NONE"]
        assert it["machine_status"] == NOT_RUN and it["priority"] == INFO
        assert "Not required" in it["evidence"]["reason"]

    def test_an_unverified_result_is_a_blocker_with_the_sql_to_recompute_it(self):
        p = _payload()
        p["evidence"][0]["validation_status"] = "FAILED"
        p["evidence_verifications"][0]["status"] = "FAILED"
        it = _one(build_verification_packet(p), "EVID-FAIL")
        assert it["priority"] == BLOCKER and it["machine_status"] == FAILED
        assert it["reproduce"]["sql"] == "SELECT 1" and it["reproduce"]["table_alias"] == "data_table"

    def test_a_result_with_no_verification_row_is_not_counted_as_verified(self):
        p = _payload(evidence_verifications=[])
        pk = build_verification_packet(p)
        assert _one(pk, "EVID-FAIL")["machine_status"] == NOT_RUN
        assert "EVID-VERIFIED" not in _by_id(pk)

    @pytest.mark.parametrize("pct,priority", [(100.0, SPOT_CHECK), (80.0, SPOT_CHECK), (77.7, REVIEW), (50.0, REVIEW), (49.9, BLOCKER)])
    def test_robustness_thresholds(self, pct, priority):
        p = _payload()
        p["multiverse"]["robustness_pct"] = pct
        assert _by_id(build_verification_packet(p))["ROBUSTNESS"]["priority"] == priority

    def test_the_hardcoded_unvalidated_verification_assumption_is_not_a_human_chore(self):
        p = _payload(); p["assumption_ledger"] = {"items": [
            {"statement": "[INDEPENDENT_VERIFICATION] Numerical claims ...", "validated": False, "sensitivity_risk": "high"},
            {"statement": "[SEASONALITY] Effects are not seasonal.", "validated": False, "sensitivity_risk": "high"},
            {"statement": "[MINOR] x", "validated": False, "sensitivity_risk": "low"}]}
        ids = [i["id"] for i in build_verification_packet(p)["items"] if i["category"] == "ASSUMPTION"]
        assert len(ids) == 1                                    # only the genuinely unvalidated high-risk one
        assert "SEASONALITY" in build_verification_packet(p)["items"][[i["id"] for i in build_verification_packet(p)["items"]].index(ids[0])]["you_confirm"]

    def test_failed_steps_and_unresolved_questions_and_unfit_data_are_surfaced(self):
        p = _payload(failed_steps=[{"step_code": "EXP-X", "step_type": "EXPERIMENT_EXECUTION", "error": "boom"}])
        p["semantic_world_model"]["contract"]["unresolved_questions"] = ["Which revenue column?"]
        p["semantic_world_model"]["data_readiness"]["fitness_verdict"] = "UNFIT"
        by = _by_id(build_verification_packet(p))
        assert by["COVERAGE-FAILED-STEPS"]["machine_status"] == FAILED
        assert by["INTERP-UNRESOLVED"]["priority"] == BLOCKER
        assert by["DATA-READINESS"]["priority"] == BLOCKER

    def test_a_flagged_leakage_indicator_downgrades_a_fit_dataset_to_review(self):
        p = _payload()
        p["semantic_world_model"]["data_readiness"]["checks"]["leakage_indicators"] = ["status_col"]
        it = _by_id(build_verification_packet(p))["DATA-READINESS"]
        assert it["priority"] == REVIEW and it["machine_status"] == WARNING

    def test_claim_gate_levels_and_stopping(self):
        p = _payload(); p["claim_gate"] = {"outcome": "REFUSE", "allowed_claim": "None"}
        assert _by_id(build_verification_packet(p))["CLAIM-GATE"]["priority"] == BLOCKER
        p["claim_gate"] = {"outcome": "QUALIFIED_ANSWER", "allowed_claim": "Association only"}
        assert _by_id(build_verification_packet(p))["CLAIM-GATE"]["priority"] == REVIEW
        assert "do not report it as a cause" in _by_id(build_verification_packet(p))["CLAIM-GATE"]["you_confirm"]
        assert "STOPPING" not in _by_id(build_verification_packet(_payload()))
        assert _by_id(build_verification_packet(_payload(stopping_criteria_met=False)))["STOPPING"]["priority"] == REVIEW


# ---------------------------------------------------------------------------
# 2. Review and sign-off state machine
# ---------------------------------------------------------------------------
def _dec(packet, item_id, decision=CONFIRMED, comment="Checked against the source system.", fp=None):
    it = _by_id(packet)[item_id]
    return {"item_id": item_id, "item_fingerprint": fp or it["fingerprint"], "decision": decision, "comment": comment,
            "reviewer": {"email": "a@b"}, "at": "t"}


class TestReviewStateMachine:
    def _packet(self):
        p = _payload(adversarial_findings=[{"leading_hypothesis": "H", "attack_status": "WEAKENED", "epistemic_impact": "e", "details": {}}])
        p["missingness"]["semantic_resolution_status"] = "UNRESOLVED_DEFAULT_SUM"
        return build_verification_packet(p)

    def test_decision_validation(self):
        pk = self._packet()
        assert validate_decision(pk, "NOPE", CONFIRMED, "x" * 20)
        assert validate_decision(pk, "INTERP-QUESTION", "MAYBE", "x" * 20)
        assert validate_decision(pk, "INTERP-QUESTION", CONFIRMED, None) is None          # REVIEW item: bare confirm is fine
        assert validate_decision(pk, "METRIC-AGGREGATION", CONFIRMED, "ok")                # BLOCKER: needs a real comment
        assert validate_decision(pk, "METRIC-AGGREGATION", CONFIRMED, "SUM is right: it is a monetary total") is None
        assert validate_decision(pk, "INTERP-QUESTION", REJECTED, "no")                    # objections always explained
        assert validate_decision(pk, "INTERP-QUESTION", REJECTED, "Wrong metric was chosen") is None

    def test_cannot_verify_until_every_required_item_is_confirmed(self):
        pk = self._packet()
        required = [i["id"] for i in pk["items"] if i["requires_decision"]]
        assert len(required) >= 3
        r = evaluate_review(pk, [])
        assert r["state"] == "NOT_STARTED" and not r["can_verify"] and set(r["pending"]) == set(required)
        log = []
        for rid in required[:-1]:
            log.append(_dec(pk, rid))
        r = evaluate_review(pk, log)
        assert r["state"] == "IN_PROGRESS" and r["pending"] == [required[-1]]
        assert validate_signoff(r, vp.SIGNOFF_VERIFIED, None)
        log.append(_dec(pk, required[-1]))
        r = evaluate_review(pk, log)
        assert r["state"] == "READY_TO_SIGN" and r["can_verify"]
        assert validate_signoff(r, vp.SIGNOFF_VERIFIED, None) is None

    def test_an_objection_blocks_verification_but_can_always_be_recorded(self):
        pk = self._packet()
        required = [i["id"] for i in pk["items"] if i["requires_decision"]]
        log = [_dec(pk, rid) for rid in required] + [_dec(pk, required[0], REJECTED, "Wrong metric was chosen")]
        r = evaluate_review(pk, log)                        # the later rejection supersedes the earlier confirm
        assert r["state"] == "OBJECTIONS_RAISED" and r["rejected"] == [required[0]] and not r["can_verify"]
        assert validate_signoff(r, vp.SIGNOFF_VERIFIED, None)
        assert validate_signoff(r, REJECTED, "short")                                      # explained
        assert validate_signoff(r, REJECTED, "Rejected: the metric is a score, not additive.") is None
        assert validate_signoff(evaluate_review(pk, []), NEEDS_REWORK, "Need the refreshed extract first.") is None

    def test_a_decision_on_a_changed_item_no_longer_counts(self):
        pk = self._packet()
        required = [i["id"] for i in pk["items"] if i["requires_decision"]]
        log = [_dec(pk, rid) for rid in required]
        assert evaluate_review(pk, log)["can_verify"]
        log[0] = _dec(pk, required[0], fp="old-fingerprint")
        r = evaluate_review(pk, log)
        assert r["stale"] == [required[0]] and required[0] in r["pending"] and not r["can_verify"]

    def test_signoff_goes_stale_when_the_result_changes(self):
        pk = self._packet()
        so = [{"outcome": "VERIFIED", "packet_fingerprint": pk["packet_fingerprint"]}]
        assert current_signoff(pk, so)["stale"] is False
        pk2 = build_verification_packet(_payload(reproducible_manifest_hash="different"))
        assert current_signoff(pk2, so)["stale"] is True
        assert current_signoff(pk, []) is None


# ---------------------------------------------------------------------------
# 3. Against the real controller and the real API functions
# ---------------------------------------------------------------------------
from tests.independent_release.test_defect_007_controller_closure import (  # noqa: E402
    BaseIsolatedControllerTest as _ControllerBase, Investigation, InvestigationController,
    InMemoryDatasetProvider, gen_uuid,
)
from apps.api.src.models.entities import InvestigationEvent, User  # noqa: E402
from apps.api.src.api.v1 import investigations as api  # noqa: E402


def _simpsons_df():
    g = np.random.default_rng(3)
    spec = {("A", "mild"): (300, 90), ("A", "severe"): (300, 50), ("B", "mild"): (300, 88),
            ("B", "severe"): (300, 48), ("C", "mild"): (40, 92), ("C", "severe"): (560, 52)}
    rows = [(d, s, v) for (d, s), (n, mu) in spec.items() for v in g.normal(mu, 3, n)]
    return pd.DataFrame(rows, columns=["drug", "severity", "recovery"])


def _positive_df():
    return pd.DataFrame({
        "datacenter_region": ["us-east", "us-west", "eu-central", "ap-south"] * 100,
        "tier": (["Enterprise", "SMB"] * 50) + (["SMB", "SMB"] * 150),
        "cost_metric": [500.0 if (i % 4 == 0 and i < 200) else 100.0 for i in range(400)],
    })


class TestVerifierWorkflowEndToEnd(_ControllerBase):
    def _run(self, df, q, name="ds"):
        inv = Investigation(id=f"INV-S8-{gen_uuid()[:8]}", project_id=self.proj.id, user_id=self.user.id, question=q, status="QUEUED")
        self.db.add(inv); self.db.commit()
        ctl = InvestigationController(session_factory=self.SessionFactory, dataset_provider=InMemoryDatasetProvider({name: df}))
        self.assertTrue(ctl.execute_investigation(investigation_id=inv.id, worker_id="w-s8"))
        self.db.expire_all()
        return inv.id

    def _get(self, inv_id):
        return api.get_investigation(inv_id, current_user=self.user, db=self.db)

    # --- defects found while wiring this up -------------------------------
    def test_defect_022_get_investigation_no_longer_raises(self):
        inv_id = self._run(_positive_df(), "Why did cost_metric surge across datacenter_region?")
        payload = self._get(inv_id)                     # was: NameError('primary_dataset_name')
        self.assertEqual(payload["status"], "COMPLETED")
        self.assertEqual(payload["semantic_world_model"]["primary_dataset"], "ds")
        self.assertEqual(payload["semantic_world_model"]["data_readiness"]["dataset_name"], "ds")

    def test_defect_023_conditional_experiment_executes_and_nothing_fails_silently(self):
        inv_id = self._run(_simpsons_df(), "Why does recovery differ across drug?", "trial")
        payload = self._get(inv_id)
        self.assertIn("EXP-COND-SEVERITY", [e["id"] for e in payload["experiments"]])
        self.assertEqual(payload["failed_steps"], [])

    def test_what_the_system_tried_to_break_is_visible_to_the_verifier(self):
        payload = self._get(self._run(_simpsons_df(), "Why does recovery differ across drug?", "trial"))
        attacks = [a for a in payload["adversarial_findings"] if a.get("attack_status")]
        self.assertEqual(len(attacks), 1)
        self.assertEqual(attacks[0]["attack_status"], "REFUTED")
        self.assertTrue(attacks[0]["is_falsified"])
        self.assertEqual(attacks[0]["details"]["simpsons_reversal"]["secondary_dimension"], "severity")
        mv = payload["multiverse"]
        self.assertTrue(mv["applicable"]); self.assertEqual(mv["total_specifications"], 9)

    def test_stopping_outcome_is_persisted(self):
        payload = self._get(self._run(_positive_df(), "Why did cost_metric surge across datacenter_region?"))
        self.assertTrue(payload["stopping_criteria_met"])
        self.assertTrue(payload["stopping_reason"])

    # --- the workflow ------------------------------------------------------
    def test_positive_finding_asks_little_of_the_human_and_can_be_verified(self):
        inv_id = self._run(_positive_df(), "Why did cost_metric surge across datacenter_region?")
        state = api.get_verification_packet(inv_id, current_user=self.user, db=self.db)
        pk = state["packet"]
        self.assertEqual(pk["machine_claim"]["verdict"], "DIAGNOSED")
        self.assertEqual(pk["summary"]["blockers"], 0)
        self.assertGreaterEqual(pk["summary"]["machine_checked_and_passed"], 4)     # most of the work is machine-checked
        self.assertLessEqual(pk["summary"]["decisions_required"], 5)                # ...and the human's list is short
        self.assertTrue(_by_id(pk)["EVID-VERIFIED"]["reproduce"]["sql"].startswith("SELECT"))
        self.assertEqual(state["review"]["state"], "NOT_STARTED")

        with self.assertRaises(HTTPException) as cm:                                # cannot rubber-stamp
            api.sign_off_verification(inv_id, api.VerificationSignoffRequest(outcome="VERIFIED"), current_user=self.user, db=self.db)
        self.assertEqual(cm.exception.status_code, 422)

        for iid in state["review"]["required_item_ids"]:
            state = api.record_verification_decision(
                inv_id, api.VerificationDecisionRequest(item_id=iid, decision="CONFIRMED", comment="Checked; matches what I asked for."),
                current_user=self.user, db=self.db)
        self.assertEqual(state["review"]["state"], "READY_TO_SIGN")
        state = api.sign_off_verification(inv_id, api.VerificationSignoffRequest(outcome="VERIFIED", comment="Reproduced the top figure."),
                                          current_user=self.user, db=self.db)
        self.assertEqual(state["signoff"]["outcome"], "VERIFIED")
        self.assertFalse(state["signoff"]["stale"])
        self.assertEqual(state["signoff"]["reviewer"]["email"], self.user.email)
        self.assertEqual(state["signoff"]["manifest_hash"], self._get(inv_id)["reproducible_manifest_hash"])

    def test_refuted_finding_puts_the_refutation_in_front_of_the_human_as_a_blocker(self):
        inv_id = self._run(_simpsons_df(), "Why does recovery differ across drug?", "trial")
        state = api.get_verification_packet(inv_id, current_user=self.user, db=self.db)
        pk = state["packet"]
        self.assertEqual(pk["machine_claim"]["verdict"], "REFUTED")
        self.assertEqual(pk["items"][0]["priority"], BLOCKER)
        chal = _one(pk, "CHALLENGE-")
        self.assertEqual(chal["machine_status"], FAILED)
        self.assertIn("mediator", chal["you_confirm"])
        # The metric was aggregated by default: a score summed across rows.
        self.assertEqual(_by_id(pk)["METRIC-AGGREGATION"]["priority"], BLOCKER)

        # A bare confirm of a blocker is refused; a human who disagrees can reject and say so.
        with self.assertRaises(HTTPException) as cm:
            api.record_verification_decision(inv_id, api.VerificationDecisionRequest(item_id=chal["id"], decision="CONFIRMED", comment="ok"),
                                             current_user=self.user, db=self.db)
        self.assertEqual(cm.exception.status_code, 422)
        state = api.record_verification_decision(
            inv_id, api.VerificationDecisionRequest(item_id=chal["id"], decision="REJECTED", comment="Severity is caused by drug choice here: a mediator."),
            current_user=self.user, db=self.db)
        self.assertEqual(state["review"]["state"], "OBJECTIONS_RAISED")
        with self.assertRaises(HTTPException):
            api.sign_off_verification(inv_id, api.VerificationSignoffRequest(outcome="VERIFIED"), current_user=self.user, db=self.db)
        state = api.sign_off_verification(inv_id, api.VerificationSignoffRequest(outcome="REJECTED", comment="Refutation rejected: severity is a mediator."),
                                          current_user=self.user, db=self.db)
        self.assertEqual(state["signoff"]["outcome"], "REJECTED")

    def test_the_review_is_an_append_only_audit_trail(self):
        inv_id = self._run(_positive_df(), "Why did cost_metric surge across datacenter_region?")
        iid = "INTERP-QUESTION"
        for d, c in (("CONFIRMED", "Looks right to me."), ("NEEDS_REWORK", "Actually I meant the EU only.")):
            api.record_verification_decision(inv_id, api.VerificationDecisionRequest(item_id=iid, decision=d, comment=c),
                                             current_user=self.user, db=self.db)
        rows = self.db.query(InvestigationEvent).filter(InvestigationEvent.investigation_id == inv_id,
                                                        InvestigationEvent.event_type == "verification.item_reviewed").order_by(InvestigationEvent.sequence).all()
        self.assertEqual([r.event_payload_json["decision"] for r in rows], ["CONFIRMED", "NEEDS_REWORK"])   # nothing overwritten
        state = api.get_verification_packet(inv_id, current_user=self.user, db=self.db)
        self.assertEqual(state["review"]["per_item"][iid]["decision"], "NEEDS_REWORK")                      # latest wins

    def test_stale_client_view_is_refused_not_misattributed(self):
        inv_id = self._run(_positive_df(), "Why did cost_metric surge across datacenter_region?")
        with self.assertRaises(HTTPException) as cm:
            api.record_verification_decision(inv_id, api.VerificationDecisionRequest(item_id="INTERP-QUESTION", decision="CONFIRMED", item_fingerprint="stale"),
                                             current_user=self.user, db=self.db)
        self.assertEqual(cm.exception.status_code, 409)
        with self.assertRaises(HTTPException) as cm:
            api.sign_off_verification(inv_id, api.VerificationSignoffRequest(outcome="NEEDS_REWORK", comment="Need more data first.", packet_fingerprint="stale"),
                                      current_user=self.user, db=self.db)
        self.assertEqual(cm.exception.status_code, 409)

    def test_only_the_project_owner_can_verify_and_running_investigations_are_not_open(self):
        inv_id = self._run(_positive_df(), "Why did cost_metric surge across datacenter_region?")
        stranger = User(id=f"usr-{gen_uuid()[:8]}", email="x@y.z", hashed_password="pw", full_name="X", is_active=True)
        self.db.add(stranger); self.db.commit()
        with self.assertRaises(HTTPException) as cm:
            api.get_verification_packet(inv_id, current_user=stranger, db=self.db)
        self.assertIn(cm.exception.status_code, (403, 404))
        row = self.db.query(Investigation).filter(Investigation.id == inv_id).one()
        row.status = "RUNNING"; self.db.commit()
        with self.assertRaises(HTTPException) as cm:
            api.get_verification_packet(inv_id, current_user=self.user, db=self.db)
        self.assertEqual(cm.exception.status_code, 409)
