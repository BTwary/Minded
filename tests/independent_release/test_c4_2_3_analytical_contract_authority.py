"""v20-C4.2.3: Unified Analytical Identity + Durable Contract Authority.

Demonstrates, by test, that exactly one durable, deterministic analytical
contract governs execution, persistence, evidence, provenance and replay:

    proposal (compiler)  ->  reconciliation  ->  FINAL CONTRACT  ->  hypothesis
        ->  experiment  ->  evidence          (every stage keyed by identity)

Cases A-K follow the C4.2.3 specification.  Nothing here calls an AI service.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import subprocess
import sys
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import (
    Evidence, Experiment, Hypothesis, Investigation, InvestigationContract, InvestigationEvent,
)
from packages.analytics_core.src.engines.method_selection import METHOD_REGISTRY, MethodSelectionEngine
from packages.analytics_core.src.intelligence.analytical_identity import (
    AnalyticalIdentityError, FinalAnalyticalContract, canonical_serialize,
    hypothesis_identity_for, stamp_experiment_identity,
)
from packages.analytics_core.src.intelligence.contract_authority import (
    ContractImmutabilityError, ContractIntegrityError, finalize_contract, load_active_final_contract,
    load_final_contract, supersede_and_replan, trace_evidence_to_contract,
)
from packages.analytics_core.src.intelligence.experiment_contract_validation import (
    AnalyticalContractSnapshot, assert_single_primary_per_identity,
    detect_plan_final_contract_conflicts, validate_experiment_against_contract,
)
from packages.analytics_core.src.intelligence.experiment_synthesizer import ExperimentSynthesizer
from packages.analytics_core.src.intelligence.predictive_hypothesis import HypothesisSynthesizer
from packages.analytics_core.src.intelligence.semantic_binding_builder import build_binding_set
from packages.analytics_core.src.intelligence.semantic_resolution_builder import (
    build_canonical_semantic_resolution,
)
from tests.independent_release.test_c4_2_2e_multi_predictor_identity import _compile_and_decide, _multi_df
from tests.independent_release.test_defect_015_forensic_completion import _run_forensic_investigation

FP = {"data_table": "fp-c423-A"}
Q_SINGLE = "Is price associated with annual_sales?"
Q_MULTI = "Are price and marketing_spend associated with annual_sales?"
Q_MULTI_REVERSED = "Are marketing_spend and price associated with annual_sales?"
Q_TRIPLE = "Are price, marketing_spend, and discount associated with annual_sales?"
Q_MIXED = "Are price and region associated with annual_sales?"


# ---------------------------------------------------------------- helpers ---
def _context(question: str, fingerprints=None):
    """Run the real pipeline exactly as the controller does, up to the FINAL contract."""
    df = _multi_df()
    intent, semantic, plan, roles, decision = _compile_and_decide(question, df)
    decision = MethodSelectionEngine.select_for_plan(
        decision, MethodSelectionEngine.canonical_plan_view(decision, plan), semantic, None, df,
    )
    est = decision.estimand
    bindings = build_binding_set(semantic, plan.task, explanatory_columns=list(est.predictor_columns))
    final = FinalAnalyticalContract.from_method_decision(
        decision, dataset_fingerprints=fingerprints or FP, semantic_bindings=bindings,
    )
    return SimpleNamespace(
        q=question, df=df, intent=intent, semantic=semantic, plan=plan, roles=roles,
        decision=decision, bindings=bindings, final=final,
    )


def _synthesize(ctx):
    """Synthesize hypotheses + experiments and stamp identity exactly as the controller does."""
    canonical = build_canonical_semantic_resolution(ctx.semantic, "CORRELATION", question_roles=ctx.roles)
    hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
        ctx.semantic, ctx.q, intent=ctx.intent, canonical_semantics=canonical, decision=ctx.decision,
    )
    exps = ExperimentSynthesizer.synthesize_candidate_experiments(hyps, ctx.semantic, decision=ctx.decision)
    for h in hyps:
        h.analytical_identity = hypothesis_identity_for(ctx.final, h)
    by_code = {h.hypothesis_code: h.analytical_identity for h in hyps}
    for e in exps:
        stamp_experiment_identity(ctx.final, e, by_code)
    return hyps, exps


def _snapshot(ctx, **overrides):
    args = dict(
        contract_id="c-test", contract_version=1, problem_class=ctx.final.canonical_task,
        method_family=ctx.decision.method_family, selected_method=ctx.final.selected_method_code,
        estimand=METHOD_REGISTRY[ctx.final.selected_method_code].estimand,
        semantic_bindings=ctx.bindings, final_contract=ctx.final,
    )
    args.update(overrides)
    return AnalyticalContractSnapshot(**args)


def _summary(question: str) -> dict:
    """Everything downstream that must be reproducible.  Also used by subprocess replay."""
    ctx = _context(question)
    hyps, exps = _synthesize(ctx)
    return {
        "analytical_identity": ctx.final.analytical_identity,
        "estimand_identity": ctx.final.estimand_identity,
        "pairs": ctx.final.pair_identities(),
        "predictors": list(ctx.final.predictor_columns),
        "method": ctx.final.selected_method_code,
        "ceiling": ctx.final.claim_ceiling,
        "regime": ctx.final.verification_regime,
        "hyps": [[h.hypothesis_code, h.analytical_identity, bool(h.is_counter_hypothesis)] for h in hyps],
        "exps": [[e.code, e.experiment_role, e.analytical_identity, e.hypothesis_analytical_identity] for e in exps],
    }


def _new_investigation(db) -> Investigation:
    inv = Investigation(id=f"INV-C423-{uuid.uuid4().hex[:10]}", project_id="proj-default", question="q", status="PLANNED")
    db.add(inv)
    db.commit()
    return inv


def _new_contract(db, inv, version=1, parent=None, active=True) -> InvestigationContract:
    c = InvestigationContract(
        investigation_id=inv.id, version=version, parent_contract_id=parent.id if parent else None,
        status="ACTIVE", original_question="q", problem_class="ASSOCIATION", claim_type="ASSOCIATION",
        target_json={"target": "annual_sales"}, explanatory_variables_json=["price"],
        candidate_methods_json=[{"code": "EXP-ASSOC-PRIMARY", "method": "correlation_or_regression"}],
        proposed_method_json={"code": "EXP-ASSOC-PRIMARY", "method": "correlation_or_regression"},
        semantic_bindings_json=[], execution_state_json={},
    )
    db.add(c)
    db.flush()
    if active:
        inv.active_contract_id = c.id
    db.commit()
    return c


def _cross_process(script: str, *args: str, hashseed: str) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONPATH=ROOT)
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script, *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


# ------------------------------------------------- identity primitives ---
class TestCanonicalSerialization(unittest.TestCase):
    def test_key_order_does_not_matter_and_format_is_pinned(self):
        self.assertEqual(canonical_serialize({"b": 1, "a": [2, 1]}), '{"a":[2,1],"b":1}')
        self.assertEqual(canonical_serialize({"a": [2, 1], "b": 1}), canonical_serialize({"b": 1, "a": [2, 1]}))

    def test_sets_are_ordered_deterministically(self):
        self.assertEqual(canonical_serialize({"s": {"z", "a", "m"}}), '{"s":["a","m","z"]}')

    def test_refuses_to_stringify_unknown_objects(self):
        """evidence_identity.canonical_json uses default=str, which would embed a memory address."""
        with self.assertRaises(AnalyticalIdentityError):
            canonical_serialize({"x": object()})

    def test_refuses_non_finite_floats(self):
        for bad in (math.nan, math.inf):
            with self.assertRaises(AnalyticalIdentityError):
                canonical_serialize({"x": bad})

    def test_refuses_non_string_keys(self):
        with self.assertRaises(AnalyticalIdentityError):
            canonical_serialize({1: "a"})


# -------------------------------------------------------------- Case A ---
class TestCaseA_SinglePredictor(unittest.TestCase):
    def test_one_target_one_predictor_one_estimand_one_identity(self):
        ctx = _context(Q_SINGLE)
        f = ctx.final
        self.assertEqual(f.target_column, "annual_sales")
        self.assertEqual(f.predictor_columns, ("price",))
        self.assertEqual(len(f.pair_identities()), 1)
        for ident in (f.analytical_identity, f.estimand_identity, f.pair_identity("price")):
            self.assertRegex(ident, r"^[0-9a-f]{64}$")
        self.assertEqual(f.selected_method_code, "association_numeric")
        self.assertEqual(f.verification_regime, "independent_correlation_recompute")

    def test_one_hypothesis_pair_one_primary_experiment_same_identity(self):
        ctx = _context(Q_SINGLE)
        hyps, exps = _synthesize(ctx)
        pair = ctx.final.pair_identity("price")
        non_counter = [h for h in hyps if not h.is_counter_hypothesis]
        self.assertEqual(len(non_counter), 1)
        # H1 and its counter-hypothesis H0 are the SAME analytical pair.
        self.assertEqual({h.analytical_identity for h in hyps}, {pair})
        self.assertEqual(len(exps), 1)
        e = exps[0]
        self.assertEqual(e.experiment_role, "PRIMARY")
        self.assertEqual(e.analytical_identity, pair)
        self.assertEqual(e.hypothesis_analytical_identity, pair)
        result = validate_experiment_against_contract(e, _snapshot(ctx))
        self.assertTrue(result.compatible, result.errors)


# -------------------------------------------------------------- Case B ---
class TestCaseB_MultiPredictorPairwise(unittest.TestCase):
    def test_each_predictor_is_an_independent_pair_no_joint_model(self):
        ctx = _context(Q_MULTI)
        f = ctx.final
        self.assertEqual(f.predictor_columns, ("marketing_spend", "price"))
        pairs = f.pair_identities()
        self.assertEqual(set(pairs), {"price", "marketing_spend"})
        self.assertEqual(len(set(pairs.values())), 2)
        hyps, exps = _synthesize(ctx)
        non_counter = [h for h in hyps if not h.is_counter_hypothesis]
        self.assertEqual(sorted(h.secondary_metric for h in non_counter), ["marketing_spend", "price"])
        self.assertEqual(len(exps), 2)  # no dropped predictor
        for h in non_counter:
            self.assertEqual(h.analytical_identity, pairs[h.secondary_metric])
        for e in exps:
            (pred,) = [p for p in e.predictors or e.metrics[1:]]
            self.assertEqual(e.analytical_identity, pairs[pred])
            self.assertEqual(e.hypothesis_analytical_identity, e.analytical_identity)
            self.assertTrue(validate_experiment_against_contract(e, _snapshot(ctx)).compatible)
        assert_single_primary_per_identity(exps)
        self.assertEqual(ctx.final.selected_method_code, "association_numeric")  # pairwise, not joint

    def test_pair_identity_is_independent_of_the_other_predictors(self):
        single = _context(Q_SINGLE).final.pair_identity("price")
        multi = _context(Q_MULTI).final.pair_identity("price")
        triple = _context(Q_TRIPLE).final.pair_identity("price")
        self.assertEqual(single, multi)
        self.assertEqual(single, triple)

    def test_predictor_order_in_the_question_never_changes_identity(self):
        a, b = _context(Q_MULTI).final, _context(Q_MULTI_REVERSED).final
        self.assertEqual(a.analytical_identity, b.analytical_identity)
        self.assertEqual(a.pair_identities(), b.pair_identities())
        self.assertEqual(a.estimand_identity, b.estimand_identity)

    def test_three_predictors_three_pairs_three_experiments(self):
        ctx = _context(Q_TRIPLE)
        hyps, exps = _synthesize(ctx)
        self.assertEqual(len(ctx.final.pair_identities()), 3)
        self.assertEqual(len(exps), 3)
        self.assertEqual(len({e.analytical_identity for e in exps}), 3)


# --------------------------------------------------------- Cases C, D, E ---
class TestCasesCDE_ValidationFailsClosed(unittest.TestCase):
    def setUp(self):
        self.ctx = _context(Q_MULTI)
        _, self.exps = _synthesize(self.ctx)
        self.price_exp = next(e for e in self.exps if "price" in e.metrics)
        self.mkt_exp = next(e for e in self.exps if "marketing_spend" in e.metrics)

    def _errors(self, exp, ctx=None):
        ctx = ctx or self.ctx
        res = validate_experiment_against_contract(exp, _snapshot(ctx))
        return res.compatible, res.errors

    def test_control_valid_experiments_pass(self):
        for e in self.exps:
            ok, errs = self._errors(e)
            self.assertTrue(ok, errs)

    def test_C_contract_price_experiment_marketing_spend_fails_closed(self):
        price_only = _context(Q_SINGLE)
        _, mkt_exps = _synthesize(_context(Q_MULTI))
        mkt = next(e for e in mkt_exps if "marketing_spend" in e.metrics)
        # Re-stamped against the price-only contract: no identity can be established.
        stamp_experiment_identity(price_only.final, mkt, {})
        ok, errs = self._errors(mkt, price_only)
        self.assertFalse(ok)
        self.assertTrue(any(e.startswith("ANALYTICAL_IDENTITY_MISSING") for e in errs), errs)

    def test_C_identity_of_the_wrong_predictor_is_rejected_not_substituted(self):
        # marketing_spend experiment carrying price's identity (forgery / bug).
        forged = dataclasses.replace(self.mkt_exp, analytical_identity=self.price_exp.analytical_identity)
        ok, errs = self._errors(forged)
        self.assertFalse(ok)
        self.assertTrue(any(e.startswith("ANALYTICAL_IDENTITY_MISMATCH") for e in errs), errs)

    def test_C_experiment_targets_a_different_pairs_hypothesis(self):
        crossed = dataclasses.replace(
            self.mkt_exp, hypothesis_analytical_identity=self.price_exp.analytical_identity,
        )
        ok, errs = self._errors(crossed)
        self.assertFalse(ok)
        self.assertTrue(any(e.startswith("HYPOTHESIS_IDENTITY_MISMATCH") for e in errs), errs)

    def test_C_predictor_outside_the_contract(self):
        outsider = dataclasses.replace(
            self.price_exp, predictors=["discount"], metrics=["annual_sales", "discount"],
        )
        ok, errs = self._errors(outsider)
        self.assertFalse(ok)
        self.assertTrue(any(e.startswith("PREDICTOR_MISMATCH") for e in errs), errs)

    def test_D_wrong_target_fails_closed(self):
        wrong = dataclasses.replace(self.price_exp, outcome="price", target_metric="price")
        ok, errs = self._errors(wrong)
        self.assertFalse(ok)
        self.assertTrue(any(e.startswith("TARGET_MISMATCH") for e in errs), errs)

    def test_E_wrong_method_fails_closed(self):
        wrong = dataclasses.replace(self.price_exp, method_code="comparison_group_effect")
        ok, errs = self._errors(wrong)
        self.assertFalse(ok)
        self.assertTrue(any(e.startswith("METHOD_MISMATCH") for e in errs), errs)

    def test_E_unknown_method_fails_closed(self):
        wrong = dataclasses.replace(self.price_exp, method_code="not_a_method")
        ok, errs = self._errors(wrong)
        self.assertFalse(ok)

    def test_snapshot_that_disagrees_with_final_contract_fails_closed(self):
        for override, tag in (
            ({"problem_class": "FORECAST"}, "FINAL_CONTRACT_PROBLEM_CLASS_DIVERGENCE"),
            ({"selected_method": "comparison_group_effect"}, "FINAL_CONTRACT_METHOD_DIVERGENCE"),
        ):
            res = validate_experiment_against_contract(self.price_exp, _snapshot(self.ctx, **override))
            self.assertFalse(res.compatible)
            self.assertTrue(any(e.startswith(tag) for e in res.errors), res.errors)

    def test_role_tagging_is_explicit_and_validated(self):
        unassigned = dataclasses.replace(self.price_exp, experiment_role="UNASSIGNED")
        ok, errs = self._errors(unassigned)
        self.assertFalse(ok)
        self.assertTrue(any(e.startswith("EXPERIMENT_ROLE_UNASSIGNED") for e in errs), errs)

    def test_a_non_primary_experiment_is_never_treated_as_the_primary_claim(self):
        """A supporting run must not satisfy a PRIMARY slot: retagging it PRIMARY without
        the hypothesis linkage fails, and duplicate PRIMARYs for one identity are refused."""
        supporting = dataclasses.replace(
            self.price_exp, experiment_role="SUPPORTING", hypothesis_analytical_identity="",
        )
        self.assertTrue(self._errors(supporting)[0])
        promoted = dataclasses.replace(supporting, experiment_role="PRIMARY")
        ok, errs = self._errors(promoted)
        self.assertFalse(ok)
        self.assertTrue(any(e.startswith("HYPOTHESIS_IDENTITY_MISSING") for e in errs), errs)
        duplicate = dataclasses.replace(self.price_exp, code="EXP-DUP")
        with self.assertRaises(ValueError):
            assert_single_primary_per_identity([self.price_exp, duplicate])

    def test_a_foreign_identity_on_a_non_primary_experiment_is_rejected(self):
        foreign = dataclasses.replace(
            self.price_exp, experiment_role="ADVERSARIAL", analytical_identity="0" * 64,
        )
        ok, errs = self._errors(foreign)
        self.assertFalse(ok)
        self.assertTrue(any(e.startswith("ANALYTICAL_IDENTITY_NOT_IN_CONTRACT") for e in errs), errs)


# -------------------------------------------------------------- Case F ---
class TestCaseF_PlanContractDisagreement(unittest.TestCase):
    def _mutated_plan(self, ctx, **semantic_overrides):
        plan = copy.deepcopy(ctx.plan)
        for k, v in semantic_overrides.items():
            object.__setattr__(plan.semantics, k, v)
        return plan

    def test_conflicting_plan_is_detected_and_final_contract_is_untouched(self):
        ctx = _context(Q_MULTI)
        before = (ctx.final.analytical_identity, ctx.final.target_column, ctx.final.predictor_columns)
        plan = self._mutated_plan(ctx, target_column="price", explanatory_columns=["discount"])
        object.__setattr__(plan, "task", "FORECAST")
        conflicts = detect_plan_final_contract_conflicts(plan, ctx.final)
        blocking = {c["field"] for c in conflicts if c["severity"] == "BLOCKING"}
        self.assertEqual(blocking, {"problem_class", "target_column", "predictor_columns"})
        self.assertEqual(before, (ctx.final.analytical_identity, ctx.final.target_column, ctx.final.predictor_columns))

    def test_the_final_contract_cannot_be_redefined_by_a_plan(self):
        ctx = _context(Q_MULTI)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.final.target_column = "price"

    def test_agreeing_plan_has_no_conflicts(self):
        ctx = _context(Q_MULTI)
        self.assertEqual(detect_plan_final_contract_conflicts(ctx.plan, ctx.final), [])

    def test_plan_naming_a_predictor_the_final_contract_lacks_is_recorded_not_ignored(self):
        """The legacy detector skipped the check whenever either predictor list was empty."""
        ctx = _context(Q_MULTI)
        final_without_predictors = dataclasses.replace(ctx.final, predictor_columns=())
        conflicts = detect_plan_final_contract_conflicts(ctx.plan, final_without_predictors)
        self.assertIn(("predictor_columns", "OVERRIDDEN"), [(c["field"], c["severity"]) for c in conflicts])

    def test_controller_refuses_to_execute_when_plan_and_final_disagree(self):
        """Integration: mutate plan.semantics AFTER canonical resolution, inside the real
        controller.  Execution must stop; the disagreement is durably recorded."""
        original = MethodSelectionEngine.select_for_plan.__func__

        def _select_then_corrupt_plan(cls, decision, plan, *args, **kwargs):
            out = original(cls, decision, plan, *args, **kwargs)
            object.__setattr__(plan.semantics, "target_column", "discount")  # stale/conflicting plan
            return out

        with mock.patch.object(MethodSelectionEngine, "select_for_plan", classmethod(_select_then_corrupt_plan)):
            inv = _run_forensic_investigation(_multi_df(), Q_SINGLE)
        db = SessionLocal()
        try:
            events = [e.event_type for e in db.query(InvestigationEvent).filter(InvestigationEvent.investigation_id == inv.id)]
            self.assertIn("investigation.plan_final_contract_conflict", events)
            self.assertIn("investigation.inconclusive_completion", events)
            self.assertEqual(db.query(Experiment).filter(Experiment.investigation_id == inv.id).count(), 0)
        finally:
            db.close()


# -------------------------------------------------------------- Case G ---
class TestCaseG_PersistenceRoundTrip(unittest.TestCase):
    def test_persist_reload_same_decision(self):
        ctx = _context(Q_MULTI)
        db = SessionLocal()
        inv = _new_investigation(db)
        c = _new_contract(db, inv)
        self.assertIsNone(c.selected_method_json, "selected method must not exist before finalization")
        finalize_contract(db, c.id, ctx.final)
        inv_id, c_id = inv.id, c.id
        db.close()

        db2 = SessionLocal()  # a fresh session: nothing is cached
        try:
            loaded = load_final_contract(db2, c_id)
            self.assertEqual(loaded, ctx.final)
            self.assertEqual(loaded.analytical_identity, ctx.final.analytical_identity)
            self.assertEqual(loaded.estimand_identity, ctx.final.estimand_identity)
            self.assertEqual(loaded.pair_identities(), ctx.final.pair_identities())
            row = db2.query(InvestigationContract).filter(InvestigationContract.id == c_id).first()
            self.assertEqual(row.selected_method_json["method_code"], "association_numeric")
            self.assertEqual(row.selected_method_json["analytical_identity"], ctx.final.analytical_identity)
            self.assertEqual(row.proposed_method_json["method"], "correlation_or_regression")
            self.assertNotEqual(row.proposed_method_json.get("method"), row.selected_method_json["method_code"])
            self.assertIsNotNone(row.finalized_at)
            self.assertEqual(load_active_final_contract(db2, inv_id), ctx.final)
        finally:
            db2.close()

    def test_finalize_is_idempotent_and_immutable(self):
        ctx = _context(Q_MULTI)
        other = _context(Q_SINGLE)
        db = SessionLocal()
        try:
            inv = _new_investigation(db)
            c = _new_contract(db, inv)
            a = finalize_contract(db, c.id, ctx.final)
            self.assertEqual(finalize_contract(db, c.id, ctx.final), a)
            with self.assertRaises(ContractImmutabilityError):
                finalize_contract(db, c.id, other.final)
            self.assertEqual(load_final_contract(db, c.id), ctx.final)
        finally:
            db.close()

    def test_unfinalized_contract_fails_closed(self):
        db = SessionLocal()
        try:
            c = _new_contract(db, _new_investigation(db))
            with self.assertRaises(ContractIntegrityError):
                load_final_contract(db, c.id)
        finally:
            db.close()

    def test_replay_in_a_new_process_reproduces_every_identity_without_ai(self):
        """Persist -> reload in a NEW process (different hash seed) -> reconstruct
        downstream identities -> identical.  No network / AI is touched."""
        ctx = _context(Q_MULTI)
        hyps, exps = _synthesize(ctx)
        db = SessionLocal()
        inv = _new_investigation(db)
        c = _new_contract(db, inv)
        finalize_contract(db, c.id, ctx.final)
        contract_id = c.id
        db.close()
        script = (
            "import json,sys\n"
            "from apps.api.src.core.database import SessionLocal\n"
            "from packages.analytics_core.src.intelligence.contract_authority import load_final_contract\n"
            "from tests.independent_release.test_c4_2_3_analytical_contract_authority import _summary\n"
            "f = load_final_contract(SessionLocal(), sys.argv[1])\n"
            "s = _summary(sys.argv[2])\n"
            "print(json.dumps({'db': {'id': f.analytical_identity, 'pairs': f.pair_identities(),\n"
            "  'method': f.selected_method_code, 'estimand': f.estimand_identity, 'ceiling': f.claim_ceiling,\n"
            "  'regime': f.verification_regime, 'target': f.target_column, 'preds': list(f.predictor_columns)},\n"
            "  'replayed': s}))\n"
        )
        out = _cross_process(script, contract_id, Q_MULTI, hashseed="424242")
        mine = _summary(Q_MULTI)
        self.assertEqual(out["db"]["id"], ctx.final.analytical_identity)
        self.assertEqual(out["db"]["pairs"], ctx.final.pair_identities())
        self.assertEqual(out["db"]["method"], "association_numeric")
        self.assertEqual(out["db"]["estimand"], ctx.final.estimand_identity)
        self.assertEqual(out["db"]["ceiling"], ctx.final.claim_ceiling)
        self.assertEqual(out["db"]["regime"], ctx.final.verification_regime)
        self.assertEqual(out["db"]["target"], "annual_sales")
        self.assertEqual(out["db"]["preds"], ["marketing_spend", "price"])
        self.assertEqual(out["replayed"], mine)  # hypotheses + experiments + identities
        self.assertEqual(out["replayed"]["analytical_identity"], out["db"]["id"])


# -------------------------------------------------------------- Case H ---
class TestCaseH_Replan(unittest.TestCase):
    def _executed_v1(self, db):
        ctx = _context(Q_SINGLE)  # price -> annual_sales
        inv = _new_investigation(db)
        v1 = _new_contract(db, inv)
        finalize_contract(db, v1.id, ctx.final)
        return ctx, inv, v1

    def test_replan_of_the_proposal_keeps_the_analytical_identity(self):
        db = SessionLocal()
        try:
            ctx, inv, v1 = self._executed_v1(db)
            new_id = supersede_and_replan(
                db, inv.id, replan_reason="evidence_changed",
                candidate_experiments=[{"code": "EXP-NEW", "method": "different_proposal"}],
                active_hypotheses=["HYP-01"],
            )
            db.expire_all()
            v1 = db.query(InvestigationContract).filter(InvestigationContract.id == v1.id).first()
            v2 = db.query(InvestigationContract).filter(InvestigationContract.id == new_id).first()
            self.assertEqual(v1.status, "SUPERSEDED")
            self.assertEqual((v2.version, v2.parent_contract_id), (2, v1.id))
            self.assertEqual(v2.analytical_identity, v1.analytical_identity)  # same claim -> same identity
            # The replanned candidate is a PROPOSAL; it never becomes the selected method.
            self.assertEqual(v2.proposed_method_json["code"], "EXP-NEW")
            self.assertEqual(v2.selected_method_json["method_code"], "association_numeric")
            self.assertEqual(load_final_contract(db, v2.id), ctx.final)
        finally:
            db.close()

    def test_material_change_creates_a_new_identity_and_leaves_the_old_one_stable(self):
        db = SessionLocal()
        try:
            ctx1, inv, v1 = self._executed_v1(db)
            old_identity = v1.analytical_identity
            old_json = copy.deepcopy(v1.final_contract_json)
            ctx2 = _context("Is marketing_spend associated with annual_sales?")  # predictor changed
            self.assertNotEqual(ctx1.final.analytical_identity, ctx2.final.analytical_identity)
            new_id = supersede_and_replan(
                db, inv.id, replan_reason="predictor_changed", candidate_experiments=[],
                active_hypotheses=[], new_final_contract=ctx2.final,
            )
            db.expire_all()
            v1 = db.query(InvestigationContract).filter(InvestigationContract.id == v1.id).first()
            v2 = db.query(InvestigationContract).filter(InvestigationContract.id == new_id).first()
            self.assertEqual(v1.analytical_identity, old_identity)      # old identity stable
            self.assertEqual(v1.final_contract_json, old_json)           # old claim untouched
            self.assertNotEqual(v2.analytical_identity, old_identity)    # new identity
            self.assertEqual(v2.analytical_identity, ctx2.final.analytical_identity)
            self.assertEqual(v2.version, 2)
            self.assertEqual(v2.explanatory_variables_json, ["marketing_spend"])
            self.assertIn("predictor_columns", v2.execution_state_json["replan_material_diff"])
            self.assertEqual(load_final_contract(db, v1.id), ctx1.final)
            self.assertEqual(load_final_contract(db, v2.id), ctx2.final)
            self.assertEqual(load_active_final_contract(db, inv.id), ctx2.final)
            # The distinct claims stay distinguishable in persistence.
            ids = {r.analytical_identity for r in db.query(InvestigationContract).filter(
                InvestigationContract.investigation_id == inv.id)}
            self.assertEqual(len(ids), 2)
        finally:
            db.close()

    def test_replan_never_inherits_the_legacy_competing_decision_key(self):
        db = SessionLocal()
        try:
            _, inv, v1 = self._executed_v1(db)
            v1.execution_state_json = {"reconciled_method_decision": {"selected_method_code": "STALE"}, "keep": 1}
            db.commit()
            new_id = supersede_and_replan(db, inv.id, replan_reason="r", candidate_experiments=[], active_hypotheses=[])
            v2 = db.query(InvestigationContract).filter(InvestigationContract.id == new_id).first()
            self.assertNotIn("reconciled_method_decision", v2.execution_state_json)
            self.assertEqual(v2.execution_state_json["keep"], 1)  # other durable state is still carried
        finally:
            db.close()


# -------------------------------------------------------------- Case I ---
class TestCaseI_EvidenceAttribution(unittest.TestCase):
    def _chain(self, db):
        ctx = _context(Q_MULTI)
        hyps, exps = _synthesize(ctx)
        inv = _new_investigation(db)
        c = _new_contract(db, inv)
        finalize_contract(db, c.id, ctx.final)
        h = next(h for h in hyps if h.secondary_metric == "price" and not h.is_counter_hypothesis)
        e = next(e for e in exps if "price" in e.metrics)
        hrow = Hypothesis(
            id=f"{inv.id}_{h.hypothesis_code}", investigation_id=inv.id, canonical_identity=h.canonical_identity,
            analytical_identity=h.analytical_identity, hypothesis_code=h.hypothesis_code, statement="narrative A",
        )
        db.add(hrow)
        db.flush()
        erow = Experiment(
            id=f"{inv.id}_{e.code}", investigation_id=inv.id, hypothesis_id=hrow.id, test_code=e.code,
            tool_name="t", experiment_role=e.experiment_role, analytical_identity=e.analytical_identity,
            hypothesis_canonical_identity=h.canonical_identity, contract_id=c.id,
        )
        db.add(erow)
        db.flush()
        ev = Evidence(
            id=f"EV_{erow.id}", investigation_id=inv.id, experiment_id=erow.id, hypothesis_id=hrow.id,
            statement="some free text", analytical_identity=e.analytical_identity,
        )
        db.add(ev)
        db.commit()
        return ctx, c, hrow, erow, ev

    def test_evidence_traces_to_experiment_hypothesis_identity_and_contract(self):
        db = SessionLocal()
        try:
            ctx, c, hrow, erow, ev = self._chain(db)
            trace = trace_evidence_to_contract(db, ev.id)
            self.assertEqual(trace["experiment_id"], erow.id)
            self.assertEqual(trace["hypothesis_id"], hrow.id)
            self.assertEqual(trace["contract_id"], c.id)
            self.assertEqual(trace["analytical_identity"], ctx.final.pair_identity("price"))
            self.assertEqual(trace["contract_analytical_identity"], ctx.final.analytical_identity)
        finally:
            db.close()

    def test_narrative_text_is_irrelevant_to_attribution(self):
        db = SessionLocal()
        try:
            _, _, hrow, _, ev = self._chain(db)
            ev.statement = "Evidence that marketing_spend drives annual_sales"  # misleading prose
            hrow.statement = "totally different narrative"
            db.commit()
            trace_evidence_to_contract(db, ev.id)  # still resolves: prose is never read
        finally:
            db.close()

    def test_identity_is_never_reconstructed_from_narrative(self):
        db = SessionLocal()
        try:
            _, _, _, _, ev = self._chain(db)
            ev.analytical_identity = None  # narrative still names price/annual_sales
            db.commit()
            with self.assertRaises(ContractIntegrityError):
                trace_evidence_to_contract(db, ev.id)
        finally:
            db.close()

    def test_every_broken_link_fails_closed(self):
        for mutate, label in (
            (lambda ctx, c, h, e, ev: setattr(ev, "analytical_identity", "f" * 64), "evidence identity"),
            (lambda ctx, c, h, e, ev: setattr(e, "analytical_identity", "f" * 64), "experiment identity"),
            (lambda ctx, c, h, e, ev: setattr(e, "contract_id", None), "experiment contract"),
            (lambda ctx, c, h, e, ev: setattr(h, "analytical_identity", ctx.final.pair_identity("marketing_spend")), "hypothesis pair"),
            (lambda ctx, c, h, e, ev: setattr(e, "hypothesis_canonical_identity", "0" * 64), "hypothesis canonical"),
        ):
            db = SessionLocal()
            try:
                ctx, c, h, e, ev = self._chain(db)
                mutate(ctx, c, h, e, ev)
                db.commit()
                with self.assertRaises(ContractIntegrityError, msg=label):
                    trace_evidence_to_contract(db, ev.id)
            finally:
                db.close()

    def test_a_corrupted_contract_breaks_the_evidence_chain(self):
        db = SessionLocal()
        try:
            _, c, _, _, ev = self._chain(db)
            row = db.query(InvestigationContract).filter(InvestigationContract.id == c.id).first()
            blob = copy.deepcopy(row.final_contract_json)
            blob["target_column"] = "price"
            row.final_contract_json = blob
            db.commit()
            with self.assertRaises(AnalyticalIdentityError):
                trace_evidence_to_contract(db, ev.id)
        finally:
            db.close()


# -------------------------------------------------------------- Case J ---
class TestCaseJ_InvalidPredictorIsolation(unittest.TestCase):
    def test_invalid_predictor_neither_suppresses_the_valid_one_nor_gets_an_identity(self):
        ctx = _context(Q_MIXED)  # region is categorical -> not a valid Pearson predictor
        self.assertEqual(ctx.final.predictor_columns, ("price",))
        self.assertEqual(set(ctx.final.pair_identities()), {"price"})
        with self.assertRaises(AnalyticalIdentityError):
            ctx.final.pair_identity("region")  # no identity can exist for an invalid predictor
        hyps, exps = _synthesize(ctx)
        self.assertEqual(len(exps), 1)
        self.assertEqual(exps[0].analytical_identity, ctx.final.pair_identity("price"))
        self.assertTrue(all(h.secondary_metric == "price" for h in hyps))
        self.assertTrue(validate_experiment_against_contract(exps[0], _snapshot(ctx)).compatible)

    def test_an_invalid_predictor_cannot_be_smuggled_in_as_a_valid_experiment(self):
        ctx = _context(Q_MIXED)
        _, exps = _synthesize(ctx)
        smuggled = dataclasses.replace(exps[0], predictors=["region"], metrics=["annual_sales", "region"])
        res = validate_experiment_against_contract(smuggled, _snapshot(ctx))
        self.assertFalse(res.compatible)


# -------------------------------------------------------------- Case K ---
class TestCaseK_DeterministicOrdering(unittest.TestCase):
    def test_repeated_execution_is_identical(self):
        first = _summary(Q_TRIPLE)
        for _ in range(3):
            self.assertEqual(_summary(Q_TRIPLE), first)
        self.assertEqual(first["predictors"], sorted(first["predictors"]))

    def test_identical_across_process_restarts_and_hash_seeds(self):
        script = (
            "import json,sys\n"
            "from tests.independent_release.test_c4_2_3_analytical_contract_authority import _summary\n"
            "print(json.dumps(_summary(sys.argv[1])))\n"
        )
        mine = _summary(Q_TRIPLE)
        for seed in ("0", "1", "987654"):
            self.assertEqual(_cross_process(script, Q_TRIPLE, hashseed=seed), mine, f"PYTHONHASHSEED={seed}")


# ------------------------------------------------- mutation attacks (§17) ---
class TestMutationAttacks(unittest.TestCase):
    def setUp(self):
        self.ctx = _context(Q_MULTI)
        self.final = self.ctx.final

    def test_final_contract_is_immutable_in_memory(self):
        for field, value in (
            ("target_column", "price"), ("predictor_columns", ("price",)), ("problem_class", "X"),
            ("selected_method_code", "comparison_group_effect"), ("claim_ceiling", "CAUSAL"),
            ("verification_regime", "none"), ("canonical_task", "FORECAST"), ("comparison_dimension", "region"),
        ):
            with self.assertRaises(dataclasses.FrozenInstanceError, msg=field):
                setattr(self.final, field, value)

    def test_every_tampered_persisted_field_is_detected_on_load(self):
        good = self.final.to_dict()
        tamperings = {
            "target": ("target_column", "price"),
            "predictor": ("predictor_columns", ["price"]),
            "problem_class": ("problem_class", "FORECASTING"),
            "canonical_task": ("canonical_task", "FORECAST"),
            "method": ("selected_method_code", "comparison_group_effect"),
            "claim_ceiling": ("claim_ceiling", "CAUSAL"),
            "verification_regime": ("verification_regime", "independent_group_recompute"),
            "estimand": ("comparison_dimension", "region"),
            "objective": ("objective", "root_cause"),
            "bindings": ("bindings", []),
            "dataset": ("dataset_identity", "0" * 64),
        }
        for label, (key, value) in tamperings.items():
            blob = copy.deepcopy(good)
            blob[key] = value
            with self.assertRaises(AnalyticalIdentityError, msg=label):
                FinalAnalyticalContract.from_dict(blob)

    def test_stored_identity_tampering_is_detected(self):
        blob = copy.deepcopy(self.final.to_dict())
        blob["analytical_identity"] = "0" * 64
        with self.assertRaises(AnalyticalIdentityError):
            FinalAnalyticalContract.from_dict(blob)
        blob = copy.deepcopy(self.final.to_dict())
        blob["pair_identities"]["price"] = blob["pair_identities"]["marketing_spend"]
        with self.assertRaises(AnalyticalIdentityError):
            FinalAnalyticalContract.from_dict(blob)

    def test_non_canonical_predictor_order_is_rejected(self):
        with self.assertRaises(AnalyticalIdentityError):
            dataclasses.replace(self.final, predictor_columns=("price", "marketing_spend"))

    def test_database_tampering_is_detected_on_load(self):
        db = SessionLocal()
        try:
            c = _new_contract(db, _new_investigation(db))
            finalize_contract(db, c.id, self.final)
            row = db.query(InvestigationContract).filter(InvestigationContract.id == c.id).first()
            # (1) competing selected method
            good_selected = copy.deepcopy(row.selected_method_json)
            row.selected_method_json = dict(good_selected, method_code="comparison_group_effect")
            db.commit()
            with self.assertRaises(ContractIntegrityError):
                load_final_contract(db, c.id)
            row.selected_method_json = good_selected
            # (2) identity column no longer matches the final contract
            row.analytical_identity = "0" * 64
            db.commit()
            with self.assertRaises(ContractIntegrityError):
                load_final_contract(db, c.id)
        finally:
            db.close()

    def test_fails_closed_when_the_dataset_cannot_be_identified(self):
        for empty in (None, {}):
            with self.assertRaises(AnalyticalIdentityError):
                FinalAnalyticalContract.from_method_decision(
                    self.ctx.decision, dataset_fingerprints=empty, semantic_bindings=self.ctx.bindings,
                )

    def test_different_dataset_is_a_different_analytical_claim(self):
        other = _context(Q_MULTI, fingerprints={"data_table": "fp-DIFFERENT"}).final
        self.assertNotEqual(other.analytical_identity, self.final.analytical_identity)
        self.assertNotEqual(other.pair_identity("price"), self.final.pair_identity("price"))

    def test_method_configuration_is_part_of_identity(self):
        """The registry has no version field; the configuration IS the version."""
        import packages.analytics_core.src.engines.method_selection as ms
        real = ms.METHOD_REGISTRY["association_numeric"]
        drifted = dataclasses.replace(real, uncertainty_method="something_else")
        persisted = self.final.to_dict()  # computed with the REAL configuration
        with mock.patch.dict(ms.METHOD_REGISTRY, {"association_numeric": drifted}):
            with self.assertRaises(AnalyticalIdentityError):  # persisted identity no longer re-derives
                FinalAnalyticalContract.from_dict(persisted)

    def test_explicit_no_method_state_is_identity_bearing(self):
        no_method = dataclasses.replace(self.final, selected_method_code=None, verification_regime=None)
        self.assertNotEqual(no_method.analytical_identity, self.final.analytical_identity)
        self.assertEqual(FinalAnalyticalContract.from_dict(no_method.to_dict()), no_method)
        with self.assertRaises(AnalyticalIdentityError):
            dataclasses.replace(self.final, verification_regime=None)  # a method requires a regime
        fb = dataclasses.replace(self.final, fallback_used=True)
        self.assertNotEqual(fb.analytical_identity, self.final.analytical_identity)


# ---------------------------------- real controller, end to end (A/B/I) ---
class TestControllerEndToEndIdentity(unittest.TestCase):
    """The identity chain, as actually persisted by InvestigationController."""

    def _assert_chain(self, inv, expected_predictors):
        db = SessionLocal()
        try:
            contracts = db.query(InvestigationContract).filter(
                InvestigationContract.investigation_id == inv.id).order_by(InvestigationContract.version).all()
            self.assertTrue(contracts)
            active = db.query(Investigation).filter(Investigation.id == inv.id).first().active_contract_id
            for c in contracts:
                final = load_final_contract(db, c.id)  # verifies integrity of EVERY version
                self.assertEqual(list(final.predictor_columns), sorted(expected_predictors))
                self.assertIsNotNone(c.proposed_method_json)
                self.assertNotIn("reconciled_method_decision", c.execution_state_json or {})
            identities = {c.analytical_identity for c in contracts}
            self.assertEqual(len(identities), 1, "replans of the same claim must keep one analytical identity")
            final = load_final_contract(db, active)

            exps = db.query(Experiment).filter(Experiment.investigation_id == inv.id).all()
            self.assertTrue(exps, "expected executed experiments")
            primaries = [e for e in exps if e.experiment_role == "PRIMARY"]
            self.assertEqual(len(primaries), len(expected_predictors))
            self.assertEqual({e.analytical_identity for e in primaries}, set(final.pair_identities().values()))
            self.assertTrue(all(e.contract_id in {c.id for c in contracts} for e in exps))
            self.assertTrue(all(e.experiment_role in {"PRIMARY", "SUPPORTING", "ADVERSARIAL", "VERIFICATION"} for e in exps))

            hyps = db.query(Hypothesis).filter(Hypothesis.investigation_id == inv.id).all()
            linked = {h.analytical_identity for h in hyps if h.analytical_identity}
            self.assertEqual(linked, set(final.pair_identities().values()))

            evidence = db.query(Evidence).filter(Evidence.investigation_id == inv.id).all()
            self.assertTrue(evidence)
            for ev in evidence:
                trace = trace_evidence_to_contract(db, ev.id)
                self.assertIn(trace["analytical_identity"], set(final.pair_identities().values()))
            return final, primaries, evidence
        finally:
            db.close()

    def test_single_predictor_investigation_persists_one_identity_chain(self):
        inv = _run_forensic_investigation(_multi_df(), Q_SINGLE)
        final, primaries, evidence = self._assert_chain(inv, ["price"])
        self.assertEqual(len(primaries), 1)

    def test_multi_predictor_investigation_persists_independent_pair_chains(self):
        inv = _run_forensic_investigation(_multi_df(), Q_MULTI)
        final, primaries, evidence = self._assert_chain(inv, ["price", "marketing_spend"])
        self.assertEqual(len({e.analytical_identity for e in primaries}), 2)
        by_pair = {}
        for ev in evidence:
            by_pair.setdefault(ev.analytical_identity, []).append(ev)
        self.assertEqual(set(by_pair), set(final.pair_identities().values()))  # no dropped predictor


if __name__ == "__main__":
    unittest.main()
