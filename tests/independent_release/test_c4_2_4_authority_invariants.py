"""v20-C4.2.4 sections 2 & 3: replan projection sync + finalize_contract
persistence-corruption hardening.

Case R (Replan projection sync):
    A replan that materially changes the analytical claim must regenerate
    every legacy proposal-era projection whose value is a function of that
    claim (estimand_json, time_window_json, semantic_bindings_json) rather
    than leaving it copied from the superseded contract -- and must never
    silently accept a materially-different binding set it cannot verify.

Case F (finalize_contract re-validation):
    An already-finalized contract must be fully re-derived and re-validated
    every time finalize_contract() is called for it (not accepted on
    identity-match alone), so a corrupted final_contract_json /
    analytical_identity / selected_method_json fails closed instead of
    being silently accepted.
"""
from __future__ import annotations

import copy
import os
import sys
import unittest
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Investigation, InvestigationContract
from packages.analytics_core.src.intelligence.analytical_identity import AnalyticalIdentityError
from packages.analytics_core.src.intelligence.contract_authority import (
    ContractImmutabilityError, ContractIntegrityError, finalize_contract, load_final_contract,
    supersede_and_replan,
)

from tests.independent_release.test_c4_2_3_analytical_contract_authority import (
    FP, Q_MULTI, Q_MULTI_REVERSED, Q_SINGLE, _context, _new_contract, _new_investigation,
)


def _finalized_v1(db):
    ctx = _context(Q_SINGLE)
    inv = _new_investigation(db)
    c = _new_contract(db, inv)
    finalize_contract(db, c.id, ctx.final)
    db.expire_all()
    c = db.query(InvestigationContract).filter(InvestigationContract.id == c.id).first()
    return ctx, inv, c


class TestCaseR_ReplanProjectionSync(unittest.TestCase):
    def test_estimand_and_time_window_projections_are_regenerated_not_copied(self):
        db = SessionLocal()
        try:
            ctx1, inv, v1 = _finalized_v1(db)
            stale_estimand = copy.deepcopy(v1.estimand_json)
            ctx2 = _context(Q_MULTI)  # different target/predictors -> material change
            self.assertNotEqual(ctx1.final.analytical_identity, ctx2.final.analytical_identity)
            new_id = supersede_and_replan(
                db, inv.id, replan_reason="target_and_predictors_changed",
                candidate_experiments=[], active_hypotheses=[], new_final_contract=ctx2.final,
            )
            v2 = db.query(InvestigationContract).filter(InvestigationContract.id == new_id).first()
            # The old estimand must not survive verbatim in the new row.
            self.assertNotEqual(v2.estimand_json, stale_estimand)
            self.assertEqual(v2.estimand_json.get("target_column"), ctx2.final.target_column)
            self.assertEqual(
                sorted(v2.estimand_json.get("predictor_columns") or []),
                sorted(ctx2.final.predictor_columns),
            )
            self.assertEqual(v2.time_window_json.get("time_column"), ctx2.final.time_column)
        finally:
            db.close()

    def test_unchanged_claim_leaves_projections_byte_identical(self):
        db = SessionLocal()
        try:
            ctx1, inv, v1 = _finalized_v1(db)
            old_estimand = copy.deepcopy(v1.estimand_json)
            old_bindings = copy.deepcopy(v1.semantic_bindings_json)
            new_id = supersede_and_replan(
                db, inv.id, replan_reason="proposal_only_change",
                candidate_experiments=[{"code": "X"}], active_hypotheses=[],
                new_final_contract=ctx1.final,  # same analytical identity
            )
            v2 = db.query(InvestigationContract).filter(InvestigationContract.id == new_id).first()
            self.assertEqual(v2.analytical_identity, v1.analytical_identity)
            self.assertEqual(v2.estimand_json, old_estimand)
            self.assertEqual(v2.semantic_bindings_json, old_bindings)
        finally:
            db.close()

    def test_bindings_change_does_not_carry_stale_bindings_forward(self):
        db = SessionLocal()
        try:
            ctx1, inv, v1 = _finalized_v1(db)
            stale_bindings = copy.deepcopy(v1.semantic_bindings_json)
            ctx2 = _context(Q_MULTI_REVERSED)
            new_id = supersede_and_replan(
                db, inv.id, replan_reason="predictor_changed",
                candidate_experiments=[], active_hypotheses=[], new_final_contract=ctx2.final,
            )
            v2 = db.query(InvestigationContract).filter(InvestigationContract.id == new_id).first()
            self.assertNotEqual(v2.semantic_bindings_json, stale_bindings)
            new_cols = {b["column"] for b in v2.semantic_bindings_json}
            self.assertTrue(set(ctx2.final.predictor_columns) <= new_cols | {ctx2.final.target_column})
        finally:
            db.close()

    def test_full_fidelity_binding_set_is_used_when_caller_supplies_one(self):
        db = SessionLocal()
        try:
            ctx1, inv, v1 = _finalized_v1(db)
            ctx2 = _context(Q_MULTI)
            new_id = supersede_and_replan(
                db, inv.id, replan_reason="predictor_changed", candidate_experiments=[],
                active_hypotheses=[], new_final_contract=ctx2.final,
                new_semantic_binding_set=ctx2.bindings,
            )
            v2 = db.query(InvestigationContract).filter(InvestigationContract.id == new_id).first()
            self.assertEqual(v2.semantic_bindings_json, ctx2.bindings.to_dict_list())
            self.assertTrue(all(not b.get("_reduced_projection") for b in v2.semantic_bindings_json))
        finally:
            db.close()


class TestCaseF_FinalizeRevalidation(unittest.TestCase):
    def test_corrupted_final_contract_json_fails_closed_on_refinalize(self):
        db = SessionLocal()
        try:
            ctx, inv, c = _finalized_v1(db)
            c.final_contract_json = {**c.final_contract_json, "target_column": "SOMETHING_ELSE"}
            db.commit()
            # any tamper to final_contract_json's own components is caught
            # by its own re-derivation (AnalyticalIdentityError), a stricter
            # fail-closed signal than the identity-column cross-check.
            with self.assertRaises(AnalyticalIdentityError):
                finalize_contract(db, c.id, ctx.final)
        finally:
            db.close()

    def test_corrupted_analytical_identity_column_fails_closed_on_refinalize(self):
        db = SessionLocal()
        try:
            ctx, inv, c = _finalized_v1(db)
            c.analytical_identity = "0" * 64
            db.commit()
            with self.assertRaises(ContractIntegrityError):
                finalize_contract(db, c.id, ctx.final)
        finally:
            db.close()

    def test_corrupted_selected_method_json_fails_closed_on_refinalize(self):
        db = SessionLocal()
        try:
            ctx, inv, c = _finalized_v1(db)
            c.selected_method_json = {**(c.selected_method_json or {}), "method_code": "TAMPERED"}
            db.commit()
            with self.assertRaises(ContractIntegrityError):
                finalize_contract(db, c.id, ctx.final)
        finally:
            db.close()

    def test_one_identity_component_mutated_fails_closed(self):
        db = SessionLocal()
        try:
            ctx, inv, c = _finalized_v1(db)
            tampered = {**c.final_contract_json, "claim_ceiling": "STRUCTURAL_CAUSAL"}
            c.final_contract_json = tampered
            db.commit()
            with self.assertRaises(AnalyticalIdentityError):
                finalize_contract(db, c.id, ctx.final)
        finally:
            db.close()

    def test_restoring_one_field_but_leaving_another_inconsistent_still_fails(self):
        db = SessionLocal()
        try:
            ctx, inv, c = _finalized_v1(db)
            original = copy.deepcopy(c.final_contract_json)
            tampered = {**c.final_contract_json, "target_column": "WRONG", "claim_ceiling": "WRONG"}
            c.final_contract_json = tampered
            db.commit()
            # "restore" only one of the two corrupted fields
            c.final_contract_json = {**tampered, "target_column": original["target_column"]}
            db.commit()
            with self.assertRaises(AnalyticalIdentityError):
                finalize_contract(db, c.id, ctx.final)
        finally:
            db.close()

    def test_uncorrupted_refinalize_of_identical_claim_still_succeeds(self):
        db = SessionLocal()
        try:
            ctx, inv, c = _finalized_v1(db)
            result = finalize_contract(db, c.id, ctx.final)  # idempotent path, nothing corrupted
            self.assertEqual(result, ctx.final.analytical_identity)
        finally:
            db.close()

    def test_materially_different_final_still_raises_immutability_not_silently_accepted(self):
        db = SessionLocal()
        try:
            ctx1, inv, c = _finalized_v1(db)
            ctx2 = _context(Q_MULTI)
            with self.assertRaises(ContractImmutabilityError):
                finalize_contract(db, c.id, ctx2.final)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
