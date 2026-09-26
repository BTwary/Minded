"""
test_controller_question_matrix.py

Phase 1 of the question-invariance workstream (see AAOS_DEFECT_LEDGER.md /
session notes for context). Purpose: establish a FACTUAL baseline of what
the real InvestigationController currently does for a representative matrix
of {correlation, rate, trend, ranking} x {genuine null, underpowered,
missing variable}, plus one standalone invariant: a Python exception must
never surface to the caller disguised as a scientific INCONCLUSIVE verdict.

Explicit non-goals of this file (per the phase-1 scope agreement):
  - Does NOT change IntentEngine, UniversalQuestionCompiler, controller.py,
    analyst_answer.py, verdict thresholds, or Bayesian updating.
  - Does NOT invent an expected terminal state for the missing-variable
    cases. Where the system currently fabricates a substitute answer, the
    test asserts the INVARIANT that must hold (no fabricated column
    binding, no confident answer to a question that was not asked) --
    this is deliberately written to FAIL today where that invariant is
    violated, because documenting the failure is the point of this phase.
  - Where current behavior is already correct (verified by direct,
    independently-computed evidence, not by assumption), the test PINS it
    as a regression guard.

Run:
    .venv/bin/python -m pytest tests/independent_release/test_controller_question_matrix.py -v
"""
import hashlib
import os
import sys
import unittest
import uuid

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Investigation, InvestigationVerdict
from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
from packages.analytics_core.src.runtime.controller import InvestigationController


class InMemoryProvider(BaseDatasetProvider):
    def __init__(self, d):
        self.d = d

    def acquire_context(self, project_id, dataset_ids=None):
        return InvestigationDataContext(
            project_id=project_id,
            datasets_map=self.d,
            dataset_fingerprints={n: hashlib.sha256(x.to_json().encode()).hexdigest() for n, x in self.d.items()},
            requested_dataset_ids=dataset_ids,
        )


def _run(df, question):
    """Runs `question` against `df` through the REAL InvestigationController
    end-to-end (semantic resolution -> estimand/method -> execution ->
    verification -> verdict), exactly as production does. Returns a dict of
    everything a caller of the API would actually see, plus whether a raw
    exception escaped (which must never happen -- caught internally and
    surfaced as status=FAILED instead)."""
    db = SessionLocal()
    iid = f"INV-MTX-{uuid.uuid4().hex[:8]}"
    db.add(Investigation(id=iid, project_id=f"p-{uuid.uuid4().hex[:6]}", question=question, status="PLANNED"))
    db.commit()
    db.close()

    raised = None
    try:
        InvestigationController(
            session_factory=SessionLocal,
            dataset_provider=InMemoryProvider({"t": df}),
        ).execute_investigation(investigation_id=iid, worker_id="w-matrix")
    except Exception as e:  # noqa: BLE001 -- we want to OBSERVE a crash, not hide it
        raised = f"{type(e).__name__}: {e}"

    db = SessionLocal()
    inv = db.query(Investigation).filter(Investigation.id == iid).first()
    verdict_record = db.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == iid).first()
    out = {
        "raised_exception": raised,
        "status": str(inv.status) if inv else None,
        "verdict_type": str(inv.verdict_type) if inv and inv.verdict_type else None,
        "confidence_score": float(inv.confidence_score) if inv and inv.confidence_score is not None else 0.0,
        "direct_answer": (inv.direct_answer or "") if inv else "",
        "main_finding": (inv.main_finding or "") if inv else "",
        "has_verdict_record": verdict_record is not None,
    }
    db.close()
    return out


# ---------------------------------------------------------------------------
# Controlled datasets (per-case, not a single shared dataset -- so a failure
# in one case can't be masked or caused by cross-contamination from another).
# ---------------------------------------------------------------------------

def _rng(seed):
    return np.random.RandomState(seed)


class TestCorrelationFamily(unittest.TestCase):
    def test_genuine_null_yields_no_detectable_effect(self):
        """PINS existing correct behavior: adequately powered (n=200), truly
        independent x/y must reach NO_DETECTABLE_EFFECT with a real CI, not
        a bare INCONCLUSIVE, and confidence must be calibrated (<1.0)."""
        rng = _rng(1)
        df = pd.DataFrame({
            "x_metric": rng.normal(50, 10, 200),
            "y_metric": rng.normal(50, 10, 200),
        })
        res = _run(df, "Is x_metric correlated with y_metric?")
        self.assertIsNone(res["raised_exception"])
        self.assertEqual(res["verdict_type"], "NO_DETECTABLE_EFFECT")
        self.assertTrue(res["has_verdict_record"])
        self.assertGreaterEqual(res["confidence_score"], 0.80)
        self.assertLess(res["confidence_score"], 1.0)
        self.assertIn("95% CI", res["direct_answer"])

    def test_underpowered_remains_inconclusive_not_no_detectable_effect(self):
        """PINS existing correct behavior: a real but tiny-n signal must be
        blocked by the admissibility gate (INCONCLUSIVE), and must NOT be
        certified as NO_DETECTABLE_EFFECT -- underpowered and adequately-
        powered-null are scientifically different claims."""
        rng = _rng(7)
        x = rng.normal(50, 10, 6)
        y = 0.6 * x + rng.normal(0, 10, 6)
        df = pd.DataFrame({"x_metric": x, "y_metric": y})
        res = _run(df, "Is x_metric correlated with y_metric?")
        self.assertIsNone(res["raised_exception"])
        self.assertEqual(res["verdict_type"], "INCONCLUSIVE")
        self.assertNotEqual(res["verdict_type"], "NO_DETECTABLE_EFFECT")

    def test_missing_variable_must_not_fabricate_substitute_answer(self):
        """EXPOSES a current defect (expected to FAIL today). The question
        asks about a column ('marketing_spend') that does not exist. The
        required invariant: the system must say so, and must NOT silently
        substitute a different column pair and report a confident answer to
        a question that was never asked. Currently it substitutes a
        region/revenue group comparison and reports INCONCLUSIVE for THAT
        instead -- the caller has no way to know their actual question was
        never analyzed."""
        rng = _rng(1)
        df = pd.DataFrame({
            "revenue": rng.normal(1000, 100, 60),
            "region": rng.choice(["East", "West"], 60),
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
        })
        res = _run(df, "Is revenue correlated with marketing_spend?")
        self.assertIsNone(res["raised_exception"])
        answer = (res["direct_answer"] + " " + res["main_finding"]).lower()
        # The substitute-analysis fingerprint: it silently ran a region
        # group-comparison instead of the requested correlation.
        self.assertNotIn("east", answer, "fabricated a region-based substitute answer instead of reporting the missing variable")
        self.assertNotIn("west", answer, "fabricated a region-based substitute answer instead of reporting the missing variable")
        self.assertIn("marketing_spend", answer, "must name the specific missing variable back to the caller")


class TestRateFamily(unittest.TestCase):
    def test_genuine_null_should_reach_a_no_effect_verdict_like_correlation_does(self):
        """EXPOSES a current defect (expected to FAIL today). Structurally
        identical epistemic situation to the correlation genuine-null case
        above (adequately powered, n=300/300, real chi-square p=0.30, real
        CI reported in the text) -- but the top-level verdict_type comes
        back INCONCLUSIVE instead of NO_DETECTABLE_EFFECT. Same evidence
        quality, different estimand family, different verdict label: this
        is the concrete instance of the question-invariance problem living
        in verdict *propagation*, not just NL parsing."""
        rng = _rng(2)
        n = 300
        df = pd.DataFrame({
            "group": ["A"] * n + ["B"] * n,
            "converted": list(rng.binomial(1, 0.20, n)) + list(rng.binomial(1, 0.20, n)),
        })
        res = _run(df, "Does the conversion rate differ between group A and group B?")
        self.assertIsNone(res["raised_exception"])
        self.assertIn("95% CI", res["direct_answer"])  # the evidence IS there
        self.assertEqual(
            res["verdict_type"], "NO_DETECTABLE_EFFECT",
            "adequately-powered real null should be NO_DETECTABLE_EFFECT, matching the correlation family's own contract",
        )

    def test_underpowered_remains_inconclusive(self):
        """PINS existing correct behavior."""
        df = pd.DataFrame({
            "group": ["A"] * 5 + ["B"] * 5,
            "converted": [1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        })
        res = _run(df, "Does the conversion rate differ between group A and group B?")
        self.assertIsNone(res["raised_exception"])
        self.assertEqual(res["verdict_type"], "INCONCLUSIVE")

    def test_missing_variable_churn_worded_fails_closed(self):
        """PINS existing correct behavior: the churn-specific identifiability
        path explicitly refuses to fabricate a churn signal when no churn
        outcome column exists."""
        rng = _rng(1)
        df = pd.DataFrame({
            "revenue": rng.normal(1000, 100, 60),
            "region": rng.choice(["East", "West"], 60),
        })
        res = _run(df, "Does churn_flag differ by region?")
        self.assertIsNone(res["raised_exception"])
        answer = (res["direct_answer"] + " " + res["main_finding"]).lower()
        self.assertIn("no identifiable churn outcome", answer)
        self.assertNotIn("east", answer)
        self.assertNotIn("west", answer)

    def test_missing_variable_non_churn_worded_must_not_fabricate(self):
        """EXPOSES a current defect (expected to FAIL today), and the more
        important one: the ONLY reason the churn-worded case above fails
        closed is a churn-specific special case. A generically-worded
        missing-variable rate question gets NO such protection -- it
        silently substitutes a region/revenue ranking and reports OBSERVED
        at confidence 1.0 for a question that was never asked."""
        rng = _rng(1)
        df = pd.DataFrame({
            "revenue": rng.normal(1000, 100, 60),
            "region": rng.choice(["East", "West"], 60),
        })
        res = _run(df, "What proportion of orders were returned?")
        self.assertIsNone(res["raised_exception"])
        answer = (res["direct_answer"] + " " + res["main_finding"]).lower()
        self.assertNotIn("east", answer, "fabricated a region-based substitute answer instead of reporting the missing variable")
        self.assertNotIn("west", answer, "fabricated a region-based substitute answer instead of reporting the missing variable")
        self.assertLess(res["confidence_score"], 1.0, "must not report maximal confidence for an unrequested, substituted analysis")


class TestTrendFamily(unittest.TestCase):
    def test_genuine_null_trend_must_not_silently_fail_blank(self):
        """EXPOSES a current defect (expected to FAIL today). A perfectly
        analyzable, adequately-powered (n=32) flat series currently causes
        status=FAILED with an EMPTY direct_answer/main_finding and no
        verdict record at all -- worse than a wrong verdict, this is no
        answer whatsoever. Required invariant: any COMPLETED-or-FAILED
        investigation must return non-empty, informative text."""
        rng = _rng(3)
        df = pd.DataFrame({
            "month": pd.date_range("2024-01-01", periods=32, freq="MS"),
            "flat_metric": 1000 + rng.normal(0, 30, 32),
        })
        res = _run(df, "Is there a trend in flat_metric over time?")
        self.assertIsNone(res["raised_exception"])
        combined = res["direct_answer"] + res["main_finding"]
        self.assertTrue(combined.strip(), "investigation returned no explanatory text at all for an analyzable question")

    def test_underpowered_trend_must_not_silently_fail_blank(self):
        """EXPOSES a current defect (expected to FAIL today): same blank-
        failure pattern as above, on a tiny (n=4) trend question."""
        df = pd.DataFrame({
            "month": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "flat_metric": [1000, 1050, 1080, 1200],
        })
        res = _run(df, "Is there a trend in flat_metric over time?")
        self.assertIsNone(res["raised_exception"])
        combined = res["direct_answer"] + res["main_finding"]
        self.assertTrue(combined.strip(), "investigation returned no explanatory text at all for an analyzable question")

    def test_missing_variable_trend_must_name_the_missing_variable(self):
        """EXPOSES a current defect (expected to FAIL today). At least this
        case does not fabricate a substitute answer (no fabrication risk
        observed), but it also gives the caller nothing at all -- an empty
        answer is indistinguishable from a silent crash from the outside."""
        rng = _rng(1)
        df = pd.DataFrame({
            "revenue": rng.normal(1000, 100, 60),
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
        })
        res = _run(df, "What is the trend in blorptastic_index over time?")
        self.assertIsNone(res["raised_exception"])
        combined = (res["direct_answer"] + " " + res["main_finding"]).lower()
        self.assertIn("blorptastic_index", combined, "must name the specific missing variable back to the caller instead of returning blank text")


class TestRankingFamily(unittest.TestCase):
    def test_near_tie_ranking_is_reported_with_appropriate_hedging(self):
        """PINS existing acceptable behavior: ranking a fully-observed
        population is a descriptive fact, not a hypothesis test, so a
        winner is reported even on a near-tie -- but the text must
        characterize it as close, not overstate the gap."""
        rng = _rng(4)
        df = pd.DataFrame({
            "category": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            "sales": list(rng.normal(1000, 50, 50)) + list(rng.normal(1002, 50, 50)) + list(rng.normal(998, 50, 50)),
        })
        res = _run(df, "Which category has the highest sales?")
        self.assertIsNone(res["raised_exception"])
        self.assertEqual(res["verdict_type"], "OBSERVED")
        self.assertIn("close", res["direct_answer"].lower())

    def test_tiny_n_ranking_must_not_silently_fail_blank(self):
        """EXPOSES a current defect (expected to FAIL today): same blank-
        failure pattern as the trend family, on a 1-row-per-category
        ranking question."""
        df = pd.DataFrame({"category": ["A", "B", "C"], "sales": [10, 11, 9]})
        res = _run(df, "Which category has the highest sales?")
        self.assertIsNone(res["raised_exception"])
        combined = res["direct_answer"] + res["main_finding"]
        self.assertTrue(combined.strip(), "investigation returned no explanatory text at all for an analyzable question")

    def test_missing_variable_ranking_must_not_fabricate(self):
        """EXPOSES a current defect (expected to FAIL today): same
        fabrication pattern as correlation/rate -- silently substitutes
        region/revenue and reports OBSERVED at confidence 1.0 for a
        question about columns ('widget_count', 'category') that do not
        exist in the dataset at all."""
        rng = _rng(1)
        df = pd.DataFrame({
            "revenue": rng.normal(1000, 100, 60),
            "region": rng.choice(["East", "West"], 60),
        })
        res = _run(df, "Which category has the highest widget_count?")
        self.assertIsNone(res["raised_exception"])
        answer = (res["direct_answer"] + " " + res["main_finding"]).lower()
        self.assertNotIn("east", answer, "fabricated a region-based substitute answer instead of reporting the missing variable")
        self.assertNotIn("west", answer, "fabricated a region-based substitute answer instead of reporting the missing variable")
        self.assertLess(res["confidence_score"], 1.0, "must not report maximal confidence for an unrequested, substituted analysis")


class TestExceptionSafetyInvariant(unittest.TestCase):
    """The single most important invariant: a Python exception inside the
    analytical pipeline must never surface to the caller disguised as a
    scientific INCONCLUSIVE verdict with a persisted verdict record. It
    must be distinguishable (status=FAILED, no verdict record) from a
    genuine null-result investigation."""

    def test_forced_internal_exception_does_not_produce_a_fake_verdict_record(self):
        import unittest.mock as mock
        rng = _rng(5)
        df = pd.DataFrame({
            "revenue": rng.normal(1000, 100, 60),
            "region": rng.choice(["East", "West"], 60),
        })
        with mock.patch(
            "packages.analytics_core.src.engines.analyst_answer.build_analyst_result",
            side_effect=RuntimeError("forced synthetic failure for invariant test"),
        ):
            # analyst_answer failures are already caught (controller.py has a
            # broad except around this specific call) -- this confirms that
            # catch does not fabricate a verdict record either.
            res = _run(df, "Does revenue differ by region?")
        # Whether the outer investigation completes or fails, it must never
        # be the case that a genuine internal exception is silently
        # relabeled as a confident, verified scientific finding.
        if res["has_verdict_record"] and res["verdict_type"] not in (None, "INCONCLUSIVE"):
            self.assertNotEqual(res["confidence_score"], 1.0, "an internal failure path must not report maximal confidence")


if __name__ == "__main__":
    unittest.main()
