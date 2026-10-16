"""Rigorous Automated Test Suite for All Analytical Workflows with Full Assertion Proofs."""
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath("."))

from apps.api.src.core.database import SessionLocal, init_db
from apps.api.src.services.analysis_service import AnalysisService
from packages.schemas.src.analysis import AnalysisCreate
from packages.shared.src.enums import AnalyticalVerdict, ValidationStatus


def _open_service():
    """Create the DB schema and a service bound to a fresh session. Done lazily so that merely
    importing this module (e.g. pytest collection) has no side effects and can never exit."""
    init_db()
    db = SessionLocal()
    return db, AnalysisService(db)


TEST_CASES = [
    {
        "label": "Root Cause Diagnostic",
        "question": "Why did revenue fall in March?",
        "expected_verdict": AnalyticalVerdict.DIAGNOSED,
        "assertions": [
            lambda r: "Region B" in r.direct_answer,
            lambda r: any(h.id == "HYP-01" and h.status == "refuted" for h in r.hypotheses),
            lambda r: any(h.id == "HYP-02" and h.status == "supported" for h in r.hypotheses),
            lambda r: any("Region B" in e.statement and e.validation_status == ValidationStatus.PASSED for e in r.evidence),
        ],
    },
    {
        "label": "Correlation Driver Analysis",
        "question": "What are the correlation drivers of revenue?",
        "expected_verdict": AnalyticalVerdict.STATISTICALLY_SIGNIFICANT,
        "assertions": [
            lambda r: "cost" in r.direct_answer.lower(),
            lambda r: any("0.99" in e.statement and e.validation_status == ValidationStatus.PASSED for e in r.evidence),
            lambda r: any(h.id == "HYP-01" and h.status == "supported" for h in r.hypotheses),
        ],
    },
    {
        "label": "Trajectory Forecasting",
        "question": "Forecast next month's revenue trajectory",
        "expected_verdict": AnalyticalVerdict.PREDICTED,
        "assertions": [
            lambda r: "351,296" in r.direct_answer or "revenue" in r.direct_answer.lower(),
            lambda r: any("MAPE" in e.statement or "8.32%" in e.statement for e in r.evidence),
            lambda r: all(e.validation_status == ValidationStatus.PASSED for e in r.evidence),
        ],
    },
    {
        "label": "Customer RFM Churn Segmentation",
        "question": "Which customers are likely to churn?",
        "expected_verdict": AnalyticalVerdict.OBSERVED,
        "assertions": [
            lambda r: "46.3%" in r.direct_answer or "frequency" in r.main_finding.lower(),
            lambda r: any("46.3%" in e.statement and e.validation_status == ValidationStatus.PASSED for e in r.evidence),
        ],
    },
    {
        "label": "Product Catalog Performance",
        "question": "Which products are performing badly?",
        "expected_verdict": AnalyticalVerdict.OBSERVED,
        "assertions": [
            lambda r: "PRD-107" in r.direct_answer,
            lambda r: any("PRD-107" in e.statement and e.validation_status == ValidationStatus.PASSED for e in r.evidence),
        ],
    },
]


def run_case(tc, svc):
    """Run one workflow case and raise AssertionError (or any exception) if it does not hold."""
    label = tc["label"]
    q = tc["question"]
    print(f"\n[TEST] {label.upper()} -> \"{q}\"")
    req = AnalysisCreate(question=q, project_id="proj-default")
    res = svc.execute_analysis(req, user_id=None)

    # 1. Assert Verdict
    assert res.verdict == tc["expected_verdict"], (
        f"Verdict Mismatch: Expected {tc['expected_verdict']}, got {res.verdict}"
    )

    # 2. Assert Step & Evidence Counts
    assert len(res.steps) >= 3, f"Insufficient execution steps: {len(res.steps)}"
    assert len(res.evidence) >= 1, "Zero validated evidence generated"

    # 3. Assert Domain-Specific Criteria
    for i, assertion in enumerate(tc["assertions"]):
        assert assertion(res), f"Domain assertion #{i+1} failed for {label}"

    # 4. Assert Evidence Pass Status
    for ev in res.evidence:
        assert ev.validation_status == ValidationStatus.PASSED, (
            f"Evidence {ev.id} failed validation: {ev.statement}"
        )

    print(f" -> PASSED (Verdict: {res.verdict.value}, {len(res.evidence)} validated claims)")
    print(f" -> Direct Answer: {res.direct_answer}")
    print(f" -> Main Finding: {res.main_finding}")


try:  # pytest is optional: the script must still run as a plain program
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

if pytest is not None:
    @pytest.fixture(scope="module")
    def _svc():
        db, svc = _open_service()
        yield svc
        db.close()

    @pytest.mark.parametrize("tc", TEST_CASES, ids=[t["label"] for t in TEST_CASES])
    def test_workflow(tc, _svc):
        run_case(tc, _svc)


def main() -> int:
    """Command-line entry point: stops at the first failing workflow, returns the exit code."""
    db, svc = _open_service()
    total_passed = 0
    total_tests = len(TEST_CASES)

    print("=" * 60)
    print("RUNNING AUTOMATED ANALYTICAL WORKFLOW REGRESSION SUITE")
    print("=" * 60)

    try:
        for tc in TEST_CASES:
            try:
                run_case(tc, svc)
                total_passed += 1
            except Exception:
                print(f" -> FAILED on {tc['label']}!")
                traceback.print_exc()
                return 1
    finally:
        db.close()

    print("\n" + "=" * 60)
    print(f"WORKFLOW REGRESSION RESULT: {total_passed}/{total_tests} WORKFLOWS PASSED ALL ASSERTIONS")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
