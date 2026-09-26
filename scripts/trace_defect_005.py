"""Reproduction / tracing harness for DEFECT-005.

Runs each of the 4 affected workflows (plus root-cause as a healthy control)
through the REAL AnalysisService -> InvestigationController path and dumps
every stage of the scientific loop: hypotheses, predictions, experiments,
evidence, verification, posterior, stopping reason, verdict.

Does not assert / does not stop on failure - this is a tracer, not a test.
"""
import os
import sys
import json
import traceback

sys.path.insert(0, os.path.abspath("."))

from apps.api.src.core.database import SessionLocal
from apps.api.src.services.analysis_service import AnalysisService
from apps.api.src.models.entities import Investigation, Hypothesis, Evidence, InvestigationVerdict, InvestigationExecution
from packages.schemas.src.analysis import AnalysisCreate

CASES = [
    ("root_cause", "Why did revenue fall in March?"),
    ("correlation", "What are the correlation drivers of revenue?"),
    ("forecast", "Forecast next month's revenue trajectory"),
    ("churn", "Which customers are likely to churn?"),
    ("segmentation", "Which products are performing badly?"),
]

out = {}

for key, question in CASES:
    db = SessionLocal()
    svc = AnalysisService(db)
    entry = {"question": question}
    try:
        req = AnalysisCreate(question=question, project_id="proj-default")
        res = svc.execute_analysis(req, user_id=None)
        entry["verdict"] = str(res.verdict)
        entry["confidence"] = res.confidence_score if hasattr(res, "confidence_score") else None
        entry["direct_answer"] = res.direct_answer
        entry["main_finding"] = res.main_finding
        entry["num_steps"] = len(res.steps)
        entry["hypotheses"] = [
            {"id": h.id, "statement": h.statement, "status": h.status, "posterior": h.priority}
            for h in res.hypotheses
        ]
        entry["evidence"] = [
            {"id": e.id, "statement": e.statement, "status": str(e.validation_status)}
            for e in res.evidence
        ]
        # Pull raw DB rows too for stopping_rationale / entropy / per-step SQL
        inv = db.query(Investigation).filter(Investigation.question == question).order_by(Investigation.created_at.desc()).first()
        if inv:
            entry["stopping_rationale"] = inv.stopping_rationale
            entry["entropy_initial"] = inv.entropy_initial
            entry["entropy_current"] = inv.entropy_current
            entry["stopping_criteria_met"] = inv.stopping_criteria_met
            entry["investigation_status"] = inv.status
            execs = db.query(InvestigationExecution).filter(InvestigationExecution.investigation_id == inv.id).all()
            entry["executions"] = [
                {"status": ex.status, "error_code": ex.error_code, "error_message": ex.error_message}
                for ex in execs
            ]
    except Exception as e:
        entry["exception"] = str(e)
        entry["traceback"] = traceback.format_exc()
    finally:
        db.close()
    out[key] = entry
    print(f"=== {key} ===")
    print(json.dumps(entry, indent=2, default=str)[:4000])
    print()

with open("DEFECT_005_TRACE_RAW.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
