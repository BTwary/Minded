import sys, os
sys.path.insert(0, os.path.abspath("."))
import traceback
from apps.api.src.core.database import SessionLocal, init_db
from apps.api.src.services.analysis_service import AnalysisService
from packages.schemas.src.analysis import AnalysisCreate

init_db()
db = SessionLocal()
svc = AnalysisService(db)
req = AnalysisCreate(question="Why did revenue fall in March?", project_id="proj-default")

try:
    res = svc.execute_analysis(req)
    print("SUCCESS!")
    print("Direct Answer:", res.direct_answer)
    print("Grain:", res.discovery.grain if res.discovery else None)
    print("Data Quality Score:", res.audit.data_quality_score if res.audit else None)
    print("Hypotheses count:", len(res.hypotheses))
    for h in res.hypotheses:
        extra = h.reason_for_rejection or (h.confirmed_findings[0] if h.confirmed_findings else "")
        print(f" - {h.id}: {h.status.upper()} -> {extra}")
    print("Scenarios count:", len(res.what_if_scenarios or []))
    for sc in res.what_if_scenarios or []:
        print(f" - {sc['scenario']}: {sc['projected_impact']}")
    print("Evidence graph nodes:", len(res.evidence_graph or []))
    print("Confidence breakdown:", res.confidence_breakdown)
except Exception as e:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    db.close()
