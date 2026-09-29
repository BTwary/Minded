"""DEFECT-030 independent release test suite.

Regression coverage for a real-run bug found while investigating a report
that natural-language question parsing/analysis "doesn't work in real"
(BUGFIX_2026-09-26_session16_calculation_trace_validation.md).

`transition.py`'s `apply_post_execution_transition` set
`raw_obs.calculation_trace = trace_dict` -- a direct attribute assignment of
a plain dict onto a Pydantic `BaseModel` (`RawObservationRecord`) field typed
`Optional[CalculationTraceSchema]`. Pydantic does not validate/coerce on
plain attribute assignment unless `validate_assignment=True` is configured
(it is not, here), so the field silently held a raw dict instead of a
validated `CalculationTraceSchema` instance.

`raw_obs` had already been appended by reference into the investigation's
runtime state (`state_mgr.record_raw_observation`) before this later
assignment, so the mistyped field was live in `runtime_state` for the rest
of the investigation. Every real investigation snapshots that runtime state
via `runtime_state.model_dump(mode="json")`
(`scientific_state_snapshot.build_scientific_state_snapshot`), and Pydantic
can only best-effort duck-type a raw dict where a `CalculationTraceSchema`
was declared -- emitting a `PydanticSerializationUnexpectedValue` warning on
*every* experiment that recorded evidence, i.e. on essentially every real
question asked against real data. This corrupts the very
audit/reproducibility snapshot this schema exists to guarantee, even though
it was invisible in most unit tests (which don't call `model_dump()` on the
full runtime-state snapshot).

Fix: validate into `CalculationTraceSchema` before assignment, matching the
pattern already used by `InvestigationStateManager.record_calculation_trace`
and `EvidenceLedger.record_claim`.

Run with: python -m unittest tests/independent_release/test_defect030_calculation_trace_validation.py
"""
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from apps.api.src.models.entities import Base, Organization, User, Project
from apps.api.src.services.dataset_service import DatasetService
from apps.api.src.services.analysis_service import AnalysisService
from packages.schemas.src.analysis import AnalysisCreate, CalculationTraceSchema


class TestDefect030CalculationTraceValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="aaos_defect030_")
        db_path = os.path.join(self.temp_dir, "defect030.db")
        self.engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db = Session()

        org = Organization(id="org-030", name="Org", slug="org-030")
        user = User(id="user-030", email="u030@example.com", hashed_password="x", full_name="U", organization_id=org.id)
        proj = Project(id="proj-030", org_id=org.id, owner_id=user.id, name="Defect030 Project")
        self.db.add_all([org, user, proj])
        self.db.commit()
        self.user = user
        self.proj = proj

        os.environ.setdefault("MINDED_DATA_DIR", self.temp_dir)

        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "region": np.random.choice(["North", "South", "East", "West"], n),
            "plan_type": np.random.choice(["Basic", "Pro", "Enterprise"], n),
            "monthly_spend": np.round(np.random.gamma(5, 40, n), 2),
            "tenure_months": np.random.randint(1, 60, n),
            "support_tickets": np.random.poisson(2, n),
            "churned": np.random.choice([0, 1], n, p=[0.75, 0.25]),
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        svc = DatasetService(self.db)
        self.dataset = svc.ingest_dataset_file(project_id=proj.id, filename="defect030_data.csv", file_bytes=csv_bytes)

    def tearDown(self):
        self.db.close()

    def test_real_investigation_calculation_trace_is_validated_pydantic_instance(self):
        """Every raw observation's calculation_trace must be a real
        CalculationTraceSchema instance (not a bare dict) once an
        experiment has executed against real, ingested data."""
        analysis_svc = AnalysisService(self.db)
        req = AnalysisCreate(
            question="Why did monthly_spend increase across region?",
            project_id=self.proj.id,
        )
        res = analysis_svc.execute_analysis(req, self.user.id)

        self.assertTrue(res.evidence, "Expected at least one evidence row for a resolvable question.")
        found_trace = False
        for ev in res.evidence:
            if ev.calculation_trace is not None:
                found_trace = True
                self.assertIsInstance(
                    ev.calculation_trace,
                    CalculationTraceSchema,
                    "calculation_trace must be a validated CalculationTraceSchema instance, not a raw dict.",
                )
        self.assertTrue(found_trace, "Expected at least one evidence row to carry a calculation_trace.")

    def test_no_pydantic_serialization_warning_on_real_run(self):
        """The full real-run path (ingest -> ask -> execute_analysis) must
        not emit PydanticSerializationUnexpectedValue warnings -- these
        indicate a field silently holds an unvalidated raw dict where a
        typed Pydantic submodel was declared."""
        analysis_svc = AnalysisService(self.db)
        req = AnalysisCreate(
            question="Is there a correlation between monthly_spend and tenure_months?",
            project_id=self.proj.id,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analysis_svc.execute_analysis(req, self.user.id)

        bad = [w for w in caught if "PydanticSerializationUnexpectedValue" in str(w.message)]
        self.assertEqual(
            bad, [],
            f"Real investigation run emitted Pydantic serialization warnings: {[str(w.message) for w in bad]}",
        )


if __name__ == "__main__":
    unittest.main()
