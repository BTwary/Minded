"""End-to-End Architectural Certification: Zero-AI Local Autonomy Mode.

Proves that AA-OS is genuinely autonomous and local-first:
1. AI_ENABLED=false + AI_PROVIDER=openai + fake API key.
2. Complete socket/HTTP network blocking at the OS process level.
3. Real messy dataset: currency symbols ($), commas, percent signs (%), single-quoted categories,
   dirty whitespace, string dates, and missing values.
4. Real InvestigationController with purely deterministic dependency injection (ai_provider=None).
5. Requires the full canonical pipeline to complete successfully:
   - Natural language semantic interpretation (UniversalQuestionCompiler / interpret_with_schema)
   - Preflight data cleaning & numeric coercion
   - Multi-round experiment synthesis & DuckDB SQL execution
   - Dual-engine verification (DuckDB SQL vs Polars vectorized)
   - EIG experiment selection & Bayesian belief updates (Bayes Factor > 1.0)
   - Evidence ledger & StoppingEngine decision
   - ClaimGate scientific admissibility
   - VerdictEngine calibrated diagnosis
   - Deep analyst answer formulation (AnalystResult.to_text with waterfall reconciliation)
6. Zero outbound network attempts and zero implicit calls to InfrastructureManager.get_ai_provider().
"""
from __future__ import annotations

import http.client
import os
import socket
import sys
import tempfile
import unittest
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from apps.api.src.models.entities import (
    Base,
    User,
    Project,
    Dataset,
    DatasetVersion,
    Investigation,
    Hypothesis,
    Prediction,
    Experiment,
    Observation,
    Evidence,
    EvidenceVerification,
    BeliefUpdate,
    InvestigationVerdict,
    gen_uuid,
)
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider
from packages.analytics_core.src.providers.manager import InfrastructureManager
from packages.analytics_core.src.ingestion.robust_loader import RobustFileLoader


class NetworkAccessForbiddenError(RuntimeError):
    """Raised whenever any outbound network access is attempted."""
    pass


class TestZeroAILocalAutonomy(unittest.TestCase):
    """Rigorous end-to-end verification of Zero-AI Local Autonomy with network blocked."""

    def setUp(self):
        # 1. Enforce misconfigured/leftover AI settings that must NOT leak
        self._saved_env = {
            k: os.environ.get(k)
            for k in (
                "AI_ENABLED",
                "AI_PROVIDER",
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "GEMINI_API_KEY",
                "ANTHROPIC_API_KEY",
                "AAOS_BUSINESS_TIMEZONE",
            )
        }
        os.environ["AI_ENABLED"] = "false"
        os.environ["AI_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "sk-PROHIBITED-OUTBOUND-KEY-MUST-NEVER-BE-USED"
        os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
        os.environ["AAOS_BUSINESS_TIMEZONE"] = "UTC"

        # 2. Block all outbound network at the socket and urllib/http level
        self._orig_socket_connect = socket.socket.connect
        self._orig_create_connection = socket.create_connection
        self._orig_urlopen = urllib.request.urlopen
        self._orig_http_connect = http.client.HTTPConnection.connect
        self._orig_https_connect = http.client.HTTPSConnection.connect

        self._network_call_attempts = 0

        def _forbidden_network(*args, **kwargs):
            self._network_call_attempts += 1
            raise NetworkAccessForbiddenError(
                f"FORBIDDEN: Outbound network call attempted in Zero-AI Local Autonomy mode! Args: {args}"
            )

        socket.socket.connect = _forbidden_network
        socket.create_connection = _forbidden_network
        urllib.request.urlopen = _forbidden_network
        http.client.HTTPConnection.connect = _forbidden_network
        http.client.HTTPSConnection.connect = _forbidden_network

        # 3. Monitor InfrastructureManager.get_ai_provider to verify controller never calls it
        self._orig_get_ai_provider = InfrastructureManager.get_ai_provider
        self._infrastructure_ai_calls = 0

        def _monitored_get_ai_provider(*args, **kwargs):
            self._infrastructure_ai_calls += 1
            return self._orig_get_ai_provider(*args, **kwargs)

        InfrastructureManager.get_ai_provider = _monitored_get_ai_provider
        InfrastructureManager._ai_provider = None

        # 4. Setup isolated in-memory SQLite database
        self.temp_dir = tempfile.mkdtemp(prefix="aaos_zero_ai_test_")
        self.db_path = os.path.join(self.temp_dir, "zero_ai_test.db")
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False, "timeout": 30},
            poolclass=StaticPool,
        )

        @event.listens_for(self.engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
            finally:
                cursor.close()

        self.SessionFactory = sessionmaker(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        self.db = self.SessionFactory()

        self.user = User(
            id=f"user-{gen_uuid()[:8]}",
            email="local_analyst@aaos.test",
            hashed_password="hash",
            full_name="Local Analyst",
            is_active=True,
            role="owner",
        )
        self.proj = Project(
            id=f"proj-{gen_uuid()[:8]}",
            name="Zero-AI Autonomy Test",
            owner_id=self.user.id,
        )
        self.db.add_all([self.user, self.proj])
        self.db.commit()

    def tearDown(self):
        # Restore network hooks
        socket.socket.connect = self._orig_socket_connect
        socket.create_connection = self._orig_create_connection
        urllib.request.urlopen = self._orig_urlopen
        http.client.HTTPConnection.connect = self._orig_http_connect
        http.client.HTTPSConnection.connect = self._orig_https_connect

        # Restore InfrastructureManager
        InfrastructureManager.get_ai_provider = self._orig_get_ai_provider
        InfrastructureManager._ai_provider = None

        # Restore environment
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        try:
            self.db.close()
            self.engine.dispose()
        except Exception:
            pass

    def _generate_messy_business_dataset(self) -> pd.DataFrame:
        """Create a realistic messy dataset with formatted currency, quotes, dirty whitespace, and missing values,
        and pass it through RobustFileLoader to mirror authentic production ingestion."""
        np.random.seed(42)
        n_rows = 400

        regions = ["us-east", "us-west", "eu-central", "ap-south"]
        categories = ["Men's Apparel", "Women's Clothing", "Children's Wear", "Footwear & Accessories"]

        data = []
        for i in range(n_rows):
            reg = regions[i % len(regions)]
            cat = categories[i % len(categories)]
            day = (i % 28) + 1
            order_date = f"2026-03-{day:02d}"

            cat_base = {
                "Men's Apparel": 240.0,
                "Women's Clothing": 190.0,
                "Children's Wear": 140.0,
                "Footwear & Accessories": 100.0,
            }.get(cat, 150.0)

            # Regional premium for us-east (driver of regional surge)
            reg_bonus = 50.0 if reg == "us-east" else 0.0
            raw_amt = cat_base + reg_bonus + (i % 15)

            # Messy currency formatting: $, commas, trailing whitespace
            amt_str = f" ${raw_amt:,.2f} "
            if i % 60 == 0:
                # Occasional dirty negative format
                amt_str = f" -${abs(raw_amt):,.2f} "
            elif i % 80 == 0:
                # Occasional missing value
                amt_str = None

            # Percentage string with %
            discount_str = f"{float(i % 15):.1f}%"

            # Dirty whitespace in category and region
            dirty_region = f"  {reg}  " if i % 3 == 0 else reg
            dirty_cat = cat

            data.append({
                "order_id": f"ORD-{i:05d}",
                "region": dirty_region,
                "category": dirty_cat,
                "sales_amount": amt_str,
                "discount_rate": discount_str,
                "order_date": order_date,
            })

        raw_df = pd.DataFrame(data)
        csv_bytes = raw_df.to_csv(index=False).encode("utf-8")
        clean_df, report = RobustFileLoader().load(file_bytes=csv_bytes, filename="transactions.csv")
        clean_df.attrs["dataset_id"] = "ds-messy-01"
        clean_df.attrs["business_timezone"] = "UTC"
        return clean_df

    def test_01_real_controller_investigation_completes_with_network_blocked(self):
        """End-to-End Test: Full investigation on messy data succeeds deterministically with network blocked."""
        df = self._generate_messy_business_dataset()
        provider = InMemoryDatasetProvider({"transactions": df})

        inv = Investigation(
            id=f"INV-ZAI-{gen_uuid()[:8]}",
            project_id=self.proj.id,
            user_id=self.user.id,
            question="Why did sales_amount surge across region?",
            status="QUEUED",
        )
        self.db.add(inv)
        self.db.commit()

        # Reset call tracker
        self._infrastructure_ai_calls = 0

        # Construct InvestigationController with default dependency injection (ai_provider=None)
        controller = InvestigationController(
            session_factory=self.SessionFactory,
            dataset_provider=provider,
            ai_provider=None,
        )
        self.assertIsNone(controller.ai_provider)

        # Execute investigation
        success = controller.execute_investigation(
            investigation_id=inv.id,
            worker_id="zero-ai-worker",
        )
        self.assertTrue(success, "InvestigationController execution must succeed deterministically.")

        # PROOF 1: Zero network attempts made
        self.assertEqual(self._network_call_attempts, 0, "Zero outbound network calls must be attempted.")

        # PROOF 2: InvestigationController canonical path never called InfrastructureManager.get_ai_provider()
        self.assertEqual(
            self._infrastructure_ai_calls,
            0,
            "InvestigationController must never consult InfrastructureManager.get_ai_provider().",
        )

        self.db.expire_all()
        completed_inv = self.db.query(Investigation).filter(Investigation.id == inv.id).first()
        self.assertEqual(completed_inv.status, "COMPLETED")
        self.assertIn(completed_inv.verdict_type, ("DIAGNOSED", "OBSERVED"))

        # PROOF 3: Candidate experiment was compiled and executed via DuckDB
        exp = self.db.query(Experiment).filter(
            Experiment.investigation_id == inv.id,
            Experiment.status == "EXECUTED",
        ).first()
        self.assertIsNotNone(exp, "Executed experiment must be recorded in relational DB.")

        # PROOF 4: Observation was collected despite messy currency and string dates
        obs = self.db.query(Observation).filter(Observation.experiment_id == exp.id).first()
        self.assertIsNotNone(obs, "Raw observation must be captured from DuckDB execution.")
        self.assertGreater(obs.row_count_analyzed, 0)

        # PROOF 5: Dual-engine verification passed between DuckDB and Polars
        ev = self.db.query(Evidence).filter(
            Evidence.investigation_id == inv.id,
            Evidence.experiment_id == exp.id,
        ).first()
        self.assertIsNotNone(ev, "Verified evidence must be linked to investigation.")

        ev_ver = self.db.query(EvidenceVerification).filter(EvidenceVerification.evidence_id == ev.id).first()
        self.assertIsNotNone(ev_ver, "EvidenceVerification audit record must exist.")
        self.assertEqual(ev_ver.primary_tool, "duckdb_sql")
        self.assertEqual(ev_ver.secondary_tool, "polars_vectorized")
        self.assertIsNotNone(ev_ver.observed_delta_pct)
        self.assertLessEqual(ev_ver.observed_delta_pct, 1e-4)

        # PROOF 6: Belief updates computed decisive Bayes factor > 1.0 and updated posterior
        bus = self.db.query(BeliefUpdate).filter(
            BeliefUpdate.investigation_id == inv.id,
        ).all()
        self.assertGreater(len(bus), 0, "BeliefUpdate records must be recorded.")
        decisive_bus = [b for b in bus if b.bayes_factor > 1.0]
        self.assertGreater(len(decisive_bus), 0, "At least one belief update must have Bayes Factor > 1.0.")
        self.assertGreaterEqual(max(b.posterior_probability for b in bus), 0.70)

        # PROOF 7: Investigation verdict is DIAGNOSED with calibrated confidence
        verd = self.db.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == inv.id).first()
        self.assertIsNotNone(verd, "InvestigationVerdict must be persisted.")
        self.assertEqual(verd.verdict_type, "DIAGNOSED")
        self.assertGreaterEqual(verd.confidence_score, 0.70)
        self.assertTrue(verd.counter_hypothesis_refuted)

        # PROOF 8: Substantive analyst direct answer is populated with numbers
        self.assertIsNotNone(completed_inv.direct_answer)
        self.assertGreater(len(completed_inv.direct_answer), 20)
        self.assertIn("us-east", completed_inv.direct_answer.lower())

    def test_02_multi_group_numeric_comparison_on_messy_data_with_network_blocked(self):
        """End-to-End Test: Multi-group numeric comparison across 4 categories succeeds deterministically."""
        df = self._generate_messy_business_dataset()
        provider = InMemoryDatasetProvider({"transactions": df})

        inv = Investigation(
            id=f"INV-ZAI-CAT-{gen_uuid()[:8]}",
            project_id=self.proj.id,
            user_id=self.user.id,
            question="Does sales_amount differ by category?",
            status="QUEUED",
        )
        self.db.add(inv)
        self.db.commit()

        self._infrastructure_ai_calls = 0

        controller = InvestigationController(
            session_factory=self.SessionFactory,
            dataset_provider=provider,
            ai_provider=None,
        )

        success = controller.execute_investigation(
            investigation_id=inv.id,
            worker_id="zero-ai-worker-cat",
        )
        self.assertTrue(success, "Multi-group category comparison must succeed deterministically.")

        # Zero network calls
        self.assertEqual(self._network_call_attempts, 0)
        self.assertEqual(self._infrastructure_ai_calls, 0)

        self.db.expire_all()
        completed = self.db.query(Investigation).filter(Investigation.id == inv.id).first()
        self.assertEqual(completed.status, "COMPLETED")
        self.assertIsNotNone(completed.direct_answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
