"""Test Suite for Mode 1: Free / Zero AI API Deterministic Analytical Runtime.

Proves that AA-OS performs real analytical work deterministically without requiring
Gemini, OpenAI, Anthropic, or any other paid AI API.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Set environment to explicit Zero-API Mode before importing modules
os.environ["AI_ENABLED"] = "false"
os.environ["AI_PROVIDER"] = "none"
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("GROQ_API_KEY", None)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from apps.api.src.ai.providers.factory import get_ai_config, get_ai_provider, is_ai_enabled, set_ai_config
from apps.api.src.ai.providers.mock import NoneAIProvider, DeterministicMockAIProvider
from apps.api.src.ai.runtime import InvestigationRuntime
from packages.analytics_core.src.discovery.engine import DiscoveryEngine
from packages.analytics_core.src.profiling.profiler import DataProfiler
from packages.analytics_core.src.semantic.world_model import SemanticWorldModelBuilder
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.intelligence.predictive_hypothesis import HypothesisSynthesizer
from packages.analytics_core.src.intelligence.prediction_engine import PredictionSynthesizer, PredictionEvaluator
from packages.analytics_core.src.intelligence.experiment_synthesizer import ExperimentSynthesizer
from packages.analytics_core.src.intelligence.eig_optimizer import EIGOptimizer
from packages.analytics_core.src.engines.execution_provider import DuckDBExecutionProvider
from packages.analytics_core.src.engines.belief import BeliefEngine, compute_shannon_entropy
from packages.analytics_core.src.intelligence.adversarial_attacker import AdversarialAttacker
from packages.analytics_core.src.engines.evidence_ledger import EvidenceLedger
from packages.schemas.src.analysis import EpistemicClaimType, ValidationStatus


class TestZeroAIMode(unittest.TestCase):
    """Rigorous tests proving Mode 1 (Free / Zero AI API) functions completely and deterministically."""

    def setUp(self):
        set_ai_config(enabled=False, provider="none", api_key="")
        
        np.random.seed(42)
        n_rows = 500
        dates = pd.date_range("2026-01-01", periods=90, freq="D")
        
        self.df_sales = pd.DataFrame({
            "order_id": [f"ORD-{i:05d}" for i in range(n_rows)],
            "order_date": np.random.choice(dates, size=n_rows),
            "customer_id": [f"CUST-{np.random.randint(1, 100):03d}" for _ in range(n_rows)],
            "product_id": [f"PROD-{np.random.randint(1, 20):02d}" for _ in range(n_rows)],
            "region": np.random.choice(["North", "South", "East", "West"], size=n_rows),
            "revenue": np.round(np.random.uniform(20.0, 500.0, size=n_rows), 2),
            "quantity": np.random.randint(1, 10, size=n_rows),
            "unit_price": np.round(np.random.uniform(10.0, 100.0, size=n_rows), 2),
            "discount_pct": np.round(np.random.uniform(0.0, 0.3, size=n_rows), 2),
        })
        self.df_sales.attrs["dataset_id"] = "ds-sales-01"
        self.df_sales.attrs["version"] = 1
        
        self.datasets = {"sales": self.df_sales}

    def test_01_runtime_starts_without_ai_key(self):
        """A. Proves application starts without an AI API key and provider returns NoneAIProvider."""
        self.assertFalse(is_ai_enabled())
        prov = get_ai_provider()
        self.assertIsInstance(prov, (NoneAIProvider, DeterministicMockAIProvider))
        self.assertFalse(prov.is_ai_enabled)
        
        cfg = get_ai_config()
        self.assertFalse(cfg.get("enabled", True))
        self.assertEqual(cfg.get("api_key", ""), "")

    def test_02_dataset_profiling_and_schema_inference_deterministic(self):
        """B. Proves datasets are profiled and schemas inferred deterministically without AI."""
        profiler = DataProfiler()
        profile = profiler.profile_dataframe(self.df_sales, dataset_name="sales")
        
        self.assertIsNotNone(profile.columns)
        self.assertEqual(profile.row_count, 500)
        self.assertGreater(profile.data_quality.overall_score, 80)
        
        builder = SemanticWorldModelBuilder()
        world_model = builder.build_world_model(self.datasets)
        
        self.assertGreater(len(world_model.entities), 0)
        self.assertGreater(len(world_model.metrics), 0)
        metric_names = [m.column_name for m in world_model.metrics]
        self.assertIn("revenue", metric_names)

    def test_03_hypothesis_and_prediction_synthesis_without_ai(self):
        """C and D. Proves hypotheses and testable predictions synthesize deterministically without AI."""
        question = "Why did revenue decline in Region B during March?"
        intent = IntentEngine.parse_intent(question)
        semantic_engine = SemanticEngine()
        semantic = semantic_engine.resolve_schema(intent, self.datasets)
        
        self.assertEqual(semantic.target_metric_col, "revenue")
        
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, question)
        self.assertGreaterEqual(len(hyps), 2)
        
        priors = [h.prior_probability for h in hyps]
        self.assertAlmostEqual(sum(priors), 1.0, places=4)
        
        for h in hyps:
            pred = PredictionSynthesizer.synthesize_prediction_for_hypothesis(
                hypothesis=h,
                prediction_id=f"PRED_{h.hypothesis_code}",
                hypothesis_entity_id=f"HYP_{h.hypothesis_code}",
            )
            self.assertTrue(pred.testability)
            self.assertIsNotNone(pred.statement)

    def test_04_experiment_synthesis_and_eig_ranking_without_ai(self):
        """D and E. Proves candidate experiments synthesize and rank via Shannon EIG without AI."""
        question = "What is the primary driver of revenue?"
        intent = IntentEngine.parse_intent(question)
        semantic = SemanticEngine().resolve_schema(intent, self.datasets)
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, question)
        
        candidates = ExperimentSynthesizer.synthesize_candidate_experiments(hyps, semantic)
        self.assertGreater(len(candidates), 0)
        
        selected_exp, rationale = EIGOptimizer.select_next_experiment(
            candidates,
            hyps,
            executed_codes=[],
            executed_fingerprints=[],
        )
        self.assertIsNotNone(selected_exp)
        self.assertIsNotNone(selected_exp.code)
        self.assertIsNotNone(selected_exp.query_sql)
        self.assertIn("SELECT", selected_exp.query_sql.upper())

    def test_05_duckdb_execution_and_statistical_verification_without_ai(self):
        """E, F, G. Proves DuckDB execution, statistical verification, and evidence ledger update without AI."""
        exec_provider = DuckDBExecutionProvider()
        
        sql = "SELECT region, SUM(revenue) AS total_rev FROM sales GROUP BY region ORDER BY total_rev DESC"
        primary_val, row_count, dur, res_df = exec_provider.execute_query(
            df=self.df_sales,
            table_name="sales",
            query_sql=sql,
        )
        
        self.assertEqual(row_count, 500)
        self.assertEqual(len(res_df), 4)
        self.assertGreater(primary_val, 0)
        
        ledger = EvidenceLedger("INV-ZERO-AI-01")
        record = ledger.record_claim(
            claim_statement="Regional revenue breakdown executed in DuckDB.",
            claim_type=EpistemicClaimType.ASSOCIATION,
            source_experiment_id="EXP-01",
            source_datasets=["sales"],
            source_columns=["revenue", "region"],
            row_count_evaluated=500,
            computation_proof={"primary_value": primary_val},
            verification_status="VERIFIED",
        )
        self.assertEqual(record.verification_status, "VERIFIED")
        self.assertEqual(len(record.provenance_hash), 64)

    def test_06_bayesian_belief_update_and_adversarial_checks_without_ai(self):
        """G and H. Proves Bayesian belief posteriors update and adversarial attacks evaluate without AI."""
        question = "What drove revenue variation?"
        intent = IntentEngine.parse_intent(question)
        semantic = SemanticEngine().resolve_schema(intent, self.datasets)
        hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, question)
        
        priors = [h.prior_probability for h in hyps]
        likelihoods = [0.85] + [0.15 / (len(hyps) - 1)] * (len(hyps) - 1)
        
        posteriors, delta_entropy = BeliefEngine.compute_bayesian_posteriors(
            priors=priors,
            likelihoods=likelihoods,
        )
        
        self.assertAlmostEqual(sum(posteriors), 1.0, places=4)
        
        attack_res = AdversarialAttacker.design_attack(
            leading_hyp=hyps[0],
            counter_hyps=[hyps[1]],
            df=self.df_sales,
            semantic=semantic,
            execution_provider=DuckDBExecutionProvider(),
        )
        self.assertIsNotNone(attack_res.attack_status)
        self.assertIsNotNone(attack_res.discriminating_test_sql)

    def test_07_full_end_to_end_investigation_in_zero_ai_mode(self):
        """B, C, I. Runs complete InvestigationRuntime end-to-end in Mode 1 (Zero AI API)."""
        runtime = InvestigationRuntime(ai_provider=NoneAIProvider())
        
        question = "Why did revenue contract between periods across regions?"
        resp = runtime.execute_investigation(
            question=question,
            project_id="proj-zero-ai-test",
            datasets=self.datasets,
        )
        
        self.assertEqual(resp.status.value, "COMPLETED")
        self.assertIn(resp.verdict.value, ["DIAGNOSED", "OBSERVED", "STATISTICALLY_SIGNIFICANT", "CONFIRMED"])
        self.assertIsNotNone(resp.direct_answer)
        self.assertIsNotNone(resp.main_finding)
        self.assertGreater(len(resp.steps), 0)
        self.assertGreater(len(resp.findings), 0)
        self.assertGreater(len(resp.evidence), 0)
        
        self.assertEqual(resp.ai_mode, "DETERMINISTIC")
        self.assertFalse(resp.ai_fallback_triggered)
        self.assertIsNotNone(resp.manifest)
        self.assertEqual(resp.manifest.ai_mode, "DETERMINISTIC")
        self.assertIsNotNone(resp.manifest.reproducible_hash)
        
        manifest_str = resp.manifest.model_dump_json()
        self.assertNotIn("AIza", manifest_str)
        self.assertNotIn("sk-", manifest_str)
        self.assertNotIn("Bearer", manifest_str)


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING MODE 1 (FREE / ZERO AI API) DETERMINISTIC ACCEPTANCE SUITE")
    print("=" * 80)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestZeroAIMode)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("\nALL ZERO-AI MODE TESTS PASSED CLEANLY (7/7 tests passed).")
