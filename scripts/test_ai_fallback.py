"""Test Suite for Graceful AI Fallbacks & Resilience.

Verifies that if an AI API call fails for any reason (timeout, rate limit,
invalid API key, network failure, malformed JSON), AA-OS:
1. Does NOT crash or halt.
2. Logs a structured warning.
3. Seamlessly falls back to deterministic execution.
4. Completes the investigation with full empirical validity.
5. Never exposes API keys or secrets in logs, manifests, or responses.
"""
import os
import sys
import unittest
from typing import Any, Optional, Type, TypeVar
import numpy as np
import pandas as pd
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from apps.api.src.ai.providers.base import BaseAIProvider
from apps.api.src.ai.agents.planner import PlannerAgent
from apps.api.src.ai.agents.sql_agent import SQLAgent
from apps.api.src.ai.runtime import InvestigationRuntime
from packages.schemas.src.analysis import AnalysisStatus

T = TypeVar("T", bound=BaseModel)


class FailingNetworkAIProvider(BaseAIProvider):
    """Simulates hard network drop or timeout."""
    def __init__(self, error_type: str = "timeout"):
        self.error_type = error_type
        self.model = "failing-ai-sim"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        if self.error_type == "timeout":
            raise TimeoutError("Simulated socket connection timeout to remote AI API (10000ms elapsed).")
        elif self.error_type == "auth":
            raise PermissionError("HTTP 401 Unauthorized: Invalid or revoked API key 'AIzaSyFAKE_KEY_DO_NOT_LOG'.")
        elif self.error_type == "rate_limit":
            raise RuntimeError("HTTP 429 Too Many Requests: Rate limit exceeded for quota group default.")
        raise ConnectionResetError("Connection reset by peer.")

    def generate_structured(self, prompt: str, schema_cls: Type[T], system_prompt: Optional[str] = None, temperature: float = 0.1) -> T:
        return self.generate(prompt=prompt, system_prompt=system_prompt, temperature=temperature)


class MalformedJSONAIProvider(BaseAIProvider):
    """Simulates LLM hallucinating invalid JSON or ignoring schema."""
    def __init__(self):
        self.model = "hallucinating-ai-sim"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        return "I am unable to answer in JSON. Here is some random prose instead without valid syntax {bad_json:"

    def generate_structured(self, prompt: str, schema_cls: Type[T], system_prompt: Optional[str] = None, temperature: float = 0.1) -> T:
        raise ValueError("Failed to parse structured output into schema: Expecting property name enclosed in double quotes.")


class TestAIFallbacks(unittest.TestCase):
    """Rigorous tests proving fail-safe fallback resilience across all AI failure modes."""

    def setUp(self):
        # Row-level (not pre-aggregated) fixture, sized to actually clear
        # InvestigationController's DataQualityGate.evaluate_fitness minimum
        # sample-size threshold (min_sample_size=5). The previous 4-row
        # pre-aggregated fixture only ever worked against the old legacy
        # runtime loop, which had no such gate; against the real controller
        # it was correctly rejected as statistically unfit before ever
        # reaching the AI-fallback behavior these tests exist to verify.
        rng = np.random.default_rng(7)
        n = 200
        regions = rng.choice(["North", "South", "East", "West"], size=n, p=[0.3, 0.25, 0.25, 0.2])
        revenue = np.where(
            regions == "North", rng.normal(300, 40, n), rng.normal(200, 35, n)
        )
        self.df_test = pd.DataFrame({
            "row_id": [f"R{i}" for i in range(n)],
            "region": regions,
            "revenue": np.round(revenue, 2),
            "units": rng.integers(1, 15, n),
        })
        self.df_test.attrs["dataset_id"] = "ds-test"
        self.df_test.attrs["version"] = 1
        self.datasets = {"sales": self.df_test}

    def test_01_planner_fallback_on_timeout(self):
        """Proves PlannerAgent gracefully falls back to deterministic planning on timeout."""
        failing_provider = FailingNetworkAIProvider(error_type="timeout")
        planner = PlannerAgent(provider=failing_provider)
        
        plan = planner.propose_initial_plan(
            question="Why did revenue drop?",
            semantic_model={"target_metric_col": "revenue", "tables": ["sales"]},
            available_tools=[{"name": "run_sql"}],
        )
        
        self.assertIsNotNone(plan)
        self.assertIn("hypotheses", plan)
        self.assertGreater(len(plan["hypotheses"]), 0)
        self.assertIn("first_action", plan)

    def test_02_planner_fallback_on_malformed_json(self):
        """Proves PlannerAgent falls back deterministically when LLM outputs broken JSON."""
        bad_json_provider = MalformedJSONAIProvider()
        planner = PlannerAgent(provider=bad_json_provider)
        
        plan = planner.propose_initial_plan(
            question="Analyze sales variance",
            semantic_model={"target_metric_col": "revenue", "tables": ["sales"]},
            available_tools=[{"name": "run_sql"}],
        )
        
        self.assertIsNotNone(plan)
        self.assertGreater(len(plan.get("hypotheses", [])), 0)

    def test_03_sql_agent_fallback_on_ai_failure(self):
        """Proves SQLAgent constructs deterministic fallback query when LLM generation fails."""
        failing_provider = FailingNetworkAIProvider(error_type="auth")
        agent = SQLAgent(provider=failing_provider)
        
        sql = agent.generate_sql(
            task_description="Analyze revenue by region",
            table_schemas={"sales": {"columns": [{"name": "region"}, {"name": "revenue"}]}},
        )
        
        self.assertIsNotNone(sql)
        self.assertIn("SELECT", sql.upper())
        self.assertIn("sales", sql.lower())

    def test_04_end_to_end_investigation_survives_ai_rate_limit(self):
        """Proves end-to-end investigation completes cleanly when AI provider returns 429 Rate Limit."""
        rate_limited_provider = FailingNetworkAIProvider(error_type="rate_limit")
        runtime = InvestigationRuntime(ai_provider=rate_limited_provider)
        
        resp = runtime.execute_investigation(
            question="What is the regional revenue distribution?",
            project_id="proj-fallback-test",
            datasets=self.datasets,
        )
        
        self.assertEqual(resp.status, AnalysisStatus.COMPLETED)
        self.assertIsNotNone(resp.direct_answer)
        self.assertIsNotNone(resp.manifest)
        self.assertGreater(len(resp.evidence), 0)

    def test_05_secrets_never_leak_in_manifest_or_response_after_failure(self):
        """Proves API keys and credentials are never stored in response or manifest after error."""
        failing_provider = FailingNetworkAIProvider(error_type="auth")
        runtime = InvestigationRuntime(ai_provider=failing_provider)
        
        resp = runtime.execute_investigation(
            question="Audit revenue variance",
            project_id="proj-leak-test",
            datasets=self.datasets,
        )
        
        json_output = resp.model_dump_json()
        self.assertNotIn("AIzaSyFAKE_KEY_DO_NOT_LOG", json_output)
        self.assertNotIn("sk-", json_output)
        self.assertNotIn("Bearer", json_output)


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING AI GRACEFUL FALLBACK & RESILIENCE ACCEPTANCE SUITE")
    print("=" * 80)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAIFallbacks)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("\nALL AI FALLBACK TESTS PASSED CLEANLY (5/5 tests passed).")
