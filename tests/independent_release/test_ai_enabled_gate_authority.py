"""Regression test for a confirmed P0 architecture violation (2026-09-25 session).

FINDING: packages/analytics_core/src/providers/manager.py's
InfrastructureManager.get_ai_provider() previously read only the AI_PROVIDER
environment variable. It never consulted
platform_local_first.resolve_ai_enabled(), which is documented elsewhere in
this repo (apps/api/src/core/config.py, apps/api/src/ai/providers/factory.py)
as the single source of truth for whether AI may run at all.

Consequence: with AI_ENABLED=false and a leftover/misconfigured
AI_PROVIDER=openai (or gemini/ollama/...) in the environment, the canonical
deterministic semantic-interpretation path in analytics_core
(UniversalQuestionPlanner.compile -> nl_semantic_interpreter.interpret_with_schema
-> InfrastructureManager.get_ai_provider()) would still construct a live
OpenAIProvider and attempt a real outbound HTTP request via
urllib.request.urlopen -- reproduced directly against this codebase before
the fix below (interpret_with_schema was called with a stubbed urlopen that
raises on any invocation; the exception fired).

FIX: InfrastructureManager.get_ai_provider() now checks resolve_ai_enabled()
first and short-circuits to NoneAIProvider() when it is False, regardless of
AI_PROVIDER's value. This test locks that behavior in and proves the
canonical entry point (interpret_with_schema, reached from
UniversalQuestionPlanner) makes zero network calls under this configuration.
"""
import os
import sys
import unittest
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class TestAIEnabledGateAuthority(unittest.TestCase):
    def setUp(self):
        self._saved_env = {
            k: os.environ.get(k) for k in ("AI_ENABLED", "AI_PROVIDER", "OPENAI_API_KEY")
        }
        self._orig_urlopen = urllib.request.urlopen

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        urllib.request.urlopen = self._orig_urlopen
        from packages.analytics_core.src.providers.manager import InfrastructureManager
        InfrastructureManager._ai_provider = None

    def _block_network(self):
        def _fail(*a, **k):
            raise AssertionError("NETWORK CALL ATTEMPTED despite AI_ENABLED=false")
        urllib.request.urlopen = _fail

    def test_01_ai_enabled_false_overrides_named_provider_at_manager_level(self):
        os.environ["AI_ENABLED"] = "false"
        os.environ["AI_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "sk-fake-does-not-matter"
        from packages.analytics_core.src.providers.manager import InfrastructureManager
        from packages.analytics_core.src.providers.ai import NoneAIProvider
        InfrastructureManager._ai_provider = None
        provider = InfrastructureManager.get_ai_provider()
        self.assertIsInstance(provider, NoneAIProvider)

    def test_02_canonical_semantic_interpreter_makes_zero_network_calls(self):
        os.environ["AI_ENABLED"] = "false"
        os.environ["AI_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "sk-fake-does-not-matter"
        from packages.analytics_core.src.providers.manager import InfrastructureManager
        InfrastructureManager._ai_provider = None
        self._block_network()
        from packages.analytics_core.src.intelligence.nl_semantic_interpreter import interpret_with_schema
        result = interpret_with_schema(
            "why did revenue fall in march", ["revenue", "region", "order_date"]
        )
        self.assertEqual(result.source, "deterministic")

    def test_03_ai_enabled_true_still_permits_the_named_provider(self):
        os.environ["AI_ENABLED"] = "true"
        os.environ["AI_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "sk-fake-does-not-matter"
        from packages.analytics_core.src.providers.manager import InfrastructureManager
        from packages.analytics_core.src.providers.ai import OpenAIProvider
        InfrastructureManager._ai_provider = None
        provider = InfrastructureManager.get_ai_provider()
        self.assertIsInstance(provider, OpenAIProvider)

    def test_04_unset_ai_enabled_keeps_legacy_named_provider_opt_in(self):
        os.environ.pop("AI_ENABLED", None)
        os.environ["AI_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "sk-fake-does-not-matter"
        from packages.analytics_core.src.providers.manager import InfrastructureManager
        from packages.analytics_core.src.providers.ai import OpenAIProvider
        InfrastructureManager._ai_provider = None
        provider = InfrastructureManager.get_ai_provider()
        self.assertIsInstance(provider, OpenAIProvider)

    def test_05_default_environment_is_none_provider(self):
        os.environ.pop("AI_ENABLED", None)
        os.environ.pop("AI_PROVIDER", None)
        from packages.analytics_core.src.providers.manager import InfrastructureManager
        from packages.analytics_core.src.providers.ai import NoneAIProvider
        InfrastructureManager._ai_provider = None
        provider = InfrastructureManager.get_ai_provider()
        self.assertIsInstance(provider, NoneAIProvider)

    def test_06_interpret_with_schema_does_not_call_infrastructure_manager(self):
        """Proves interpret_with_schema is completely decoupled from InfrastructureManager."""
        from packages.analytics_core.src.providers.manager import InfrastructureManager
        from packages.analytics_core.src.intelligence.nl_semantic_interpreter import interpret_with_schema

        def _forbidden_get_ai():
            raise AssertionError("FATAL: InfrastructureManager.get_ai_provider() was called by deterministic interpreter!")

        orig_get_ai = InfrastructureManager.get_ai_provider
        InfrastructureManager.get_ai_provider = _forbidden_get_ai
        try:
            # Calling without ai_provider must not touch InfrastructureManager
            proposal = interpret_with_schema("why did revenue fall", ["revenue", "region", "date"])
            self.assertEqual(proposal.source, "deterministic")
            self.assertEqual(proposal.task, "DIAGNOSTIC")
        finally:
            InfrastructureManager.get_ai_provider = orig_get_ai

    def test_07_controller_and_planner_propagate_injected_ai_provider_only_when_passed(self):
        """Proves InvestigationController and UniversalQuestionCompiler use dependency-injected ai_provider."""
        from packages.analytics_core.src.runtime.controller import InvestigationController
        from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
        import pandas as pd

        # 1. Default controller has ai_provider=None
        ctrl = InvestigationController()
        self.assertIsNone(ctrl.ai_provider)

        # 2. Universal planner accepts explicit mock provider
        class MockProposalAI:
            provider_type = "mock"
            def generate_response(self, prompt, **kwargs):
                return '{"task": "DIAGNOSTIC", "target": "sales", "group": "category"}'

        mock_prov = MockProposalAI()
        df = pd.DataFrame({"sales": [10, 20], "category": ["A", "B"]})
        class DummySemantic:
            primary_dataset_name = "test_table"
            metrics = []
            dimensions = []
        plan = UniversalQuestionCompiler.compile(
            "why did sales drop by category",
            semantic=DummySemantic(),
            df=df,
            ai_provider=mock_prov,
        )
        self.assertIn("nl_proposal", plan.semantics.semantic_evidence)
        self.assertEqual(plan.semantics.semantic_evidence["nl_proposal"]["source"], "byom_validated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
