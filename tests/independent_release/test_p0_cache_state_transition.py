"""P0-1 Regression: InfrastructureManager AI gate is checked on every call.

Proves that a runtime transition AI_ENABLED=true -> AI_ENABLED=false is
immediately reflected by get_ai_provider(), with no cached live provider
being returned after the flag is cleared.
"""
import os
import sys
import unittest
sys.path.insert(0, '.')

from packages.analytics_core.src.providers.manager import InfrastructureManager
from packages.analytics_core.src.providers.ai import NoneAIProvider


class TestCacheStateTransition(unittest.TestCase):
    def setUp(self):
        InfrastructureManager._ai_provider = None
        # Remove env vars that would persist across tests
        for key in ('AI_ENABLED', 'AI_PROVIDER', 'OPENAI_API_KEY', 'GEMINI_API_KEY'):
            os.environ.pop(key, None)

    def tearDown(self):
        InfrastructureManager._ai_provider = None
        for key in ('AI_ENABLED', 'AI_PROVIDER', 'OPENAI_API_KEY', 'GEMINI_API_KEY'):
            os.environ.pop(key, None)

    def test_false_to_true_transition_not_served_from_none_cache(self):
        """AI_ENABLED=false followed by AI_ENABLED=true with mock provider serves mock."""
        os.environ['AI_ENABLED'] = 'false'
        os.environ['AI_PROVIDER'] = 'mock'
        p1 = InfrastructureManager.get_ai_provider()
        self.assertIsInstance(p1, NoneAIProvider, 'AI_ENABLED=false must return NoneAIProvider')

        # Transition to enabled
        os.environ['AI_ENABLED'] = 'true'
        p2 = InfrastructureManager.get_ai_provider()
        self.assertNotIsInstance(p2, NoneAIProvider, 'AI_ENABLED=true must NOT return NoneAIProvider')
        self.assertEqual(p2.provider_type, 'mock')

    def test_true_to_false_transition_cached_provider_not_returned(self):
        """AI_ENABLED=true (mock cached) -> AI_ENABLED=false must return NoneAIProvider."""
        os.environ['AI_ENABLED'] = 'true'
        os.environ['AI_PROVIDER'] = 'mock'
        p1 = InfrastructureManager.get_ai_provider()
        self.assertEqual(p1.provider_type, 'mock', 'first call should return mock')
        # Cache now has the mock provider
        self.assertIsNotNone(InfrastructureManager._ai_provider)

        # Disable AI at runtime
        os.environ['AI_ENABLED'] = 'false'
        p2 = InfrastructureManager.get_ai_provider()
        self.assertIsInstance(p2, NoneAIProvider,
            'After AI_ENABLED=false, cached mock must NOT be returned; NoneAIProvider required')

    def test_disabled_never_returns_live_provider_regardless_of_ai_provider_env(self):
        """AI_ENABLED=false + AI_PROVIDER=openai must always return NoneAIProvider."""
        os.environ['AI_ENABLED'] = 'false'
        os.environ['AI_PROVIDER'] = 'openai'
        os.environ['OPENAI_API_KEY'] = 'sk-fake-key'
        p = InfrastructureManager.get_ai_provider()
        self.assertIsInstance(p, NoneAIProvider)
        self.assertEqual(p.provider_type, 'none')

    def test_repeated_calls_when_disabled_all_return_none(self):
        """Multiple consecutive calls with AI_ENABLED=false all return NoneAIProvider."""
        os.environ['AI_ENABLED'] = 'false'
        os.environ['AI_PROVIDER'] = 'mock'
        for _ in range(5):
            p = InfrastructureManager.get_ai_provider()
            self.assertIsInstance(p, NoneAIProvider)


if __name__ == '__main__':
    unittest.main()
