import unittest
from packages.schemas.src.analysis import AnalysisCreate
from packages.schemas.src.chat import ChatRequest
from packages.analytics_core.src.engines.local_explanation import LocalExplanationEngine


class TestAnalysisModes(unittest.TestCase):
    def test_deterministic_is_default(self):
        self.assertEqual(AnalysisCreate(question="What happened?", project_id="p").analysis_mode, "DETERMINISTIC")
        self.assertEqual(ChatRequest(message="What happened?", project_id="p").analysis_mode, "DETERMINISTIC")

    def test_ai_mode_is_optional_route(self):
        self.assertEqual(
            AnalysisCreate(question="Explain this", project_id="p", analysis_mode="AI_AUGMENTED").analysis_mode,
            "AI_AUGMENTED",
        )

    def test_local_explanation_never_requires_ai(self):
        result = LocalExplanationEngine.build(
            question="Why?",
            verdict_type="INCONCLUSIVE",
            direct_answer="Insufficient evidence.",
            justification="The evidence is insufficient.",
        )
        self.assertEqual(result.mode, "DETERMINISTIC")
        self.assertIn("inconclusive", result.summary.lower())


if __name__ == "__main__":
    unittest.main()
