"""Unit tests for deterministic autonomous question synthesis."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.schemas.src.dataset import (
    ColumnProfileSchema,
    DataQualityBreakdownSchema,
    DatasetProfileSchema,
)
from packages.shared.src.enums import ColumnDataType, SemanticType
from minded.questions import synthesize_questions


class QuestionSynthesisTests(unittest.TestCase):
    def test_time_and_metric_question(self):
        profile = DatasetProfileSchema(
            dataset_name="sales",
            version=1,
            row_count=100,
            column_count=3,
            columns=[
                ColumnProfileSchema(
                    name="order_date",
                    data_type=ColumnDataType.DATETIME,
                    semantic_type=SemanticType.TIMESTAMP,
                ),
                ColumnProfileSchema(
                    name="revenue",
                    data_type=ColumnDataType.FLOAT,
                    semantic_type=SemanticType.METRIC,
                ),
                ColumnProfileSchema(
                    name="region",
                    data_type=ColumnDataType.STRING,
                    semantic_type=SemanticType.DIMENSION,
                    cardinality_ratio=0.02,
                ),
            ],
            data_quality=DataQualityBreakdownSchema(overall_score=92.0),
        )
        questions = synthesize_questions(profile)
        self.assertTrue(questions)
        self.assertIn("revenue", questions[0])
        self.assertIn("order_date", questions[0])

    def test_low_quality_adds_followup(self):
        profile = DatasetProfileSchema(
            dataset_name="messy",
            version=1,
            row_count=20,
            column_count=1,
            columns=[
                ColumnProfileSchema(
                    name="amount",
                    data_type=ColumnDataType.FLOAT,
                    semantic_type=SemanticType.METRIC,
                )
            ],
            data_quality=DataQualityBreakdownSchema(overall_score=40.0),
        )
        questions = synthesize_questions(profile, max_questions=3)
        self.assertGreaterEqual(len(questions), 2)
        self.assertTrue(any("quality" in q.lower() for q in questions))


if __name__ == "__main__":
    unittest.main()
