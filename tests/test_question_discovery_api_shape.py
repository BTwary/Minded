from packages.schemas.src.analysis import AnalysisResponse


def test_analysis_response_exposes_dataset_question_discovery():
    fields = AnalysisResponse.model_fields
    assert "suggested_questions" in fields
    default = fields["suggested_questions"].default_factory()
    assert default == []
