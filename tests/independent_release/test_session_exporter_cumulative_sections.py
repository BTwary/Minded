from io import BytesIO

from docx import Document
from pypdf import PdfReader

from packages.analytics_core.src.reporting.session_exporter import (
    export_analysis_session_to_docx,
    export_analysis_session_to_pdf,
)


def _question(question: str, answer: str, value: float) -> dict:
    return {
        "id": question.lower().replace(" ", "-"),
        "question": question,
        "status": "COMPLETED",
        "verdict_type": "OBSERVED",
        "review_status": "ACCEPTED_AND_VERIFIED",
        "direct_answer": answer,
        "analyst_result": {
            "kind": "RANKING",
            "headline": answer,
            "executive_summary": answer,
            "details": [],
            "caveats": ["Descriptive only; this fixture does not establish causation."],
            "numbers": {
                "operation": "SUM",
                "target_column": "revenue",
                "grouping_columns": ["region"],
                "table": {"West": {"value": value, "n": 10}},
            },
        },
        "analysis_plan": {"requested_aggregation": "SUM", "group_dimension": "region"},
        "execution_record": {"aggregation": "SUM", "grouping": "region", "eligible_input_records": 10},
        "report_contract": {"interpreted": [], "calculation": [], "visualizations": []},
        "evidence": [],
        "findings": [],
    }


def test_cumulative_docx_contains_each_accepted_question_section():
    questions = [
        _question("Which region had the highest Q3 revenue?", "West: 100", 100),
        _question("Why did revenue change in Q4?", "Revenue rose by 25", 125),
    ]
    blob = export_analysis_session_to_docx(
        {"title": "Quarterly Revenue Investigation", "status": "OPEN", "report_revision": 2},
        questions,
    )
    doc = Document(BytesIO(blob))
    content = "\n".join(p.text for p in doc.paragraphs)
    assert "Which region had the highest Q3 revenue?" in content
    assert "Why did revenue change in Q4?" in content
    assert "Questions included: 2" in content


def test_cumulative_pdf_has_cover_and_one_section_per_question():
    questions = [
        _question("Which region had the highest Q3 revenue?", "West: 100", 100),
        _question("Why did revenue change in Q4?", "Revenue rose by 25", 125),
    ]
    blob = export_analysis_session_to_pdf(
        {"title": "Quarterly Revenue Investigation", "status": "OPEN", "report_revision": 2},
        questions,
    )
    reader = PdfReader(BytesIO(blob))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert len(reader.pages) >= 3
    assert "Quarterly Revenue Investigation" in extracted
    assert "Which region had the highest Q3 revenue?" in extracted
    assert "Why did revenue change in Q4?" in extracted
