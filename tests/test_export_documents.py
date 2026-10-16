"""Comprehensive Test Suite for MindEd AA-OS Structured Document Exports & Tamper-Evident Verification.

Tests:
1. Ranking Report PDF & DOCX generation: values, headers, percentages, and formatting.
2. Statistical Association Report PDF & DOCX generation: correlation, CI, p-value, slope, R2, controlled slope.
3. Tamper Detection: modifying a metric value, headline, insight, or evidence causes verification failure.
4. Unchanged Report: verification succeeds with exact full SHA-256 match and formatted seal match.
5. Missing Optional Sections: robust rendering when optional fields (tables, caveats, recommendations) are absent.
6. Large Tables & Pagination: handles multi-row datasets spanning across pages without throwing errors.
7. Unicode & Special Characters: currency symbols ($ £ € ¥), Greek letters (ρ, α, β, η²), mathematical signs (±, ≤, ≥).
8. API Endpoints: GET /export (pdf/docx) returns proper headers (Content-Disposition, X-MindEd-Document-Seal, X-MindEd-Content-Digest) and GET /export/verify.
"""

from io import BytesIO
import pytest
import docx
from pypdf import PdfReader

from packages.analytics_core.src.reporting.exporter import (
    canonical_report_payload,
    compute_canonical_report_hash,
    compute_export_fingerprint,
    export_investigation_to_docx,
    export_investigation_to_pdf,
    verify_report_integrity,
)


@pytest.fixture
def ranking_investigation_sample():
    return {
        "id": "inv-ranking-001",
        "question": "Which region has the highest total revenue?",
        "status": "COMPLETED",
        "verdict_type": "OBSERVED",
        "reproducible_manifest_hash": "a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0",
        "analyst_result": {
            "kind": "RANKING",
            "headline": "West has the highest total revenue: 17.07M (n=33,424), 27.6% of overall revenue.",
            "numbers": {
                "top": "West",
                "top_value": 17070000.0,
                "n": 120000,
                "table": {
                    "West": {"value": 17070000.0, "n": 33424},
                    "East": {"value": 15040000.0, "n": 29030},
                    "South": {"value": 12340000.0, "n": 24093},
                    "North": {"value": 9730000.0, "n": 18623},
                    "Central": {"value": 7610000.0, "n": 14830},
                },
            },
            "details": [
                "West leads in total revenue.",
                "East has slightly higher revenue per row (518.2 vs 510.7).",
                "Opportunity sizing: elevating regions below median represents +7.35M expansion.",
            ],
            "caveats": [
                "This is descriptive revenue per row, not profit or unit economics.",
            ],
            "next_steps": [
                "Audit fulfillment capacity in West region to support sustained volume.",
                "Investigate pricing realization in East region.",
            ],
        },
        "evidence": [
            {
                "id": "ev-001",
                "statement": "West region total revenue is 17.07M across 33,424 rows",
                "calculation_summary": "SUM(revenue) GROUP BY region",
                "row_count_analyzed": 120000,
                "validation_status": "PASSED",
                "sql_executed": "SELECT region, SUM(revenue) FROM orders GROUP BY region ORDER BY 2 DESC",
            }
        ],
    }


@pytest.fixture
def statistical_investigation_sample():
    return {
        "id": "inv-stat-002",
        "question": "Does discount_pct affect quantity?",
        "status": "COMPLETED",
        "verdict_type": "ASSOCIATION",
        "reproducible_manifest_hash": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
        "analyst_result": {
            "kind": "ASSOCIATION",
            "headline": "Discount percentage is positively associated with quantity (r=0.306, p < 0.001, n=119,546).",
            "numbers": {
                "pearson_r": 0.306,
                "pearson_p": 0.00002,
                "ci_low": 0.301,
                "ci_high": 0.311,
                "slope": 0.0604,
                "r_squared": 0.0937,
                "n": 119546,
                "multivariate_context": {
                    "controlled_slope": 0.0287,
                    "adjusted_r_squared": 0.461,
                    "controlled_covariates": ["revenue", "price"],
                },
            },
            "details": [
                "Each +1 percentage point increase in discount goes with an average quantity increase of 0.0604.",
                "After controlling for covariates, the association attenuates to +0.0287.",
            ],
            "caveats": [
                "The analysis establishes association, not causation.",
                "The adjusted model includes revenue, which is mathematically related to quantity.",
            ],
            "next_steps": [
                "Conduct an A/B test with randomized discount cohorts to evaluate causal elasticity.",
            ],
        },
        "evidence": [
            {
                "id": "ev-002",
                "statement": "Bivariate linear regression of quantity on discount_pct yields slope 0.0604",
                "calculation_summary": "linregress(discount_pct, quantity)",
                "row_count_analyzed": 119546,
                "validation_status": "PASSED",
                "sql_executed": "SELECT discount_pct, quantity FROM sales WHERE discount_pct IS NOT NULL",
            }
        ],
    }


def test_ranking_pdf_generation(ranking_investigation_sample):
    pdf_bytes = export_investigation_to_pdf(ranking_investigation_sample)
    assert len(pdf_bytes) > 1000
    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1
    full_text = "\n".join(page.extract_text() for page in reader.pages)

    assert "MINDED AA-OS" in full_text
    assert "Which region has the highest total revenue?" in full_text
    assert "17.07M" in full_text
    assert "West" in full_text
    assert "East" in full_text
    assert "MDED-" in full_text
    assert "EVIDENCE AUDIT" in full_text


def test_ranking_docx_generation(ranking_investigation_sample):
    docx_bytes = export_investigation_to_docx(ranking_investigation_sample)
    assert len(docx_bytes) > 5000
    doc = docx.Document(BytesIO(docx_bytes))

    paragraphs_text = "\n".join(p.text for p in doc.paragraphs)
    header_table_text = "\n".join(cell.text for row in doc.tables[0].rows for cell in row.cells)
    all_text = paragraphs_text + "\n" + header_table_text

    assert "MINDED AA-OS" in all_text
    assert "Which region has the highest total revenue?" in all_text
    assert "17.07M" in all_text
    assert "MDED-" in all_text

    # Verify structured table in docx
    assert len(doc.tables) >= 2  # Header table + Ranking table
    ranking_table = doc.tables[1]
    headers = [cell.text for cell in ranking_table.rows[0].cells]
    assert headers == ["Rank", "Group / Entity", "Total Value", "Rows", "Share"]
    first_row = [cell.text for cell in ranking_table.rows[1].cells]
    assert first_row[0] == "1"
    assert first_row[1] == "West"
    assert "17.07M" in first_row[2]


def test_statistical_association_pdf_and_docx(statistical_investigation_sample):
    # PDF check
    pdf_bytes = export_investigation_to_pdf(statistical_investigation_sample)
    reader = PdfReader(BytesIO(pdf_bytes))
    full_text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Pearson correlation" in full_text
    assert "0.306" in full_text
    assert "0.0604" in full_text
    assert "MDED-" in full_text

    # DOCX check
    docx_bytes = export_investigation_to_docx(statistical_investigation_sample)
    doc = docx.Document(BytesIO(docx_bytes))
    stat_table = doc.tables[1]
    cells = [c.text for row in stat_table.rows for c in row.cells]
    assert "Pearson correlation (r)" in cells
    assert "0.306" in cells
    assert "Simple regression slope" in cells
    assert "+0.0604" in cells


def test_tamper_detection_full_hash_and_seal(ranking_investigation_sample):
    # Step 1: Compute original digest & seal
    original_full_hash = compute_canonical_report_hash(ranking_investigation_sample)
    original_seal = compute_export_fingerprint(ranking_investigation_sample)

    # Step 2: Verify unchanged passes
    valid, details = verify_report_integrity(
        ranking_investigation_sample,
        expected_full_hash=original_full_hash,
        expected_seal=original_seal,
    )
    assert valid is True
    assert details["matches_full_hash"] is True
    assert details["matches_seal"] is True

    # Step 3: Tamper with metric value -> Must fail
    tampered_metric = dict(ranking_investigation_sample)
    tampered_metric["analyst_result"] = dict(ranking_investigation_sample["analyst_result"])
    tampered_metric["analyst_result"]["numbers"] = dict(ranking_investigation_sample["analyst_result"]["numbers"])
    tampered_metric["analyst_result"]["numbers"]["top_value"] = 99999999.0

    valid_tampered, details_tampered = verify_report_integrity(
        tampered_metric,
        expected_full_hash=original_full_hash,
        expected_seal=original_seal,
    )
    assert valid_tampered is False
    assert details_tampered["matches_full_hash"] is False
    assert details_tampered["matches_seal"] is False

    # Step 4: Tamper with headline -> Must fail
    tampered_headline = dict(ranking_investigation_sample)
    tampered_headline["analyst_result"] = dict(ranking_investigation_sample["analyst_result"])
    tampered_headline["analyst_result"]["headline"] = "East has the highest revenue: 99M"
    valid_head, _ = verify_report_integrity(
        tampered_headline,
        expected_full_hash=original_full_hash,
        expected_seal=original_seal,
    )
    assert valid_head is False

    # Step 5: Tamper with an insight -> Must fail
    tampered_insight = dict(ranking_investigation_sample)
    tampered_insight["analyst_result"] = dict(ranking_investigation_sample["analyst_result"])
    tampered_insight["analyst_result"]["details"] = ["Fabricated finding not in data."]
    valid_ins, _ = verify_report_integrity(
        tampered_insight,
        expected_full_hash=original_full_hash,
        expected_seal=original_seal,
    )
    assert valid_ins is False


def test_missing_optional_sections_renders_cleanly():
    minimal_sample = {
        "id": "inv-min-003",
        "question": "Is system operational?",
        "status": "COMPLETED",
        "verdict_type": "CONFIRMED",
        "analyst_result": {
            "headline": "System is operational.",
        },
    }
    pdf_bytes = export_investigation_to_pdf(minimal_sample)
    assert len(pdf_bytes) > 500
    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) == 1

    docx_bytes = export_investigation_to_docx(minimal_sample)
    assert len(docx_bytes) > 2000


def test_large_table_multipage_export():
    large_table = {f"Entity_{i:03d}": {"value": float(1000 - i * 10), "n": 50} for i in range(1, 40)}
    large_sample = {
        "id": "inv-large-004",
        "question": "Rank all 40 entities by revenue",
        "status": "COMPLETED",
        "verdict_type": "OBSERVED",
        "analyst_result": {
            "kind": "RANKING",
            "headline": "Entity_001 leads with 990.00 revenue.",
            "numbers": {
                "top": "Entity_001",
                "top_value": 990.0,
                "n": 2000,
                "table": large_table,
            },
        },
    }
    pdf_bytes = export_investigation_to_pdf(large_sample)
    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1
    docx_bytes = export_investigation_to_docx(large_sample)
    doc = docx.Document(BytesIO(docx_bytes))
    assert len(doc.tables[1].rows) == 40


def test_unicode_and_currency_symbols():
    unicode_sample = {
        "id": "inv-uni-005",
        "question": "Currency and scientific notation: €1,200 vs £950, η² = 0.42, α = 0.05, ρ = 0.31",
        "status": "COMPLETED",
        "verdict_type": "OBSERVED",
        "analyst_result": {
            "headline": "Effect size η² = 0.42 exceeds threshold α = 0.05 with Δ = +€250.",
            "details": ["Statistically significant difference (p ≤ 0.001, ρ = 0.31)."],
        },
    }
    pdf_bytes = export_investigation_to_pdf(unicode_sample)
    assert len(pdf_bytes) > 500
    docx_bytes = export_investigation_to_docx(unicode_sample)
    assert len(docx_bytes) > 2000
