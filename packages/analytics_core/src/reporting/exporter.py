"""Structured Document Export Engine for MindEd AA-OS (PDF and DOCX).

Renders the canonical structured analytical answer with:
- MindEd Emblem / Logo and Cryptographic Provenance Fingerprint
- Canonical SHA-256 Content Digest & Verifiable Document Seal
- Status, Claim & Epistemic Verification Badges
- Headline (One sentence answer)
- Key Result (Primary metric value, sample size, observation counts)
- Task-Specific Structured Table (Rankings, Statistical associations, Group comparisons)
- Key Insights (crisp one-line insights)
- Limitations & Assumptions (epistemic bounds, non-causal caveats)
- Recommended Actions (numbered operational recommendations)
- Audited Evidence & Provenance (calculations, rows analyzed, SQL queries, SHA-256 manifest)
"""

from io import BytesIO
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from PIL import Image, ImageDraw

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def _fmt_compact(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    try:
        val = float(val)
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:.2f}M"
        if abs(val) >= 1_000:
            return f"{val:,.2f}"
        return f"{val:,.4f}"
    except (ValueError, TypeError):
        return str(val)


def canonical_report_payload(inv_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a canonical, normalized dictionary representation of the report content.

    Any material change in the investigation identity, question, headline, metrics,
    tables, insights, limitations, recommended actions, manifest hash, or evidence
    will alter this canonical payload.
    """
    analyst_result = inv_data.get("analyst_result") or {}
    numbers = analyst_result.get("numbers") or {}
    evidence_items = inv_data.get("evidence") or []

    # Sort evidence items deterministically by ID or statement
    sorted_evidence = []
    for ev in evidence_items:
        sorted_evidence.append({
            "id": str(ev.get("id", "")),
            "statement": str(ev.get("statement", "")),
            "calculation_summary": str(ev.get("calculation_summary", "")),
            "row_count_analyzed": ev.get("row_count_analyzed"),
            "validation_status": str(ev.get("validation_status", "UNVERIFIED")),
            "sql_executed": str(ev.get("sql_executed") or ""),
        })
    sorted_evidence.sort(key=lambda x: (x["id"], x["statement"]))

    return {
        "investigation_id": str(inv_data.get("id", "")),
        "question": str(inv_data.get("question", "")).strip(),
        "status": str(inv_data.get("status", "")).strip().upper(),
        "verdict": str(inv_data.get("verdict_type") or inv_data.get("verdict") or "").strip().upper(),
        "headline": str(analyst_result.get("headline") or inv_data.get("headline") or inv_data.get("direct_answer") or "").strip(),
        "kind": str(analyst_result.get("kind") or "ANALYSIS").strip().upper(),
        "numbers": {
            k: v for k, v in sorted(numbers.items(), key=lambda item: item[0])
            if isinstance(v, (str, int, float, bool, dict, list))
        },
        "insights": [str(d).strip() for d in (analyst_result.get("details") or []) if str(d).strip()][:3],
        "limitations": [str(c).strip() for c in (analyst_result.get("caveats") or []) if str(c).strip()],
        "actions": [str(a).strip() for a in (analyst_result.get("next_steps") or []) if str(a).strip()],
        "manifest_hash": str(inv_data.get("reproducible_manifest_hash") or "").strip(),
        "evidence": sorted_evidence,
    }


def compute_canonical_report_hash(inv_data: Dict[str, Any]) -> str:
    """Compute full 64-character SHA-256 hash of the canonical report payload."""
    payload = canonical_report_payload(inv_data)
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_export_fingerprint(inv_data: Dict[str, Any]) -> str:
    """Compute the user-facing short document seal from the canonical full report hash.

    Format: MDED-XXXX-XXXX-XXXX-XXXX (first 16 hex characters).
    Backed by the 64-character canonical report hash for complete tamper-evident audit.
    """
    full_hash = compute_canonical_report_hash(inv_data)
    return f"MDED-{full_hash[:4].upper()}-{full_hash[4:8].upper()}-{full_hash[8:12].upper()}-{full_hash[12:16].upper()}"


def verify_report_integrity(
    inv_data: Dict[str, Any],
    expected_full_hash: Optional[str] = None,
    expected_seal: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Verify whether a report's data matches an expected full hash and/or visible seal.

    Returns:
        (is_valid, details_dict)
    """
    computed_full_hash = compute_canonical_report_hash(inv_data)
    computed_seal = f"MDED-{computed_full_hash[:4].upper()}-{computed_full_hash[4:8].upper()}-{computed_full_hash[8:12].upper()}-{computed_full_hash[12:16].upper()}"

    matches_full = (expected_full_hash.lower() == computed_full_hash.lower()) if expected_full_hash else True
    matches_seal = (expected_seal.upper() == computed_seal.upper()) if expected_seal else True

    is_valid = matches_full and matches_seal
    return is_valid, {
        "is_valid": is_valid,
        "computed_full_hash": computed_full_hash,
        "computed_seal": computed_seal,
        "matches_full_hash": matches_full,
        "matches_seal": matches_seal,
    }


def _generate_minded_logo_png(size: int = 120) -> bytes:
    """Draw a clean, crisp MindEd hexagonal lattice prism emblem as PNG bytes."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Coordinates for hexagon
    cx, cy = size / 2, size / 2
    r = size * 0.44

    import math
    points = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        points.append((px, py))

    # Dark graphite backing
    draw.polygon(points, fill=(17, 24, 39, 255), outline=(245, 166, 35, 255), width=max(2, int(size * 0.05)))

    # Amber interior lattice lines
    amber_inner = (255, 204, 112, 230)
    w_line = max(1, int(size * 0.035))
    draw.line([points[0], points[3]], fill=amber_inner, width=w_line)
    draw.line([points[1], points[4]], fill=amber_inner, width=w_line)
    draw.line([points[2], points[5]], fill=amber_inner, width=w_line)

    # Core center circle (representing nucleus of intelligence)
    r_core = size * 0.12
    draw.ellipse(
        [cx - r_core, cy - r_core, cx + r_core, cy + r_core],
        fill=(245, 166, 35, 255),
        outline=(255, 255, 255, 200),
        width=1,
    )

    out = BytesIO()
    img.save(out, format="PNG")
    out.seek(0)
    return out.getvalue()


def _extract_table_data(kind: str, numbers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    kind = (kind or "").upper()
    if kind == "RANKING" and "table" in numbers and isinstance(numbers["table"], dict):
        raw_table = numbers["table"]
        entries = []
        for name, data in raw_table.items():
            val = data.get("value", 0.0) if isinstance(data, dict) else 0.0
            n_rows = data.get("n", 0) if isinstance(data, dict) else 0
            entries.append({"name": str(name), "value": float(val), "rows": int(n_rows)})
        entries.sort(key=lambda x: x["value"], reverse=True)
        grand = sum(x["value"] for x in entries)
        rows = []
        for idx, item in enumerate(entries, start=1):
            share_str = f"{(item['value'] / grand * 100):.1f}%" if grand > 0 else "N/A"
            rows.append([
                str(idx),
                item["name"],
                _fmt_compact(item["value"]),
                f"{item['rows']:,}",
                share_str,
            ])
        return {
            "title": "Ranking by Metric",
            "headers": ["Rank", "Group / Entity", "Total Value", "Rows", "Share"],
            "rows": rows,
        }

    if kind in ("ASSOCIATION", "CORRELATION", "REGRESSION"):
        rows = []
        if "pearson_r" in numbers and numbers["pearson_r"] is not None:
            rows.append(["Pearson correlation (r)", f"{float(numbers['pearson_r']):.3f}"])
        if "ci_low" in numbers and "ci_high" in numbers:
            rows.append(["95% confidence interval", f"{float(numbers['ci_low']):.3f} to {float(numbers['ci_high']):.3f}"])
        if "pearson_p" in numbers and numbers["pearson_p"] is not None:
            p = float(numbers["pearson_p"])
            rows.append(["P-value", "p < 0.001" if p < 0.001 else f"{p:.4f}"])
        if "slope" in numbers and numbers["slope"] is not None:
            slope = float(numbers["slope"])
            rows.append(["Simple regression slope", f"{slope:+.4f}"])
        if "r_squared" in numbers and numbers["r_squared"] is not None:
            rows.append(["Variation explained (R²)", f"{(float(numbers['r_squared']) * 100):.2f}%"])

        multiv = numbers.get("multivariate_context")
        if isinstance(multiv, dict):
            if "controlled_slope" in multiv:
                rows.append(["Adjusted slope (controlled)", f"{float(multiv['controlled_slope']):+.4f}"])
            if "adjusted_r_squared" in multiv:
                rows.append(["Adjusted model R²", f"{(float(multiv['adjusted_r_squared']) * 100):.2f}%"])

        if rows:
            return {
                "title": "Statistical Evidence",
                "headers": ["Metric", "Result"],
                "rows": rows,
            }

    if "table" in numbers and isinstance(numbers["table"], dict):
        raw_table = numbers["table"]
        rows = []
        for k, v in list(raw_table.items())[:30]:
            val_str = _fmt_compact(v.get("value")) if isinstance(v, dict) else str(v)
            rows.append([str(k), val_str])
        return {
            "title": "Results Table",
            "headers": ["Item / Group", "Value"],
            "rows": rows,
        }

    return None


def export_investigation_to_pdf(investigation_data: Dict[str, Any]) -> bytes:
    """Generate professional publication-quality PDF matching MindEd design standard."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()

    brand_style = ParagraphStyle(
        "BrandHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#B45309"),
    )
    fingerprint_style = ParagraphStyle(
        "Fingerprint",
        parent=styles["Normal"],
        fontName="Courier-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569"),
        alignment=2,
    )
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#111827"),
    )
    headline_style = ParagraphStyle(
        "Headline",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1E293B"),
    )
    section_h_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#475569"),
        textTransform="uppercase",
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#92400E"),
    )
    metric_val_style = ParagraphStyle(
        "MetricVal",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#D97706"),
    )
    sub_metric_style = ParagraphStyle(
        "SubMetric",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#64748B"),
    )

    story = []

    # 1. Header with Emblem Logo + Fingerprint Banner
    logo_bytes = _generate_minded_logo_png(size=80)
    logo_img = RLImage(BytesIO(logo_bytes), width=0.42 * inch, height=0.42 * inch)
    fingerprint_code = compute_export_fingerprint(investigation_data)
    canonical_hash = compute_canonical_report_hash(investigation_data)

    header_table_data = [
        [
            logo_img,
            Paragraph("<b>MINDED AA-OS</b><br/><font color='#64748B' size=8>Autonomous Analytical Operating System</font>", brand_style),
            Paragraph(f"<b>AUTHENTIC EXPORT SEAL</b><br/><font color='#D97706'><b>{fingerprint_code}</b></font>", fingerprint_style),
        ]
    ]
    header_table = Table(header_table_data, colWidths=[0.55 * inch, 3.6 * inch, 2.85 * inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#F5A623"), spaceAfter=10))

    # Question Title
    question = investigation_data.get("question", "Analytical Investigation")
    story.append(Paragraph(question, title_style))
    story.append(Spacer(1, 4))

    verdict = investigation_data.get("verdict_type") or investigation_data.get("verdict") or "OBSERVED"
    status = investigation_data.get("status", "COMPLETED")
    date_str = datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC")
    meta_text = f"<b>Status:</b> {status} &nbsp;|&nbsp; <b>Verdict:</b> {verdict} &nbsp;|&nbsp; <b>Certified:</b> {date_str}"
    story.append(Paragraph(meta_text, sub_metric_style))
    story.append(Spacer(1, 10))

    # Analyst Result Extraction
    analyst_result = investigation_data.get("analyst_result") or {}
    headline = analyst_result.get("headline") or investigation_data.get("headline") or investigation_data.get("direct_answer") or ""
    numbers = analyst_result.get("numbers") or {}
    kind = analyst_result.get("kind") or "ANALYSIS"

    # 2. Headline
    if headline:
        story.append(Paragraph("HEADLINE", section_h_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph(headline, headline_style))
        story.append(Spacer(1, 10))

    # 3. Key Result Metric Banner
    top_val = numbers.get("top_value")
    pearson_r = numbers.get("pearson_r")
    if top_val is not None or pearson_r is not None:
        val_str = _fmt_compact(top_val) if top_val is not None else f"r = {float(pearson_r):.3f}"
        sub_str = f"Category: {kind} · Sample: {numbers.get('n', 'N/A')} observations"
        story.append(Paragraph(val_str, metric_val_style))
        story.append(Paragraph(sub_str, sub_metric_style))
        story.append(Spacer(1, 10))

    # 4. Structured Results Table
    table_meta = _extract_table_data(kind, numbers)
    if table_meta:
        story.append(Paragraph(table_meta["title"], section_h_style))
        story.append(Spacer(1, 4))
        table_content = [table_meta["headers"]] + table_meta["rows"]
        t = Table(table_content, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
            ("TOPPADDING", (0, 0), (-1, -1), 4.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#1E293B")),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    # 5. Key Insights
    details = analyst_result.get("details") or []
    if details:
        story.append(Paragraph("KEY INSIGHTS", section_h_style))
        story.append(Spacer(1, 4))
        for d in details[:3]:
            story.append(Paragraph(f"• {d}", body_style))
            story.append(Spacer(1, 2.5))
        story.append(Spacer(1, 8))

    # 6. Limitations & Assumptions
    caveats = analyst_result.get("caveats") or []
    if caveats or kind in ("ASSOCIATION", "CORRELATION"):
        story.append(Paragraph("LIMITATIONS & ASSUMPTIONS", section_h_style))
        story.append(Spacer(1, 4))
        all_caveats = list(caveats)
        if kind in ("ASSOCIATION", "CORRELATION") and not any("causation" in c.lower() for c in all_caveats):
            all_caveats.append("The analysis establishes statistical association, not mechanistic causation.")
        for c in all_caveats:
            story.append(Paragraph(f"• {c}", callout_style))
            story.append(Spacer(1, 2.5))
        story.append(Spacer(1, 8))

    # 7. Recommended Actions
    next_steps = analyst_result.get("next_steps") or []
    if next_steps:
        story.append(Paragraph("RECOMMENDED ACTIONS", section_h_style))
        story.append(Spacer(1, 4))
        for idx, act in enumerate(next_steps, start=1):
            story.append(Paragraph(f"{idx}. {act}", body_style))
            story.append(Spacer(1, 2.5))
        story.append(Spacer(1, 8))

    # 8. Evidence & Provenance Audit with Security Seal
    evidence = investigation_data.get("evidence") or []
    manifest_hash = investigation_data.get("reproducible_manifest_hash") or "N/A"
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceAfter=8))
    story.append(Paragraph("EVIDENCE AUDIT & PROVENANCE", section_h_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Full Canonical Digest (SHA-256):</b> <font face='Courier' size=7.5>{canonical_hash}</font>", sub_metric_style))
    story.append(Paragraph(f"<b>Manifest Hash:</b> <font face='Courier' size=8>{manifest_hash}</font> &nbsp;|&nbsp; <b>Seal:</b> <font face='Courier' color='#D97706'><b>{fingerprint_code}</b></font>", sub_metric_style))
    story.append(Spacer(1, 4))

    for ev in evidence[:5]:
        stmt = ev.get("statement", "Evidence Item")
        v_status = ev.get("validation_status", "UNVERIFIED")
        calc = ev.get("calculation_summary", "")
        rows_n = ev.get("row_count_analyzed")
        rows_str = f" | Rows analyzed: {rows_n:,}" if rows_n else ""
        story.append(Paragraph(f"• <b>{stmt}</b> [{v_status}]{rows_str}", body_style))
        if calc:
            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<i>Calculation:</i> {calc}", sub_metric_style))
        story.append(Spacer(1, 2))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def export_investigation_to_docx(investigation_data: Dict[str, Any]) -> bytes:
    """Generate clean structured Microsoft Word document (.docx) with MindEd branding and fingerprint."""
    doc = docx.Document()

    fingerprint_code = compute_export_fingerprint(investigation_data)
    canonical_hash = compute_canonical_report_hash(investigation_data)
    logo_bytes = _generate_minded_logo_png(size=64)

    # Document Header Table (Emblem / Logo + Brand + Fingerprint)
    tbl_hdr = doc.add_table(rows=1, cols=2)
    tbl_hdr.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_left, c_right = tbl_hdr.rows[0].cells

    p_logo = c_left.paragraphs[0]
    run_img = p_logo.add_run()
    run_img.add_picture(BytesIO(logo_bytes), width=Inches(0.45), height=Inches(0.45))
    run_brand = p_logo.add_run("  MINDED AA-OS\n  Autonomous Analytical Intelligence")
    run_brand.font.size = Pt(9.5)
    run_brand.font.color.rgb = RGBColor(180, 83, 9)
    run_brand.bold = True

    p_fp = c_right.paragraphs[0]
    p_fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run_fp_label = p_fp.add_run("AUTHENTIC EXPORT SEAL\n")
    run_fp_label.font.size = Pt(8)
    run_fp_label.font.color.rgb = RGBColor(100, 116, 139)
    run_fp_val = p_fp.add_run(fingerprint_code)
    run_fp_val.font.size = Pt(10)
    run_fp_val.font.color.rgb = RGBColor(217, 119, 6)
    run_fp_val.bold = True

    doc.add_paragraph()

    # Inquiry Title
    h1 = doc.add_heading(level=1)
    run_title = h1.add_run(investigation_data.get("question", "Analytical Investigation Report"))
    run_title.font.size = Pt(17)
    run_title.bold = True

    # Metadata
    verdict = investigation_data.get("verdict_type") or investigation_data.get("verdict") or "OBSERVED"
    status = investigation_data.get("status", "COMPLETED")
    p_meta = doc.add_paragraph()
    p_meta.add_run(f"Status: {status}  |  Verdict: {verdict}  |  Certified: {datetime.now(timezone.utc).strftime('%b %d, %Y UTC')}").italic = True

    analyst_result = investigation_data.get("analyst_result") or {}
    headline = analyst_result.get("headline") or investigation_data.get("headline") or investigation_data.get("direct_answer") or ""
    numbers = analyst_result.get("numbers") or {}
    kind = analyst_result.get("kind") or "ANALYSIS"

    # Headline
    if headline:
        doc.add_heading("Headline", level=2)
        p_head = doc.add_paragraph()
        run_h = p_head.add_run(headline)
        run_h.bold = True
        run_h.font.size = Pt(12)

    # Key Result Metric
    top_val = numbers.get("top_value")
    pearson_r = numbers.get("pearson_r")
    if top_val is not None or pearson_r is not None:
        val_str = _fmt_compact(top_val) if top_val is not None else f"r = {float(pearson_r):.3f}"
        p_metric = doc.add_paragraph()
        r_m = p_metric.add_run(f"{val_str}\n")
        r_m.bold = True
        r_m.font.size = Pt(19)
        r_m.font.color.rgb = RGBColor(217, 119, 6)
        p_metric.add_run(f"Category: {kind}  |  Observations: {numbers.get('n', 'N/A')}").italic = True

    # Results Table
    table_meta = _extract_table_data(kind, numbers)
    if table_meta:
        doc.add_heading(table_meta["title"], level=2)
        headers = table_meta["headers"]
        rows = table_meta["rows"]
        t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.LEFT

        # Header Row
        hdr_cells = t.rows[0].cells
        for i, header_text in enumerate(headers):
            hdr_cells[i].text = header_text
            for p in hdr_cells[i].paragraphs:
                for r in p.runs:
                    r.bold = True
                    r.font.size = Pt(9.5)

        # Data Rows
        for row_idx, row_data in enumerate(rows):
            row_cells = t.rows[row_idx + 1].cells
            for col_idx, cell_value in enumerate(row_data):
                row_cells[col_idx].text = str(cell_value)
                for p in row_cells[col_idx].paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(9.5)

    # Key Insights
    details = analyst_result.get("details") or []
    if details:
        doc.add_heading("Key Insights", level=2)
        for d in details[:3]:
            doc.add_paragraph(d, style="List Bullet")

    # Limitations & Assumptions
    caveats = analyst_result.get("caveats") or []
    if caveats or kind in ("ASSOCIATION", "CORRELATION"):
        doc.add_heading("Limitations & Assumptions", level=2)
        all_caveats = list(caveats)
        if kind in ("ASSOCIATION", "CORRELATION") and not any("causation" in c.lower() for c in all_caveats):
            all_caveats.append("The analysis establishes statistical association, not mechanistic causation.")
        for c in all_caveats:
            doc.add_paragraph(c, style="List Bullet")

    # Recommended Actions
    next_steps = analyst_result.get("next_steps") or []
    if next_steps:
        doc.add_heading("Recommended Actions", level=2)
        for act in next_steps:
            doc.add_paragraph(act, style="List Number")

    # Evidence & Provenance
    evidence = investigation_data.get("evidence") or []
    manifest_hash = investigation_data.get("reproducible_manifest_hash") or "N/A"
    doc.add_heading("Evidence Audit & Provenance", level=2)
    p_prov = doc.add_paragraph()
    p_prov.add_run(f"Canonical Digest (SHA-256): {canonical_hash}\nManifest Hash: {manifest_hash}\nDocument Seal: {fingerprint_code}").italic = True

    for ev in evidence[:5]:
        stmt = ev.get("statement", "Evidence Item")
        v_status = ev.get("validation_status", "UNVERIFIED")
        calc = ev.get("calculation_summary", "")
        p_ev = doc.add_paragraph(f"{stmt} [{v_status}]", style="List Bullet")
        if calc:
            p_ev.add_run(f"\nCalculation: {calc}").italic = True

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
