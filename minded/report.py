"""Write autonomous analysis reports to Markdown and JSON."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _enum_val(v: Any) -> str:
    if v is None:
        return ""
    return getattr(v, "value", v)


def analysis_to_dict(result: Any) -> Dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unsupported analysis result type: {type(result)!r}")


def render_markdown(
    *,
    source_files: List[str],
    question: str,
    profile_summaries: List[str],
    result: Any,
) -> str:
    data = analysis_to_dict(result)
    lines = [
        "# Minded autonomous analysis",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Question: {question}",
        f"- Sources: {', '.join(source_files) or '(none)'}",
        f"- Investigation: {data.get('id', '')}",
        f"- Status: {_enum_val(data.get('status'))}",
        f"- Verdict: {_enum_val(data.get('verdict'))}",
        f"- Confidence: {_enum_val(data.get('confidence'))}",
        "",
        "## Answer",
        "",
        data.get("direct_answer") or data.get("main_finding") or "_No answer produced._",
        "",
    ]
    if profile_summaries:
        lines.extend(["## Dataset profile", ""])
        for s in profile_summaries:
            lines.append(f"- {s}")
        lines.append("")

    hyps = data.get("hypotheses") or []
    if hyps:
        lines.extend(["## Hypotheses", ""])
        for h in hyps:
            stmt = h.get("statement") if isinstance(h, dict) else getattr(h, "statement", "")
            status = h.get("status") if isinstance(h, dict) else getattr(h, "status", "")
            post = h.get("posterior_probability") if isinstance(h, dict) else getattr(h, "posterior_probability", None)
            extra = f" (posterior={post:.3f})" if isinstance(post, (int, float)) else ""
            lines.append(f"- **{status}**{extra}: {stmt}")
        lines.append("")

    steps = data.get("steps") or []
    if steps:
        lines.extend(["## Experiments run", ""])
        for s in steps:
            title = s.get("title") if isinstance(s, dict) else getattr(s, "title", "")
            desc = s.get("description") if isinstance(s, dict) else getattr(s, "description", "")
            lines.append(f"- {title}: {desc}")
        lines.append("")

    evidence = data.get("evidence") or []
    if evidence:
        lines.extend(["## Evidence", ""])
        for e in evidence[:20]:
            stmt = e.get("statement") if isinstance(e, dict) else getattr(e, "statement", "")
            lines.append(f"- {stmt}")
        lines.append("")

    bullets = data.get("executive_bullets") or []
    if bullets:
        lines.extend(["## Executive bullets", ""])
        for b in bullets:
            lines.append(f"- {b}")
        lines.append("")

    follow = data.get("suggested_followups") or []
    if follow:
        lines.extend(["## Suggested follow-ups", ""])
        for f in follow:
            lines.append(f"- {f}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    out_dir: Path,
    stem: str,
    *,
    source_files: List[str],
    question: str,
    profile_summaries: List[str],
    result: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}.json"
    payload = {
        "question": question,
        "source_files": source_files,
        "profile_summaries": profile_summaries,
        "analysis": analysis_to_dict(result),
        **(extra or {}),
    }
    md_path.write_text(
        render_markdown(
            source_files=source_files,
            question=question,
            profile_summaries=profile_summaries,
            result=result,
        ),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {"markdown": md_path, "json": json_path}
