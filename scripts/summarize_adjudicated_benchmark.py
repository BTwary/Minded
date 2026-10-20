#!/usr/bin/env python3
"""Strictly score a completed 152-case adjudicated acceptance run.

Input case JSON: [{"id": "CASE-001", "answerable": true, "expected": {...}}, ...]
Input result JSON: [{"id": "CASE-001", "outcome": "ANSWER|SAFE_REFUSAL|REFUSAL", "actual": {...}}, ...]
The runner deliberately does not infer missing answers or silently score missing cases.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _normal(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normal(v) for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [_normal(v) for v in value]
    if isinstance(value, float) and math.isfinite(value):
        return round(value, 9)
    return value


def score_cases(cases: list[dict], results: list[dict], expected_count: int = 152) -> dict:
    if len(cases) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} adjudicated cases, got {len(cases)}")
    case_ids = [str(item.get("id", "")) for item in cases]
    result_ids = [str(item.get("id", "")) for item in results]
    if any(not item for item in case_ids + result_ids):
        raise ValueError("Every case and result must have a non-empty id")
    if len(set(case_ids)) != len(case_ids) or len(set(result_ids)) != len(result_ids):
        raise ValueError("Duplicate case/result IDs are not allowed")
    if set(case_ids) != set(result_ids):
        missing = sorted(set(case_ids) - set(result_ids))
        unexpected = sorted(set(result_ids) - set(case_ids))
        raise ValueError(f"Case/result ID mismatch; missing={missing}, unexpected={unexpected}")

    result_by_id = {str(item["id"]): item for item in results}
    scored = []
    counts = {key: 0 for key in ("correct_answers", "wrong_answers", "safe_refusals", "answerable_refusals")}
    for case in cases:
        cid = str(case["id"])
        result = result_by_id[cid]
        answerable = case.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError(f"Case {cid}: answerable must be a boolean")
        outcome = str(result.get("outcome", "")).upper()
        if outcome not in {"ANSWER", "SAFE_REFUSAL", "REFUSAL"}:
            raise ValueError(f"Case {cid}: unsupported outcome {outcome!r}")
        if outcome == "ANSWER":
            category = "correct_answers" if answerable and _normal(result.get("actual")) == _normal(case.get("expected")) else "wrong_answers"
        elif outcome == "SAFE_REFUSAL" and not answerable:
            category = "safe_refusals"
        elif outcome in {"SAFE_REFUSAL", "REFUSAL"} and answerable:
            category = "answerable_refusals"
        else:
            category = "wrong_answers"
        counts[category] += 1
        scored.append({"id": cid, "category": category, "expected_answerable": answerable,
                       "outcome": outcome, "expected": case.get("expected"), "actual": result.get("actual"),
                       "reason": result.get("reason")})
    return {"benchmark": "adjudicated_acceptance", "expected_case_count": expected_count,
            "observed_case_count": len(scored), "counts": counts,
            "score_percent_answerable_correct": round(100 * counts["correct_answers"] / max(1, sum(bool(c.get("answerable")) for c in cases)), 2),
            "cases": scored}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="Original adjudicated case fixture JSON")
    parser.add_argument("--results", required=True, type=Path, help="Per-case results captured from the updated production run")
    parser.add_argument("--output", required=True, type=Path, help="Path for the scored JSON report")
    parser.add_argument("--expected-count", type=int, default=152)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    report = score_cases(cases, results, args.expected_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": report["counts"],
                      "score_percent_answerable_correct": report["score_percent_answerable_correct"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
