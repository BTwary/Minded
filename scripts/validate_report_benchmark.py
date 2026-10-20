#!/usr/bin/env python3
"""Validate a normalized MindEd report JSON against its release benchmark contract.

Usage:
  python scripts/validate_report_benchmark.py report.json
  python scripts/validate_report_benchmark.py report.json --metrics golden_metrics.json

Exit status: 0 = report-level gate passed, 1 = report rejected, 2 = input/runtime error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.analytics_core.src.reporting.report_benchmark import validate_report_contract


def _read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Normalized report contract JSON")
    parser.add_argument("--metrics", type=Path, help="Optional golden-suite benchmark metrics JSON")
    args = parser.parse_args()
    try:
        report = _read_json(args.report)
        metrics = _read_json(args.metrics) if args.metrics else None
        if not isinstance(report, dict):
            raise ValueError("Report JSON root must be an object.")
        if metrics is not None and not isinstance(metrics, dict):
            raise ValueError("Benchmark metrics JSON root must be an object.")
        result = validate_report_contract(report, metrics)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0 if result.publishable else 1
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"valid": False, "publishable": False, "grade": "FAIL", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
