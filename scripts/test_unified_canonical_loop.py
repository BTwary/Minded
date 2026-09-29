"""Canonical release proof runner.

This script deliberately does not recreate the scientific loop by manually
calling individual engines. The authoritative proof is controller-driven:
the listed tests instantiate InvestigationController and inspect persisted
state. Component-level tests remain useful, but they are not substitutes for
this proof.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PROOFS = [
    "scripts/test_autonomous_controller_messy_input.py",
    "tests/independent_release/test_defect_007_controller_closure.py",
    "tests/independent_release/test_defect_015_churn_autonomous_integration.py",
]


def main() -> int:
    print("CANONICAL CONTROLLER PROOF: real InvestigationController tests")
    for test_path in CONTROLLER_PROOFS:
        print(f"\n>>> {test_path}")
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", test_path], cwd=ROOT)
        if proc.returncode != 0:
            print(f"CANONICAL CONTROLLER PROOF: FAIL ({test_path})")
            return proc.returncode or 1
    print("CANONICAL CONTROLLER PROOF: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
