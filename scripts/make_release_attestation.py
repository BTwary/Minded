"""Bind release evidence to the exact artifact that produced it.

The 32-question stress result files are bare lists; on their own they do not say which source,
dependency lock, datasets, or manifest produced them. This script writes (or verifies)
``release_evidence/ATTESTATION.json`` containing SHA-256 digests of:

  * the source tree (packages/, apps/, scripts/, tests/, migrations/, minded/, packaging/ -- sorted,
    raw bytes, caches, build output and compiled binaries excluded). packaging/ covers the Windows
    build scripts, installer sources and requirements-desktop.txt
  * every root requirements*.txt (including requirements-lock.txt), alembic.ini, the release
    manifest and the release-status document
  * the release-gate JSON report itself, copied next to the attestation so its SHA-256 can be
    re-checked (release_evidence/GATE_REPORT.json)
  * data/certification/checksums.sha256 and every certification dataset
  * real_stress_results_raw.json / real_stress_results_clean.json
  * (optional) the release-gate JSON report, with its decision and per-check result counts

Usage:
    python -m scripts.make_release_attestation --gate-report gate_report.json   # write
    python -m scripts.make_release_attestation --verify                         # check current tree

This is evidence binding, not a signature: it detects drift between the tree and the recorded
evidence. It does not prove the evidence is genuine -- re-run the release gate for that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "release_evidence" / "ATTESTATION.json"
GATE_REPORT_COPY = ROOT / "release_evidence" / "GATE_REPORT.json"
SOURCE_DIRS = ("packages", "apps", "scripts", "tests", "migrations", "minded", "packaging")
EXCLUDE_PARTS = {"__pycache__", "node_modules", ".next", ".pytest_cache", "out", "data_store", "bundle", "build", "dist", "temp_p2_test"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".tsbuildinfo", ".db", ".log", ".exe", ".g.cs")
SINGLE_FILES = (
    "alembic.ini",
    "requirements-lock.txt",
    "tests/independent_release/manifest.toml",
    "docs/AAOS_V1_0_0_RELEASE_STATUS.md",
    "data/certification/checksums.sha256",
    "real_stress_results_raw.json",
    "real_stress_results_clean.json",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_tree_digest() -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for d in SOURCE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or EXCLUDE_PARTS & set(p.relative_to(ROOT).parts):
                continue
            if p.name.endswith(EXCLUDE_SUFFIXES):
                continue
            h.update(f"{p.relative_to(ROOT).as_posix()}\0{_sha(p)}\n".encode())
            n += 1
    return h.hexdigest(), n


def _single_files() -> list[str]:
    names = set(SINGLE_FILES)
    names.update(p.name for p in ROOT.glob("requirements*.txt"))
    return sorted(names)


def build(gate_report: Path | None) -> dict:
    tree, count = source_tree_digest()
    att: dict = {
        "schema": 1,
        "source_tree_sha256": tree,
        "source_files": count,
        "files": {rel: _sha(ROOT / rel) for rel in _single_files() if (ROOT / rel).exists()},
        "certification_datasets": {
            p.name: _sha(p) for p in sorted((ROOT / "data" / "certification").glob("*.csv"))
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if gate_report is not None:
        rep = json.loads(gate_report.read_text(encoding="utf-8"))
        att["gate"] = {
            "report_sha256": _sha(gate_report),
            "report_file": GATE_REPORT_COPY.relative_to(ROOT).as_posix(),
            "decision": rep.get("release_decision") or rep.get("decision") or rep.get("status"),
            "check_results": dict(Counter(v.get("result") for v in rep.get("checks", {}).values())),
            "clean_room_verified": (rep.get("clean_room") or {}).get("verified"),
        }
    return att


def verify() -> int:
    if not OUT.exists():
        print(f"ATTESTATION: FAIL - {OUT} missing")
        return 1
    recorded = json.loads(OUT.read_text(encoding="utf-8"))
    current = build(None)
    problems = []
    if recorded.get("source_tree_sha256") != current["source_tree_sha256"]:
        problems.append("source tree changed since attestation")
    for key in ("files", "certification_datasets"):
        for name, digest in recorded.get(key, {}).items():
            if current[key].get(name) != digest:
                problems.append(f"{key}:{name} differs")
    gate = recorded.get("gate")
    if gate:
        if not GATE_REPORT_COPY.is_file():
            problems.append(f"gate report {GATE_REPORT_COPY.relative_to(ROOT).as_posix()} is not shipped with the attestation")
        elif _sha(GATE_REPORT_COPY) != gate.get("report_sha256"):
            problems.append("shipped gate report does not match recorded report_sha256")
    if problems:
        print("ATTESTATION: FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("ATTESTATION: PASS (evidence matches current tree)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-report", type=Path)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        return verify()
    att = build(args.gate_report)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if args.gate_report is not None and args.gate_report.resolve() != GATE_REPORT_COPY.resolve():
        shutil.copyfile(args.gate_report, GATE_REPORT_COPY)
    OUT.write_text(json.dumps(att, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} (tree {att['source_tree_sha256'][:12]}…, {att['source_files']} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
