"""Fail-closed, manifest-driven release gate for MindEd AA-OS."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "independent_release" / "manifest.toml"


@dataclass
class CheckResult:
    id: str
    result: str  # PASS | FAIL | MISSING | ERROR | TIMEOUT | SKIPPED
    detail: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0


@dataclass
class GateReport:
    phase: str = "release"
    status: str = "BLOCKED"
    release_decision: str = "RELEASE-BLOCKED"
    checks: dict[str, CheckResult] = field(default_factory=dict)
    failed_checks: list[str] = field(default_factory=list)
    missing_checks: list[str] = field(default_factory=list)
    manifest_check_count: int = 0
    checks_executed: int = 0
    release_status: dict[str, Any] = field(default_factory=dict)
    serialization_audit: dict[str, Any] = field(default_factory=dict)
    dependency_lock: dict[str, Any] = field(default_factory=dict)
    frontend_lock: dict[str, Any] = field(default_factory=dict)
    migration_drift: dict[str, Any] = field(default_factory=dict)
    clean_room: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except Exception:
        return []
    checks = data.get("check", [])
    return checks if isinstance(checks, list) else []


def _substitute(value: str) -> str:
    return value.replace("{python}", sys.executable).replace("{root}", str(ROOT))


def _command_missing_reason(entry: dict[str, Any], argv: list[str]) -> str | None:
    kind = str(entry.get("kind", "command"))
    if kind == "pytest":
        # Every explicit Python test file/directory named by the manifest must exist.
        for raw in argv:
            if raw.startswith("-"):
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            if candidate.suffix == ".py" and not candidate.exists():
                return f"required test file missing: {raw}"
            if raw.endswith("/") and not candidate.exists():
                return f"required test directory missing: {raw}"
    elif kind == "python_module":
        if "-m" in argv:
            index = argv.index("-m") + 1
            if index < len(argv):
                module_name = argv[index]
                parts = module_name.split(".")
                module_file = ROOT.joinpath(*parts).with_suffix(".py")
                package_init = ROOT.joinpath(*parts, "__init__.py")
                if module_file.exists() or package_init.exists():
                    return None
                try:
                    if importlib.util.find_spec(module_name) is None:
                        return f"required Python module missing: {module_name}"
                except ModuleNotFoundError:
                    return f"required Python module missing: {module_name}"
    elif kind == "command":
        for raw in argv[1:]:
            if raw.endswith(".py"):
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = ROOT / candidate
                if not candidate.exists():
                    return f"required script missing: {raw}"
    return None


def _run_one(entry: dict[str, Any], timeout_s: int) -> CheckResult:
    check_id = str(entry.get("id", "unnamed"))
    argv = [_substitute(str(arg)) for arg in entry.get("command", [])]
    if not argv:
        return CheckResult(check_id, "ERROR", "manifest entry has an empty command")
    missing = _command_missing_reason(entry, argv)
    if missing:
        return CheckResult(check_id, "MISSING", missing)

    expected = int(entry.get("expected_exit_code", 0))
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(check_id, "TIMEOUT", f"exceeded {timeout_s}s", None, time.monotonic() - started)
    except OSError as exc:
        return CheckResult(check_id, "ERROR", f"process launch failed: {exc}", None, time.monotonic() - started)

    duration = time.monotonic() - started
    output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    if proc.returncode != expected:
        tail = output[-2000:] if output else "no stdout/stderr"
        return CheckResult(check_id, "FAIL", f"exit={proc.returncode}, expected={expected}\n{tail}", proc.returncode, duration)

    if str(entry.get("kind")) == "pytest":
        skipped = 0
        matches = re.findall(r"(?:^|,|\s)(\d+)\s+skipped\b", output)
        if matches:
            skipped = max(int(item) for item in matches)
        allowed = int(entry.get("max_skips", 0))
        if skipped > allowed:
            return CheckResult(check_id, "SKIPPED", f"skipped={skipped} exceeds max_skips={allowed}", proc.returncode, duration)

    return CheckResult(check_id, "PASS", output[-2000:], proc.returncode, duration)


def _run_audit(module: str, args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        return {
            "module": module,
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    except Exception as exc:
        return {
            "module": module,
            "exit_code": None,
            "passed": False,
            "stdout_tail": "",
            "stderr_tail": f"{type(exc).__name__}: {exc}",
        }


def _clean_room_verdict(lock_path: Path) -> dict[str, Any]:
    pip_check_proc = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False)
    frontend_lock_ok = False
    try:
        from scripts.audit_frontend_lock import main as check_frontend_lock
        frontend_lock_ok = (check_frontend_lock() == 0)
    except Exception:
        frontend_lock_ok = False

    checks = {
        "marker_env_set": os.environ.get("MINDED_CLEAN_ROOM") == "1",
        "inside_virtual_environment": bool(os.environ.get("VIRTUAL_ENV")) or sys.prefix != sys.base_prefix,
        "lockfile_present": lock_path.exists(),
        "lockfile_nonempty": False,
        "lockfile_has_sha256_hashes": False,
        "pip_check_passed": pip_check_proc.returncode == 0,
        "frontend_lock_verified": frontend_lock_ok,
    }
    if lock_path.exists():
        try:
            lines = lock_path.read_text(encoding="utf-8").splitlines()
            checks["lockfile_nonempty"] = any(
                line.strip() and not line.strip().startswith("#") for line in lines
            )
            checks["lockfile_has_sha256_hashes"] = any("--hash=sha256:" in line for line in lines)
        except OSError:
            checks["lockfile_nonempty"] = False
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "note": (
            "Clean room verified: marker, virtual environment, non-empty lockfile, sha256 lock hashes, pip check, and frontend lock verification are all verified."
            if all(checks.values())
            else "CLEAN_ROOM_INCOMPLETE: requires MINDED_CLEAN_ROOM=1, active virtualenv, non-empty sha256 lockfile, pip check passing with zero conflicts, and frontend lock alignment."
        ),
    }


def _collect_versions() -> dict[str, Any]:
    import importlib.metadata as metadata

    result: dict[str, Any] = {}
    for name in ("duckdb", "polars", "pyarrow", "pydantic", "sqlalchemy", "alembic"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def run_gate(*, human_signoff: bool = False) -> GateReport:
    report = GateReport(timestamp=datetime.now(timezone.utc).isoformat())
    entries = _load_manifest(MANIFEST)
    report.manifest_check_count = len(entries)

    # Side audits always run so BLOCKED output remains diagnostically useful.
    report.release_status = _run_audit("scripts.audit_release_status", ["--root", str(ROOT)])
    report.serialization_audit = _run_audit(
        "scripts.audit_canonical_serialization", ["--root", str(ROOT)]
    )
    lock_path = ROOT / "requirements-lock.txt"
    report.dependency_lock = _run_audit(
        "scripts.audit_dependency_lock", ["--lock", str(lock_path)]
    )
    report.frontend_lock = _run_audit(
        "scripts.audit_frontend_lock", []
    )
    report.clean_room = _clean_room_verdict(lock_path)

    seen: set[str] = set()
    for entry in entries:
        check_id = str(entry.get("id", "unnamed"))
        if check_id in seen:
            report.checks[check_id] = CheckResult(check_id, "ERROR", "duplicate manifest id")
            continue
        seen.add(check_id)
        timeout_s = int(entry.get("timeout_seconds", 900))
        report.checks[check_id] = _run_one(entry, timeout_s)

    report.checks_executed = sum(
        result.result in {"PASS", "FAIL", "ERROR", "TIMEOUT", "SKIPPED"}
        for result in report.checks.values()
    )
    report.failed_checks = [
        check_id
        for check_id, result in report.checks.items()
        if result.result in {"FAIL", "ERROR", "TIMEOUT", "SKIPPED"}
    ]
    report.missing_checks = [
        check_id for check_id, result in report.checks.items() if result.result == "MISSING"
    ]

    migration = report.checks.get("alembic_no_drift")
    report.migration_drift = {
        "declared_in_manifest": migration is not None,
        "passed": migration is not None and migration.result == "PASS",
        "result": migration.result if migration else "MISSING",
    }

    side_failures: list[str] = []
    for label, audit in (
        ("release_status", report.release_status),
        ("serialization", report.serialization_audit),
        ("dependency_lock", report.dependency_lock),
    ):
        if not audit.get("passed", False):
            side_failures.append(label)
    if not report.clean_room.get("verified", False):
        side_failures.append("clean_room")
    if report.manifest_check_count == 0:
        side_failures.append("manifest_empty")
    if report.checks_executed != report.manifest_check_count:
        side_failures.append("manifest_not_fully_executed")
    if migration is None or migration.result != "PASS":
        side_failures.append("migration_drift")

    all_passed = (
        report.manifest_check_count > 0
        and report.checks_executed == report.manifest_check_count
        and not report.failed_checks
        and not report.missing_checks
        and not side_failures
    )
    report.status = "PASS" if all_passed else "BLOCKED"
    report.release_decision = (
        "RELEASE-READY" if all_passed and human_signoff else
        "RELEASE-CANDIDATE" if all_passed else
        "RELEASE-BLOCKED"
    )
    report.environment = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "os": platform.system(),
        "arch": platform.machine(),
        "root": str(ROOT),
    }
    report.versions = _collect_versions()
    return report


def _serialize(report: GateReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["checks"] = {key: asdict(value) for key, value in report.checks.items()}
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default="phase_release_report.json")
    parser.add_argument("--human-signoff", action="store_true")
    args = parser.parse_args(argv)

    signoff = args.human_signoff or os.environ.get("MINDED_HUMAN_SIGNOFF") == "1"
    report = run_gate(human_signoff=signoff)
    payload = _serialize(report)
    output = Path(args.json_out)
    output_path = output if output.is_absolute() else ROOT / output
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"RELEASE GATE: {report.status}")
    print(f"RELEASE DECISION: {report.release_decision}")
    print(f"Manifest checks: {report.checks_executed}/{report.manifest_check_count} executed")
    if report.failed_checks:
        print("Failed checks:")
        for check_id in report.failed_checks:
            detail = report.checks[check_id].detail.splitlines()[0] if report.checks[check_id].detail else ""
            print(f"  - {check_id}: {detail}")
    if report.missing_checks:
        print("Missing checks:")
        for check_id in report.missing_checks:
            print(f"  - {check_id}: {report.checks[check_id].detail}")
    for label in ("release_status", "serialization_audit", "dependency_lock"):
        print(f"Side audit {label}: {'PASS' if getattr(report, label).get('passed') else 'FAIL'}")
    print(f"Clean room: {'VERIFIED' if report.clean_room.get('verified') else 'CLEAN_ROOM_INCOMPLETE'}")
    print(f"Migration drift: {'PASS' if report.migration_drift.get('passed') else 'FAIL'}")
    print(f"Full report: {output_path}")
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
