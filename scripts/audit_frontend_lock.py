"""Fail-closed audit for the frontend package/lock dependency contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "apps" / "web" / "package.json"
LOCK = ROOT / "apps" / "web" / "package-lock.json"


def main() -> int:
    errors: list[str] = []
    try:
        package = json.loads(PACKAGE.read_text(encoding="utf-8"))
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"FRONTEND_LOCK_AUDIT: ERROR ({type(exc).__name__}: {exc})")
        return 2

    declared = package.get("dependencies", {}).get("next")
    root_deps = lock.get("packages", {}).get("", {}).get("dependencies", {})
    locked_root = root_deps.get("next")
    locked_pkg = lock.get("packages", {}).get("node_modules/next", {})
    locked_version = locked_pkg.get("version")

    if declared != locked_root:
        errors.append(f"root dependency mismatch: package.json next={declared!r}, package-lock root next={locked_root!r}")
    if declared != locked_version:
        errors.append(f"installed package graph mismatch: package.json next={declared!r}, node_modules/next={locked_version!r}")

    # The release must also resolve @next/env consistently with the Next.js major/minor line.
    if locked_version is not None:
        env_version = lock.get("packages", {}).get("node_modules/@next/env", {}).get("version")
        if env_version is not None and env_version != locked_version:
            errors.append(f"@next/env mismatch: next={locked_version!r}, @next/env={env_version!r}")

    if errors:
        print("FRONTEND_LOCK_AUDIT: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"FRONTEND_LOCK_AUDIT: PASS (next={declared})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
