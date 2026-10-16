"""Fail unless every ``name==version`` pin in a lock file is installed at exactly that version.

Used by packaging/windows/build_exe_scope.ps1 so a release EXE is only built from the
hash-locked desktop toolchain (pywebview / PyInstaller and their closure).

    python scripts/verify_desktop_pins.py packaging/windows/requirements-desktop.txt
"""
from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)")


def parse_pins(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in text.splitlines():
        m = _PIN.match(line.strip())
        if m:
            pins[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return pins


def check(lock: Path) -> list[str]:
    problems: list[str] = []
    pins = parse_pins(lock.read_text(encoding="utf-8"))
    if not pins:
        return [f"{lock}: no pinned requirements found"]
    for name, want in sorted(pins.items()):
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name}: not installed (need {want})")
            continue
        if have != want:
            problems.append(f"{name}: installed {have}, locked {want}")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: verify_desktop_pins.py <lockfile>")
        return 2
    problems = check(Path(args[0]))
    if problems:
        print("DESKTOP_PIN_AUDIT: FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print(f"DESKTOP_PIN_AUDIT: PASS ({len(parse_pins(Path(args[0]).read_text(encoding='utf-8')))} pins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
