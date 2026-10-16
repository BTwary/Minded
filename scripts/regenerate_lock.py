"""Regenerate hashes / the Windows desktop closure for the project's hash-locked requirement files.

  python scripts/regenerate_lock.py --rehash requirements-lock.txt
      Keep every ``name==version`` pin, refresh its --hash list from PyPI (all published files of
      that exact version, so the lock stays platform independent).

  python scripts/regenerate_lock.py --desktop
      Resolve pywebview + pyinstaller (floors below) for Windows x64 / CPython 3.12 with win32
      environment markers evaluated, pick the newest compatible versions, and rewrite
      packaging/windows/requirements-desktop.txt (header comments preserved).

Adding a new package to requirements-lock.txt: resolve it in a venv built from the lock using the
existing pins as constraints (``pip install -c <pins> <pkg>``), append ``name==version`` lines, then
run ``--rehash``. Needs network access to pypi.org and the ``packaging`` library.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
DESKTOP_LOCK = ROOT / "packaging" / "windows" / "requirements-desktop.txt"
DESKTOP_ROOTS = ("pywebview>=5.2", "pyinstaller>=6.10.0")
WIN_ENV = {
    "sys_platform": "win32", "platform_system": "Windows", "os_name": "nt", "platform_machine": "AMD64",
    "implementation_name": "cpython", "platform_python_implementation": "CPython",
    "python_version": "3.12", "python_full_version": "3.12.3", "extra": "",
}
TARGET_PY = Version("3.12.3")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https PyPI endpoint
        return json.load(resp)


def _hashes(name: str, version: str) -> list[str]:
    data = _get(f"https://pypi.org/pypi/{_norm(name)}/{version}/json")
    found = sorted({u["digests"]["sha256"] for u in data["urls"]})
    if not found:
        raise SystemExit(f"no files published for {name}=={version}")
    return found


def _pin_line(name: str, version: str) -> str:
    return f"{_norm(name)}=={version} " + " ".join(f"--hash=sha256:{h}" for h in _hashes(name, version))


def rehash(lock: Path) -> None:
    lines = lock.read_text(encoding="utf-8").splitlines()
    head = [l for l in lines if l.startswith("#") or not l.strip()]
    pins = {}
    for l in lines:
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)", l)
        if m:
            pins[_norm(m.group(1))] = m.group(2)
    body = [_pin_line(n, v) for n, v in sorted(pins.items())]
    lock.write_text("\n".join([l for l in head if l.strip()]) + "\n" + "\n".join(body) + "\n", encoding="utf-8")
    print(f"rehashed {len(body)} pins in {lock}")


def _best(name: str, spec: SpecifierSet) -> Version:
    best = None
    for v, files in _get(f"https://pypi.org/pypi/{_norm(name)}/json")["releases"].items():
        try:
            ver = Version(v)
        except ValueError:
            continue
        if ver.is_prerelease or not files or all(f.get("yanked") for f in files) or not spec.contains(ver):
            continue
        rp = files[0].get("requires_python")
        if rp and not SpecifierSet(rp).contains(TARGET_PY):
            continue
        best = ver if best is None or ver > best else best
    if best is None:
        raise SystemExit(f"no installable version of {name} satisfies {spec}")
    return best


def desktop() -> None:
    pins: dict[str, Version] = {}
    cons: dict[str, SpecifierSet] = {}
    queue = [Requirement(r) for r in DESKTOP_ROOTS]
    while queue:
        req = queue.pop(0)
        n = _norm(req.name)
        cons[n] = cons.get(n, SpecifierSet()) & req.specifier
        ver = _best(n, cons[n])
        if pins.get(n) == ver:
            continue
        pins[n] = ver
        for r in _get(f"https://pypi.org/pypi/{n}/{ver}/json")["info"].get("requires_dist") or []:
            q = Requirement(r)
            if q.marker is None or q.marker.evaluate(WIN_ENV):
                queue.append(q)
    for n, s in cons.items():
        assert s.contains(pins[n]), f"unsatisfiable: {n} {s} vs {pins[n]}"
    old = DESKTOP_LOCK.read_text(encoding="utf-8").splitlines()
    head = [l for l in old if l.startswith("#")]
    DESKTOP_LOCK.write_text("\n".join(head) + "\n" + "\n".join(_pin_line(n, str(v)) for n, v in sorted(pins.items())) + "\n", encoding="utf-8")
    print(f"wrote {len(pins)} pins to {DESKTOP_LOCK}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rehash", type=Path, metavar="LOCKFILE")
    g.add_argument("--desktop", action="store_true")
    a = ap.parse_args()
    rehash(a.rehash) if a.rehash else desktop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
