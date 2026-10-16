"""Strict offline desktop release certification.

This verifier is intentionally conservative. It does not claim that a source
repository is a self-contained desktop application.  Certification is based
on an extracted/staged desktop bundle and requires the runtime/frontend/assets
needed to launch without Python, Node, npm, pip, or network access from the
host machine.

Usage:
    python scripts/offline_desktop_certification.py --bundle <bundle-root>

Exit code 0 only when every required offline invariant passes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit


NETWORK_URL_RE = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)
LOCAL_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def _check(label: str, condition: bool, detail: str) -> tuple[bool, str]:
    status = "PASS" if condition else "FAIL"
    return condition, f"[{status}] {label}: {detail}"


def _runtime_candidates(bundle: Path) -> list[Path]:
    names = [
        "runtime/python.exe",
        "python/python.exe",
        "python/python3.exe",
        "MindedAAOS/python.exe",
        "MindedAAOS/_internal/python.exe",
    ]
    return [bundle / name for name in names]


def _node_candidates(bundle: Path) -> list[Path]:
    names = [
        "runtime/node.exe",
        "node/node.exe",
        "MindedAAOS/node.exe",
    ]
    return [bundle / name for name in names]


def _frontend_candidates(bundle: Path) -> list[Path]:
    return [
        bundle / "frontend",
        bundle / "web",
        bundle / "MindedAAOS" / "frontend",
        bundle / "MindedAAOS" / "web",
        bundle / "apps" / "web" / ".next" / "standalone",
    ]


# Exact, lower-case hostnames that may appear as *text* in a packaged bundle (XML/SVG
# namespaces, documentation links inside vendored frameworks, ...). Matching is on the
# parsed hostname only -- never a string prefix or substring -- so "https://a.evil.example"
# or "https://evil.example/?x=localhost" cannot satisfy an entry. Single-label fragments that
# minifiers produce (e.g. "http://n") are NOT allowlisted: if a real bundle trips on one,
# review it and add the exact host here deliberately.
ALLOWED_EXTERNAL_HOSTS = frozenset({
    "www.w3.org",
    "react.dev",
    "reactjs.org",
    "nextjs.org",
    "github.com",
    "tailwindcss.com",
})


# Inert URL *literals* emitted by Next.js 15.5.x runtime chunks of the static export (verified against a
# real `npm run build` output): the router parses relative paths with ``new URL(path, "http://n")`` and
# the URL polyfill feature-tests the parser with the others. They are never fetched. They are matched as
# WHOLE strings -- "http://n.evil.example" or "https://a/anything-else" are still rejected.
INERT_URL_LITERALS = frozenset({
    "http://n",
    "http://f",
    "https://a",
    "https://x",
    "https://a@b",
    "https://a/c%20d?a=1&c=3",
    "https://a#\u0431",
    "https://\u0442\u0435\u0441\u0442",
})


def _url_host(url: str) -> str | None:
    """Return the parsed lower-case hostname of ``url`` (scheme-relative allowed), else None."""
    candidate = url.strip().strip("\"'")
    if candidate.startswith("//"):
        candidate = "https:" + candidate
    try:
        return (urlsplit(candidate).hostname or "").lower() or None
    except ValueError:
        return None


def _is_url_allowed(url: str) -> bool:
    if url in INERT_URL_LITERALS:
        return True
    host = _url_host(url)
    if host is None:
        return False
    return host in LOCAL_ALLOWED_HOSTS or host in ALLOWED_EXTERNAL_HOSTS


def _wheelhouse_state(bundle: Path) -> tuple[Path | None, int]:
    """Locate the bundled local wheelhouse and count the wheels in it."""
    for cand in (bundle / "wheelhouse", bundle / "MindedAAOS" / "wheelhouse"):
        if cand.is_dir():
            return cand, sum(1 for p in cand.rglob("*.whl") if p.is_file())
    return None, 0


def _find_external_urls(target: Path, base_bundle: Path) -> list[str]:
    hits: list[str] = []
    excluded = {".git", "__pycache__", ".pytest_cache", "node_modules"}
    for path in target.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".js", ".mjs", ".ts", ".tsx", ".json", ".html", ".css", ".ps1", ".yml", ".yaml", ".txt"}:
            continue
        text = _read_text(path)
        for url in NETWORK_URL_RE.findall(text):
            if not _is_url_allowed(url):
                hits.append(f"{path.relative_to(base_bundle)} -> {url[:180]}")
    return hits


def certify(bundle: Path) -> tuple[bool, dict]:
    checks: list[tuple[bool, str]] = []

    exe_candidates = [bundle / "MindedAAOS.exe", bundle / "MindedAAOS" / "MindedAAOS.exe"]
    frontend_candidates = [bundle / "frontend", bundle / "MindedAAOS" / "frontend"]
    exe = next((p for p in exe_candidates if p.is_file()), None)
    frontend = next((p for p in frontend_candidates if p.is_dir()), None)
    manifest_candidates = [
        bundle / "MindedAAOS.exe.manifest.json",
        bundle / "MindedAAOS" / "MindedAAOS.exe.manifest.json",
    ]
    manifest_path = next((p for p in manifest_candidates if p.is_file()), None)
    marker_candidates = [bundle / "OFFLINE_MODE", bundle / "MindedAAOS" / "OFFLINE_MODE"]
    offline_marker = next((p for p in marker_candidates if p.is_file()), None)

    checks.append(_check("desktop executable", exe is not None, str(exe.relative_to(bundle)) if exe else "MindedAAOS.exe missing"))
    pyinstaller_payload = bool(exe and (exe.parent / "_internal").is_dir())
    checks.append(_check("bundled Python payload", pyinstaller_payload, "PyInstaller _internal payload present" if pyinstaller_payload else "PyInstaller _internal payload missing"))
    checks.append(_check("static frontend bundle", frontend is not None and (frontend / "index.html").is_file(), "frontend/index.html present" if frontend and (frontend / "index.html").is_file() else "static frontend/index.html missing"))
    checks.append(_check("offline marker", offline_marker is not None, "OFFLINE_MODE marker present" if offline_marker else "OFFLINE_MODE marker missing"))

    wheelhouse, wheel_count = _wheelhouse_state(bundle)
    checks.append(_check(
        "local wheelhouse",
        wheel_count > 0,
        f"{wheelhouse.relative_to(bundle)} contains {wheel_count} wheel(s)" if wheel_count else "non-empty wheelhouse/ (*.whl) missing",
    ))

    external_resource_hits: list[str] = []
    if frontend:
        for path in frontend.rglob("*.html"):
            text = _read_text(path)
            for match in re.finditer(r"(?:src|href)\s*=\s*[\"'](https?:[^\"']*|//[^\"']*)", text, re.IGNORECASE):
                value = match.group(1)
                if not _is_url_allowed(value):
                    external_resource_hits.append(f"{path.relative_to(bundle)} -> {value[:180]}")
        # Rigorously scan all JS, CSS, JSON, and static assets in the frontend bundle
        external_resource_hits.extend(_find_external_urls(frontend, bundle))
    checks.append(_check("frontend external resources", not external_resource_hits, "none found" if not external_resource_hits else f"{len(external_resource_hits)} external resource(s) found"))

    if manifest_path:
        try:
            manifest = json.loads(_read_text(manifest_path))
            checks.append(_check("manifest offline", manifest.get("offline") is True, "offline=true" if manifest.get("offline") is True else str(manifest.get("offline"))))
            checks.append(_check("manifest network policy", manifest.get("network_policy") == "loopback-only", "loopback-only" if manifest.get("network_policy") == "loopback-only" else str(manifest.get("network_policy"))))
            checks.append(_check("manifest runtime dependency", manifest.get("runtime_dependency") == "self-contained", "self-contained" if manifest.get("runtime_dependency") == "self-contained" else str(manifest.get("runtime_dependency"))))
        except (json.JSONDecodeError, OSError) as exc:
            checks.append(_check("bundle manifest readable", False, f"invalid manifest: {exc}"))
    else:
        checks.append(_check("bundle manifest", False, "MindedAAOS.exe.manifest.json missing"))

    passed = all(ok for ok, _ in checks)
    payload = {
        "certification": "OFFLINE_DESKTOP",
        "status": "PASS" if passed else "BLOCKED",
        "bundle": str(bundle),
        "checks": [{"status": "PASS" if ok else "FAIL", "message": msg} for ok, msg in checks],
        "external_frontend_resources": external_resource_hits[:50],
    }
    for _, msg in checks:
        print(msg)
    print(f"OFFLINE DESKTOP CERTIFICATION: {'PASS' if passed else 'BLOCKED'}")
    return passed, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="Extracted final desktop bundle root")
    parser.add_argument("--json", dest="json_path", help="Write machine-readable certification JSON")
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    if not bundle.is_dir():
        print(f"[FAIL] bundle: directory does not exist: {bundle}")
        return 2

    passed, payload = certify(bundle)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
