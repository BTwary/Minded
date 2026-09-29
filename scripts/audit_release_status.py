"""Fail closed unless the release-identity-bound status document reports no unfinished work.

Release authority is never inferred by sorting ``*RELEASE_STATUS*.md`` filenames.
Instead, ``tests/independent_release/manifest.toml`` declares under a ``[release]``
table exactly which artifact/version is current and which document is authoritative
for it. This script resolves *that specific document* and refuses to substitute any
other file for it, however new that other file's name sorts, however many older
status documents remain in ``docs/``.

Fails closed whenever:
  - the manifest does not declare a complete release identity;
  - the declared status document does not exist;
  - the document does not declare a release version, or declares one that does not
    match the manifest;
  - the document does not declare an artifact, or declares one that does not match
    the manifest;
  - the document reports unfinished work or an explicit not-release-ready status.
"""
from __future__ import annotations

import argparse
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

_EXPLICIT_BLOCKER = re.compile(
    r"release\s+status\s*[:=][^\n]{0,80}\b(not\s+release[ -]ready|not\s+releasable|blocked)\b",
    re.IGNORECASE,
)
_SECTION = re.compile(r"\bSections?\s+(\d+)(?:\s*[-\u2013]\s*(\d+))?", re.IGNORECASE)
_UNFINISHED = (
    "not started",
    "unfinished",
    "not complete",
    "not yet complete",
    "pending",
    "deferred",
    "incomplete",
)
_VERSION_RE = re.compile(r"^#\s*AA-OS\s+v(\S+)\s+Release Status", re.IGNORECASE | re.MULTILINE)
_ARTIFACT_RE = re.compile(r"\*\*Archive:\*\*\s*`?([^`\n]+?)`?\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class ReleaseIdentity:
    version: str
    status_document: str
    artifact_name: str


def _normalize_version(value: str) -> str:
    return value.strip().lower().lstrip("v")


def _load_release_identity(manifest_path: Path) -> tuple[ReleaseIdentity | None, str | None]:
    """Read the manifest's [release] table. Returns (identity, None) or (None, error)."""
    if not manifest_path.exists():
        return None, f"manifest missing: {manifest_path}"
    try:
        with manifest_path.open("rb") as handle:
            data = tomllib.load(handle)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"manifest unreadable: {exc}"

    release = data.get("release")
    if not isinstance(release, dict):
        return None, "manifest has no [release] table declaring release identity"

    version = release.get("version")
    status_document = release.get("status_document")
    artifact_name = release.get("artifact_name")
    missing = [
        name
        for name, value in (
            ("version", version),
            ("status_document", status_document),
            ("artifact_name", artifact_name),
        )
        if not value
    ]
    if missing:
        return None, f"manifest [release] table missing required field(s): {', '.join(missing)}"

    return ReleaseIdentity(str(version), str(status_document), str(artifact_name)), None


def _scan_unfinished(text: str) -> list[str]:
    """Detect explicit not-release-ready blockers and unfinished-section markers."""
    reasons: list[str] = []
    explicit = _EXPLICIT_BLOCKER.search(text)
    if explicit:
        reasons.append(f"explicit blocker: {explicit.group(0).strip()}")

    lines = text.splitlines()
    for index, line in enumerate(lines):
        section_match = _SECTION.search(line)
        if not section_match:
            continue
        window = " ".join(lines[index : min(index + 4, len(lines))]).lower()
        markers = [marker for marker in _UNFINISHED if marker in window]
        if markers:
            label = section_match.group(0)
            reasons.append(f"{label} near line {index + 1}: {', '.join(markers)}")
    return reasons


# Backwards-compatible alias — tests/test_release_status_audit.py imports `_scan`.
_scan = _scan_unfinished


def _check(root: Path, manifest_path: Path) -> tuple[bool, str, list[str]]:
    """Returns (passed, resolved_document_label, reasons_for_failure)."""
    identity, error = _load_release_identity(manifest_path)
    if identity is None:
        return False, "", [f"cannot resolve release identity: {error}"]

    document = root / identity.status_document
    if not document.exists():
        return False, identity.status_document, [
            f"declared status document missing: {identity.status_document}"
        ]

    text = document.read_text(encoding="utf-8", errors="replace")
    reasons: list[str] = []

    version_match = _VERSION_RE.search(text)
    if not version_match:
        reasons.append(
            "document does not declare a release version "
            f"(expected a heading like '# AA-OS v{identity.version} Release Status')"
        )
    elif _normalize_version(version_match.group(1)) != _normalize_version(identity.version):
        reasons.append(
            f"document declares version '{version_match.group(1)}', "
            f"manifest requires '{identity.version}'"
        )

    artifact_match = _ARTIFACT_RE.search(text)
    if not artifact_match:
        reasons.append("document does not declare an **Archive:** artifact name")
    elif artifact_match.group(1).strip() != identity.artifact_name:
        reasons.append(
            f"document references artifact '{artifact_match.group(1).strip()}', "
            f"manifest requires '{identity.artifact_name}'"
        )

    reasons.extend(_scan_unfinished(text))
    return (len(reasons) == 0), identity.status_document, reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="Repository root. The manifest and the status document it declares "
        "are both resolved relative to this path.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to the manifest.toml declaring [release] identity "
        "(default: <root>/tests/independent_release/manifest.toml)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else root / "tests" / "independent_release" / "manifest.toml"
    )

    passed, document_label, reasons = _check(root, manifest_path)
    if passed:
        print(f"RELEASE_STATUS_AUDIT: PASS ({document_label})")
        return 0

    label = document_label or "<unresolved>"
    print(f"RELEASE_STATUS_AUDIT: FAIL ({label})")
    for reason in reasons:
        print(f"  - {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
