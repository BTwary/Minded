"""Regression tests for the release-identity-bound status audit.

These prove the manifest-declared [release] document is the sole source of
release authority, and that no other RELEASE_STATUS document -- historical,
unrelated, or simply newer-sorting by filename -- can substitute for it.
"""
import textwrap
from pathlib import Path

from scripts.audit_release_status import _check, _scan


def _write_manifest(
    root: Path,
    *,
    version: str = "31",
    artifact_name: str = "Minded_AAOS_v31_2026-09-26_blank_terminal_state_fix.zip",
    status_document: str = "docs/AAOS_V31_RELEASE_STATUS.md",
) -> Path:
    manifest_dir = root / "tests" / "independent_release"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.toml"
    manifest_path.write_text(
        textwrap.dedent(
            f"""\
            [release]
            version = "{version}"
            artifact_name = "{artifact_name}"
            status_document = "{status_document}"
            """
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_status_doc(
    root: Path,
    relative_path: str,
    *,
    version: str = "31",
    artifact_name: str = "Minded_AAOS_v31_2026-09-26_blank_terminal_state_fix.zip",
    release_status: str = "RELEASE READY",
) -> Path:
    doc_path = root / relative_path
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        textwrap.dedent(
            f"""\
            # AA-OS v{version} Release Status

            **Date:** 2026-09-26
            **Archive:** `{artifact_name}`
            **Release Status:** **{release_status}**
            """
        ),
        encoding="utf-8",
    )
    return doc_path


def test_markdown_blocked_release_status_is_detected():
    reasons = _scan("**Release Status:** **BLOCKED** — final desktop certification pending")
    assert any("explicit blocker" in reason for reason in reasons)


def test_fail_when_v31_manifest_and_only_v29_status_exists(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    _write_status_doc(
        tmp_path,
        "docs/AAOS_V29_RELEASE_STATUS.md",
        version="29",
        artifact_name="Minded_AAOS_v29_offline_desktop_certified_2026-09-25.zip",
        release_status="HARDENED & CERTIFIED FOR OFFLINE DESKTOP VALIDATION",
    )

    passed, document_label, reasons = _check(tmp_path, manifest_path)

    assert not passed
    assert document_label == "docs/AAOS_V31_RELEASE_STATUS.md"
    assert any("missing" in reason for reason in reasons)


def test_fail_when_v31_manifest_and_only_v30_status_exists(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    _write_status_doc(
        tmp_path,
        "docs/AAOS_V30_RELEASE_STATUS.md",
        version="30",
        artifact_name="Minded_AAOS_v30_2026-09-26.zip",
    )

    passed, document_label, reasons = _check(tmp_path, manifest_path)

    assert not passed
    assert any("missing" in reason for reason in reasons)


def test_fail_when_matching_document_declares_wrong_version(tmp_path):
    # The right filename/path, but the header text was copy-pasted from an older doc.
    manifest_path = _write_manifest(tmp_path)
    _write_status_doc(tmp_path, "docs/AAOS_V31_RELEASE_STATUS.md", version="30")

    passed, document_label, reasons = _check(tmp_path, manifest_path)

    assert not passed
    assert any("declares version '30'" in reason for reason in reasons)


def test_fail_when_matching_document_references_wrong_artifact(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    _write_status_doc(
        tmp_path,
        "docs/AAOS_V31_RELEASE_STATUS.md",
        artifact_name="Minded_AAOS_v30_wrong_artifact.zip",
    )

    passed, document_label, reasons = _check(tmp_path, manifest_path)

    assert not passed
    assert any("references artifact" in reason for reason in reasons)


def test_fail_when_matching_v31_status_is_not_release_ready(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    _write_status_doc(
        tmp_path, "docs/AAOS_V31_RELEASE_STATUS.md", release_status="NOT RELEASE READY"
    )

    passed, document_label, reasons = _check(tmp_path, manifest_path)

    assert not passed
    assert any("explicit blocker" in reason for reason in reasons)


def test_pass_when_matching_v31_status_is_release_ready(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    _write_status_doc(
        tmp_path, "docs/AAOS_V31_RELEASE_STATUS.md", release_status="RELEASE READY"
    )

    passed, document_label, reasons = _check(tmp_path, manifest_path)

    assert passed
    assert reasons == []
    assert document_label == "docs/AAOS_V31_RELEASE_STATUS.md"


def test_extra_older_status_documents_are_irrelevant(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    _write_status_doc(
        tmp_path, "docs/AAOS_V31_RELEASE_STATUS.md", release_status="RELEASE READY"
    )
    # Historical documents remain in the tree, exactly as in the real repo.
    _write_status_doc(
        tmp_path,
        "docs/AAOS_V29_RELEASE_STATUS.md",
        version="29",
        artifact_name="Minded_AAOS_v29_offline_desktop_certified_2026-09-25.zip",
    )
    _write_status_doc(
        tmp_path, "docs/AAOS_V28_RELEASE_STATUS.md", version="28", artifact_name="Minded_AAOS_v28.zip"
    )

    passed, document_label, reasons = _check(tmp_path, manifest_path)

    assert passed
    assert reasons == []


def test_lexicographically_newer_unrelated_document_is_never_substituted(tmp_path):
    """This is the exact failure mode the old
    ``max(candidates, key=lambda p: p.name.upper())`` filename-sort logic had:
    a document whose filename sorts after the real one, and which itself claims
    to be release-ready, must never be picked up in place of the manifest's
    declared document.
    """
    manifest_path = _write_manifest(tmp_path)
    _write_status_doc(
        tmp_path, "docs/AAOS_V31_RELEASE_STATUS.md", release_status="NOT RELEASE READY"
    )
    _write_status_doc(
        tmp_path,
        "docs/AAOS_VZZZ_UNRELATED_RELEASE_STATUS.md",
        version="ZZZ",
        artifact_name="not_the_real_artifact.zip",
        release_status="RELEASE READY",
    )

    passed, document_label, reasons = _check(tmp_path, manifest_path)

    assert not passed
    assert document_label == "docs/AAOS_V31_RELEASE_STATUS.md"
    assert any("explicit blocker" in reason for reason in reasons)


def test_fail_when_manifest_has_no_release_table(tmp_path):
    manifest_dir = tmp_path / "tests" / "independent_release"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "manifest.toml"
    manifest_path.write_text('[[check]]\nid = "x"\n', encoding="utf-8")

    passed, document_label, reasons = _check(tmp_path, manifest_path)

    assert not passed
    assert document_label == ""
    assert any("release identity" in reason for reason in reasons)
